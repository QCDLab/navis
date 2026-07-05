//! Monte Carlo integration backend, wrapping `mchep`'s VEGAS+ integrator
//! (adaptive stratified sampling + importance sampling).
//!
//! `mchep`'s `Integrand` trait is a plain `fn eval(&self, x: &[f64]) -> f64`
//! with no side channel, but every Monte Carlo point here also needs to
//! emit PineAPPL grid fills (one per channel/order, not just contribute to
//! a single running total) -- exactly the same requirement the Fortran's
//! `DPLUS` has, calling `pineappl_grid_fill2` on every point rather than
//! only at the end. To support that, `mchep` gained a small additive
//! `ObservableIntegrand` trait (see `mchep::integrand`) that hands back a
//! `fill_weight` per point -- the VEGAS+ jacobian already normalized by
//! this point's share of its hypercube's sample count and by the number of
//! iterations counted in the final result -- so callers can scale their own
//! per-channel weights by it directly instead of re-deriving VEGAS's
//! internal bookkeeping.
//!
//! The `xx` coordinates handed to the wrapped integrand are still uniform
//! points in `[0, 1]^5` after VEGAS+'s own importance-sampling remap (the
//! `boundaries` passed to `VegasPlus::new` are always `[(0, 1); 5]`), so
//! `navis_core::kinematics::map_phase_space` and everything downstream of
//! it is unchanged from the previous VEGAS backend.

use mchep::integrand::ObservableIntegrand;
use mchep::vegasplus::VegasPlus;
use pineappl::grid::Grid;
use rayon::prelude::*;

/// VEGAS always integrates in 5 dimensions in `sihp-pp`: `W`, `V`, `X3`, and
/// up to two more for the rapidity/pT integration.
pub const VEGAS_NDIM: usize = 5;

/// Number of bins per dimension in VEGAS+'s adaptive grid.
const N_BINS: usize = 50;
/// Grid-refinement damping factor.
const ALPHA: f64 = 1.5;
/// Hypercube-resampling damping factor.
const BETA: f64 = 0.75;

/// One PineAPPL grid-fill record, the side effect an integrand emits per
/// Monte Carlo point (mirrors a single `pineappl_grid_fill2` call in
/// `DPLUS`).
#[derive(Debug, Clone)]
pub struct GridFill {
    pub order: usize,
    pub observable: f64,
    pub channel: usize,
    pub ntuple: [f64; 4],
    pub weight: f64,
}

/// Final result of a VEGAS+ run: the weighted-average integral estimate,
/// its standard deviation, and the chi-squared per degree of freedom,
/// matching `ERG1`/`ERG2`/`ERG3` (`COMMON /RESULT/`) as read by `FUNCDG` in
/// `hadrive-ms.f`.
#[derive(Debug, Clone, Copy)]
pub struct VegasOutcome {
    pub integral: f64,
    pub std_dev: f64,
    pub chi2_per_it: f64,
}

/// Adapts a `Fn(&[f64; VEGAS_NDIM], f64) -> (f64, Vec<GridFill>)` closure
/// (point, this-point's `fill_weight`) into `mchep`'s `ObservableIntegrand`.
struct ClosureIntegrand<F> {
    eval: F,
}

impl<F> ObservableIntegrand for ClosureIntegrand<F>
where
    F: Fn(&[f64; VEGAS_NDIM], f64) -> (f64, Vec<GridFill>) + Sync,
{
    type Observation = GridFill;

    fn dim(&self) -> usize {
        VEGAS_NDIM
    }

    fn eval(&self, x: &[f64], fill_weight: f64) -> (f64, Vec<GridFill>) {
        let mut xx = [0.0_f64; VEGAS_NDIM];
        xx.copy_from_slice(x);
        (self.eval)(&xx, fill_weight)
    }
}

/// Picks a safe number of stratification bins per dimension for VEGAS+'s
/// hypercubes.
fn choose_n_strat(n_eval: usize, dim: usize) -> usize {
    let mut n_strat = ((n_eval as f64 / 2.0).powf(1.0 / dim as f64))
        .floor()
        .max(1.0) as usize;
    while n_strat > 1 && 2 * n_strat.pow(dim as u32) > n_eval {
        n_strat -= 1;
    }
    n_strat.max(1)
}

/// Runs VEGAS+ on `integrand`, filling `grid` as a side effect (once per
/// counted iteration -- the first, "warm-up" iteration is excluded from
/// both the reported result and the grid fills, matching `mchep`'s own
/// convention) and returning the final estimate.
///
/// `integrand(xx, fill_weight)` mirrors `DPLUS(XX,grid,calls,wgt)`'s role:
/// it receives a point `xx` in `[0, 1]^5` together with this point's total
/// Monte Carlo weight, and must return the *unscaled* integrand value (used
/// to drive VEGAS+'s own importance-sampling estimator) plus any grid-fill
/// records for that point, already scaled by `fill_weight` where
/// appropriate.
#[allow(clippy::too_many_arguments)]
pub fn vegas<F>(
    integrand: F,
    n_iter: usize,
    n_eval: usize,
    seed: u64,
    grid: &mut Grid,
) -> VegasOutcome
where
    F: Fn(&[f64; VEGAS_NDIM], f64) -> (f64, Vec<GridFill>) + Sync,
{
    let boundaries = [(0.0_f64, 1.0_f64); VEGAS_NDIM];
    let n_strat = choose_n_strat(n_eval, VEGAS_NDIM);

    let mut integrator = VegasPlus::new(n_iter, n_eval, N_BINS, ALPHA, n_strat, BETA, &boundaries);
    integrator.set_seed(seed);

    // `grid` is guaranteed empty here (every call site builds a fresh
    // per-bin grid immediately before calling `vegas`), so this clone is a
    // cheap structural template -- no filled subgrids to copy -- used below
    // to give each parallel fill task its own accumulator.
    let empty_template = grid.clone();

    let wrapped = ClosureIntegrand { eval: integrand };
    let result = integrator.integrate_with_observations(&wrapped, None, |fills| {
        // `grid.fill` runs real interpolation-kernel work per call (it
        // touches every node touched by the interpolation order in each
        // dimension), and at NLO there are dozens of fills per Monte Carlo
        // point -- cheap per call but this loop dominates wall time if run
        // serially. Fan it out across one chunk per thread (a fine-grained
        // rayon fold/reduce here -- both with a fixed split count and with
        // `with_min_len`-bounded adaptive splitting -- was measured to be no
        // better, and sometimes worse, than this simple fixed chunking, since
        // the callers of `vegas` already run many bins concurrently via their
        // own outer `rayon` parallelism; the dominant remaining cost in that
        // regime is bin-to-bin load imbalance, which no fill-splitting
        // strategy here can fix), each accumulating into its own partial grid
        // (cloned from `empty_template`, so still empty at chunk-start), then
        // merged back sequentially; `Grid::merge` just sums overlapping bins,
        // far cheaper than re-deriving interpolation weights from scratch.
        let n_chunks = rayon::current_num_threads().max(1);
        let chunk_len = fills.len().div_ceil(n_chunks).max(1);
        let partials: Vec<Grid> = fills
            .par_chunks(chunk_len)
            .map(|chunk| {
                let mut acc = empty_template.clone();
                for fill in chunk {
                    if fill.weight.is_finite() {
                        acc.fill(
                            fill.order,
                            fill.observable,
                            fill.channel,
                            &fill.ntuple,
                            fill.weight,
                        );
                    }
                }
                acc
            })
            .collect();
        for partial in partials {
            grid.merge(partial)
                .expect("partial grid shares the same schema");
        }
    });

    VegasOutcome {
        integral: result.value,
        std_dev: result.error,
        chi2_per_it: result.chi2_dof,
    }
}
