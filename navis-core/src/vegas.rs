//! Monte Carlo integration backend, wrapping `mchep`'s VEGAS+ integrator
//! (adaptive stratified sampling + importance sampling).
//!
//! `mchep`'s `Integrand` trait is a plain `fn eval(&self, x: &[f64]) -> f64`
//! with no side channel, but every Monte Carlo point here also needs to
//! emit PineAPPL grid fills (one per channel/order, not just contribute to
//! a single running total), calling `pineappl_grid_fill2` on every point rather
//! than only at the end. To support that, `mchep` gained a small additive
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

use mchep::integrand::{Integrand, ObservableIntegrand};
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
const BETA: f64 = 0.0;

/// One PineAPPL grid-fill record, the side effect an integrand emits per
/// Monte Carlo point.
#[derive(Debug, Clone)]
pub struct GridFill {
    pub order: usize,
    pub observable: f64,
    pub channel: usize,
    pub ntuple: [f64; 4],
    pub weight: f64,
}

/// Final result of a VEGAS+ run: the weighted-average integral estimate,
/// its standard deviation, and the chi-squared per degree of freedom.
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

/// Adapts a `Fn(&[f64; VEGAS_NDIM]) -> f64` closure into `mchep`'s plain
/// `Integrand`, used when no grid fills are needed (see [`vegas_value_only`]).
struct ValueClosureIntegrand<F> {
    eval: F,
}

impl<F> Integrand for ValueClosureIntegrand<F>
where
    F: Fn(&[f64; VEGAS_NDIM]) -> f64 + Sync,
{
    fn dim(&self) -> usize {
        VEGAS_NDIM
    }

    fn eval(&self, x: &[f64]) -> f64 {
        let mut xx = [0.0_f64; VEGAS_NDIM];
        xx.copy_from_slice(x);
        (self.eval)(&xx)
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

    let empty_template = grid.clone();

    let wrapped = ClosureIntegrand { eval: integrand };
    let result = integrator.integrate_with_observations(&wrapped, None, |mut fills| {
        // PineAPPL's `PackedArray` stores its non-default entries in ascending
        // raveled-index order and inserts a not-yet-seen index by shifting
        // everything after its insertion point -- cheap (amortized O(1)) when
        // the new index is at or near the end of what's already stored, but
        // O(current size) when it lands in the middle. VEGAS+ hands points in
        // random order, so without sorting, most insertions land in the middle
        // of an ever-growing array. This dominated grid-filling time end to end.
        // Sorting by the same key `Grid::fill` uses to place an entry -- first
        // `(order, channel)` to pick the subgrid (matching the `[order, bin, channel]`
        // indexing in `Grid::fill`), then `observable` to pick the bin, then
        // `ntuple` in the same order as the `Scale(0), X(0), X(1), X(2)` kinematics
        // set up in `GridConfig::new` -- makes each subsequent fill (and each
        // chunk's later merge back into the running `grid`) land at or near the
        // tail of what's already stored.
        fills.sort_unstable_by(|a, b| {
            a.order
                .cmp(&b.order)
                .then_with(|| a.channel.cmp(&b.channel))
                .then_with(|| a.observable.total_cmp(&b.observable))
                .then_with(|| a.ntuple[0].total_cmp(&b.ntuple[0]))
                .then_with(|| a.ntuple[1].total_cmp(&b.ntuple[1]))
                .then_with(|| a.ntuple[2].total_cmp(&b.ntuple[2]))
                .then_with(|| a.ntuple[3].total_cmp(&b.ntuple[3]))
        });

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

/// Runs VEGAS+ on `integrand` and returns the final estimate, without
/// recording any grid fills. Uses `mchep`'s plain (non-observation)
/// `VegasPlus::integrate`, which skips the per-point observation buffering
/// and the PineAPPL interpolation-grid fill/merge that otherwise dominates
/// the runtime -- for when only the Monte Carlo prediction is wanted (see
/// `RunCard::generate_grids`).
pub fn vegas_value_only<F>(integrand: F, n_iter: usize, n_eval: usize, seed: u64) -> VegasOutcome
where
    F: Fn(&[f64; VEGAS_NDIM]) -> f64 + Sync,
{
    let boundaries = [(0.0_f64, 1.0_f64); VEGAS_NDIM];
    let n_strat = choose_n_strat(n_eval, VEGAS_NDIM);

    let mut integrator = VegasPlus::new(n_iter, n_eval, N_BINS, ALPHA, n_strat, BETA, &boundaries);
    integrator.set_seed(seed);

    let wrapped = ValueClosureIntegrand { eval: integrand };
    let result = integrator.integrate(&wrapped, None);

    VegasOutcome {
        integral: result.value,
        std_dev: result.error,
        chi2_per_it: result.chi2_dof,
    }
}
