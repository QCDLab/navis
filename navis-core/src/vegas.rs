//! Port of G.P. Lepage's adaptive Monte Carlo integrator from `intvegas.f`
//!
//! Differences from the Fortran source, all deliberate:
//!
//! - Every call starts from a fresh, uniform grid. In the Fortran code this
//!   is also true in practice (the main programs always invoke the `vegas`
//!   entry point, never `vegas1`/`vegas2`/`vegas3` directly, and `ndo`
//!   starts at 1 on a fresh call), which lets the general "regrid an
//!   existing non-trivial grid" routine collapse to simply placing `nd`
//!   equal-width bins — this is proven by hand-simulating the Fortran
//!   rebin loop starting from `ndo = 1`.
//! - `xjac` is hardcoded to `1.0`: the integration domain is always the
//!   unit hypercube (`xl = 0`, `xu = 1` for every dimension in the Fortran
//!   `DATA xl,xu/10*0.,10*1./`), so the Jacobian of that domain is trivially 1.
//! - Random numbers come from `rand`'s `SmallRng`, not Fortran's `ranf`;
//!   only statistical equivalence is required (see project plan).
//! - Point evaluations within one iteration are independent aside from
//!   accumulating into shared reduction state, so they're parallelized with
//!   `rayon` (fold-per-task, then reduce). PineAPPL grid fills — a side
//!   effect of the integrand on *every* point, not just the final average —
//!   are collected the same way and applied to the `Grid` once per
//!   iteration (bounding memory to one iteration's worth rather than the
//!   whole run).

use pineappl::grid::Grid;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// VEGAS always integrates in 5 dimensions in `sihp-pp`
/// (`CALL VEGAS(DPLUS,1.D-6,5,NCALLS,ITMAX,0,0,GRID)`): `W`, `V`, `X3`, and
/// up to two more for the rapidity/pT integration.
pub const VEGAS_NDIM: usize = 5;

const NDMX: usize = 50;
const ALPH: f64 = 1.5;

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

/// Final result of a VEGAS run: the weighted-average integral estimate,
/// its standard deviation, and the chi-squared per iteration, matching
/// `ERG1`/`ERG2`/`ERG3` (`COMMON /RESULT/`) as read by `FUNCDG` in
/// `hadrive-ms.f`.
#[derive(Debug, Clone, Copy)]
pub struct VegasOutcome {
    pub integral: f64,
    pub std_dev: f64,
    pub chi2_per_it: f64,
}

#[derive(Clone)]
struct CellAccum {
    ti: f64,
    tsi: f64,
    d: Vec<[f64; VEGAS_NDIM]>,
    di: Vec<[f64; VEGAS_NDIM]>,
    nxi: Vec<[u32; VEGAS_NDIM]>,
    fills: Vec<GridFill>,
}

impl CellAccum {
    fn new(nd: usize) -> Self {
        Self {
            ti: 0.0,
            tsi: 0.0,
            d: vec![[0.0; VEGAS_NDIM]; nd],
            di: vec![[0.0; VEGAS_NDIM]; nd],
            nxi: vec![[0; VEGAS_NDIM]; nd],
            fills: Vec::new(),
        }
    }

    fn merge(mut self, mut other: Self) -> Self {
        self.ti += other.ti;
        self.tsi += other.tsi;
        for i in 0..self.d.len() {
            for j in 0..VEGAS_NDIM {
                self.d[i][j] += other.d[i][j];
                self.di[i][j] += other.di[i][j];
                self.nxi[i][j] += other.nxi[i][j];
            }
        }
        self.fills.append(&mut other.fills);
        self
    }
}

/// Run VEGAS on `integrand`, filling `grid` as a side effect (once per
/// iteration) and returning the final estimate.
///
/// `integrand(xx, wgt, calls)` mirrors `DPLUS(XX,grid,calls,wgt)`: it
/// receives the mapped point `xx` together with the VEGAS point weight and
/// the total number of calls for the iteration (both needed to reproduce
/// `DPLUS`'s own `/calls*wgt` grid-fill normalization), and must return the
/// *unscaled* integrand value — VEGAS itself multiplies by `wgt` to form the
/// Monte Carlo estimator — plus any grid-fill records for that point.
#[allow(clippy::too_many_arguments)]
pub fn vegas<F>(
    integrand: F,
    bcc: f64,
    ncall: i64,
    itmx: usize,
    seed: u64,
    grid: &mut Grid,
) -> VegasOutcome
where
    F: Fn(&[f64; VEGAS_NDIM], f64, f64) -> (f64, Vec<GridFill>) + Sync,
{
    let ndim = VEGAS_NDIM;

    // --- one-time grid setup (Fortran labels before `9  it=it+1`) ---
    let ng_formula = ((ncall as f64 * 0.5).powf(1.0 / ndim as f64)) as i64;
    let (ng, nd, mds_ge_zero) = if 2 * ng_formula - NDMX as i64 >= 0 {
        let npg0 = ng_formula / NDMX as i64 + 1;
        let nd0 = (ng_formula / npg0).max(1);
        (npg0 * nd0, nd0, false)
    } else {
        (ng_formula.max(1), NDMX as i64, true)
    };
    let ng = ng as usize;
    let nd = nd as usize;

    let total_cells: usize = ng.pow(ndim as u32);
    let npg = ((ncall / total_cells as i64).max(2)) as usize;
    let calls = (npg * total_cells) as f64;

    let dxg_initial = 1.0 / ng as f64;
    let dv2g = dxg_initial.powi(2 * ndim as i32) / (npg as f64) / (npg as f64) / (npg as f64 - 1.0);
    let xnd = nd as f64;
    let ndm = nd - 1;
    let dxg = dxg_initial * xnd;

    // xi[0][j] = 0.0 is a padding entry standing in for the Fortran code's
    // implicit `xi(0,j) = 0`; xi[1..=nd][j] are the real grid edges,
    // initially `nd` equal-width bins (see module docs).
    let mut xi = vec![[0.0_f64; VEGAS_NDIM]; nd + 1];
    for j in 0..ndim {
        for (i, row) in xi.iter_mut().enumerate().take(nd + 1).skip(1) {
            row[j] = i as f64 / xnd;
        }
    }

    let mut si = 0.0_f64;
    let mut si2 = 0.0_f64;
    let mut swgt = 0.0_f64;
    let mut schi = 0.0_f64;

    let mut avgi = 0.0_f64;
    let mut sd = 0.0_f64;
    let mut chi2a = 0.0_f64;

    for it in 1..=itmx {
        let xi_ref = &xi;
        let mut result: CellAccum = (0..total_cells)
            .into_par_iter()
            .fold(
                || CellAccum::new(nd),
                |mut acc, cell_idx| {
                    // Decode the cell's multi-index. The Fortran odometer
                    // increments the last dimension fastest; cell
                    // enumeration order doesn't affect the Monte Carlo
                    // result (only RNG consumption order, which we don't
                    // need to match), so any decoding works.
                    let mut kg = [0usize; VEGAS_NDIM];
                    let mut rem = cell_idx;
                    for j in (0..ndim).rev() {
                        kg[j] = rem % ng;
                        rem /= ng;
                    }

                    let mut rng =
                        SmallRng::seed_from_u64(seed ^ ((it as u64) << 48) ^ (cell_idx as u64));

                    let mut fb = 0.0_f64;
                    let mut f2b = 0.0_f64;
                    let mut ia = [1usize; VEGAS_NDIM];

                    for _ in 0..npg {
                        let mut x = [0.0_f64; VEGAS_NDIM];
                        let mut wgt = 1.0_f64; // xjac = 1
                        for j in 0..ndim {
                            let qran: f64 = rng.gen();
                            let xn = (kg[j] as f64 + 1.0 - qran) * dxg + 1.0;
                            let iaj = (xn as i64).clamp(1, nd as i64) as usize;
                            ia[j] = iaj;
                            let xo = xi_ref[iaj][j] - xi_ref[iaj - 1][j];
                            let rc = xi_ref[iaj - 1][j] + (xn - iaj as f64) * xo;
                            x[j] = rc; // xl = 0, dx = 1
                            wgt *= xo * xnd;
                        }

                        let (raw, fills) = integrand(&x, wgt, calls);
                        let f = raw * wgt;
                        let f2 = f * f;
                        fb += f;
                        f2b += f2;

                        (0..ndim).for_each(|j| {
                            let i0 = ia[j] - 1;
                            acc.di[i0][j] += f / calls;
                            acc.nxi[i0][j] += 1;
                            if mds_ge_zero {
                                acc.d[i0][j] += f2;
                            }
                        });
                        acc.fills.extend(fills);
                    }

                    let mut f2b_var = (f2b * npg as f64).sqrt();
                    f2b_var = (f2b_var - fb) * (f2b_var + fb);

                    acc.ti += fb;
                    acc.tsi += f2b_var;

                    if !mds_ge_zero {
                        (0..ndim).for_each(|j| {
                            let i0 = ia[j] - 1;
                            acc.d[i0][j] += f2b_var;
                        });
                    }

                    acc
                },
            )
            .reduce(|| CellAccum::new(nd), CellAccum::merge);

        // Apply this iteration's grid fills, then drop them before moving
        // on to bound peak memory to one iteration's worth.
        for fill in result.fills.drain(..) {
            grid.fill(
                fill.order,
                fill.observable,
                fill.channel,
                &fill.ntuple,
                fill.weight,
            );
        }

        let ti = result.ti / calls;
        let tsi = result.tsi * dv2g;
        let ti2 = ti * ti;
        let iter_weight = ti2 / tsi;

        si += ti * iter_weight;
        si2 += ti2;
        swgt += iter_weight;
        schi += ti2 * iter_weight;

        avgi = si / swgt;
        sd = swgt * it as f64 / si2;
        chi2a = if it > 1 {
            sd * (schi / swgt - avgi * avgi) / (it as f64 - 1.0)
        } else {
            0.0
        };
        sd = (1.0 / sd).sqrt();

        // --- grid refinement (Fortran labels 23..28) ---
        let mut dt = [0.0_f64; VEGAS_NDIM];
        (0..ndim).for_each(|j| {
            for i in 0..nd {
                if result.nxi[i][j] > 0 {
                    result.d[i][j] /= f64::from(result.nxi[i][j]);
                }
                dt[j] += result.d[i][j];
            }
        });

        for j in 0..ndim {
            let mut r = vec![0.0_f64; nd];
            let mut rc = 0.0_f64;
            (0..nd).for_each(|i| {
                if result.d[i][j] > 0.0 {
                    let xo = dt[j] / result.d[i][j];
                    r[i] = ((xo - 1.0) / xo / xo.ln()).powf(ALPH);
                }
                rc += r[i];
            });
            rc /= xnd;

            let mut xin = vec![0.0_f64; ndm];
            let mut k = 0usize;
            let mut xn = 0.0_f64;
            let mut xo_prev = 0.0_f64;
            let mut dr = 0.0_f64;
            let mut i = 0usize;
            while i < ndm {
                while rc > dr {
                    k += 1;
                    dr += r[k - 1];
                    xo_prev = xn;
                    xn = xi[k][j];
                }
                dr -= rc;
                xin[i] = xn - (xn - xo_prev) * dr / r[k - 1];
                i += 1;
            }
            for (idx, &value) in xin.iter().enumerate() {
                xi[idx + 1][j] = value;
            }
            xi[nd][j] = 1.0;
        }

        if it >= itmx || bcc.abs() >= (sd / avgi).abs() {
            break;
        }
    }

    VegasOutcome {
        integral: avgi,
        std_dev: sd,
        chi2_per_it: chi2a,
    }
}
