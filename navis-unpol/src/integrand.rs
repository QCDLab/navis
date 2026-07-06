//! Wires `navis-core`'s phase-space mapping and this crate's matrix
//! elements into the `DPLUS`-equivalent VEGAS integrand.

use navis_core::constants::Params;
use navis_core::grid_setup::PertOrder;
use navis_core::kinematics::{map_phase_space, BinInputs, RunParams};
use navis_core::vegas::GridFill;

use crate::mes::{
    fbor, fdel1, fdel2, fresc1, fresc2, fvlo1, fvlo2, fvwpl1, fvwpl2, stru, MeContext,
};
use crate::pdfs::PdfFf;

/// Target-hadron selectors for `STRUCI`'s `ITAR` argument.
#[derive(Debug, Clone, Copy)]
pub struct Targets {
    pub ih1: i32,
    pub ih2: i32,
}

/// Channel indices touched by `FHODEL1`'s extra terms (`DATA IA1`), 0-indexed.
const IA1: [usize; 10] = [0, 2, 4, 5, 10, 11, 12, 13, 14, 15];
/// Channel indices touched by `FHODEL2`'s extra terms (`DATA IA2`), 0-indexed.
const IA2: [usize; 6] = [0, 2, 4, 10, 12, 13];
/// Channel indices touched by `FHOREST2`'s extra terms (`DATA IA3`), 0-indexed.
const IA3: [usize; 9] = [0, 2, 4, 7, 8, 9, 10, 12, 13];

/// One evaluation of the `DPLUS` integrand at VEGAS point `xx`, with this
/// point's total Monte Carlo weight `fill_weight` (see
/// `navis_core::vegas::vegas`'s doc comment for the calling convention --
/// it replaces the Fortran's separate `/calls*wgt/iter_max` grid-fill
/// normalization with one pre-combined factor).
///
/// `bin_width` is `PTUP - PTDO` for this bin.
#[allow(clippy::too_many_arguments)]
pub fn dplus(
    xx: &[f64; 5],
    fill_weight: f64,
    bin: &BinInputs,
    run: &RunParams,
    params: &Params,
    order: PertOrder,
    targets: Targets,
    pdf_ff: &PdfFf,
    bin_width: f64,
) -> (f64, Vec<GridFill>) {
    let ps = map_phase_space(xx, bin, run, true);

    let alpord = pdf_ff.alphas_q2(ps.q2mu);
    let alpasho = alpord * if order.is_nlo() { 1.0 } else { 0.0 };

    let ff = pdf_ff.strucf(ps.x3, ps.q2frag);
    let pdf1 = pdf_ff.struci(ps.bx1, ps.q2fac, targets.ih1);
    let pdf2 = pdf_ff.struci(ps.bx2, ps.q2fac, targets.ih2);
    let (grrt, grrc) = stru(&pdf1, &pdf2, &ff);

    let mut fills = Vec::new();

    // --- Born cross sections (FBOR calls + LO grid fills) ---
    let f01 = fbor(ps.v, ps.shd_born, params.ca, params.cf);
    let f02 = fbor(1.0 - ps.v, ps.shd_born, params.ca, params.cf);

    let mut born = [0.0_f64; 16];
    for j0 in 0..16 {
        fills.push(GridFill {
            order: PertOrder::born_index(),
            observable: ps.pt,
            channel: j0,
            ntuple: [ps.q2mu, ps.bx1, ps.bx2, ps.x3],
            weight: f01[j0] * fill_weight * ps.bx1 * ps.bx2 * ps.bxjac * ps.phase_space * bin_width,
        });
        fills.push(GridFill {
            order: PertOrder::born_index(),
            observable: ps.pt,
            channel: j0,
            ntuple: [ps.q2mu, ps.bx2, ps.bx1, ps.x3],
            weight: f02[j0] * fill_weight * ps.bx1 * ps.bx2 * ps.bxjac * ps.phase_space * bin_width,
        });
        born[j0] = (f01[j0] * grrt[j0] + f02[j0] * grrc[j0]) * ps.bxjac;
    }

    let mut ghd = 0.0_f64;
    let mut ghe = 0.0_f64;

    if order.is_nlo() {
        let ctx = MeContext {
            ca: params.ca,
            cf: params.cf,
            nf: params.nf,
            q2fac: ps.q2fac,
            q2mu: ps.q2mu,
            q2frag: ps.q2frag,
        };
        let un = 1.0_f64;

        let pdf1_nlo = pdf_ff.struci(ps.x1, ps.q2fac, targets.ih1);
        let (gppt, gppc) = stru(&pdf1_nlo, &pdf2, &ff);

        let mut fhodel1 = [0.0_f64; 16];
        for &j0 in &IA1 {
            let j0f = j0 + 1;
            fhodel1[j0] = (fdel1(ps.v, ps.x3, ps.gv, ps.gw, run.s, j0f, &ctx)
                + fvwpl1(un, ps.v, ps.x3, ps.gv, ps.gw, run.s, j0f, &ctx) * (1.0 - ps.wmin).ln()
                + fvlo1(un, ps.v, ps.x3, ps.gv, ps.gw, run.s, j0f, &ctx)
                    * (1.0 - ps.wmin).ln().powi(2)
                    / 2.0)
                / (1.0 - ps.v)
                - (ps.wmax - ps.wmin) / (1.0 - ps.v)
                    * (fvwpl1(un, ps.v, ps.x3, ps.gv, ps.gw, run.s, j0f, &ctx)
                        + fvlo1(un, ps.v, ps.x3, ps.gv, ps.gw, run.s, j0f, &ctx)
                            * (1.0 - ps.w).ln())
                    / (1.0 - ps.w);
            fhodel1[j0] = fhodel1[j0] / params.cc[j0] * ps.bxjac;
        }

        let mut fhodel2 = fhodel1;
        for &j0 in &IA2 {
            let j0f = j0 + 1;
            fhodel2[j0] = (fdel2(ps.v, ps.x3, ps.gv, ps.gw, run.s, j0f, &ctx)
                + fvwpl2(un, ps.v, ps.x3, ps.gv, ps.gw, run.s, j0f, &ctx) * (1.0 - ps.wmin).ln()
                + fvlo2(un, ps.v, ps.x3, ps.gv, ps.gw, run.s, j0f, &ctx)
                    * (1.0 - ps.wmin).ln().powi(2)
                    / 2.0)
                / (1.0 - ps.v)
                - (ps.wmax - ps.wmin) / (1.0 - ps.v)
                    * (fvwpl2(un, ps.v, ps.x3, ps.gv, ps.gw, run.s, j0f, &ctx)
                        + fvlo2(un, ps.v, ps.x3, ps.gv, ps.gw, run.s, j0f, &ctx)
                            * (1.0 - ps.w).ln())
                    / (1.0 - ps.w);
            fhodel2[j0] = fhodel2[j0] / params.cc[j0] * ps.bxjac;
        }

        let shd_nlo = ps.shd_nlo(run);
        let pre1 = crate::mes::precalc(ps.v, ps.w, shd_nlo, &ctx);

        let mut fhorest1 = [0.0_f64; 16];
        (0..16).for_each(|j0| {
            let j0f = j0 + 1;
            fhorest1[j0] = (ps.wmax - ps.wmin) / (1.0 - ps.v)
                * (fvwpl1(ps.w, ps.v, ps.x3, ps.gv, ps.gw, run.s, j0f, &ctx) / ps.w / (1.0 - ps.w)
                    + fvlo1(ps.w, ps.v, ps.x3, ps.gv, ps.gw, run.s, j0f, &ctx) / ps.w
                        * (1.0 - ps.w).ln()
                        / (1.0 - ps.w)
                    + fresc1(ps.w, ps.v, ps.x3, ps.gv, ps.gw, run.s, j0f, &ctx, &pre1) / ps.w);
            fhorest1[j0] = fhorest1[j0] / params.cc[j0] * ps.xjac;
        });

        let mut fhorest2 = fhorest1;
        let vx = 1.0 - ps.v * ps.w;
        let wx = (1.0 - ps.v) / (1.0 - ps.v * ps.w);
        let pre2 = crate::mes::precalc(vx, wx, shd_nlo, &ctx);
        for &j0 in &IA3 {
            let j0f = j0 + 1;
            fhorest2[j0] = (ps.wmax - ps.wmin) / (1.0 - ps.v)
                * (fvwpl2(ps.w, ps.v, ps.x3, ps.gv, ps.gw, run.s, j0f, &ctx) / ps.w / (1.0 - ps.w)
                    + fvlo2(ps.w, ps.v, ps.x3, ps.gv, ps.gw, run.s, j0f, &ctx) / ps.w
                        * (1.0 - ps.w).ln()
                        / (1.0 - ps.w)
                    + fresc2(ps.w, ps.v, ps.x3, ps.gv, ps.gw, run.s, j0f, &ctx, &pre2) / ps.w);
            fhorest2[j0] = fhorest2[j0] / params.cc[j0] * ps.xjac;
        }

        for j0 in 0..16 {
            fills.push(GridFill {
                order: PertOrder::nlo_index(),
                observable: ps.pt,
                channel: j0,
                ntuple: [ps.q2mu, ps.bx1, ps.bx2, ps.x3],
                weight: fhodel1[j0] * fill_weight * ps.bx1 * ps.x2 * ps.phase_space * bin_width,
            });
            fills.push(GridFill {
                order: PertOrder::nlo_index(),
                observable: ps.pt,
                channel: j0,
                ntuple: [ps.q2mu, ps.bx2, ps.bx1, ps.x3],
                weight: fhodel2[j0] * fill_weight * ps.bx1 * ps.x2 * ps.phase_space * bin_width,
            });
            fills.push(GridFill {
                order: PertOrder::nlo_index(),
                observable: ps.pt,
                channel: j0,
                ntuple: [ps.q2mu, ps.x1, ps.x2, ps.x3],
                weight: fhorest1[j0] * fill_weight * ps.x1 * ps.x2 * ps.phase_space * bin_width,
            });
            fills.push(GridFill {
                order: PertOrder::nlo_index(),
                observable: ps.pt,
                channel: j0,
                ntuple: [ps.q2mu, ps.x2, ps.x1, ps.x3],
                weight: fhorest2[j0] * fill_weight * ps.x1 * ps.x2 * ps.phase_space * bin_width,
            });

            ghd += (fhodel1[j0] * grrt[j0] + fhodel2[j0] * grrc[j0]) * alpasho + born[j0];
            ghe += (fhorest1[j0] * gppt[j0] + fhorest2[j0] * gppc[j0]) * alpasho;
        }
    } else {
        (0..16).for_each(|j0| {
            ghd += born[j0];
        });
    }

    let dplus = (ghd + ghe) * alpord.powi(2) * ps.phase_space;

    (dplus, fills)
}
