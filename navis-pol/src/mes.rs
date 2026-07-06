//! Polarized partonic matrix elements.

/// Scale/color context threaded through the matrix elements.
#[derive(Debug, Clone, Copy)]
pub struct MeContext {
    pub ca: f64,
    pub cf: f64,
    pub nf: f64,
    pub pi: f64,
    pub q2fac: f64,
    pub q2mu: f64,
    pub q2frag: f64,
}

/// `FBOR(V,SHD,F0)`: Born cross sections (including the `1/v/(1-v)` phase
/// space factor), one entry per of the 16 channels shared by this
/// function, [`avwpl`], [`avdel`], [`avlo`], and [`struv`] (1-indexed,
/// matching the Fortran `J0`), identified from the flavor bookkeeping in
/// [`stru`] (confirmed byte-for-byte identical to the unpolarized
/// package's `stru`):
///
/// 01. `qq' -> qq'`.
/// 02. `qq' -> qq'`.
/// 03. `qqbar' -> qqbar'`.
/// 04. `qqbar' -> qqbar'`.
/// 05. `qqbar -> q'qbar'`.
/// 06. `qq -> qq`.
/// 07. `qq -> qq`.
/// 08. `qg -> q q'qbar'`.
/// 09. `qg -> q q'qbar'`.
/// 10. `qg -> q qqbar`.
/// 11. `qqbar -> qqbar`.
/// 12. `qqbar -> gg`.
/// 13. `qg -> qg`.
/// 14. `qg -> qg`.
/// 15. `gg -> gg`.
/// 16. `gg -> qqbar`.
#[must_use]
pub fn fbor(v: f64, shd: f64, nc: f64, cf: f64) -> [f64; 16] {
    let v2 = v.powi(2);
    let vc = nc * nc - 1.0;
    let nc2 = nc * nc;
    let vm = 1.0 - v;
    let vm2 = vm.powi(2);
    let prelo = std::f64::consts::PI / shd / v / vm;

    let mut f0 = [0.0_f64; 16];

    f0[0] = cf / nc * prelo * (-v2 + 1.0) / vm2;
    f0[1] = 0.0;
    f0[2] = cf / nc * prelo * (-v2 + 1.0) / vm2;
    f0[3] = 0.0;
    f0[4] = -cf / nc * prelo * (2.0 * v2 - 2.0 * v + 1.0);
    f0[5] = 2.0 * cf / nc2 * prelo * vm * v * (2.0 - 3.0 * v + 3.0 * v2) / v2 / vm2;
    f0[6] = 0.0;
    f0[7] = 0.0;
    f0[8] = 0.0;
    f0[9] = 0.0;
    f0[10] = 2.0 * cf / nc2 * prelo * v * vm * (6.0 - 7.0 * v + 3.0 * v2) / vm2;
    f0[11] = -cf / nc2
        * prelo
        * (2.0 * v2 - 2.0 * v + 1.0)
        * (2.0 * nc2 * v2 - 2.0 * nc2 * v + nc2 - 1.0)
        / v
        / vm;
    f0[12] =
        prelo / (2.0 * nc2) * (-v2 + 1.0) * ((nc2 - 1.0) * v2 + 2.0 * v + (nc2 - 1.0)) / v / vm2;
    f0[13] = prelo / (2.0 * nc2) * (2.0 * v - v2) * ((nc2 - 1.0) * v2 - 2.0 * nc2 * v + 2.0 * nc2)
        / v2
        / vm;
    f0[14] = (4.0 * nc2) / vc * prelo * (1.0 - v + v2) * (2.0 - v + v2) / v / vm;
    f0[15] = -prelo / (2.0 * nc) / vc * (v2 + vm2) * (2.0 * nc2 * (v2 - v) + nc2 - 1.0) / v / vm;

    f0
}

/// `PRECALC(V,W,S)`: precomputed powers of `V`/`W` and logs of the scale
/// ratios, shared by `STRUV`/`STRUV1..16`.
#[derive(Debug, Clone, Copy)]
pub struct Precalc {
    pub l1v: f64,
    pub lv: f64,
    pub l1w: f64,
    pub lw: f64,
    pub lvw: f64,
    pub l1vw: f64,
    pub lmu: f64,
    pub lms: f64,
    pub lmss: f64,
    pub cacf: f64,
    pub ca2: f64,
    pub ca4: f64,
    pub v2: f64,
    pub v3: f64,
    pub v4: f64,
    pub v5: f64,
    pub v6: f64,
    pub v7: f64,
    pub v8: f64,
    pub v9: f64,
    pub v10: f64,
    pub v11: f64,
    pub v12: f64,
    pub w2: f64,
    pub w3: f64,
    pub w4: f64,
    pub w5: f64,
    pub w6: f64,
    pub w7: f64,
    pub w8: f64,
    pub w9: f64,
    pub w10: f64,
    pub w11: f64,
    pub w12: f64,
}

#[must_use]
pub fn precalc(v: f64, w: f64, s: f64, ctx: &MeContext) -> Precalc {
    Precalc {
        l1v: (1.0 - v).ln(),
        lv: v.ln(),
        l1w: (1.0 - w).ln(),
        lw: w.ln(),
        lvw: ((1.0 - v) / (1.0 - v * w)).ln() / (1.0 - w),
        l1vw: (1.0 - v + v * w).ln() / (1.0 - w),
        lmu: (ctx.q2mu / s).ln(),
        lms: (ctx.q2fac / s).ln(),
        lmss: (ctx.q2frag / s).ln(),
        cacf: ctx.ca * ctx.cf,
        ca2: ctx.ca.powi(2),
        ca4: ctx.ca.powi(4),
        v2: v.powi(2),
        v3: v.powi(3),
        v4: v.powi(4),
        v5: v.powi(5),
        v6: v.powi(6),
        v7: v.powi(7),
        v8: v.powi(8),
        v9: v.powi(9),
        v10: v.powi(10),
        v11: v.powi(11),
        v12: v.powi(12),
        w2: w.powi(2),
        w3: w.powi(3),
        w4: w.powi(4),
        w5: w.powi(5),
        w6: w.powi(6),
        w7: w.powi(7),
        w8: w.powi(8),
        w9: w.powi(9),
        w10: w.powi(10),
        w11: w.powi(11),
        w12: w.powi(12),
    }
}

/// `FDEL1(V,X3)`.
#[must_use]
pub fn fdel1(v: f64, x3: f64, gv: f64, gw: f64, gs: f64, j0: usize, ctx: &MeContext) -> f64 {
    let bx1 = gv * gw / v / x3;
    let bx2 = (1.0 - gv) / (1.0 - v) / x3;
    let shd = bx1 * bx2 * gs;
    let fkel = avdel(j0, v, shd, ctx);
    fkel / shd
}

/// `FDEL2(V,X3)`.
#[must_use]
pub fn fdel2(v: f64, x3: f64, gv: f64, gw: f64, gs: f64, j0: usize, ctx: &MeContext) -> f64 {
    let bx1 = gv * gw / v / x3;
    let bx2 = (1.0 - gv) / (1.0 - v) / x3;
    let shd = bx1 * bx2 * gs;
    let un = 1.0;
    let fkelc = (avdel(j0, 1.0 - v, shd, ctx)
        + (v / (1.0 - v)).ln() * avwpl(j0, un, 1.0 - v, shd, ctx)
        + 0.5 * avlo(j0, un, 1.0 - v, shd, ctx) * ((1.0 - v) / v).ln().powi(2))
        * (1.0 - v)
        / v;
    fkelc / shd
}

/// `FVWPL1(W,V,X3)`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fvwpl1(
    w: f64,
    v: f64,
    x3: f64,
    gv: f64,
    gw: f64,
    gs: f64,
    j0: usize,
    ctx: &MeContext,
) -> f64 {
    let x1 = gv * gw / v / w / x3;
    let x2 = (1.0 - gv) / (1.0 - v) / x3;
    let sh = x1 * x2 * gs;
    let rvwpl = avwpl(j0, w, v, sh, ctx);
    rvwpl / sh
}

/// `FVWPL2(W,V,X3)`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fvwpl2(
    w: f64,
    v: f64,
    x3: f64,
    gv: f64,
    gw: f64,
    gs: f64,
    j0: usize,
    ctx: &MeContext,
) -> f64 {
    let x1 = gv * gw / v / w / x3;
    let x2 = (1.0 - gv) / (1.0 - v) / x3;
    let vx = 1.0 - v * w;
    let wx = (1.0 - v) / (1.0 - v * w);
    let sh = x1 * x2 * gs;
    let rvwplc = (avwpl(j0, wx, vx, sh, ctx) + avlo(j0, wx, vx, sh, ctx) * (v / vx).ln()) * vx / v;
    rvwplc / sh
}

/// `FVLO1(W,V,X3)`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fvlo1(
    w: f64,
    v: f64,
    x3: f64,
    gv: f64,
    gw: f64,
    gs: f64,
    j0: usize,
    ctx: &MeContext,
) -> f64 {
    let x1 = gv * gw / v / w / x3;
    let x2 = (1.0 - gv) / (1.0 - v) / x3;
    let sh = x1 * x2 * gs;
    let rvwlo = avlo(j0, w, v, sh, ctx);
    rvwlo / sh
}

/// `FVLO2(W,V,X3)`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fvlo2(
    w: f64,
    v: f64,
    x3: f64,
    gv: f64,
    gw: f64,
    gs: f64,
    j0: usize,
    ctx: &MeContext,
) -> f64 {
    let x1 = gv * gw / v / w / x3;
    let x2 = (1.0 - gv) / (1.0 - v) / x3;
    let vx = 1.0 - v * w;
    let wx = (1.0 - v) / (1.0 - v * w);
    let sh = x1 * x2 * gs;
    let rvwloc = avlo(j0, wx, vx, sh, ctx) * vx / v;
    rvwloc / sh
}

/// `FRESC1(W,V,X3)`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fresc1(
    w: f64,
    v: f64,
    x3: f64,
    gv: f64,
    gw: f64,
    gs: f64,
    j0: usize,
    ctx: &MeContext,
    pre: &Precalc,
) -> f64 {
    let x1 = gv * gw / v / w / x3;
    let x2 = (1.0 - gv) / (1.0 - v) / x3;
    let sh = x1 * x2 * gs;
    let rresc = struv(j0, w, v, x3, sh, ctx, pre);
    rresc / sh
}

/// `FRESC2(W,V,X3)`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fresc2(
    w: f64,
    v: f64,
    x3: f64,
    gv: f64,
    gw: f64,
    gs: f64,
    j0: usize,
    ctx: &MeContext,
    pre: &Precalc,
) -> f64 {
    let x1 = gv * gw / v / w / x3;
    let x2 = (1.0 - gv) / (1.0 - v) / x3;
    let vx = 1.0 - v * w;
    let wx = (1.0 - v) / (1.0 - v * w);
    let sh = x1 * x2 * gs;
    let rrescc = struv(j0, wx, vx, x3, sh, ctx, pre);
    rrescc / sh
}

/// `AVWPL(W,V,S)`: all `1/(1-W)+` pieces, selected by channel `j0`. See
/// the channel table on [`fbor`] for what each index physically
/// represents. Unlike the unpolarized version,
/// `M=DSQRT(Q2FAC)`/`MP=DSQRT(Q2FRAG)` collapse to the same
/// `LMS=ln(Q2FAC/S)`/`LMSS=ln(Q2FRAG/S)` (squaring undoes the square root),
/// so those are computed directly rather than via an intermediate mass.
#[must_use]
pub fn avwpl(j0: usize, _w: f64, v: f64, s: f64, ctx: &MeContext) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let l1v = (1.0 - v).ln();
    let lv = v.ln();
    let lms = (ctx.q2fac / s).ln();
    let lmss = (ctx.q2frag / s).ln();
    let nf = ctx.nf; // Fortran recomputes `Nf = 2.*GTR` where `GTR = NF/2`.

    match j0 {
        // 1: qq' -> qq' (unlike flavor, elastic), hadron from quark
        1 => {
            (-6.0 * ca * cf.powi(2) * (1.0 + v)) / ((1.0 - v) * v)
                - (8.0 * (1.0 + ca.powi(2)) * cf * l1v * (1.0 + v)) / ((1.0 - v) * v)
                - (16.0 * ca * cf.powi(2) * lms * (1.0 + v)) / ((1.0 - v) * v)
                - (8.0 * ca * cf.powi(2) * lmss * (1.0 + v)) / ((1.0 - v) * v)
                - (4.0 * (11.0 - 7.0 * ca.powi(2)) * cf * lv * (1.0 + v)) / ((1.0 - v) * v)
        }
        // 2: qq' -> qq' + extra gluon, hadron from gluon
        2 => 0.0,
        // 3: qqbar' -> qqbar' (unlike flavor, elastic), hadron from quark
        3 => {
            (-6.0 * ca * cf.powi(2) * (1.0 + v)) / ((1.0 - v) * v)
                - (8.0 * (1.0 + ca.powi(2)) * cf * l1v * (1.0 + v)) / ((1.0 - v) * v)
                - (16.0 * ca * cf.powi(2) * lms * (1.0 + v)) / ((1.0 - v) * v)
                - (8.0 * ca * cf.powi(2) * lmss * (1.0 + v)) / ((1.0 - v) * v)
                + (4.0 * (5.0 + 3.0 * ca.powi(2)) * cf * lv * (1.0 + v)) / ((1.0 - v) * v)
        }
        // 4: qqbar' -> qqbar' + extra gluon, hadron from gluon
        4 => 0.0,
        // 5: qqbar -> q'qbar' (annihilation, unlike flavor), hadron from quark
        5 => {
            let t1 = 1.0 - 2.0 * v + 2.0 * v.powi(2);
            (6.0 * ca * cf.powi(2) * (t1)) / v
                + (8.0 * (3.0 - ca.powi(2)) * cf * l1v * (t1)) / v
                + (16.0 * ca * cf.powi(2) * lms * (t1)) / v
                + (8.0 * ca * cf.powi(2) * lmss * (t1)) / v
                - (4.0 * (5.0 + 3.0 * ca.powi(2)) * cf * lv * (t1)) / v
        }
        // 6: qq -> qq (identical flavor, elastic), hadron from quark
        6 => {
            let t1 = (1.0 - v) * v.powi(2);
            (16.0
                * cf
                * l1v
                * (1.0 - 3.0 * ca + ca.powi(2) + ca.powi(3) + 4.0 * ca * v
                    - 2.0 * ca.powi(3) * v
                    - 2.0 * ca * v.powi(2)))
                / (ca * t1)
                + (12.0 * cf.powi(2) * (1.0 - ca + ca * v - ca * v.powi(2))) / (t1)
                + (32.0 * cf.powi(2) * lms * (1.0 - ca + ca * v - ca * v.powi(2))) / (t1)
                + (16.0 * cf.powi(2) * lmss * (1.0 - ca + ca * v - ca * v.powi(2))) / (t1)
                + (8.0
                    * cf
                    * lv
                    * (7.0 - 7.0 * ca - 3.0 * ca.powi(2) + 3.0 * ca.powi(3) + 5.0 * ca * v
                        - ca.powi(3) * v
                        - 9.0 * ca * v.powi(2)
                        + 5.0 * ca.powi(3) * v.powi(2)))
                    / (ca * t1)
        }
        7..=10 => 0.0,
        // 11: qqbar -> qqbar (same flavor, elastic + annihilation), hadron from quark
        11 => {
            let t1 = 2.0 * ca - v - 2.0 * ca * v + ca * v.powi(2);
            (-12.0 * cf.powi(2) * (t1)) / (1.0 - v)
                - (32.0 * cf.powi(2) * lms * (t1)) / (1.0 - v)
                - (16.0 * cf.powi(2) * lmss * (t1)) / (1.0 - v)
                + (8.0
                    * cf
                    * lv
                    * (10.0 * ca + 6.0 * ca.powi(3)
                        - v
                        - 10.0 * ca * v
                        - 7.0 * ca.powi(2) * v
                        - 6.0 * ca.powi(3) * v
                        + 5.0 * ca * v.powi(2)
                        + 3.0 * ca.powi(3) * v.powi(2)))
                    / (ca * (1.0 - v))
                + (16.0
                    * cf
                    * l1v
                    * (ca - ca.powi(3) - 5.0 * ca * v + ca.powi(3) * v + v.powi(2)
                        - 6.0 * ca * v.powi(2)
                        + ca.powi(2) * v.powi(2)
                        - 2.0 * ca.powi(3) * v.powi(2)
                        - 3.0 * ca * v.powi(3)
                        + ca.powi(3) * v.powi(3)))
                    / (ca * (1.0 - v) * v)
        }
        // 12: qqbar -> gg (same-flavor annihilation), hadron from gluon
        12 => {
            let t1 = 1.0 - 2.0 * v + 2.0 * v.powi(2);
            (16.0 * ca.powi(2) * cf * lmss * (t1) * (cf - ca * v + ca * v.powi(2)))
                / ((1.0 - v) * v.powi(2))
                - (22.0
                    * ca
                    * cf
                    * (t1)
                    * (-2.0 * ca * cf + 2.0 * ca.powi(2) * v - 2.0 * ca.powi(2) * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(2))
                - (16.0
                    * cf.powi(2)
                    * lms
                    * (t1)
                    * (-2.0 * ca * cf + 2.0 * ca.powi(2) * v - 2.0 * ca.powi(2) * v.powi(2)))
                    / ((1.0 - v) * v.powi(2))
                + (4.0
                    * cf
                    * nf
                    * (t1)
                    * (-2.0 * ca * cf + 2.0 * ca.powi(2) * v - 2.0 * ca.powi(2) * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(2))
                + (8.0
                    * cf
                    * l1v
                    * (t1)
                    * (1.0 + ca.powi(4) + 2.0 * ca.powi(2) * v
                        - 2.0 * ca.powi(4) * v
                        - 2.0 * ca.powi(2) * v.powi(2)))
                    / (ca * (1.0 - v) * v.powi(2))
                - (8.0
                    * cf
                    * lv
                    * (t1)
                    * (1.0 - 5.0 * ca.powi(2) + 4.0 * ca.powi(4) + 2.0 * ca.powi(2) * v
                        - 8.0 * ca.powi(4) * v
                        - 2.0 * ca.powi(2) * v.powi(2)
                        + 6.0 * ca.powi(4) * v.powi(2)))
                    / (ca * (1.0 - v) * v.powi(2))
        }
        // 13: qg -> qg (Compton), hadron from quark
        13 => {
            let t1 = -2.0 * ca * cf - 2.0 * v + v.powi(2) - ca.powi(2) * v.powi(2);
            (22.0 * ca * cf * (1.0 + v) * (t1)) / (3.0 * (1.0 - v) * v.powi(2))
                - (4.0 * (1.0 - 3.0 * ca.powi(2)) * cf * lms * (1.0 + v) * (t1))
                    / (ca * (1.0 - v) * v.powi(2))
                - (4.0 * cf * nf * (1.0 + v) * (t1)) / (3.0 * (1.0 - v) * v.powi(2))
                - (8.0
                    * cf.powi(2)
                    * lmss
                    * (1.0 + v)
                    * (2.0 * ca * cf + 2.0 * v - v.powi(2) + ca.powi(2) * v.powi(2)))
                    / ((1.0 - v) * v.powi(2))
                + (8.0
                    * cf
                    * l1v
                    * (1.0 + v)
                    * (1.0 + 2.0 * ca.powi(2) - ca.powi(4) - 2.0 * v - 6.0 * ca.powi(2) * v
                        + v.powi(2)
                        + 2.0 * ca.powi(2) * v.powi(2)
                        - ca.powi(4) * v.powi(2)))
                    / (ca * (1.0 - v) * v.powi(2))
                + (8.0
                    * cf
                    * lv
                    * (1.0 + v)
                    * (1.0 - 5.0 * ca.powi(2) + 4.0 * ca.powi(4) - 2.0 * v
                        + 8.0 * ca.powi(2) * v
                        + v.powi(2)
                        - 5.0 * ca.powi(2) * v.powi(2)
                        + 2.0 * ca.powi(4) * v.powi(2)))
                    / (ca * (1.0 - v) * v.powi(2))
        }
        // 14: qg -> qg (Compton), hadron from gluon
        14 => {
            let t1 = (1.0 - v) * v.powi(2);
            (-16.0 * ca.powi(3) * cf * l1v * (1.0 - v) * (2.0 - v)) / v.powi(2)
                - (6.0
                    * cf.powi(2)
                    * (2.0 - v)
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / (t1)
                + (4.0
                    * (1.0 - 3.0 * ca.powi(2))
                    * cf
                    * lms
                    * (2.0 - v)
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / (ca * t1)
                - (8.0
                    * ca
                    * cf
                    * lmss
                    * (2.0 - v)
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / (t1)
                - (4.0
                    * cf
                    * lv
                    * (2.0 - v)
                    * (2.0 * ca.powi(2) - 18.0 * ca.powi(4) - 2.0 * ca.powi(2) * v
                        + 18.0 * ca.powi(4) * v
                        - v.powi(2)
                        + 6.0 * ca.powi(2) * v.powi(2)
                        - 9.0 * ca.powi(4) * v.powi(2)))
                    / (ca * t1)
        }
        // 15: gg -> gg, hadron from gluon
        15 => {
            let t1 = 2.0 - v + v.powi(2);
            (-256.0 * ca.powi(3) * l1v * (1.0 - v) * (t1)) / v.powi(2)
                - (704.0 * ca.powi(3) * (1.0 - v + v.powi(2)) * (t1))
                    / (3.0 * (1.0 - v) * v.powi(2))
                - (512.0 * ca.powi(3) * lms * (1.0 - v + v.powi(2)) * (t1))
                    / ((1.0 - v) * v.powi(2))
                - (256.0 * ca.powi(3) * lmss * (1.0 - v + v.powi(2)) * (t1))
                    / ((1.0 - v) * v.powi(2))
                + (128.0 * ca.powi(3) * nf * (1.0 - v + v.powi(2)) * (t1))
                    / (9.0 * (1.0 - v) * v.powi(2))
                + (256.0 * ca.powi(3) * lv * (t1) * (5.0 - 5.0 * v + 4.0 * v.powi(2)))
                    / ((1.0 - v) * v.powi(2))
        }
        // 16: gg -> qqbar, hadron from quark
        16 => {
            let t1 = 1.0 - 2.0 * v + 2.0 * v.powi(2);
            (16.0 * ca.powi(3) * cf * l1v * (1.0 - v) * (t1)) / v.powi(2)
                + (12.0 * ca * cf.powi(2) * (t1) * (cf - ca * v + ca * v.powi(2)))
                    / ((1.0 - v) * v.powi(2))
                + (16.0 * ca * cf.powi(2) * lmss * (t1) * (cf - ca * v + ca * v.powi(2)))
                    / ((1.0 - v) * v.powi(2))
                - (4.0
                    * cf
                    * lv
                    * (t1)
                    * (1.0 - 10.0 * ca.powi(2) + 9.0 * ca.powi(4) + 2.0 * ca.powi(2) * v
                        - 18.0 * ca.powi(4) * v
                        - 2.0 * ca.powi(2) * v.powi(2)
                        + 14.0 * ca.powi(4) * v.powi(2)))
                    / (ca * (1.0 - v) * v.powi(2))
                + (32.0
                    * ca
                    * cf
                    * lms
                    * (ca * cf + v - 2.0 * ca.powi(2) * v - v.powi(2)
                        + 4.0 * ca.powi(2) * v.powi(2)
                        - 4.0 * ca.powi(2) * v.powi(3)
                        + 2.0 * ca.powi(2) * v.powi(4)))
                    / ((1.0 - v) * v.powi(2))
        }
        _ => unreachable!("channel index out of range: {j0}"),
    }
}

/// `AVDEL(V,S)`: the `delta(1-W)` term, selected by channel `j0`. See
/// the channel table on [`fbor`] for what each index physically represents.
#[must_use]
pub fn avdel(j0: usize, v: f64, s: f64, ctx: &MeContext) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let pi = ctx.pi;
    let l1v = (1.0 - v).ln();
    let lv = v.ln();
    let lmu = (ctx.q2mu / s).ln();
    let lms = (ctx.q2fac / s).ln();
    let lmss = (ctx.q2frag / s).ln();
    let nf = ctx.nf;

    match j0 {
        // 1: qq' -> qq' (unlike flavor, elastic), hadron from quark
        1 => {
            let t1 = 3.0 * (1.0 - v) * v;
            (2.0 * (1.0 + 2.0 * ca.powi(2)) * cf * l1v.powi(2) * (1.0 + v)) / ((1.0 - v) * v)
                - (12.0 * ca * cf.powi(2) * lms * (1.0 + v)) / ((1.0 - v) * v)
                + (8.0 * ca * cf.powi(2) * l1v * lms * (1.0 + v)) / ((1.0 - v) * v)
                - (6.0 * ca * cf.powi(2) * lmss * (1.0 + v)) / ((1.0 - v) * v)
                + (44.0 * ca.powi(2) * cf * lmu * (1.0 + v)) / (t1)
                + (4.0 * (5.0 - 4.0 * ca.powi(2)) * cf * l1v * lv * (1.0 + v)) / ((1.0 - v) * v)
                - (8.0 * ca * cf.powi(2) * lms * lv * (1.0 + v)) / ((1.0 - v) * v)
                - (8.0 * ca * cf.powi(2) * lmss * lv * (1.0 + v)) / ((1.0 - v) * v)
                - (2.0 * (16.0 - 9.0 * ca.powi(2)) * cf * lv.powi(2) * (1.0 + v)) / ((1.0 - v) * v)
                - (40.0 * ca * cf * nf * (1.0 + v)) / (9.0 * (1.0 - v) * v)
                + (8.0 * ca * cf * l1v * nf * (1.0 + v)) / (t1)
                - (8.0 * ca * cf * lmu * nf * (1.0 + v)) / (t1)
                + (cf
                    * (225.0
                        + 115.0 * ca.powi(2)
                        + 42.0 * pi.powi(2)
                        + 12.0 * ca.powi(2) * pi.powi(2))
                    * (1.0 + v))
                    / (9.0 * (1.0 - v) * v)
                - (4.0 * cf * l1v * (3.0 + 5.0 * ca.powi(2) + 15.0 * v + 2.0 * ca.powi(2) * v))
                    / (t1)
                - (cf * lv * (5.0 - ca.powi(2) - 3.0 * v + 3.0 * ca.powi(2) * v)) / ((1.0 - v) * v)
        }
        // 2: qq' -> qq' + extra gluon, hadron from gluon
        2 => 0.0,
        // 3: qqbar' -> qqbar' (unlike flavor, elastic), hadron from quark
        3 => {
            let t1 = 3.0 * (1.0 - v) * v;
            (2.0 * (1.0 + 2.0 * ca.powi(2)) * cf * l1v.powi(2) * (1.0 + v)) / ((1.0 - v) * v)
                - (12.0 * ca * cf.powi(2) * lms * (1.0 + v)) / ((1.0 - v) * v)
                + (8.0 * ca * cf.powi(2) * l1v * lms * (1.0 + v)) / ((1.0 - v) * v)
                - (6.0 * ca * cf.powi(2) * lmss * (1.0 + v)) / ((1.0 - v) * v)
                + (44.0 * ca.powi(2) * cf * lmu * (1.0 + v)) / (t1)
                - (4.0 * (7.0 + ca.powi(2)) * cf * l1v * lv * (1.0 + v)) / ((1.0 - v) * v)
                - (8.0 * ca * cf.powi(2) * lms * lv * (1.0 + v)) / ((1.0 - v) * v)
                - (8.0 * ca * cf.powi(2) * lmss * lv * (1.0 + v)) / ((1.0 - v) * v)
                + (4.0 * (6.0 + ca.powi(2)) * cf * lv.powi(2) * (1.0 + v)) / ((1.0 - v) * v)
                - (40.0 * ca * cf * nf * (1.0 + v)) / (9.0 * (1.0 - v) * v)
                + (8.0 * ca * cf * l1v * nf * (1.0 + v)) / (t1)
                - (8.0 * ca * cf * lmu * nf * (1.0 + v)) / (t1)
                + (5.0
                    * cf
                    * (45.0 + 23.0 * ca.powi(2) - 6.0 * pi.powi(2) + 6.0 * ca.powi(2) * pi.powi(2))
                    * (1.0 + v))
                    / (9.0 * (1.0 - v) * v)
                + (cf * lv * (11.0 - 3.0 * ca.powi(2) + 3.0 * v - 3.0 * ca.powi(2) * v))
                    / ((1.0 - v) * v)
                - (4.0 * cf * l1v * (15.0 + 2.0 * ca.powi(2) + 3.0 * v + 5.0 * ca.powi(2) * v))
                    / (t1)
        }
        // 4: qqbar' -> qqbar' + extra gluon, hadron from gluon
        4 => 0.0,
        // 5: qqbar -> q'qbar' (annihilation, unlike flavor), hadron from quark
        5 => {
            let t1 = 1.0 - 2.0 * v + 2.0 * v.powi(2);
            4.0 * (2.0 - ca.powi(2)) * cf * l1v + (12.0 * ca * cf.powi(2) * lms * (t1)) / v
                - (8.0 * ca * cf.powi(2) * l1v * lms * (t1)) / v
                + (6.0 * ca * cf.powi(2) * lmss * (t1)) / v
                - (44.0 * ca.powi(2) * cf * lmu * (t1)) / (3.0 * v)
                + (4.0 * (7.0 - ca.powi(2)) * cf * l1v * lv * (t1)) / v
                + (8.0 * ca * cf.powi(2) * lms * lv * (t1)) / v
                + (8.0 * ca * cf.powi(2) * lmss * lv * (t1)) / v
                + (40.0 * ca * cf * nf * (t1)) / (9.0 * v)
                + (8.0 * ca * cf * lmu * nf * (t1)) / (3.0 * v)
                - (cf
                    * (225.0 + 115.0 * ca.powi(2)
                        - 30.0 * pi.powi(2)
                        - 6.0 * ca.powi(2) * pi.powi(2))
                    * (t1))
                    / (9.0 * v)
                - (cf
                    * lv
                    * (11.0 - 3.0 * ca.powi(2) - 14.0 * v + 6.0 * ca.powi(2) * v + 6.0 * v.powi(2)
                        - 6.0 * ca.powi(2) * v.powi(2)))
                    / v
                + (2.0
                    * cf
                    * l1v.powi(2)
                    * (3.0 - 2.0 * ca.powi(2) - 6.0 * v + 4.0 * ca.powi(2) * v + 2.0 * v.powi(2)
                        - 2.0 * ca.powi(2) * v.powi(2)))
                    / v
                - (4.0
                    * cf
                    * lv.powi(2)
                    * (6.0 + ca.powi(2) - 12.0 * v - 2.0 * ca.powi(2) * v
                        + 14.0 * v.powi(2)
                        + 2.0 * ca.powi(2) * v.powi(2)))
                    / v
        }
        // 6: qq -> qq (identical flavor, elastic), hadron from quark
        6 => {
            let t1 = (1.0 - v) * v.powi(2);
            (-8.0 * cf * l1v * nf * (1.0 - ca * v - ca * v.powi(2))) / (3.0 * t1)
                + (24.0 * cf.powi(2) * lms * (1.0 - ca + ca * v - ca * v.powi(2))) / (t1)
                - (16.0 * cf.powi(2) * l1v * lms * (1.0 - ca + ca * v - ca * v.powi(2))) / (t1)
                + (12.0 * cf.powi(2) * lmss * (1.0 - ca + ca * v - ca * v.powi(2))) / (t1)
                - (88.0 * ca * cf * lmu * (1.0 - ca + ca * v - ca * v.powi(2))) / (3.0 * t1)
                + (16.0 * cf.powi(2) * lms * lv * (1.0 - ca + ca * v - ca * v.powi(2))) / (t1)
                + (16.0 * cf.powi(2) * lmss * lv * (1.0 - ca + ca * v - ca * v.powi(2))) / (t1)
                + (80.0 * cf * nf * (1.0 - ca + ca * v - ca * v.powi(2))) / (9.0 * t1)
                + (16.0 * cf * lmu * nf * (1.0 - ca + ca * v - ca * v.powi(2))) / (3.0 * t1)
                - (8.0 * cf * lv * nf * (1.0 - 2.0 * ca + 3.0 * ca * v - ca * v.powi(2)))
                    / (3.0 * t1)
                + (2.0
                    * cf
                    * lv
                    * (15.0 - 27.0 * ca + 13.0 * ca.powi(2) - 23.0 * ca.powi(3) - 12.0 * v
                        + 45.0 * ca * v
                        + 33.0 * ca.powi(3) * v
                        - 21.0 * ca * v.powi(2)
                        - 13.0 * ca.powi(3) * v.powi(2)))
                    / (3.0 * ca * t1)
                + (4.0
                    * cf
                    * l1v
                    * (6.0 - 6.0 * ca
                        + 2.0 * ca.powi(2)
                        + 3.0 * ca.powi(3)
                        + 6.0 * v
                        + 3.0 * ca * v
                        - 8.0 * ca.powi(3) * v
                        - 15.0 * ca * v.powi(2)
                        - 2.0 * ca.powi(3) * v.powi(2)))
                    / (3.0 * ca * t1)
                - (2.0
                    * cf
                    * l1v.powi(2)
                    * (1.0 - 2.0 * ca + 4.0 * ca.powi(2) + 2.0 * v + 2.0 * ca * v
                        - 2.0 * ca.powi(3) * v
                        - 2.0 * v.powi(2)
                        - 2.0 * ca * v.powi(2)
                        - 2.0 * ca.powi(3) * v.powi(2)))
                    / (ca * t1)
                - (4.0
                    * cf
                    * l1v
                    * lv
                    * (5.0 - 2.0 * ca - 2.0 * ca.powi(2) - 2.0 * v - 2.0 * ca * v
                        + 4.0 * ca.powi(3) * v
                        + 2.0 * v.powi(2)
                        - 6.0 * ca * v.powi(2)
                        + 4.0 * ca.powi(3) * v.powi(2)))
                    / (ca * t1)
                + (2.0
                    * cf
                    * lv.powi(2)
                    * (17.0 - 16.0 * ca - 6.0 * ca.powi(2) + 6.0 * ca.powi(3) - 2.0 * v
                        + 8.0 * ca * v
                        + 2.0 * v.powi(2)
                        - 24.0 * ca * v.powi(2)
                        + 12.0 * ca.powi(3) * v.powi(2)))
                    / (ca * t1)
                - (2.0
                    * cf
                    * (225.0 - 225.0 * ca + 115.0 * ca.powi(2) - 115.0 * ca.powi(3)
                        + 15.0 * pi.powi(2)
                        - 42.0 * ca * pi.powi(2)
                        + 30.0 * ca.powi(2) * pi.powi(2)
                        - 12.0 * ca.powi(3) * pi.powi(2)
                        + 225.0 * ca * v
                        + 115.0 * ca.powi(3) * v
                        + 18.0 * pi.powi(2) * v
                        + 42.0 * ca * pi.powi(2) * v
                        + 12.0 * ca.powi(3) * pi.powi(2) * v
                        - 225.0 * ca * v.powi(2)
                        - 115.0 * ca.powi(3) * v.powi(2)
                        - 18.0 * pi.powi(2) * v.powi(2)
                        - 42.0 * ca * pi.powi(2) * v.powi(2)
                        - 12.0 * ca.powi(3) * pi.powi(2) * v.powi(2)))
                    / (9.0 * ca * t1)
        }
        7..=10 => 0.0,
        // 11: qqbar -> qqbar (same flavor, elastic + annihilation), hadron from quark
        11 => {
            let t1 = 2.0 * ca - v - 2.0 * ca * v + ca * v.powi(2);
            (8.0 * cf * l1v * nf * (ca + ca * v - v.powi(2))) / (3.0 * (1.0 - v) * v)
                - (24.0 * cf.powi(2) * lms * (t1)) / (1.0 - v)
                + (16.0 * cf.powi(2) * l1v * lms * (t1)) / (1.0 - v)
                - (12.0 * cf.powi(2) * lmss * (t1)) / (1.0 - v)
                + (88.0 * ca * cf * lmu * (t1)) / (3.0 * (1.0 - v))
                - (16.0 * cf.powi(2) * lms * lv * (t1)) / (1.0 - v)
                - (16.0 * cf.powi(2) * lmss * lv * (t1)) / (1.0 - v)
                - (80.0 * cf * nf * (t1)) / (9.0 * (1.0 - v))
                - (16.0 * cf * lmu * nf * (t1)) / (3.0 * (1.0 - v))
                + (2.0
                    * cf
                    * lv
                    * (14.0 * ca - 6.0 * ca.powi(3) - 3.0 * v - 10.0 * ca * v
                        + 3.0 * ca.powi(2) * v
                        + 6.0 * ca.powi(3) * v
                        + 3.0 * ca * v.powi(2)
                        - 3.0 * ca.powi(3) * v.powi(2)))
                    / (ca * (1.0 - v))
                - (4.0
                    * cf
                    * l1v
                    * (15.0 * ca + 2.0 * ca.powi(3) - 6.0 * v - 3.0 * ca * v
                        + 8.0 * ca.powi(3) * v
                        - 6.0 * v.powi(2)
                        + 6.0 * ca * v.powi(2)
                        - 2.0 * ca.powi(2) * v.powi(2)
                        - 3.0 * ca.powi(3) * v.powi(2)))
                    / (3.0 * ca * (1.0 - v) * v)
                + (8.0
                    * cf
                    * lv.powi(2)
                    * (12.0 * ca + 2.0 * ca.powi(3)
                        - 3.0 * v
                        - 13.0 * ca * v
                        - 5.0 * ca.powi(2) * v
                        - 2.0 * ca.powi(3) * v
                        + 7.0 * ca * v.powi(2)
                        + ca.powi(3) * v.powi(2)))
                    / (ca * (1.0 - v))
                - (8.0
                    * cf
                    * l1v
                    * lv
                    * (ca.powi(3) + 14.0 * ca * v
                        - ca.powi(3) * v
                        - 4.0 * v.powi(2)
                        - 14.0 * ca * v.powi(2)
                        - 4.0 * ca.powi(2) * v.powi(2)
                        + 2.0 * ca.powi(3) * v.powi(2)
                        + 7.0 * ca * v.powi(3)
                        - ca.powi(3) * v.powi(3)))
                    / (ca * (1.0 - v) * v)
                + (2.0
                    * cf
                    * l1v.powi(2)
                    * (2.0 + 4.0 * ca - 2.0 * v - 8.0 * ca * v + 8.0 * ca.powi(3) * v - v.powi(2)
                        + 8.0 * ca * v.powi(2)
                        - 4.0 * ca.powi(2) * v.powi(2)
                        - 6.0 * ca.powi(3) * v.powi(2)
                        - 2.0 * ca * v.powi(3)
                        + 2.0 * ca.powi(3) * v.powi(3)))
                    / (ca * (1.0 - v) * v)
                + (2.0
                    * cf
                    * (18.0 * ca.powi(3) * pi.powi(2) + 450.0 * ca * v + 230.0 * ca.powi(3) * v
                        - 60.0 * ca * pi.powi(2) * v
                        + 6.0 * ca.powi(3) * pi.powi(2) * v
                        - 225.0 * v.powi(2)
                        - 450.0 * ca * v.powi(2)
                        - 115.0 * ca.powi(2) * v.powi(2)
                        - 230.0 * ca.powi(3) * v.powi(2)
                        + 12.0 * pi.powi(2) * v.powi(2)
                        + 60.0 * ca * pi.powi(2) * v.powi(2)
                        - 12.0 * ca.powi(2) * pi.powi(2) * v.powi(2)
                        + 12.0 * ca.powi(3) * pi.powi(2) * v.powi(2)
                        + 225.0 * ca * v.powi(3)
                        + 115.0 * ca.powi(3) * v.powi(3)
                        - 30.0 * ca * pi.powi(2) * v.powi(3)
                        - 6.0 * ca.powi(3) * pi.powi(2) * v.powi(3)))
                    / (9.0 * ca * (1.0 - v) * v)
        }
        // 12: qqbar -> gg (same-flavor annihilation), hadron from gluon
        12 => {
            let t1 = 1.0 - 2.0 * v + 2.0 * v.powi(2);
            (-2.0
                * cf
                * l1v
                * (2.0 + 2.0 * ca.powi(2) + v - 5.0 * ca.powi(2) * v)
                * (1.0 - ca.powi(2) * v))
                / (ca * (1.0 - v) * v)
                - (16.0 * ca * cf.powi(2) * l1v * lms * (t1) * (cf - ca * v + ca * v.powi(2)))
                    / ((1.0 - v) * v.powi(2))
                + (44.0 * ca.powi(2) * cf * lmss * (t1) * (cf - ca * v + ca * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(2))
                - (88.0 * ca.powi(2) * cf * lmu * (t1) * (cf - ca * v + ca * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(2))
                + (16.0 * ca * cf.powi(2) * lms * lv * (t1) * (cf - ca * v + ca * v.powi(2)))
                    / ((1.0 - v) * v.powi(2))
                + (16.0 * ca.powi(2) * cf * lmss * lv * (t1) * (cf - ca * v + ca * v.powi(2)))
                    / ((1.0 - v) * v.powi(2))
                - (8.0 * ca * cf * lmss * nf * (t1) * (cf - ca * v + ca * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(2))
                + (16.0 * ca * cf * lmu * nf * (t1) * (cf - ca * v + ca * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(2))
                - (12.0
                    * cf.powi(2)
                    * lms
                    * (t1)
                    * (-2.0 * ca * cf + 2.0 * ca.powi(2) * v - 2.0 * ca.powi(2) * v.powi(2)))
                    / ((1.0 - v) * v.powi(2))
                - (20.0
                    * cf
                    * nf
                    * (t1)
                    * (-2.0 * ca * cf + 2.0 * ca.powi(2) * v - 2.0 * ca.powi(2) * v.powi(2)))
                    / (9.0 * (1.0 - v) * v.powi(2))
                + (4.0
                    * cf
                    * lv
                    * nf
                    * (t1)
                    * (-2.0 * ca * cf + 2.0 * ca.powi(2) * v - 2.0 * ca.powi(2) * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(2))
                + (4.0
                    * cf
                    * l1v
                    * lv
                    * (t1)
                    * (1.0 - 4.0 * ca.powi(2) + 3.0 * ca.powi(4) + 2.0 * ca.powi(2) * v
                        - 6.0 * ca.powi(4) * v
                        - 2.0 * ca.powi(2) * v.powi(2)
                        + 2.0 * ca.powi(4) * v.powi(2)))
                    / (ca * (1.0 - v) * v.powi(2))
                - (2.0
                    * cf
                    * lv
                    * (9.0 - 7.0 * ca.powi(2) - 2.0 * ca.powi(4) - 12.0 * v
                        + 23.0 * ca.powi(2) * v
                        + 11.0 * ca.powi(4) * v
                        + 3.0 * v.powi(2)
                        - 8.0 * ca.powi(2) * v.powi(2)
                        - 49.0 * ca.powi(4) * v.powi(2)
                        + 3.0 * ca.powi(2) * v.powi(3)
                        + 73.0 * ca.powi(4) * v.powi(3)
                        - 44.0 * ca.powi(4) * v.powi(4)))
                    / (3.0 * ca * (1.0 - v) * v.powi(2))
                - (2.0
                    * cf
                    * l1v.powi(2)
                    * (3.0 + ca.powi(4) - 4.0 * v + 2.0 * ca.powi(2) * v - 6.0 * ca.powi(4) * v
                        + 3.0 * v.powi(2)
                        - 7.0 * ca.powi(2) * v.powi(2)
                        + 13.0 * ca.powi(4) * v.powi(2)
                        + 7.0 * ca.powi(2) * v.powi(3)
                        - 12.0 * ca.powi(4) * v.powi(3)
                        - 4.0 * ca.powi(2) * v.powi(4)
                        + 4.0 * ca.powi(4) * v.powi(4)))
                    / (ca * (1.0 - v) * v.powi(2))
                - (2.0
                    * cf
                    * lv.powi(2)
                    * (2.0 - 12.0 * ca.powi(2) + 10.0 * ca.powi(4) - 2.0 * v
                        + 27.0 * ca.powi(2) * v
                        - 40.0 * ca.powi(4) * v
                        + 3.0 * v.powi(2)
                        - 30.0 * ca.powi(2) * v.powi(2)
                        + 73.0 * ca.powi(4) * v.powi(2)
                        + 9.0 * ca.powi(2) * v.powi(3)
                        - 68.0 * ca.powi(4) * v.powi(3)
                        - 4.0 * ca.powi(2) * v.powi(4)
                        + 28.0 * ca.powi(4) * v.powi(4)))
                    / (ca * (1.0 - v) * v.powi(2))
                + (2.0
                    * cf
                    * (63.0 - 59.0 * ca.powi(2) - 4.0 * ca.powi(4) - 9.0 * pi.powi(2)
                        + 12.0 * ca.powi(2) * pi.powi(2)
                        - 3.0 * ca.powi(4) * pi.powi(2)
                        - 126.0 * v
                        + 253.0 * ca.powi(2) * v
                        + 25.0 * ca.powi(4) * v
                        + 18.0 * pi.powi(2) * v
                        - 42.0 * ca.powi(2) * pi.powi(2) * v
                        + 12.0 * ca.powi(4) * pi.powi(2) * v
                        + 126.0 * v.powi(2)
                        - 541.0 * ca.powi(2) * v.powi(2)
                        - 77.0 * ca.powi(4) * v.powi(2)
                        - 18.0 * pi.powi(2) * v.powi(2)
                        + 78.0 * ca.powi(2) * pi.powi(2) * v.powi(2)
                        - 24.0 * ca.powi(4) * pi.powi(2) * v.powi(2)
                        + 576.0 * ca.powi(2) * v.powi(3)
                        + 104.0 * ca.powi(4) * v.powi(3)
                        - 72.0 * ca.powi(2) * pi.powi(2) * v.powi(3)
                        + 24.0 * ca.powi(4) * pi.powi(2) * v.powi(3)
                        - 288.0 * ca.powi(2) * v.powi(4)
                        - 52.0 * ca.powi(4) * v.powi(4)
                        + 36.0 * ca.powi(2) * pi.powi(2) * v.powi(4)
                        - 12.0 * ca.powi(4) * pi.powi(2) * v.powi(4)))
                    / (9.0 * ca * (1.0 - v) * v.powi(2))
        }
        // 13: qg -> qg (Compton), hadron from quark
        13 => {
            let t1 = 2.0 * ca * cf + 2.0 * v - v.powi(2) + ca.powi(2) * v.powi(2);
            (-2.0 * ca * (5.0 - ca.powi(2)) * cf * l1v * (1.0 + v)) / ((1.0 - v) * v)
                - ((9.0 - 31.0 * ca.powi(2))
                    * cf
                    * lms
                    * (1.0 + v)
                    * (-2.0 * ca * cf - 2.0 * v + v.powi(2) - ca.powi(2) * v.powi(2)))
                    / (3.0 * ca * (1.0 - v) * v.powi(2))
                + (20.0
                    * cf
                    * nf
                    * (1.0 + v)
                    * (-2.0 * ca * cf - 2.0 * v + v.powi(2) - ca.powi(2) * v.powi(2)))
                    / (9.0 * (1.0 - v) * v.powi(2))
                - (4.0
                    * cf
                    * lms
                    * nf
                    * (1.0 + v)
                    * (-2.0 * ca * cf - 2.0 * v + v.powi(2) - ca.powi(2) * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(2))
                - (4.0
                    * cf
                    * lv
                    * nf
                    * (1.0 + v)
                    * (-2.0 * ca * cf - 2.0 * v + v.powi(2) - ca.powi(2) * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(2))
                + (8.0 * ca * cf * l1v * lms * (1.0 + v) * (t1)) / ((1.0 - v) * v.powi(2))
                - (6.0 * cf.powi(2) * lmss * (1.0 + v) * (t1)) / ((1.0 - v) * v.powi(2))
                + (44.0 * ca * cf * lmu * (1.0 + v) * (t1)) / (3.0 * (1.0 - v) * v.powi(2))
                - (8.0 * ca * cf * lms * lv * (1.0 + v) * (t1)) / ((1.0 - v) * v.powi(2))
                - (8.0 * cf.powi(2) * lmss * lv * (1.0 + v) * (t1)) / ((1.0 - v) * v.powi(2))
                - (8.0 * cf * lmu * nf * (1.0 + v) * (t1)) / (3.0 * (1.0 - v) * v.powi(2))
                + (2.0
                    * cf
                    * l1v.powi(2)
                    * (1.0 + v)
                    * (1.0 - 2.0 * ca.powi(2) + ca.powi(4) - 2.0 * v
                        + 5.0 * ca.powi(2) * v
                        + v.powi(2)
                        - 2.0 * ca.powi(2) * v.powi(2)
                        + ca.powi(4) * v.powi(2)))
                    / (ca * (1.0 - v) * v.powi(2))
                + (2.0
                    * cf
                    * lv
                    * (9.0
                        - 7.0 * ca.powi(2)
                        - 2.0 * ca.powi(4)
                        - 15.0 * v
                        - 2.0 * ca.powi(2) * v
                        - 5.0 * ca.powi(4) * v
                        + 6.0 * v.powi(2)
                        - 5.0 * ca.powi(2) * v.powi(2)
                        - 11.0 * ca.powi(4) * v.powi(2)
                        + 11.0 * ca.powi(2) * v.powi(3)
                        - 11.0 * ca.powi(4) * v.powi(3)))
                    / (3.0 * ca * (1.0 - v) * v.powi(2))
                + (4.0
                    * cf
                    * l1v
                    * lv
                    * (4.0 * ca.powi(2) - 4.0 * ca.powi(4) + 2.0 * v
                        - 5.0 * ca.powi(2) * v
                        - 4.0 * ca.powi(4) * v
                        - 3.0 * v.powi(2)
                        - 8.0 * ca.powi(2) * v.powi(2)
                        - ca.powi(4) * v.powi(2)
                        + v.powi(3)
                        + 4.0 * ca.powi(2) * v.powi(3)
                        - ca.powi(4) * v.powi(3)))
                    / (ca * (1.0 - v) * v.powi(2))
                + (2.0
                    * cf
                    * lv.powi(2)
                    * (2.0 - 12.0 * ca.powi(2) + 10.0 * ca.powi(4) - 4.0 * v
                        + 9.0 * ca.powi(2) * v
                        + 10.0 * ca.powi(4) * v
                        + v.powi(2)
                        + 12.0 * ca.powi(2) * v.powi(2)
                        + 3.0 * ca.powi(4) * v.powi(2)
                        + v.powi(3)
                        - 12.0 * ca.powi(2) * v.powi(3)
                        + 3.0 * ca.powi(4) * v.powi(3)))
                    / (ca * (1.0 - v) * v.powi(2))
                - (2.0
                    * cf
                    * (63.0 - 59.0 * ca.powi(2) - 4.0 * ca.powi(4) - 9.0 * pi.powi(2)
                        + 12.0 * ca.powi(2) * pi.powi(2)
                        - 3.0 * ca.powi(4) * pi.powi(2)
                        - 63.0 * v
                        - 76.0 * ca.powi(2) * v
                        - 13.0 * ca.powi(4) * v
                        + 27.0 * pi.powi(2) * v
                        - 3.0 * ca.powi(2) * pi.powi(2) * v
                        - 3.0 * ca.powi(4) * pi.powi(2) * v
                        - 63.0 * v.powi(2)
                        - 76.0 * ca.powi(2) * v.powi(2)
                        - 13.0 * ca.powi(4) * v.powi(2)
                        - 18.0 * pi.powi(2) * v.powi(2)
                        - 30.0 * ca.powi(2) * pi.powi(2) * v.powi(2)
                        - 12.0 * ca.powi(4) * pi.powi(2) * v.powi(2)
                        + 63.0 * v.powi(3)
                        - 59.0 * ca.powi(2) * v.powi(3)
                        - 4.0 * ca.powi(4) * v.powi(3)
                        + 12.0 * ca.powi(2) * pi.powi(2) * v.powi(3)
                        - 12.0 * ca.powi(4) * pi.powi(2) * v.powi(3)))
                    / (9.0 * ca * (1.0 - v) * v.powi(2))
        }
        // 14: qg -> qg (Compton), hadron from gluon
        14 => {
            let t1 = (1.0 - v) * v.powi(2);
            (-2.0
                * cf
                * l1v
                * (ca.powi(2) - v)
                * (1.0 - 5.0 * ca.powi(2) + 2.0 * v + 2.0 * ca.powi(2) * v))
                / (ca * t1)
                + ((9.0 - 31.0 * ca.powi(2))
                    * cf
                    * lms
                    * (2.0 - v)
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / (3.0 * ca * t1)
                + (8.0
                    * ca
                    * cf
                    * l1v
                    * lms
                    * (2.0 - v)
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / (t1)
                - (22.0
                    * ca
                    * cf
                    * lmss
                    * (2.0 - v)
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / (3.0 * t1)
                + (44.0
                    * ca
                    * cf
                    * lmu
                    * (2.0 - v)
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / (3.0 * t1)
                - (8.0
                    * ca
                    * cf
                    * lms
                    * lv
                    * (2.0 - v)
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / (t1)
                - (8.0
                    * ca
                    * cf
                    * lmss
                    * lv
                    * (2.0 - v)
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / (t1)
                + (4.0
                    * cf
                    * lms
                    * nf
                    * (2.0 - v)
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / (3.0 * t1)
                + (4.0
                    * cf
                    * lmss
                    * nf
                    * (2.0 - v)
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / (3.0 * t1)
                - (8.0
                    * cf
                    * lmu
                    * nf
                    * (2.0 - v)
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / (3.0 * t1)
                - (cf
                    * lv
                    * (2.0 - v)
                    * (4.0 * ca.powi(2) + 4.0 * ca.powi(4)
                        - 4.0 * ca.powi(2) * v
                        - 4.0 * ca.powi(4) * v
                        + 3.0 * v.powi(2)
                        - 6.0 * ca.powi(2) * v.powi(2)
                        + 3.0 * ca.powi(4) * v.powi(2)))
                    / (ca * t1)
                + (2.0
                    * cf
                    * lv.powi(2)
                    * (2.0 - v)
                    * (ca.powi(2) + 22.0 * ca.powi(4) - ca.powi(2) * v - 22.0 * ca.powi(4) * v
                        + v.powi(2)
                        - 4.0 * ca.powi(2) * v.powi(2)
                        + 11.0 * ca.powi(4) * v.powi(2)))
                    / (ca * t1)
                - (4.0
                    * cf
                    * l1v
                    * lv
                    * (ca.powi(2) + 14.0 * ca.powi(4)
                        - v
                        - 3.0 * ca.powi(2) * v
                        - 23.0 * ca.powi(4) * v
                        + 2.0 * v.powi(2)
                        + 2.0 * ca.powi(2) * v.powi(2)
                        + 16.0 * ca.powi(4) * v.powi(2)
                        - 4.0 * ca.powi(4) * v.powi(3)))
                    / (ca * t1)
                + (2.0
                    * cf
                    * l1v.powi(2)
                    * (ca.powi(2) + 6.0 * ca.powi(4)
                        - v
                        - 3.0 * ca.powi(2) * v
                        - 11.0 * ca.powi(4) * v
                        + 2.0 * v.powi(2)
                        - 2.0 * ca.powi(2) * v.powi(2)
                        + 8.0 * ca.powi(4) * v.powi(2)
                        + 2.0 * ca.powi(2) * v.powi(3)
                        - 2.0 * ca.powi(4) * v.powi(3)))
                    / (ca * t1)
                + (cf
                    * (96.0 * ca.powi(2) - 72.0 * ca.powi(4)
                        + 14.0 * ca.powi(2) * pi.powi(2)
                        + 28.0 * ca.powi(4) * pi.powi(2)
                        - 144.0 * ca.powi(2) * v
                        + 108.0 * ca.powi(4) * v
                        - 6.0 * pi.powi(2) * v
                        - 30.0 * ca.powi(2) * pi.powi(2) * v
                        - 54.0 * ca.powi(4) * pi.powi(2) * v
                        - 42.0 * v.powi(2)
                        + 132.0 * ca.powi(2) * v.powi(2)
                        - 78.0 * ca.powi(4) * v.powi(2)
                        + 8.0 * pi.powi(2) * v.powi(2)
                        + 40.0 * ca.powi(4) * pi.powi(2) * v.powi(2)
                        + 21.0 * v.powi(3)
                        - 42.0 * ca.powi(2) * v.powi(3)
                        + 21.0 * ca.powi(4) * v.powi(3)
                        + 2.0 * pi.powi(2) * v.powi(3)
                        + 8.0 * ca.powi(2) * pi.powi(2) * v.powi(3)
                        - 10.0 * ca.powi(4) * pi.powi(2) * v.powi(3)))
                    / (3.0 * ca * t1)
        }
        // 15: gg -> gg, hadron from gluon
        15 => {
            let t1 = (1.0 - v) * v.powi(2);
            (16.0 * ca.powi(3) * lv.powi(2) * nf * (2.0 - v) * (1.0 - v + v.powi(2))) / (3.0 * t1)
                + (16.0 * ca.powi(3) * l1v.powi(2) * nf * (1.0 + v) * (1.0 - v + v.powi(2)))
                    / (3.0 * t1)
                - (1408.0 * ca.powi(3) * lms * (1.0 - v + v.powi(2)) * (2.0 - v + v.powi(2)))
                    / (3.0 * t1)
                + (256.0 * ca.powi(3) * l1v * lms * (1.0 - v + v.powi(2)) * (2.0 - v + v.powi(2)))
                    / (t1)
                - (704.0 * ca.powi(3) * lmss * (1.0 - v + v.powi(2)) * (2.0 - v + v.powi(2)))
                    / (3.0 * t1)
                + (1408.0 * ca.powi(3) * lmu * (1.0 - v + v.powi(2)) * (2.0 - v + v.powi(2)))
                    / (3.0 * t1)
                - (256.0 * ca.powi(3) * lms * lv * (1.0 - v + v.powi(2)) * (2.0 - v + v.powi(2)))
                    / (t1)
                - (256.0 * ca.powi(3) * lmss * lv * (1.0 - v + v.powi(2)) * (2.0 - v + v.powi(2)))
                    / (t1)
                + (256.0 * ca.powi(3) * lms * nf * (1.0 - v + v.powi(2)) * (2.0 - v + v.powi(2)))
                    / (9.0 * t1)
                + (128.0 * ca.powi(3) * lmss * nf * (1.0 - v + v.powi(2)) * (2.0 - v + v.powi(2)))
                    / (9.0 * t1)
                - (256.0 * ca.powi(3) * lmu * nf * (1.0 - v + v.powi(2)) * (2.0 - v + v.powi(2)))
                    / (9.0 * t1)
                + (64.0
                    * ca.powi(3)
                    * l1v.powi(2)
                    * (1.0 - v + v.powi(2))
                    * (3.0 - 3.0 * v + 2.0 * v.powi(2)))
                    / (t1)
                - (32.0 * ca.powi(3) * l1v * lv * nf * (1.0 - 2.0 * v + 2.0 * v.powi(2)))
                    / (3.0 * t1)
                - (64.0
                    * ca.powi(3)
                    * l1v
                    * lv
                    * (2.0 - v + v.powi(2))
                    * (7.0 - 8.0 * v + 4.0 * v.powi(2)))
                    / (t1)
                - (32.0 * ca.powi(3) * l1v * nf * (1.0 + v) * (5.0 - 7.0 * v + 5.0 * v.powi(2)))
                    / (9.0 * t1)
                + (64.0 * ca.powi(3) * l1v * (1.0 + v) * (7.0 + v + 7.0 * v.powi(2))) / (3.0 * t1)
                + (64.0
                    * ca.powi(3)
                    * lv
                    * (8.0 - 12.0 * v - 15.0 * v.powi(2) + 15.0 * v.powi(3) - 11.0 * v.powi(4)))
                    / (3.0 * t1)
                + (32.0
                    * ca.powi(3)
                    * lv
                    * nf
                    * (2.0 - 3.0 * v + 3.0 * v.powi(2) - 3.0 * v.powi(3) + 4.0 * v.powi(4)))
                    / (9.0 * t1)
                + (64.0
                    * ca.powi(3)
                    * lv.powi(2)
                    * (22.0 - 33.0 * v + 37.0 * v.powi(2) - 19.0 * v.powi(3) + 8.0 * v.powi(4)))
                    / (t1)
                + (16.0
                    * ca.powi(3)
                    * nf
                    * (116.0 + 9.0 * pi.powi(2) - 174.0 * v - 18.0 * pi.powi(2) * v
                        + 214.0 * v.powi(2)
                        + 18.0 * pi.powi(2) * v.powi(2)
                        - 80.0 * v.powi(3)
                        + 40.0 * v.powi(4)))
                    / (27.0 * t1)
                - (32.0
                    * ca.powi(3)
                    * (286.0 - 30.0 * pi.powi(2) - 429.0 * v
                        + 63.0 * pi.powi(2) * v
                        + 563.0 * v.powi(2)
                        - 87.0 * pi.powi(2) * v.powi(2)
                        - 268.0 * v.powi(3)
                        + 48.0 * pi.powi(2) * v.powi(3)
                        + 134.0 * v.powi(4)
                        - 24.0 * pi.powi(2) * v.powi(4)))
                    / (9.0 * t1)
        }
        // 16: gg -> qqbar, hadron from quark
        16 => {
            let t1 = (1.0 - v) * v.powi(2);
            (-2.0
                * cf
                * l1v
                * (2.0 + 2.0 * ca.powi(2) + v - 5.0 * ca.powi(2) * v)
                * (1.0 - ca.powi(2) * v))
                / (ca * (1.0 - v) * v)
                - (16.0
                    * ca
                    * cf
                    * l1v
                    * lv
                    * (1.0 + ca - ca * v)
                    * (1.0 - ca + ca * v)
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2)))
                    / (t1)
                - (16.0
                    * ca.powi(2)
                    * cf
                    * l1v
                    * lms
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (cf - ca * v + ca * v.powi(2)))
                    / (t1)
                + (12.0
                    * ca
                    * cf.powi(2)
                    * lmss
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (cf - ca * v + ca * v.powi(2)))
                    / (t1)
                - (88.0
                    * ca.powi(2)
                    * cf
                    * lmu
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (cf - ca * v + ca * v.powi(2)))
                    / (3.0 * t1)
                + (16.0
                    * ca.powi(2)
                    * cf
                    * lms
                    * lv
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (cf - ca * v + ca * v.powi(2)))
                    / (t1)
                + (16.0
                    * ca
                    * cf.powi(2)
                    * lmss
                    * lv
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (cf - ca * v + ca * v.powi(2)))
                    / (t1)
                + (16.0
                    * ca
                    * cf
                    * lmu
                    * nf
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (cf - ca * v + ca * v.powi(2)))
                    / (3.0 * t1)
                - (44.0
                    * ca
                    * cf
                    * lms
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (-2.0 * ca * cf + 2.0 * ca.powi(2) * v - 2.0 * ca.powi(2) * v.powi(2)))
                    / (3.0 * t1)
                + (8.0
                    * cf
                    * lms
                    * nf
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (-2.0 * ca * cf + 2.0 * ca.powi(2) * v - 2.0 * ca.powi(2) * v.powi(2)))
                    / (3.0 * t1)
                - (cf
                    * lv
                    * (3.0 - 6.0 * ca.powi(2) + 3.0 * ca.powi(4) - 2.0 * v + 12.0 * ca.powi(2) * v
                        - 10.0 * ca.powi(4) * v
                        - 4.0 * v.powi(2)
                        + 10.0 * ca.powi(2) * v.powi(2)
                        + 2.0 * ca.powi(4) * v.powi(2)
                        - 22.0 * ca.powi(2) * v.powi(3)
                        + 14.0 * ca.powi(4) * v.powi(3)
                        + 12.0 * ca.powi(2) * v.powi(4)
                        - 12.0 * ca.powi(4) * v.powi(4)))
                    / (ca * t1)
                - (2.0
                    * cf
                    * l1v.powi(2)
                    * (2.0 + 2.0 * ca.powi(4) - 2.0 * v - 10.0 * ca.powi(4) * v + v.powi(2)
                        - ca.powi(2) * v.powi(2)
                        + 21.0 * ca.powi(4) * v.powi(2)
                        - ca.powi(2) * v.powi(3)
                        - 20.0 * ca.powi(4) * v.powi(3)
                        + 8.0 * ca.powi(4) * v.powi(4)))
                    / (ca * t1)
                - (2.0
                    * cf
                    * lv.powi(2)
                    * (2.0 - 12.0 * ca.powi(2) + 10.0 * ca.powi(4) - 2.0 * v
                        + 27.0 * ca.powi(2) * v
                        - 40.0 * ca.powi(4) * v
                        + 3.0 * v.powi(2)
                        - 30.0 * ca.powi(2) * v.powi(2)
                        + 73.0 * ca.powi(4) * v.powi(2)
                        + 9.0 * ca.powi(2) * v.powi(3)
                        - 68.0 * ca.powi(4) * v.powi(3)
                        - 4.0 * ca.powi(2) * v.powi(4)
                        + 28.0 * ca.powi(4) * v.powi(4)))
                    / (ca * t1)
                + (cf
                    * (21.0 - 42.0 * ca.powi(2) + 21.0 * ca.powi(4) - 4.0 * pi.powi(2)
                        + 8.0 * ca.powi(2) * pi.powi(2)
                        - 4.0 * ca.powi(4) * pi.powi(2)
                        - 42.0 * v
                        + 132.0 * ca.powi(2) * v
                        - 78.0 * ca.powi(4) * v
                        + 8.0 * pi.powi(2) * v
                        - 24.0 * ca.powi(2) * pi.powi(2) * v
                        + 16.0 * ca.powi(4) * pi.powi(2) * v
                        + 42.0 * v.powi(2)
                        - 240.0 * ca.powi(2) * v.powi(2)
                        + 138.0 * ca.powi(4) * v.powi(2)
                        - 8.0 * pi.powi(2) * v.powi(2)
                        + 40.0 * ca.powi(2) * pi.powi(2) * v.powi(2)
                        - 32.0 * ca.powi(4) * pi.powi(2) * v.powi(2)
                        + 216.0 * ca.powi(2) * v.powi(3)
                        - 120.0 * ca.powi(4) * v.powi(3)
                        - 32.0 * ca.powi(2) * pi.powi(2) * v.powi(3)
                        + 32.0 * ca.powi(4) * pi.powi(2) * v.powi(3)
                        - 108.0 * ca.powi(2) * v.powi(4)
                        + 60.0 * ca.powi(4) * v.powi(4)
                        + 16.0 * ca.powi(2) * pi.powi(2) * v.powi(4)
                        - 16.0 * ca.powi(4) * pi.powi(2) * v.powi(4)))
                    / (3.0 * ca * t1)
        }
        _ => unreachable!("channel index out of range: {j0}"),
    }
}

/// `AVLO(W,V,S)`: the `log(1-W)/(1-W)+` term. See the channel table on
/// [`fbor`] for what each index physically represents.
#[must_use]
pub fn avlo(j0: usize, _w: f64, v: f64, _s: f64, ctx: &MeContext) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;

    match j0 {
        // 1: qq' -> qq' (unlike flavor, elastic), hadron from quark
        1 => (-40.0 * ca * cf.powi(2) * (1.0 + v)) / ((-1.0 + v) * v),
        // 2: qq' -> qq' + extra gluon, hadron from gluon
        2 => 0.0,
        // 3: qqbar' -> qqbar' (unlike flavor, elastic), hadron from quark
        3 => (-40.0 * ca * cf.powi(2) * (1.0 + v)) / ((-1.0 + v) * v),
        // 4: qqbar' -> qqbar' + extra gluon, hadron from gluon
        4 => 0.0,
        // 5: qqbar -> q'qbar' (annihilation, unlike flavor), hadron from quark
        5 => (-40.0 * ca * cf.powi(2) * (1.0 - 2.0 * v + 2.0 * v.powi(2))) / v,
        // 6: qq -> qq (identical flavor, elastic), hadron from quark
        6 => {
            (-80.0 * cf.powi(2) * (-1.0 + ca - ca * v + ca * v.powi(2))) / ((-1.0 + v) * v.powi(2))
        }
        7..=10 => 0.0,
        // 11: qqbar -> qqbar (same flavor, elastic + annihilation), hadron from quark
        11 => (-80.0 * cf.powi(2) * (2.0 * ca - v - 2.0 * ca * v + ca * v.powi(2))) / (-1.0 + v),
        // 12: qqbar -> gg (same-flavor annihilation), hadron from gluon
        12 => {
            (16.0
                * (-2.0 + 3.0 * ca.powi(2))
                * cf
                * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                * (cf - ca * v + ca * v.powi(2)))
                / ((-1.0 + v) * v.powi(2))
        }
        // 13: qg -> qg (Compton), hadron from quark
        13 => {
            (-8.0
                * (-2.0 + 3.0 * ca.powi(2))
                * cf
                * (1.0 + v)
                * (2.0 * ca * cf + 2.0 * v - v.powi(2) + ca.powi(2) * v.powi(2)))
                / (ca * (-1.0 + v) * v.powi(2))
        }
        // 14: qg -> qg (Compton), hadron from gluon
        14 => {
            (4.0 * (-1.0 + 3.0 * ca)
                * (1.0 + 3.0 * ca)
                * cf
                * (-2.0 + v)
                * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2) + ca.powi(2) * v.powi(2)))
                / (ca * (-1.0 + v) * v.powi(2))
        }
        // 15: gg -> gg, hadron from gluon
        15 => {
            (-1280.0 * ca.powi(3) * (1.0 - v + v.powi(2)) * (2.0 - v + v.powi(2)))
                / ((-1.0 + v) * v.powi(2))
        }
        // 16: gg -> qqbar, hadron from quark
        16 => {
            (8.0 * (-1.0 + 3.0 * ca)
                * (1.0 + 3.0 * ca)
                * cf
                * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                * (cf - ca * v + ca * v.powi(2)))
                / ((-1.0 + v) * v.powi(2))
        }
        _ => unreachable!("channel index out of range: {j0}"),
    }
}

/// `STRUV(W,V,X3,S)`: dispatches to the per-channel regular remainder terms
/// (formerly `STRUV1..16`) by channel `j0`. See the channel table on
/// [`fbor`] for what each index physically represents.
#[must_use]
pub fn struv(j0: usize, w: f64, v: f64, x3: f64, s: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    match j0 {
        1 => qqprime_elastic_quark_frag(w, v, x3, s, ctx, pre),
        2 => qqprime_elastic_gluon_frag(w, v, x3, s, ctx, pre),
        3 => qqbarprime_elastic_quark_frag(w, v, x3, s, ctx, pre),
        4 => qqbarprime_elastic_gluon_frag(w, v, x3, s, ctx, pre),
        5 => qqbar_to_qprimeqbarprime_quark_frag(w, v, x3, s, ctx, pre),
        6 => qq_identical_elastic_quark_frag(w, v, x3, s, ctx, pre),
        7 => qq_identical_elastic_gluon_frag(w, v, x3, s, ctx, pre),
        8 => qg_seapair_unlike_quark_frag(w, v, x3, s, ctx, pre),
        9 => qg_seapair_unlike_antiquark_frag(w, v, x3, s, ctx, pre),
        10 => qg_seapair_same_flavor_frag(w, v, x3, s, ctx, pre),
        11 => qqbar_elastic_quark_frag(w, v, x3, s, ctx, pre),
        12 => qqbar_to_gg_gluon_frag(w, v, x3, s, ctx, pre),
        13 => qg_compton_quark_frag(w, v, x3, s, ctx, pre),
        14 => qg_compton_gluon_frag(w, v, x3, s, ctx, pre),
        15 => gg_to_gg_gluon_frag(w, v, x3, s, ctx, pre),
        16 => gg_to_qqbar_quark_frag(w, v, x3, s, ctx, pre),
        _ => unreachable!("channel index out of range: {j0}"),
    }
}

/// `STRUV1(W,V,X3,S)`. `x3`/`s` are unused, matching the Fortran (the
/// function's value depends only on `w`, `v`, and the precomputed powers
/// and logs in `pre`).
#[must_use]
pub fn qqprime_elastic_quark_frag(
    w: f64,
    v: f64,
    _x3: f64,
    _s: f64,
    ctx: &MeContext,
    pre: &Precalc,
) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5) = (pre.v2, pre.v3, pre.v4, pre.v5);
    let (w2, w3, w4, w5, w6) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6);
    let ca2 = pre.ca2;
    let cacf = pre.cacf;
    let cacf2 = ca * cf.powi(2);
    let (l1v, lv, l1w, lw, lvw, l1vw, lms, lmss) = (
        pre.l1v, pre.lv, pre.l1w, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.lmss,
    );

    let t1 = (1.0 - v) * v * w * (1.0 - v * w).powi(2);

    (-4.0 * cf * l1v * (2.0 - 4.0 * ca2 - 4.0 * v + ca2 * v + 2.0 * v * w + ca2 * v * w))
        / ((1.0 - v) * (1.0 - v * w))
        + (4.0
            * cf
            * lvw
            * (1.0 + v * w)
            * (2.0 * cacf - v + ca2 * v + w + 2.0 * ca2 * w + 3.0 * v * w
                - 3.0 * ca2 * v * w
                - 2.0 * v * w2
                + 2.0 * ca2 * v * w2))
            / ((1.0 - v) * v * w)
        + (4.0 * cacf2 * lmss * (4.0 - 5.0 * v + v2 + 5.0 * v * w - 3.0 * v2 * w + 2.0 * v2 * w2))
            / ((1.0 - v) * (1.0 - v + v * w))
        + (4.0
            * cf
            * l1vw
            * (4.0 - 2.0 * ca2 + 4.0 * cacf - 5.0 * v + 4.0 * ca2 * v - 8.0 * ca * cf * v
                + 2.0 * v2
                - 3.0 * ca2 * v2
                + 6.0 * ca * cf * v2
                - v3
                + ca2 * v3
                - 2.0 * ca * cf * v3
                + 9.0 * v * w
                - 6.0 * ca2 * v * w
                + 12.0 * ca * cf * v * w
                - 9.0 * v2 * w
                + 9.0 * ca2 * v2 * w
                - 16.0 * ca * cf * v2 * w
                + 4.0 * v3 * w
                - 4.0 * ca2 * v3 * w
                + 8.0 * ca * cf * v3 * w
                + 7.0 * v2 * w2
                - 6.0 * ca2 * v2 * w2
                + 10.0 * ca * cf * v2 * w2
                - 5.0 * v3 * w2
                + 5.0 * ca2 * v3 * w2
                - 10.0 * ca * cf * v3 * w2
                + 2.0 * v3 * w3
                - 2.0 * ca2 * v3 * w3
                + 4.0 * ca * cf * v3 * w3))
            / ((1.0 - v) * v * (1.0 - v + v * w))
        - (2.0
            * cf
            * lms
            * (4.0 * ca * cf + 4.0 * ca * cf * v - 2.0 * ca * cf * w + 2.0 * v * w
                - 16.0 * ca * cf * v * w
                + 2.0 * v2 * w
                + 4.0 * ca * cf * v * w2
                - 3.0 * v2 * w2
                + ca2 * v2 * w2
                + 8.0 * ca * cf * v2 * w2
                + v3 * w2
                + ca2 * v3 * w2
                + 4.0 * ca * cf * v3 * w2
                - 2.0 * ca * cf * v2 * w3
                - 3.0 * v3 * w3
                - ca2 * v3 * w3
                - 4.0 * ca * cf * v3 * w3
                - v4 * w3
                + ca2 * v4 * w3
                + 2.0 * v4 * w4
                - 2.0 * ca2 * v4 * w4))
            / (t1)
        - (cf
            * (2.0 - 2.0 * ca2 - 4.0 * v + 4.0 * ca2 * v + 2.0 * v2 - 2.0 * ca2 * v2 + 6.0 * w
                - 2.0 * ca2 * w
                - 13.0 * v * w
                + ca2 * v * w
                + 7.0 * v2 * w
                + 9.0 * ca2 * v2 * w
                - 8.0 * ca2 * v3 * w
                - 6.0 * v * w2
                + 2.0 * ca2 * v * w2
                + 39.0 * v2 * w2
                - 15.0 * ca2 * v2 * w2
                - 26.0 * v3 * w2
                + 26.0 * ca2 * v3 * w2
                - 8.0 * ca2 * v4 * w2
                - 6.0 * v2 * w3
                + 2.0 * ca2 * v2 * w3
                - 15.0 * v3 * w3
                - 13.0 * ca2 * v3 * w3
                + 13.0 * v4 * w3
                + 19.0 * ca2 * v4 * w3
                + 2.0 * v5 * w3
                - 2.0 * ca2 * v5 * w3
                + 6.0 * v3 * w4
                - 2.0 * ca2 * v3 * w4
                - 5.0 * v4 * w4
                - 11.0 * ca2 * v4 * w4
                - 2.0 * v5 * w4
                + 2.0 * ca2 * v5 * w4))
            / (t1 * (1.0 - v + v * w))
        - (2.0
            * cf
            * l1w
            * (2.0 - 2.0 * ca2 - 2.0 * v2 + 2.0 * ca2 * v2 - w + ca2 * w - 23.0 * v * w
                + 17.0 * ca2 * v * w
                + 35.0 * v2 * w
                - 25.0 * ca2 * v2 * w
                - 7.0 * v3 * w
                + 3.0 * ca2 * v3 * w
                + v * w2
                - ca2 * v * w2
                + 2.0 * v2 * w2
                - 31.0 * v3 * w2
                + 19.0 * ca2 * v3 * w2
                + 9.0 * v4 * w2
                - ca2 * v4 * w2
                + v2 * w3
                - ca2 * v2 * w3
                + 25.0 * v3 * w3
                - 15.0 * ca2 * v3 * w3
                - 4.0 * v4 * w3
                - 6.0 * ca2 * v4 * w3
                - 2.0 * v5 * w3
                + 2.0 * ca2 * v5 * w3
                - v3 * w4
                + ca2 * v3 * w4
                - 6.0 * v4 * w4
                + 8.0 * ca2 * v4 * w4
                + 6.0 * v5 * w4
                - 6.0 * ca2 * v5 * w4
                - 4.0 * v5 * w5
                + 4.0 * ca2 * v5 * w5))
            / (t1 * (1.0 - v + v * w))
        - (2.0
            * cf
            * lv
            * (2.0 - 2.0 * ca2 - 2.0 * v2 + 2.0 * ca2 * v2 - w + ca2 * w - 43.0 * v * w
                + 25.0 * ca2 * v * w
                + 55.0 * v2 * w
                - 35.0 * ca2 * v2 * w
                - 7.0 * v3 * w
                + 5.0 * ca2 * v3 * w
                + v * w2
                - ca2 * v * w2
                + 6.0 * v2 * w2
                - 2.0 * ca2 * v2 * w2
                - 55.0 * v3 * w2
                + 29.0 * ca2 * v3 * w2
                + 9.0 * v4 * w2
                - 3.0 * ca2 * v4 * w2
                + v2 * w3
                - ca2 * v2 * w3
                + 45.0 * v3 * w3
                - 23.0 * ca2 * v3 * w3
                - 6.0 * ca2 * v4 * w3
                - 2.0 * v5 * w3
                + 2.0 * ca2 * v5 * w3
                - v3 * w4
                + ca2 * v3 * w4
                - 10.0 * v4 * w4
                + 10.0 * ca2 * v4 * w4
                + 6.0 * v5 * w4
                - 6.0 * ca2 * v5 * w4
                - 4.0 * v5 * w5
                + 4.0 * ca2 * v5 * w5))
            / (t1 * (1.0 - v + v * w))
        - (2.0
            * cf
            * lw
            * (3.0 - 3.0 * ca2 + 2.0 * ca * cf - 3.0 * v2 + 3.0 * ca2 * v2 - 2.0 * ca * cf * v2
                + 14.0 * w
                - 8.0 * ca2 * w
                + 8.0 * ca * cf * w
                - 31.0 * v * w
                + 19.0 * ca2 * v * w
                - 18.0 * ca * cf * v * w
                + 19.0 * v2 * w
                - 15.0 * ca2 * v2 * w
                + 14.0 * ca * cf * v2 * w
                + 4.0 * v3 * w
                - 2.0 * ca2 * v3 * w
                + w2
                - ca2 * w2
                + 2.0 * ca * cf * w2
                + 14.0 * v * w2
                - 8.0 * ca2 * v * w2
                + 8.0 * ca * cf * v * w2
                - 14.0 * v2 * w2
                + 12.0 * ca2 * v2 * w2
                - 12.0 * ca * cf * v2 * w2
                - 8.0 * v3 * w2
                + 4.0 * ca2 * v3 * w2
                - 2.0 * v4 * w2
                - v * w3
                + ca2 * v * w3
                - 2.0 * ca * cf * v * w3
                - 19.0 * v2 * w3
                + 11.0 * ca2 * v2 * w3
                - 10.0 * ca * cf * v2 * w3
                + 35.0 * v3 * w3
                - 21.0 * ca2 * v3 * w3
                + 18.0 * ca * cf * v3 * w3
                - 7.0 * v4 * w3
                + 7.0 * ca2 * v4 * w3
                - 6.0 * ca * cf * v4 * w3
                - v5 * w3
                + ca2 * v5 * w3
                - 2.0 * ca * cf * v5 * w3
                - v2 * w4
                + ca2 * v2 * w4
                - 2.0 * ca * cf * v2 * w4
                - 14.0 * v3 * w4
                + 8.0 * ca2 * v3 * w4
                - 8.0 * ca * cf * v3 * w4
                + 2.0 * v4 * w4
                - 2.0 * ca2 * v4 * w4
                + 4.0 * v5 * w4
                - 4.0 * ca2 * v5 * w4
                + 8.0 * ca * cf * v5 * w4
                + v3 * w5
                - ca2 * v3 * w5
                + 2.0 * ca * cf * v3 * w5
                + 7.0 * v4 * w5
                - 5.0 * ca2 * v4 * w5
                + 6.0 * ca * cf * v4 * w5
                - 5.0 * v5 * w5
                + 5.0 * ca2 * v5 * w5
                - 10.0 * ca * cf * v5 * w5
                + 2.0 * v5 * w6
                - 2.0 * ca2 * v5 * w6
                + 4.0 * ca * cf * v5 * w6))
            / ((1.0 - v) * v * (1.0 - w) * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w))
}

/// `STRUV2(W,V,X3,S)`.
#[must_use]
pub fn qqprime_elastic_gluon_frag(
    w: f64,
    v: f64,
    _x3: f64,
    _s: f64,
    ctx: &MeContext,
    pre: &Precalc,
) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7);
    let (w2, w3, w4, w5, w6) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6);
    let ca2 = pre.ca2;
    let cacf2 = ca * cf.powi(2);
    let (l1v, lv, l1w, lw, lvw, l1vw, lms, lmss) = (
        pre.l1v, pre.lv, pre.l1w, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.lmss,
    );

    let t1 = (1.0 - v) * v2 * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2);

    (-4.0
        * cf
        * l1v
        * (1.0 - v - v * w)
        * (2.0 + ca2 + 2.0 * v - ca2 * v - 2.0 * v * w + ca2 * v * w))
        / ((1.0 - v) * v * w * (1.0 - v + v * w))
        - (8.0
            * cacf2
            * lmss
            * (1.0 + v2 - 2.0 * v2 * w + v2 * w2)
            * (1.0 - 2.0 * v + v2 + v * w - v2 * w + v2 * w2))
            / ((1.0 - v) * v * w * (1.0 - v + v * w).powi(2))
        - (4.0
            * cf
            * lvw
            * (1.0 - w)
            * (2.0 - 5.0 * ca2 + 2.0 * v - 3.0 * ca2 * v - 3.0 * v * w + 6.0 * ca2 * v * w
                - v2 * w
                + ca2 * v2 * w
                + 2.0 * v2 * w2
                - 2.0 * ca2 * v2 * w2))
            / ((1.0 - v) * v * w)
        + (4.0
            * cf
            * lw
            * (4.0 * ca2 + 2.0 * v - 2.0 * ca2 * v - 4.0 * v2 + 2.0 * v3
                - 2.0 * ca2 * v3
                - 5.0 * v * w
                + ca2 * v * w
                + 4.0 * ca2 * v2 * w
                - 3.0 * v3 * w
                + 3.0 * ca2 * v3 * w
                + 3.0 * v2 * w2
                - 3.0 * ca2 * v2 * w2
                + v3 * w2
                - ca2 * v3 * w2))
            / ((1.0 - v) * v * w * (1.0 - v + v * w))
        - (16.0
            * cacf2
            * l1vw
            * (1.0 - w)
            * (1.0 - v - v2 + v3 + v * w + 4.0 * v2 * w - 5.0 * v3 * w - 3.0 * v2 * w2
                + 5.0 * v3 * w2
                - v3 * w3))
            / ((1.0 - v) * v * w * (1.0 - v + v * w).powi(2))
        + (2.0
            * cf
            * lms
            * (2.0 - 2.0 * ca2 - 2.0 * v - 6.0 * ca2 * v - 2.0 * v2 - 6.0 * ca2 * v2 + 2.0 * v3
                - 2.0 * ca2 * v3
                - w
                + ca2 * w
                - 3.0 * v * w
                + 5.0 * ca2 * v * w
                + 5.0 * v2 * w
                + 19.0 * ca2 * v2 * w
                + 3.0 * v3 * w
                + 11.0 * ca2 * v3 * w
                - 4.0 * v4 * w
                + 4.0 * ca2 * v4 * w
                + 2.0 * v * w2
                - 2.0 * ca2 * v * w2
                - 4.0 * ca2 * v2 * w2
                - 2.0 * v3 * w2
                - 20.0 * ca2 * v3 * w2
                + 2.0 * v4 * w2
                - 8.0 * ca2 * v4 * w2
                + 2.0 * v5 * w2
                - 2.0 * ca2 * v5 * w2
                - v2 * w3
                + ca2 * v2 * w3
                + v3 * w3
                + ca2 * v3 * w3
                - 4.0 * v4 * w3
                + 10.0 * ca2 * v4 * w3
                - 2.0 * v5 * w3
                + 2.0 * ca2 * v5 * w3
                + 2.0 * v5 * w4
                - 2.0 * ca2 * v5 * w4))
            / ((1.0 - v) * v2 * w * (1.0 - v * w).powi(2))
        + (2.0
            * cf
            * (3.0 - 3.0 * ca2 - 10.0 * v + 6.0 * ca2 * v + 14.0 * v2
                - 18.0 * ca2 * v2
                - 10.0 * v3
                + 30.0 * ca2 * v3
                + 3.0 * v4
                - 15.0 * ca2 * v4
                - 4.0 * w
                + 4.0 * ca2 * w
                + 17.0 * v * w
                - 9.0 * ca2 * v * w
                - 30.0 * v2 * w
                + 18.0 * ca2 * v2 * w
                - 4.0 * v3 * w
                - 28.0 * ca2 * v3 * w
                + 30.0 * v4 * w
                - 10.0 * ca2 * v4 * w
                - 9.0 * v5 * w
                + 25.0 * ca2 * v5 * w
                - 14.0 * v2 * w2
                + 14.0 * ca2 * v2 * w2
                + 76.0 * v3 * w2
                - 20.0 * ca2 * v3 * w2
                - 42.0 * v4 * w2
                + 34.0 * ca2 * v4 * w2
                - 30.0 * v5 * w2
                - 30.0 * ca2 * v5 * w2
                + 12.0 * v6 * w2
                - 16.0 * ca2 * v6 * w2
                + 8.0 * v2 * w3
                - 8.0 * ca2 * v2 * w3
                - 26.0 * v3 * w3
                + 10.0 * ca2 * v3 * w3
                - 28.0 * v4 * w3
                + 4.0 * ca2 * v4 * w3
                + 58.0 * v5 * w3
                + 6.0 * ca2 * v5 * w3
                - 14.0 * v6 * w3
                + 34.0 * ca2 * v6 * w3
                - 2.0 * v7 * w3
                + 2.0 * ca2 * v7 * w3
                + 11.0 * v4 * w4
                - 11.0 * ca2 * v4 * w4
                - 18.0 * v5 * w4
                - 2.0 * ca2 * v5 * w4
                + 8.0 * v6 * w4
                - 28.0 * ca2 * v6 * w4
                + 4.0 * v7 * w4
                - 4.0 * ca2 * v7 * w4
                - 4.0 * v4 * w5
                + 4.0 * ca2 * v4 * w5
                + 9.0 * v5 * w5
                - ca2 * v5 * w5
                - 6.0 * v6 * w5
                + 10.0 * ca2 * v6 * w5
                - 2.0 * v7 * w5
                + 2.0 * ca2 * v7 * w5))
            / (t1)
        - (2.0
            * cf
            * l1w
            * (2.0 - 2.0 * ca2 - 10.0 * ca2 * v - 4.0 * v2 + 18.0 * ca2 * v2 + 4.0 * v3
                - 2.0 * ca2 * v3
                - 6.0 * v4
                + 4.0 * v5
                - 4.0 * ca2 * v5
                - w
                + ca2 * w
                + 3.0 * v * w
                - ca2 * v * w
                + 4.0 * v2 * w
                + 6.0 * ca2 * v2 * w
                - 8.0 * v3 * w
                - 40.0 * ca2 * v3 * w
                + 9.0 * v4 * w
                + 17.0 * ca2 * v4 * w
                + v5 * w
                + 9.0 * ca2 * v5 * w
                - 8.0 * v6 * w
                + 8.0 * ca2 * v6 * w
                - 6.0 * v2 * w2
                + 6.0 * ca2 * v2 * w2
                + 18.0 * ca2 * v3 * w2
                + 16.0 * v4 * w2
                + 2.0 * ca2 * v4 * w2
                - 20.0 * v5 * w2
                - 28.0 * ca2 * v5 * w2
                + 18.0 * v6 * w2
                - 22.0 * ca2 * v6 * w2
                + 4.0 * v7 * w2
                - 4.0 * ca2 * v7 * w2
                + 2.0 * v2 * w3
                - 2.0 * ca2 * v2 * w3
                - 4.0 * v3 * w3
                - 10.0 * v4 * w3
                - 4.0 * ca2 * v4 * w3
                + 6.0 * v5 * w3
                + 30.0 * ca2 * v5 * w3
                - 14.0 * v6 * w3
                + 34.0 * ca2 * v6 * w3
                - 12.0 * v7 * w3
                + 12.0 * ca2 * v7 * w3
                + 4.0 * v4 * w4
                - 4.0 * ca2 * v4 * w4
                + 4.0 * v5 * w4
                - 12.0 * ca2 * v5 * w4
                + 10.0 * v6 * w4
                - 30.0 * ca2 * v6 * w4
                + 16.0 * v7 * w4
                - 16.0 * ca2 * v7 * w4
                - v4 * w5
                + ca2 * v4 * w5
                + v5 * w5
                + ca2 * v5 * w5
                - 6.0 * v6 * w5
                + 10.0 * ca2 * v6 * w5
                - 12.0 * v7 * w5
                + 12.0 * ca2 * v7 * w5
                + 4.0 * v7 * w6
                - 4.0 * ca2 * v7 * w6))
            / (t1)
        - (2.0
            * cf
            * lv
            * (2.0 - 2.0 * ca2 - 4.0 * v - 12.0 * ca2 * v + 24.0 * ca2 * v2 + 8.0 * v3
                - 8.0 * ca2 * v3
                - 10.0 * v4
                + 2.0 * ca2 * v4
                + 4.0 * v5
                - 4.0 * ca2 * v5
                - w
                + ca2 * w
                + 3.0 * v * w
                - ca2 * v * w
                + 16.0 * v2 * w
                + 8.0 * ca2 * v2 * w
                - 24.0 * v3 * w
                - 48.0 * ca2 * v3 * w
                + 5.0 * v4 * w
                + 27.0 * ca2 * v4 * w
                + 9.0 * v5 * w
                + 5.0 * ca2 * v5 * w
                - 8.0 * v6 * w
                + 8.0 * ca2 * v6 * w
                - 6.0 * v2 * w2
                + 6.0 * ca2 * v2 * w2
                - 8.0 * v3 * w2
                + 22.0 * ca2 * v3 * w2
                + 40.0 * v4 * w2
                - 2.0 * ca2 * v4 * w2
                - 24.0 * v5 * w2
                - 30.0 * ca2 * v5 * w2
                + 14.0 * v6 * w2
                - 20.0 * ca2 * v6 * w2
                + 4.0 * v7 * w2
                - 4.0 * ca2 * v7 * w2
                + 2.0 * v2 * w3
                - 2.0 * ca2 * v2 * w3
                - 4.0 * v3 * w3
                - 18.0 * v4 * w3
                - 8.0 * ca2 * v4 * w3
                - 10.0 * v5 * w3
                + 38.0 * ca2 * v5 * w3
                - 10.0 * v6 * w3
                + 32.0 * ca2 * v6 * w3
                - 12.0 * v7 * w3
                + 12.0 * ca2 * v7 * w3
                + 4.0 * v4 * w4
                - 4.0 * ca2 * v4 * w4
                + 16.0 * v5 * w4
                - 14.0 * ca2 * v5 * w4
                + 14.0 * v6 * w4
                - 32.0 * ca2 * v6 * w4
                + 16.0 * v7 * w4
                - 16.0 * ca2 * v7 * w4
                - v4 * w5
                + ca2 * v4 * w5
                + v5 * w5
                + ca2 * v5 * w5
                - 10.0 * v6 * w5
                + 12.0 * ca2 * v6 * w5
                - 12.0 * v7 * w5
                + 12.0 * ca2 * v7 * w5
                + 4.0 * v7 * w6
                - 4.0 * ca2 * v7 * w6))
            / (t1)
}

/// `STRUV3(W,V,X3,S)`.
#[must_use]
pub fn qqbarprime_elastic_quark_frag(
    w: f64,
    v: f64,
    _x3: f64,
    _s: f64,
    ctx: &MeContext,
    pre: &Precalc,
) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5) = (pre.v2, pre.v3, pre.v4, pre.v5);
    let (w2, w3, w4, w5, w6) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6);
    let ca2 = pre.ca2;
    let cacf2 = ca * cf.powi(2);
    let (l1v, lv, l1w, lw, lvw, l1vw, lms, lmss) = (
        pre.l1v, pre.lv, pre.l1w, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.lmss,
    );

    let t1 = (1.0 - v) * v * w * (1.0 - v * w).powi(2);

    (4.0 * cf * l1v * (6.0 + 2.0 * ca2 - 4.0 * v + ca2 * v - 2.0 * v * w - ca2 * v * w))
        / ((1.0 - v) * (1.0 - v * w))
        + (4.0
            * cf
            * lvw
            * (1.0 + v * w)
            * (2.0 * ca * cf - v + ca2 * v + w + 2.0 * ca2 * w + 3.0 * v * w
                - 3.0 * ca2 * v * w
                - 2.0 * v * w2
                + 2.0 * ca2 * v * w2))
            / ((1.0 - v) * v * w)
        + (4.0 * cacf2 * lmss * (4.0 - 5.0 * v + v2 + 5.0 * v * w - 3.0 * v2 * w + 2.0 * v2 * w2))
            / ((1.0 - v) * (1.0 - v + v * w))
        - (4.0
            * cf
            * l1vw
            * (1.0 - v + 2.0 * v * w)
            * (ca2 - 4.0 * ca * cf + v - 2.0 * ca2 * v + 4.0 * ca * cf * v - v2 + ca2 * v2
                - 2.0 * ca * cf * v2
                - v * w
                + 2.0 * ca2 * v * w
                - 4.0 * ca * cf * v * w
                + 2.0 * v2 * w
                - 2.0 * ca2 * v2 * w
                + 4.0 * ca * cf * v2 * w
                - v2 * w2
                + ca2 * v2 * w2
                - 2.0 * ca * cf * v2 * w2))
            / ((1.0 - v) * v * (1.0 - v + v * w))
        - (2.0
            * cf
            * lms
            * (4.0 * ca * cf + 4.0 * ca * cf * v - 2.0 * ca * cf * w + 2.0 * v * w
                - 16.0 * ca * cf * v * w
                + 2.0 * v2 * w
                + 4.0 * ca * cf * v * w2
                - 3.0 * v2 * w2
                + ca2 * v2 * w2
                + 8.0 * ca * cf * v2 * w2
                + v3 * w2
                + ca2 * v3 * w2
                + 4.0 * ca * cf * v3 * w2
                - 2.0 * ca * cf * v2 * w3
                - 3.0 * v3 * w3
                - ca2 * v3 * w3
                - 4.0 * ca * cf * v3 * w3
                - v4 * w3
                + ca2 * v4 * w3
                + 2.0 * v4 * w4
                - 2.0 * ca2 * v4 * w4))
            / (t1)
        - (cf
            * (2.0 - 2.0 * ca2 - 4.0 * v + 4.0 * ca2 * v + 2.0 * v2 - 2.0 * ca2 * v2 - 10.0 * w
                + 2.0 * ca2 * w
                + 35.0 * v * w
                - 11.0 * ca2 * v * w
                - 25.0 * v2 * w
                + 17.0 * ca2 * v2 * w
                - 8.0 * ca2 * v3 * w
                + 10.0 * v * w2
                - 2.0 * ca2 * v * w2
                - 41.0 * v2 * w2
                + 5.0 * ca2 * v2 * w2
                + 38.0 * v3 * w2
                + 10.0 * ca2 * v3 * w2
                - 8.0 * ca2 * v4 * w2
                + 10.0 * v2 * w3
                - 2.0 * ca2 * v2 * w3
                + v3 * w3
                - 17.0 * ca2 * v3 * w3
                - 19.0 * v4 * w3
                + 27.0 * ca2 * v4 * w3
                + 2.0 * v5 * w3
                - 2.0 * ca2 * v5 * w3
                - 10.0 * v3 * w4
                + 2.0 * ca2 * v3 * w4
                + 11.0 * v4 * w4
                - 15.0 * ca2 * v4 * w4
                - 2.0 * v5 * w4
                + 2.0 * ca2 * v5 * w4))
            / (t1 * (1.0 - v + v * w))
        - (2.0
            * cf
            * lv
            * (2.0 - 2.0 * ca2 - 2.0 * v2 + 2.0 * ca2 * v2 - w
                + ca2 * w
                + 21.0 * v * w
                + 9.0 * ca2 * v * w
                - 25.0 * v2 * w
                - 15.0 * ca2 * v2 * w
                + 9.0 * v3 * w
                + ca2 * v3 * w
                + v * w2
                - ca2 * v * w2
                - 10.0 * v2 * w2
                + 2.0 * ca2 * v2 * w2
                + 25.0 * v3 * w2
                + 9.0 * ca2 * v3 * w2
                - 7.0 * v4 * w2
                + ca2 * v4 * w2
                + v2 * w3
                - ca2 * v2 * w3
                - 19.0 * v3 * w3
                - 7.0 * ca2 * v3 * w3
                - 6.0 * ca2 * v4 * w3
                - 2.0 * v5 * w3
                + 2.0 * ca2 * v5 * w3
                - v3 * w4
                + ca2 * v3 * w4
                + 6.0 * v4 * w4
                + 6.0 * ca2 * v4 * w4
                + 6.0 * v5 * w4
                - 6.0 * ca2 * v5 * w4
                + 4.0 * ca2 * v5 * w5))
        - 4.0 * v5 * w5 / (t1 * (1.0 - v + v * w))
        - (2.0
            * cf
            * l1w
            * (2.0 - 2.0 * ca2 - 2.0 * v2 + 2.0 * ca2 * v2 - w + ca2 * w - 7.0 * v * w
                + 13.0 * ca2 * v * w
                + 3.0 * v2 * w
                - 17.0 * ca2 * v2 * w
                + 9.0 * v3 * w
                - ca2 * v3 * w
                + v * w2
                - ca2 * v * w2
                + 2.0 * v2 * w2
                - 15.0 * v3 * w2
                + 15.0 * ca2 * v3 * w2
                - 7.0 * v4 * w2
                + 3.0 * ca2 * v4 * w2
                + v2 * w3
                - ca2 * v2 * w3
                + 9.0 * v3 * w3
                - 11.0 * ca2 * v3 * w3
                + 12.0 * v4 * w3
                - 10.0 * ca2 * v4 * w3
                - 2.0 * v5 * w3
                + 2.0 * ca2 * v5 * w3
                - v3 * w4
                + ca2 * v3 * w4
                - 6.0 * v4 * w4
                + 8.0 * ca2 * v4 * w4
                + 6.0 * v5 * w4
                - 6.0 * ca2 * v5 * w4
                - 4.0 * v5 * w5
                + 4.0 * ca2 * v5 * w5))
            / (t1 * (1.0 - v + v * w))
        - (2.0
            * cf
            * lw
            * (3.0 - 3.0 * ca2 + 2.0 * ca * cf - 3.0 * v2 + 3.0 * ca2 * v2
                - 2.0 * ca * cf * v2
                - 10.0 * w
                - 2.0 * ca2 * w
                + 8.0 * ca * cf * w
                + 9.0 * v * w
                + 9.0 * ca2 * v * w
                - 18.0 * ca * cf * v * w
                + 3.0 * v2 * w
                - 11.0 * ca2 * v2 * w
                + 14.0 * ca * cf * v2 * w
                + 4.0 * v3 * w
                - 2.0 * ca2 * v3 * w
                + w2
                - ca2 * w2
                + 2.0 * ca * cf * w2
                - 2.0 * v * w2
                - 4.0 * ca2 * v * w2
                + 8.0 * ca * cf * v * w2
                - 6.0 * v2 * w2
                + 10.0 * ca2 * v2 * w2
                - 12.0 * ca * cf * v2 * w2
                - 8.0 * v3 * w2
                + 4.0 * ca2 * v3 * w2
                - 2.0 * v4 * w2
                - v * w3
                + ca2 * v * w3
                - 2.0 * ca * cf * v * w3
                + 13.0 * v2 * w3
                + 3.0 * ca2 * v2 * w3
                - 10.0 * ca * cf * v2 * w3
                - 5.0 * v3 * w3
                - 11.0 * ca2 * v3 * w3
                + 18.0 * ca * cf * v3 * w3
                + 9.0 * v4 * w3
                + 3.0 * ca2 * v4 * w3
                - 6.0 * ca * cf * v4 * w3
                - v5 * w3
                + ca2 * v5 * w3
                - 2.0 * ca * cf * v5 * w3
                - v2 * w4
                + ca2 * v2 * w4
                - 2.0 * ca * cf * v2 * w4
                + 2.0 * v3 * w4
                + 4.0 * ca2 * v3 * w4
                - 8.0 * ca * cf * v3 * w4
                - 6.0 * v4 * w4
                + 4.0 * v5 * w4
                - 4.0 * ca2 * v5 * w4
                + 8.0 * ca * cf * v5 * w4
                + v3 * w5
                - ca2 * v3 * w5
                + 2.0 * ca * cf * v3 * w5
                - v4 * w5
                - 3.0 * ca2 * v4 * w5
                + 6.0 * ca * cf * v4 * w5
                - 5.0 * v5 * w5
                + 5.0 * ca2 * v5 * w5
                - 10.0 * ca * cf * v5 * w5
                + 2.0 * v5 * w6
                - 2.0 * ca2 * v5 * w6
                + 4.0 * ca * cf * v5 * w6))
            / ((1.0 - v) * v * (1.0 - w) * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w))
}

/// `STRUV4(W,V,X3,S)`.
#[must_use]
pub fn qqbarprime_elastic_gluon_frag(
    w: f64,
    v: f64,
    _x3: f64,
    _s: f64,
    ctx: &MeContext,
    pre: &Precalc,
) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7);
    let (w2, w3, w4, w5, w6) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6);
    let ca2 = pre.ca2;
    let cacf2 = ca * cf.powi(2);
    let (l1v, lv, l1w, lw, lvw, l1vw, lms, lmss) = (
        pre.l1v, pre.lv, pre.l1w, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.lmss,
    );

    let t1 = (1.0 - v) * v2 * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2);

    (-8.0 * cf * l1v * (1.0 - v - v * w) * (2.0 * ca * cf - v + v * w))
        / ((1.0 - v) * v * w * (1.0 - v + v * w))
        - (8.0
            * cacf2
            * lmss
            * (1.0 + v2 - 2.0 * v2 * w + v2 * w2)
            * (1.0 - 2.0 * v + v2 + v * w - v2 * w + v2 * w2))
            / ((1.0 - v) * v * w * (1.0 - v + v * w).powi(2))
        + (4.0
            * cf
            * lvw
            * (1.0 - w)
            * (2.0 + 4.0 * ca2 + 2.0 * v + 2.0 * ca2 * v - v * w - 5.0 * ca2 * v * w + v2 * w
                - ca2 * v2 * w
                - 2.0 * v2 * w2
                + 2.0 * ca2 * v2 * w2))
            / ((1.0 - v) * v * w)
        + (4.0
            * cf
            * lw
            * (4.0 * ca2 + 2.0 * v - 2.0 * ca2 * v - 4.0 * v2 + 2.0 * v3 - 2.0 * ca2 * v3
                + 3.0 * v * w
                - ca2 * v * w
                + 8.0 * v2 * w
                + 2.0 * ca2 * v2 * w
                - 3.0 * v3 * w
                + 3.0 * ca2 * v3 * w
                - 5.0 * v2 * w2
                - ca2 * v2 * w2
                + v3 * w2
                - ca2 * v3 * w2))
            / ((1.0 - v) * v * w * (1.0 - v + v * w))
        - (8.0
            * cf
            * l1vw
            * (1.0 - w)
            * (3.0 - 3.0 * v - 3.0 * v2
                + 3.0 * v3
                + 3.0 * v * w
                + 4.0 * v2 * w
                + 2.0 * ca2 * v2 * w
                - 7.0 * v3 * w
                - 2.0 * ca2 * v3 * w
                - v2 * w2
                - 2.0 * ca2 * v2 * w2
                + 7.0 * v3 * w2
                + 2.0 * ca2 * v3 * w2
                - 3.0 * v3 * w3))
            / ((1.0 - v) * v * w * (1.0 - v + v * w).powi(2))
        + (2.0
            * cf
            * lms
            * (2.0 - 2.0 * ca2 - 2.0 * v - 6.0 * ca2 * v - 2.0 * v2 - 6.0 * ca2 * v2 + 2.0 * v3
                - 2.0 * ca2 * v3
                - w
                + ca2 * w
                - 3.0 * v * w
                + 5.0 * ca2 * v * w
                + 5.0 * v2 * w
                + 19.0 * ca2 * v2 * w
                + 3.0 * v3 * w
                + 11.0 * ca2 * v3 * w
                - 4.0 * v4 * w
                + 4.0 * ca2 * v4 * w
                + 2.0 * v * w2
                - 2.0 * ca2 * v * w2
                - 4.0 * ca2 * v2 * w2
                - 2.0 * v3 * w2
                - 20.0 * ca2 * v3 * w2
                + 2.0 * v4 * w2
                - 8.0 * ca2 * v4 * w2
                + 2.0 * v5 * w2
                - 2.0 * ca2 * v5 * w2
                - v2 * w3
                + ca2 * v2 * w3
                + v3 * w3
                + ca2 * v3 * w3
                - 4.0 * v4 * w3
                + 10.0 * ca2 * v4 * w3
                - 2.0 * v5 * w3
                + 2.0 * ca2 * v5 * w3
                + 2.0 * v5 * w4
                - 2.0 * ca2 * v5 * w4))
            / ((1.0 - v) * v2 * w * (1.0 - v * w).powi(2))
        + (2.0
            * cf
            * (3.0 - 3.0 * ca2 - 10.0 * v + 6.0 * ca2 * v + 14.0 * v2
                - 18.0 * ca2 * v2
                - 10.0 * v3
                + 30.0 * ca2 * v3
                + 3.0 * v4
                - 15.0 * ca2 * v4
                - 4.0 * w
                + 4.0 * ca2 * w
                + 9.0 * v * w
                - 7.0 * ca2 * v * w
                + 2.0 * v2 * w
                + 10.0 * ca2 * v2 * w
                - 4.0 * v3 * w
                - 28.0 * ca2 * v3 * w
                - 2.0 * v4 * w
                - 2.0 * ca2 * v4 * w
                - v5 * w
                + 23.0 * ca2 * v5 * w
                - 14.0 * v2 * w2
                + 14.0 * ca2 * v2 * w2
                + 4.0 * v3 * w2
                - 2.0 * ca2 * v3 * w2
                + 14.0 * v4 * w2
                + 20.0 * ca2 * v4 * w2
                - 6.0 * v5 * w2
                - 36.0 * ca2 * v5 * w2
                + 4.0 * v6 * w2
                - 14.0 * ca2 * v6 * w2
                + 8.0 * v2 * w3
                - 8.0 * ca2 * v2 * w3
                - 10.0 * v3 * w3
                + 6.0 * ca2 * v3 * w3
                - 4.0 * v4 * w3
                - 2.0 * ca2 * v4 * w3
                + 10.0 * v5 * w3
                + 18.0 * ca2 * v5 * w3
                - 6.0 * v6 * w3
                + 32.0 * ca2 * v6 * w3
                - 2.0 * v7 * w3
                + 2.0 * ca2 * v7 * w3
                + 11.0 * v4 * w4
                - 11.0 * ca2 * v4 * w4
                - 10.0 * v5 * w4
                - 4.0 * ca2 * v5 * w4
                - 26.0 * ca2 * v6 * w4
                + 4.0 * v7 * w4
                - 4.0 * ca2 * v7 * w4
                - 4.0 * v4 * w5
                + 4.0 * ca2 * v4 * w5
                + v5 * w5
                + ca2 * v5 * w5
                + 2.0 * v6 * w5
                + 8.0 * ca2 * v6 * w5
                - 2.0 * v7 * w5
                + 2.0 * ca2 * v7 * w5))
            / (t1)
        - (2.0
            * cf
            * l1w
            * (2.0 - 2.0 * ca2 - 8.0 * v - 8.0 * ca2 * v + 4.0 * v2 + 16.0 * ca2 * v2 + 12.0 * v3
                - 4.0 * ca2 * v3
                - 14.0 * v4
                + 2.0 * ca2 * v4
                + 4.0 * v5
                - 4.0 * ca2 * v5
                - w
                + ca2 * w
                + 3.0 * v * w
                - ca2 * v * w
                - 4.0 * v2 * w
                + 8.0 * ca2 * v2 * w
                - 8.0 * v3 * w
                - 40.0 * ca2 * v3 * w
                + v4 * w
                + 19.0 * ca2 * v4 * w
                + 17.0 * v5 * w
                + 5.0 * ca2 * v5 * w
                - 8.0 * v6 * w
                + 8.0 * ca2 * v6 * w
                - 6.0 * v2 * w2
                + 6.0 * ca2 * v2 * w2
                + 32.0 * v3 * w2
                + 10.0 * ca2 * v3 * w2
                - 16.0 * v4 * w2
                + 10.0 * ca2 * v4 * w2
                - 28.0 * v5 * w2
                - 26.0 * ca2 * v5 * w2
                + 10.0 * v6 * w2
                - 20.0 * ca2 * v6 * w2
                + 4.0 * v7 * w2
                - 4.0 * ca2 * v7 * w2
                + 2.0 * v2 * w3
                - 2.0 * ca2 * v2 * w3
                - 4.0 * v3 * w3
                - 10.0 * v4 * w3
                - 4.0 * ca2 * v4 * w3
                + 38.0 * v5 * w3
                + 22.0 * ca2 * v5 * w3
                - 6.0 * v6 * w3
                + 32.0 * ca2 * v6 * w3
                - 12.0 * v7 * w3
                + 12.0 * ca2 * v7 * w3
                + 4.0 * v4 * w4
                - 4.0 * ca2 * v4 * w4
                - 20.0 * v5 * w4
                - 6.0 * ca2 * v5 * w4
                + 2.0 * v6 * w4
                - 28.0 * ca2 * v6 * w4
                + 16.0 * v7 * w4
                - 16.0 * ca2 * v7 * w4
                - v4 * w5
                + ca2 * v4 * w5
                + v5 * w5
                + ca2 * v5 * w5
                + 2.0 * v6 * w5
                + 8.0 * ca2 * v6 * w5
                - 12.0 * v7 * w5
                + 12.0 * ca2 * v7 * w5
                + 4.0 * v7 * w6
                - 4.0 * ca2 * v7 * w6))
            / (t1)
        - (2.0
            * cf
            * lv
            * (2.0 - 2.0 * ca2 - 4.0 * v - 12.0 * ca2 * v + 24.0 * ca2 * v2 + 8.0 * v3
                - 8.0 * ca2 * v3
                - 10.0 * v4
                + 2.0 * ca2 * v4
                + 4.0 * v5
                - 4.0 * ca2 * v5
                - w
                + ca2 * w
                + 3.0 * v * w
                - ca2 * v * w
                - 16.0 * v2 * w
                + 16.0 * ca2 * v2 * w
                + 8.0 * v3 * w
                - 56.0 * ca2 * v3 * w
                + 5.0 * v4 * w
                + 27.0 * ca2 * v4 * w
                + 9.0 * v5 * w
                + 5.0 * ca2 * v5 * w
                - 8.0 * v6 * w
                + 8.0 * ca2 * v6 * w
                - 6.0 * v2 * w2
                + 6.0 * ca2 * v2 * w2
                + 40.0 * v3 * w2
                + 10.0 * ca2 * v3 * w2
                - 40.0 * v4 * w2
                + 18.0 * ca2 * v4 * w2
                - 24.0 * v5 * w2
                - 30.0 * ca2 * v5 * w2
                + 14.0 * v6 * w2
                - 20.0 * ca2 * v6 * w2
                + 4.0 * v7 * w2
                - 4.0 * ca2 * v7 * w2
                + 2.0 * v2 * w3
                - 2.0 * ca2 * v2 * w3
                - 4.0 * v3 * w3
                - 2.0 * v4 * w3
                - 12.0 * ca2 * v4 * w3
                + 54.0 * v5 * w3
                + 22.0 * ca2 * v5 * w3
                - 10.0 * v6 * w3
                + 32.0 * ca2 * v6 * w3
                - 12.0 * v7 * w3
                + 12.0 * ca2 * v7 * w3
                + 4.0 * v4 * w4
                - 4.0 * ca2 * v4 * w4
                - 32.0 * v5 * w4
                - 2.0 * ca2 * v5 * w4
                - 2.0 * v6 * w4
                - 28.0 * ca2 * v6 * w4
                + 16.0 * v7 * w4
                - 16.0 * ca2 * v7 * w4
                - v4 * w5
                + ca2 * v4 * w5
                + v5 * w5
                + ca2 * v5 * w5
                + 6.0 * v6 * w5
                + 8.0 * ca2 * v6 * w5
                - 12.0 * v7 * w5
                + 12.0 * ca2 * v7 * w5
                + 4.0 * v7 * w6
                - 4.0 * ca2 * v7 * w6))
            / (t1)
}

/// `STRUV5(W,V,X3,S)`.
#[must_use]
pub fn qqbar_to_qprimeqbarprime_quark_frag(
    w: f64,
    v: f64,
    _x3: f64,
    _s: f64,
    ctx: &MeContext,
    pre: &Precalc,
) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7);
    let (w2, w3, w4, w5, w6, w7, w8) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8);
    let ca2 = pre.ca2;
    let cacf2 = ca * cf.powi(2);
    let (l1v, lv, l1w, lw, lvw, l1vw, lms, lmss) = (
        pre.l1v, pre.lv, pre.l1w, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.lmss,
    );

    let t1 = (1.0 - v) * v * w * (1.0 - v * w) * (1.0 - v + v * w).powi(3);

    let part1 = (4.0
        * cf
        * l1v
        * (10.0 - 2.0 * ca2 - 20.0 * v + 5.0 * ca2 * v + 14.0 * v2 - 4.0 * ca2 * v2 - 8.0 * v * w
            + ca2 * v * w
            + 2.0 * v2 * w2))
        / (1.0 - v + v * w)
        + (4.0
            * cf
            * lvw
            * (4.0 - 3.0 * ca2 - 5.0 * v + 4.0 * ca2 * v + 2.0 * v2
                - 2.0 * ca2 * v2
                - 7.0 * v * w
                + 5.0 * ca2 * v * w
                + 6.0 * v2 * w
                - 4.0 * ca2 * v2 * w
                - 2.0 * v3 * w
                + 2.0 * ca2 * v3 * w
                + 4.0 * v2 * w2
                - 4.0 * ca2 * v2 * w2
                - 2.0 * v3 * w3
                + 2.0 * ca2 * v3 * w3))
            / ((1.0 - v) * v)
        + (4.0
            * cacf2
            * lms
            * (1.0 - 3.0 * v + 4.0 * v2 - 2.0 * v3 - w + 6.0 * v * w - 8.0 * v2 * w
                + 2.0 * v3 * w
                + 2.0 * v4 * w
                + v * w2
                - 12.0 * v2 * w2
                + 14.0 * v3 * w2
                - 6.0 * v4 * w2
                + 6.0 * v3 * w3
                - 2.0 * v4 * w3
                - 2.0 * v4 * w4))
            / ((1.0 - v) * v * w * (1.0 - v * w));

    let part2 = -(4.0
        * cf
        * l1vw
        * (4.0 - 5.0 * ca2 + 4.0 * ca * cf - 14.0 * v + 18.0 * ca2 * v - 12.0 * ca * cf * v
            + 20.0 * v2
            - 28.0 * ca2 * v2
            + 14.0 * ca * cf * v2
            - 14.0 * v3
            + 22.0 * ca2 * v3
            - 8.0 * ca * cf * v3
            + 4.0 * v4
            - 7.0 * ca2 * v4
            + 2.0 * ca * cf * v4
            + 6.0 * v * w
            - 8.0 * ca2 * v * w
            + 4.0 * ca * cf * v * w
            - 16.0 * v2 * w
            + 22.0 * ca2 * v2 * w
            - 12.0 * ca * cf * v2 * w
            + 18.0 * v3 * w
            - 24.0 * ca2 * v3 * w
            + 12.0 * ca * cf * v3 * w
            - 8.0 * v4 * w
            + 10.0 * ca2 * v4 * w
            - 4.0 * ca * cf * v4 * w
            - 4.0 * v2 * w2
            - 2.0 * ca2 * v2 * w2
            + 6.0 * ca * cf * v2 * w2
            + 6.0 * v3 * w2
            + 2.0 * ca2 * v3 * w2
            - 8.0 * ca * cf * v3 * w2
            - 4.0 * ca2 * v4 * w2
            + 4.0 * ca * cf * v4 * w2
            - 10.0 * v3 * w3
            + 4.0 * ca * cf * v3 * w3
            + 8.0 * v4 * w3
            + 6.0 * ca2 * v4 * w3
            - 4.0 * ca * cf * v4 * w3
            - 4.0 * v4 * w4
            - 5.0 * ca2 * v4 * w4
            + 2.0 * ca * cf * v4 * w4))
        / (v * (1.0 - v + v * w).powi(3))
        + (2.0
            * cf
            * lmss
            * (2.0 * ca * cf + 2.0 * v - 2.0 * ca2 * v - 8.0 * ca * cf * v - 9.0 * v2
                + 9.0 * ca2 * v2
                + 14.0 * ca * cf * v2
                + 18.0 * v3
                - 18.0 * ca2 * v3
                - 12.0 * ca * cf * v3
                - 19.0 * v4
                + 19.0 * ca2 * v4
                + 4.0 * ca * cf * v4
                + 10.0 * v5
                - 10.0 * ca2 * v5
                - 2.0 * v6
                + 2.0 * ca2 * v6
                - 2.0 * v * w
                + 12.0 * ca * cf * v * w
                + 14.0 * v2 * w
                - 4.0 * ca2 * v2 * w
                - 54.0 * ca * cf * v2 * w
                - 40.0 * v3 * w
                + 18.0 * ca2 * v3 * w
                + 94.0 * ca * cf * v3 * w
                + 54.0 * v4 * w
                - 28.0 * ca2 * v4 * w
                - 82.0 * ca * cf * v4 * w
                - 34.0 * v5 * w
                + 18.0 * ca2 * v5 * w
                + 38.0 * ca * cf * v5 * w
                + 8.0 * v6 * w
                - 4.0 * ca2 * v6 * w
                - 8.0 * ca * cf * v6 * w
                - 5.0 * v2 * w2
                + ca2 * v2 * w2
                + 12.0 * ca * cf * v2 * w2
                + 30.0 * v3 * w2
                - 10.0 * ca2 * v3 * w2
                - 46.0 * ca * cf * v3 * w2
                - 58.0 * v4 * w2
                + 22.0 * ca2 * v4 * w2
                + 74.0 * ca * cf * v4 * w2
                + 48.0 * v5 * w2
                - 20.0 * ca2 * v5 * w2
                - 54.0 * ca * cf * v5 * w2
                - 14.0 * v6 * w2
                + 6.0 * ca2 * v6 * w2
                + 16.0 * ca * cf * v6 * w2
                - 8.0 * v3 * w3
                + 2.0 * ca2 * v3 * w3
                + 4.0 * ca * cf * v3 * w3
                + 30.0 * v4 * w3
                - 12.0 * ca2 * v4 * w3
                - 14.0 * ca * cf * v4 * w3
                - 40.0 * v5 * w3
                + 20.0 * ca2 * v5 * w3
                + 14.0 * ca * cf * v5 * w3
                + 16.0 * v6 * w3
                - 8.0 * ca2 * v6 * w3
                - 8.0 * ca * cf * v6 * w3
                - 7.0 * v4 * w4
                + 3.0 * ca2 * v4 * w4
                + 2.0 * ca * cf * v4 * w4
                + 22.0 * v5 * w4
                - 10.0 * ca2 * v5 * w4
                + 2.0 * ca * cf * v5 * w4
                - 14.0 * v6 * w4
                + 6.0 * ca2 * v6 * w4
                - 6.0 * v5 * w5
                + 2.0 * ca2 * v5 * w5
                + 8.0 * v6 * w5
                - 4.0 * ca2 * v6 * w5
                - 2.0 * v6 * w6
                + 2.0 * ca2 * v6 * w6))
            / ((1.0 - v) * v * w * (1.0 - v + v * w).powi(3));

    let part3 = -(cf
        * (4.0 * ca * cf - 16.0 * ca * cf * v + 24.0 * ca * cf * v2 - 16.0 * ca * cf * v3
            + 4.0 * ca * cf * v4
            + 8.0 * w
            - 4.0 * ca * cf * w
            - 15.0 * v * w
            - 9.0 * ca2 * v * w
            + 24.0 * ca * cf * v * w
            - 8.0 * v2 * w
            + 32.0 * ca2 * v2 * w
            - 36.0 * ca * cf * v2 * w
            + 40.0 * v3 * w
            - 40.0 * ca2 * v3 * w
            + 4.0 * ca * cf * v3 * w
            - 42.0 * v4 * w
            + 18.0 * ca2 * v4 * w
            + 24.0 * ca * cf * v4 * w
            + 23.0 * v5 * w
            + ca2 * v5 * w
            - 12.0 * ca * cf * v5 * w
            - 6.0 * v6 * w
            - 2.0 * ca2 * v6 * w
            + 16.0 * v * w2
            - 8.0 * ca * cf * v * w2
            - 90.0 * v2 * w2
            + 14.0 * ca2 * v2 * w2
            + 12.0 * ca * cf * v2 * w2
            + 171.0 * v3 * w2
            - 67.0 * ca2 * v3 * w2
            + 40.0 * ca * cf * v3 * w2
            - 143.0 * v4 * w2
            + 103.0 * ca2 * v4 * w2
            - 80.0 * ca * cf * v4 * w2
            + 59.0 * v5 * w2
            - 59.0 * ca2 * v5 * w2
            + 32.0 * ca * cf * v5 * w2
            - 19.0 * v6 * w2
            + 7.0 * ca2 * v6 * w2
            + 4.0 * ca * cf * v6 * w2
            + 6.0 * v7 * w2
            + 2.0 * ca2 * v7 * w2
            - 16.0 * v3 * w3
            + 36.0 * ca2 * v3 * w3
            - 36.0 * ca * cf * v3 * w3
            + 27.0 * v4 * w3
            - 71.0 * ca2 * v4 * w3
            + 80.0 * ca * cf * v4 * w3
            - 32.0 * v5 * w3
            + 36.0 * ca2 * v5 * w3
            - 32.0 * ca * cf * v5 * w3
            + 33.0 * v6 * w3
            + 3.0 * ca2 * v6 * w3
            - 12.0 * ca * cf * v6 * w3
            - 12.0 * v7 * w3
            - 4.0 * ca2 * v7 * w3
            - 16.0 * v3 * w4
            + 8.0 * ca * cf * v3 * w4
            + 50.0 * v4 * w4
            - 6.0 * ca2 * v4 * w4
            - 32.0 * ca * cf * v4 * w4
            - 31.0 * v5 * w4
            + 7.0 * ca2 * v5 * w4
            + 16.0 * ca * cf * v5 * w4
            - 9.0 * v6 * w4
            - 11.0 * ca2 * v6 * w4
            + 12.0 * ca * cf * v6 * w4
            + 6.0 * v7 * w4
            + 10.0 * ca2 * v7 * w4
            - 8.0 * v4 * w5
            + 4.0 * ca * cf * v4 * w5
            + 7.0 * v5 * w5
            + 5.0 * ca2 * v5 * w5
            - 4.0 * ca * cf * v5 * w5
            + v6 * w5
            + 11.0 * ca2 * v6 * w5
            - 4.0 * ca * cf * v6 * w5
            - 16.0 * ca2 * v7 * w5
            - 8.0 * ca2 * v6 * w6
            + 8.0 * ca2 * v7 * w6))
        / (t1);

    let part4 = (2.0
        * cf
        * l1w
        * (2.0 - 2.0 * ca2 - 12.0 * v + 12.0 * ca2 * v + 32.0 * v2 - 32.0 * ca2 * v2 - 48.0 * v3
            + 48.0 * ca2 * v3
            + 42.0 * v4
            - 42.0 * ca2 * v4
            - 20.0 * v5
            + 20.0 * ca2 * v5
            + 4.0 * v6
            - 4.0 * ca2 * v6
            - w
            + ca2 * w
            + 15.0 * v * w
            - 21.0 * ca2 * v * w
            - 67.0 * v2 * w
            + 91.0 * ca2 * v2 * w
            + 141.0 * v3 * w
            - 177.0 * ca2 * v3 * w
            - 152.0 * v4 * w
            + 176.0 * ca2 * v4 * w
            + 76.0 * v5 * w
            - 82.0 * ca2 * v5 * w
            - 8.0 * v6 * w
            + 8.0 * ca2 * v6 * w
            - 4.0 * v7 * w
            + 4.0 * ca2 * v7 * w
            - 2.0 * v * w2
            + 2.0 * ca2 * v * w2
            + 19.0 * v2 * w2
            - 15.0 * ca2 * v2 * w2
            - 54.0 * v3 * w2
            + 32.0 * ca2 * v3 * w2
            + 48.0 * v4 * w2
            - 10.0 * ca2 * v4 * w2
            + 28.0 * v5 * w2
            - 54.0 * ca2 * v5 * w2
            - 67.0 * v6 * w2
            + 73.0 * ca2 * v6 * w2
            + 28.0 * v7 * w2
            - 28.0 * ca2 * v7 * w2
            - 7.0 * v3 * w3
            + 23.0 * ca2 * v3 * w3
            + 60.0 * v4 * w3
            - 90.0 * ca2 * v4 * w3
            - 144.0 * v5 * w3
            + 156.0 * ca2 * v5 * w3
            + 143.0 * v6 * w3
            - 141.0 * ca2 * v6 * w3
            - 52.0 * v7 * w3
            + 52.0 * ca2 * v7 * w3
            + 2.0 * v3 * w4
            - 2.0 * ca2 * v3 * w4
            - 27.0 * v4 * w4
            + 19.0 * ca2 * v4 * w4
            + 74.0 * v5 * w4
            - 52.0 * ca2 * v5 * w4
            - 83.0 * v6 * w4
            + 69.0 * ca2 * v6 * w4
            + 36.0 * v7 * w4
            - 36.0 * ca2 * v7 * w4
            + v4 * w5
            - ca2 * v4 * w5
            - 6.0 * v5 * w5
            + 11.0 * v6 * w5
            - 5.0 * ca2 * v6 * w5
            - 12.0 * v7 * w5
            + 12.0 * ca2 * v7 * w5
            + 8.0 * v7 * w6
            - 8.0 * ca2 * v7 * w6
            - 4.0 * v7 * w7
            + 4.0 * ca2 * v7 * w7))
        / (t1)
        + (2.0
            * cf
            * lv
            * (2.0 - 2.0 * ca2 - 12.0 * v + 12.0 * ca2 * v + 32.0 * v2
                - 32.0 * ca2 * v2
                - 48.0 * v3
                + 48.0 * ca2 * v3
                + 42.0 * v4
                - 42.0 * ca2 * v4
                - 20.0 * v5
                + 20.0 * ca2 * v5
                + 4.0 * v6
                - 4.0 * ca2 * v6
                - w
                + ca2 * w
                - 13.0 * v * w
                - 17.0 * ca2 * v * w
                + 73.0 * v2 * w
                + 69.0 * ca2 * v2 * w
                - 147.0 * v3 * w
                - 127.0 * ca2 * v3 * w
                + 152.0 * v4 * w
                + 118.0 * ca2 * v4 * w
                - 88.0 * v5 * w
                - 48.0 * ca2 * v5 * w
                + 28.0 * v6 * w
                - 4.0 * v7 * w
                + 4.0 * ca2 * v7 * w
                - 2.0 * v * w2
                + 2.0 * ca2 * v * w2
                + 7.0 * v2 * w2
                - 13.0 * ca2 * v2 * w2
                - 18.0 * v3 * w2
                + 24.0 * ca2 * v3 * w2
                + 32.0 * v4 * w2
                - 2.0 * ca2 * v4 * w2
                - 36.0 * v5 * w2
                - 46.0 * ca2 * v5 * w2
                + 25.0 * v6 * w2
                + 55.0 * ca2 * v6 * w2
                - 8.0 * v7 * w2
                - 20.0 * ca2 * v7 * w2
                + 41.0 * v3 * w3
                + 17.0 * ca2 * v3 * w3
                - 108.0 * v4 * w3
                - 66.0 * ca2 * v4 * w3
                + 112.0 * v5 * w3
                + 112.0 * ca2 * v5 * w3
                - 65.0 * v6 * w3
                - 99.0 * ca2 * v6 * w3
                + 20.0 * v7 * w3
                + 36.0 * ca2 * v7 * w3
                + 2.0 * v3 * w4
                - 2.0 * ca2 * v3 * w4
                - 11.0 * v4 * w4
                + 17.0 * ca2 * v4 * w4
                + 34.0 * v5 * w4
                - 44.0 * ca2 * v5 * w4
                - 27.0 * v6 * w4
                + 55.0 * ca2 * v6 * w4
                + 4.0 * v7 * w4
                - 28.0 * ca2 * v7 * w4
                + v4 * w5
                - ca2 * v4 * w5
                - 26.0 * v5 * w5
                + 2.0 * ca2 * v5 * w5
                + 39.0 * v6 * w5
                - 7.0 * ca2 * v6 * w5
                - 20.0 * v7 * w5
                + 12.0 * ca2 * v7 * w5
                - 4.0 * v6 * w6
                + 12.0 * v7 * w6
                - 8.0 * ca2 * v7 * w6
                - 4.0 * v7 * w7
                + 4.0 * ca2 * v7 * w7))
            / (t1);

    let part5 = (2.0
        * cf
        * lw
        * (3.0 - 3.0 * ca2 + 2.0 * ca * cf - 18.0 * v + 18.0 * ca2 * v - 12.0 * ca * cf * v
            + 48.0 * v2
            - 48.0 * ca2 * v2
            + 32.0 * ca * cf * v2
            - 72.0 * v3
            + 72.0 * ca2 * v3
            - 48.0 * ca * cf * v3
            + 63.0 * v4
            - 63.0 * ca2 * v4
            + 42.0 * ca * cf * v4
            - 30.0 * v5
            + 30.0 * ca2 * v5
            - 20.0 * ca * cf * v5
            + 6.0 * v6
            - 6.0 * ca2 * v6
            + 4.0 * ca * cf * v6
            - 10.0 * w
            - 2.0 * ca2 * w
            + 8.0 * ca * cf * w
            + 54.0 * v * w
            - 28.0 * ca * cf * v * w
            - 129.0 * v2 * w
            + 23.0 * ca2 * v2 * w
            + 34.0 * ca * cf * v2 * w
            + 173.0 * v3 * w
            - 55.0 * ca2 * v3 * w
            - 10.0 * ca * cf * v3 * w
            - 129.0 * v4 * w
            + 51.0 * ca2 * v4 * w
            - 10.0 * ca * cf * v4 * w
            + 39.0 * v5 * w
            - 11.0 * ca2 * v5 * w
            + 2.0 * ca * cf * v5 * w
            + 8.0 * v6 * w
            - 12.0 * ca2 * v6 * w
            + 8.0 * ca * cf * v6 * w
            - 6.0 * v7 * w
            + 6.0 * ca2 * v7 * w
            - 4.0 * ca * cf * v7 * w
            + w2
            - ca2 * w2
            + 2.0 * ca * cf * w2
            - 14.0 * v * w2
            + 8.0 * ca2 * v * w2
            - 12.0 * ca * cf * v * w2
            + 46.0 * v2 * w2
            - 30.0 * ca2 * v2 * w2
            + 40.0 * ca * cf * v2 * w2
            - 69.0 * v3 * w2
            + 49.0 * ca2 * v3 * w2
            - 66.0 * ca * cf * v3 * w2
            + 30.0 * v4 * w2
            - 22.0 * ca2 * v4 * w2
            + 48.0 * ca * cf * v4 * w2
            + 49.0 * v5 * w2
            - 35.0 * ca2 * v5 * w2
            + 2.0 * ca * cf * v5 * w2
            - 65.0 * v6 * w2
            + 49.0 * ca2 * v6 * w2
            - 26.0 * ca * cf * v6 * w2
            + 22.0 * v7 * w2
            - 18.0 * ca2 * v7 * w2
            + 12.0 * ca * cf * v7 * w2
            + 2.0 * v * w3
            - 2.0 * ca2 * v * w3
            + 4.0 * ca * cf * v * w3
            - 15.0 * v2 * w3
            + 13.0 * ca2 * v2 * w3
            - 22.0 * ca * cf * v2 * w3
            + 35.0 * v3 * w3
            - 27.0 * ca2 * v3 * w3
            + 42.0 * ca * cf * v3 * w3
            + 4.0 * v4 * w3
            + 4.0 * ca2 * v4 * w3
            - 20.0 * ca * cf * v4 * w3
            - 102.0 * v5 * w3
            + 52.0 * ca2 * v5 * w3
            - 28.0 * ca * cf * v5 * w3
            + 114.0 * v6 * w3
            - 66.0 * ca2 * v6 * w3
            + 44.0 * ca * cf * v6 * w3
            - 38.0 * v7 * w3
            + 26.0 * ca2 * v7 * w3
            - 20.0 * ca * cf * v7 * w3
            - 13.0 * v3 * w4
            - 5.0 * ca2 * v3 * w4
            + 14.0 * ca * cf * v3 * w4
            - 14.0 * v4 * w4
            + 34.0 * ca2 * v4 * w4
            - 60.0 * ca * cf * v4 * w4
            + 114.0 * v5 * w4
            - 72.0 * ca2 * v5 * w4
            + 84.0 * ca * cf * v5 * w4
            - 132.0 * v6 * w4
            + 72.0 * ca2 * v6 * w4
            - 64.0 * ca * cf * v6 * w4
            + 46.0 * v7 * w4
            - 30.0 * ca2 * v7 * w4
            + 28.0 * ca * cf * v7 * w4
            - 2.0 * v3 * w5
            + 2.0 * ca2 * v3 * w5
            - 4.0 * ca * cf * v3 * w5
            + 27.0 * v4 * w5
            - 17.0 * ca2 * v4 * w5
            + 26.0 * ca * cf * v4 * w5
            - 97.0 * v5 * w5
            + 39.0 * ca2 * v5 * w5
            - 38.0 * ca * cf * v5 * w5
            + 110.0 * v6 * w5
            - 46.0 * ca2 * v6 * w5
            + 36.0 * ca * cf * v6 * w5
            - 42.0 * v7 * w5
            + 26.0 * ca2 * v7 * w5
            - 28.0 * ca * cf * v7 * w5
            - v4 * w6
            + ca2 * v4 * w6
            - 2.0 * ca * cf * v4 * w6
            + 27.0 * v5 * w6
            - 3.0 * ca2 * v5 * w6
            - 2.0 * ca * cf * v5 * w6
            - 45.0 * v6 * w6
            + 9.0 * ca2 * v6 * w6
            - 2.0 * ca * cf * v6 * w6
            + 26.0 * v7 * w6
            - 14.0 * ca2 * v7 * w6
            + 20.0 * ca * cf * v7 * w6
            + 4.0 * v6 * w7
            - 10.0 * v7 * w7
            + 6.0 * ca2 * v7 * w7
            - 12.0 * ca * cf * v7 * w7
            + 2.0 * v7 * w8
            - 2.0 * ca2 * v7 * w8
            + 4.0 * ca * cf * v7 * w8))
        / ((1.0 - v) * v * (1.0 - w) * w * (1.0 - v * w) * (1.0 - v + v * w).powi(3));

    part1 + part2 + part3 + part4 + part5
}

/// `STRUV6(W,V,X3,S)`.
#[must_use]
pub fn qq_identical_elastic_quark_frag(
    w: f64,
    v: f64,
    _x3: f64,
    _s: f64,
    ctx: &MeContext,
    pre: &Precalc,
) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6);
    let (w2, w3, w4, w5, w6) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6);
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let ca2cf = ca2 * cf;
    let (l1v, lv, l1w, lw, lvw, l1vw, lms, lmss) = (
        pre.l1v, pre.lv, pre.l1w, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.lmss,
    );

    let t1 = ca * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w);

    let part1 = (4.0
        * cf
        * lvw
        * (1.0 - 6.0 * ca2 + 2.0 * ca3 - 2.0 * v - 3.0 * ca * v + 5.0 * ca2 * v + ca3 * v + v2
            - ca * v2
            - 3.0 * ca2 * v2
            + ca3 * v2
            + 3.0 * ca * v * w
            - 5.0 * ca2 * v * w
            - ca3 * v * w
            + 3.0 * ca * v2 * w
            + 6.0 * ca2 * v2 * w
            - 2.0 * ca3 * v2 * w
            - ca * v3 * w
            + ca3 * v3 * w
            + v2 * w2
            - 2.0 * ca * v2 * w2
            - 3.0 * ca2 * v2 * w2
            + 5.0 * ca3 * v2 * w2
            + 3.0 * ca * v3 * w2
            - 3.0 * ca3 * v3 * w2
            - 2.0 * ca * v3 * w3
            + 2.0 * ca3 * v3 * w3))
        / (ca * (1.0 - v) * v2 * w)
        + (8.0
            * cf.powi(2)
            * lmss
            * (2.0 - 2.0 * ca - 6.0 * v + 6.0 * ca * v + 7.0 * v2 - 7.0 * ca * v2 - 4.0 * v3
                + 4.0 * ca * v3
                + v4
                - ca * v4
                + 2.0 * v * w
                - 2.0 * ca * v * w
                - 7.0 * v2 * w
                + 7.0 * ca * v2 * w
                + 8.0 * v3 * w
                - 7.0 * ca * v3 * w
                - 3.0 * v4 * w
                + 2.0 * ca * v4 * w
                - 4.0 * v3 * w2
                + 3.0 * ca * v3 * w2
                + 3.0 * v4 * w2
                - 2.0 * ca * v4 * w2
                - v4 * w3
                + ca * v4 * w3))
            / ((1.0 - v) * v2 * w * (1.0 - v + v * w))
        - (4.0
            * cf
            * l1v
            * (4.0 * ca - ca2 - 2.0 * ca3 - 4.0 * v - 6.0 * ca * v - ca2 * v
                + 3.0 * ca3 * v
                + 3.0 * v2
                + 2.0 * ca * v2
                + ca2 * v2
                - ca3 * v2
                - v3
                - ca2 * v3
                + 5.0 * v * w
                - 4.0 * ca * v * w
                + 4.0 * ca2 * v * w
                + 3.0 * ca3 * v * w
                - 3.0 * v2 * w
                + 8.0 * ca * v2 * w
                - ca2 * v2 * w
                - 8.0 * ca3 * v2 * w
                + 2.0 * v3 * w
                - 6.0 * ca * v3 * w
                + 2.0 * ca2 * v3 * w
                + 2.0 * ca3 * v3 * w
                - v4 * w
                + ca2 * v4 * w
                - v2 * w2
                + ca2 * v2 * w2
                - ca3 * v2 * w2
                - v3 * w2
                + 2.0 * ca * v3 * w2
                - ca2 * v3 * w2
                + 2.0 * ca3 * v3 * w2
                + 2.0 * v4 * w2
                - 2.0 * ca2 * v4 * w2
                - v4 * w3
                + ca2 * v4 * w3))
            / (ca * (1.0 - v) * v2 * w * (1.0 - v * w));

    let part2 = -(4.0
        * cf
        * l1vw
        * (4.0 - 8.0 * ca - 4.0 * ca2 + 4.0 * ca3 + 8.0 * ca * cf - 8.0 * ca2cf - 12.0 * v
            + 22.0 * ca * v
            + 12.0 * ca2 * v
            - 13.0 * ca3 * v
            - 24.0 * ca * cf * v
            + 24.0 * ca2cf * v
            + 14.0 * v2
            - 22.0 * ca * v2
            - 14.0 * ca2 * v2
            + 16.0 * ca3 * v2
            + 28.0 * ca * cf * v2
            - 28.0 * ca2cf * v2
            - 8.0 * v3
            + 10.0 * ca * v3
            + 8.0 * ca2 * v3
            - 9.0 * ca3 * v3
            - 16.0 * ca * cf * v3
            + 16.0 * ca2cf * v3
            + 2.0 * v4
            - 2.0 * ca * v4
            - 2.0 * ca2 * v4
            + 2.0 * ca3 * v4
            + 4.0 * ca * cf * v4
            - 4.0 * ca2cf * v4
            + 12.0 * v * w
            - 14.0 * ca * v * w
            - 12.0 * ca2 * v * w
            + 9.0 * ca3 * v * w
            + 24.0 * ca * cf * v * w
            - 16.0 * ca2cf * v * w
            - 28.0 * v2 * w
            + 26.0 * ca * v2 * w
            + 28.0 * ca2 * v2 * w
            - 22.0 * ca3 * v2 * w
            - 56.0 * ca * cf * v2 * w
            + 40.0 * ca2cf * v2 * w
            + 24.0 * v3 * w
            - 18.0 * ca * v3 * w
            - 24.0 * ca2 * v3 * w
            + 19.0 * ca3 * v3 * w
            + 48.0 * ca * cf * v3 * w
            - 36.0 * ca2cf * v3 * w
            - 8.0 * v4 * w
            + 6.0 * ca * v4 * w
            + 8.0 * ca2 * v4 * w
            - 6.0 * ca3 * v4 * w
            - 16.0 * ca * cf * v4 * w
            + 12.0 * ca2cf * v4 * w
            + 14.0 * v2 * w2
            - 12.0 * ca * v2 * w2
            - 14.0 * ca2 * v2 * w2
            + 10.0 * ca3 * v2 * w2
            + 28.0 * ca * cf * v2 * w2
            - 20.0 * ca2cf * v2 * w2
            - 24.0 * v3 * w2
            + 16.0 * ca * v3 * w2
            + 24.0 * ca2 * v3 * w2
            - 17.0 * ca3 * v3 * w2
            - 48.0 * ca * cf * v3 * w2
            + 32.0 * ca2cf * v3 * w2
            + 12.0 * v4 * w2
            - 8.0 * ca * v4 * w2
            - 12.0 * ca2 * v4 * w2
            + 8.0 * ca3 * v4 * w2
            + 24.0 * ca * cf * v4 * w2
            - 16.0 * ca2cf * v4 * w2
            + 8.0 * v3 * w3
            - 8.0 * ca * v3 * w3
            - 8.0 * ca2 * v3 * w3
            + 7.0 * ca3 * v3 * w3
            + 16.0 * ca * cf * v3 * w3
            - 12.0 * ca2cf * v3 * w3
            - 8.0 * v4 * w3
            + 6.0 * ca * v4 * w3
            + 8.0 * ca2 * v4 * w3
            - 6.0 * ca3 * v4 * w3
            - 16.0 * ca * cf * v4 * w3
            + 12.0 * ca2cf * v4 * w3
            + 2.0 * v4 * w4
            - 2.0 * ca * v4 * w4
            - 2.0 * ca2 * v4 * w4
            + 2.0 * ca3 * v4 * w4
            + 4.0 * ca * cf * v4 * w4
            - 4.0 * ca2cf * v4 * w4))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v + v * w))
        - (2.0
            * cf
            * lms
            * (8.0 * ca2 - 12.0 * cf + 12.0 * ca * cf - v - 11.0 * ca2 * v + 8.0 * cf * v
                - 12.0 * ca * cf * v
                - 5.0 * v2
                + 9.0 * ca2 * v2
                - 4.0 * cf * v2
                + 8.0 * ca * cf * v2
                + 2.0 * v3
                - 2.0 * ca2 * v3
                - 4.0 * ca2 * w
                + 4.0 * cf * w
                - 4.0 * ca * cf * w
                - 10.0 * ca2 * v * w
                + 24.0 * cf * v * w
                - 20.0 * ca * cf * v * w
                + 5.0 * v2 * w
                + 19.0 * ca2 * v2 * w
                - 4.0 * cf * v2 * w
                + 8.0 * ca * cf * v2 * w
                + 8.0 * v3 * w
                - 14.0 * ca2 * v3 * w
                - 12.0 * ca * cf * v3 * w
                - 4.0 * v4 * w
                + 4.0 * ca2 * v4 * w
                + 8.0 * ca2 * v * w2
                - 8.0 * cf * v * w2
                + 8.0 * ca * cf * v * w2
                - 4.0 * ca2 * v2 * w2
                - 12.0 * cf * v2 * w2
                + 4.0 * ca * cf * v2 * w2
                - 5.0 * v3 * w2
                - 5.0 * ca2 * v3 * w2
                + 8.0 * ca * cf * v3 * w2
                - v4 * w2
                + 7.0 * ca2 * v4 * w2
                + 8.0 * ca * cf * v4 * w2
                + 2.0 * v5 * w2
                - 2.0 * ca2 * v5 * w2
                - 4.0 * ca2 * v2 * w3
                + 4.0 * cf * v2 * w3
                - 4.0 * ca * cf * v2 * w3
                + 6.0 * ca2 * v3 * w3
                + 4.0 * ca * cf * v3 * w3
                - v4 * w3
                - 5.0 * ca2 * v4 * w3
                - 8.0 * ca * cf * v4 * w3
                - 2.0 * v5 * w3
                + 2.0 * ca2 * v5 * w3
                + 2.0 * v5 * w4
                - 2.0 * ca2 * v5 * w4))
            / ((1.0 - v) * v2 * w * (1.0 - v * w).powi(2));

    let part3 = (2.0
        * cf
        * l1w
        * (18.0 - 18.0 * ca - 12.0 * ca2 + 22.0 * ca3 - 46.0 * v + 43.0 * ca * v + 22.0 * ca2 * v
            - 53.0 * ca3 * v
            + 46.0 * v2
            - 44.0 * ca * v2
            - 16.0 * ca2 * v2
            + 52.0 * ca3 * v2
            - 22.0 * v3
            + 23.0 * ca * v3
            + 6.0 * ca2 * v3
            - 25.0 * ca3 * v3
            + 4.0 * v4
            - 4.0 * ca * v4
            + 4.0 * ca3 * v4
            - 2.0 * w
            + 2.0 * ca * w
            + 2.0 * ca2 * w
            - 6.0 * ca3 * w
            - 14.0 * v * w
            + 22.0 * ca * v * w
            + 16.0 * ca2 * v * w
            - 10.0 * ca3 * v * w
            + 36.0 * v2 * w
            - 46.0 * ca * v2 * w
            - 32.0 * ca2 * v2 * w
            + 46.0 * ca3 * v2 * w
            - 30.0 * v3 * w
            + 34.0 * ca * v3 * w
            + 16.0 * ca2 * v3 * w
            - 50.0 * ca3 * v3 * w
            + 18.0 * v4 * w
            - 28.0 * ca * v4 * w
            - 10.0 * ca2 * v4 * w
            + 36.0 * ca3 * v4 * w
            - 4.0 * v5 * w
            + 8.0 * ca * v5 * w
            - 8.0 * ca3 * v5 * w
            + 2.0 * v * w2
            - 2.0 * ca * v * w2
            - 2.0 * ca2 * v * w2
            + 6.0 * ca3 * v * w2
            - 20.0 * v2 * w2
            + 16.0 * ca * v2 * w2
            + 18.0 * ca2 * v2 * w2
            - 40.0 * ca3 * v2 * w2
            + 10.0 * v3 * w2
            + 2.0 * ca2 * v3 * w2
            + 32.0 * ca3 * v3 * w2
            - 14.0 * v4 * w2
            + 20.0 * ca * v4 * w2
            + 6.0 * ca2 * v4 * w2
            - 20.0 * ca3 * v4 * w2
            + 2.0 * v5 * w2
            - ca * v5 * w2
            + 6.0 * ca2 * v5 * w2
            - 9.0 * ca3 * v5 * w2
            - 4.0 * ca * v6 * w2
            + 4.0 * ca3 * v6 * w2
            + 2.0 * v2 * w3
            - 2.0 * ca * v2 * w3
            - 2.0 * ca2 * v2 * w3
            + 6.0 * ca3 * v2 * w3
            + 22.0 * v3 * w3
            - 26.0 * ca * v3 * w3
            - 20.0 * ca2 * v3 * w3
            + 22.0 * ca3 * v3 * w3
            - 6.0 * v4 * w3
            + 6.0 * ca2 * v4 * w3
            - 20.0 * ca3 * v4 * w3
            + 4.0 * v5 * w3
            - 8.0 * ca * v5 * w3
            - 12.0 * ca2 * v5 * w3
            + 28.0 * ca3 * v5 * w3
            + 8.0 * ca * v6 * w3
            - 8.0 * ca3 * v6 * w3
            - 2.0 * v3 * w4
            + 2.0 * ca * v3 * w4
            + 2.0 * ca2 * v3 * w4
            - 6.0 * ca3 * v3 * w4
            - 2.0 * v4 * w4
            + 6.0 * ca * v4 * w4
            - 2.0 * ca2 * v4 * w4
            + 6.0 * ca3 * v4 * w4
            + 2.0 * v5 * w4
            + 3.0 * ca * v5 * w4
            + 6.0 * ca2 * v5 * w4
            - 13.0 * ca3 * v5 * w4
            - 8.0 * ca * v6 * w4
            + 8.0 * ca3 * v6 * w4
            - 4.0 * v5 * w5
            + 4.0 * ca * v6 * w5
            - 4.0 * ca3 * v6 * w5))
        / (t1)
        + (2.0
            * cf
            * lv
            * (26.0 - 26.0 * ca - 10.0 * ca2 + 18.0 * ca3 - 62.0 * v
                + 63.0 * ca * v
                + 22.0 * ca2 * v
                - 43.0 * ca3 * v
                + 60.0 * v2
                - 60.0 * ca * v2
                - 20.0 * ca2 * v2
                + 44.0 * ca3 * v2
                - 30.0 * v3
                + 27.0 * ca * v3
                + 10.0 * ca2 * v3
                - 23.0 * ca3 * v3
                + 6.0 * v4
                - 4.0 * ca * v4
                - 2.0 * ca2 * v4
                + 4.0 * ca3 * v4
                - 2.0 * w
                + 2.0 * ca * w
                + 2.0 * ca2 * w
                - 6.0 * ca3 * w
                - 24.0 * v * w
                + 22.0 * ca * v * w
                + 8.0 * ca2 * v * w
                - 4.0 * ca3 * v * w
                + 52.0 * v2 * w
                - 38.0 * ca * v2 * w
                - 20.0 * ca2 * v2 * w
                + 20.0 * ca3 * v2 * w
                - 36.0 * v3 * w
                + 34.0 * ca * v3 * w
                + 12.0 * ca2 * v3 * w
                - 24.0 * ca3 * v3 * w
                + 26.0 * v4 * w
                - 36.0 * ca * v4 * w
                - 10.0 * ca2 * v4 * w
                + 30.0 * ca3 * v4 * w
                - 8.0 * v5 * w
                + 8.0 * ca * v5 * w
                + 4.0 * ca2 * v5 * w
                - 8.0 * ca3 * v5 * w
                + 2.0 * v * w2
                - 2.0 * ca * v * w2
                - 2.0 * ca2 * v * w2
                + 6.0 * ca3 * v * w2
                - 30.0 * v2 * w2
                + 32.0 * ca * v2 * w2
                + 14.0 * ca2 * v2 * w2
                - 38.0 * ca3 * v2 * w2
                + 16.0 * v3 * w2
                - 32.0 * ca * v3 * w2
                - 4.0 * ca2 * v3 * w2
                + 38.0 * ca3 * v3 * w2
                - 32.0 * v4 * w2
                + 52.0 * ca * v4 * w2
                + 12.0 * ca2 * v4 * w2
                - 38.0 * ca3 * v4 * w2
                + 8.0 * v5 * w2
                + 3.0 * ca * v5 * w2
                - 4.0 * ca2 * v5 * w2
                - 5.0 * ca3 * v5 * w2
                + 2.0 * v6 * w2
                - 4.0 * ca * v6 * w2
                - 2.0 * ca2 * v6 * w2
                + 4.0 * ca3 * v6 * w2
                + 2.0 * v2 * w3
                - 2.0 * ca * v2 * w3
                - 2.0 * ca2 * v2 * w3
                + 6.0 * ca3 * v2 * w3
                + 28.0 * v3 * w3
                - 26.0 * ca * v3 * w3
                - 12.0 * ca2 * v3 * w3
                + 16.0 * ca3 * v3 * w3
                - 8.0 * ca * v4 * w3
                - 6.0 * ca3 * v4 * w3
                + 8.0 * v5 * w3
                - 24.0 * ca * v5 * w3
                - 4.0 * ca2 * v5 * w3
                + 28.0 * ca3 * v5 * w3
                - 6.0 * v6 * w3
                + 8.0 * ca * v6 * w3
                + 6.0 * ca2 * v6 * w3
                - 8.0 * ca3 * v6 * w3
                - 2.0 * v3 * w4
                + 2.0 * ca * v3 * w4
                + 2.0 * ca2 * v3 * w4
                - 6.0 * ca3 * v3 * w4
                - 2.0 * ca * v4 * w4
                + 8.0 * ca3 * v4 * w4
                - 8.0 * v5 * w4
                + 15.0 * ca * v5 * w4
                + 4.0 * ca2 * v5 * w4
                - 17.0 * ca3 * v5 * w4
                + 6.0 * v6 * w4
                - 8.0 * ca * v6 * w4
                - 6.0 * ca2 * v6 * w4
                + 8.0 * ca3 * v6 * w4
                - 2.0 * v6 * w5
                + 4.0 * ca * v6 * w5
                + 2.0 * ca2 * v6 * w5
                - 4.0 * ca3 * v6 * w5))
            / (t1);

    let part4 = -(cf
        * (16.0 + 2.0 * ca - 16.0 * ca2 + 22.0 * ca3 + 40.0 * ca * cf - 52.0 * v - 3.0 * ca * v
            + 52.0 * ca2 * v
            - 57.0 * ca3 * v
            - 112.0 * ca * cf * v
            + 56.0 * v2
            - 56.0 * ca2 * v2
            + 48.0 * ca3 * v2
            + 128.0 * ca * cf * v2
            - 20.0 * v3
            + ca * v3
            + 20.0 * ca2 * v3
            - 13.0 * ca3 * v3
            - 64.0 * ca * cf * v3
            + 8.0 * ca * cf * v4
            - 10.0 * w
            + 4.0 * ca * w
            - 2.0 * ca2 * w
            - 20.0 * ca3 * w
            - 8.0 * ca * cf * w
            + 22.0 * v * w
            - 6.0 * ca * v * w
            + 18.0 * ca2 * v * w
            + 26.0 * ca3 * v * w
            - 8.0 * ca * cf * v * w
            - 4.0 * v2 * w
            + 2.0 * ca * v2 * w
            - 36.0 * ca2 * v2 * w
            + 34.0 * ca3 * v2 * w
            + 56.0 * ca * cf * v2 * w
            - 68.0 * v3 * w
            - 10.0 * ca * v3 * w
            + 76.0 * ca2 * v3 * w
            - 50.0 * ca3 * v3 * w
            - 144.0 * ca * cf * v3 * w
            + 60.0 * v4 * w
            + 10.0 * ca * v4 * w
            - 56.0 * ca2 * v4 * w
            + 10.0 * ca3 * v4 * w
            + 136.0 * ca * cf * v4 * w
            - 16.0 * ca * cf * v5 * w
            + 10.0 * v * w2
            - 4.0 * ca * v * w2
            + 2.0 * ca2 * v * w2
            + 20.0 * ca3 * v * w2
            + 8.0 * ca * cf * v * w2
            - 64.0 * v2 * w2
            + 6.0 * ca * v2 * w2
            + 12.0 * ca2 * v2 * w2
            - 90.0 * ca3 * v2 * w2
            - 80.0 * ca * cf * v2 * w2
            + 116.0 * v3 * w2
            + 42.0 * ca * v3 * w2
            - 56.0 * ca2 * v3 * w2
            + 66.0 * ca3 * v3 * w2
            + 160.0 * ca * cf * v3 * w2
            - 34.0 * v4 * w2
            - 24.0 * ca * v4 * w2
            + 30.0 * ca2 * v4 * w2
            + 20.0 * ca3 * v4 * w2
            - 96.0 * ca * cf * v4 * w2
            - 62.0 * v5 * w2
            - 11.0 * ca * v5 * w2
            + 46.0 * ca2 * v5 * w2
            - 13.0 * ca3 * v5 * w2
            - 80.0 * ca * cf * v5 * w2
            + 8.0 * ca * cf * v6 * w2
            + 10.0 * v2 * w3
            - 4.0 * ca * v2 * w3
            + 2.0 * ca2 * v2 * w3
            + 20.0 * ca3 * v2 * w3
            + 8.0 * ca * cf * v2 * w3
            - 2.0 * v3 * w3
            - 2.0 * ca * v3 * w3
            - 14.0 * ca2 * v3 * w3
            + 14.0 * ca3 * v3 * w3
            + 24.0 * ca * cf * v3 * w3
            - 54.0 * v4 * w3
            - 34.0 * ca * v4 * w3
            + 26.0 * ca2 * v4 * w3
            - 58.0 * ca3 * v4 * w3
            - 72.0 * ca * cf * v4 * w3
            + 88.0 * v5 * w3
            + 26.0 * ca * v5 * w3
            - 56.0 * ca2 * v5 * w3
            + 38.0 * ca3 * v5 * w3
            + 136.0 * ca * cf * v5 * w3
            + 12.0 * v6 * w3
            + 4.0 * ca * v6 * w3
            - 12.0 * ca2 * v6 * w3
            - 4.0 * ca3 * v6 * w3
            - 10.0 * v3 * w4
            + 4.0 * ca * v3 * w4
            - 2.0 * ca2 * v3 * w4
            - 20.0 * ca3 * v3 * w4
            - 8.0 * ca * cf * v3 * w4
            + 28.0 * v4 * w4
            + 28.0 * ca3 * v4 * w4
            + 24.0 * ca * cf * v4 * w4
            - 26.0 * v5 * w4
            + ca * v5 * w4
            + 10.0 * ca2 * v5 * w4
            - 25.0 * ca3 * v5 * w4
            - 40.0 * ca * cf * v5 * w4
            - 12.0 * v6 * w4
            - 4.0 * ca * v6 * w4
            + 12.0 * ca2 * v6 * w4
            + 4.0 * ca3 * v6 * w4
            - 8.0 * ca * cf * v6 * w5))
        / (t1);

    let part5 = (4.0
        * cf
        * lw
        * (10.0 - 9.0 * ca - 10.0 * ca2 + 9.0 * ca3 + 10.0 * ca * cf - 10.0 * ca2cf - 22.0 * v
            + 21.0 * ca * v
            + 22.0 * ca2 * v
            - 21.0 * ca3 * v
            - 22.0 * ca * cf * v
            + 24.0 * ca2cf * v
            + 20.0 * v2
            - 24.0 * ca * v2
            - 20.0 * ca2 * v2
            + 24.0 * ca3 * v2
            + 20.0 * ca * cf * v2
            - 24.0 * ca2cf * v2
            - 10.0 * v3
            + 15.0 * ca * v3
            + 10.0 * ca2 * v3
            - 15.0 * ca3 * v3
            - 10.0 * ca * cf * v3
            + 12.0 * ca2cf * v3
            + 2.0 * v4
            - 3.0 * ca * v4
            - 2.0 * ca2 * v4
            + 3.0 * ca3 * v4
            + 2.0 * ca * cf * v4
            - 2.0 * ca2cf * v4
            + 3.0 * ca2 * w
            - 2.0 * ca3 * w
            - v * w
            - 2.0 * ca * v * w
            - 8.0 * ca2 * v * w
            + ca3 * v * w
            + 2.0 * ca * cf * v * w
            + 6.0 * ca2cf * v * w
            + 5.0 * v2 * w
            + 2.0 * ca * v2 * w
            + 2.0 * ca2 * v2 * w
            + 6.0 * ca3 * v2 * w
            - 18.0 * ca2cf * v2 * w
            - 5.0 * v3 * w
            + 9.0 * ca * v3 * w
            + 4.0 * ca2 * v3 * w
            - 13.0 * ca3 * v3 * w
            - 2.0 * ca * cf * v3 * w
            + 20.0 * ca2cf * v3 * w
            + 9.0 * v4 * w
            - 21.0 * ca * v4 * w
            - 9.0 * ca2 * v4 * w
            + 20.0 * ca3 * v4 * w
            + 8.0 * ca * cf * v4 * w
            - 16.0 * ca2cf * v4 * w
            - 4.0 * v5 * w
            + 6.0 * ca * v5 * w
            + 4.0 * ca2 * v5 * w
            - 6.0 * ca3 * v5 * w
            - 4.0 * ca * cf * v5 * w
            + 4.0 * ca2cf * v5 * w
            + w2
            - ca * w2
            - ca2 * w2
            + ca3 * w2
            + 2.0 * ca * cf * w2
            - 2.0 * ca2cf * w2
            + 6.0 * ca * v * w2
            + ca2 * v * w2
            - ca3 * v * w2
            - 2.0 * ca * cf * v * w2
            + 4.0 * ca2cf * v * w2
            - 10.0 * v2 * w2
            - 8.0 * ca * v2 * w2
            + 13.0 * ca2 * v2 * w2
            - 5.0 * ca3 * v2 * w2
            - 10.0 * ca * cf * v2 * w2
            - 2.0 * ca2cf * v2 * w2
            + 14.0 * v3 * w2
            - 9.0 * ca * v3 * w2
            - 21.0 * ca2 * v3 * w2
            + 11.0 * ca3 * v3 * w2
            + 16.0 * ca * cf * v3 * w2
            - 23.0 * v4 * w2
            + 31.0 * ca * v4 * w2
            + 24.0 * ca2 * v4 * w2
            - 25.0 * ca3 * v4 * w2
            - 22.0 * ca * cf * v4 * w2
            + 10.0 * ca2cf * v4 * w2
            + 8.0 * v5 * w2
            - ca * v5 * w2
            - 8.0 * ca2 * v5 * w2
            + 2.0 * ca3 * v5 * w2
            + 8.0 * ca * cf * v5 * w2
            + 2.0 * v6 * w2
            - 3.0 * ca * v6 * w2
            - 2.0 * ca2 * v6 * w2
            + 3.0 * ca3 * v6 * w2
            + 2.0 * ca * cf * v6 * w2
            - 2.0 * ca2cf * v6 * w2
            - v * w3
            + ca * v * w3
            + ca2 * v * w3
            - ca3 * v * w3
            - 2.0 * ca * cf * v * w3
            + 2.0 * ca2cf * v * w3
            + 3.0 * v2 * w3
            - 7.0 * ca * v2 * w3
            - 4.0 * ca2 * v2 * w3
            + 6.0 * ca3 * v2 * w3
            + 4.0 * ca * cf * v2 * w3
            - 6.0 * ca2cf * v2 * w3
            - 6.0 * v3 * w3
            + 24.0 * ca * v3 * w3
            + 7.0 * ca2 * v3 * w3
            - 9.0 * ca3 * v3 * w3
            - 4.0 * ca * cf * v3 * w3
            + 6.0 * ca2cf * v3 * w3
            + 18.0 * v4 * w3
            - 31.0 * ca * v4 * w3
            - 14.0 * ca2 * v4 * w3
            + 16.0 * ca3 * v4 * w3
            + 12.0 * ca * cf * v4 * w3
            - 10.0 * ca2cf * v4 * w3
            - 12.0 * ca * v5 * w3
            + 10.0 * ca3 * v5 * w3
            - 6.0 * ca2cf * v5 * w3
            - 8.0 * v6 * w3
            + 8.0 * ca * v6 * w3
            + 8.0 * ca2 * v6 * w3
            - 8.0 * ca3 * v6 * w3
            - 8.0 * ca * cf * v6 * w3
            + 6.0 * ca2cf * v6 * w3
            - v2 * w4
            + ca * v2 * w4
            + ca2 * v2 * w4
            - ca3 * v2 * w4
            - 2.0 * ca * cf * v2 * w4
            + 2.0 * ca2cf * v2 * w4
            - 4.0 * ca * v3 * w4
            + ca2 * v3 * w4
            - ca3 * v3 * w4
            - 2.0 * ca * cf * v3 * w4
            - 3.0 * v4 * w4
            + 5.0 * ca * v4 * w4
            + 2.0 * ca2 * v4 * w4
            - 2.0 * ca3 * v4 * w4
            + 4.0 * ca2cf * v4 * w4
            - 12.0 * v5 * w4
            + 18.0 * ca * v5 * w4
            + 8.0 * ca2 * v5 * w4
            - 13.0 * ca3 * v5 * w4
            - 8.0 * ca * cf * v5 * w4
            + 8.0 * ca2cf * v5 * w4
            + 12.0 * v6 * w4
            - 8.0 * ca * v6 * w4
            - 12.0 * ca2 * v6 * w4
            + 8.0 * ca3 * v6 * w4
            + 12.0 * ca * cf * v6 * w4
            - 8.0 * ca2cf * v6 * w4
            + v3 * w5
            - ca * v3 * w5
            - ca2 * v3 * w5
            + ca3 * v3 * w5
            + 2.0 * ca * cf * v3 * w5
            - 2.0 * ca2cf * v3 * w5
            - v4 * w5
            + 5.0 * ca * v4 * w5
            - ca2 * v4 * w5
            - 2.0 * ca3 * v4 * w5
            + 2.0 * ca2cf * v4 * w5
            + 10.0 * v5 * w5
            - 11.0 * ca * v5 * w5
            - 4.0 * ca2 * v5 * w5
            + 7.0 * ca3 * v5 * w5
            + 4.0 * ca * cf * v5 * w5
            - 6.0 * ca2cf * v5 * w5
            - 8.0 * v6 * w5
            + 4.0 * ca * v6 * w5
            + 8.0 * ca2 * v6 * w5
            - 4.0 * ca3 * v6 * w5
            - 8.0 * ca * cf * v6 * w5
            + 6.0 * ca2cf * v6 * w5
            - 2.0 * v5 * w6
            + 2.0 * v6 * w6
            - ca * v6 * w6
            - 2.0 * ca2 * v6 * w6
            + ca3 * v6 * w6
            + 2.0 * ca * cf * v6 * w6
            - 2.0 * ca2cf * v6 * w6))
        / (ca * (1.0 - v) * v2 * (1.0 - w) * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w));

    part1 + part2 + part3 + part4 + part5
}

/// `STRUV7(W,V,X3,S)`.
#[must_use]
pub fn qq_identical_elastic_gluon_frag(
    w: f64,
    v: f64,
    _x3: f64,
    _s: f64,
    ctx: &MeContext,
    pre: &Precalc,
) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7);
    let (w2, w3, w4, w5, w6) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6);
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let cacf2 = ca * cf.powi(2);
    let (l1v, lv, l1w, lw, lvw, l1vw, lms, lmss) = (
        pre.l1v, pre.lv, pre.l1w, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.lmss,
    );

    let t1 = ca * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2);

    let part1 = (8.0
        * cf.powi(2)
        * lmss
        * (1.0 + v2 - 2.0 * v2 * w + v2 * w2)
        * (1.0 - ca - 2.0 * v + 2.0 * ca * v + v2 - ca * v2 + 2.0 * v * w
            - ca * v * w
            - 2.0 * v2 * w
            + ca * v2 * w
            + v2 * w2
            - ca * v2 * w2))
        / ((1.0 - v) * v * w * (1.0 - v + v * w).powi(2))
        - (4.0
            * cf
            * lvw
            * (1.0 - w)
            * (2.0 - 5.0 * ca2 + 2.0 * v - 3.0 * ca2 * v - 3.0 * v * w + 6.0 * ca2 * v * w
                - v2 * w
                + ca2 * v2 * w
                + 2.0 * v2 * w2
                - 2.0 * ca2 * v2 * w2))
            / ((1.0 - v) * v * w)
        - (16.0
            * cacf2
            * l1vw
            * (1.0 - w)
            * (1.0 - v - v2 + v3 + v * w + 4.0 * v2 * w - 5.0 * v3 * w - 3.0 * v2 * w2
                + 5.0 * v3 * w2
                - v3 * w3))
            / ((1.0 - v) * v * w * (1.0 - v + v * w).powi(2))
        + (2.0
            * cf
            * lms
            * (2.0 - 2.0 * ca2 - 2.0 * v - 6.0 * ca2 * v - 2.0 * v2 - 6.0 * ca2 * v2 + 2.0 * v3
                - 2.0 * ca2 * v3
                - w
                + ca2 * w
                - 3.0 * v * w
                + 5.0 * ca2 * v * w
                + 5.0 * v2 * w
                + 19.0 * ca2 * v2 * w
                + 3.0 * v3 * w
                + 11.0 * ca2 * v3 * w
                - 4.0 * v4 * w
                + 4.0 * ca2 * v4 * w
                + 2.0 * v * w2
                - 2.0 * ca2 * v * w2
                - 4.0 * ca2 * v2 * w2
                - 2.0 * v3 * w2
                - 20.0 * ca2 * v3 * w2
                + 2.0 * v4 * w2
                - 8.0 * ca2 * v4 * w2
                + 2.0 * v5 * w2
                - 2.0 * ca2 * v5 * w2
                - v2 * w3
                + ca2 * v2 * w3
                + v3 * w3
                + ca2 * v3 * w3
                - 4.0 * v4 * w3
                + 10.0 * ca2 * v4 * w3
                - 2.0 * v5 * w3
                + 2.0 * ca2 * v5 * w3
                + 2.0 * v5 * w4
                - 2.0 * ca2 * v5 * w4))
            / ((1.0 - v) * v2 * w * (1.0 - v * w).powi(2));

    let part2 = (4.0
        * cf
        * l1v
        * (1.0 + ca2 - 2.0 * ca * v - ca3 * v + 2.0 * ca3 * v2 + 2.0 * ca * v3
            - ca3 * v3
            - v4
            - ca2 * v4
            - v * w
            - ca2 * v * w
            - 2.0 * v2 * w
            + 6.0 * ca * v2 * w
            + ca3 * v2 * w
            - 2.0 * ca2 * v3 * w
            - 2.0 * ca3 * v3 * w
            + 2.0 * v4 * w
            - 2.0 * ca * v4 * w
            + 4.0 * ca2 * v4 * w
            + ca3 * v4 * w
            + v5 * w
            - ca2 * v5 * w
            + v2 * w2
            + ca2 * v2 * w2
            + v3 * w2
            - 6.0 * ca * v3 * w2
            + 3.0 * ca2 * v3 * w2
            + ca3 * v3 * w2
            - v4 * w2
            - 5.0 * ca2 * v4 * w2
            - 3.0 * v5 * w2
            + 3.0 * ca2 * v5 * w2
            - v3 * w3
            - ca2 * v3 * w3
            + 2.0 * ca * v4 * w3
            + 2.0 * ca2 * v4 * w3
            - ca3 * v4 * w3
            + 3.0 * v5 * w3
            - 3.0 * ca2 * v5 * w3
            - v5 * w4
            + ca2 * v5 * w4))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v * w) * (1.0 - v + v * w))
        + (4.0
            * cf
            * lw
            * (2.0 * ca * cf + 2.0 * v - 2.0 * ca2 * v + 4.0 * ca3 * v - 2.0 * v2
                + 2.0 * ca * v2
                + 2.0 * ca2 * v2
                - 2.0 * ca3 * v2
                + 2.0 * v3
                - 4.0 * ca * v3
                - 2.0 * ca2 * v3
                - v4
                + 2.0 * ca * v4
                + ca2 * v4
                - 2.0 * ca3 * v4
                + v * w
                + ca2 * v * w
                + 2.0 * v2 * w
                - 5.0 * ca * v2 * w
                - 3.0 * ca3 * v2 * w
                - 2.0 * v3 * w
                - 2.0 * ca * v3 * w
                + 4.0 * ca2 * v3 * w
                + 6.0 * ca3 * v3 * w
                + 2.0 * v4 * w
                + ca * v4 * w
                + 3.0 * ca3 * v4 * w
                + v5 * w
                - 2.0 * ca * v5 * w
                - ca2 * v5 * w
                + 2.0 * ca3 * v5 * w
                - v2 * w2
                - ca2 * v2 * w2
                - v3 * w2
                + 8.0 * ca * v3 * w2
                - 3.0 * ca2 * v3 * w2
                - 4.0 * ca3 * v3 * w2
                - v4 * w2
                + ca * v4 * w2
                - 5.0 * ca2 * v4 * w2
                - 5.0 * ca3 * v4 * w2
                - 3.0 * v5 * w2
                + 3.0 * ca * v5 * w2
                + 3.0 * ca2 * v5 * w2
                - 3.0 * ca3 * v5 * w2
                + v3 * w3
                + ca2 * v3 * w3
                - 3.0 * ca * v4 * w3
                + 6.0 * ca2 * v4 * w3
                + 3.0 * ca3 * v4 * w3
                + 3.0 * v5 * w3
                - ca * v5 * w3
                - 3.0 * ca2 * v5 * w3
                + ca3 * v5 * w3
                - 2.0 * ca2 * v4 * w4
                - v5 * w4
                + ca2 * v5 * w4))
            / (ca * (1.0 - v) * v2 * w * (1.0 - v * w) * (1.0 - v + v * w));

    let part3 = (2.0
        * cf
        * (3.0 * ca - 3.0 * ca3 + 2.0 * v - 10.0 * ca * v - 2.0 * ca2 * v + 6.0 * ca3 * v
            - 8.0 * v2
            + 14.0 * ca * v2
            + 8.0 * ca2 * v2
            - 18.0 * ca3 * v2
            + 12.0 * v3
            - 10.0 * ca * v3
            - 12.0 * ca2 * v3
            + 30.0 * ca3 * v3
            - 8.0 * v4
            + 3.0 * ca * v4
            + 8.0 * ca2 * v4
            - 15.0 * ca3 * v4
            + 2.0 * v5
            - 2.0 * ca2 * v5
            - 4.0 * ca * w
            + 4.0 * ca3 * w
            + 17.0 * ca * v * w
            - 9.0 * ca3 * v * w
            + 4.0 * v2 * w
            - 30.0 * ca * v2 * w
            - 4.0 * ca2 * v2 * w
            + 18.0 * ca3 * v2 * w
            - 8.0 * v3 * w
            - 4.0 * ca * v3 * w
            + 8.0 * ca2 * v3 * w
            - 28.0 * ca3 * v3 * w
            + 30.0 * ca * v4 * w
            - 10.0 * ca3 * v4 * w
            + 8.0 * v5 * w
            - 9.0 * ca * v5 * w
            - 8.0 * ca2 * v5 * w
            + 25.0 * ca3 * v5 * w
            - 4.0 * v6 * w
            + 4.0 * ca2 * v6 * w
            - 14.0 * ca * v2 * w2
            + 14.0 * ca3 * v2 * w2
            - 2.0 * v3 * w2
            + 76.0 * ca * v3 * w2
            + 2.0 * ca2 * v3 * w2
            - 20.0 * ca3 * v3 * w2
            + 16.0 * v4 * w2
            - 42.0 * ca * v4 * w2
            - 16.0 * ca2 * v4 * w2
            + 34.0 * ca3 * v4 * w2
            - 24.0 * v5 * w2
            - 30.0 * ca * v5 * w2
            + 24.0 * ca2 * v5 * w2
            - 30.0 * ca3 * v5 * w2
            + 8.0 * v6 * w2
            + 12.0 * ca * v6 * w2
            - 8.0 * ca2 * v6 * w2
            - 16.0 * ca3 * v6 * w2
            + 2.0 * v7 * w2
            - 2.0 * ca2 * v7 * w2
            + 8.0 * ca * v2 * w3
            - 8.0 * ca3 * v2 * w3
            - 26.0 * ca * v3 * w3
            + 10.0 * ca3 * v3 * w3
            - 8.0 * v4 * w3
            - 28.0 * ca * v4 * w3
            + 8.0 * ca2 * v4 * w3
            + 4.0 * ca3 * v4 * w3
            + 16.0 * v5 * w3
            + 58.0 * ca * v5 * w3
            - 16.0 * ca2 * v5 * w3
            + 6.0 * ca3 * v5 * w3
            - 14.0 * ca * v6 * w3
            + 34.0 * ca3 * v6 * w3
            - 8.0 * v7 * w3
            - 2.0 * ca * v7 * w3
            + 8.0 * ca2 * v7 * w3
            + 2.0 * ca3 * v7 * w3
            + 11.0 * ca * v4 * w4
            - 11.0 * ca3 * v4 * w4
            - 2.0 * v5 * w4
            - 18.0 * ca * v5 * w4
            + 2.0 * ca2 * v5 * w4
            - 2.0 * ca3 * v5 * w4
            - 8.0 * v6 * w4
            + 8.0 * ca * v6 * w4
            + 8.0 * ca2 * v6 * w4
            - 28.0 * ca3 * v6 * w4
            + 12.0 * v7 * w4
            + 4.0 * ca * v7 * w4
            - 12.0 * ca2 * v7 * w4
            - 4.0 * ca3 * v7 * w4
            - 4.0 * ca * v4 * w5
            + 4.0 * ca3 * v4 * w5
            + 9.0 * ca * v5 * w5
            - ca3 * v5 * w5
            + 4.0 * v6 * w5
            - 6.0 * ca * v6 * w5
            - 4.0 * ca2 * v6 * w5
            + 10.0 * ca3 * v6 * w5
            - 8.0 * v7 * w5
            - 2.0 * ca * v7 * w5
            + 8.0 * ca2 * v7 * w5
            + 2.0 * ca3 * v7 * w5
            + 2.0 * v7 * w6
            - 2.0 * ca2 * v7 * w6))
        / (t1)
        - (2.0
            * cf
            * lv
            * (2.0 * ca - 2.0 * ca3 - 2.0 * v - 4.0 * ca * v + 6.0 * ca2 * v - 12.0 * ca3 * v
                + 4.0 * v2
                - 12.0 * ca2 * v2
                + 24.0 * ca3 * v2
                - 4.0 * v3
                + 8.0 * ca * v3
                + 12.0 * ca2 * v3
                - 8.0 * ca3 * v3
                + 4.0 * v4
                - 10.0 * ca * v4
                - 12.0 * ca2 * v4
                + 2.0 * ca3 * v4
                - 2.0 * v5
                + 4.0 * ca * v5
                + 6.0 * ca2 * v5
                - 4.0 * ca3 * v5
                - ca * w
                + ca3 * w
                + 3.0 * ca * v * w
                - ca3 * v * w
                - 4.0 * v2 * w
                + 16.0 * ca * v2 * w
                + 8.0 * ca3 * v2 * w
                + 4.0 * v3 * w
                - 24.0 * ca * v3 * w
                - 4.0 * ca2 * v3 * w
                - 48.0 * ca3 * v3 * w
                - 8.0 * v4 * w
                + 5.0 * ca * v4 * w
                + 16.0 * ca2 * v4 * w
                + 27.0 * ca3 * v4 * w
                + 4.0 * v5 * w
                + 9.0 * ca * v5 * w
                - 4.0 * ca2 * v5 * w
                + 5.0 * ca3 * v5 * w
                + 4.0 * v6 * w
                - 8.0 * ca * v6 * w
                - 8.0 * ca2 * v6 * w
                + 8.0 * ca3 * v6 * w
                - 6.0 * ca * v2 * w2
                + 6.0 * ca3 * v2 * w2
                + 2.0 * v3 * w2
                - 8.0 * ca * v3 * w2
                - 2.0 * ca2 * v3 * w2
                + 22.0 * ca3 * v3 * w2
                + 4.0 * v4 * w2
                + 40.0 * ca * v4 * w2
                - 4.0 * ca2 * v4 * w2
                - 2.0 * ca3 * v4 * w2
                - 24.0 * ca * v5 * w2
                - 16.0 * ca2 * v5 * w2
                - 30.0 * ca3 * v5 * w2
                - 16.0 * v6 * w2
                + 14.0 * ca * v6 * w2
                + 24.0 * ca2 * v6 * w2
                - 20.0 * ca3 * v6 * w2
                - 2.0 * v7 * w2
                + 4.0 * ca * v7 * w2
                + 2.0 * ca2 * v7 * w2
                - 4.0 * ca3 * v7 * w2
                + 2.0 * ca * v2 * w3
                - 2.0 * ca3 * v2 * w3
                - 4.0 * ca * v3 * w3
                - 18.0 * ca * v4 * w3
                - 8.0 * ca3 * v4 * w3
                - 4.0 * v5 * w3
                - 10.0 * ca * v5 * w3
                + 20.0 * ca2 * v5 * w3
                + 38.0 * ca3 * v5 * w3
                + 24.0 * v6 * w3
                - 10.0 * ca * v6 * w3
                - 24.0 * ca2 * v6 * w3
                + 32.0 * ca3 * v6 * w3
                + 8.0 * v7 * w3
                - 12.0 * ca * v7 * w3
                - 8.0 * ca2 * v7 * w3
                + 12.0 * ca3 * v7 * w3
                + 4.0 * ca * v4 * w4
                - 4.0 * ca3 * v4 * w4
                + 2.0 * v5 * w4
                + 16.0 * ca * v5 * w4
                - 6.0 * ca2 * v5 * w4
                - 14.0 * ca3 * v5 * w4
                - 16.0 * v6 * w4
                + 14.0 * ca * v6 * w4
                + 8.0 * ca2 * v6 * w4
                - 32.0 * ca3 * v6 * w4
                - 12.0 * v7 * w4
                + 16.0 * ca * v7 * w4
                + 12.0 * ca2 * v7 * w4
                - 16.0 * ca3 * v7 * w4
                - ca * v4 * w5
                + ca3 * v4 * w5
                + ca * v5 * w5
                + ca3 * v5 * w5
                + 4.0 * v6 * w5
                - 10.0 * ca * v6 * w5
                + 12.0 * ca3 * v6 * w5
                + 8.0 * v7 * w5
                - 12.0 * ca * v7 * w5
                - 8.0 * ca2 * v7 * w5
                + 12.0 * ca3 * v7 * w5
                - 2.0 * v7 * w6
                + 4.0 * ca * v7 * w6
                + 2.0 * ca2 * v7 * w6
                - 4.0 * ca3 * v7 * w6))
            / (t1);

    let part4 = (2.0
        * cf
        * l1w
        * (2.0 - 2.0 * ca - 2.0 * ca2 + 2.0 * ca3 - 4.0 * v
            + 10.0 * ca3 * v
            + 4.0 * v2
            + 4.0 * ca * v2
            + 4.0 * ca2 * v2
            - 18.0 * ca3 * v2
            - 4.0 * v3
            - 4.0 * ca * v3
            - 4.0 * ca2 * v3
            + 2.0 * ca3 * v3
            + 2.0 * v4
            + 6.0 * ca * v4
            + 6.0 * ca2 * v4
            - 4.0 * ca * v5
            - 4.0 * ca2 * v5
            + 4.0 * ca3 * v5
            + ca * w
            - ca3 * w
            - 2.0 * v * w
            - 3.0 * ca * v * w
            - 2.0 * ca2 * v * w
            + ca3 * v * w
            + 4.0 * v2 * w
            - 4.0 * ca * v2 * w
            - 6.0 * ca3 * v2 * w
            + 8.0 * ca * v3 * w
            + 40.0 * ca3 * v3 * w
            + 4.0 * v4 * w
            - 9.0 * ca * v4 * w
            - 12.0 * ca2 * v4 * w
            - 17.0 * ca3 * v4 * w
            - 6.0 * v5 * w
            - ca * v5 * w
            + 10.0 * ca2 * v5 * w
            - 9.0 * ca3 * v5 * w
            + 8.0 * ca * v6 * w
            + 4.0 * ca2 * v6 * w
            - 8.0 * ca3 * v6 * w
            + 6.0 * ca * v2 * w2
            + 4.0 * ca2 * v2 * w2
            - 6.0 * ca3 * v2 * w2
            - 18.0 * ca3 * v3 * w2
            - 12.0 * v4 * w2
            - 16.0 * ca * v4 * w2
            + 12.0 * ca2 * v4 * w2
            - 2.0 * ca3 * v4 * w2
            + 12.0 * v5 * w2
            + 20.0 * ca * v5 * w2
            - 12.0 * ca2 * v5 * w2
            + 28.0 * ca3 * v5 * w2
            + 4.0 * v6 * w2
            - 18.0 * ca * v6 * w2
            - 16.0 * ca2 * v6 * w2
            + 22.0 * ca3 * v6 * w2
            - 4.0 * ca * v7 * w2
            + 4.0 * ca3 * v7 * w2
            - 2.0 * ca * v2 * w3
            + 2.0 * ca3 * v2 * w3
            + 4.0 * ca * v3 * w3
            + 8.0 * v4 * w3
            + 10.0 * ca * v4 * w3
            - 8.0 * ca2 * v4 * w3
            + 4.0 * ca3 * v4 * w3
            - 4.0 * v5 * w3
            - 6.0 * ca * v5 * w3
            + 12.0 * ca2 * v5 * w3
            - 30.0 * ca3 * v5 * w3
            - 12.0 * v6 * w3
            + 14.0 * ca * v6 * w3
            + 28.0 * ca2 * v6 * w3
            - 34.0 * ca3 * v6 * w3
            + 12.0 * ca * v7 * w3
            - 12.0 * ca3 * v7 * w3
            - 2.0 * v4 * w4
            - 4.0 * ca * v4 * w4
            + 2.0 * ca2 * v4 * w4
            + 4.0 * ca3 * v4 * w4
            - 4.0 * v5 * w4
            - 4.0 * ca * v5 * w4
            - 8.0 * ca2 * v5 * w4
            + 12.0 * ca3 * v5 * w4
            + 12.0 * v6 * w4
            - 10.0 * ca * v6 * w4
            - 28.0 * ca2 * v6 * w4
            + 30.0 * ca3 * v6 * w4
            - 16.0 * ca * v7 * w4
            + 16.0 * ca3 * v7 * w4
            + ca * v4 * w5
            - ca3 * v4 * w5
            + 2.0 * v5 * w5
            - ca * v5 * w5
            + 2.0 * ca2 * v5 * w5
            - ca3 * v5 * w5
            - 4.0 * v6 * w5
            + 6.0 * ca * v6 * w5
            + 16.0 * ca2 * v6 * w5
            - 10.0 * ca3 * v6 * w5
            + 12.0 * ca * v7 * w5
            - 12.0 * ca3 * v7 * w5
            - 4.0 * ca2 * v6 * w6
            - 4.0 * ca * v7 * w6
            + 4.0 * ca3 * v7 * w6))
        / (t1);

    part1 + part2 + part3 + part4
}

/// `STRUV8(W,V,X3,S)`.
#[must_use]
pub fn qg_seapair_unlike_quark_frag(
    w: f64,
    v: f64,
    _x3: f64,
    _s: f64,
    ctx: &MeContext,
    pre: &Precalc,
) -> f64 {
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7);
    let (w2, w3, w4, w5, w6) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6);
    let ca2 = pre.ca2;
    let (l1v, lv, l1w, lw, lvw, l1vw, lms, lmss) = (
        pre.l1v, pre.lv, pre.l1w, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.lmss,
    );

    let t1 = (1.0 - v) * v2 * w * (1.0 - v * w) * (1.0 - v + v * w).powi(2);

    let part1 = -(2.0
        * cf
        * lmss
        * (2.0 - 2.0 * v + v * w)
        * (1.0 - 2.0 * v + 2.0 * v2 + 2.0 * v * w - 4.0 * v2 * w + 2.0 * v2 * w2)
        * (2.0 * ca2 - 4.0 * ca2 * v + 2.0 * ca2 * v2 + 2.0 * ca2 * v * w
            - 2.0 * ca2 * v2 * w
            - v2 * w2
            + ca2 * v2 * w2))
        / ((1.0 - v) * v * w * (1.0 - v + v * w).powi(2))
        + (4.0
            * cf
            * l1vw
            * (1.0 - w)
            * (2.0 - 8.0 * v + 2.0 * ca2 * v + 10.0 * v2 - 4.0 * ca2 * v2 - 4.0 * v3
                + 2.0 * ca2 * v3
                + 5.0 * v * w
                - ca2 * v * w
                - 13.0 * v2 * w
                + 3.0 * ca2 * v2 * w
                + 8.0 * v3 * w
                - 2.0 * ca2 * v3 * w
                + 4.0 * v2 * w2
                - 6.0 * v3 * w2
                + 2.0 * v3 * w3))
            / (v * w * (1.0 - v + v * w).powi(2))
        - (8.0
            * cf
            * l1v
            * (1.0 + ca2 - 2.0 * v - ca2 * v - 2.0 * v * w + 6.0 * v2 * w + ca2 * v2 * w
                - 3.0 * v2 * w2
                - 2.0 * ca2 * v3 * w2
                + ca2 * v3 * w3))
            / (v * w * (1.0 - v * w))
        + (8.0
            * cf
            * lvw
            * (1.0 - w)
            * (ca2 + v - 2.0 * ca2 * v + ca2 * v * w - v2 * w - v2 * w2
                + ca2 * v2 * w2
                + 2.0 * v3 * w2
                - 2.0 * ca2 * v3 * w2
                - v3 * w3
                + ca2 * v3 * w3))
            / ((1.0 - v) * v * w);

    let part2 = (4.0
        * cf
        * lw
        * (4.0 * ca2 + 2.0 * v - 14.0 * ca2 * v + 16.0 * ca2 * v2 - 8.0 * ca2 * v3 - 5.0 * v * w
            + 3.0 * ca2 * v * w
            + 2.0 * v2 * w
            - 8.0 * ca2 * v3 * w
            + 8.0 * ca2 * v4 * w
            + v2 * w2
            - 3.0 * ca2 * v2 * w2
            + 10.0 * ca2 * v3 * w2
            - 8.0 * ca2 * v4 * w2
            - 2.0 * ca2 * v3 * w3
            + 2.0 * ca2 * v4 * w3))
        / ((1.0 - v) * v * w * (1.0 - v * w))
        - (2.0
            * cf
            * lms
            * (2.0 - 2.0 * ca2 - 6.0 * v + 10.0 * ca2 * v + 8.0 * v2 - 20.0 * ca2 * v2
                + 16.0 * ca2 * v3
                - 8.0 * ca2 * v4
                - w
                + ca2 * w
                + 2.0 * v * w
                - 4.0 * ca2 * v * w
                - 4.0 * v2 * w
                + 10.0 * ca2 * v2 * w
                - 8.0 * ca2 * v3 * w
                + 4.0 * ca2 * v4 * w
                - 2.0 * v3 * w2
                + 2.0 * ca2 * v3 * w2
                + 4.0 * v4 * w2
                - 4.0 * ca2 * v4 * w2
                - 2.0 * v4 * w3
                + 2.0 * ca2 * v4 * w3))
            / ((1.0 - v) * v2 * w)
        - (2.0
            * cf
            * (3.0 - 3.0 * ca2 - 13.0 * v + 17.0 * ca2 * v + 27.0 * v2
                - 55.0 * ca2 * v2
                - 27.0 * v3
                + 103.0 * ca2 * v3
                + 10.0 * v4
                - 110.0 * ca2 * v4
                + 64.0 * ca2 * v5
                - 16.0 * ca2 * v6
                - 4.0 * w
                + 4.0 * ca2 * w
                + 25.0 * v * w
                - 27.0 * ca2 * v * w
                - 56.0 * v2 * w
                + 88.0 * ca2 * v2 * w
                + 67.0 * v3 * w
                - 193.0 * ca2 * v3 * w
                - 34.0 * v4 * w
                + 250.0 * ca2 * v4 * w
                + 2.0 * v5 * w
                - 174.0 * ca2 * v5 * w
                + 52.0 * ca2 * v6 * w
                - 8.0 * v * w2
                + 8.0 * ca2 * v * w2
                + 33.0 * v2 * w2
                - 37.0 * ca2 * v2 * w2
                - 55.0 * v3 * w2
                + 105.0 * ca2 * v3 * w2
                + 47.0 * v4 * w2
                - 189.0 * ca2 * v4 * w2
                - 19.0 * v5 * w2
                + 175.0 * ca2 * v5 * w2
                + 8.0 * v6 * w2
                - 68.0 * ca2 * v6 * w2
                - 4.0 * v2 * w3
                + 4.0 * ca2 * v2 * w3
                + 11.0 * v3 * w3
                - 13.0 * ca2 * v3 * w3
                - 22.0 * v4 * w3
                + 48.0 * ca2 * v4 * w3
                + 31.0 * v5 * w3
                - 83.0 * ca2 * v5 * w3
                - 22.0 * v6 * w3
                + 50.0 * ca2 * v6 * w3
                - 14.0 * v5 * w4
                + 18.0 * ca2 * v5 * w4
                + 20.0 * v6 * w4
                - 24.0 * ca2 * v6 * w4
                - 6.0 * v6 * w5
                + 6.0 * ca2 * v6 * w5))
            / ((1.0 - v) * v2 * w * (1.0 - v + v * w).powi(2));

    let part3 = (2.0
        * cf
        * l1w
        * (2.0 - 2.0 * ca2 - 6.0 * v + 22.0 * ca2 * v + 2.0 * v2 - 78.0 * ca2 * v2
            + 14.0 * v3
            + 134.0 * ca2 * v3
            - 20.0 * v4
            - 128.0 * ca2 * v4
            + 8.0 * v5
            + 68.0 * ca2 * v5
            - 16.0 * ca2 * v6
            - w
            + ca2 * w
            + 6.0 * v * w
            - 8.0 * ca2 * v * w
            - 15.0 * v2 * w
            + 47.0 * ca2 * v2 * w
            + 32.0 * v3 * w
            - 114.0 * ca2 * v3 * w
            - 54.0 * v4 * w
            + 118.0 * ca2 * v4 * w
            + 56.0 * v5 * w
            - 40.0 * ca2 * v5 * w
            - 24.0 * v6 * w
            - 20.0 * ca2 * v6 * w
            + 16.0 * ca2 * v7 * w
            - v * w2
            + ca2 * v * w2
            - 2.0 * ca2 * v2 * w2
            - 21.0 * v3 * w2
            + 11.0 * ca2 * v3 * w2
            + 96.0 * v4 * w2
            - 138.0 * v5 * w2
            - 54.0 * ca2 * v5 * w2
            + 68.0 * v6 * w2
            + 80.0 * ca2 * v6 * w2
            - 40.0 * ca2 * v7 * w2
            + v2 * w3
            - ca2 * v2 * w3
            - 6.0 * v3 * w3
            + 8.0 * ca2 * v3 * w3
            - 25.0 * v4 * w3
            - 23.0 * ca2 * v4 * w3
            + 80.0 * v5 * w3
            + 50.0 * ca2 * v5 * w3
            - 50.0 * v6 * w3
            - 66.0 * ca2 * v6 * w3
            - 8.0 * v7 * w3
            + 40.0 * ca2 * v7 * w3
            + v3 * w4
            - ca2 * v3 * w4
            - 2.0 * v4 * w4
            + 4.0 * ca2 * v4 * w4
            - 9.0 * v5 * w4
            - 13.0 * ca2 * v5 * w4
            - 2.0 * v6 * w4
            + 30.0 * ca2 * v6 * w4
            + 20.0 * v7 * w4
            - 28.0 * ca2 * v7 * w4
            + 8.0 * v6 * w5
            - 8.0 * ca2 * v6 * w5
            - 16.0 * v7 * w5
            + 16.0 * ca2 * v7 * w5
            + 4.0 * v7 * w6
            - 4.0 * ca2 * v7 * w6))
        / (t1)
        + (2.0
            * cf
            * lv
            * (2.0 - 2.0 * ca2 - 10.0 * v + 26.0 * ca2 * v + 22.0 * v2
                - 94.0 * ca2 * v2
                - 22.0 * v3
                + 158.0 * ca2 * v3
                + 8.0 * v4
                - 144.0 * ca2 * v4
                + 72.0 * ca2 * v5
                - 16.0 * ca2 * v6
                - w
                + ca2 * w
                + 6.0 * v * w
                - 8.0 * ca2 * v * w
                - 31.0 * v2 * w
                + 59.0 * ca2 * v2 * w
                + 80.0 * v3 * w
                - 154.0 * ca2 * v3 * w
                - 94.0 * v4 * w
                + 166.0 * ca2 * v4 * w
                + 56.0 * v5 * w
                - 64.0 * ca2 * v5 * w
                - 16.0 * v6 * w
                - 16.0 * ca2 * v6 * w
                + 16.0 * ca2 * v7 * w
                - v * w2
                + ca2 * v * w2
                - 2.0 * ca2 * v2 * w2
                - 37.0 * v3 * w2
                + 23.0 * ca2 * v3 * w2
                + 112.0 * v4 * w2
                - 24.0 * ca2 * v4 * w2
                - 118.0 * v5 * w2
                - 50.0 * ca2 * v5 * w2
                + 48.0 * v6 * w2
                + 96.0 * ca2 * v6 * w2
                - 48.0 * ca2 * v7 * w2
                + v2 * w3
                - ca2 * v2 * w3
                - 6.0 * v3 * w3
                + 8.0 * ca2 * v3 * w3
                - 25.0 * v4 * w3
                - 23.0 * ca2 * v4 * w3
                + 64.0 * v5 * w3
                + 70.0 * ca2 * v5 * w3
                - 34.0 * v6 * w3
                - 106.0 * ca2 * v6 * w3
                - 8.0 * v7 * w3
                + 60.0 * ca2 * v7 * w3
                + v3 * w4
                - ca2 * v3 * w4
                - 2.0 * v4 * w4
                + 4.0 * ca2 * v4 * w4
                - 5.0 * v5 * w4
                - 21.0 * ca2 * v5 * w4
                - 6.0 * v6 * w4
                + 54.0 * ca2 * v6 * w4
                + 20.0 * v7 * w4
                - 44.0 * ca2 * v7 * w4
                + 8.0 * v6 * w5
                - 12.0 * ca2 * v6 * w5
                - 16.0 * v7 * w5
                + 20.0 * ca2 * v7 * w5
                + 4.0 * v7 * w6
                - 4.0 * ca2 * v7 * w6))
            / (t1);

    part1 + part2 + part3
}

/// `STRUV9(W,V,X3,S)`.
#[must_use]
pub fn qg_seapair_unlike_antiquark_frag(
    w: f64,
    v: f64,
    _x3: f64,
    _s: f64,
    ctx: &MeContext,
    pre: &Precalc,
) -> f64 {
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7);
    let (w2, w3, w4, w5, w6) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6);
    let ca2 = pre.ca2;
    let (l1v, lv, l1w, lw, lvw, l1vw, lms, lmss) = (
        pre.l1v, pre.lv, pre.l1w, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.lmss,
    );

    let t1 = (1.0 - v) * v2 * w * (1.0 - v * w) * (1.0 - v + v * w).powi(2);

    let part1 = -(2.0
        * cf
        * lmss
        * (2.0 - 2.0 * v + v * w)
        * (1.0 - 2.0 * v + 2.0 * v2 + 2.0 * v * w - 4.0 * v2 * w + 2.0 * v2 * w2)
        * (2.0 * ca2 - 4.0 * ca2 * v + 2.0 * ca2 * v2 + 2.0 * ca2 * v * w
            - 2.0 * ca2 * v2 * w
            - v2 * w2
            + ca2 * v2 * w2))
        / ((1.0 - v) * v * w * (1.0 - v + v * w).powi(2))
        + (4.0
            * cf
            * l1v
            * (2.0 - 3.0 * ca2 - 4.0 * v + 4.0 * ca2 * v - 4.0 * v * w
                + 2.0 * ca2 * v * w
                + 12.0 * v2 * w
                - 8.0 * ca2 * v2 * w
                - 6.0 * v2 * w2
                + 3.0 * ca2 * v2 * w2
                + 4.0 * ca2 * v3 * w2
                - 2.0 * ca2 * v3 * w3))
            / (v * w * (1.0 - v * w))
        - (4.0
            * cf
            * l1vw
            * (1.0 - w)
            * (2.0 - ca2 - 8.0 * v + 2.0 * ca2 * v + 10.0 * v2 - ca2 * v2 - 4.0 * v3
                + 7.0 * v * w
                - 2.0 * ca2 * v * w
                - 19.0 * v2 * w
                + 5.0 * ca2 * v2 * w
                + 12.0 * v3 * w
                - 3.0 * ca2 * v3 * w
                + 8.0 * v2 * w2
                - 3.0 * ca2 * v2 * w2
                - 10.0 * v3 * w2
                + 4.0 * ca2 * v3 * w2
                + 2.0 * v3 * w3
                - ca2 * v3 * w3))
            / (v * w * (1.0 - v + v * w).powi(2))
        + (8.0
            * cf
            * lvw
            * (1.0 - w)
            * (ca2 + v - 2.0 * ca2 * v + ca2 * v * w - v2 * w - v2 * w2
                + ca2 * v2 * w2
                + 2.0 * v3 * w2
                - 2.0 * ca2 * v3 * w2
                - v3 * w3
                + ca2 * v3 * w3))
            / ((1.0 - v) * v * w);

    let part2 = (4.0
        * cf
        * lw
        * (4.0 * ca2 + 2.0 * v - 14.0 * ca2 * v + 16.0 * ca2 * v2 - 8.0 * ca2 * v3
            + 3.0 * v * w
            + ca2 * v * w
            - 6.0 * v2 * w
            + 2.0 * ca2 * v2 * w
            - 8.0 * ca2 * v3 * w
            + 8.0 * ca2 * v4 * w
            + v2 * w2
            - 3.0 * ca2 * v2 * w2
            + 10.0 * ca2 * v3 * w2
            - 8.0 * ca2 * v4 * w2
            - 2.0 * ca2 * v3 * w3
            + 2.0 * ca2 * v4 * w3))
        / ((1.0 - v) * v * w * (1.0 - v * w))
        - (2.0
            * cf
            * lms
            * (2.0 - 2.0 * ca2 - 6.0 * v + 10.0 * ca2 * v + 8.0 * v2 - 20.0 * ca2 * v2
                + 16.0 * ca2 * v3
                - 8.0 * ca2 * v4
                - w
                + ca2 * w
                + 2.0 * v * w
                - 4.0 * ca2 * v * w
                - 4.0 * v2 * w
                + 10.0 * ca2 * v2 * w
                - 8.0 * ca2 * v3 * w
                + 4.0 * ca2 * v4 * w
                - 2.0 * v3 * w2
                + 2.0 * ca2 * v3 * w2
                + 4.0 * v4 * w2
                - 4.0 * ca2 * v4 * w2
                - 2.0 * v4 * w3
                + 2.0 * ca2 * v4 * w3))
            / ((1.0 - v) * v2 * w)
        - (2.0
            * cf
            * (3.0 - 3.0 * ca2 - 13.0 * v + 17.0 * ca2 * v + 27.0 * v2
                - 55.0 * ca2 * v2
                - 27.0 * v3
                + 103.0 * ca2 * v3
                + 10.0 * v4
                - 110.0 * ca2 * v4
                + 64.0 * ca2 * v5
                - 16.0 * ca2 * v6
                - 4.0 * w
                + 4.0 * ca2 * w
                + 17.0 * v * w
                - 25.0 * ca2 * v * w
                - 40.0 * v2 * w
                + 84.0 * ca2 * v2 * w
                + 59.0 * v3 * w
                - 191.0 * ca2 * v3 * w
                - 34.0 * v4 * w
                + 250.0 * ca2 * v4 * w
                + 2.0 * v5 * w
                - 174.0 * ca2 * v5 * w
                + 52.0 * ca2 * v6 * w
                - 8.0 * v * w2
                + 8.0 * ca2 * v * w2
                + 17.0 * v2 * w2
                - 33.0 * ca2 * v2 * w2
                - 31.0 * v3 * w2
                + 99.0 * ca2 * v3 * w2
                + 39.0 * v4 * w2
                - 187.0 * ca2 * v4 * w2
                - 19.0 * v5 * w2
                + 175.0 * ca2 * v5 * w2
                + 8.0 * v6 * w2
                - 68.0 * ca2 * v6 * w2
                - 4.0 * v2 * w3
                + 4.0 * ca2 * v2 * w3
                + 3.0 * v3 * w3
                - 11.0 * ca2 * v3 * w3
                - 14.0 * v4 * w3
                + 46.0 * ca2 * v4 * w3
                + 31.0 * v5 * w3
                - 83.0 * ca2 * v5 * w3
                - 22.0 * v6 * w3
                + 50.0 * ca2 * v6 * w3
                - 14.0 * v5 * w4
                + 18.0 * ca2 * v5 * w4
                + 20.0 * v6 * w4
                - 24.0 * ca2 * v6 * w4
                - 6.0 * v6 * w5
                + 6.0 * ca2 * v6 * w5))
            / ((1.0 - v) * v2 * w * (1.0 - v + v * w).powi(2));

    let part3 = (2.0
        * cf
        * l1w
        * (2.0 - 2.0 * ca2 - 14.0 * v + 24.0 * ca2 * v + 42.0 * v2 - 88.0 * ca2 * v2 - 58.0 * v3
            + 152.0 * ca2 * v3
            + 36.0 * v4
            - 142.0 * ca2 * v4
            - 8.0 * v5
            + 72.0 * ca2 * v5
            - 16.0 * ca2 * v6
            - w
            + ca2 * w
            + 6.0 * v * w
            - 8.0 * ca2 * v * w
            - 15.0 * v2 * w
            + 47.0 * ca2 * v2 * w
            - 106.0 * ca2 * v3 * w
            + 58.0 * v4 * w
            + 90.0 * ca2 * v4 * w
            - 72.0 * v5 * w
            - 8.0 * ca2 * v5 * w
            + 24.0 * v6 * w
            - 32.0 * ca2 * v6 * w
            + 16.0 * ca2 * v7 * w
            - v * w2
            + ca2 * v * w2
            - 2.0 * ca2 * v2 * w2
            + 27.0 * v3 * w2
            - ca2 * v3 * w2
            - 112.0 * v4 * w2
            + 52.0 * ca2 * v4 * w2
            + 142.0 * v5 * w2
            - 124.0 * ca2 * v5 * w2
            - 52.0 * v6 * w2
            + 110.0 * ca2 * v6 * w2
            - 40.0 * ca2 * v7 * w2
            + v2 * w3
            - ca2 * v2 * w3
            - 6.0 * v3 * w3
            + 8.0 * ca2 * v3 * w3
            + 39.0 * v4 * w3
            - 39.0 * ca2 * v4 * w3
            - 80.0 * v5 * w3
            + 90.0 * ca2 * v5 * w3
            + 46.0 * v6 * w3
            - 90.0 * ca2 * v6 * w3
            - 8.0 * v7 * w3
            + 40.0 * ca2 * v7 * w3
            + v3 * w4
            - ca2 * v3 * w4
            - 2.0 * v4 * w4
            + 4.0 * ca2 * v4 * w4
            + 15.0 * v5 * w4
            - 19.0 * ca2 * v5 * w4
            - 26.0 * v6 * w4
            + 36.0 * ca2 * v6 * w4
            + 20.0 * v7 * w4
            - 28.0 * ca2 * v7 * w4
            + 8.0 * v6 * w5
            - 8.0 * ca2 * v6 * w5
            - 16.0 * v7 * w5
            + 16.0 * ca2 * v7 * w5
            + 4.0 * v7 * w6
            - 4.0 * ca2 * v7 * w6))
        / (t1)
        + (2.0
            * cf
            * lv
            * (2.0 - 2.0 * ca2 - 10.0 * v + 26.0 * ca2 * v + 22.0 * v2
                - 94.0 * ca2 * v2
                - 22.0 * v3
                + 158.0 * ca2 * v3
                + 8.0 * v4
                - 144.0 * ca2 * v4
                + 72.0 * ca2 * v5
                - 16.0 * ca2 * v6
                - w
                + ca2 * w
                + 6.0 * v * w
                - 8.0 * ca2 * v * w
                + v2 * w
                + 51.0 * ca2 * v2 * w
                - 48.0 * v3 * w
                - 122.0 * ca2 * v3 * w
                + 98.0 * v4 * w
                + 118.0 * ca2 * v4 * w
                - 72.0 * v5 * w
                - 32.0 * ca2 * v5 * w
                + 16.0 * v6 * w
                - 24.0 * ca2 * v6 * w
                + 16.0 * ca2 * v7 * w
                - v * w2
                + ca2 * v * w2
                - 2.0 * ca2 * v2 * w2
                + 43.0 * v3 * w2
                + 3.0 * ca2 * v3 * w2
                - 128.0 * v4 * w2
                + 36.0 * ca2 * v4 * w2
                + 122.0 * v5 * w2
                - 110.0 * ca2 * v5 * w2
                - 32.0 * v6 * w2
                + 116.0 * ca2 * v6 * w2
                - 48.0 * ca2 * v7 * w2
                + v2 * w3
                - ca2 * v2 * w3
                - 6.0 * v3 * w3
                + 8.0 * ca2 * v3 * w3
                + 39.0 * v4 * w3
                - 39.0 * ca2 * v4 * w3
                - 64.0 * v5 * w3
                + 102.0 * ca2 * v5 * w3
                + 30.0 * v6 * w3
                - 122.0 * ca2 * v6 * w3
                - 8.0 * v7 * w3
                + 60.0 * ca2 * v7 * w3
                + v3 * w4
                - ca2 * v3 * w4
                - 2.0 * v4 * w4
                + 4.0 * ca2 * v4 * w4
                + 11.0 * v5 * w4
                - 25.0 * ca2 * v5 * w4
                - 22.0 * v6 * w4
                + 58.0 * ca2 * v6 * w4
                + 20.0 * v7 * w4
                - 44.0 * ca2 * v7 * w4
                + 8.0 * v6 * w5
                - 12.0 * ca2 * v6 * w5
                - 16.0 * v7 * w5
                + 20.0 * ca2 * v7 * w5
                + 4.0 * v7 * w6
                - 4.0 * ca2 * v7 * w6))
            / (t1);

    part1 + part2 + part3
}

/// `STRUV10(W,V,X3,S)`.
#[must_use]
pub fn qg_seapair_same_flavor_frag(
    w: f64,
    v: f64,
    _x3: f64,
    _s: f64,
    ctx: &MeContext,
    pre: &Precalc,
) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7);
    let (w2, w3, w4, w5, w6) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6);
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let (l1v, lv, l1w, lw, lvw, l1vw, lms, lmss) = (
        pre.l1v, pre.lv, pre.l1w, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.lmss,
    );

    let t1 = ca * (1.0 - v) * v2 * w * (1.0 - v * w) * (1.0 - v + v * w);

    let part1 = -(2.0
        * cf
        * lmss
        * (2.0 - 2.0 * v + v * w)
        * (1.0 - 2.0 * v + 2.0 * v2 + 2.0 * v * w - 4.0 * v2 * w + 2.0 * v2 * w2)
        * (2.0 * ca2 - 4.0 * ca2 * v + 2.0 * ca2 * v2 + 2.0 * ca2 * v * w
            - 2.0 * ca2 * v2 * w
            - v2 * w2
            + ca2 * v2 * w2))
        / ((1.0 - v) * v * w * (1.0 - v + v * w).powi(2))
        - (4.0
            * cf
            * l1vw
            * (1.0 - w)
            * (2.0 - ca2 - 8.0 * v + 2.0 * ca2 * v + 10.0 * v2 - ca2 * v2 - 4.0 * v3
                + 7.0 * v * w
                - 2.0 * ca2 * v * w
                - 19.0 * v2 * w
                + 5.0 * ca2 * v2 * w
                + 12.0 * v3 * w
                - 3.0 * ca2 * v3 * w
                + 8.0 * v2 * w2
                - 3.0 * ca2 * v2 * w2
                - 10.0 * v3 * w2
                + 4.0 * ca2 * v3 * w2
                + 2.0 * v3 * w3
                - ca2 * v3 * w3))
            / (v * w * (1.0 - v + v * w).powi(2))
        + (8.0
            * cf
            * lvw
            * (1.0 - w)
            * (ca3 - 2.0 * ca * cf - 2.0 * v + ca * v + 2.0 * ca2 * v - 2.0 * ca3 * v
                + ca3 * v * w
                + 2.0 * v2 * w
                - ca * v2 * w
                - 2.0 * ca2 * v2 * w
                - v2 * w2
                - ca * v2 * w2
                + ca2 * v2 * w2
                + ca3 * v2 * w2
                + 2.0 * ca * v3 * w2
                - 2.0 * ca3 * v3 * w2
                - ca * v3 * w3
                + ca3 * v3 * w3))
            / (ca * (1.0 - v) * v * w)
        - (2.0
            * cf
            * lms
            * (2.0 * ca - 2.0 * ca3 + 2.0 * v - 6.0 * ca * v - 2.0 * ca2 * v + 10.0 * ca3 * v
                - 4.0 * v2
                + 8.0 * ca * v2
                + 4.0 * ca2 * v2
                - 20.0 * ca3 * v2
                + 16.0 * ca3 * v3
                - 8.0 * ca3 * v4
                - ca * w
                + ca3 * w
                + 2.0 * ca * v * w
                - 4.0 * ca3 * v * w
                - 4.0 * ca * v2 * w
                + 10.0 * ca3 * v2 * w
                + 4.0 * v3 * w
                - 4.0 * ca2 * v3 * w
                - 8.0 * ca3 * v3 * w
                + 4.0 * ca3 * v4 * w
                - 2.0 * v3 * w2
                - 2.0 * ca * v3 * w2
                + 2.0 * ca2 * v3 * w2
                + 2.0 * ca3 * v3 * w2
                + 4.0 * ca * v4 * w2
                - 4.0 * ca3 * v4 * w2
                - 2.0 * ca * v4 * w3
                + 2.0 * ca3 * v4 * w3))
            / (ca * (1.0 - v) * v2 * w);

    let part2 = -(4.0
        * cf
        * lw
        * (2.0 * ca * cf + 3.0 * v - 3.0 * ca2 * v - 4.0 * ca3 * v - 2.0 * v2 - 2.0 * ca * v2
            + 2.0 * ca2 * v2
            + 18.0 * ca3 * v2
            + 2.0 * ca * v3
            - 30.0 * ca3 * v3
            + 24.0 * ca3 * v4
            - 8.0 * ca3 * v5
            - v * w
            - ca2 * v * w
            + 2.0 * v2 * w
            - 3.0 * ca * v2 * w
            + 8.0 * ca2 * v2 * w
            - 5.0 * ca3 * v2 * w
            - 4.0 * v3 * w
            + 7.0 * ca * v3 * w
            - 12.0 * ca2 * v3 * w
            + 13.0 * ca3 * v3 * w
            + 4.0 * v4 * w
            - 6.0 * ca * v4 * w
            + 4.0 * ca2 * v4 * w
            - 6.0 * ca3 * v4 * w
            - 8.0 * ca3 * v5 * w
            + 8.0 * ca3 * v6 * w
            + v2 * w2
            - 3.0 * ca2 * v2 * w2
            - 2.0 * v3 * w2
            - 4.0 * ca * v3 * w2
            + 4.0 * ca2 * v3 * w2
            + 2.0 * ca3 * v3 * w2
            + 2.0 * v4 * w2
            + 7.0 * ca * v4 * w2
            + 6.0 * ca2 * v4 * w2
            - 15.0 * ca3 * v4 * w2
            - 4.0 * v5 * w2
            - 4.0 * ca2 * v5 * w2
            + 26.0 * ca3 * v5 * w2
            - 16.0 * ca3 * v6 * w2
            + v3 * w3
            + ca2 * v3 * w3
            - 2.0 * v4 * w3
            - ca * v4 * w3
            - 8.0 * ca2 * v4 * w3
            + 5.0 * ca3 * v4 * w3
            + 4.0 * v5 * w3
            + 4.0 * ca2 * v5 * w3
            - 14.0 * ca3 * v5 * w3
            + 10.0 * ca3 * v6 * w3
            + 2.0 * ca2 * v4 * w4
            - v5 * w4
            - ca2 * v5 * w4
            + 2.0 * ca3 * v5 * w4
            - 2.0 * ca3 * v6 * w4))
        / (t1)
        + (4.0
            * cf
            * l1v
            * (1.0 + ca2 - 5.0 * v + 2.0 * ca * v - 3.0 * ca2 * v - 3.0 * ca3 * v + 8.0 * v2
                - 8.0 * ca * v2
                + 2.0 * ca2 * v2
                + 10.0 * ca3 * v2
                - 4.0 * v3
                + 10.0 * ca * v3
                - 11.0 * ca3 * v3
                - 4.0 * ca * v4
                + 4.0 * ca3 * v4
                + v * w
                + ca2 * v * w
                - 2.0 * ca * v2 * w
                - ca3 * v2 * w
                - 8.0 * v3 * w
                + 14.0 * ca * v3 * w
                - 2.0 * ca2 * v3 * w
                - 5.0 * ca3 * v3 * w
                + 8.0 * v4 * w
                - 24.0 * ca * v4 * w
                + 14.0 * ca3 * v4 * w
                + 12.0 * ca * v5 * w
                - 8.0 * ca3 * v5 * w
                - v2 * w2
                - ca2 * v2 * w2
                + 6.0 * v3 * w2
                - 10.0 * ca * v3 * w2
                + 2.0 * ca2 * v3 * w2
                + 5.0 * ca3 * v3 * w2
                - 4.0 * v4 * w2
                + 28.0 * ca * v4 * w2
                + 2.0 * ca2 * v4 * w2
                - 12.0 * ca3 * v4 * w2
                - 4.0 * v5 * w2
                - 18.0 * ca * v5 * w2
                + 3.0 * ca3 * v5 * w2
                + 4.0 * ca3 * v6 * w2
                - v3 * w3
                - ca2 * v3 * w3
                - 6.0 * ca * v4 * w3
                + ca3 * v4 * w3
                + 4.0 * v5 * w3
                + 6.0 * ca * v5 * w3
                - 2.0 * ca2 * v5 * w3
                + 5.0 * ca3 * v5 * w3
                - 6.0 * ca3 * v6 * w3
                - v5 * w4
                + ca2 * v5 * w4
                - 2.0 * ca3 * v5 * w4
                + 2.0 * ca3 * v6 * w4))
            / (t1);

    let part3 = -(2.0
        * cf
        * (3.0 * ca - 3.0 * ca3 - 13.0 * ca * v + 17.0 * ca3 * v - 4.0 * v2
            + 27.0 * ca * v2
            + 4.0 * ca2 * v2
            - 55.0 * ca3 * v2
            + 8.0 * v3
            - 27.0 * ca * v3
            - 8.0 * ca2 * v3
            + 103.0 * ca3 * v3
            - 4.0 * v4
            + 10.0 * ca * v4
            + 4.0 * ca2 * v4
            - 110.0 * ca3 * v4
            + 64.0 * ca3 * v5
            - 16.0 * ca3 * v6
            - 4.0 * ca * w
            + 4.0 * ca3 * w
            + 17.0 * ca * v * w
            - 25.0 * ca3 * v * w
            + 4.0 * v2 * w
            - 40.0 * ca * v2 * w
            - 4.0 * ca2 * v2 * w
            + 84.0 * ca3 * v2 * w
            - 12.0 * v3 * w
            + 59.0 * ca * v3 * w
            + 12.0 * ca2 * v3 * w
            - 191.0 * ca3 * v3 * w
            + 4.0 * v4 * w
            - 34.0 * ca * v4 * w
            - 4.0 * ca2 * v4 * w
            + 250.0 * ca3 * v4 * w
            + 4.0 * v5 * w
            + 2.0 * ca * v5 * w
            - 4.0 * ca2 * v5 * w
            - 174.0 * ca3 * v5 * w
            + 52.0 * ca3 * v6 * w
            - 8.0 * ca * v * w2
            + 8.0 * ca3 * v * w2
            + 17.0 * ca * v2 * w2
            - 33.0 * ca3 * v2 * w2
            + 4.0 * v3 * w2
            - 31.0 * ca * v3 * w2
            - 4.0 * ca2 * v3 * w2
            + 99.0 * ca3 * v3 * w2
            + 4.0 * v4 * w2
            + 39.0 * ca * v4 * w2
            - 4.0 * ca2 * v4 * w2
            - 187.0 * ca3 * v4 * w2
            - 12.0 * v5 * w2
            - 19.0 * ca * v5 * w2
            + 12.0 * ca2 * v5 * w2
            + 175.0 * ca3 * v5 * w2
            + 8.0 * ca * v6 * w2
            - 68.0 * ca3 * v6 * w2
            - 4.0 * ca * v2 * w3
            + 4.0 * ca3 * v2 * w3
            + 3.0 * ca * v3 * w3
            - 11.0 * ca3 * v3 * w3
            - 4.0 * v4 * w3
            - 14.0 * ca * v4 * w3
            + 4.0 * ca2 * v4 * w3
            + 46.0 * ca3 * v4 * w3
            + 12.0 * v5 * w3
            + 31.0 * ca * v5 * w3
            - 12.0 * ca2 * v5 * w3
            - 83.0 * ca3 * v5 * w3
            - 22.0 * ca * v6 * w3
            + 50.0 * ca3 * v6 * w3
            - 4.0 * v5 * w4
            - 14.0 * ca * v5 * w4
            + 4.0 * ca2 * v5 * w4
            + 18.0 * ca3 * v5 * w4
            + 20.0 * ca * v6 * w4
            - 24.0 * ca3 * v6 * w4
            - 6.0 * ca * v6 * w5
            + 6.0 * ca3 * v6 * w5))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v + v * w).powi(2));

    let part4 = (2.0
        * cf
        * lv
        * (2.0 * ca - 2.0 * ca3 + 2.0 * v - 10.0 * ca * v - 6.0 * ca2 * v + 26.0 * ca3 * v
            - 8.0 * v2
            + 22.0 * ca * v2
            + 24.0 * ca2 * v2
            - 94.0 * ca3 * v2
            + 10.0 * v3
            - 22.0 * ca * v3
            - 30.0 * ca2 * v3
            + 158.0 * ca3 * v3
            - 4.0 * v4
            + 8.0 * ca * v4
            + 12.0 * ca2 * v4
            - 144.0 * ca3 * v4
            + 72.0 * ca3 * v5
            - 16.0 * ca3 * v6
            - ca * w
            + ca3 * w
            + 6.0 * ca * v * w
            - 8.0 * ca3 * v * w
            - 2.0 * v2 * w
            + ca * v2 * w
            - 6.0 * ca2 * v2 * w
            + 51.0 * ca3 * v2 * w
            + 16.0 * v3 * w
            - 48.0 * ca * v3 * w
            + 4.0 * ca2 * v3 * w
            - 122.0 * ca3 * v3 * w
            - 30.0 * v4 * w
            + 98.0 * ca * v4 * w
            + 18.0 * ca2 * v4 * w
            + 118.0 * ca3 * v4 * w
            + 16.0 * v5 * w
            - 72.0 * ca * v5 * w
            - 16.0 * ca2 * v5 * w
            - 32.0 * ca3 * v5 * w
            + 16.0 * ca * v6 * w
            - 24.0 * ca3 * v6 * w
            + 16.0 * ca3 * v7 * w
            - ca * v * w2
            + ca3 * v * w2
            - 2.0 * ca3 * v2 * w2
            - 8.0 * v3 * w2
            + 43.0 * ca * v3 * w2
            + 8.0 * ca2 * v3 * w2
            + 3.0 * ca3 * v3 * w2
            + 20.0 * v4 * w2
            - 128.0 * ca * v4 * w2
            - 32.0 * ca2 * v4 * w2
            + 36.0 * ca3 * v4 * w2
            - 2.0 * v5 * w2
            + 122.0 * ca * v5 * w2
            + 22.0 * ca2 * v5 * w2
            - 110.0 * ca3 * v5 * w2
            - 12.0 * v6 * w2
            - 32.0 * ca * v6 * w2
            + 4.0 * ca2 * v6 * w2
            + 116.0 * ca3 * v6 * w2
            - 48.0 * ca3 * v7 * w2
            + ca * v2 * w3
            - ca3 * v2 * w3
            - 6.0 * ca * v3 * w3
            + 8.0 * ca3 * v3 * w3
            + 39.0 * ca * v4 * w3
            + 8.0 * ca2 * v4 * w3
            - 39.0 * ca3 * v4 * w3
            - 16.0 * v5 * w3
            - 64.0 * ca * v5 * w3
            - 4.0 * ca2 * v5 * w3
            + 102.0 * ca3 * v5 * w3
            + 22.0 * v6 * w3
            + 30.0 * ca * v6 * w3
            - 10.0 * ca2 * v6 * w3
            - 122.0 * ca3 * v6 * w3
            - 8.0 * ca * v7 * w3
            + 60.0 * ca3 * v7 * w3
            + ca * v3 * w4
            - ca3 * v3 * w4
            - 2.0 * ca * v4 * w4
            + 4.0 * ca3 * v4 * w4
            + 6.0 * v5 * w4
            + 11.0 * ca * v5 * w4
            - 2.0 * ca2 * v5 * w4
            - 25.0 * ca3 * v5 * w4
            - 12.0 * v6 * w4
            - 22.0 * ca * v6 * w4
            + 8.0 * ca2 * v6 * w4
            + 58.0 * ca3 * v6 * w4
            + 20.0 * ca * v7 * w4
            - 44.0 * ca3 * v7 * w4
            + 2.0 * v6 * w5
            + 8.0 * ca * v6 * w5
            - 2.0 * ca2 * v6 * w5
            - 12.0 * ca3 * v6 * w5
            - 16.0 * ca * v7 * w5
            + 20.0 * ca3 * v7 * w5
            + 4.0 * ca * v7 * w6
            - 4.0 * ca3 * v7 * w6))
        / (t1.powi(2));

    let part5 = -(2.0
        * cf
        * l1w
        * (2.0 - 2.0 * ca - 2.0 * ca2 + 2.0 * ca3 - 10.0 * v + 14.0 * ca * v + 14.0 * ca2 * v
            - 24.0 * ca3 * v
            + 18.0 * v2
            - 42.0 * ca * v2
            - 34.0 * ca2 * v2
            + 88.0 * ca3 * v2
            - 14.0 * v3
            + 58.0 * ca * v3
            + 34.0 * ca2 * v3
            - 152.0 * ca3 * v3
            + 4.0 * v4
            - 36.0 * ca * v4
            - 12.0 * ca2 * v4
            + 142.0 * ca3 * v4
            + 8.0 * ca * v5
            - 72.0 * ca3 * v5
            + 16.0 * ca3 * v6
            + ca * w
            - ca3 * w
            + 4.0 * v * w
            - 6.0 * ca * v * w
            + 8.0 * ca3 * v * w
            - 10.0 * v2 * w
            + 15.0 * ca * v2 * w
            - 6.0 * ca2 * v2 * w
            - 47.0 * ca3 * v2 * w
            + 32.0 * ca2 * v3 * w
            + 106.0 * ca3 * v3 * w
            + 14.0 * v4 * w
            - 58.0 * ca * v4 * w
            - 50.0 * ca2 * v4 * w
            - 90.0 * ca3 * v4 * w
            - 8.0 * v5 * w
            + 72.0 * ca * v5 * w
            + 24.0 * ca2 * v5 * w
            + 8.0 * ca3 * v5 * w
            - 24.0 * ca * v6 * w
            + 32.0 * ca3 * v6 * w
            - 16.0 * ca3 * v7 * w
            + ca * v * w2
            - ca3 * v * w2
            + 8.0 * ca2 * v2 * w2
            + 2.0 * ca3 * v2 * w2
            + 10.0 * v3 * w2
            - 27.0 * ca * v3 * w2
            - 38.0 * ca2 * v3 * w2
            + ca3 * v3 * w2
            - 20.0 * v4 * w2
            + 112.0 * ca * v4 * w2
            + 52.0 * ca2 * v4 * w2
            - 52.0 * ca3 * v4 * w2
            + 6.0 * v5 * w2
            - 142.0 * ca * v5 * w2
            - 10.0 * ca2 * v5 * w2
            + 124.0 * ca3 * v5 * w2
            + 4.0 * v6 * w2
            + 52.0 * ca * v6 * w2
            - 12.0 * ca2 * v6 * w2
            - 110.0 * ca3 * v6 * w2
            + 40.0 * ca3 * v7 * w2
            - ca * v2 * w3
            + ca3 * v2 * w3
            - 4.0 * v3 * w3
            + 6.0 * ca * v3 * w3
            + 4.0 * ca2 * v3 * w3
            - 8.0 * ca3 * v3 * w3
            + 10.0 * v4 * w3
            - 39.0 * ca * v4 * w3
            + 2.0 * ca2 * v4 * w3
            + 39.0 * ca3 * v4 * w3
            + 80.0 * ca * v5 * w3
            - 32.0 * ca2 * v5 * w3
            - 90.0 * ca3 * v5 * w3
            - 6.0 * v6 * w3
            - 46.0 * ca * v6 * w3
            + 26.0 * ca2 * v6 * w3
            + 90.0 * ca3 * v6 * w3
            + 8.0 * ca * v7 * w3
            - 40.0 * ca3 * v7 * w3
            - ca * v3 * w4
            + ca3 * v3 * w4
            - 2.0 * v4 * w4
            + 2.0 * ca * v4 * w4
            - 6.0 * ca2 * v4 * w4
            - 4.0 * ca3 * v4 * w4
            - 15.0 * ca * v5 * w4
            + 24.0 * ca2 * v5 * w4
            + 19.0 * ca3 * v5 * w4
            + 2.0 * v6 * w4
            + 26.0 * ca * v6 * w4
            - 18.0 * ca2 * v6 * w4
            - 36.0 * ca3 * v6 * w4
            - 20.0 * ca * v7 * w4
            + 28.0 * ca3 * v7 * w4
            - 4.0 * ca2 * v5 * w5
            - 8.0 * ca * v6 * w5
            + 4.0 * ca2 * v6 * w5
            + 8.0 * ca3 * v6 * w5
            + 16.0 * ca * v7 * w5
            - 16.0 * ca3 * v7 * w5
            - 4.0 * ca * v7 * w6
            + 4.0 * ca3 * v7 * w6))
        / (t1.powi(2));

    part1 + part2 + part3 + part4 + part5
}

/// `STRUV11(W,V,X3,S)`.
#[must_use]
pub fn qqbar_elastic_quark_frag(
    w: f64,
    v: f64,
    _x3: f64,
    _s: f64,
    ctx: &MeContext,
    pre: &Precalc,
) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7, v8) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8);
    let (w2, w3, w4, w5, w6, w7) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7);
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let ca2cf = ca2 * cf;
    let (l1v, lvw, l1vw, lms, lmss) = (pre.l1v, pre.lvw, pre.l1vw, pre.lms, pre.lmss);

    let t1 = (1.0 - v) * v * w * (1.0 - v * w).powi(2);

    let part1 = -(4.0
        * cf
        * lvw
        * (ca - ca3 + ca * v - ca3 * v - 2.0 * w - 5.0 * ca * w
            + ca3 * w
            + 2.0 * v * w
            + 3.0 * ca * v * w
            - 2.0 * ca3 * v * w
            - v2 * w
            - ca * v2 * w
            + 3.0 * ca2 * v2 * w
            + ca3 * v2 * w
            + 8.0 * ca * v * w2
            - 9.0 * ca3 * v * w2
            - 9.0 * ca * v2 * w2
            - ca2 * v2 * w2
            + 7.0 * ca3 * v2 * w2
            + 2.0 * ca * v3 * w2
            - 2.0 * ca3 * v3 * w2
            - 2.0 * ca * v2 * w3
            + 4.0 * ca2 * v2 * w3
            + 2.0 * ca3 * v2 * w3
            + 2.0 * ca * v3 * w4
            - 2.0 * ca3 * v3 * w4))
        / (ca * (1.0 - v) * v * w)
        + (4.0
            * cf
            * l1v
            * (v - ca2 * v - v2 + ca2 * v2 + 16.0 * ca * w
                - 6.0 * v * w
                - 40.0 * ca * v * w
                - 2.0 * ca2 * v * w
                + 6.0 * ca3 * v * w
                + 8.0 * v2 * w
                + 38.0 * ca * v2 * w
                + 2.0 * ca2 * v2 * w
                - 10.0 * ca3 * v2 * w
                - v3 * w
                - 14.0 * ca * v3 * w
                - ca2 * v3 * w
                + 4.0 * ca3 * v3 * w
                - 2.0 * v * w2
                - 14.0 * ca * v * w2
                - 6.0 * ca2 * v * w2
                + 4.0 * ca3 * v * w2
                + 2.0 * v2 * w2
                + 36.0 * ca * v2 * w2
                + 6.0 * ca2 * v2 * w2
                - 6.0 * ca3 * v2 * w2
                - 4.0 * v3 * w2
                - 34.0 * ca * v3 * w2
                - 2.0 * ca2 * v3 * w2
                + 9.0 * ca3 * v3 * w2
                + 14.0 * ca * v4 * w2
                - 4.0 * ca3 * v4 * w2
                + 2.0 * v2 * w3
                + 8.0 * ca * v2 * w3
                - 2.0 * ca3 * v2 * w3
                - v3 * w3
                - 10.0 * ca * v3 * w3
                - 2.0 * ca2 * v3 * w3
                + ca3 * v3 * w3
                + 2.0 * v3 * w4
                - 2.0 * ca * v3 * w4
                + 2.0 * ca2 * v3 * w4
                + 2.0 * ca * v4 * w4))
            / (ca * (1.0 - v) * w * (1.0 - v * w) * (1.0 - v + v * w))
        - (2.0
            * cf
            * lms
            * (2.0 * ca * cf + 10.0 * ca * cf * v - 4.0 * cf * v2 - 8.0 * ca * cf * v2
                + 4.0 * ca * cf * v3
                + 2.0 * v * w
                - 26.0 * ca * cf * v * w
                + 2.0 * v2 * w
                + 12.0 * cf * v2 * w
                + 10.0 * ca * cf * v2 * w
                + 8.0 * cf * v3 * w
                + 4.0 * ca * cf * v3 * w
                - 8.0 * ca * cf * v4 * w
                - 3.0 * v2 * w2
                + ca2 * v2 * w2
                + 8.0 * cf * v2 * w2
                + 44.0 * ca * cf * v2 * w2
                + v3 * w2
                + ca2 * v3 * w2
                - 24.0 * cf * v3 * w2
                - 40.0 * ca * cf * v3 * w2
                - 4.0 * cf * v4 * w2
                + 16.0 * ca * cf * v4 * w2
                + 4.0 * ca * cf * v5 * w2
                - 3.0 * v3 * w3
                - ca2 * v3 * w3
                - 8.0 * cf * v3 * w3
                - 40.0 * ca * cf * v3 * w3
                - v4 * w3
                + ca2 * v4 * w3
                + 8.0 * cf * v4 * w3
                + 32.0 * ca * cf * v4 * w3
                - 12.0 * ca * cf * v5 * w3
                + 2.0 * v4 * w4
                - 2.0 * ca2 * v4 * w4
                + 4.0 * cf * v4 * w4
                + 16.0 * ca * cf * v4 * w4
                - 4.0 * ca * cf * v5 * w4
                - 4.0 * ca * cf * v5 * w5))
            / (t1);

    let part2 = -(4.0
        * cf
        * l1vw
        * (2.0 + 4.0 * ca - 4.0 * ca3 - 8.0 * v - 17.0 * ca * v
            + 18.0 * ca3 * v
            + 13.0 * v2
            + 30.0 * ca * v2
            - 3.0 * ca2 * v2
            - 36.0 * ca3 * v2
            - 11.0 * v3
            - 28.0 * ca * v3
            + 9.0 * ca2 * v3
            + 40.0 * ca3 * v3
            + 5.0 * v4
            + 14.0 * ca * v4
            - 9.0 * ca2 * v4
            - 24.0 * ca3 * v4
            - v5
            - 3.0 * ca * v5
            + 3.0 * ca2 * v5
            + 6.0 * ca3 * v5
            + 6.0 * v * w
            + 5.0 * ca * v * w
            - 2.0 * ca3 * v * w
            - 16.0 * ca2cf * v * w
            - 18.0 * v2 * w
            - 13.0 * ca * v2 * w
            + ca2 * v2 * w
            + 6.0 * ca3 * v2 * w
            + 48.0 * ca2cf * v2 * w
            + 21.0 * v3 * w
            + 13.0 * ca * v3 * w
            - 12.0 * ca2 * v3 * w
            - 10.0 * ca3 * v3 * w
            - 56.0 * ca2cf * v3 * w
            - 12.0 * v4 * w
            - 7.0 * ca * v4 * w
            + 21.0 * ca2 * v4 * w
            + 10.0 * ca3 * v4 * w
            + 32.0 * ca2cf * v4 * w
            + 3.0 * v5 * w
            + 2.0 * ca * v5 * w
            - 10.0 * ca2 * v5 * w
            - 4.0 * ca3 * v5 * w
            - 8.0 * ca2cf * v5 * w
            + 6.0 * v2 * w2
            - 9.0 * ca * v2 * w2
            - 4.0 * ca2 * v2 * w2
            + 12.0 * ca3 * v2 * w2
            + 8.0 * ca * cf * v2 * w2
            - 32.0 * ca2cf * v2 * w2
            - 12.0 * v3 * w2
            + 34.0 * ca * v3 * w2
            + 15.0 * ca2 * v3 * w2
            - 38.0 * ca3 * v3 * w2
            - 16.0 * ca * cf * v3 * w2
            + 80.0 * ca2cf * v3 * w2
            + 9.0 * v4 * w2
            - 39.0 * ca * v4 * w2
            - 27.0 * ca2 * v4 * w2
            + 36.0 * ca3 * v4 * w2
            + 12.0 * ca * cf * v4 * w2
            - 72.0 * ca2cf * v4 * w2
            - 3.0 * v5 * w2
            + 14.0 * ca * v5 * w2
            + 16.0 * ca2 * v5 * w2
            - 10.0 * ca3 * v5 * w2
            - 4.0 * ca * cf * v5 * w2
            + 24.0 * ca2cf * v5 * w2
            + 2.0 * v3 * w3
            - 19.0 * ca * v3 * w3
            - 12.0 * ca2 * v3 * w3
            + 16.0 * ca3 * v3 * w3
            + 16.0 * ca * cf * v3 * w3
            - 32.0 * ca2cf * v3 * w3
            - 2.0 * v4 * w3
            + 43.0 * ca * v4 * w3
            + 27.0 * ca2 * v4 * w3
            - 26.0 * ca3 * v4 * w3
            - 24.0 * ca * cf * v4 * w3
            + 56.0 * ca2cf * v4 * w3
            + v5 * w3
            - 24.0 * ca * v5 * w3
            - 18.0 * ca2 * v5 * w3
            + 10.0 * ca3 * v5 * w3
            + 12.0 * ca * cf * v5 * w3
            - 28.0 * ca2cf * v5 * w3
            - 11.0 * ca * v4 * w4
            - 12.0 * ca2 * v4 * w4
            + 4.0 * ca3 * v4 * w4
            + 12.0 * ca * cf * v4 * w4
            - 16.0 * ca2cf * v4 * w4
            + 13.0 * ca * v5 * w4
            + 13.0 * ca2 * v5 * w4
            - 4.0 * ca3 * v5 * w4
            - 12.0 * ca * cf * v5 * w4
            + 16.0 * ca2cf * v5 * w4
            - 2.0 * ca * v5 * w5
            - 4.0 * ca2 * v5 * w5
            + 2.0 * ca3 * v5 * w5
            + 4.0 * ca * cf * v5 * w5
            - 4.0 * ca2cf * v5 * w5))
        / (ca * (1.0 - v) * v * (1.0 - v + v * w).powi(3))
        + (2.0
            * cf
            * lmss
            * (2.0 * ca * cf + 2.0 * v - 2.0 * ca2 * v - 8.0 * ca * cf * v - 9.0 * v2
                + 9.0 * ca2 * v2
                + 14.0 * ca * cf * v2
                + 18.0 * v3
                - 18.0 * ca2 * v3
                - 12.0 * ca * cf * v3
                - 19.0 * v4
                + 19.0 * ca2 * v4
                + 4.0 * ca * cf * v4
                + 10.0 * v5
                - 10.0 * ca2 * v5
                - 2.0 * v6
                + 2.0 * ca2 * v6
                - 2.0 * v * w
                + 20.0 * ca * cf * v * w
                + 14.0 * v2 * w
                - 4.0 * ca2 * v2 * w
                - 8.0 * cf * v2 * w
                - 80.0 * ca * cf * v2 * w
                - 40.0 * v3 * w
                + 18.0 * ca2 * v3 * w
                + 24.0 * cf * v3 * w
                + 124.0 * ca * cf * v3 * w
                + 54.0 * v4 * w
                - 28.0 * ca2 * v4 * w
                - 24.0 * cf * v4 * w
                - 96.0 * ca * cf * v4 * w
                - 34.0 * v5 * w
                + 18.0 * ca2 * v5 * w
                + 8.0 * cf * v5 * w
                + 40.0 * ca * cf * v5 * w
                + 8.0 * v6 * w
                - 4.0 * ca2 * v6 * w
                - 8.0 * ca * cf * v6 * w
                - 5.0 * v2 * w2
                + ca2 * v2 * w2
                - 8.0 * cf * v2 * w2
                + 38.0 * ca * cf * v2 * w2
                + 30.0 * v3 * w2
                - 10.0 * ca2 * v3 * w2
                - 108.0 * ca * cf * v3 * w2
                - 58.0 * v4 * w2
                + 22.0 * ca2 * v4 * w2
                + 24.0 * cf * v4 * w2
                + 120.0 * ca * cf * v4 * w2
                + 48.0 * v5 * w2
                - 20.0 * ca2 * v5 * w2
                - 16.0 * cf * v5 * w2
                - 64.0 * ca * cf * v5 * w2
                - 14.0 * v6 * w2
                + 6.0 * ca2 * v6 * w2
                + 16.0 * ca * cf * v6 * w2
                - 8.0 * v3 * w3
                + 2.0 * ca2 * v3 * w3
                - 16.0 * cf * v3 * w3
                + 36.0 * ca * cf * v3 * w3
                + 30.0 * v4 * w3
                - 12.0 * ca2 * v4 * w3
                + 12.0 * cf * v4 * w3
                - 64.0 * ca * cf * v4 * w3
                - 40.0 * v5 * w3
                + 20.0 * ca2 * v5 * w3
                + 4.0 * cf * v5 * w3
                + 32.0 * ca * cf * v5 * w3
                + 16.0 * v6 * w3
                - 8.0 * ca2 * v6 * w3
                - 8.0 * ca * cf * v6 * w3
                - 7.0 * v4 * w4
                + 3.0 * ca2 * v4 * w4
                - 12.0 * cf * v4 * w4
                + 20.0 * ca * cf * v4 * w4
                + 22.0 * v5 * w4
                - 10.0 * ca2 * v5 * w4
                + 8.0 * cf * v5 * w4
                - 12.0 * ca * cf * v5 * w4
                - 14.0 * v6 * w4
                + 6.0 * ca2 * v6 * w4
                - 6.0 * v5 * w5
                + 2.0 * ca2 * v5 * w5
                - 4.0 * cf * v5 * w5
                + 4.0 * ca * cf * v5 * w5
                + 8.0 * v6 * w5
                - 4.0 * ca2 * v6 * w5
                - 2.0 * v6 * w6
                + 2.0 * ca2 * v6 * w6))
            / ((1.0 - v) * v * w * (1.0 - v + v * w).powi(3));

    let part3 = (cf
        * (4.0 - 2.0 * ca - 4.0 * ca2 + 2.0 * ca3 + 8.0 * ca * cf - 4.0 * ca2cf - 24.0 * v
            + 8.0 * ca * v
            + 24.0 * ca2 * v
            - 8.0 * ca3 * v
            - 48.0 * ca * cf * v
            + 16.0 * ca2cf * v
            + 56.0 * v2
            - 12.0 * ca * v2
            - 56.0 * ca2 * v2
            + 12.0 * ca3 * v2
            + 104.0 * ca * cf * v2
            - 24.0 * ca2cf * v2
            - 64.0 * v3
            + 8.0 * ca * v3
            + 64.0 * ca2 * v3
            - 8.0 * ca3 * v3
            - 104.0 * ca * cf * v3
            + 16.0 * ca2cf * v3
            + 36.0 * v4
            - 2.0 * ca * v4
            - 36.0 * ca2 * v4
            + 2.0 * ca3 * v4
            + 48.0 * ca * cf * v4
            - 4.0 * ca2cf * v4
            - 8.0 * v5
            + 8.0 * ca2 * v5
            - 8.0 * ca * cf * v5
            - 4.0 * w
            + 2.0 * ca * w
            + 4.0 * ca2 * w
            - 2.0 * ca3 * w
            - 8.0 * ca * cf * w
            + 4.0 * ca2cf * w
            + 60.0 * v * w
            - 44.0 * ca * v * w
            - 60.0 * ca2 * v * w
            + 28.0 * ca3 * v * w
            + 120.0 * ca * cf * v * w
            - 20.0 * ca2cf * v * w
            - 172.0 * v2 * w
            + 125.0 * ca * v2 * w
            + 156.0 * ca2 * v2 * w
            - 85.0 * ca3 * v2 * w
            - 336.0 * ca * cf * v2 * w
            + 20.0 * ca2cf * v2 * w
            + 180.0 * v3 * w
            - 137.0 * ca * v3 * w
            - 132.0 * ca2 * v3 * w
            + 105.0 * ca3 * v3 * w
            + 328.0 * ca * cf * v3 * w
            + 20.0 * ca2cf * v3 * w
            - 48.0 * v4 * w
            + 71.0 * ca * v4 * w
            - 55.0 * ca3 * v4 * w
            - 72.0 * ca * cf * v4 * w
            - 40.0 * ca2cf * v4 * w
            - 32.0 * v5 * w
            - 23.0 * ca * v5 * w
            + 48.0 * ca2 * v5 * w
            + 7.0 * ca3 * v5 * w
            - 48.0 * ca * cf * v5 * w
            + 16.0 * ca2cf * v5 * w
            + 16.0 * v6 * w
            + 6.0 * ca * v6 * w
            - 16.0 * ca2 * v6 * w
            + 2.0 * ca3 * v6 * w
            + 16.0 * ca * cf * v6 * w
            - 4.0 * v * w2
            + 2.0 * ca * v * w2
            + 4.0 * ca2 * v * w2
            - 2.0 * ca3 * v * w2
            - 8.0 * ca * cf * v * w2
            + 4.0 * ca2cf * v * w2
            + 72.0 * v2 * w2
            + 44.0 * ca * v2 * w2
            - 16.0 * ca2 * v2 * w2
            - 4.0 * ca3 * v2 * w2
            + 56.0 * ca * cf * v2 * w2
            + 12.0 * ca2cf * v2 * w2
            - 80.0 * v3 * w2
            - 185.0 * ca * v3 * w2
            - 104.0 * ca2 * v3 * w2
            + 41.0 * ca3 * v3 * w2
            + 112.0 * ca * cf * v3 * w2
            - 76.0 * ca2cf * v3 * w2
            - 80.0 * v4 * w2
            + 248.0 * ca * v4 * w2
            + 276.0 * ca2 * v4 * w2
            - 68.0 * ca3 * v4 * w2
            - 440.0 * ca * cf * v4 * w2
            + 84.0 * ca2cf * v4 * w2
            + 124.0 * v5 * w2
            - 139.0 * ca * v5 * w2
            - 188.0 * ca2 * v5 * w2
            + 35.0 * ca3 * v5 * w2
            + 328.0 * ca * cf * v5 * w2
            - 8.0 * ca2cf * v5 * w2
            - 24.0 * v6 * w2
            + 42.0 * ca * v6 * w2
            + 20.0 * ca2 * v6 * w2
            + 2.0 * ca3 * v6 * w2
            - 40.0 * ca * cf * v6 * w2
            - 16.0 * ca2cf * v6 * w2
            - 8.0 * v7 * w2
            - 12.0 * ca * v7 * w2
            + 8.0 * ca2 * v7 * w2
            - 4.0 * ca3 * v7 * w2
            - 8.0 * ca * cf * v7 * w2
            + 8.0 * v2 * w3
            - 4.0 * ca * v2 * w3
            - 8.0 * ca2 * v2 * w3
            + 4.0 * ca3 * v2 * w3
            + 16.0 * ca * cf * v2 * w3
            - 8.0 * ca2cf * v2 * w3
            - 52.0 * v3 * w3
            + 12.0 * ca * v3 * w3
            + 108.0 * ca2 * v3 * w3
            - 12.0 * ca3 * v3 * w3
            - 192.0 * ca * cf * v3 * w3
            + 48.0 * ca2cf * v3 * w3
            + 172.0 * v4 * w3
            + 22.0 * ca * v4 * w3
            - 260.0 * ca2 * v4 * w3
            - 82.0 * ca3 * v4 * w3
            + 480.0 * ca * cf * v4 * w3
            - 40.0 * ca2cf * v4 * w3
            - 116.0 * v5 * w3
            - 76.0 * ca * v5 * w3
            + 120.0 * ca2 * v5 * w3
            + 184.0 * ca3 * v5 * w3
            - 240.0 * ca * cf * v5 * w3
            - 48.0 * ca2cf * v5 * w3
            - 42.0 * v6 * w3
            + 49.0 * ca * v6 * w3
            + 70.0 * ca2 * v6 * w3
            - 109.0 * ca3 * v6 * w3
            - 120.0 * ca * cf * v6 * w3
            + 44.0 * ca2cf * v6 * w3
            + 30.0 * v7 * w3
            - 9.0 * ca * v7 * w3
            - 30.0 * ca2 * v7 * w3
            + 13.0 * ca3 * v7 * w3
            + 48.0 * ca * cf * v7 * w3
            + 4.0 * ca2cf * v7 * w3
            + 6.0 * ca * v8 * w3
            + 2.0 * ca3 * v8 * w3
            + 8.0 * v3 * w4
            - 4.0 * ca * v3 * w4
            - 8.0 * ca2 * v3 * w4
            + 4.0 * ca3 * v3 * w4
            + 16.0 * ca * cf * v3 * w4
            - 8.0 * ca2cf * v3 * w4
            - 56.0 * v4 * w4
            - 38.0 * ca * v4 * w4
            + 48.0 * ca2 * v4 * w4
            + 86.0 * ca3 * v4 * w4
            - 88.0 * ca * cf * v4 * w4
            - 4.0 * ca2cf * v4 * w4
            - 12.0 * v5 * w4
            + 94.0 * ca * v5 * w4
            + 52.0 * ca2 * v5 * w4
            - 210.0 * ca3 * v5 * w4
            - 112.0 * ca * cf * v5 * w4
            + 64.0 * ca2cf * v5 * w4
            + 126.0 * v6 * w4
            - 80.0 * ca * v6 * w4
            - 158.0 * ca2 * v6 * w4
            + 132.0 * ca3 * v6 * w4
            + 328.0 * ca * cf * v6 * w4
            - 44.0 * ca2cf * v6 * w4
            - 40.0 * v7 * w4
            + 33.0 * ca * v7 * w4
            + 40.0 * ca2 * v7 * w4
            - 13.0 * ca3 * v7 * w4
            - 88.0 * ca * cf * v7 * w4
            - 12.0 * ca2cf * v7 * w4
            - 12.0 * ca * v8 * w4
            - 4.0 * ca3 * v8 * w4
            - 4.0 * v4 * w5
            + 2.0 * ca * v4 * w5
            + 4.0 * ca2 * v4 * w5
            - 2.0 * ca3 * v4 * w5
            - 8.0 * ca * cf * v4 * w5
            + 4.0 * ca2cf * v4 * w5
            + 40.0 * v5 * w5
            - 48.0 * ca2 * v5 * w5
            + 40.0 * ca3 * v5 * w5
            + 104.0 * ca * cf * v5 * w5
            - 28.0 * ca2cf * v5 * w5
            - 96.0 * v6 * w5
            + 13.0 * ca * v6 * w5
            + 104.0 * ca2 * v6 * w5
            - 65.0 * ca3 * v6 * w5
            - 240.0 * ca * cf * v6 * w5
            + 20.0 * ca2cf * v6 * w5
            + 14.0 * v7 * w5
            - 15.0 * ca * v7 * w5
            - 14.0 * ca2 * v7 * w5
            + 11.0 * ca3 * v7 * w5
            + 64.0 * ca * cf * v7 * w5
            + 12.0 * ca2cf * v7 * w5
            + 6.0 * ca * v8 * w5
            + 10.0 * ca3 * v8 * w5
            - 4.0 * v5 * w6
            + 2.0 * ca * v5 * w6
            + 4.0 * ca2 * v5 * w6
            - 2.0 * ca3 * v5 * w6
            - 8.0 * ca * cf * v5 * w6
            + 4.0 * ca2cf * v5 * w6
            + 20.0 * v6 * w6
            - 4.0 * ca * v6 * w6
            - 20.0 * ca2 * v6 * w6
            + 28.0 * ca3 * v6 * w6
            + 56.0 * ca * cf * v6 * w6
            - 4.0 * ca2cf * v6 * w6
            + 12.0 * v7 * w6
            + 3.0 * ca * v7 * w6
            - 12.0 * ca2 * v7 * w6
            + ca3 * v7 * w6
            - 16.0 * ca * cf * v7 * w6
            - 4.0 * ca2cf * v7 * w6
            - 16.0 * ca3 * v8 * w6
            - 8.0 * v7 * w7
            + 8.0 * ca2 * v7 * w7
            - 8.0 * ca3 * v7 * w7
            + 8.0 * ca3 * v8 * w7))
        / (ca * t1 * (1.0 - v + v * w).powi(3));

    part1 + part2 + part3 + qqbar_elastic_quark_frag_part4(w, v, ctx, pre)
}

fn qqbar_elastic_quark_frag_part4(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7);
    let (w2, w3, w4, w5, w6, w7, w8) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8);
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let ca2cf = ca2 * cf;
    let (l1w, lv, lw) = (pre.l1w, pre.lv, pre.lw);

    let t1 = ca * (1.0 - v) * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(3);

    let part4a = -(2.0
        * cf
        * l1w
        * (8.0 * ca - 8.0 * ca3 - 32.0 * ca * v + 32.0 * ca3 * v + 52.0 * ca * v2
            - 52.0 * ca3 * v2
            - 44.0 * ca * v3
            + 44.0 * ca3 * v3
            + 20.0 * ca * v4
            - 20.0 * ca3 * v4
            - 4.0 * ca * v5
            + 4.0 * ca3 * v5
            - 14.0 * ca * w
            + 26.0 * ca3 * w
            + 6.0 * v * w
            + 67.0 * ca * v * w
            - 14.0 * ca2 * v * w
            - 117.0 * ca3 * v * w
            - 18.0 * v2 * w
            - 117.0 * ca * v2 * w
            + 42.0 * ca2 * v2 * w
            + 195.0 * ca3 * v2 * w
            + 14.0 * v3 * w
            + 93.0 * ca * v3 * w
            - 42.0 * ca2 * v3 * w
            - 147.0 * ca3 * v3 * w
            + 2.0 * v4 * w
            - 25.0 * ca * v4 * w
            + 14.0 * ca2 * v4 * w
            + 39.0 * ca3 * v4 * w
            - 4.0 * v5 * w
            - 12.0 * ca * v5 * w
            + 12.0 * ca3 * v5 * w
            + 8.0 * ca * v6 * w
            - 8.0 * ca3 * v6 * w
            + 12.0 * v * w2
            - 14.0 * ca * v * w2
            - 20.0 * ca2 * v * w2
            + 18.0 * ca3 * v * w2
            - 30.0 * v2 * w2
            - 11.0 * ca * v2 * w2
            + 46.0 * ca2 * v2 * w2
            + 13.0 * ca3 * v2 * w2
            + 46.0 * v3 * w2
            + 128.0 * ca * v3 * w2
            - 62.0 * ca2 * v3 * w2
            - 160.0 * ca3 * v3 * w2
            - 54.0 * v4 * w2
            - 199.0 * ca * v4 * w2
            + 66.0 * ca2 * v4 * w2
            + 241.0 * ca3 * v4 * w2
            + 22.0 * v5 * w2
            + 136.0 * ca * v5 * w2
            - 30.0 * ca2 * v5 * w2
            - 152.0 * ca3 * v5 * w2
            + 4.0 * v6 * w2
            - 36.0 * ca * v6 * w2
            + 36.0 * ca3 * v6 * w2
            - 4.0 * ca * v7 * w2
            + 4.0 * ca3 * v7 * w2
            + 12.0 * v2 * w3
            + 28.0 * ca * v2 * w3
            - 20.0 * ca2 * v2 * w3
            - 32.0 * ca3 * v2 * w3
            - 30.0 * v3 * w3
            - 150.0 * ca * v3 * w3
            + 40.0 * ca2 * v3 * w3
            + 146.0 * ca3 * v3 * w3
            + 46.0 * v4 * w3
            + 200.0 * ca * v4 * w3
            - 50.0 * ca2 * v4 * w3
            - 180.0 * ca3 * v4 * w3
            - 10.0 * v5 * w3
            - 85.0 * ca * v5 * w3
            + 16.0 * ca2 * v5 * w3
            + 67.0 * ca3 * v5 * w3
            - 22.0 * v6 * w3
            - 17.0 * ca * v6 * w3
            + 14.0 * ca2 * v6 * w3
            + 23.0 * ca3 * v6 * w3
            + 28.0 * ca * v7 * w3
            - 28.0 * ca3 * v7 * w3
            - 8.0 * v3 * w4
            + 34.0 * ca * v3 * w4
            + 20.0 * ca2 * v3 * w4
            - 10.0 * ca3 * v3 * w4
            + 10.0 * v4 * w4
            - 6.0 * ca * v4 * w4
            - 40.0 * ca2 * v4 * w4
            - 42.0 * ca3 * v4 * w4
            - 20.0 * v5 * w4
            - 114.0 * ca * v5 * w4
            + 32.0 * ca2 * v5 * w4
            + 134.0 * ca3 * v5 * w4
            + 36.0 * v6 * w4
            + 117.0 * ca * v6 * w4
            - 26.0 * ca2 * v6 * w4
            - 115.0 * ca3 * v6 * w4
            - 52.0 * ca * v7 * w4
            + 52.0 * ca3 * v7 * w4
            - 8.0 * v4 * w5
            - 26.0 * ca * v4 * w5
            + 20.0 * ca2 * v4 * w5
            + 26.0 * ca3 * v4 * w5
            + 8.0 * v5 * w5
            + 107.0 * ca * v5 * w5
            - 10.0 * ca2 * v5 * w5
            - 93.0 * ca3 * v5 * w5
            - 18.0 * v6 * w5
            - 89.0 * ca * v6 * w5
            + 2.0 * ca2 * v6 * w5
            + 75.0 * ca3 * v6 * w5
            + 36.0 * ca * v7 * w5
            - 36.0 * ca3 * v7 * w5
            + 4.0 * v5 * w6
            - 20.0 * ca * v5 * w6
            - 8.0 * ca2 * v5 * w6
            + 16.0 * ca3 * v5 * w6
            - 4.0 * v6 * w6
            + 17.0 * ca * v6 * w6
            + 18.0 * ca2 * v6 * w6
            - 11.0 * ca3 * v6 * w6
            - 12.0 * ca * v7 * w6
            + 12.0 * ca3 * v7 * w6
            + 4.0 * v6 * w7
            - 8.0 * ca2 * v6 * w7
            + 8.0 * ca * v7 * w7
            - 8.0 * ca3 * v7 * w7
            - 4.0 * ca * v7 * w8
            + 4.0 * ca3 * v7 * w8))
        / (t1);

    let part4b = -(2.0
        * cf
        * lv
        * (8.0 * ca - 8.0 * ca3 + 2.0 * v - 32.0 * ca * v - 2.0 * ca2 * v + 32.0 * ca3 * v
            - 6.0 * v2
            + 52.0 * ca * v2
            + 6.0 * ca2 * v2
            - 52.0 * ca3 * v2
            + 6.0 * v3
            - 44.0 * ca * v3
            - 6.0 * ca2 * v3
            + 44.0 * ca3 * v3
            - 2.0 * v4
            + 20.0 * ca * v4
            + 2.0 * ca2 * v4
            - 20.0 * ca3 * v4
            - 4.0 * ca * v5
            + 4.0 * ca3 * v5
            + 42.0 * ca * w
            + 18.0 * ca3 * w
            - 14.0 * v * w
            - 157.0 * ca * v * w
            - 18.0 * ca2 * v * w
            - 85.0 * ca3 * v * w
            + 44.0 * v2 * w
            + 255.0 * ca * v2 * w
            + 52.0 * ca2 * v2 * w
            + 139.0 * ca3 * v2 * w
            - 46.0 * v3 * w
            - 239.0 * ca * v3 * w
            - 54.0 * ca2 * v3 * w
            - 91.0 * ca3 * v3 * w
            + 16.0 * v4 * w
            + 139.0 * ca * v4 * w
            + 24.0 * ca2 * v4 * w
            + 7.0 * ca3 * v4 * w
            - 48.0 * ca * v5 * w
            - 4.0 * ca2 * v5 * w
            + 20.0 * ca3 * v5 * w
            + 8.0 * ca * v6 * w
            - 8.0 * ca3 * v6 * w
            + 14.0 * ca * v * w2
            - 32.0 * ca2 * v * w2
            + 14.0 * ca3 * v * w2
            - 14.0 * v2 * w2
            + 45.0 * ca * v2 * w2
            + 78.0 * ca2 * v2 * w2
            + ca3 * v2 * w2
            + 8.0 * v3 * w2
            - 180.0 * ca * v3 * w2
            - 96.0 * ca2 * v3 * w2
            - 106.0 * ca3 * v3 * w2
            + 26.0 * v4 * w2
            + 209.0 * ca * v4 * w2
            + 82.0 * ca2 * v4 * w2
            + 169.0 * ca3 * v4 * w2
            - 22.0 * v5 * w2
            - 120.0 * ca * v5 * w2
            - 34.0 * ca2 * v5 * w2
            - 102.0 * ca3 * v5 * w2
            + 2.0 * v6 * w2
            + 36.0 * ca * v6 * w2
            + 2.0 * ca2 * v6 * w2
            + 20.0 * ca3 * v6 * w2
            - 4.0 * ca * v7 * w2
            + 4.0 * ca3 * v7 * w2
            - 56.0 * ca * v2 * w3
            - 32.0 * ca2 * v2 * w3
            - 20.0 * ca3 * v2 * w3
            + 4.0 * v3 * w3
            + 174.0 * ca * v3 * w3
            + 52.0 * ca2 * v3 * w3
            + 96.0 * ca3 * v3 * w3
            - 28.0 * v4 * w3
            - 156.0 * ca * v4 * w3
            - 44.0 * ca2 * v4 * w3
            - 122.0 * ca3 * v4 * w3
            + 20.0 * v5 * w3
            + 47.0 * ca * v5 * w3
            + 8.0 * ca2 * v5 * w3
            + 41.0 * ca3 * v5 * w3
            + 2.0 * v6 * w3
            + 3.0 * ca * v6 * w3
            + 14.0 * ca2 * v6 * w3
            + 21.0 * ca3 * v6 * w3
            - 8.0 * ca * v7 * w3
            - 20.0 * ca3 * v7 * w3
            + 12.0 * v3 * w4
            + 10.0 * ca * v3 * w4
            + 36.0 * ca2 * v3 * w4
            - 6.0 * ca3 * v3 * w4
            - 20.0 * v4 * w4
            - 86.0 * ca * v4 * w4
            - 84.0 * ca2 * v4 * w4
            - 28.0 * ca3 * v4 * w4
            + 12.0 * v5 * w4
            + 122.0 * ca * v5 * w4
            + 60.0 * ca2 * v5 * w4
            + 92.0 * ca3 * v5 * w4
            - 8.0 * v6 * w4
            - 59.0 * ca * v6 * w4
            - 28.0 * ca2 * v6 * w4
            - 81.0 * ca3 * v6 * w4
            + 20.0 * ca * v7 * w4
            + 36.0 * ca3 * v7 * w4
            + 12.0 * v4 * w5
            + 6.0 * ca * v4 * w5
            + 36.0 * ca2 * v4 * w5
            + 22.0 * ca3 * v4 * w5
            - 6.0 * v5 * w5
            + 3.0 * ca * v5 * w5
            - 18.0 * ca2 * v5 * w5
            - 75.0 * ca3 * v5 * w5
            - 2.0 * v6 * w5
            - 25.0 * ca * v6 * w5
            - 6.0 * ca2 * v6 * w5
            + 61.0 * ca3 * v6 * w5
            + 4.0 * ca * v7 * w5
            - 28.0 * ca3 * v7 * w5
            - 4.0 * v5 * w6
            - 24.0 * ca * v5 * w6
            - 12.0 * ca2 * v5 * w6
            + 16.0 * ca3 * v5 * w6
            + 10.0 * v6 * w6
            + 41.0 * ca * v6 * w6
            + 30.0 * ca2 * v6 * w6
            - 13.0 * ca3 * v6 * w6
            - 20.0 * ca * v7 * w6
            + 12.0 * ca3 * v7 * w6
            - 4.0 * v6 * w7
            - 4.0 * ca * v6 * w7
            - 12.0 * ca2 * v6 * w7
            + 12.0 * ca * v7 * w7
            - 8.0 * ca3 * v7 * w7
            - 4.0 * ca * v7 * w8
            + 4.0 * ca3 * v7 * w8))
        / (t1);

    let part4c = -(4.0
        * cf
        * lw
        * (6.0 * ca - 6.0 * ca3 + 4.0 * ca2cf - 2.0 * v - 24.0 * ca * v
            + 2.0 * ca2 * v
            + 24.0 * ca3 * v
            - 2.0 * ca * cf * v
            - 16.0 * ca2cf * v
            + 6.0 * v2
            + 39.0 * ca * v2
            - 6.0 * ca2 * v2
            - 39.0 * ca3 * v2
            + 6.0 * ca * cf * v2
            + 26.0 * ca2cf * v2
            - 6.0 * v3
            - 33.0 * ca * v3
            + 6.0 * ca2 * v3
            + 33.0 * ca3 * v3
            - 6.0 * ca * cf * v3
            - 22.0 * ca2cf * v3
            + 2.0 * v4
            + 15.0 * ca * v4
            - 2.0 * ca2 * v4
            - 15.0 * ca3 * v4
            + 2.0 * ca * cf * v4
            + 10.0 * ca2cf * v4
            - 3.0 * ca * v5
            + 3.0 * ca3 * v5
            - 2.0 * ca2cf * v5
            - 8.0 * ca * w
            + 2.0 * ca3 * w
            + v * w
            + 40.0 * ca * v * w
            + 3.0 * ca2 * v * w
            - 15.0 * ca3 * v * w
            + 4.0 * ca2cf * v * w
            - 5.0 * v2 * w
            - 62.0 * ca * v2 * w
            - 7.0 * ca2 * v2 * w
            + 21.0 * ca3 * v2 * w
            - 2.0 * ca * cf * v2 * w
            - 4.0 * ca2cf * v2 * w
            + 3.0 * v3 * w
            + 29.0 * ca * v3 * w
            + 9.0 * ca2 * v3 * w
            + 4.0 * ca3 * v3 * w
            - 10.0 * ca2cf * v3 * w
            + 5.0 * v4 * w
            + 14.0 * ca * v4 * w
            - 9.0 * ca2 * v4 * w
            - 27.0 * ca3 * v4 * w
            + 6.0 * ca * cf * v4 * w
            + 20.0 * ca2cf * v4 * w
            - 4.0 * v5 * w
            - 19.0 * ca * v5 * w
            + 4.0 * ca2 * v5 * w
            + 21.0 * ca3 * v5 * w
            - 4.0 * ca * cf * v5 * w
            - 14.0 * ca2cf * v5 * w
            + 6.0 * ca * v6 * w
            - 6.0 * ca3 * v6 * w
            + 4.0 * ca2cf * v6 * w
            - 10.0 * ca * w2
            - 8.0 * ca3 * w2
            + 20.0 * ca2cf * w2
            + v * w2
            + 24.0 * ca * v * w2
            - ca2 * v * w2
            + 33.0 * ca3 * v * w2
            - 2.0 * ca * cf * v * w2
            - 72.0 * ca2cf * v * w2
            - 2.0 * v2 * w2
            - 35.0 * ca * v2 * w2
            + 6.0 * ca2 * v2 * w2
            - 43.0 * ca3 * v2 * w2
            + 6.0 * ca * cf * v2 * w2
            + 98.0 * ca2cf * v2 * w2
            + 8.0 * v3 * w2
            + 75.0 * ca * v3 * w2
            - 8.0 * ca2 * v3 * w2
            - 5.0 * ca3 * v3 * w2
            - 2.0 * ca * cf * v3 * w2
            - 50.0 * ca2cf * v3 * w2
            - 18.0 * v4 * w2
            - 95.0 * ca * v4 * w2
            + 6.0 * ca2 * v4 * w2
            + 47.0 * ca3 * v4 * w2
            - 10.0 * ca * cf * v4 * w2
            - 6.0 * ca2cf * v4 * w2
            + 9.0 * v5 * w2
            + 51.0 * ca * v5 * w2
            - ca2 * v5 * w2
            - 30.0 * ca3 * v5 * w2
            + 6.0 * ca * cf * v5 * w2
            + 14.0 * ca2cf * v5 * w2
            + 2.0 * v6 * w2
            - 7.0 * ca * v6 * w2
            - 2.0 * ca2 * v6 * w2
            + 3.0 * ca3 * v6 * w2
            + 2.0 * ca * cf * v6 * w2
            - 2.0 * ca2cf * v6 * w2
            - 3.0 * ca * v7 * w2
            + 3.0 * ca3 * v7 * w2
            - 2.0 * ca2cf * v7 * w2
            + 2.0 * v * w3
            + 10.0 * ca2 * v * w3
            - 6.0 * ca3 * v * w3
            - 8.0 * ca * cf * v * w3
            + 12.0 * ca2cf * v * w3
            - 5.0 * v2 * w3
            - 10.0 * ca * v2 * w3
            - 31.0 * ca2 * v2 * w3
            + 9.0 * ca3 * v2 * w3
            + 22.0 * ca * cf * v2 * w3
            - 12.0 * ca2cf * v2 * w3
            - 4.0 * v3 * w3
            - 21.0 * ca * v3 * w3
            + 31.0 * ca2 * v3 * w3
            + 25.0 * ca3 * v3 * w3
            - 32.0 * ca * cf * v3 * w3
            - 30.0 * ca2cf * v3 * w3
            + 22.0 * v4 * w3
            + 62.0 * ca * v4 * w3
            - 8.0 * ca2 * v4 * w3
            - 50.0 * ca3 * v4 * w3
            + 34.0 * ca * cf * v4 * w3
            + 52.0 * ca2cf * v4 * w3
            - 10.0 * v5 * w3
            - 25.0 * ca * v5 * w3
            - 3.0 * ca2 * v5 * w3
            + 16.0 * ca3 * v5 * w3
            - 14.0 * ca * cf * v5 * w3
            - 22.0 * ca2cf * v5 * w3
            - 7.0 * v6 * w3
            - 14.0 * ca * v6 * w3
            + 3.0 * ca2 * v6 * w3
            + 12.0 * ca3 * v6 * w3
            - 4.0 * ca * cf * v6 * w3
            - 4.0 * ca2cf * v6 * w3
            + 11.0 * ca * v7 * w3
            - 9.0 * ca3 * v7 * w3
            + 6.0 * ca2cf * v7 * w3
            + 2.0 * v2 * w4
            + 14.0 * ca * v2 * w4
            + 10.0 * ca2 * v2 * w4
            + 10.0 * ca3 * v2 * w4
            - 8.0 * ca * cf * v2 * w4
            - 24.0 * ca2cf * v2 * w4
            + 9.0 * v3 * w4
            - 2.0 * ca * v3 * w4
            - 14.0 * ca2 * v3 * w4
            - 43.0 * ca3 * v3 * w4
            + 24.0 * ca * cf * v3 * w4
            + 80.0 * ca2cf * v3 * w4
            - 32.0 * v4 * w4
            - 36.0 * ca * v4 * w4
            - 3.0 * ca2 * v4 * w4
            + 54.0 * ca3 * v4 * w4
            - 36.0 * ca * cf * v4 * w4
            - 76.0 * ca2cf * v4 * w4
            + 15.0 * v5 * w4
            - 3.0 * ca * v5 * w4
            + 4.0 * ca2 * v5 * w4
            - 8.0 * ca3 * v5 * w4
            + 20.0 * ca * cf * v5 * w4
            + 14.0 * ca2cf * v5 * w4
            + 11.0 * v6 * w4
            + 37.0 * ca * v6 * w4
            + 2.0 * ca2 * v6 * w4
            - 21.0 * ca3 * v6 * w4
            + 2.0 * ca * cf * v6 * w4
            + 14.0 * ca2cf * v6 * w4
            - 19.0 * ca * v7 * w4
            + 13.0 * ca3 * v7 * w4
            - 10.0 * ca2cf * v7 * w4
            - 8.0 * v3 * w5
            - 12.0 * ca * v3 * w5
            - 10.0 * ca2 * v3 * w5
            + 10.0 * ca3 * v3 * w5
            + 4.0 * ca * cf * v3 * w5
            - 16.0 * ca2cf * v3 * w5
            + 29.0 * v4 * w5
            + 30.0 * ca * v4 * w5
            + 26.0 * ca2 * v4 * w5
            - 7.0 * ca3 * v4 * w5
            - 4.0 * ca2cf * v4 * w5
            - 14.0 * v5 * w5
            + 21.0 * ca * v5 * w5
            - 11.0 * ca2 * v5 * w5
            - 22.0 * ca3 * v5 * w5
            - 6.0 * ca * cf * v5 * w5
            + 42.0 * ca2cf * v5 * w5
            - 13.0 * v6 * w5
            - 52.0 * ca * v6 * w5
            - 8.0 * ca2 * v6 * w5
            + 30.0 * ca3 * v6 * w5
            + 2.0 * ca * cf * v6 * w5
            - 32.0 * ca2cf * v6 * w5
            + 23.0 * ca * v7 * w5
            - 15.0 * ca3 * v7 * w5
            + 14.0 * ca2cf * v7 * w5
            - 8.0 * v4 * w6
            - 10.0 * ca2 * v4 * w6
            - 8.0 * ca3 * v4 * w6
            + 4.0 * ca * cf * v4 * w6
            + 16.0 * ca2cf * v4 * w6
            + 2.0 * v5 * w6
            - 35.0 * ca * v5 * w6
            + 3.0 * ca2 * v5 * w6
            + 25.0 * ca3 * v5 * w6
            + 2.0 * ca * cf * v5 * w6
            - 38.0 * ca2cf * v5 * w6
            + 13.0 * v6 * w6
            + 50.0 * ca * v6 * w6
            + 12.0 * ca2 * v6 * w6
            - 24.0 * ca3 * v6 * w6
            - 8.0 * ca * cf * v6 * w6
            + 24.0 * ca2cf * v6 * w6
            - 21.0 * ca * v7 * w6
            + 13.0 * ca3 * v7 * w6
            - 14.0 * ca2cf * v7 * w6
            + 2.0 * v5 * w7
            + 13.0 * ca * v5 * w7
            + 4.0 * ca2 * v5 * w7
            - 5.0 * ca3 * v5 * w7
            - 4.0 * ca * cf * v5 * w7
            + 6.0 * ca2cf * v5 * w7
            - 8.0 * v6 * w7
            - 22.0 * ca * v6 * w7
            - 11.0 * ca2 * v6 * w7
            + 6.0 * ca3 * v6 * w7
            + 10.0 * ca * cf * v6 * w7
            - 4.0 * ca2cf * v6 * w7
            + 13.0 * ca * v7 * w7
            - 7.0 * ca3 * v7 * w7
            + 10.0 * ca2cf * v7 * w7
            + 2.0 * v6 * w8
            + 2.0 * ca * v6 * w8
            + 4.0 * ca2 * v6 * w8
            - 4.0 * ca * cf * v6 * w8
            - 5.0 * ca * v7 * w8
            + 3.0 * ca3 * v7 * w8
            - 6.0 * ca2cf * v7 * w8
            + ca * v7 * pre.w9
            - ca3 * v7 * pre.w9
            + 2.0 * ca2cf * v7 * pre.w9))
        / (ca * (1.0 - v) * (1.0 - w) * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(3));

    part4a + part4b + part4c
}

/// `STRUV12(W,V,X3,S)`, part A: the four `Nf`-proportional terms sharing
/// the common factor `(1-2V+V2+V2W2)*(1+V2-2V2W+V2W2)/(1-V+VW)^4`
/// (verbatim identical in the Fortran source across all four terms, so
/// factored here rather than retyped four times -- not a physics
/// simplification, just avoiding four copies of the same bracket).
fn qqbar_to_gg_gluon_frag_part_a(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let nf = ctx.nf;
    let v2 = pre.v2;
    let w2 = pre.w2;
    let cacf2 = ca * cf.powi(2);
    let (l1w, lv, lmss, l1vw) = (pre.l1w, pre.lv, pre.lmss, pre.l1vw);

    let common = cacf2 * nf * (1.0 - 2.0 * v + v2 + v2 * w2) * (1.0 + v2 - 2.0 * v2 * w + v2 * w2)
        / (1.0 - v + v * w).powi(4);

    common * (-8.0 * l1w + 8.0 * lmss - 8.0 * lv - 16.0 * (1.0 - w) * l1vw)
}

/// `STRUV12(W,V,X3,S)`, part B: the `lvw`/`l1v`/`lw`/`lms` term groups.
fn qqbar_to_gg_gluon_frag_part_b(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6);
    let (w2, w3, w4, w5) = (pre.w2, pre.w3, pre.w4, pre.w5);
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let ca4 = pre.ca4;
    let (lvw, l1v, lw, lms) = (pre.lvw, pre.l1v, pre.lw, pre.lms);

    let term5 = -(4.0
        * cf
        * lvw
        * (4.0 - 2.0 * ca2 + 2.0 * ca4 - 2.0 * v - 2.0 * ca * v + 3.0 * ca2 * v
            - 4.0 * ca3 * v
            - 2.0 * ca4 * v
            + 2.0 * v2
            - 2.0 * ca * v2
            - ca2 * v2
            - 2.0 * ca3 * v2
            + ca4 * v2
            - 4.0 * v * w
            + 2.0 * ca * v * w
            + 5.0 * ca2 * v * w
            + 4.0 * ca3 * v * w
            - 8.0 * ca4 * v * w
            - v2 * w
            + 3.0 * ca * v2 * w
            - 4.0 * ca2 * v2 * w
            + 7.0 * ca3 * v2 * w
            + 7.0 * ca4 * v2 * w
            - v3 * w
            - ca * v3 * w
            - 2.0 * ca2 * v3 * w
            + ca3 * v3 * w
            - 4.0 * ca4 * v3 * w
            + 4.0 * v2 * w2
            - ca * v2 * w2
            - 12.0 * ca2 * v2 * w2
            - 5.0 * ca3 * v2 * w2
            + 13.0 * ca4 * v2 * w2
            + 2.0 * v3 * w2
            + 3.0 * ca * v3 * w2
            + 11.0 * ca2 * v3 * w2
            - 3.0 * ca3 * v3 * w2
            - 7.0 * ca4 * v3 * w2
            - 4.0 * ca2 * v4 * w2
            + 4.0 * ca4 * v4 * w2
            - v3 * w3
            - 2.0 * ca * v3 * w3
            + 6.0 * ca2 * v3 * w3
            + 2.0 * ca3 * v3 * w3
            - 9.0 * ca4 * v3 * w3
            - 4.0 * ca2 * v4 * w4
            + 4.0 * ca4 * v4 * w4))
        / (ca * (1.0 - v) * v2 * w);

    let term6 = (4.0
        * cf
        * l1v
        * (2.0 * ca4 - 2.0 * ca * cf - v + 2.0 * ca * v + 4.0 * ca2 * v
            - 2.0 * ca3 * v
            - 5.0 * ca4 * v
            + v2
            - 4.0 * ca2 * v2
            + 2.0 * ca3 * v2
            + 6.0 * ca4 * v2
            - 3.0 * v3
            - 2.0 * ca * v3
            + 2.0 * ca2 * v3
            - 4.0 * ca4 * v3
            + 2.0 * v4
            - ca2 * v4
            + ca4 * v4
            - v * w
            - ca2 * v * w
            + 3.0 * v2 * w
            - 6.0 * ca * v2 * w
            + 6.0 * ca2 * v2 * w
            + 4.0 * ca3 * v2 * w
            - 7.0 * ca4 * v2 * w
            + v3 * w
            - 16.0 * ca2 * v3 * w
            - 2.0 * ca3 * v3 * w
            + 12.0 * ca4 * v3 * w
            + 3.0 * v4 * w
            + 2.0 * ca * v4 * w
            + 18.0 * ca2 * v4 * w
            - 5.0 * ca4 * v4 * w
            - 2.0 * v5 * w
            - 5.0 * ca2 * v5 * w
            - ca4 * v5 * w
            - 3.0 * v2 * w2
            - ca2 * v2 * w2
            - 2.0 * ca4 * v2 * w2
            - v3 * w2
            + 6.0 * ca * v3 * w2
            + ca2 * v3 * w2
            - 2.0 * ca3 * v3 * w2
            + 14.0 * ca4 * v3 * w2
            - 7.0 * v4 * w2
            + 3.0 * ca2 * v4 * w2
            - 19.0 * ca4 * v4 * w2
            + v5 * w2
            - 11.0 * ca2 * v5 * w2
            + 9.0 * ca4 * v5 * w2
            + 4.0 * ca2 * v6 * w2
            - v3 * w3
            - ca2 * v3 * w3
            + 7.0 * v4 * w3
            - 2.0 * ca * v4 * w3
            - ca4 * v4 * w3
            + v5 * w3
            + 5.0 * ca2 * v5 * w3
            + ca4 * v5 * w3
            - 2.0 * v5 * w4
            + 3.0 * ca2 * v5 * w4
            - ca4 * v5 * w4
            - 4.0 * ca2 * v6 * w4))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v * w) * (1.0 - v + v * w));

    let term7 = -(4.0
        * cf
        * lw
        * (2.0 - 5.0 * ca2 + 3.0 * ca4 - 5.0 * v + 15.0 * ca2 * v
            - 4.0 * ca3 * v
            - 10.0 * ca4 * v
            + 5.0 * v2
            - 2.0 * ca * v2
            - 22.0 * ca2 * v2
            + 2.0 * ca3 * v2
            + 17.0 * ca4 * v2
            - 3.0 * v3
            + 4.0 * ca * v3
            + 21.0 * ca2 * v3
            - 18.0 * ca4 * v3
            + v4
            - 2.0 * ca * v4
            - 13.0 * ca2 * v4
            + 2.0 * ca3 * v4
            + 12.0 * ca4 * v4
            + 4.0 * ca2 * v5
            - 4.0 * ca4 * v5
            - w
            + ca2 * w
            + 5.0 * v * w
            - ca2 * v * w
            + 4.0 * ca3 * v * w
            - 5.0 * ca4 * v * w
            - 3.0 * v2 * w
            - ca * v2 * w
            + 2.0 * ca2 * v2 * w
            + 3.0 * ca3 * v2 * w
            + 9.0 * ca4 * v2 * w
            + 2.0 * v3 * w
            - 10.0 * ca * v3 * w
            - 6.0 * ca2 * v3 * w
            - 4.0 * ca3 * v3 * w
            - 10.0 * ca4 * v3 * w
            + 2.0 * v4 * w
            + ca * v4 * w
            + ca2 * v4 * w
            - 5.0 * ca3 * v4 * w
            + 10.0 * ca4 * v4 * w
            - v5 * w
            + 2.0 * ca * v5 * w
            + 5.0 * ca2 * v5 * w
            - 2.0 * ca3 * v5 * w
            - 8.0 * ca4 * v5 * w
            - 4.0 * ca2 * v6 * w
            + 4.0 * ca4 * v6 * w
            - v * w2
            - ca2 * v * w2
            + 3.0 * ca * v2 * w2
            + 4.0 * ca2 * v2 * w2
            - 5.0 * ca3 * v2 * w2
            + 7.0 * ca4 * v2 * w2
            - 4.0 * v3 * w2
            + 14.0 * ca * v3 * w2
            - 3.0 * ca2 * v3 * w2
            + 4.0 * ca3 * v3 * w2
            - 7.0 * ca4 * v3 * w2
            - 5.0 * v4 * w2
            + 8.0 * ca * v4 * w2
            + 12.0 * ca2 * v4 * w2
            + 6.0 * ca3 * v4 * w2
            - 5.0 * ca * v5 * w2
            - 15.0 * ca2 * v5 * w2
            + 5.0 * ca3 * v5 * w2
            + 4.0 * ca4 * v5 * w2
            + 8.0 * ca2 * v6 * w2
            - 4.0 * ca4 * v6 * w2
            - v2 * w3
            - ca2 * v2 * w3
            + 5.0 * v3 * w3
            - 8.0 * ca * v3 * w3
            - 2.0 * ca2 * v3 * w3
            - 6.0 * ca4 * v3 * w3
            + 6.0 * v4 * w3
            - 12.0 * ca * v4 * w3
            - 8.0 * ca2 * v4 * w3
            - 4.0 * ca3 * v4 * w3
            + 6.0 * ca4 * v4 * w3
            + 3.0 * v5 * w3
            + 4.0 * ca * v5 * w3
            + 13.0 * ca2 * v5 * w3
            - 4.0 * ca3 * v5 * w3
            - 3.0 * ca4 * v5 * w3
            - 8.0 * ca2 * v6 * w3
            + 4.0 * ca4 * v6 * w3
            - v3 * w4
            - ca2 * v3 * w4
            - 4.0 * v4 * w4
            + 5.0 * ca * v4 * w4
            + 9.0 * ca2 * v4 * w4
            + ca3 * v4 * w4
            - 4.0 * v5 * w4
            - ca * v5 * w4
            - 12.0 * ca2 * v5 * w4
            + ca3 * v5 * w4
            + 2.0 * ca4 * v5 * w4
            + 8.0 * ca2 * v6 * w4
            - 4.0 * ca4 * v6 * w4
            - 2.0 * ca2 * v4 * w5
            + 2.0 * v5 * w5
            + 5.0 * ca2 * v5 * w5
            + ca4 * v5 * w5
            - 4.0 * ca2 * v6 * w5))
        / (ca * (1.0 - v) * v2 * (1.0 - w) * w * (1.0 - v * w) * (1.0 - v + v * w));

    let term8 = (2.0
        * cf
        * lms
        * (2.0 - 2.0 * ca2 - 6.0 * cf + 6.0 * ca2 * cf - 2.0 * v - 6.0 * ca2 * v + 8.0 * cf * v
            - 12.0 * ca2 * cf * v
            - 2.0 * v2
            - 6.0 * ca2 * v2
            - 6.0 * cf * v2
            + 18.0 * ca2 * cf * v2
            + 2.0 * v3
            - 2.0 * ca2 * v3
            - 16.0 * ca2 * cf * v3
            + 8.0 * ca2 * cf * v4
            - w
            + ca2 * w
            + 2.0 * cf * w
            - 2.0 * ca2 * cf * w
            - 3.0 * v * w
            + 5.0 * ca2 * v * w
            + 8.0 * cf * v * w
            - 4.0 * ca2 * cf * v * w
            + 5.0 * v2 * w
            + 19.0 * ca2 * v2 * w
            - 6.0 * cf * v2 * w
            - 14.0 * ca2 * cf * v2 * w
            + 3.0 * v3 * w
            + 11.0 * ca2 * v3 * w
            + 12.0 * cf * v3 * w
            + 8.0 * ca2 * cf * v3 * w
            - 4.0 * v4 * w
            + 4.0 * ca2 * v4 * w
            + 8.0 * ca2 * cf * v4 * w
            - 16.0 * ca2 * cf * v5 * w
            + 2.0 * v * w2
            - 2.0 * ca2 * v * w2
            - 4.0 * cf * v * w2
            + 4.0 * ca2 * cf * v * w2
            - 4.0 * ca2 * v2 * w2
            + 2.0 * cf * v2 * w2
            - 10.0 * ca2 * cf * v2 * w2
            - 2.0 * v3 * w2
            - 20.0 * ca2 * v3 * w2
            - 8.0 * cf * v3 * w2
            + 80.0 * ca2 * cf * v3 * w2
            + 2.0 * v4 * w2
            - 8.0 * ca2 * v4 * w2
            - 8.0 * cf * v4 * w2
            - 80.0 * ca2 * cf * v4 * w2
            + 2.0 * v5 * w2
            - 2.0 * ca2 * v5 * w2
            + 32.0 * ca2 * cf * v5 * w2
            + 8.0 * ca2 * cf * v6 * w2
            - v2 * w3
            + ca2 * v2 * w3
            + 2.0 * cf * v2 * w3
            - 2.0 * ca2 * cf * v2 * w3
            + v3 * w3
            + ca2 * v3 * w3
            - 4.0 * cf * v3 * w3
            + 8.0 * ca2 * cf * v3 * w3
            - 4.0 * v4 * w3
            + 10.0 * ca2 * v4 * w3
            + 8.0 * cf * v4 * w3
            - 80.0 * ca2 * cf * v4 * w3
            - 2.0 * v5 * w3
            + 2.0 * ca2 * v5 * w3
            + 64.0 * ca2 * cf * v5 * w3
            - 24.0 * ca2 * cf * v6 * w3
            + 2.0 * v5 * w4
            - 2.0 * ca2 * v5 * w4
            + 32.0 * ca2 * cf * v5 * w4
            - 8.0 * ca2 * cf * v6 * w4
            - 8.0 * ca2 * cf * v6 * w5))
        / ((1.0 - v) * v2 * w * (1.0 - v * w).powi(2));

    term5 + term6 + term7 + term8
}

/// `STRUV12(W,V,X3,S)`, part C: the `Nf`/`l1vw`/`lmss` term groups over
/// `(1-v+vw)^4`.
fn qqbar_to_gg_gluon_frag_part_c(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let nf = ctx.nf;
    let (v2, v3, v4, v5, v6, v7, v8) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8);
    let (w2, w3, w4, w5, w6, w7) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7);
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let ca4 = pre.ca4;
    let ca3cf = ca3 * cf;
    let (l1vw, lmss) = (pre.l1vw, pre.lmss);

    let term9 = (4.0
        * cf
        * nf
        * (-2.0 * ca * cf - 4.0 * v + 4.0 * ca2 * v + 7.0 * v2 - 7.0 * ca2 * v2 - 8.0 * v3
            + 8.0 * ca2 * v3
            + 7.0 * v4
            - 7.0 * ca2 * v4
            - 4.0 * v5
            + 4.0 * ca2 * v5
            + v6
            - ca2 * v6
            + 4.0 * v * w
            - 4.0 * ca2 * v * w
            - 13.0 * v2 * w
            + 24.0 * ca2 * v2 * w
            + 14.0 * v3 * w
            - 58.0 * ca2 * v3 * w
            - 4.0 * v4 * w
            + 84.0 * ca2 * v4 * w
            - 2.0 * v5 * w
            - 88.0 * ca2 * v5 * w
            + v6 * w
            + 64.0 * ca2 * v6 * w
            - 26.0 * ca2 * v7 * w
            + 4.0 * ca2 * v8 * w
            + 6.0 * v2 * w2
            - 6.0 * ca2 * v2 * w2
            - 10.0 * v3 * w2
            + 23.0 * ca2 * v3 * w2
            + 30.0 * v4 * w2
            - 81.0 * ca2 * v4 * w2
            - 54.0 * v5 * w2
            + 153.0 * ca2 * v5 * w2
            + 28.0 * v6 * w2
            - 137.0 * ca2 * v6 * w2
            + 60.0 * ca2 * v7 * w2
            - 12.0 * ca2 * v8 * w2
            + 4.0 * v3 * w3
            - 4.0 * ca2 * v3 * w3
            - 34.0 * v4 * w3
            + 41.0 * ca2 * v4 * w3
            + 58.0 * v5 * w3
            - 82.0 * ca2 * v5 * w3
            - 28.0 * v6 * w3
            + 77.0 * ca2 * v6 * w3
            - 44.0 * ca2 * v7 * w3
            + 12.0 * ca2 * v8 * w3
            + v4 * w4
            - ca2 * v4 * w4
            + 2.0 * v5 * w4
            - 3.0 * ca2 * v5 * w4
            - v6 * w4
            - 6.0 * ca2 * v6 * w4
            + 12.0 * ca2 * v7 * w4
            - 4.0 * ca2 * v8 * w4
            - v6 * w5
            + 3.0 * ca2 * v6 * w5
            - 2.0 * ca2 * v7 * w5))
        / (3.0 * (1.0 - v) * v2 * w * (1.0 - v + v * w).powi(4));

    let term10 = (4.0
        * cf
        * l1vw
        * (3.0 - 2.0 * ca2 + 3.0 * ca4 - 8.0 * ca3cf - 11.0 * v - 6.0 * ca * v + 8.0 * ca2 * v
            - 12.0 * ca4 * v
            + 32.0 * ca3cf * v
            + 15.0 * v2
            + 18.0 * ca * v2
            - 12.0 * ca2 * v2
            + 18.0 * ca4 * v2
            - 48.0 * ca3cf * v2
            - 7.0 * v3
            - 12.0 * ca * v3
            + 10.0 * ca2 * v3
            - 13.0 * ca4 * v3
            + 32.0 * ca3cf * v3
            - 7.0 * v4
            - 12.0 * ca * v4
            - 10.0 * ca2 * v4
            + 7.0 * ca4 * v4
            - 8.0 * ca3cf * v4
            + 15.0 * v5
            + 18.0 * ca * v5
            + 12.0 * ca2 * v5
            - 6.0 * ca4 * v5
            - 11.0 * v6
            - 6.0 * ca * v6
            - 8.0 * ca2 * v6
            + 4.0 * ca4 * v6
            + 3.0 * v7
            + 2.0 * ca2 * v7
            - ca4 * v7
            + 9.0 * v * w
            + 6.0 * ca * v * w
            - 3.0 * ca2 * v * w
            + 6.0 * ca4 * v * w
            - 16.0 * ca3cf * v * w
            - 26.0 * v2 * w
            - 36.0 * ca * v2 * w
            + 16.0 * ca2 * v2 * w
            - 8.0 * ca4 * v2 * w
            + 48.0 * ca3cf * v2 * w
            + 19.0 * v3 * w
            + 40.0 * ca * v3 * w
            - 37.0 * ca2 * v3 * w
            - 4.0 * ca3 * v3 * w
            - 19.0 * ca4 * v3 * w
            - 48.0 * ca3cf * v3 * w
            + 24.0 * v4 * w
            + 36.0 * ca * v4 * w
            + 60.0 * ca2 * v4 * w
            + 12.0 * ca3 * v4 * w
            + 52.0 * ca4 * v4 * w
            + 16.0 * ca3cf * v4 * w
            - 61.0 * v5 * w
            - 78.0 * ca * v5 * w
            - 73.0 * ca2 * v5 * w
            - 12.0 * ca3 * v5 * w
            - 52.0 * ca4 * v5 * w
            + 50.0 * v6 * w
            + 32.0 * ca * v6 * w
            + 52.0 * ca2 * v6 * w
            + 4.0 * ca3 * v6 * w
            + 28.0 * ca4 * v6 * w
            - 15.0 * v7 * w
            - 15.0 * ca2 * v7 * w
            - 7.0 * ca4 * v7 * w
            + 13.0 * v2 * w2
            + 18.0 * ca * v2 * w2
            - 9.0 * ca2 * v2 * w2
            - 4.0 * ca4 * v2 * w2
            - 16.0 * ca3cf * v2 * w2
            - 19.0 * v3 * w2
            - 44.0 * ca * v3 * w2
            + 47.0 * ca2 * v3 * w2
            + 8.0 * ca3 * v3 * w2
            + 23.0 * ca4 * v3 * w2
            + 32.0 * ca3cf * v3 * w2
            - 30.0 * v4 * w2
            - 40.0 * ca * v4 * w2
            - 110.0 * ca2 * v4 * w2
            - 32.0 * ca3 * v4 * w2
            - 50.0 * ca4 * v4 * w2
            - 16.0 * ca3cf * v4 * w2
            + 98.0 * v5 * w2
            + 140.0 * ca * v5 * w2
            + 162.0 * ca2 * v5 * w2
            + 40.0 * ca3 * v5 * w2
            + 64.0 * ca4 * v5 * w2
            - 95.0 * v6 * w2
            - 74.0 * ca * v6 * w2
            - 137.0 * ca2 * v6 * w2
            - 16.0 * ca3 * v6 * w2
            - 50.0 * ca4 * v6 * w2
            + 33.0 * v7 * w2
            + 47.0 * ca2 * v7 * w2
            + 17.0 * ca4 * v7 * w2
            + 7.0 * v3 * w3
            + 16.0 * ca * v3 * w3
            - 20.0 * ca2 * v3 * w3
            - 4.0 * ca3 * v3 * w3
            + 9.0 * ca4 * v3 * w3
            - 16.0 * ca3cf * v3 * w3
            + 16.0 * v4 * w3
            + 20.0 * ca * v4 * w3
            + 80.0 * ca2 * v4 * w3
            + 28.0 * ca3 * v4 * w3
            - 20.0 * ca4 * v4 * w3
            + 16.0 * ca3cf * v4 * w3
            - 78.0 * v5 * w3
            - 132.0 * ca * v5 * w3
            - 168.0 * ca2 * v5 * w3
            - 48.0 * ca3 * v5 * w3
            + 20.0 * ca4 * v5 * w3
            + 100.0 * v6 * w3
            + 96.0 * ca * v6 * w3
            + 188.0 * ca2 * v6 * w3
            + 24.0 * ca3 * v6 * w3
            - 45.0 * v7 * w3
            - 80.0 * ca2 * v7 * w3
            - 9.0 * ca4 * v7 * w3
            - 3.0 * v4 * w4
            - 4.0 * ca * v4 * w4
            - 20.0 * ca2 * v4 * w4
            - 8.0 * ca3 * v4 * w4
            + 11.0 * ca4 * v4 * w4
            - 8.0 * ca3cf * v4 * w4
            + 31.0 * v5 * w4
            + 66.0 * ca * v5 * w4
            + 82.0 * ca2 * v5 * w4
            + 24.0 * ca3 * v5 * w4
            - 26.0 * ca4 * v5 * w4
            - 65.0 * v6 * w4
            - 74.0 * ca * v6 * w4
            - 142.0 * ca2 * v6 * w4
            - 16.0 * ca3 * v6 * w4
            + 8.0 * ca4 * v6 * w4
            + 45.0 * v7 * w4
            + 80.0 * ca2 * v7 * w4
            + 9.0 * ca4 * v7 * w4
            - 5.0 * v5 * w5
            - 14.0 * ca * v5 * w5
            - 15.0 * ca2 * v5 * w5
            - 4.0 * ca3 * v5 * w5
            + 26.0 * v6 * w5
            + 32.0 * ca * v6 * w5
            + 56.0 * ca2 * v6 * w5
            + 4.0 * ca3 * v6 * w5
            + 20.0 * ca4 * v6 * w5
            - 33.0 * v7 * w5
            - 47.0 * ca2 * v7 * w5
            - 17.0 * ca4 * v7 * w5
            - 5.0 * v6 * w6
            - 6.0 * ca * v6 * w6
            - 9.0 * ca2 * v6 * w6
            - 10.0 * ca4 * v6 * w6
            + 15.0 * v7 * w6
            + 15.0 * ca2 * v7 * w6
            + 7.0 * ca4 * v7 * w6
            - 3.0 * v7 * w7
            - 2.0 * ca2 * v7 * w7
            + ca4 * v7 * w7))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v + v * w).powi(4));

    let term11 = (4.0
        * cf
        * lmss
        * (4.0 * ca3cf + ca * v + 4.0 * ca2 * v - ca3 * v - 4.0 * ca4 * v + 2.0 * ca * cf * v
            - 16.0 * ca3cf * v
            + 4.0 * v2
            - 4.0 * ca * v2
            - 26.0 * ca2 * v2
            + 4.0 * ca3 * v2
            + 22.0 * ca4 * v2
            + 24.0 * ca3cf * v2
            - 7.0 * v3
            + 7.0 * ca * v3
            + 59.0 * ca2 * v3
            - 7.0 * ca3 * v3
            - 52.0 * ca4 * v3
            - 16.0 * ca3cf * v3
            + 8.0 * v4
            - 8.0 * ca * v4
            - 78.0 * ca2 * v4
            + 8.0 * ca3 * v4
            + 70.0 * ca4 * v4
            + 4.0 * ca3cf * v4
            - 7.0 * v5
            + 7.0 * ca * v5
            + 67.0 * ca2 * v5
            - 7.0 * ca3 * v5
            - 60.0 * ca4 * v5
            + 4.0 * v6
            - 4.0 * ca * v6
            - 38.0 * ca2 * v6
            + 4.0 * ca3 * v6
            + 34.0 * ca4 * v6
            - v7
            + ca * v7
            + 13.0 * ca2 * v7
            - ca3 * v7
            - 12.0 * ca4 * v7
            - 2.0 * ca2 * v8
            + 2.0 * ca4 * v8
            + 16.0 * ca3cf * v * w
            - v2 * w
            + 3.0 * ca * v2 * w
            + 15.0 * ca2 * v2 * w
            - 3.0 * ca3 * v2 * w
            - 22.0 * ca4 * v2 * w
            - 64.0 * ca3cf * v2 * w
            + 5.0 * v3 * w
            - 11.0 * ca * v3 * w
            - 73.0 * ca2 * v3 * w
            + 11.0 * ca3 * v3 * w
            + 108.0 * ca4 * v3 * w
            + 104.0 * ca3cf * v3 * w
            - 12.0 * v4 * w
            + 20.0 * ca * v4 * w
            + 154.0 * ca2 * v4 * w
            - 20.0 * ca3 * v4 * w
            - 230.0 * ca4 * v4 * w
            - 88.0 * ca3cf * v4 * w
            + 16.0 * v5 * w
            - 24.0 * ca * v5 * w
            - 184.0 * ca2 * v5 * w
            + 24.0 * ca3 * v5 * w
            + 276.0 * ca4 * v5 * w
            + 40.0 * ca3cf * v5 * w
            - 11.0 * v6 * w
            + 17.0 * ca * v6 * w
            + 133.0 * ca2 * v6 * w
            - 17.0 * ca3 * v6 * w
            - 198.0 * ca4 * v6 * w
            - 8.0 * ca3cf * v6 * w
            + 3.0 * v7 * w
            - 5.0 * ca * v7 * w
            - 55.0 * ca2 * v7 * w
            + 5.0 * ca3 * v7 * w
            + 80.0 * ca4 * v7 * w
            + 10.0 * ca2 * v8 * w
            - 14.0 * ca4 * v8 * w
            + 24.0 * ca3cf * v2 * w2
            - v3 * w2
            + 5.0 * ca * v3 * w2
            + 25.0 * ca2 * v3 * w2
            - 5.0 * ca3 * v3 * w2
            - 32.0 * ca4 * v3 * w2
            - 88.0 * ca3cf * v3 * w2
            + 6.0 * v4 * w2
            - 18.0 * ca * v4 * w2
            - 108.0 * ca2 * v4 * w2
            + 18.0 * ca3 * v4 * w2
            + 150.0 * ca4 * v4 * w2
            + 128.0 * ca3cf * v4 * w2
            - 12.0 * v5 * w2
            + 32.0 * ca * v5 * w2
            + 196.0 * ca2 * v5 * w2
            - 32.0 * ca3 * v5 * w2
            - 292.0 * ca4 * v5 * w2
            - 88.0 * ca3cf * v5 * w2
            + 10.0 * v6 * w2
            - 30.0 * ca * v6 * w2
            - 190.0 * ca2 * v6 * w2
            + 30.0 * ca3 * v6 * w2
            + 296.0 * ca4 * v6 * w2
            + 24.0 * ca3cf * v6 * w2
            - 3.0 * v7 * w2
            + 11.0 * ca * v7 * w2
            + 99.0 * ca2 * v7 * w2
            - 11.0 * ca3 * v7 * w2
            - 156.0 * ca4 * v7 * w2
            - 22.0 * ca2 * v8 * w2
            + 34.0 * ca4 * v8 * w2
            + 16.0 * ca3cf * v3 * w3
            - 2.0 * v4 * w3
            + 6.0 * ca * v4 * w3
            + 32.0 * ca2 * v4 * w3
            - 6.0 * ca3 * v4 * w3
            - 30.0 * ca4 * v4 * w3
            - 48.0 * ca3cf * v4 * w3
            + 4.0 * v5 * w3
            - 20.0 * ca * v5 * w3
            - 108.0 * ca2 * v5 * w3
            + 20.0 * ca3 * v5 * w3
            + 108.0 * ca4 * v5 * w3
            + 56.0 * ca3cf * v5 * w3
            - 4.0 * v6 * w3
            + 28.0 * ca * v6 * w3
            + 152.0 * ca2 * v6 * w3
            - 28.0 * ca3 * v6 * w3
            - 168.0 * ca4 * v6 * w3
            - 24.0 * ca3cf * v6 * w3
            + 2.0 * v7 * w3
            - 14.0 * ca * v7 * w3
            - 106.0 * ca2 * v7 * w3
            + 14.0 * ca3 * v7 * w3
            + 128.0 * ca4 * v7 * w3
            + 30.0 * ca2 * v8 * w3
            - 38.0 * ca4 * v8 * w3
            + 4.0 * ca3cf * v4 * w4
            - v5 * w4
            + 5.0 * ca * v5 * w4
            + 29.0 * ca2 * v5 * w4
            - 5.0 * ca3 * v5 * w4
            - 16.0 * ca4 * v5 * w4
            - 8.0 * ca3cf * v5 * w4
            + 2.0 * v6 * w4
            - 14.0 * ca * v6 * w4
            - 76.0 * ca2 * v6 * w4
            + 14.0 * ca3 * v6 * w4
            + 46.0 * ca4 * v6 * w4
            + 8.0 * ca3cf * v6 * w4
            - 3.0 * v7 * w4
            + 11.0 * ca * v7 * w4
            + 79.0 * ca2 * v7 * w4
            - 11.0 * ca3 * v7 * w4
            - 52.0 * ca4 * v7 * w4
            - 30.0 * ca2 * v8 * w4
            + 22.0 * ca4 * v8 * w4
            - v6 * w5
            + 3.0 * ca * v6 * w5
            + 19.0 * ca2 * v6 * w5
            - 3.0 * ca3 * v6 * w5
            + 10.0 * ca4 * v6 * w5
            + 3.0 * v7 * w5
            - 5.0 * ca * v7 * w5
            - 39.0 * ca2 * v7 * w5
            + 5.0 * ca3 * v7 * w5
            + 16.0 * ca4 * v7 * w5
            + 22.0 * ca2 * v8 * w5
            - 10.0 * ca4 * v8 * w5
            - v7 * w6
            + ca * v7 * w6
            + 9.0 * ca2 * v7 * w6
            - ca3 * v7 * w6
            - 4.0 * ca4 * v7 * w6
            - 10.0 * ca2 * v8 * w6
            + 6.0 * ca4 * v8 * w6
            + 2.0 * ca2 * v8 * w7
            - 2.0 * ca4 * v8 * w7))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v + v * w).powi(4));

    term9 + term10 + term11
}

/// `STRUV12(W,V,X3,S)`, term 12: the pure (no-log) `CF`-polynomial piece.
fn qqbar_to_gg_gluon_frag_term12(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7, v8, v9, v10) = (
        pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8, pre.v9, pre.v10,
    );
    let (w2, w3, w4, w5, w6, w7, w8, w9) = (
        pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8, pre.w9,
    );
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let ca4 = pre.ca4;

    -(2.0
        * cf
        * (9.0 - 9.0 * ca - 7.0 * ca2 + 9.0 * ca3 - 2.0 * ca4 - 48.0 * v
            + 48.0 * ca * v
            + 52.0 * ca2 * v
            - 36.0 * ca3 * v
            - 4.0 * ca4 * v
            + 129.0 * v2
            - 111.0 * ca * v2
            - 187.0 * ca2 * v2
            + 99.0 * ca3 * v2
            + 58.0 * ca4 * v2
            - 222.0 * v3
            + 144.0 * ca * v3
            + 398.0 * ca2 * v3
            - 216.0 * ca3 * v3
            - 176.0 * ca4 * v3
            + 243.0 * v4
            - 111.0 * ca * v4
            - 529.0 * ca2 * v4
            + 279.0 * ca3 * v4
            + 286.0 * ca4 * v4
            - 156.0 * v5
            + 48.0 * ca * v5
            + 448.0 * ca2 * v5
            - 180.0 * ca3 * v5
            - 292.0 * ca4 * v5
            + 51.0 * v6
            - 9.0 * ca * v6
            - 241.0 * ca2 * v6
            + 45.0 * ca3 * v6
            + 190.0 * ca4 * v6
            - 6.0 * v7
            + 78.0 * ca2 * v7
            - 72.0 * ca4 * v7
            - 12.0 * ca2 * v8
            + 12.0 * ca4 * v8
            + 12.0 * ca * w
            - 6.0 * ca2 * w
            - 12.0 * ca3 * w
            + 6.0 * ca4 * w
            + 27.0 * v * w
            - 69.0 * ca * v * w
            + 31.0 * ca2 * v * w
            + 63.0 * ca3 * v * w
            - 46.0 * ca4 * v * w
            - 108.0 * v2 * w
            + 138.0 * ca * v2 * w
            - 31.0 * ca2 * v2 * w
            - 138.0 * ca3 * v2 * w
            + 191.0 * ca4 * v2 * w
            + 165.0 * v3 * w
            - 147.0 * ca * v3 * w
            - 111.0 * ca2 * v3 * w
            + 309.0 * ca3 * v3 * w
            - 448.0 * ca4 * v3 * w
            - 60.0 * v4 * w
            + 120.0 * ca * v4 * w
            + 300.0 * ca2 * v4 * w
            - 480.0 * ca3 * v4 * w
            + 688.0 * ca4 * v4 * w
            - 147.0 * v5 * w
            - 75.0 * ca * v5 * w
            - 239.0 * ca2 * v5 * w
            + 273.0 * ca3 * v5 * w
            - 778.0 * ca4 * v5 * w
            + 204.0 * v6 * w
            + 18.0 * ca * v6 * w
            - 15.0 * ca2 * v6 * w
            + 54.0 * ca3 * v6 * w
            + 643.0 * ca4 * v6 * w
            - 93.0 * v7 * w
            + 3.0 * ca * v7 * w
            + 143.0 * ca2 * v7 * w
            - 69.0 * ca3 * v7 * w
            - 360.0 * ca4 * v7 * w
            + 12.0 * v8 * w
            - 96.0 * ca2 * v8 * w
            + 128.0 * ca4 * v8 * w
            + 24.0 * ca2 * v9 * w
            - 24.0 * ca4 * v9 * w
            + 24.0 * ca * v * w2
            - 12.0 * ca2 * v * w2
            - 24.0 * ca3 * v * w2
            + 12.0 * ca4 * v * w2
            + 9.0 * v2 * w2
            - 45.0 * ca * v2 * w2
            + 73.0 * ca2 * v2 * w2
            + 33.0 * ca3 * v2 * w2
            - 58.0 * ca4 * v2 * w2
            + 72.0 * v3 * w2
            - 24.0 * ca * v3 * w2
            - 165.0 * ca2 * v3 * w2
            - 30.0 * ca3 * v3 * w2
            + 3.0 * ca4 * v3 * w2
            - 378.0 * v4 * w2
            + 18.0 * ca * v4 * w2
            + 270.0 * ca2 * v4 * w2
            + 168.0 * ca3 * v4 * w2
            + 182.0 * ca4 * v4 * w2
            + 666.0 * v5 * w2
            + 108.0 * ca * v5 * w2
            - 534.0 * ca2 * v5 * w2
            - 12.0 * ca3 * v5 * w2
            - 290.0 * ca4 * v5 * w2
            - 519.0 * v6 * w2
            - 105.0 * ca * v6 * w2
            + 807.0 * ca2 * v6 * w2
            - 339.0 * ca3 * v6 * w2
            + 346.0 * ca4 * v6 * w2
            + 132.0 * v7 * w2
            + 36.0 * ca * v7 * w2
            - 687.0 * ca2 * v7 * w2
            + 162.0 * ca3 * v7 * w2
            - 317.0 * ca4 * v7 * w2
            + 24.0 * v8 * w2
            - 12.0 * ca * v8 * w2
            + 302.0 * ca2 * v8 * w2
            + 42.0 * ca3 * v8 * w2
            + 150.0 * ca4 * v8 * w2
            - 6.0 * v9 * w2
            - 42.0 * ca2 * v9 * w2
            - 40.0 * ca4 * v9 * w2
            - 12.0 * ca2 * v10 * w2
            + 12.0 * ca4 * v10 * w2
            - 12.0 * ca * v2 * w3
            + 6.0 * ca2 * v2 * w3
            + 12.0 * ca3 * v2 * w3
            - 6.0 * ca4 * v2 * w3
            - 45.0 * v3 * w3
            + 135.0 * ca * v3 * w3
            - 41.0 * ca2 * v3 * w3
            - 129.0 * ca3 * v3 * w3
            + 74.0 * ca4 * v3 * w3
            + 258.0 * v4 * w3
            - 186.0 * ca * v4 * w3
            + 42.0 * ca2 * v4 * w3
            + 132.0 * ca3 * v4 * w3
            - 166.0 * ca4 * v4 * w3
            - 444.0 * v5 * w3
            - 72.0 * ca * v5 * w3
            + 240.0 * ca2 * v5 * w3
            - 132.0 * ca3 * v5 * w3
            + 422.0 * ca4 * v5 * w3
            + 216.0 * v6 * w3
            + 216.0 * ca * v6 * w3
            - 682.0 * ca2 * v6 * w3
            + 360.0 * ca3 * v6 * w3
            - 704.0 * ca4 * v6 * w3
            + 159.0 * v7 * w3
            - 117.0 * ca * v7 * w3
            + 655.0 * ca2 * v7 * w3
            - 69.0 * ca3 * v7 * w3
            + 518.0 * ca4 * v7 * w3
            - 162.0 * v8 * w3
            + 30.0 * ca * v8 * w3
            - 214.0 * ca2 * v8 * w3
            - 168.0 * ca3 * v8 * w3
            - 136.0 * ca4 * v8 * w3
            + 18.0 * v9 * w3
            + 6.0 * ca * v9 * w3
            - 66.0 * ca2 * v9 * w3
            - 6.0 * ca3 * v9 * w3
            + 14.0 * ca4 * v9 * w3
            + 60.0 * ca2 * v10 * w3
            - 16.0 * ca4 * v10 * w3
            - 48.0 * ca * v3 * w4
            + 24.0 * ca2 * v3 * w4
            + 48.0 * ca3 * v3 * w4
            - 24.0 * ca4 * v3 * w4
            - 45.0 * v4 * w4
            + 117.0 * ca * v4 * w4
            - 125.0 * ca2 * v4 * w4
            - 93.0 * ca3 * v4 * w4
            + 122.0 * ca4 * v4 * w4
            + 66.0 * v5 * w4
            + 48.0 * ca * v5 * w4
            + 120.0 * ca2 * v5 * w4
            - 286.0 * ca4 * v5 * w4
            + 141.0 * v6 * w4
            - 219.0 * ca * v6 * w4
            + 165.0 * ca2 * v6 * w4
            - 93.0 * ca3 * v6 * w4
            - 38.0 * ca4 * v6 * w4
            - 378.0 * v7 * w4
            + 132.0 * ca * v7 * w4
            - 254.0 * ca2 * v7 * w4
            - 108.0 * ca3 * v7 * w4
            + 562.0 * ca4 * v7 * w4
            + 252.0 * v8 * w4
            - 12.0 * ca * v8 * w4
            - 46.0 * ca2 * v8 * w4
            + 276.0 * ca3 * v8 * w4
            - 544.0 * ca4 * v8 * w4
            - 18.0 * v9 * w4
            - 24.0 * ca * v9 * w4
            + 234.0 * ca2 * v9 * w4
            + 24.0 * ca3 * v9 * w4
            + 204.0 * ca4 * v9 * w4
            - 132.0 * ca2 * v10 * w4
            - 12.0 * ca * v4 * w5
            + 6.0 * ca2 * v4 * w5
            + 12.0 * ca3 * v4 * w5
            - 6.0 * ca4 * v4 * w5
            + 9.0 * v5 * w5
            - 63.0 * ca * v5 * w5
            - 11.0 * ca2 * v5 * w5
            + 69.0 * ca3 * v5 * w5
            - 10.0 * ca4 * v5 * w5
            - 120.0 * v6 * w5
            + 150.0 * ca * v6 * w5
            - 87.0 * ca2 * v6 * w5
            - 66.0 * ca3 * v6 * w5
            + 319.0 * ca4 * v6 * w5
            + 261.0 * v7 * w5
            - 75.0 * ca * v7 * w5
            + 129.0 * ca2 * v7 * w5
            + 117.0 * ca3 * v7 * w5
            - 516.0 * ca4 * v7 * w5
            - 204.0 * v8 * w5
            - 24.0 * ca * v8 * w5
            + 94.0 * ca2 * v8 * w5
            - 240.0 * ca3 * v8 * w5
            + 424.0 * ca4 * v8 * w5
            + 12.0 * v9 * w5
            + 36.0 * ca * v9 * w5
            - 276.0 * ca2 * v9 * w5
            - 36.0 * ca3 * v9 * w5
            - 156.0 * ca4 * v9 * w5
            + 180.0 * ca2 * v10 * w5
            - 48.0 * ca4 * v10 * w5
            + 24.0 * ca * v5 * w6
            - 12.0 * ca2 * v5 * w6
            - 24.0 * ca3 * v5 * w6
            + 12.0 * ca4 * v5 * w6
            + 27.0 * v6 * w6
            - 63.0 * ca * v6 * w6
            + 59.0 * ca2 * v6 * w6
            + 51.0 * ca3 * v6 * w6
            - 62.0 * ca4 * v6 * w6
            - 84.0 * v7 * w6
            + 24.0 * ca * v7 * w6
            - 85.0 * ca2 * v7 * w6
            - 30.0 * ca3 * v7 * w6
            + 39.0 * ca4 * v7 * w6
            + 108.0 * v8 * w6
            + 24.0 * ca * v8 * w6
            - 44.0 * ca2 * v8 * w6
            + 114.0 * ca3 * v8 * w6
            - 70.0 * ca4 * v8 * w6
            - 18.0 * v9 * w6
            - 24.0 * ca * v9 * w6
            + 210.0 * ca2 * v9 * w6
            + 24.0 * ca3 * v9 * w6
            - 36.0 * ca4 * v9 * w6
            - 180.0 * ca2 * v10 * w6
            + 136.0 * ca4 * v10 * w6
            + 12.0 * ca * v6 * w7
            - 6.0 * ca2 * v6 * w7
            - 12.0 * ca3 * v6 * w7
            + 6.0 * ca4 * v6 * w7
            + 9.0 * v7 * w7
            - 3.0 * ca * v7 * w7
            + 21.0 * ca2 * v7 * w7
            - 3.0 * ca3 * v7 * w7
            - 18.0 * ca4 * v7 * w7
            - 30.0 * v8 * w7
            - 6.0 * ca * v8 * w7
            + 16.0 * ca2 * v8 * w7
            - 24.0 * ca3 * v8 * w7
            + 36.0 * ca4 * v8 * w7
            + 18.0 * v9 * w7
            + 6.0 * ca * v9 * w7
            - 114.0 * ca2 * v9 * w7
            - 6.0 * ca3 * v9 * w7
            + 62.0 * ca4 * v9 * w7
            + 132.0 * ca2 * v10 * w7
            - 132.0 * ca4 * v10 * w7
            - 6.0 * v9 * w8
            + 30.0 * ca2 * v9 * w8
            - 24.0 * ca4 * v9 * w8
            - 60.0 * ca2 * v10 * w8
            + 60.0 * ca4 * v10 * w8
            + 12.0 * ca2 * v10 * w9
            - 12.0 * ca4 * v10 * w9))
        / (3.0 * ca * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(4))
}

/// `STRUV12(W,V,X3,S)`, term 13: the `lv` piece.
fn qqbar_to_gg_gluon_frag_term13(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7, v8, v9, v10) = (
        pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8, pre.v9, pre.v10,
    );
    let (w2, w3, w4, w5, w6, w7) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7);
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let ca4 = pre.ca4;
    let lv = pre.lv;

    -(2.0
        * cf
        * lv
        * (3.0 + 2.0 * ca - 18.0 * ca2 - 2.0 * ca3 + 15.0 * ca4 - 18.0 * v - 8.0 * ca * v
            + 108.0 * ca2 * v
            - 8.0 * ca3 * v
            - 86.0 * ca4 * v
            + 45.0 * v2
            + 10.0 * ca * v2
            - 288.0 * ca2 * v2
            + 46.0 * ca3 * v2
            + 227.0 * ca4 * v2
            - 62.0 * v3
            + 4.0 * ca * v3
            + 458.0 * ca2 * v3
            - 68.0 * ca3 * v3
            - 368.0 * ca4 * v3
            + 53.0 * v4
            - 26.0 * ca * v4
            - 490.0 * ca2 * v4
            + 42.0 * ca3 * v4
            + 405.0 * ca4 * v4
            - 30.0 * v5
            + 32.0 * ca * v5
            + 368.0 * ca2 * v5
            - 16.0 * ca3 * v5
            - 310.0 * ca4 * v5
            + 11.0 * v6
            - 18.0 * ca * v6
            - 188.0 * ca2 * v6
            + 10.0 * ca3 * v6
            + 161.0 * ca4 * v6
            - 2.0 * v7
            + 4.0 * ca * v7
            + 58.0 * ca2 * v7
            - 4.0 * ca3 * v7
            - 52.0 * ca4 * v7
            - 8.0 * ca2 * v8
            + 8.0 * ca4 * v8
            - w
            - ca * w
            + 2.0 * ca2 * w
            + ca3 * w
            - ca4 * w
            + 12.0 * v * w
            + 9.0 * ca * v * w
            - 50.0 * ca2 * v * w
            - 7.0 * ca3 * v * w
            + 38.0 * ca4 * v * w
            - 33.0 * v2 * w
            - 35.0 * ca * v2 * w
            + 218.0 * ca2 * v2 * w
            - ca3 * v2 * w
            - 221.0 * ca4 * v2 * w
            + 40.0 * v3 * w
            + 51.0 * ca * v3 * w
            - 460.0 * ca2 * v3 * w
            - 17.0 * ca3 * v3 * w
            + 620.0 * ca4 * v3 * w
            - 19.0 * v4 * w
            - 11.0 * ca * v4 * w
            + 586.0 * ca2 * v4 * w
            + 91.0 * ca3 * v4 * w
            - 1007.0 * ca4 * v4 * w
            - 16.0 * v5 * w
            - 29.0 * ca * v5 * w
            - 454.0 * ca2 * v5 * w
            - 85.0 * ca3 * v5 * w
            + 978.0 * ca4 * v5 * w
            + 33.0 * v6 * w
            + 7.0 * ca * v6 * w
            + 154.0 * ca2 * v6 * w
            + 13.0 * ca3 * v6 * w
            - 531.0 * ca4 * v6 * w
            - 20.0 * v7 * w
            + 17.0 * ca * v7 * w
            + 52.0 * ca2 * v7 * w
            - 3.0 * ca3 * v7 * w
            + 108.0 * ca4 * v7 * w
            + 4.0 * v8 * w
            - 8.0 * ca * v8 * w
            - 64.0 * ca2 * v8 * w
            + 8.0 * ca3 * v8 * w
            + 32.0 * ca4 * v8 * w
            + 16.0 * ca2 * v9 * w
            - 16.0 * ca4 * v9 * w
            - 2.0 * v * w2
            - 2.0 * ca * v * w2
            + 4.0 * ca2 * v * w2
            + 2.0 * ca3 * v * w2
            - 2.0 * ca4 * v * w2
            + 5.0 * v2 * w2
            + 4.0 * ca * v2 * w2
            - 2.0 * ca2 * v2 * w2
            - 3.0 * ca4 * v2 * w2
            + 4.0 * v3 * w2
            + 10.0 * ca * v3 * w2
            - 64.0 * ca2 * v3 * w2
            + 20.0 * ca3 * v3 * w2
            + 68.0 * ca4 * v3 * w2
            - 30.0 * v4 * w2
            - 78.0 * ca * v4 * w2
            + 240.0 * ca2 * v4 * w2
            - 116.0 * ca3 * v4 * w2
            - 318.0 * ca4 * v4 * w2
            + 80.0 * v5 * w2
            + 98.0 * ca * v5 * w2
            - 518.0 * ca2 * v5 * w2
            + 102.0 * ca3 * v5 * w2
            + 830.0 * ca4 * v5 * w2
            - 95.0 * v6 * w2
            + 20.0 * ca * v6 * w2
            + 758.0 * ca2 * v6 * w2
            + 16.0 * ca3 * v6 * w2
            - 1259.0 * ca4 * v6 * w2
            + 32.0 * v7 * w2
            - 78.0 * ca * v7 * w2
            - 676.0 * ca2 * v7 * w2
            + 8.0 * ca3 * v7 * w2
            + 1100.0 * ca4 * v7 * w2
            + 8.0 * v8 * w2
            + 22.0 * ca * v8 * w2
            + 308.0 * ca2 * v8 * w2
            - 28.0 * ca3 * v8 * w2
            - 516.0 * ca4 * v8 * w2
            - 2.0 * v9 * w2
            + 4.0 * ca * v9 * w2
            - 42.0 * ca2 * v9 * w2
            - 4.0 * ca3 * v9 * w2
            + 92.0 * ca4 * v9 * w2
            - 8.0 * ca2 * v10 * w2
            + 8.0 * ca4 * v10 * w2
            + v2 * w3
            + ca * v2 * w3
            - 2.0 * ca2 * v2 * w3
            - ca3 * v2 * w3
            + ca4 * v2 * w3
            - 22.0 * v3 * w3
            - 17.0 * ca * v3 * w3
            + 94.0 * ca2 * v3 * w3
            + 15.0 * ca3 * v3 * w3
            - 72.0 * ca4 * v3 * w3
            + 56.0 * v4 * w3
            + 84.0 * ca * v4 * w3
            - 354.0 * ca2 * v4 * w3
            + 10.0 * ca3 * v4 * w3
            + 410.0 * ca4 * v4 * w3
            + 102.0 * v5 * w3
            - 98.0 * ca * v5 * w3
            + 722.0 * ca2 * v5 * w3
            + 6.0 * ca3 * v5 * w3
            - 1140.0 * ca4 * v5 * w3
            + 93.0 * v6 * w3
            - 83.0 * ca * v6 * w3
            - 998.0 * ca2 * v6 * w3
            - 93.0 * ca3 * v6 * w3
            + 1693.0 * ca4 * v6 * w3
            + 30.0 * v7 * w3
            + 147.0 * ca * v7 * w3
            + 818.0 * ca2 * v7 * w3
            - 5.0 * ca3 * v7 * w3
            - 1320.0 * ca4 * v7 * w3
            - 58.0 * v8 * w3
            - 14.0 * ca * v8 * w3
            - 266.0 * ca2 * v8 * w3
            + 48.0 * ca3 * v8 * w3
            + 436.0 * ca4 * v8 * w3
            + 2.0 * v9 * w3
            - 20.0 * ca * v9 * w3
            - 62.0 * ca2 * v9 * w3
            + 20.0 * ca3 * v9 * w3
            + 64.0 * ca4 * v9 * w3
            + 48.0 * ca2 * v10 * w3
            - 72.0 * ca4 * v10 * w3
            + 4.0 * v3 * w4
            + 4.0 * ca * v3 * w4
            - 8.0 * ca2 * v3 * w4
            - 4.0 * ca3 * v3 * w4
            + 4.0 * ca4 * v3 * w4
            - 19.0 * v4 * w4
            - 14.0 * ca * v4 * w4
            + 58.0 * ca2 * v4 * w4
            + 6.0 * ca3 * v4 * w4
            - 39.0 * ca4 * v4 * w4
            + 40.0 * v5 * w4
            + 4.0 * ca * v5 * w4
            - 130.0 * ca2 * v5 * w4
            - 8.0 * ca3 * v5 * w4
            + 150.0 * ca4 * v5 * w4
            - 19.0 * v6 * w4
            + 138.0 * ca * v6 * w4
            + 208.0 * ca2 * v6 * w4
            + 58.0 * ca3 * v6 * w4
            - 125.0 * ca4 * v6 * w4
            - 114.0 * v7 * w4
            - 164.0 * ca * v7 * w4
            - 98.0 * ca2 * v7 * w4
            + 28.0 * ca3 * v7 * w4
            - 260.0 * ca4 * v7 * w4
            + 100.0 * v8 * w4
            - 24.0 * ca * v8 * w4
            - 220.0 * ca2 * v8 * w4
            - 56.0 * ca3 * v8 * w4
            + 596.0 * ca4 * v8 * w4
            + 14.0 * v9 * w4
            + 44.0 * ca * v9 * w4
            + 302.0 * ca2 * v9 * w4
            - 44.0 * ca3 * v9 * w4
            - 504.0 * ca4 * v9 * w4
            - 112.0 * ca2 * v10 * w4
            + 184.0 * ca4 * v10 * w4
            + v4 * w5
            + ca * v4 * w5
            - 2.0 * ca2 * v4 * w5
            - ca3 * v4 * w5
            + ca4 * v4 * w5
            + 8.0 * v5 * w5
            + 7.0 * ca * v5 * w5
            - 38.0 * ca2 * v5 * w5
            - 9.0 * ca3 * v5 * w5
            + 30.0 * ca4 * v5 * w5
            - 29.0 * v6 * w5
            - 71.0 * ca * v6 * w5
            + 118.0 * ca2 * v6 * w5
            - ca3 * v6 * w5
            - 221.0 * ca4 * v6 * w5
            + 112.0 * v7 * w5
            + 91.0 * ca * v7 * w5
            - 232.0 * ca2 * v7 * w5
            - 33.0 * ca3 * v7 * w5
            + 576.0 * ca4 * v7 * w5
            - 80.0 * v8 * w5
            + 56.0 * ca * v8 * w5
            + 400.0 * ca2 * v8 * w5
            + 40.0 * ca3 * v8 * w5
            - 740.0 * ca4 * v8 * w5
            - 40.0 * v9 * w5
            - 56.0 * ca * v9 * w5
            - 372.0 * ca2 * v9 * w5
            + 56.0 * ca3 * v9 * w5
            + 528.0 * ca4 * v9 * w5
            + 136.0 * ca2 * v10 * w5
            - 200.0 * ca4 * v10 * w5
            - 2.0 * v5 * w6
            - 2.0 * ca * v5 * w6
            + 4.0 * ca2 * v5 * w6
            + 2.0 * ca3 * v5 * w6
            - 2.0 * ca4 * v5 * w6
            + 11.0 * v6 * w6
            + 8.0 * ca * v6 * w6
            - 38.0 * ca2 * v6 * w6
            - 4.0 * ca3 * v6 * w6
            + 27.0 * ca4 * v6 * w6
            - 40.0 * v7 * w6
            - 18.0 * ca * v7 * w6
            + 84.0 * ca2 * v7 * w6
            + 8.0 * ca3 * v7 * w6
            - 108.0 * ca4 * v7 * w6
            + 28.0 * v8 * w6
            - 46.0 * ca * v8 * w6
            - 176.0 * ca2 * v8 * w6
            - 12.0 * ca3 * v8 * w6
            + 184.0 * ca4 * v8 * w6
            + 46.0 * v9 * w6
            + 44.0 * ca * v9 * w6
            + 218.0 * ca2 * v9 * w6
            - 44.0 * ca3 * v9 * w6
            - 164.0 * ca4 * v9 * w6
            - 104.0 * ca2 * v10 * w6
            + 104.0 * ca4 * v10 * w6
            - v6 * w7
            - ca * v6 * w7
            + 2.0 * ca2 * v6 * w7
            + ca3 * v6 * w7
            - ca4 * v6 * w7
            + 2.0 * v7 * w7
            + ca * v7 * w7
            - 6.0 * ca2 * v7 * w7
            + ca3 * v7 * w7
            + 4.0 * ca4 * v7 * w7
            - 2.0 * v8 * w7
            + 14.0 * ca * v8 * w7
            + 26.0 * ca2 * v8 * w7
            - 26.0 * v9 * w7
            - 20.0 * ca * v9 * w7
            - 78.0 * ca2 * v9 * w7
            + 20.0 * ca3 * v9 * w7
            + 64.0 * ca2 * v10 * w7
            - 40.0 * ca4 * v10 * w7
            + 6.0 * v9 * pre.w8
            + 4.0 * ca * v9 * pre.w8
            + 18.0 * ca2 * v9 * pre.w8
            - 4.0 * ca3 * v9 * pre.w8
            - 32.0 * ca2 * v10 * pre.w8
            + 24.0 * ca4 * v10 * pre.w8
            + 8.0 * ca2 * v10 * pre.w9
            - 8.0 * ca4 * v10 * pre.w9))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(4))
}

/// `STRUV12(W,V,X3,S)`, term 14: the `l1w` piece.
fn qqbar_to_gg_gluon_frag_term14(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7, v8, v9, v10) = (
        pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8, pre.v9, pre.v10,
    );
    let (w2, w3, w4, w5, w6, w7) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7);
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let ca4 = pre.ca4;
    let l1w = pre.l1w;

    -(2.0
        * cf
        * l1w
        * (5.0 + 2.0 * ca - 16.0 * ca2 - 2.0 * ca3 + 11.0 * ca4 - 26.0 * v - 12.0 * ca * v
            + 98.0 * ca2 * v
            - 4.0 * ca3 * v
            - 64.0 * ca4 * v
            + 59.0 * v2
            + 22.0 * ca * v2
            - 266.0 * ca2 * v2
            + 30.0 * ca3 * v2
            + 173.0 * ca4 * v2
            - 74.0 * v3
            - 4.0 * ca * v3
            + 432.0 * ca2 * v3
            - 44.0 * ca3 * v3
            - 290.0 * ca4 * v3
            + 51.0 * v4
            - 34.0 * ca * v4
            - 476.0 * ca2 * v4
            + 26.0 * ca3 * v4
            + 333.0 * ca4 * v4
            - 14.0 * v5
            + 44.0 * ca * v5
            + 370.0 * ca2 * v5
            - 12.0 * ca3 * v5
            - 268.0 * ca4 * v5
            - 3.0 * v6
            - 22.0 * ca * v6
            - 194.0 * ca2 * v6
            + 10.0 * ca3 * v6
            + 147.0 * ca4 * v6
            + 2.0 * v7
            + 4.0 * ca * v7
            + 60.0 * ca2 * v7
            - 4.0 * ca3 * v7
            - 50.0 * ca4 * v7
            - 8.0 * ca2 * v8
            + 8.0 * ca4 * v8
            - w
            - ca * w
            + 2.0 * ca2 * w
            + ca3 * w
            - ca4 * w
            + 14.0 * v * w
            + 9.0 * ca * v * w
            - 48.0 * ca2 * v * w
            - 7.0 * ca3 * v * w
            + 30.0 * ca4 * v * w
            - 35.0 * v2 * w
            - 31.0 * ca * v2 * w
            + 222.0 * ca2 * v2 * w
            - ca3 * v2 * w
            - 175.0 * ca4 * v2 * w
            + 32.0 * v3 * w
            + 27.0 * ca * v3 * w
            - 510.0 * ca2 * v3 * w
            - 9.0 * ca3 * v3 * w
            + 500.0 * ca4 * v3 * w
            + 9.0 * v4 * w
            + 29.0 * ca * v4 * w
            + 726.0 * ca2 * v4 * w
            + 67.0 * ca3 * v4 * w
            - 835.0 * ca4 * v4 * w
            - 54.0 * v5 * w
            - 45.0 * ca * v5 * w
            - 648.0 * ca2 * v5 * w
            - 61.0 * ca3 * v5 * w
            + 846.0 * ca4 * v5 * w
            + 47.0 * v6 * w
            - 5.0 * ca * v6 * w
            + 294.0 * ca2 * v6 * w
            + 5.0 * ca3 * v6 * w
            - 489.0 * ca4 * v6 * w
            - 8.0 * v7 * w
            + 25.0 * ca * v7 * w
            + 6.0 * ca2 * v7 * w
            - 3.0 * ca3 * v7 * w
            + 112.0 * ca4 * v7 * w
            - 4.0 * v8 * w
            - 8.0 * ca * v8 * w
            - 60.0 * ca2 * v8 * w
            + 8.0 * ca3 * v8 * w
            + 28.0 * ca4 * v8 * w
            + 16.0 * ca2 * v9 * w
            - 16.0 * ca4 * v9 * w
            - 2.0 * v * w2
            - 2.0 * ca * v * w2
            + 4.0 * ca2 * v * w2
            + 2.0 * ca3 * v * w2
            - 2.0 * ca4 * v * w2
            - 5.0 * v2 * w2
            + 4.0 * ca * v2 * w2
            - 8.0 * ca2 * v2 * w2
            + ca4 * v2 * w2
            + 52.0 * v3 * w2
            + 22.0 * ca * v3 * w2
            - 28.0 * ca2 * v3 * w2
            + 8.0 * ca3 * v3 * w2
            + 44.0 * ca4 * v3 * w2
            - 122.0 * v4 * w2
            - 90.0 * ca * v4 * w2
            + 160.0 * ca2 * v4 * w2
            - 84.0 * ca3 * v4 * w2
            - 232.0 * ca4 * v4 * w2
            + 164.0 * v5 * w2
            + 66.0 * ca * v5 * w2
            - 452.0 * ca2 * v5 * w2
            + 78.0 * ca3 * v5 * w2
            + 636.0 * ca4 * v5 * w2
            - 105.0 * v6 * w2
            + 68.0 * ca * v6 * w2
            + 808.0 * ca2 * v6 * w2
            + 16.0 * ca3 * v6 * w2
            - 1015.0 * ca4 * v6 * w2
            - 8.0 * v7 * w2
            - 90.0 * ca * v7 * w2
            - 812.0 * ca2 * v7 * w2
            + 12.0 * ca3 * v7 * w2
            + 948.0 * ca4 * v7 * w2
            + 24.0 * v8 * w2
            + 18.0 * ca * v8 * w2
            + 392.0 * ca2 * v8 * w2
            - 28.0 * ca3 * v8 * w2
            - 482.0 * ca4 * v8 * w2
            + 2.0 * v9 * w2
            + 4.0 * ca * v9 * w2
            - 56.0 * ca2 * v9 * w2
            - 4.0 * ca3 * v9 * w2
            + 94.0 * ca4 * v9 * w2
            - 8.0 * ca2 * v10 * w2
            + 8.0 * ca4 * v10 * w2
            + v2 * w3
            + ca * v2 * w3
            - 2.0 * ca2 * v2 * w3
            - ca3 * v2 * w3
            + ca4 * v2 * w3
            - 40.0 * v3 * w3
            - 17.0 * ca * v3 * w3
            + 84.0 * ca2 * v3 * w3
            + 15.0 * ca3 * v3 * w3
            - 56.0 * ca4 * v3 * w3
            + 106.0 * v4 * w3
            + 72.0 * ca * v4 * w3
            - 346.0 * ca2 * v4 * w3
            + 10.0 * ca3 * v4 * w3
            + 312.0 * ca4 * v4 * w3
            - 136.0 * v5 * w3
            - 50.0 * ca * v5 * w3
            + 798.0 * ca2 * v5 * w3
            - 10.0 * ca3 * v5 * w3
            - 892.0 * ca4 * v5 * w3
            + 57.0 * v6 * w3
            - 115.0 * ca * v6 * w3
            - 1230.0 * ca2 * v6 * w3
            - 69.0 * ca3 * v6 * w3
            + 1393.0 * ca4 * v6 * w3
            + 88.0 * v7 * w3
            + 131.0 * ca * v7 * w3
            + 1096.0 * ca2 * v7 * w3
            - 13.0 * ca3 * v7 * w3
            - 1172.0 * ca4 * v7 * w3
            - 64.0 * v8 * w3
            - 2.0 * ca * v8 * w3
            - 386.0 * ca2 * v8 * w3
            + 48.0 * ca3 * v8 * w3
            + 446.0 * ca4 * v8 * w3
            - 12.0 * v9 * w3
            - 20.0 * ca * v9 * w3
            - 70.0 * ca2 * v9 * w3
            + 20.0 * ca3 * v9 * w3
            + 40.0 * ca4 * v9 * w3
            + 56.0 * ca2 * v10 * w3
            - 72.0 * ca4 * v10 * w3
            + 4.0 * v3 * w4
            + 4.0 * ca * v3 * w4
            - 8.0 * ca2 * v3 * w4
            - 4.0 * ca3 * v3 * w4
            + 4.0 * ca4 * v3 * w4
            - 21.0 * v4 * w4
            - 14.0 * ca * v4 * w4
            + 52.0 * ca2 * v4 * w4
            + 6.0 * ca3 * v4 * w4
            - 35.0 * ca4 * v4 * w4
            + 12.0 * v5 * w4
            - 8.0 * ca * v5 * w4
            - 134.0 * ca2 * v5 * w4
            + 4.0 * ca3 * v5 * w4
            + 130.0 * ca4 * v5 * w4
            + 59.0 * v6 * w4
            + 126.0 * ca * v6 * w4
            + 246.0 * ca2 * v6 * w4
            + 42.0 * ca3 * v6 * w4
            - 135.0 * ca4 * v6 * w4
            - 170.0 * v7 * w4
            - 124.0 * ca * v7 * w4
            - 128.0 * ca2 * v7 * w4
            + 28.0 * ca3 * v7 * w4
            - 150.0 * ca4 * v7 * w4
            + 92.0 * v8 * w4
            - 32.0 * ca * v8 * w4
            - 288.0 * ca2 * v8 * w4
            - 56.0 * ca3 * v8 * w4
            + 456.0 * ca4 * v8 * w4
            + 30.0 * v9 * w4
            + 44.0 * ca * v9 * w4
            + 400.0 * ca2 * v9 * w4
            - 44.0 * ca3 * v9 * w4
            - 446.0 * ca4 * v9 * w4
            - 136.0 * ca2 * v10 * w4
            + 184.0 * ca4 * v10 * w4
            + v4 * w5
            + ca * v4 * w5
            - 2.0 * ca2 * v4 * w5
            - ca3 * v4 * w5
            + ca4 * v4 * w5
            + 22.0 * v5 * w5
            + 7.0 * ca * v5 * w5
            - 40.0 * ca2 * v5 * w5
            - 9.0 * ca3 * v5 * w5
            + 22.0 * ca4 * v5 * w5
            - 75.0 * v6 * w5
            - 59.0 * ca * v6 * w5
            + 138.0 * ca2 * v6 * w5
            - ca3 * v6 * w5
            - 163.0 * ca4 * v6 * w5
            + 140.0 * v7 * w5
            + 67.0 * ca * v7 * w5
            - 298.0 * ca2 * v7 * w5
            - 25.0 * ca3 * v7 * w5
            + 440.0 * ca4 * v7 * w5
            - 72.0 * v8 * w5
            + 48.0 * ca * v8 * w5
            + 516.0 * ca2 * v8 * w5
            + 40.0 * ca3 * v8 * w5
            - 612.0 * ca4 * v8 * w5
            - 40.0 * v9 * w5
            - 56.0 * ca * v9 * w5
            - 468.0 * ca2 * v9 * w5
            + 56.0 * ca3 * v9 * w5
            + 480.0 * ca4 * v9 * w5
            + 152.0 * ca2 * v10 * w5
            - 200.0 * ca4 * v10 * w5
            - 2.0 * v5 * w6
            - 2.0 * ca * v5 * w6
            + 4.0 * ca2 * v5 * w6
            + 2.0 * ca3 * v5 * w6
            - 2.0 * ca4 * v5 * w6
            + 21.0 * v6 * w6
            + 8.0 * ca * v6 * w6
            - 32.0 * ca2 * v6 * w6
            - 4.0 * ca3 * v6 * w6
            + 23.0 * ca4 * v6 * w6
            - 48.0 * v7 * w6
            - 14.0 * ca * v7 * w6
            + 72.0 * ca2 * v7 * w6
            + 4.0 * ca3 * v7 * w6
            - 84.0 * ca4 * v7 * w6
            + 28.0 * v8 * w6
            - 34.0 * ca * v8 * w6
            - 164.0 * ca2 * v8 * w6
            - 12.0 * ca3 * v8 * w6
            + 162.0 * ca4 * v8 * w6
            + 30.0 * v9 * w6
            + 44.0 * ca * v9 * w6
            + 208.0 * ca2 * v9 * w6
            - 44.0 * ca3 * v9 * w6
            - 158.0 * ca4 * v9 * w6
            - 88.0 * ca2 * v10 * w6
            + 104.0 * ca4 * v10 * w6
            - v6 * w7
            - ca * v6 * w7
            + 2.0 * ca2 * v6 * w7
            + ca3 * v6 * w7
            - ca4 * v6 * w7
            + 4.0 * v7 * w7
            + ca * v7 * w7
            + 4.0 * ca2 * v7 * w7
            + ca3 * v7 * w7
            + 4.0 * ca4 * v7 * w7
            - 4.0 * v8 * w7
            + 10.0 * ca * v8 * w7
            - 6.0 * ca2 * v8 * w7
            - 6.0 * ca4 * v8 * w7
            - 12.0 * v9 * w7
            - 20.0 * ca * v9 * w7
            - 38.0 * ca2 * v9 * w7
            + 20.0 * ca3 * v9 * w7
            + 8.0 * ca4 * v9 * w7
            + 40.0 * ca2 * v10 * w7
            - 40.0 * ca4 * v10 * w7
            + 4.0 * ca2 * v8 * pre.w8
            + 2.0 * v9 * pre.w8
            + 4.0 * ca * v9 * pre.w8
            + 8.0 * ca2 * v9 * pre.w8
            - 4.0 * ca3 * v9 * pre.w8
            - 2.0 * ca4 * v9 * pre.w8
            - 24.0 * ca2 * v10 * pre.w8
            + 24.0 * ca4 * v10 * pre.w8
            + 8.0 * ca2 * v10 * pre.w9
            - 8.0 * ca4 * v10 * pre.w9))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(4))
}

/// `STRUV12(W,V,X3,S)`, part D: the pure/`lv`/`l1w` term groups over
/// `(1-vw)^2*(1-v+vw)^4`.
fn qqbar_to_gg_gluon_frag_part_d(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    qqbar_to_gg_gluon_frag_term12(w, v, ctx, pre)
        + qqbar_to_gg_gluon_frag_term13(w, v, ctx, pre)
        + qqbar_to_gg_gluon_frag_term14(w, v, ctx, pre)
}

/// `STRUV12(W,V,X3,S)`.
#[must_use]
pub fn qqbar_to_gg_gluon_frag(
    w: f64,
    v: f64,
    _x3: f64,
    _s: f64,
    ctx: &MeContext,
    pre: &Precalc,
) -> f64 {
    qqbar_to_gg_gluon_frag_part_a(w, v, ctx, pre)
        + qqbar_to_gg_gluon_frag_part_b(w, v, ctx, pre)
        + qqbar_to_gg_gluon_frag_part_c(w, v, ctx, pre)
        + qqbar_to_gg_gluon_frag_part_d(w, v, ctx, pre)
}

/// `STRUV13(W,V,X3,S)`, part A: the three `Nf`-proportional terms sharing
/// `(1+VW)(1-2V+VW)/((1-V)(1-VW)^2)` (identical bracket in the Fortran
/// source, factored to avoid retyping), plus the `Nf` polynomial term.
fn qg_compton_quark_frag_part_a(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let cf = ctx.cf;
    let ca = ctx.ca;
    let nf = ctx.nf;
    let (v2, v3, v4, v5) = (pre.v2, pre.v3, pre.v4, pre.v5);
    let (w2, w3) = (pre.w2, pre.w3);
    let ca2 = pre.ca2;
    let cacf2 = ca * cf.powi(2);
    let (l1w, lms, lv) = (pre.l1w, pre.lms, pre.lv);

    let common = 8.0 * cacf2 * nf * (1.0 + v * w) * (1.0 - 2.0 * v + v * w)
        / ((1.0 - v) * (1.0 - v * w).powi(2));

    let terms123 = common * (l1w - lms + lv);

    let term4 = -(4.0
        * cf
        * nf
        * (-2.0 * ca * cf - 3.0 * v + 3.0 * ca2 * v + 2.0 * v2 - 2.0 * ca2 * v2 - 2.0 * v * w
            + 2.0 * ca2 * v * w
            + 2.0 * v2 * w
            + 9.0 * ca2 * v2 * w
            + 9.0 * v3 * w
            - 18.0 * ca2 * v3 * w
            + v2 * w2
            - ca2 * v2 * w2
            - 13.0 * v3 * w2
            + 18.0 * ca2 * v3 * w2
            + 12.0 * v4 * w2
            - 21.0 * ca2 * v4 * w2
            - 10.0 * v4 * w3
            + 12.0 * ca2 * v4 * w3
            + v5 * w3
            - ca2 * v5 * w3))
        / (3.0 * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2));

    terms123 + term4
}

/// `STRUV13(W,V,X3,S)`, part B: the `lvw`/`l1v`/`l1vw`/`lmss` term groups.
fn qg_compton_quark_frag_part_b(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6);
    let (w2, w3, w4, w5) = (pre.w2, pre.w3, pre.w4, pre.w5);
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let ca4 = pre.ca4;
    let cacf2 = ca * cf.powi(2);
    let ca3cf = ca3 * cf;
    let ca2cf2 = ca2 * cf.powi(2);
    let (lvw, l1v, l1vw, lmss) = (pre.lvw, pre.l1v, pre.l1vw, pre.lmss);

    let term5 = (4.0
        * cf
        * lvw
        * (1.0 - 4.0 * ca2 + 3.0 * ca4 - 2.0 * v + 2.0 * ca3 * v - ca4 * v + 2.0 * ca * v2
            - 2.0 * ca2 * v2
            - 4.0 * ca3 * v2
            + 5.0 * ca4 * v2
            + v * w
            + 5.0 * ca2 * v * w
            - 2.0 * ca3 * v * w
            + 4.0 * ca4 * v * w
            - 2.0 * ca * v2 * w
            + 4.0 * ca2 * v2 * w
            + 6.0 * ca3 * v2 * w
            - 10.0 * ca4 * v2 * w
            - 2.0 * ca * v3 * w
            - 2.0 * ca2 * v3 * w
            + 5.0 * ca4 * v3 * w
            - v2 * w2
            + 3.0 * ca2 * v2 * w2
            - 2.0 * ca3 * v2 * w2
            + 8.0 * ca4 * v2 * w2
            + 2.0 * v3 * w2
            + 4.0 * ca2 * v3 * w2
            + 2.0 * ca3 * v3 * w2
            - 9.0 * ca4 * v3 * w2
            + 4.0 * ca * v4 * w2
            - 4.0 * ca3 * v4 * w2
            - v3 * w3
            + 2.0 * ca * v3 * w3
            - 6.0 * ca2 * v3 * w3
            - 2.0 * ca3 * v3 * w3
            + 7.0 * ca4 * v3 * w3
            - 6.0 * ca * v4 * w3
            + 6.0 * ca3 * v4 * w3
            + 2.0 * ca * v4 * w4
            - 2.0 * ca3 * v4 * w4))
        / (ca * (1.0 - v) * v2 * w);

    let term6 = (4.0
        * cf
        * l1v
        * (1.0 + 3.0 * ca2 - 2.0 * ca4 - 4.0 * v - 2.0 * ca * v - 7.0 * ca2 * v - 2.0 * ca3 * v
            + 5.0 * ca4 * v
            + 5.0 * v2
            + 8.0 * ca * v2
            + 6.0 * ca2 * v2
            + 6.0 * ca3 * v2
            - 6.0 * ca4 * v2
            - 2.0 * v3
            - 10.0 * ca * v3
            - 2.0 * ca2 * v3
            - 6.0 * ca3 * v3
            + 4.0 * ca4 * v3
            + 4.0 * ca * v4
            + 2.0 * ca3 * v4
            - ca4 * v4
            + v * w
            + ca2 * v * w
            - v2 * w
            + 2.0 * ca * v2 * w
            + 8.0 * ca2 * v2 * w
            - 2.0 * ca3 * v2 * w
            + 5.0 * ca4 * v2 * w
            - 2.0 * v3 * w
            - 14.0 * ca * v3 * w
            - 12.0 * ca2 * v3 * w
            + 2.0 * ca3 * v3 * w
            - 7.0 * ca4 * v3 * w
            + 2.0 * v4 * w
            + 24.0 * ca * v4 * w
            + 4.0 * ca2 * v4 * w
            + 2.0 * ca3 * v4 * w
            + 2.0 * ca4 * v4 * w
            - 12.0 * ca * v5 * w
            - 2.0 * ca3 * v5 * w
            - ca4 * v5 * w
            - v2 * w2
            - 3.0 * ca2 * v2 * w2
            + 2.0 * ca4 * v2 * w2
            + 10.0 * ca * v3 * w2
            + 2.0 * ca2 * v3 * w2
            + 4.0 * ca4 * v3 * w2
            - 28.0 * ca * v4 * w2
            + 6.0 * ca2 * v4 * w2
            + 2.0 * ca3 * v4 * w2
            - ca4 * v4 * w2
            + 18.0 * ca * v5 * w2
            - 2.0 * ca2 * v5 * w2
            - 6.0 * ca3 * v5 * w2
            + 2.0 * ca4 * v5 * w2
            + 4.0 * ca3 * v6 * w2
            - v3 * w3
            - ca2 * v3 * w3
            + v4 * w3
            + 6.0 * ca * v4 * w3
            - 4.0 * ca2 * v4 * w3
            - 2.0 * ca3 * v4 * w3
            - ca4 * v4 * w3
            - v5 * w3
            - 6.0 * ca * v5 * w3
            + 8.0 * ca3 * v5 * w3
            - 6.0 * ca3 * v6 * w3
            + ca2 * v5 * w4
            - 2.0 * ca3 * v5 * w4
            - ca4 * v5 * w4
            + 2.0 * ca3 * v6 * w4))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v * w) * (1.0 - v + v * w));

    let term7 = -(4.0
        * cf
        * l1vw
        * (2.0 - 4.0 * ca2 + 2.0 * ca4 - 8.0 * ca2cf2 - 8.0 * v - 2.0 * ca * v + 15.0 * ca2 * v
            - 8.0 * ca4 * v
            - 8.0 * ca * cf * v
            + 8.0 * ca3cf * v
            + 16.0 * ca2cf2 * v
            + 12.0 * v2
            + 10.0 * ca * v2
            - 23.0 * ca2 * v2
            - 2.0 * ca3 * v2
            + 13.0 * ca4 * v2
            + 20.0 * ca * cf * v2
            - 20.0 * ca3cf * v2
            - 12.0 * ca2cf2 * v2
            - 8.0 * v3
            - 18.0 * ca * v3
            + 19.0 * ca2 * v3
            + 6.0 * ca3 * v3
            - 11.0 * ca4 * v3
            - 20.0 * ca * cf * v3
            + 20.0 * ca3cf * v3
            + 4.0 * ca2cf2 * v3
            + 2.0 * v4
            + 14.0 * ca * v4
            - 9.0 * ca2 * v4
            - 6.0 * ca3 * v4
            + 5.0 * ca4 * v4
            + 10.0 * ca * cf * v4
            - 10.0 * ca3cf * v4
            - 4.0 * ca * v5
            + 2.0 * ca2 * v5
            + 2.0 * ca3 * v5
            - ca4 * v5
            - 2.0 * ca * cf * v5
            + 2.0 * ca3cf * v5
            + 8.0 * v * w
            + 2.0 * ca * v * w
            - 13.0 * ca2 * v * w
            + 10.0 * ca4 * v * w
            - 8.0 * ca3cf * v * w
            - 24.0 * ca2cf2 * v * w
            - 27.0 * v2 * w
            - 15.0 * ca * v2 * w
            + 38.0 * ca2 * v2 * w
            + 3.0 * ca3 * v2 * w
            - 35.0 * ca4 * v2 * w
            - 24.0 * ca * cf * v2 * w
            + 48.0 * ca3cf * v2 * w
            + 32.0 * ca2cf2 * v2 * w
            + 32.0 * v3 * w
            + 36.0 * ca * v3 * w
            - 48.0 * ca2 * v3 * w
            - 10.0 * ca3 * v3 * w
            + 47.0 * ca4 * v3 * w
            + 44.0 * ca * cf * v3 * w
            - 72.0 * ca3cf * v3 * w
            - 16.0 * ca2cf2 * v3 * w
            - 15.0 * v4 * w
            - 35.0 * ca * v4 * w
            + 34.0 * ca2 * v4 * w
            + 11.0 * ca3 * v4 * w
            - 29.0 * ca4 * v4 * w
            - 32.0 * ca * cf * v4 * w
            + 48.0 * ca3cf * v4 * w
            + 2.0 * v5 * w
            + 12.0 * ca * v5 * w
            - 11.0 * ca2 * v5 * w
            - 4.0 * ca3 * v5 * w
            + 7.0 * ca4 * v5 * w
            + 8.0 * ca * cf * v5 * w
            - 12.0 * ca3cf * v5 * w
            + 10.0 * v2 * w2
            + 5.0 * ca * v2 * w2
            - 16.0 * ca2 * v2 * w2
            - ca3 * v2 * w2
            + 23.0 * ca4 * v2 * w2
            - 32.0 * ca3cf * v2 * w2
            - 20.0 * ca2cf2 * v2 * w2
            - 24.0 * v3 * w2
            - 22.0 * ca * v3 * w2
            + 38.0 * ca2 * v3 * w2
            + 4.0 * ca3 * v3 * w2
            - 64.0 * ca4 * v3 * w2
            - 20.0 * ca * cf * v3 * w2
            + 92.0 * ca3cf * v3 * w2
            + 20.0 * ca2cf2 * v3 * w2
            + 16.0 * v4 * w2
            + 31.0 * ca * v4 * w2
            - 43.0 * ca2 * v4 * w2
            - 5.0 * ca3 * v4 * w2
            + 60.0 * ca4 * v4 * w2
            + 30.0 * ca * cf * v4 * w2
            - 90.0 * ca3cf * v4 * w2
            - 2.0 * v5 * w2
            - 14.0 * ca * v5 * w2
            + 21.0 * ca2 * v5 * w2
            + 2.0 * ca3 * v5 * w2
            - 19.0 * ca4 * v5 * w2
            - 10.0 * ca * cf * v5 * w2
            + 30.0 * ca3cf * v5 * w2
            + 3.0 * v3 * w3
            + 4.0 * ca * v3 * w3
            - 13.0 * ca2 * v3 * w3
            + 29.0 * ca4 * v3 * w3
            - 44.0 * ca3cf * v3 * w3
            - 8.0 * ca2cf2 * v3 * w3
            - v4 * w3
            - 12.0 * ca * v4 * w3
            + 28.0 * ca2 * v4 * w3
            - 55.0 * ca4 * v4 * w3
            - 8.0 * ca * cf * v4 * w3
            + 80.0 * ca3cf * v4 * w3
            - 3.0 * v5 * w3
            + 8.0 * ca * v5 * w3
            - 21.0 * ca2 * v5 * w3
            + 26.0 * ca4 * v5 * w3
            + 4.0 * ca * cf * v5 * w3
            - 40.0 * ca3cf * v5 * w3
            - 2.0 * v4 * w4
            + 2.0 * ca * v4 * w4
            - 10.0 * ca2 * v4 * w4
            + 19.0 * ca4 * v4 * w4
            - 28.0 * ca3cf * v4 * w4
            + 4.0 * v5 * w4
            - 2.0 * ca * v5 * w4
            + 13.0 * ca2 * v5 * w4
            - 18.0 * ca4 * v5 * w4
            + 28.0 * ca3cf * v5 * w4
            - v5 * w5
            - 4.0 * ca2 * v5 * w5
            + 5.0 * ca4 * v5 * w5
            - 8.0 * ca3cf * v5 * w5))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v + v * w).powi(2));

    let term8 = -(2.0
        * cf
        * lmss
        * (8.0 * cacf2 + 4.0 * ca2 * v + 8.0 * cf * v
            - 8.0 * ca2 * cf * v
            - 16.0 * cacf2 * v
            - 20.0 * ca2 * v2
            - 20.0 * cf * v2
            + 20.0 * ca2 * cf * v2
            + 12.0 * cacf2 * v2
            + 44.0 * ca2 * v3
            + 20.0 * cf * v3
            - 20.0 * ca2 * cf * v3
            - 4.0 * cacf2 * v3
            - 52.0 * ca2 * v4
            - 10.0 * cf * v4
            + 10.0 * ca2 * cf * v4
            + 32.0 * ca2 * v5
            + 2.0 * cf * v5
            - 2.0 * ca2 * cf * v5
            - 8.0 * ca2 * v6
            + 16.0 * cacf2 * v * w
            + 14.0 * ca2 * v2 * w
            + 16.0 * cf * v2 * w
            - 32.0 * ca2 * cf * v2 * w
            - 12.0 * cacf2 * v2 * w
            - 64.0 * ca2 * v3 * w
            - 20.0 * cf * v3 * w
            + 56.0 * ca2 * cf * v3 * w
            + 4.0 * cacf2 * v3 * w
            + 114.0 * ca2 * v4 * w
            + 10.0 * cf * v4 * w
            - 34.0 * ca2 * cf * v4 * w
            - 92.0 * ca2 * v5 * w
            - 2.0 * cf * v5 * w
            + 6.0 * ca2 * cf * v5 * w
            + 28.0 * ca2 * v6 * w
            + 8.0 * cacf2 * v2 * w2
            - 2.0 * v3 * w2
            + 24.0 * ca2 * v3 * w2
            + 8.0 * cf * v3 * w2
            - 44.0 * ca2 * cf * v3 * w2
            + 6.0 * v4 * w2
            - 84.0 * ca2 * v4 * w2
            - 4.0 * cf * v4 * w2
            + 56.0 * ca2 * cf * v4 * w2
            - 8.0 * v5 * w2
            + 100.0 * ca2 * v5 * w2
            - 16.0 * ca2 * cf * v5 * w2
            + 4.0 * v6 * w2
            - 40.0 * ca2 * v6 * w2
            - 5.0 * v4 * w3
            + 21.0 * ca2 * v4 * w3
            - 28.0 * ca2 * cf * v4 * w3
            + 14.0 * v5 * w3
            - 50.0 * ca2 * v5 * w3
            + 20.0 * ca2 * cf * v5 * w3
            - 10.0 * v6 * w3
            + 30.0 * ca2 * v6 * w3
            - 6.0 * v5 * w4
            + 10.0 * ca2 * v5 * w4
            - 8.0 * ca2 * cf * v5 * w4
            + 8.0 * v6 * w4
            - 12.0 * ca2 * v6 * w4
            - 2.0 * v6 * w5
            + 2.0 * ca2 * v6 * w5))
        / ((1.0 - v) * v2 * w * (1.0 - v + v * w).powi(2));

    term5 + term6 + term7 + term8
}

/// `STRUV13(W,V,X3,S)`, part C: the `lw`/`lms` term groups.
fn qg_compton_quark_frag_part_c(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6);
    let (w2, w3, w4, w5) = (pre.w2, pre.w3, pre.w4, pre.w5);
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let ca4 = pre.ca4;
    let (lw, lms) = (pre.lw, pre.lms);

    let term9 = (4.0
        * cf
        * lw
        * (2.0 - 5.0 * ca2 + 3.0 * ca4 - 5.0 * v + 10.0 * ca2 * v + 4.0 * ca3 * v - 5.0 * ca4 * v
            + 3.0 * v2
            + 2.0 * ca * v2
            - 10.0 * ca2 * v2
            - 18.0 * ca3 * v2
            + 7.0 * ca4 * v2
            - 2.0 * ca * v3
            + 5.0 * ca2 * v3
            + 30.0 * ca3 * v3
            - 5.0 * ca4 * v3
            - 24.0 * ca3 * v4
            + 8.0 * ca3 * v5
            - w
            + ca2 * w
            - 4.0 * ca2 * v * w
            - 4.0 * ca3 * v * w
            + 5.0 * ca4 * v * w
            + 7.0 * v2 * w
            - 7.0 * ca * v2 * w
            + 8.0 * ca2 * v2 * w
            + 25.0 * ca3 * v2 * w
            - 11.0 * ca4 * v2 * w
            - 11.0 * v3 * w
            + 11.0 * ca * v3 * w
            - 10.0 * ca2 * v3 * w
            - 47.0 * ca3 * v3 * w
            + 13.0 * ca4 * v3 * w
            + 4.0 * v4 * w
            - 2.0 * ca * v4 * w
            + 32.0 * ca3 * v4 * w
            - ca4 * v4 * w
            - 8.0 * ca3 * v6 * w
            + v * w2
            + ca2 * v * w2
            - 6.0 * v2 * w2
            + 5.0 * ca * v2 * w2
            + 8.0 * ca2 * v2 * w2
            - 7.0 * ca3 * v2 * w2
            + ca4 * v2 * w2
            + 14.0 * v3 * w2
            - 13.0 * ca * v3 * w2
            - 7.0 * ca2 * v3 * w2
            + 17.0 * ca3 * v3 * w2
            - 8.0 * ca4 * v3 * w2
            - 9.0 * v4 * w2
            + 3.0 * ca * v4 * w2
            + 12.0 * ca2 * v4 * w2
            + 5.0 * ca3 * v4 * w2
            + 3.0 * ca4 * v4 * w2
            - 5.0 * ca2 * v5 * w2
            - 34.0 * ca3 * v5 * w2
            + 24.0 * ca3 * v6 * w2
            + v2 * w3
            - 3.0 * ca2 * v2 * w3
            - 4.0 * v3 * w3
            + 4.0 * ca * v3 * w3
            + 3.0 * ca2 * v3 * w3
            + 5.0 * v4 * w3
            - 7.0 * ca2 * v4 * w3
            - 18.0 * ca3 * v4 * w3
            - 4.0 * ca4 * v4 * w3
            + v5 * w3
            + 8.0 * ca2 * v5 * w3
            + 40.0 * ca3 * v5 * w3
            + ca4 * v5 * w3
            - 26.0 * ca3 * v6 * w3
            - v3 * w4
            - ca2 * v3 * w4
            - ca * v4 * w4
            - 3.0 * ca2 * v4 * w4
            + 5.0 * ca3 * v4 * w4
            + 2.0 * ca4 * v4 * w4
            - v5 * w4
            - 4.0 * ca2 * v5 * w4
            - 16.0 * ca3 * v5 * w4
            - 2.0 * ca4 * v5 * w4
            + 12.0 * ca3 * v6 * w4
            + 2.0 * ca2 * v4 * w5
            + ca2 * v5 * w5
            + 2.0 * ca3 * v5 * w5
            + ca4 * v5 * w5
            - 2.0 * ca3 * v6 * w5))
        / (ca * (1.0 - v) * v2 * (1.0 - w) * w * (1.0 - v * w) * (1.0 - v + v * w));

    let term10 = -(2.0
        * cf
        * lms
        * (1.0 + 2.0 * ca - 6.0 * ca2 - 2.0 * ca3 + 5.0 * ca4 + v - 6.0 * ca * v
            + 2.0 * ca2 * v
            + 10.0 * ca3 * v
            - 3.0 * ca4 * v
            - 5.0 * v2
            + 8.0 * ca * v2
            - 4.0 * ca2 * v2
            - 20.0 * ca3 * v2
            + 9.0 * ca4 * v2
            + v3
            - 2.0 * ca2 * v3
            + 16.0 * ca3 * v3
            + ca4 * v3
            - 8.0 * ca3 * v4
            - w
            - ca * w
            + 2.0 * ca2 * w
            + ca3 * w
            - ca4 * w
            - v * w
            - 2.0 * ca * v * w
            + 12.0 * ca2 * v * w
            - 11.0 * ca4 * v * w
            + 5.0 * v2 * w
            + 8.0 * ca * v2 * w
            - 6.0 * ca2 * v2 * w
            - 10.0 * ca3 * v2 * w
            - 15.0 * ca4 * v2 * w
            + v3 * w
            - 16.0 * ca * v3 * w
            + 14.0 * ca2 * v3 * w
            + 32.0 * ca3 * v3 * w
            + ca4 * v3 * w
            - 2.0 * v4 * w
            + 4.0 * ca2 * v4 * w
            - 28.0 * ca3 * v4 * w
            - 2.0 * ca4 * v4 * w
            + 16.0 * ca3 * v5 * w
            + 2.0 * v * w2
            + 2.0 * ca * v * w2
            - 4.0 * ca2 * v * w2
            - 2.0 * ca3 * v * w2
            + 2.0 * ca4 * v * w2
            - v2 * w2
            - 2.0 * ca * v2 * w2
            - 6.0 * ca2 * v2 * w2
            + 6.0 * ca3 * v2 * w2
            + 7.0 * ca4 * v2 * w2
            - 3.0 * v3 * w2
            + 4.0 * ca2 * v3 * w2
            - 8.0 * ca3 * v3 * w2
            - ca4 * v3 * w2
            + v4 * w2
            + 12.0 * ca * v4 * w2
            - 4.0 * ca2 * v4 * w2
            - 8.0 * ca3 * v4 * w2
            + 19.0 * ca4 * v4 * w2
            + v5 * w2
            - 2.0 * ca2 * v5 * w2
            + 8.0 * ca3 * v5 * w2
            + ca4 * v5 * w2
            - 8.0 * ca3 * v6 * w2
            - v2 * w3
            - ca * v2 * w3
            + 2.0 * ca2 * v2 * w3
            + ca3 * v2 * w3
            - ca4 * v2 * w3
            + v3 * w3
            + 2.0 * ca * v3 * w3
            - 4.0 * ca3 * v3 * w3
            - ca4 * v3 * w3
            + 3.0 * v4 * w3
            - 2.0 * ca * v4 * w3
            - 14.0 * ca2 * v4 * w3
            + 8.0 * ca3 * v4 * w3
            - 5.0 * ca4 * v4 * w3
            - 5.0 * v5 * w3
            - 8.0 * ca * v5 * w3
            + 2.0 * ca2 * v5 * w3
            + 3.0 * ca4 * v5 * w3
            + 4.0 * ca3 * v6 * w3
            + 2.0 * v5 * w4
            + 2.0 * ca * v5 * w4
            + 6.0 * ca2 * v5 * w4
            - 2.0 * ca3 * v5 * w4
            - 8.0 * ca4 * v5 * w4
            + 4.0 * ca * v6 * w4
            - 4.0 * ca3 * v6 * w4
            - 2.0 * ca * v6 * w5
            + 2.0 * ca3 * v6 * w5))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2));

    term9 + term10
}

/// `STRUV13(W,V,X3,S)`, term 11: the pure (no-log) `CF`-polynomial piece.
fn qg_compton_quark_frag_term11(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7, v8) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8);
    let (w2, w3, w4, w5, w6, w7) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7);
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let ca4 = pre.ca4;

    (2.0 * cf
        * (9.0 - 9.0 * ca - 7.0 * ca2 + 9.0 * ca3 - 2.0 * ca4 - 39.0 * v
            + 39.0 * ca * v
            + 17.0 * ca2 * v
            - 51.0 * ca3 * v
            + 22.0 * ca4 * v
            + 75.0 * v2
            - 81.0 * ca * v2
            - 9.0 * ca2 * v2
            + 165.0 * ca3 * v2
            - 66.0 * ca4 * v2
            - 69.0 * v3
            + 81.0 * ca * v3
            - 5.0 * ca2 * v3
            - 309.0 * ca3 * v3
            + 74.0 * ca4 * v3
            + 24.0 * v4
            - 30.0 * ca * v4
            + 4.0 * ca2 * v4
            + 330.0 * ca3 * v4
            - 28.0 * ca4 * v4
            - 192.0 * ca3 * v5
            + 48.0 * ca3 * v6
            + 12.0 * ca * w
            - 6.0 * ca2 * w
            - 12.0 * ca3 * w
            + 6.0 * ca4 * w
            - 9.0 * v * w
            - 57.0 * ca * v * w
            + 9.0 * ca2 * v * w
            + 63.0 * ca3 * v * w
            - 12.0 * ca4 * v * w
            + 12.0 * v2 * w
            + 90.0 * ca * v2 * w
            + 17.0 * ca2 * v2 * w
            - 162.0 * ca3 * v2 * w
            + 95.0 * ca4 * v2 * w
            + 15.0 * v3 * w
            - 39.0 * ca * v3 * w
            - 88.0 * ca2 * v3 * w
            + 249.0 * ca3 * v3 * w
            - 277.0 * ca4 * v3 * w
            - 6.0 * v4 * w
            - 60.0 * ca * v4 * w
            + 119.0 * ca2 * v4 * w
            - 132.0 * ca3 * v4 * w
            + 263.0 * ca4 * v4 * w
            - 12.0 * v5 * w
            + 54.0 * ca * v5 * w
            - 51.0 * ca2 * v5 * w
            - 138.0 * ca3 * v5 * w
            - 75.0 * ca4 * v5 * w
            + 228.0 * ca3 * v6 * w
            - 96.0 * ca3 * v7 * w
            - 18.0 * v2 * w2
            + 42.0 * ca * v2 * w2
            + 2.0 * ca2 * v2 * w2
            - 42.0 * ca3 * v2 * w2
            + 16.0 * ca4 * v2 * w2
            + 36.0 * v3 * w2
            - 132.0 * ca * v3 * w2
            + 55.0 * ca2 * v3 * w2
            + 162.0 * ca3 * v3 * w2
            + 191.0 * ca4 * v3 * w2
            - 54.0 * v4 * w2
            + 180.0 * ca * v4 * w2
            - 184.0 * ca2 * v4 * w2
            - 426.0 * ca3 * v4 * w2
            - 552.0 * ca4 * v4 * w2
            + 60.0 * v5 * w2
            - 66.0 * ca * v5 * w2
            + 145.0 * ca2 * v5 * w2
            + 666.0 * ca3 * v5 * w2
            + 441.0 * ca4 * v5 * w2
            - 12.0 * v6 * w2
            - 42.0 * ca * v6 * w2
            - 18.0 * ca2 * v6 * w2
            - 510.0 * ca3 * v6 * w2
            - 108.0 * ca4 * v6 * w2
            + 120.0 * ca3 * v7 * w2
            + 48.0 * ca3 * v8 * w2
            - 24.0 * ca * v2 * w3
            + 12.0 * ca2 * v2 * w3
            + 24.0 * ca3 * v2 * w3
            - 12.0 * ca4 * v2 * w3
            + 18.0 * v3 * w3
            + 90.0 * ca * v3 * w3
            - 6.0 * ca2 * v3 * w3
            - 102.0 * ca3 * v3 * w3
            + 12.0 * ca4 * v3 * w3
            - 42.0 * v4 * w3
            - 96.0 * ca * v4 * w3
            + 117.0 * ca2 * v4 * w3
            + 222.0 * ca3 * v4 * w3
            + 259.0 * ca4 * v4 * w3
            + 42.0 * v5 * w3
            - 12.0 * ca * v5 * w3
            - 233.0 * ca2 * v5 * w3
            - 306.0 * ca3 * v5 * w3
            - 563.0 * ca4 * v5 * w3
            - 54.0 * v6 * w3
            + 54.0 * ca * v6 * w3
            + 105.0 * ca2 * v6 * w3
            + 150.0 * ca3 * v6 * w3
            + 307.0 * ca4 * v6 * w3
            + 24.0 * v7 * w3
            + 42.0 * ca * v7 * w3
            - 7.0 * ca2 * v7 * w3
            + 114.0 * ca3 * v7 * w3
            - 17.0 * ca4 * v7 * w3
            - 156.0 * ca3 * v8 * w3
            + 9.0 * v4 * w4
            - 33.0 * ca * v4 * w4
            + 5.0 * ca2 * v4 * w4
            + 33.0 * ca3 * v4 * w4
            - 14.0 * ca4 * v4 * w4
            - 33.0 * v5 * w4
            + 75.0 * ca * v5 * w4
            + 72.0 * ca2 * v5 * w4
            - 81.0 * ca3 * v5 * w4
            + 231.0 * ca4 * v5 * w4
            + 75.0 * v6 * w4
            - 15.0 * ca * v6 * w4
            - 73.0 * ca2 * v6 * w4
            + 141.0 * ca3 * v6 * w4
            - 316.0 * ca4 * v6 * w4
            - 51.0 * v7 * w4
            - 75.0 * ca * v7 * w4
            - 4.0 * ca2 * v7 * w4
            - 225.0 * ca3 * v7 * w4
            + 55.0 * ca4 * v7 * w4
            - 24.0 * ca * v8 * w4
            + 204.0 * ca3 * v8 * w4
            + 12.0 * ca * v4 * w5
            - 6.0 * ca2 * v4 * w5
            - 12.0 * ca3 * v4 * w5
            + 6.0 * ca4 * v4 * w5
            - 9.0 * v5 * w5
            - 33.0 * ca * v5 * w5
            - 3.0 * ca2 * v5 * w5
            + 39.0 * ca3 * v5 * w5
            - 18.0 * v6 * w5
            + 10.0 * ca2 * v6 * w5
            - 54.0 * ca3 * v6 * w5
            + 102.0 * ca4 * v6 * w5
            + 39.0 * v7 * w5
            + 27.0 * ca * v7 * w5
            + 11.0 * ca2 * v7 * w5
            + 105.0 * ca3 * v7 * w5
            - 50.0 * ca4 * v7 * w5
            + 66.0 * ca * v8 * w5
            - 150.0 * ca3 * v8 * w5
            - 12.0 * v7 * w6
            + 6.0 * ca * v7 * w6
            - 18.0 * ca3 * v7 * w6
            + 12.0 * ca4 * v7 * w6
            - 60.0 * ca * v8 * w6
            + 72.0 * ca3 * v8 * w6
            + 18.0 * ca * v8 * w7
            - 18.0 * ca3 * v8 * w7))
        / (3.0 * ca * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2))
}

/// `STRUV13(W,V,X3,S)`, term 12: the `lv` piece.
fn qg_compton_quark_frag_term12(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7, v8) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8);
    let (w2, w3, w4, w5, w6, w7) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7);
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let ca4 = pre.ca4;
    let lv = pre.lv;

    (2.0 * cf
        * lv
        * (3.0 + 2.0 * ca - 18.0 * ca2 - 2.0 * ca3 + 15.0 * ca4 - 9.0 * v - 10.0 * ca * v
            + 54.0 * ca2 * v
            + 26.0 * ca3 * v
            - 49.0 * ca4 * v
            + 7.0 * v2
            + 22.0 * ca * v2
            - 68.0 * ca2 * v2
            - 94.0 * ca3 * v2
            + 77.0 * ca4 * v2
            + v3
            - 22.0 * ca * v3
            + 50.0 * ca2 * v3
            + 158.0 * ca3 * v3
            - 71.0 * ca4 * v3
            - 2.0 * v4
            + 8.0 * ca * v4
            - 22.0 * ca2 * v4
            - 144.0 * ca3 * v4
            + 32.0 * ca4 * v4
            + 4.0 * ca2 * v5
            + 72.0 * ca3 * v5
            - 4.0 * ca4 * v5
            - 16.0 * ca3 * v6
            - w
            - ca * w
            + 2.0 * ca2 * w
            + ca3 * w
            - ca4 * w
            + 3.0 * v * w
            + 4.0 * ca * v * w
            - 4.0 * ca2 * v * w
            - 6.0 * ca3 * v * w
            + ca4 * v * w
            + 15.0 * v2 * w
            - 21.0 * ca * v2 * w
            - 38.0 * ca2 * v2 * w
            + 33.0 * ca3 * v2 * w
            - 21.0 * ca4 * v2 * w
            - 53.0 * v3 * w
            + 58.0 * ca * v3 * w
            + 94.0 * ca2 * v3 * w
            - 60.0 * ca3 * v3 * w
            + 75.0 * ca4 * v3 * w
            + 50.0 * v4 * w
            - 72.0 * ca * v4 * w
            - 92.0 * ca2 * v4 * w
            + 8.0 * ca3 * v4 * w
            - 50.0 * ca4 * v4 * w
            - 14.0 * v5 * w
            + 48.0 * ca * v5 * w
            + 46.0 * ca2 * v5 * w
            + 80.0 * ca3 * v5 * w
            - 8.0 * ca4 * v5 * w
            - 16.0 * ca * v6 * w
            - 8.0 * ca2 * v6 * w
            - 88.0 * ca3 * v6 * w
            + 4.0 * ca4 * v6 * w
            + 32.0 * ca3 * v7 * w
            - 8.0 * v2 * w2
            - 6.0 * ca * v2 * w2
            + 40.0 * ca2 * v2 * w2
            + 6.0 * ca3 * v2 * w2
            - 32.0 * ca4 * v2 * w2
            + 36.0 * v3 * w2
            - 6.0 * ca * v3 * w2
            - 72.0 * ca2 * v3 * w2
            - 36.0 * ca3 * v3 * w2
            - 38.0 * v4 * w2
            + 32.0 * ca * v4 * w2
            + 34.0 * ca2 * v4 * w2
            + 130.0 * ca3 * v4 * w2
            + 32.0 * ca4 * v4 * w2
            - 6.0 * v5 * w2
            - 24.0 * ca * v5 * w2
            + 2.0 * ca2 * v5 * w2
            - 216.0 * ca3 * v5 * w2
            + 12.0 * ca4 * v5 * w2
            + 14.0 * v6 * w2
            - 8.0 * ca * v6 * w2
            - 14.0 * ca2 * v6 * w2
            + 160.0 * ca3 * v6 * w2
            + 4.0 * ca4 * v6 * w2
            + 16.0 * ca * v7 * w2
            + 4.0 * ca2 * v7 * w2
            - 32.0 * ca3 * v7 * w2
            - 16.0 * ca3 * v8 * w2
            + 2.0 * v2 * w3
            + 2.0 * ca * v2 * w3
            - 4.0 * ca2 * v2 * w3
            - 2.0 * ca3 * v2 * w3
            + 2.0 * ca4 * v2 * w3
            - 4.0 * v3 * w3
            - 6.0 * ca * v3 * w3
            + 4.0 * ca2 * v3 * w3
            + 10.0 * ca3 * v3 * w3
            + 2.0 * v4 * w3
            + 12.0 * ca * v4 * w3
            + 26.0 * ca2 * v4 * w3
            - 46.0 * ca3 * v4 * w3
            - 16.0 * ca4 * v4 * w3
            + 22.0 * v5 * w3
            - 48.0 * ca * v5 * w3
            - 2.0 * ca2 * v5 * w3
            + 94.0 * ca3 * v5 * w3
            - 8.0 * ca4 * v5 * w3
            - 18.0 * v6 * w3
            + 84.0 * ca * v6 * w3
            - 8.0 * ca2 * v6 * w3
            - 56.0 * ca3 * v6 * w3
            - 30.0 * ca4 * v6 * w3
            - 6.0 * v7 * w3
            - 56.0 * ca * v7 * w3
            - 2.0 * ca2 * v7 * w3
            - 36.0 * ca3 * v7 * w3
            + 4.0 * ca4 * v7 * w3
            + 48.0 * ca3 * v8 * w3
            + 5.0 * v4 * w4
            + 4.0 * ca * v4 * w4
            - 22.0 * ca2 * v4 * w4
            - 4.0 * ca3 * v4 * w4
            + 17.0 * ca4 * v4 * w4
            - 13.0 * v5 * w4
            + 20.0 * ca * v5 * w4
            - 8.0 * ca2 * v5 * w4
            + 2.0 * ca3 * v5 * w4
            - 3.0 * ca4 * v5 * w4
            + 7.0 * v6 * w4
            - 70.0 * ca * v6 * w4
            + 26.0 * ca2 * v6 * w4
            - 16.0 * ca3 * v6 * w4
            + 51.0 * ca4 * v6 * w4
            + 13.0 * v7 * w4
            + 54.0 * ca * v7 * w4
            + 8.0 * ca2 * v7 * w4
            + 62.0 * ca3 * v7 * w4
            - 21.0 * ca4 * v7 * w4
            + 8.0 * ca * v8 * w4
            - 60.0 * ca3 * v8 * w4
            - v4 * w5
            - ca * v4 * w5
            + 2.0 * ca2 * v4 * w5
            + ca3 * v4 * w5
            - ca4 * v4 * w5
            + v5 * w5
            + 2.0 * ca * v5 * w5
            - 4.0 * ca3 * v5 * w5
            - ca4 * v5 * w5
            - v6 * w5
            + 13.0 * ca * v6 * w5
            - 4.0 * ca2 * v6 * w5
            + 9.0 * ca3 * v6 * w5
            - 27.0 * ca4 * v6 * w5
            - 9.0 * v7 * w5
            - 10.0 * ca * v7 * w5
            - 20.0 * ca2 * v7 * w5
            - 34.0 * ca3 * v7 * w5
            + 29.0 * ca4 * v7 * w5
            - 20.0 * ca * v8 * w5
            + 44.0 * ca3 * v8 * w5
            + 2.0 * v7 * w6
            - 4.0 * ca * v7 * w6
            + 10.0 * ca2 * v7 * w6
            + 8.0 * ca3 * v7 * w6
            - 12.0 * ca4 * v7 * w6
            + 16.0 * ca * v8 * w6
            - 20.0 * ca3 * v8 * w6
            - 4.0 * ca * v8 * w7
            + 4.0 * ca3 * v8 * w7))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2))
}

/// `STRUV13(W,V,X3,S)`, term 13: the `l1w` piece.
fn qg_compton_quark_frag_term13(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7, v8) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8);
    let (w2, w3, w4, w5, w6, w7) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7);
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let ca4 = pre.ca4;
    let l1w = pre.l1w;

    (2.0 * cf
        * l1w
        * (5.0 + 2.0 * ca - 16.0 * ca2 - 2.0 * ca3 + 11.0 * ca4 - 19.0 * v - 6.0 * ca * v
            + 46.0 * ca2 * v
            + 22.0 * ca3 * v
            - 35.0 * ca4 * v
            + 25.0 * v2
            + 2.0 * ca * v2
            - 50.0 * ca2 * v2
            - 78.0 * ca3 * v2
            + 55.0 * ca4 * v2
            - 13.0 * v3
            + 14.0 * ca * v3
            + 26.0 * ca2 * v3
            + 134.0 * ca3 * v3
            - 51.0 * ca4 * v3
            + 2.0 * v4
            - 20.0 * ca * v4
            - 6.0 * ca2 * v4
            - 128.0 * ca3 * v4
            + 22.0 * ca4 * v4
            + 8.0 * ca * v5
            + 68.0 * ca3 * v5
            - 2.0 * ca4 * v5
            - 16.0 * ca3 * v6
            - w
            - ca * w
            + 2.0 * ca2 * w
            + ca3 * w
            - ca4 * w
            + 5.0 * v * w
            + 4.0 * ca * v * w
            - 2.0 * ca2 * v * w
            - 6.0 * ca3 * v * w
            + ca4 * v * w
            + 13.0 * v2 * w
            - 9.0 * ca * v2 * w
            - 30.0 * ca2 * v2 * w
            + 25.0 * ca3 * v2 * w
            - 19.0 * ca4 * v2 * w
            - 55.0 * v3 * w
            + 30.0 * ca * v3 * w
            + 76.0 * ca2 * v3 * w
            - 36.0 * ca3 * v3 * w
            + 61.0 * ca4 * v3 * w
            + 52.0 * v4 * w
            - 68.0 * ca * v4 * w
            - 76.0 * ca2 * v4 * w
            - 16.0 * ca3 * v4 * w
            - 36.0 * ca4 * v4 * w
            - 14.0 * v5 * w
            + 76.0 * ca * v5 * w
            + 30.0 * ca2 * v5 * w
            + 88.0 * ca3 * v5 * w
            - 6.0 * ca4 * v5 * w
            - 32.0 * ca * v6 * w
            - 88.0 * ca3 * v6 * w
            + 32.0 * ca3 * v7 * w
            - 12.0 * v2 * w2
            - 6.0 * ca * v2 * w2
            + 32.0 * ca2 * v2 * w2
            + 6.0 * ca3 * v2 * w2
            - 24.0 * ca4 * v2 * w2
            + 44.0 * v3 * w2
            - 6.0 * ca * v3 * w2
            - 54.0 * ca2 * v3 * w2
            - 36.0 * ca3 * v3 * w2
            - 10.0 * ca4 * v3 * w2
            - 34.0 * v4 * w2
            + 64.0 * ca * v4 * w2
            + 30.0 * ca2 * v4 * w2
            + 114.0 * ca3 * v4 * w2
            + 40.0 * ca4 * v4 * w2
            - 10.0 * v5 * w2
            - 84.0 * ca * v5 * w2
            + 18.0 * ca2 * v5 * w2
            - 172.0 * ca3 * v5 * w2
            - 12.0 * ca4 * v5 * w2
            + 10.0 * v6 * w2
            + 12.0 * ca * v6 * w2
            - 30.0 * ca2 * v6 * w2
            + 120.0 * ca3 * v6 * w2
            + 18.0 * ca4 * v6 * w2
            + 24.0 * ca * v7 * w2
            - 20.0 * ca3 * v7 * w2
            + 2.0 * ca4 * v7 * w2
            - 16.0 * ca3 * v8 * w2
            + 2.0 * v2 * w3
            + 2.0 * ca * v2 * w3
            - 4.0 * ca2 * v2 * w3
            - 2.0 * ca3 * v2 * w3
            + 2.0 * ca4 * v2 * w3
            - 8.0 * v3 * w3
            - 6.0 * ca * v3 * w3
            + 10.0 * ca3 * v3 * w3
            + 2.0 * v4 * w3
            - 4.0 * ca * v4 * w3
            + 6.0 * ca2 * v4 * w3
            - 34.0 * ca3 * v4 * w3
            - 16.0 * ca4 * v4 * w3
            + 14.0 * v5 * w3
            - 16.0 * ca * v5 * w3
            + 2.0 * ca2 * v5 * w3
            + 50.0 * ca3 * v5 * w3
            + 12.0 * ca4 * v5 * w3
            + 88.0 * ca * v6 * w3
            + 8.0 * ca2 * v6 * w3
            - 12.0 * ca3 * v6 * w3
            - 44.0 * ca4 * v6 * w3
            - 6.0 * v7 * w3
            - 76.0 * ca * v7 * w3
            + 14.0 * ca2 * v7 * w3
            - 40.0 * ca3 * v7 * w3
            - 2.0 * ca4 * v7 * w3
            + 40.0 * ca3 * v8 * w3
            + 7.0 * v4 * w4
            + 4.0 * ca * v4 * w4
            - 12.0 * ca2 * v4 * w4
            - 4.0 * ca3 * v4 * w4
            + 13.0 * ca4 * v4 * w4
            - 11.0 * v5 * w4
            + 16.0 * ca * v5 * w4
            - 16.0 * ca2 * v5 * w4
            + 10.0 * ca3 * v5 * w4
            - 5.0 * ca4 * v5 * w4
            - 7.0 * v6 * w4
            - 82.0 * ca * v6 * w4
            + 12.0 * ca2 * v6 * w4
            - 20.0 * ca3 * v6 * w4
            + 57.0 * ca4 * v6 * w4
            + 11.0 * v7 * w4
            + 70.0 * ca * v7 * w4
            - 16.0 * ca2 * v7 * w4
            + 38.0 * ca3 * v7 * w4
            - 17.0 * ca4 * v7 * w4
            + 8.0 * ca * v8 * w4
            - 40.0 * ca3 * v8 * w4
            - v4 * w5
            - ca * v4 * w5
            + 2.0 * ca2 * v4 * w5
            + ca3 * v4 * w5
            - ca4 * v4 * w5
            + 3.0 * v5 * w5
            + 2.0 * ca * v5 * w5
            + 2.0 * ca2 * v5 * w5
            - 4.0 * ca3 * v5 * w5
            - ca4 * v5 * w5
            + v6 * w5
            + 17.0 * ca * v6 * w5
            + 8.0 * ca2 * v6 * w5
            + 5.0 * ca3 * v6 * w5
            - 29.0 * ca4 * v6 * w5
            - 7.0 * v7 * w5
            - 14.0 * ca * v7 * w5
            - 6.0 * ca2 * v7 * w5
            - 14.0 * ca3 * v7 * w5
            + 31.0 * ca4 * v7 * w5
            - 20.0 * ca * v8 * w5
            + 28.0 * ca3 * v8 * w5
            - 4.0 * ca2 * v6 * w6
            + 2.0 * v7 * w6
            - 4.0 * ca * v7 * w6
            + 8.0 * ca2 * v7 * w6
            + 4.0 * ca3 * v7 * w6
            - 14.0 * ca4 * v7 * w6
            + 16.0 * ca * v8 * w6
            - 16.0 * ca3 * v8 * w6
            - 4.0 * ca * v8 * w7
            + 4.0 * ca3 * v8 * w7))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2))
}

/// `STRUV13(W,V,X3,S)`, part D: the pure/`lv`/`l1w` term groups.
fn qg_compton_quark_frag_part_d(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    qg_compton_quark_frag_term11(w, v, ctx, pre)
        + qg_compton_quark_frag_term12(w, v, ctx, pre)
        + qg_compton_quark_frag_term13(w, v, ctx, pre)
}

/// `STRUV13(W,V,X3,S)`.
#[must_use]
pub fn qg_compton_quark_frag(
    w: f64,
    v: f64,
    _x3: f64,
    _s: f64,
    ctx: &MeContext,
    pre: &Precalc,
) -> f64 {
    qg_compton_quark_frag_part_a(w, v, ctx, pre)
        + qg_compton_quark_frag_part_b(w, v, ctx, pre)
        + qg_compton_quark_frag_part_c(w, v, ctx, pre)
        + qg_compton_quark_frag_part_d(w, v, ctx, pre)
}

/// `STRUV14(W,V,X3,S)`, part A: the `lvw`/`l1v`/`l1vw`/`lw` term groups.
fn qg_compton_gluon_frag_part_a(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6);
    let (w2, w3, w4) = (pre.w2, pre.w3, pre.w4);
    let ca2 = pre.ca2;
    let ca4 = pre.ca4;
    let (lvw, l1v, l1vw, lw) = (pre.lvw, pre.l1v, pre.l1vw, pre.lw);

    let term1 = (4.0
        * cf
        * lvw
        * (ca2 + 14.0 * ca4 - v - ca2 * v - 11.0 * ca4 * v + 2.0 * v2 - ca2 * v2
            + 28.0 * ca4 * v2
            - 2.0 * ca2 * v * w
            - 12.0 * ca4 * v * w
            + 3.0 * ca2 * v2 * w
            - 37.0 * ca4 * v2 * w
            - 3.0 * v3 * w
            + 7.0 * ca2 * v3 * w
            - 9.0 * ca4 * v3 * w
            - 8.0 * ca2 * v2 * w2
            + 25.0 * ca4 * v2 * w2
            + 3.0 * v3 * w2
            - 7.0 * ca2 * v3 * w2
            + 9.0 * ca4 * v3 * w2
            - 8.0 * ca2 * v4 * w2
            + 8.0 * ca4 * v4 * w2
            + 4.0 * ca2 * v3 * w3
            - 4.0 * ca4 * v3 * w3
            + 12.0 * ca2 * v4 * w3
            - 12.0 * ca4 * v4 * w3
            - 4.0 * ca2 * v4 * w4
            + 4.0 * ca4 * v4 * w4))
        / (ca * (1.0 - v) * v2 * w);

    let term2 = -(4.0
        * cf
        * l1v
        * (8.0 * ca4 - 29.0 * ca4 * v + 42.0 * ca4 * v2 + v3 - ca2 * v3 - 28.0 * ca4 * v3 - v4
            + ca2 * v4
            + 7.0 * ca4 * v4
            + 6.0 * ca4 * v * w
            - 16.0 * ca4 * v2 * w
            - 2.0 * v3 * w
            + 6.0 * ca2 * v3 * w
            - 7.0 * ca4 * v3 * w
            + 2.0 * v4 * w
            - 6.0 * ca2 * v4 * w
            + 25.0 * ca4 * v4 * w
            + v5 * w
            + ca2 * v5 * w
            - 11.0 * ca4 * v5 * w
            - 8.0 * ca4 * v2 * w2
            - 2.0 * ca2 * v3 * w2
            + 33.0 * ca4 * v3 * w2
            + 2.0 * ca2 * v4 * w2
            - 33.0 * ca4 * v4 * w2
            - 3.0 * v5 * w2
            - ca2 * v5 * w2
            - ca4 * v5 * w2
            + 8.0 * ca4 * v6 * w2
            - 6.0 * ca4 * v3 * w3
            + 6.0 * ca4 * v4 * w3
            + 2.0 * v5 * w3
            + 16.0 * ca4 * v5 * w3
            - 12.0 * ca4 * v6 * w3
            + 2.0 * ca2 * v5 * w4
            - 6.0 * ca4 * v5 * w4
            + 4.0 * ca4 * v6 * w4))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v * w) * (1.0 - v + v * w));

    let term3 = -(4.0
        * cf
        * l1vw
        * (ca2 - 2.0 * ca4 - 3.0 * ca2 * v + 6.0 * ca4 * v + 3.0 * ca2 * v2
            - 6.0 * ca4 * v2
            - 3.0 * ca2 * v4
            + 6.0 * ca4 * v4
            + 3.0 * ca2 * v5
            - 6.0 * ca4 * v5
            - ca2 * v6
            + 2.0 * ca4 * v6
            + v * w
            + 3.0 * ca2 * v * w
            - ca4 * v * w
            - 2.0 * v2 * w
            - 7.0 * ca2 * v2 * w
            + 7.0 * ca4 * v2 * w
            + v3 * w
            + 8.0 * ca2 * v3 * w
            - 5.0 * ca4 * v3 * w
            + v4 * w
            - 6.0 * ca2 * v4 * w
            - 7.0 * ca4 * v4 * w
            - 2.0 * v5 * w
            + ca2 * v5 * w
            + 6.0 * ca4 * v5 * w
            + v6 * w
            + ca2 * v6 * w
            + 2.0 * v2 * w2
            + 3.0 * ca2 * v2 * w2
            - 5.0 * ca4 * v2 * w2
            - 3.0 * v3 * w2
            - 3.0 * ca2 * v3 * w2
            + 6.0 * ca4 * v3 * w2
            + 2.0 * v4 * w2
            + 11.0 * ca2 * v4 * w2
            + 3.0 * ca4 * v4 * w2
            + 3.0 * v5 * w2
            - 11.0 * ca2 * v5 * w2
            + 10.0 * ca4 * v5 * w2
            - 4.0 * v6 * w2
            - 14.0 * ca4 * v6 * w2
            + v3 * w3
            - 5.0 * ca2 * v3 * w3
            - 5.0 * v4 * w3
            + 2.0 * ca2 * v4 * w3
            - 5.0 * ca4 * v4 * w3
            + 9.0 * ca2 * v5 * w3
            - 15.0 * ca4 * v5 * w3
            + 6.0 * v6 * w3
            + 4.0 * ca2 * v6 * w3
            + 22.0 * ca4 * v6 * w3
            + 2.0 * v4 * w4
            - 4.0 * ca2 * v4 * w4
            + 3.0 * ca4 * v4 * w4
            - v5 * w4
            - 4.0 * ca2 * v5 * w4
            + 6.0 * ca4 * v5 * w4
            - 4.0 * v6 * w4
            - 7.0 * ca2 * v6 * w4
            - 12.0 * ca4 * v6 * w4
            + 2.0 * ca2 * v5 * pre.w5
            - ca4 * v5 * pre.w5
            + v6 * pre.w5
            + 3.0 * ca2 * v6 * pre.w5
            + 2.0 * ca4 * v6 * pre.w5))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v + v * w).powi(3));

    let term4 = (4.0
        * cf
        * lw
        * (2.0 * ca2 + 14.0 * ca4 - 5.0 * ca2 * v - 35.0 * ca4 * v
            + 2.0 * v2
            + 2.0 * ca2 * v2
            + 76.0 * ca4 * v2
            - 2.0 * v3
            + ca2 * v3
            - 87.0 * ca4 * v3
            + 48.0 * ca4 * v4
            - 16.0 * ca4 * v5
            - 2.0 * ca4 * w
            + 5.0 * ca4 * v * w
            + 8.0 * ca2 * v2 * w
            - 66.0 * ca4 * v2 * w
            - 3.0 * v3 * w
            - 9.0 * ca2 * v3 * w
            + 110.0 * ca4 * v3 * w
            + 5.0 * v4 * w
            + ca2 * v4 * w
            - 35.0 * ca4 * v4 * w
            + 16.0 * ca4 * v6 * w
            - 6.0 * ca2 * v2 * w2
            + 22.0 * ca4 * v2 * w2
            + 2.0 * v3 * w2
            + 7.0 * ca2 * v3 * w2
            - 43.0 * ca4 * v3 * w2
            - 4.0 * v4 * w2
            + ca2 * v4 * w2
            - 65.0 * ca4 * v4 * w2
            - 3.0 * v5 * w2
            - 2.0 * ca2 * v5 * w2
            + 64.0 * ca4 * v5 * w2
            - 48.0 * ca4 * v6 * w2
            - 2.0 * ca4 * v2 * w3
            + 5.0 * ca4 * v3 * w3
            - 6.0 * ca2 * v4 * w3
            + 81.0 * ca4 * v4 * w3
            + 5.0 * v5 * w3
            + 6.0 * ca2 * v5 * w3
            - 73.0 * ca4 * v5 * w3
            + 52.0 * ca4 * v6 * w3
            + 4.0 * ca2 * v4 * w4
            - 30.0 * ca4 * v4 * w4
            - 2.0 * v5 * w4
            - 4.0 * ca2 * v5 * w4
            + 29.0 * ca4 * v5 * w4
            - 24.0 * ca4 * v6 * w4
            + 4.0 * ca4 * v4 * pre.w5
            - 4.0 * ca4 * v5 * pre.w5
            + 4.0 * ca4 * v6 * pre.w5))
        / (ca * (1.0 - v) * v2 * (1.0 - w) * w * (1.0 - v * w) * (1.0 - v + v * w));

    term1 + term2 + term3 + term4
}

/// `STRUV14(W,V,X3,S)`, part B: the `lms`/`lmss` term groups.
fn qg_compton_gluon_frag_part_b(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7);
    let (w2, w3, w4, w5) = (pre.w2, pre.w3, pre.w4, pre.w5);
    let ca2 = pre.ca2;
    let ca4 = pre.ca4;
    let (lms, lmss) = (pre.lms, pre.lmss);

    let term5 = (2.0
        * cf
        * lms
        * (4.0 * ca2 - 52.0 * ca4 + v - 8.0 * ca2 * v + 71.0 * ca4 * v - 4.0 * v2
            + 10.0 * ca2 * v2
            - 102.0 * ca4 * v2
            + v3
            - 2.0 * ca2 * v3
            + 33.0 * ca4 * v3
            - 16.0 * ca4 * v4
            - 4.0 * ca2 * w
            + 20.0 * ca4 * w
            - 2.0 * ca2 * v * w
            + 74.0 * ca4 * v * w
            + v2 * w
            + 2.0 * ca2 * v2 * w
            - 67.0 * ca4 * v2 * w
            + 7.0 * v3 * w
            - 18.0 * ca2 * v3 * w
            + 171.0 * ca4 * v3 * w
            - 2.0 * v4 * w
            + 4.0 * ca2 * v4 * w
            - 58.0 * ca4 * v4 * w
            + 32.0 * ca4 * v5 * w
            + 8.0 * ca2 * v * w2
            - 40.0 * ca4 * v * w2
            - 8.0 * ca2 * v2 * w2
            + 8.0 * ca4 * v2 * w2
            - 4.0 * v3 * w2
            + 14.0 * ca2 * v3 * w2
            - 58.0 * ca4 * v3 * w2
            - 4.0 * v4 * w2
            + 30.0 * ca2 * v4 * w2
            - 74.0 * ca4 * v4 * w2
            + v5 * w2
            - 2.0 * ca2 * v5 * w2
            + 17.0 * ca4 * v5 * w2
            - 16.0 * ca4 * v6 * w2
            - 4.0 * ca2 * v2 * w3
            + 20.0 * ca4 * v2 * w3
            + 6.0 * ca2 * v3 * w3
            - 30.0 * ca4 * v3 * w3
            + 4.0 * v4 * w3
            - 28.0 * ca2 * v4 * w3
            + 72.0 * ca4 * v4 * w3
            - v5 * w3
            - 18.0 * ca2 * v5 * w3
            + 3.0 * ca4 * v5 * w3
            + 8.0 * ca4 * v6 * w3
            + 12.0 * ca2 * v5 * w4
            - 12.0 * ca4 * v5 * w4
            + 8.0 * ca2 * v6 * w4
            - 8.0 * ca4 * v6 * w4
            - 4.0 * ca2 * v6 * w5
            + 4.0 * ca4 * v6 * w5))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2));

    let term6 = -(2.0
        * cf
        * lmss
        * (16.0 * ca4 + v - 2.0 * ca2 * v - 79.0 * ca4 * v - 3.0 * v2
            + 6.0 * ca2 * v2
            + 189.0 * ca4 * v2
            + 4.0 * v3
            - 8.0 * ca2 * v3
            - 284.0 * ca4 * v3
            - 4.0 * v4
            + 8.0 * ca2 * v4
            + 284.0 * ca4 * v4
            + 3.0 * v5
            - 6.0 * ca2 * v5
            - 189.0 * ca4 * v5
            - v6
            + 2.0 * ca2 * v6
            + 79.0 * ca4 * v6
            - 16.0 * ca4 * v7
            + 48.0 * ca4 * v * w
            + 2.0 * v2 * w
            + 2.0 * ca2 * v2 * w
            - 228.0 * ca4 * v2 * w
            - 6.0 * v3 * w
            - 12.0 * ca2 * v3 * w
            + 514.0 * ca4 * v3 * w
            + 10.0 * v4 * w
            + 12.0 * ca2 * v4 * w
            - 694.0 * ca4 * v4 * w
            - 10.0 * v5 * w
            + 4.0 * ca2 * v5 * w
            + 590.0 * ca4 * v5 * w
            + 4.0 * v6 * w
            - 6.0 * ca2 * v6 * w
            - 302.0 * ca4 * v6 * w
            + 72.0 * ca4 * v7 * w
            + 48.0 * ca4 * v2 * w2
            + v3 * w2
            + 12.0 * ca2 * v3 * w2
            - 237.0 * ca4 * v3 * w2
            - 7.0 * v4 * w2
            - 40.0 * ca2 * v4 * w2
            + 543.0 * ca4 * v4 * w2
            + 11.0 * v5 * w2
            + 32.0 * ca2 * v5 * w2
            - 675.0 * ca4 * v5 * w2
            - 5.0 * v6 * w2
            - 12.0 * ca2 * v6 * w2
            + 457.0 * ca4 * v6 * w2
            + 8.0 * ca2 * v7 * w2
            - 136.0 * ca4 * v7 * w2
            + 16.0 * ca4 * v3 * w3
            + 2.0 * v4 * w3
            + 22.0 * ca2 * v4 * w3
            - 136.0 * ca4 * v4 * w3
            - 4.0 * v5 * w3
            - 52.0 * ca2 * v5 * w3
            + 336.0 * ca4 * v5 * w3
            + 2.0 * v6 * w3
            + 46.0 * ca2 * v6 * w3
            - 344.0 * ca4 * v6 * w3
            - 28.0 * ca2 * v7 * w3
            + 140.0 * ca4 * v7 * w3
            + 22.0 * ca2 * v5 * w4
            - 62.0 * ca4 * v5 * w4
            - 42.0 * ca2 * v6 * w4
            + 130.0 * ca4 * v6 * w4
            + 36.0 * ca2 * v7 * w4
            - 84.0 * ca4 * v7 * w4
            + 12.0 * ca2 * v6 * w5
            - 20.0 * ca4 * v6 * w5
            - 20.0 * ca2 * v7 * w5
            + 28.0 * ca4 * v7 * w5
            + 4.0 * ca2 * v7 * pre.w6
            - 4.0 * ca4 * v7 * pre.w6))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v + v * w).powi(3));

    term5 + term6
}

/// `STRUV14(W,V,X3,S)`, term 7: the pure (no-log) `CF`-polynomial piece.
fn qg_compton_gluon_frag_term7(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7, v8, v9) = (
        pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8, pre.v9,
    );
    let (w2, w3, w4, w5, w6, w7, w8) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8);
    let ca2 = pre.ca2;
    let ca4 = pre.ca4;

    -(cf * (96.0 * ca4 - 2.0 * v + 4.0 * ca2 * v - 434.0 * ca4 * v + 10.0 * v2 - 24.0 * ca2 * v2
        + 958.0 * ca4 * v2
        - 18.0 * v3
        + 48.0 * ca2 * v3
        - 1294.0 * ca4 * v3
        + 14.0 * v4
        - 40.0 * ca2 * v4
        + 1066.0 * ca4 * v4
        - 4.0 * v5
        + 12.0 * ca2 * v5
        - 520.0 * ca4 * v5
        + 160.0 * ca4 * v6
        - 32.0 * ca4 * v7
        + 4.0 * ca2 * w
        - 92.0 * ca4 * w
        - 18.0 * ca2 * v * w
        + 510.0 * ca4 * v * w
        - 6.0 * v2 * w
        + 92.0 * ca2 * v2 * w
        - 1118.0 * ca4 * v2 * w
        + 17.0 * v3 * w
        - 232.0 * ca2 * v3 * w
        + 1395.0 * ca4 * v3 * w
        - 7.0 * v4 * w
        + 252.0 * ca2 * v4 * w
        - 785.0 * ca4 * v4 * w
        - 13.0 * v5 * w
        - 110.0 * ca2 * v5 * w
        - 205.0 * ca4 * v5 * w
        + 9.0 * v6 * w
        + 12.0 * ca2 * v6 * w
        + 423.0 * ca4 * v6 * w
        - 192.0 * ca4 * v7 * w
        + 64.0 * ca4 * v8 * w
        + 4.0 * ca2 * v * w2
        - 92.0 * ca4 * v * w2
        - 6.0 * ca2 * v2 * w2
        - 54.0 * ca4 * v2 * w2
        + 2.0 * v3 * w2
        + 46.0 * ca2 * v3 * w2
        + 844.0 * ca4 * v3 * w2
        - 37.0 * v4 * w2
        - 122.0 * ca2 * v4 * w2
        - 1817.0 * ca4 * v4 * w2
        + 56.0 * v5 * w2
        + 124.0 * ca2 * v5 * w2
        + 2300.0 * ca4 * v5 * w2
        - 11.0 * v6 * w2
        - 80.0 * ca2 * v6 * w2
        - 1449.0 * ca4 * v6 * w2
        - 10.0 * v7 * w2
        + 34.0 * ca2 * v7 * w2
        + 396.0 * ca4 * v7 * w2
        - 96.0 * ca4 * v8 * w2
        - 32.0 * ca4 * v9 * w2
        - 8.0 * ca2 * v2 * w3
        + 184.0 * ca4 * v2 * w3
        + 36.0 * ca2 * v3 * w3
        - 1020.0 * ca4 * v3 * w3
        + 12.0 * v4 * w3
        + 12.0 * ca2 * v4 * w3
        + 1696.0 * ca4 * v4 * w3
        + 38.0 * v5 * w3
        - 216.0 * ca2 * v5 * w3
        - 1574.0 * ca4 * v5 * w3
        - 112.0 * v6 * w3
        + 314.0 * ca2 * v6 * w3
        + 514.0 * ca4 * v6 * w3
        + 71.0 * v7 * w3
        - 128.0 * ca2 * v7 * w3
        + 341.0 * ca4 * v7 * w3
        - 5.0 * v8 * w3
        - 26.0 * ca2 * v8 * w3
        - 129.0 * ca4 * v8 * w3
        + 128.0 * ca4 * v9 * w3
        - 8.0 * ca2 * v3 * w4
        + 184.0 * ca4 * v3 * w4
        + 12.0 * ca2 * v4 * w4
        - 180.0 * ca4 * v4 * w4
        - 58.0 * v5 * w4
        + 132.0 * ca2 * v5 * w4
        - 362.0 * ca4 * v5 * w4
        + 132.0 * v6 * w4
        - 304.0 * ca2 * v6 * w4
        + 948.0 * ca4 * v6 * w4
        - 96.0 * v7 * w4
        + 116.0 * ca2 * v7 * w4
        - 1092.0 * ca4 * v7 * w4
        + 9.0 * v8 * w4
        + 86.0 * ca2 * v8 * w4
        + 321.0 * ca4 * v8 * w4
        + 16.0 * ca2 * v9 * w4
        - 208.0 * ca4 * v9 * w4
        + 4.0 * ca2 * v4 * w5
        - 92.0 * ca4 * v4 * w5
        - 18.0 * ca2 * v5 * w5
        + 510.0 * ca4 * v5 * w5
        - 22.0 * v6 * w5
        + 80.0 * ca2 * v6 * w5
        - 746.0 * ca4 * v6 * w5
        + 41.0 * v7 * w5
        - 16.0 * ca2 * v7 * w5
        + 739.0 * ca4 * v7 * w5
        - 3.0 * v8 * w5
        - 86.0 * ca2 * v8 * w5
        - 199.0 * ca4 * v8 * w5
        - 64.0 * ca2 * v9 * w5
        + 192.0 * ca4 * v9 * w5
        + 4.0 * ca2 * v5 * w6
        - 92.0 * ca4 * v5 * w6
        - 6.0 * ca2 * v6 * w6
        + 138.0 * ca4 * v6 * w6
        - 6.0 * v7 * w6
        - 6.0 * ca2 * v7 * w6
        - 160.0 * ca4 * v7 * w6
        - v8 * w6
        + 18.0 * ca2 * v8 * w6
        + 47.0 * ca4 * v8 * w6
        + 96.0 * ca2 * v9 * w6
        - 128.0 * ca4 * v9 * w6
        + 8.0 * ca2 * v8 * w7
        - 8.0 * ca4 * v8 * w7
        - 64.0 * ca2 * v9 * w7
        + 64.0 * ca4 * v9 * w7
        + 16.0 * ca2 * v9 * w8
        - 16.0 * ca4 * v9 * w8))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(3))
}

/// `STRUV14(W,V,X3,S)`, term 8: the `lv` piece.
fn qg_compton_gluon_frag_term8(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7, v8, v9) = (
        pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8, pre.v9,
    );
    let (w2, w3, w4, w5, w6, w7, w8) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8);
    let ca2 = pre.ca2;
    let ca4 = pre.ca4;
    let lv = pre.lv;

    -(2.0
        * cf
        * lv
        * (4.0 * ca2 - 100.0 * ca4 - 18.0 * ca2 * v + 450.0 * ca4 * v - 4.0 * v2 + 40.0 * ca2 * v2
            - 948.0 * ca4 * v2
            + 8.0 * v3
            - 48.0 * ca2 * v3
            + 1224.0 * ca4 * v3
            + 24.0 * ca2 * v4
            - 1016.0 * ca4 * v4
            - 8.0 * v5
            + 2.0 * ca2 * v5
            + 534.0 * ca4 * v5
            + 4.0 * v6
            - 4.0 * ca2 * v6
            - 176.0 * ca4 * v6
            + 32.0 * ca4 * v7
            - 4.0 * ca2 * w
            + 20.0 * ca4 * w
            + 22.0 * ca2 * v * w
            - 190.0 * ca4 * v * w
            + 4.0 * v2 * w
            - 58.0 * ca2 * v2 * w
            + 462.0 * ca4 * v2 * w
            - 6.0 * v3 * w
            + 80.0 * ca2 * v3 * w
            - 570.0 * ca4 * v3 * w
            - 20.0 * v4 * w
            - 24.0 * ca2 * v4 * w
            + 244.0 * ca4 * v4 * w
            + 34.0 * v5 * w
            - 42.0 * ca2 * v5 * w
            + 292.0 * ca4 * v5 * w
            - 4.0 * v6 * w
            + 22.0 * ca2 * v6 * w
            - 410.0 * ca4 * v6 * w
            - 8.0 * v7 * w
            + 4.0 * ca2 * v7 * w
            + 216.0 * ca4 * v7 * w
            - 64.0 * ca4 * v8 * w
            - 4.0 * ca2 * v * w2
            + 20.0 * ca4 * v * w2
            - 2.0 * ca2 * v2 * w2
            + 170.0 * ca4 * v2 * w2
            + 4.0 * v3 * w2
            + 26.0 * ca2 * v3 * w2
            - 798.0 * ca4 * v3 * w2
            + 14.0 * v4 * w2
            - 60.0 * ca2 * v4 * w2
            + 1478.0 * ca4 * v4 * w2
            - 16.0 * v5 * w2
            + 56.0 * ca2 * v5 * w2
            - 1788.0 * ca4 * v5 * w2
            - 40.0 * v6 * w2
            + 34.0 * ca2 * v6 * w2
            + 1258.0 * ca4 * v6 * w2
            + 34.0 * v7 * w2
            - 50.0 * ca2 * v7 * w2
            - 460.0 * ca4 * v7 * w2
            + 4.0 * v8 * w2
            + 88.0 * ca4 * v8 * w2
            + 32.0 * ca4 * v9 * w2
            + 8.0 * ca2 * v2 * w3
            - 40.0 * ca4 * v2 * w3
            - 44.0 * ca2 * v3 * w3
            + 380.0 * ca4 * v3 * w3
            - 6.0 * v4 * w3
            + 60.0 * ca2 * v4 * w3
            - 726.0 * ca4 * v4 * w3
            - 7.0 * v5 * w3
            + 4.0 * ca2 * v5 * w3
            + 835.0 * ca4 * v5 * w3
            + 70.0 * v6 * w3
            - 118.0 * ca2 * v6 * w3
            - 468.0 * ca4 * v6 * w3
            - 43.0 * v7 * w3
            + 76.0 * ca2 * v7 * w3
            - 69.0 * ca4 * v7 * w3
            - 20.0 * v8 * w3
            + 30.0 * ca2 * v8 * w3
            + 130.0 * ca4 * v8 * w3
            - 128.0 * ca4 * v9 * w3
            + 8.0 * ca2 * v3 * w4
            - 40.0 * ca4 * v3 * w4
            - 8.0 * ca2 * v4 * w4
            - 40.0 * ca4 * v4 * w4
            + 4.0 * v5 * w4
            - 42.0 * ca2 * v5 * w4
            + 302.0 * ca4 * v5 * w4
            - 37.0 * v6 * w4
            + 112.0 * ca2 * v6 * w4
            - 475.0 * ca4 * v6 * w4
            + 20.0 * v7 * w4
            - 44.0 * ca2 * v7 * w4
            + 596.0 * ca4 * v7 * w4
            + 35.0 * v8 * w4
            - 64.0 * ca2 * v8 * w4
            - 327.0 * ca4 * v8 * w4
            - 16.0 * ca2 * v9 * w4
            + 216.0 * ca4 * v9 * w4
            - 4.0 * ca2 * v4 * w5
            + 20.0 * ca4 * v4 * w5
            + 22.0 * ca2 * v5 * w5
            - 190.0 * ca4 * v5 * w5
            + 6.0 * v6 * w5
            - 46.0 * ca2 * v6 * w5
            + 296.0 * ca4 * v6 * w5
            - 7.0 * v7 * w5
            + 20.0 * ca2 * v7 * w5
            - 389.0 * ca4 * v7 * w5
            - 26.0 * v8 * w5
            + 34.0 * ca2 * v8 * w5
            + 244.0 * ca4 * v8 * w5
            + 56.0 * ca2 * v9 * w5
            - 208.0 * ca4 * v9 * w5
            - 4.0 * ca2 * v5 * w6
            + 20.0 * ca4 * v5 * w6
            + 6.0 * ca2 * v6 * w6
            - 30.0 * ca4 * v6 * w6
            + 4.0 * v7 * w6
            - 6.0 * ca2 * v7 * w6
            + 74.0 * ca4 * v7 * w6
            + 7.0 * v8 * w6
            + 4.0 * ca2 * v8 * w6
            - 83.0 * ca4 * v8 * w6
            - 72.0 * ca2 * v9 * w6
            + 128.0 * ca4 * v9 * w6
            - 4.0 * ca2 * v8 * w7
            + 12.0 * ca4 * v8 * w7
            + 40.0 * ca2 * v9 * w7
            - 48.0 * ca4 * v9 * w7
            - 8.0 * ca2 * v9 * w8
            + 8.0 * ca4 * v9 * w8))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(3))
}

/// `STRUV14(W,V,X3,S)`, term 9: the `l1w` piece.
fn qg_compton_gluon_frag_term9(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7, v8, v9) = (
        pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8, pre.v9,
    );
    let (w2, w3, w4, w5, w6, w7, w8) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8);
    let ca2 = pre.ca2;
    let ca4 = pre.ca4;
    let l1w = pre.l1w;

    -(2.0
        * cf
        * l1w
        * (4.0 * ca2 - 96.0 * ca4 - 18.0 * ca2 * v + 432.0 * ca4 * v - 4.0 * v2 + 40.0 * ca2 * v2
            - 912.0 * ca4 * v2
            + 10.0 * v3
            - 50.0 * ca2 * v3
            + 1182.0 * ca4 * v3
            - 6.0 * v4
            + 30.0 * ca2 * v4
            - 986.0 * ca4 * v4
            - 2.0 * v5
            - 4.0 * ca2 * v5
            + 522.0 * ca4 * v5
            + 2.0 * v6
            - 2.0 * ca2 * v6
            - 174.0 * ca4 * v6
            + 32.0 * ca4 * v7
            - 4.0 * ca2 * w
            + 20.0 * ca4 * w
            + 22.0 * ca2 * v * w
            - 186.0 * ca4 * v * w
            + 4.0 * v2 * w
            - 66.0 * ca2 * v2 * w
            + 424.0 * ca4 * v2 * w
            - 10.0 * v3 * w
            + 112.0 * ca2 * v3 * w
            - 456.0 * ca4 * v3 * w
            - 6.0 * v4 * w
            - 70.0 * ca2 * v4 * w
            + 76.0 * ca4 * v4 * w
            + 22.0 * v5 * w
            - 18.0 * ca2 * v5 * w
            + 430.0 * ca4 * v5 * w
            - 6.0 * v6 * w
            + 24.0 * ca2 * v6 * w
            - 472.0 * ca4 * v6 * w
            - 4.0 * v7 * w
            + 228.0 * ca4 * v7 * w
            - 64.0 * ca4 * v8 * w
            - 4.0 * ca2 * v * w2
            + 20.0 * ca4 * v * w2
            - 2.0 * ca2 * v2 * w2
            + 170.0 * ca4 * v2 * w2
            + 4.0 * v3 * w2
            + 18.0 * ca2 * v3 * w2
            - 826.0 * ca4 * v3 * w2
            + 10.0 * v4 * w2
            - 60.0 * ca2 * v4 * w2
            + 1526.0 * ca4 * v4 * w2
            - 20.0 * v5 * w2
            + 100.0 * ca2 * v5 * w2
            - 1778.0 * ca4 * v5 * w2
            - 16.0 * v6 * w2
            - 18.0 * ca2 * v6 * w2
            + 1166.0 * ca4 * v6 * w2
            + 20.0 * v7 * w2
            - 36.0 * ca2 * v7 * w2
            - 368.0 * ca4 * v7 * w2
            + 2.0 * v8 * w2
            + 2.0 * ca2 * v8 * w2
            + 58.0 * ca4 * v8 * w2
            + 32.0 * ca4 * v9 * w2
            + 8.0 * ca2 * v2 * w3
            - 40.0 * ca4 * v2 * w3
            - 44.0 * ca2 * v3 * w3
            + 380.0 * ca4 * v3 * w3
            - 6.0 * v4 * w3
            + 76.0 * ca2 * v4 * w3
            - 682.0 * ca4 * v4 * w3
            + v5 * w3
            - 60.0 * ca2 * v5 * w3
            + 663.0 * ca4 * v5 * w3
            + 42.0 * v6 * w3
            - 66.0 * ca2 * v6 * w3
            - 206.0 * ca4 * v6 * w3
            - 31.0 * v7 * w3
            + 84.0 * ca2 * v7 * w3
            - 263.0 * ca4 * v7 * w3
            - 10.0 * v8 * w3
            + 20.0 * ca2 * v8 * w3
            + 168.0 * ca4 * v8 * w3
            - 112.0 * ca4 * v9 * w3
            + 8.0 * ca2 * v3 * w4
            - 40.0 * ca4 * v3 * w4
            - 8.0 * ca2 * v4 * w4
            - 52.0 * ca4 * v4 * w4
            + 4.0 * v5 * w4
            - 26.0 * ca2 * v5 * w4
            + 388.0 * ca4 * v5 * w4
            - 29.0 * v6 * w4
            + 112.0 * ca2 * v6 * w4
            - 607.0 * ca4 * v6 * w4
            + 22.0 * v7 * w4
            - 86.0 * ca2 * v7 * w4
            + 668.0 * ca4 * v7 * w4
            + 17.0 * v8 * w4
            - 50.0 * ca2 * v8 * w4
            - 281.0 * ca4 * v8 * w4
            - 16.0 * ca2 * v9 * w4
            + 160.0 * ca4 * v9 * w4
            - 4.0 * ca2 * v4 * w5
            + 20.0 * ca4 * v4 * w5
            + 22.0 * ca2 * v5 * w5
            - 202.0 * ca4 * v5 * w5
            + 6.0 * v6 * w5
            - 54.0 * ca2 * v6 * w5
            + 298.0 * ca4 * v6 * w5
            - 11.0 * v7 * w5
            + 52.0 * ca2 * v7 * w5
            - 339.0 * ca4 * v7 * w5
            - 12.0 * v8 * w5
            + 28.0 * ca2 * v8 * w5
            + 150.0 * ca4 * v8 * w5
            + 56.0 * ca2 * v9 * w5
            - 136.0 * ca4 * v9 * w5
            - 4.0 * ca2 * v5 * w6
            + 20.0 * ca4 * v5 * w6
            + 6.0 * ca2 * v6 * w6
            - 22.0 * ca4 * v6 * w6
            + 4.0 * v7 * w6
            - 14.0 * ca2 * v7 * w6
            + 34.0 * ca4 * v7 * w6
            + 3.0 * v8 * w6
            + 4.0 * ca2 * v8 * w6
            - 35.0 * ca4 * v8 * w6
            - 72.0 * ca2 * v9 * w6
            + 88.0 * ca4 * v9 * w6
            + 8.0 * ca4 * v7 * w7
            - 4.0 * ca2 * v8 * w7
            + 4.0 * ca4 * v8 * w7
            + 40.0 * ca2 * v9 * w7
            - 40.0 * ca4 * v9 * w7
            - 8.0 * ca2 * v9 * w8
            + 8.0 * ca4 * v9 * w8))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(3))
}

/// `STRUV14(W,V,X3,S)`, part C: the pure/`lv`/`l1w` term groups.
fn qg_compton_gluon_frag_part_c(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    qg_compton_gluon_frag_term7(w, v, ctx, pre)
        + qg_compton_gluon_frag_term8(w, v, ctx, pre)
        + qg_compton_gluon_frag_term9(w, v, ctx, pre)
}

/// `STRUV14(W,V,X3,S)`.
#[must_use]
pub fn qg_compton_gluon_frag(
    w: f64,
    v: f64,
    _x3: f64,
    _s: f64,
    ctx: &MeContext,
    pre: &Precalc,
) -> f64 {
    qg_compton_gluon_frag_part_a(w, v, ctx, pre)
        + qg_compton_gluon_frag_part_b(w, v, ctx, pre)
        + qg_compton_gluon_frag_part_c(w, v, ctx, pre)
}

/// `STRUV15(W,V,X3,S)`, part A: terms 1-4 (`lmss`/`lvw`(x2)/`l1v` `Nf`-and-
/// direct pieces).
fn gg_to_gg_gluon_frag_part_a(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let nf = ctx.nf;
    let (v2, v3, v4) = (pre.v2, pre.v3, pre.v4);
    let (w2, w3, w4) = (pre.w2, pre.w3, pre.w4);
    let ca3 = ca.powi(3);
    let (lmss, lvw, l1v) = (pre.lmss, pre.lvw, pre.l1v);

    let term1 = (256.0
        * ca3
        * lmss
        * nf
        * (1.0 - 2.0 * v + v2 + v2 * w2)
        * (1.0 + v2 - 2.0 * v2 * w + v2 * w2)
        * (4.0 - 8.0 * v + 4.0 * v2 - v * w + v2 * w + 4.0 * v2 * w2))
        / (243.0 * (1.0 - v) * v * w * (1.0 - v + v * w).powi(4));

    let term2 = (32.0
        * ca3
        * lvw
        * nf
        * (1.0 + v * w)
        * (81.0 - 143.0 * v + 79.0 * v2 + 62.0 * v * w - 15.0 * v2 * w + 17.0 * v2 * w2))
        / (243.0 * (1.0 - v) * v2 * w);

    let term3 = (128.0
        * ca3
        * lvw
        * (7.0 - 4.0 * v + 12.0 * v2 - 8.0 * v * w - 16.0 * v2 * w - 4.0 * v3 * w
            + 20.0 * v2 * w2
            + 8.0 * v4 * w2
            - 5.0 * v3 * w3
            - 12.0 * v4 * w3
            + 8.0 * v4 * w4))
        / ((1.0 - v) * v2 * w);

    let term4 = -(32.0
        * ca3
        * l1v
        * nf
        * (9.0 - 18.0 * v + 8.0 * v2 + v3 + 20.0 * w - 40.0 * v * w + 60.0 * v2 * w
            - 40.0 * v3 * w
            + v4 * w
            - 14.0 * v2 * w2
            + 14.0 * v3 * w2
            + 24.0 * v4 * w2
            + 20.0 * v2 * w3
            - 20.0 * v3 * w3
            - 24.0 * v4 * w3
            + 17.0 * v4 * w4))
        / (243.0 * (1.0 - v) * v * w * (1.0 - v * w) * (1.0 - v + v * w));

    term1 + term2 + term3 + term4
}

/// `STRUV15(W,V,X3,S)`, part B: terms 5-9 (`lms`(x2)/`l1v`/`lw`(x2)).
fn gg_to_gg_gluon_frag_part_b(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let nf = ctx.nf;
    let (v2, v3, v4, v5, v6) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6);
    let (w2, w3, w4, w5) = (pre.w2, pre.w3, pre.w4, pre.w5);
    let ca3 = ca.powi(3);
    let (lms, l1v, lw) = (pre.lms, pre.l1v, pre.lw);

    let term5 = (256.0
        * ca3
        * lms
        * nf
        * (18.0 - 31.0 * v + 25.0 * v2 - 4.0 * v3 - 36.0 * w + 18.0 * v * w + 11.0 * v2 * w
            - 16.0 * v3 * w
            + 8.0 * v4 * w
            + 72.0 * v * w2
            - 90.0 * v2 * w2
            + 31.0 * v3 * w2
            + 11.0 * v4 * w2
            - 4.0 * v5 * w2
            - 36.0 * v2 * w3
            + 54.0 * v3 * w3
            - 43.0 * v4 * w3
            + 16.0 * v5 * w3
            - 4.0 * v5 * w4))
        / (243.0 * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2));

    let term6 = -(128.0
        * ca3
        * l1v
        * (4.0 - 12.0 * v + 16.0 * v2 - 11.0 * v3 + 3.0 * v4 + 3.0 * v * w - 10.0 * v2 * w
            + 6.0 * v3 * w
            + 3.0 * v4 * w
            - 3.0 * v5 * w
            - 4.0 * v2 * w2
            + 16.0 * v3 * w2
            - 20.0 * v4 * w2
            + 6.0 * v5 * w2
            + 2.0 * v6 * w2
            - 3.0 * v3 * w3
            + 3.0 * v4 * w3
            + 6.0 * v5 * w3
            - 6.0 * v6 * w3
            - 5.0 * v5 * w4
            + 4.0 * v6 * w4))
        / ((1.0 - v) * v2 * w * (1.0 - v * w) * (1.0 - v + v * w));

    let term7 = (32.0
        * ca3
        * lw
        * nf
        * (144.0 - 360.0 * v + 304.0 * v2 - 88.0 * v3 + 18.0 * w - 45.0 * v * w + 186.0 * v2 * w
            - 258.0 * v3 * w
            + 117.0 * v4 * w
            - 24.0 * v2 * w2
            + 68.0 * v3 * w2
            - 80.0 * v4 * w2
            - 9.0 * v5 * w2
            + 20.0 * v2 * w3
            - 46.0 * v3 * w3
            + 190.0 * v4 * w3
            - 46.0 * v5 * w3
            - 148.0 * v4 * w4
            + 56.0 * v5 * w4
            + 2.0 * v4 * w5
            - v5 * w5))
        / (243.0 * (1.0 - v) * v2 * (1.0 - w) * w * (1.0 - v * w) * (1.0 - v + v * w));

    let term8 = -(256.0
        * ca3
        * lms
        * (6.0 - 8.0 * v + 12.0 * v2 - 4.0 * v3 + 2.0 * v4 - 4.0 * w - 6.0 * v * w + v2 * w
            - 16.0 * v3 * w
            + 5.0 * v4 * w
            - 4.0 * v5 * w
            + 8.0 * v * w2
            - 6.0 * v2 * w2
            + 19.0 * v3 * w2
            + 3.0 * v4 * w2
            + 2.0 * v5 * w2
            + 2.0 * v6 * w2
            - 4.0 * v2 * w3
            + 6.0 * v3 * w3
            - 19.0 * v4 * w3
            + 4.0 * v5 * w3
            - 3.0 * v6 * w3
            + 5.0 * v5 * w4
            + v6 * w4
            - 2.0 * v6 * w5))
        / ((1.0 - v) * v2 * w * (1.0 - v * w).powi(2));

    let term9 = (128.0
        * ca3
        * lw
        * (8.0 - 20.0 * v + 40.0 * v2 - 44.0 * v3 + 24.0 * v4 - 8.0 * v5 - 2.0 * w + 5.0 * v * w
            - 34.0 * v2 * w
            + 54.0 * v3 * w
            - 17.0 * v4 * w
            + 8.0 * v6 * w
            + 12.0 * v2 * w2
            - 25.0 * v3 * w2
            - 26.0 * v4 * w2
            + 28.0 * v5 * w2
            - 24.0 * v6 * w2
            + v3 * w3
            + 41.0 * v4 * w3
            - 38.0 * v5 * w3
            + 30.0 * v6 * w3
            - 17.0 * v4 * w4
            + 19.0 * v5 * w4
            - 18.0 * v6 * w4
            + 2.0 * v4 * w5
            - 3.0 * v5 * w5
            + 4.0 * v6 * w5))
        / ((1.0 - v) * v2 * (1.0 - w) * w * (1.0 - v * w) * (1.0 - v + v * w));

    term5 + term6 + term7 + term8 + term9
}

/// `STRUV15(W,V,X3,S)`, part C: terms 10-12 (`l1vw`(x2)/`lmss`).
fn gg_to_gg_gluon_frag_part_c(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let nf = ctx.nf;
    let (v2, v3, v4, v5, v6, v7, v8) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8);
    let (w2, w3, w4, w5, w6, w7) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7);
    let ca3 = ca.powi(3);
    let (l1vw, lmss) = (pre.l1vw, pre.lmss);

    let term10 = -(32.0
        * ca3
        * l1vw
        * nf
        * (81.0 - 324.0 * v + 486.0 * v2 - 243.0 * v3 - 243.0 * v4 + 486.0 * v5 - 324.0 * v6
            + 81.0 * v7
            + 405.0 * v * w
            - 1728.0 * v2 * w
            + 2750.0 * v3 * w
            - 1716.0 * v4 * w
            - 147.0 * v5 * w
            + 652.0 * v6 * w
            - 216.0 * v7 * w
            + 1161.0 * v2 * w2
            - 3808.0 * v3 * w2
            + 4684.0 * v4 * w2
            - 2354.0 * v5 * w2
            + 83.0 * v6 * w2
            + 234.0 * v7 * w2
            + 1301.0 * v3 * w3
            - 3824.0 * v4 * w3
            + 3660.0 * v5 * w3
            - 948.0 * v6 * w3
            - 189.0 * v7 * w3
            + 1099.0 * v4 * w4
            - 1812.0 * v5 * w4
            + 686.0 * v6 * w4
            + 189.0 * v7 * w4
            + 167.0 * v5 * w5
            - 176.0 * v6 * w5
            - 234.0 * v7 * w5
            + 27.0 * v6 * w6
            + 216.0 * v7 * w6
            - 81.0 * v7 * w7))
        / (243.0 * (1.0 - v) * v2 * w * (1.0 - v + v * w).powi(4));

    let term11 = (128.0
        * ca3
        * l1vw
        * (1.0 - 4.0 * v + 6.0 * v2 - 3.0 * v3 - 3.0 * v4 + 6.0 * v5 - 4.0 * v6
            + v7
            + 2.0 * v * w
            - 10.0 * v2 * w
            + 17.0 * v3 * w
            - 12.0 * v4 * w
            + 4.0 * v5 * w
            - 2.0 * v6 * w
            + v7 * w
            + 6.0 * v2 * w2
            - 17.0 * v3 * w2
            + 18.0 * v4 * w2
            - 20.0 * v5 * w2
            + 24.0 * v6 * w2
            - 11.0 * v7 * w2
            + 3.0 * v3 * w3
            - 4.0 * v4 * w3
            + 16.0 * v5 * w3
            - 36.0 * v6 * w3
            + 21.0 * v7 * w3
            + v4 * w4
            - 10.0 * v5 * w4
            + 32.0 * v6 * w4
            - 21.0 * v7 * w4
            + 4.0 * v5 * w5
            - 18.0 * v6 * w5
            + 11.0 * v7 * w5
            + 4.0 * v6 * w6
            - v7 * w6
            - v7 * w7))
        / ((1.0 - v) * v2 * w * (1.0 - v + v * w).powi(4));

    let term12 = -(256.0
        * ca3
        * lmss
        * (2.0 - 12.0 * v + 34.0 * v2 - 60.0 * v3 + 72.0 * v4 - 60.0 * v5 + 34.0 * v6 - 12.0 * v7
            + 2.0 * v8
            + 8.0 * v * w
            - 47.0 * v2 * w
            + 128.0 * v3 * w
            - 209.0 * v4 * w
            + 221.0 * v5 * w
            - 152.0 * v6 * w
            + 63.0 * v7 * w
            - 12.0 * v8 * w
            + 12.0 * v2 * w2
            - 72.0 * v3 * w2
            + 193.0 * v4 * w2
            - 291.0 * v5 * w2
            + 262.0 * v6 * w2
            - 135.0 * v7 * w2
            + 31.0 * v8 * w2
            + 8.0 * v3 * w3
            - 63.0 * v4 * w3
            + 173.0 * v5 * w3
            - 230.0 * v6 * w3
            + 158.0 * v7 * w3
            - 46.0 * v8 * w3
            + 2.0 * v4 * w4
            - 41.0 * v5 * w4
            + 109.0 * v6 * w4
            - 112.0 * v7 * w4
            + 44.0 * v8 * w4
            - 23.0 * v6 * w5
            + 47.0 * v7 * w5
            - 28.0 * v8 * w5
            - 9.0 * v7 * w6
            + 11.0 * v8 * w6
            - 2.0 * v8 * w7))
        / ((1.0 - v) * v2 * w * (1.0 - v + v * w).powi(4));

    term10 + term11 + term12
}

/// `STRUV15(W,V,X3,S)`, term 13: the `lv*Nf` piece.
fn gg_to_gg_gluon_frag_term13(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let nf = ctx.nf;
    let (v2, v3, v4, v5, v6, v7, v8, v9) = (
        pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8, pre.v9,
    );
    let (w2, w3, w4, w5, w6, w7) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7);
    let ca3 = ca.powi(3);
    let lv = pre.lv;

    -(64.0
        * ca3
        * lv
        * nf
        * (72.0 - 396.0 * v + 964.0 * v2 - 1344.0 * v3 + 1136.0 * v4 - 556.0 * v5 + 132.0 * v6
            - 8.0 * v7
            - 144.0 * w
            + 936.0 * v * w
            - 2504.0 * v2 * w
            + 3644.0 * v3 * w
            - 3112.0 * v4 * w
            + 1479.0 * v5 * w
            - 269.0 * v6 * w
            - 47.0 * v7 * w
            + 17.0 * v8 * w
            - 288.0 * v * w2
            + 936.0 * v2 * w2
            - 676.0 * v3 * w2
            - 636.0 * v4 * w2
            + 1242.0 * v5 * w2
            - 928.0 * v6 * w2
            + 445.0 * v7 * w2
            - 86.0 * v8 * w2
            - 9.0 * v9 * w2
            + 144.0 * v2 * w3
            - 1656.0 * v3 * w3
            + 3984.0 * v4 * w3
            - 3510.0 * v5 * w3
            + 858.0 * v6 * w3
            + 524.0 * v7 * w3
            - 489.0 * v8 * w3
            + 145.0 * v9 * w3
            + 576.0 * v3 * w4
            - 2088.0 * v4 * w4
            + 1644.0 * v5 * w4
            + 1640.0 * v6 * w4
            - 3114.0 * v7 * w4
            + 1824.0 * v8 * w4
            - 427.0 * v9 * w4
            + 144.0 * v4 * w5
            + 504.0 * v5 * w5
            - 2648.0 * v6 * w5
            + 3492.0 * v7 * w5
            - 2194.0 * v8 * w5
            + 573.0 * v9 * w5
            - 288.0 * v5 * w6
            + 1080.0 * v6 * w6
            - 1436.0 * v7 * w6
            + 1200.0 * v8 * w6
            - 400.0 * v9 * w6
            - 144.0 * v6 * w7
            + 216.0 * v7 * w7
            - 272.0 * v8 * w7
            + 118.0 * v9 * w7))
        / (243.0 * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(4))
}

/// `STRUV15(W,V,X3,S)`, term 14: the pure `Nf` (no-log) piece.
fn gg_to_gg_gluon_frag_term14(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let nf = ctx.nf;
    let (v2, v3, v4, v5, v6, v7, v8, v9, v10) = (
        pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8, pre.v9, pre.v10,
    );
    let (w2, w3, w4, w5, w6, w7, w8) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8);
    let ca3 = ca.powi(3);

    (64.0
        * ca3
        * nf
        * (396.0 - 2194.0 * v + 5268.0 * v2 - 7100.0 * v3 + 5740.0 * v4 - 2706.0 * v5 + 644.0 * v6
            - 48.0 * v7
            - 594.0 * w
            + 4059.0 * v * w
            - 11099.0 * v2 * w
            + 16005.0 * v3 * w
            - 13032.0 * v4 * w
            + 5603.0 * v5 * w
            - 789.0 * v6 * w
            - 195.0 * v7 * w
            + 42.0 * v8 * w
            - 1188.0 * v * w2
            + 3762.0 * v2 * w2
            - 1970.0 * v3 * w2
            - 5006.0 * v4 * w2
            + 8199.0 * v5 * w2
            - 5551.0 * v6 * w2
            + 2183.0 * v7 * w2
            - 489.0 * v8 * w2
            + 60.0 * v9 * w2
            + 594.0 * v2 * w3
            - 7227.0 * v3 * w3
            + 18163.0 * v4 * w3
            - 17138.0 * v5 * w3
            + 4815.0 * v6 * w3
            + 2413.0 * v7 * w3
            - 2038.0 * v8 * w3
            + 472.0 * v9 * w3
            - 54.0 * v10 * w3
            + 2376.0 * v3 * w4
            - 8712.0 * v4 * w4
            + 6820.0 * v5 * w4
            + 7870.0 * v6 * w4
            - 14853.0 * v7 * w4
            + 8361.0 * v8 * w4
            - 1834.0 * v9 * w4
            + 162.0 * v10 * w4
            + 594.0 * v4 * w5
            + 2277.0 * v5 * w5
            - 11867.0 * v6 * w5
            + 16105.0 * v7 * w5
            - 9783.0 * v8 * w5
            + 2388.0 * v9 * w5
            - 162.0 * v10 * w5
            - 1188.0 * v5 * w6
            + 4554.0 * v6 * w6
            - 6280.0 * v7 * w6
            + 4936.0 * v8 * w6
            - 1510.0 * v9 * w6
            + 54.0 * v10 * w6
            - 594.0 * v6 * w7
            + 891.0 * v7 * w7
            - 1029.0 * v8 * w7
            + 472.0 * v9 * w7
            - 48.0 * v9 * w8))
        / (243.0 * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(4))
}

/// `STRUV15(W,V,X3,S)`, term 15: the `l1w*Nf` piece.
fn gg_to_gg_gluon_frag_term15(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let nf = ctx.nf;
    let (v2, v3, v4, v5, v6, v7, v8, v9) = (
        pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8, pre.v9,
    );
    let (w2, w3, w4, w5, w6, w7, w8) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8);
    let ca3 = ca.powi(3);
    let l1w = pre.l1w;

    -(32.0
        * ca3
        * l1w
        * nf
        * (126.0 - 693.0 * v + 1685.0 * v2 - 2329.0 * v3 + 1916.0 * v4 - 875.0 * v5
            + 169.0 * v6
            + v7
            - 288.0 * w
            + 1836.0 * v * w
            - 4864.0 * v2 * w
            + 7029.0 * v3 * w
            - 5944.0 * v4 * w
            + 2806.0 * v5 * w
            - 576.0 * v6 * w
            + v7 * w
            - 576.0 * v * w2
            + 1852.0 * v2 * w2
            - 1300.0 * v3 * w2
            - 1327.0 * v4 * w2
            + 2411.0 * v5 * w2
            - 1544.0 * v6 * w2
            + 582.0 * v7 * w2
            - 97.0 * v8 * w2
            - v9 * w2
            + 288.0 * v2 * w3
            - 3316.0 * v3 * w3
            + 7964.0 * v4 * w3
            - 6883.0 * v5 * w3
            + 1284.0 * v6 * w3
            + 1464.0 * v7 * w3
            - 1016.0 * v8 * w3
            + 215.0 * v9 * w3
            + 1152.0 * v3 * w4
            - 4160.0 * v4 * w4
            + 3166.0 * v5 * w4
            + 3639.0 * v6 * w4
            - 6567.0 * v7 * w4
            + 3588.0 * v8 * w4
            - 707.0 * v9 * w4
            + 288.0 * v4 * w5
            + 1044.0 * v5 * w5
            - 5440.0 * v6 * w5
            + 7127.0 * v7 * w5
            - 4276.0 * v8 * w5
            + 986.0 * v9 * w5
            - 576.0 * v5 * w6
            + 2180.0 * v6 * w6
            - 2900.0 * v7 * w6
            + 2339.0 * v8 * w6
            - 707.0 * v9 * w6
            - 288.0 * v6 * w7
            + 436.0 * v7 * w7
            - 540.0 * v8 * w7
            + 215.0 * v9 * w7
            + 2.0 * v8 * w8
            - v9 * w8))
        / (243.0 * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(4))
}

/// `STRUV15(W,V,X3,S)`, term 16: the pure (no-log, no-`Nf`) polynomial piece.
fn gg_to_gg_gluon_frag_term16(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let (v2, v3, v4, v5, v6, v7, v8, v9, v10) = (
        pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8, pre.v9, pre.v10,
    );
    let (w2, w3, w4, w5, w6, w7, w8, w9) = (
        pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8, pre.w9,
    );
    let ca3 = ca.powi(3);

    -(64.0
        * ca3
        * (82.0 - 451.0 * v + 1162.0 * v2 - 1826.0 * v3 + 1858.0 * v4 - 1219.0 * v5 + 514.0 * v6
            - 144.0 * v7
            + 24.0 * v8
            - 114.0 * w
            + 791.0 * v * w
            - 2241.0 * v2 * w
            + 3637.0 * v3 * w
            - 3601.0 * v4 * w
            + 1907.0 * v5 * w
            - 241.0 * v6 * w
            - 247.0 * v7 * w
            + 157.0 * v8 * w
            - 48.0 * v9 * w
            - 228.0 * v * w2
            + 716.0 * v2 * w2
            - 501.0 * v3 * w2
            - 794.0 * v4 * w2
            + 2348.0 * v5 * w2
            - 2794.0 * v6 * w2
            + 1691.0 * v7 * w2
            - 580.0 * v8 * w2
            + 118.0 * v9 * w2
            + 24.0 * v10 * w2
            + 114.0 * v2 * w3
            - 1411.0 * v3 * w3
            + 3703.0 * v4 * w3
            - 4400.0 * v5 * w3
            + 2809.0 * v6 * w3
            - 589.0 * v7 * w3
            - 225.0 * v8 * w3
            + 130.0 * v9 * w3
            - 131.0 * v10 * w3
            + 456.0 * v3 * w4
            - 1678.0 * v4 * w4
            + 1489.0 * v5 * w4
            + 956.0 * v6 * w4
            - 2877.0 * v7 * w4
            + 2109.0 * v8 * w4
            - 639.0 * v9 * w4
            + 297.0 * v10 * w4
            + 114.0 * v4 * w5
            + 449.0 * v5 * w5
            - 2185.0 * v6 * w5
            + 3357.0 * v7 * w5
            - 2448.0 * v8 * w5
            + 786.0 * v9 * w5
            - 393.0 * v10 * w5
            - 228.0 * v5 * w6
            + 880.0 * v6 * w6
            - 1321.0 * v7 * w6
            + 1208.0 * v8 * w6
            - 525.0 * v9 * w6
            + 371.0 * v10 * w6
            - 114.0 * v6 * w7
            + 171.0 * v7 * w7
            - 245.0 * v8 * w7
            + 226.0 * v9 * w7
            - 264.0 * v10 * w7
            - 48.0 * v9 * w8
            + 120.0 * v10 * w8
            - 24.0 * v10 * w9))
        / (3.0 * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(4))
}

/// `STRUV15(W,V,X3,S)`, term 17: the `l1w` piece.
fn gg_to_gg_gluon_frag_term17(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let (v2, v3, v4, v5, v6, v7, v8, v9, v10) = (
        pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8, pre.v9, pre.v10,
    );
    let (w2, w3, w4, w5, w6, w7, w8, w9) = (
        pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8, pre.w9,
    );
    let ca3 = ca.powi(3);
    let l1w = pre.l1w;

    (128.0
        * ca3
        * l1w
        * (22.0 - 121.0 * v + 309.0 * v2 - 485.0 * v3 + 508.0 * v4 - 359.0 * v5 + 169.0 * v6
            - 51.0 * v7
            + 8.0 * v8
            - 8.0 * w
            + 88.0 * v * w
            - 306.0 * v2 * w
            + 580.0 * v3 * w
            - 664.0 * v4 * w
            + 414.0 * v5 * w
            - 76.0 * v6 * w
            - 66.0 * v7 * w
            + 54.0 * v8 * w
            - 16.0 * v9 * w
            - 16.0 * v * w2
            + 32.0 * v2 * w2
            + 63.0 * v3 * w2
            - 316.0 * v4 * w2
            + 678.0 * v5 * w2
            - 851.0 * v6 * w2
            + 596.0 * v7 * w2
            - 237.0 * v8 * w2
            + 43.0 * v9 * w2
            + 8.0 * v10 * w2
            + 8.0 * v2 * w3
            - 168.0 * v3 * w3
            + 521.0 * v4 * w3
            - 864.0 * v5 * w3
            + 927.0 * v6 * w3
            - 524.0 * v7 * w3
            + 110.0 * v8 * w3
            + 36.0 * v9 * w3
            - 46.0 * v10 * w3
            + 32.0 * v3 * w4
            - 132.0 * v4 * w4
            + 148.0 * v5 * w4
            - 25.0 * v6 * w4
            - 248.0 * v7 * w4
            + 351.0 * v8 * w4
            - 209.0 * v9 * w4
            + 106.0 * v10 * w4
            + 8.0 * v4 * w5
            + 72.0 * v5 * w5
            - 250.0 * v6 * w5
            + 428.0 * v7 * w5
            - 443.0 * v8 * w5
            + 244.0 * v9 * w5
            - 134.0 * v10 * w5
            - 16.0 * v5 * w6
            + 80.0 * v6 * w6
            - 137.0 * v7 * w6
            + 184.0 * v8 * w6
            - 137.0 * v9 * w6
            + 114.0 * v10 * w6
            - 8.0 * v6 * w7
            + 8.0 * v7 * w7
            - 25.0 * v8 * w7
            + 52.0 * v9 * w7
            - 76.0 * v10 * w7
            - 2.0 * v8 * w8
            - 13.0 * v9 * w8
            + 36.0 * v10 * w8
            - 8.0 * v10 * w9))
        / ((1.0 - v) * v2 * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(4))
}

/// `STRUV15(W,V,X3,S)`, term 18: the `lv` piece.
fn gg_to_gg_gluon_frag_term18(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let (v2, v3, v4, v5, v6, v7, v8, v9, v10) = (
        pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8, pre.v9, pre.v10,
    );
    let (w2, w3, w4, w5, w6, w7, w8, w9) = (
        pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8, pre.w9,
    );
    let ca3 = ca.powi(3);
    let lv = pre.lv;

    (128.0
        * ca3
        * lv
        * (24.0 - 132.0 * v + 336.0 * v2 - 524.0 * v3 + 544.0 * v4 - 380.0 * v5 + 176.0 * v6
            - 52.0 * v7
            + 8.0 * v8
            - 8.0 * w
            + 92.0 * v * w
            - 324.0 * v2 * w
            + 618.0 * v3 * w
            - 714.0 * v4 * w
            + 456.0 * v5 * w
            - 96.0 * v6 * w
            - 62.0 * v7 * w
            + 54.0 * v8 * w
            - 16.0 * v9 * w
            - 16.0 * v * w2
            + 32.0 * v2 * w2
            + 64.0 * v3 * w2
            - 312.0 * v4 * w2
            + 665.0 * v5 * w2
            - 849.0 * v6 * w2
            + 613.0 * v7 * w2
            - 251.0 * v8 * w2
            + 46.0 * v9 * w2
            + 8.0 * v10 * w2
            + 8.0 * v2 * w3
            - 172.0 * v3 * w3
            + 534.0 * v4 * w3
            - 883.0 * v5 * w3
            + 971.0 * v6 * w3
            - 602.0 * v7 * w3
            + 165.0 * v8 * w3
            + 27.0 * v9 * w3
            - 48.0 * v10 * w3
            + 32.0 * v3 * w4
            - 136.0 * v4 * w4
            + 164.0 * v5 * w4
            - 74.0 * v6 * w4
            - 151.0 * v7 * w4
            + 281.0 * v8 * w4
            - 212.0 * v9 * w4
            + 118.0 * v10 * w4
            + 8.0 * v4 * w5
            + 68.0 * v5 * w5
            - 234.0 * v6 * w5
            + 388.0 * v7 * w5
            - 419.0 * v8 * w5
            + 280.0 * v9 * w5
            - 162.0 * v10 * w5
            - 16.0 * v5 * w6
            + 80.0 * v6 * w6
            - 140.0 * v7 * w6
            + 198.0 * v8 * w6
            - 182.0 * v9 * w6
            + 146.0 * v10 * w6
            - 8.0 * v6 * w7
            + 12.0 * v7 * w7
            - 36.0 * v8 * w7
            + 73.0 * v9 * w7
            - 94.0 * v10 * w7
            - 16.0 * v9 * w8
            + 40.0 * v10 * w8
            - 8.0 * v10 * w9))
        / ((1.0 - v) * v2 * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(4))
}

/// `STRUV15(W,V,X3,S)`, part D: terms 13-18.
fn gg_to_gg_gluon_frag_part_d(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    gg_to_gg_gluon_frag_term13(w, v, ctx, pre)
        + gg_to_gg_gluon_frag_term14(w, v, ctx, pre)
        + gg_to_gg_gluon_frag_term15(w, v, ctx, pre)
        + gg_to_gg_gluon_frag_term16(w, v, ctx, pre)
        + gg_to_gg_gluon_frag_term17(w, v, ctx, pre)
        + gg_to_gg_gluon_frag_term18(w, v, ctx, pre)
}

/// `STRUV15(W,V,X3,S)`.
#[must_use]
pub fn gg_to_gg_gluon_frag(
    w: f64,
    v: f64,
    _x3: f64,
    _s: f64,
    ctx: &MeContext,
    pre: &Precalc,
) -> f64 {
    gg_to_gg_gluon_frag_part_a(w, v, ctx, pre)
        + gg_to_gg_gluon_frag_part_b(w, v, ctx, pre)
        + gg_to_gg_gluon_frag_part_c(w, v, ctx, pre)
        + gg_to_gg_gluon_frag_part_d(w, v, ctx, pre)
}

/// `STRUV16(W,V,X3,S)`, part A: terms 1-4 (`lvw`/`l1v`/`lms`/`lw`).
fn gg_to_qqbar_quark_frag_part_a(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6);
    let (w2, w3, w4) = (pre.w2, pre.w3, pre.w4);
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let ca4 = pre.ca4;
    let (lvw, l1v, lms, lw) = (pre.lvw, pre.l1v, pre.lms, pre.lw);

    let term1 = -(4.0
        * cf
        * lvw
        * (2.0 - 2.0 * ca2 + 4.0 * ca4 - 3.0 * v + 3.0 * ca2 * v - 7.0 * ca4 * v + 3.0 * v2
            - 11.0 * ca2 * v2
            + 15.0 * ca4 * v2
            + v * w
            + ca2 * v * w
            - 11.0 * ca4 * v * w
            - 2.0 * v2 * w
            + 17.0 * ca2 * v2 * w
            - 13.0 * ca4 * v2 * w
            - 2.0 * v3 * w
            + ca2 * v3 * w
            - 14.0 * ca4 * v3 * w
            - 11.0 * ca2 * v2 * w2
            + 35.0 * ca4 * v2 * w2
            + 3.0 * v3 * w2
            - 3.0 * ca2 * v3 * w2
            - ca4 * v3 * w2
            + 32.0 * ca4 * v4 * w2
            - v3 * w3
            + ca2 * v3 * w3
            - 21.0 * ca4 * v3 * w3
            - 48.0 * ca4 * v4 * w3
            + 32.0 * ca4 * v4 * w4))
        / (ca * (1.0 - v) * v2 * w);

    let term2 = (4.0
        * cf
        * l1v
        * (4.0 * ca4 + v + ca2 * v - 19.0 * ca4 * v - v2 - ca2 * v2
            + 39.0 * ca4 * v2
            + v3
            + ca2 * v3
            - 37.0 * ca4 * v3
            - v4
            - ca2 * v4
            + 13.0 * ca4 * v4
            + 6.0 * ca4 * v * w
            + 2.0 * ca2 * v2 * w
            - 29.0 * ca4 * v2 * w
            - v3 * w
            - 3.0 * ca2 * v3 * w
            + 31.0 * ca4 * v3 * w
            + 2.0 * v4 * w
            + 4.0 * ca2 * v4 * w
            + 5.0 * ca4 * v4 * w
            + v5 * w
            - ca2 * v5 * w
            - 13.0 * ca4 * v5 * w
            - 4.0 * ca4 * v2 * w2
            + 2.0 * ca2 * v3 * w2
            + 38.0 * ca4 * v3 * w2
            - v4 * w2
            - 5.0 * ca2 * v4 * w2
            - 68.0 * ca4 * v4 * w2
            - 3.0 * v5 * w2
            + 3.0 * ca2 * v5 * w2
            + 24.0 * ca4 * v5 * w2
            + 8.0 * ca4 * v6 * w2
            - 6.0 * ca4 * v3 * w3
            + 2.0 * ca2 * v4 * w3
            + 11.0 * ca4 * v4 * w3
            + 3.0 * v5 * w3
            - 3.0 * ca2 * v5 * w3
            + 24.0 * ca4 * v5 * w3
            - 24.0 * ca4 * v6 * w3
            - v5 * w4
            + ca2 * v5 * w4
            - 21.0 * ca4 * v5 * w4
            + 16.0 * ca4 * v6 * w4))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v * w) * (1.0 - v + v * w));

    let term3 = (2.0
        * cf
        * lms
        * (1.0 - 6.0 * ca2 + 5.0 * ca4 + 16.0 * ca3 * cf - v + 24.0 * ca2 * v
            - 39.0 * ca4 * v
            - v2
            - 32.0 * ca2 * v2
            + 81.0 * ca4 * v2
            + v3
            - 2.0 * ca2 * v3
            - 63.0 * ca4 * v3
            + 32.0 * ca4 * v4
            - 2.0 * w
            + 12.0 * ca2 * w
            - 10.0 * ca4 * w
            - 4.0 * ca2 * v * w
            + 20.0 * ca4 * v * w
            - 32.0 * ca3 * cf * v * w
            + 4.0 * v2 * w
            - 14.0 * ca2 * v2 * w
            - 22.0 * ca4 * v2 * w
            + 64.0 * ca2 * v3 * w
            - 80.0 * ca4 * v3 * w
            - 2.0 * v4 * w
            + 4.0 * ca2 * v4 * w
            + 78.0 * ca4 * v4 * w
            - 64.0 * ca4 * v5 * w
            + 4.0 * v * w2
            - 24.0 * ca2 * v * w2
            + 20.0 * ca4 * v * w2
            - 3.0 * v2 * w2
            + 26.0 * ca2 * v2 * w2
            - 55.0 * ca4 * v2 * w2
            + 16.0 * ca3 * cf * v2 * w2
            - 7.0 * v3 * w2
            - 32.0 * ca2 * v3 * w2
            + 183.0 * ca4 * v3 * w2
            + 7.0 * v4 * w2
            - 52.0 * ca2 * v4 * w2
            - 67.0 * ca4 * v4 * w2
            + v5 * w2
            - 2.0 * ca2 * v5 * w2
            + 33.0 * ca4 * v5 * w2
            + 32.0 * ca4 * v6 * w2
            - 2.0 * v2 * w3
            + 12.0 * ca2 * v2 * w3
            - 10.0 * ca4 * v2 * w3
            + 2.0 * v3 * w3
            - 16.0 * ca2 * v3 * w3
            + 30.0 * ca4 * v3 * w3
            + v4 * w3
            + 36.0 * ca2 * v4 * w3
            - 181.0 * ca4 * v4 * w3
            - 4.0 * v5 * w3
            + 8.0 * ca2 * v5 * w3
            + 60.0 * ca4 * v5 * w3
            - 48.0 * ca4 * v6 * w3
            + v5 * w4
            - 2.0 * ca2 * v5 * w4
            + 81.0 * ca4 * v5 * w4
            + 16.0 * ca4 * v6 * w4
            - 32.0 * ca4 * v6 * pre.w5))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2));

    let term4 = -(4.0
        * cf
        * lw
        * (1.0 - 6.0 * ca2 + 5.0 * ca4 - 2.0 * v + 13.0 * ca2 * v - 27.0 * ca4 * v + v2
            - 20.0 * ca2 * v2
            + 83.0 * ca4 * v2
            + 13.0 * ca2 * v3
            - 125.0 * ca4 * v3
            + 96.0 * ca4 * v4
            - 32.0 * ca4 * v5
            + 2.0 * ca2 * w
            - 2.0 * ca4 * w
            + v * w
            + 12.0 * ca4 * v * w
            + 11.0 * ca2 * v2 * w
            - 89.0 * ca4 * v2 * w
            - v3 * w
            - 12.0 * ca2 * v3 * w
            + 173.0 * ca4 * v3 * w
            - 13.0 * ca2 * v4 * w
            - 114.0 * ca4 * v4 * w
            + 32.0 * ca4 * v6 * w
            - 8.0 * ca2 * v2 * w2
            + 45.0 * ca4 * v2 * w2
            + 13.0 * ca2 * v3 * w2
            - 98.0 * ca4 * v3 * w2
            + v4 * w2
            + 30.0 * ca2 * v4 * w2
            - 5.0 * ca4 * v4 * w2
            + 113.0 * ca4 * v5 * w2
            - 96.0 * ca4 * v6 * w2
            - 6.0 * ca4 * v2 * w3
            - 5.0 * ca2 * v3 * w3
            + 9.0 * ca4 * v3 * w3
            - 34.0 * ca2 * v4 * w3
            + 91.0 * ca4 * v4 * w3
            - v5 * w3
            + 2.0 * ca2 * v5 * w3
            - 155.0 * ca4 * v5 * w3
            + 120.0 * ca4 * v6 * w3
            - v4 * w4
            + 18.0 * ca2 * v4 * w4
            - 48.0 * ca4 * v4 * w4
            + 2.0 * v5 * w4
            - 3.0 * ca2 * v5 * w4
            + 77.0 * ca4 * v5 * w4
            - 72.0 * ca4 * v6 * w4
            - 2.0 * ca2 * v4 * pre.w5
            + 8.0 * ca4 * v4 * pre.w5
            - v5 * pre.w5
            + ca2 * v5 * pre.w5
            - 11.0 * ca4 * v5 * pre.w5
            + 16.0 * ca4 * v6 * pre.w5))
        / (ca * (1.0 - v) * v2 * (1.0 - w) * w * (1.0 - v * w) * (1.0 - v + v * w));

    term1 + term2 + term3 + term4
}

/// `STRUV16(W,V,X3,S)`, term 5: the `l1vw` piece.
fn gg_to_qqbar_quark_frag_term5(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6);
    let (w2, w3, w4, w5, w6) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6);
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let ca4 = pre.ca4;
    let ca3cf = ca3 * cf;
    let ca2cf2 = ca2 * cf.powi(2);
    let l1vw = pre.l1vw;

    (4.0 * cf
        * l1vw
        * (3.0 - 2.0 * ca2 + 3.0 * ca4 - 8.0 * ca2cf2 - 14.0 * v + 11.0 * ca2 * v
            - 15.0 * ca4 * v
            - 8.0 * ca * cf * v
            + 8.0 * ca3cf * v
            + 24.0 * ca2cf2 * v
            + 27.0 * v2
            - 25.0 * ca2 * v2
            + 35.0 * ca4 * v2
            + 28.0 * ca * cf * v2
            - 28.0 * ca3cf * v2
            - 28.0 * ca2cf2 * v2
            - 28.0 * v3
            + 30.0 * ca2 * v3
            - 50.0 * ca4 * v3
            - 40.0 * ca * cf * v3
            + 40.0 * ca3cf * v3
            + 16.0 * ca2cf2 * v3
            + 17.0 * v4
            - 20.0 * ca2 * v4
            + 45.0 * ca4 * v4
            + 30.0 * ca * cf * v4
            - 30.0 * ca3cf * v4
            - 4.0 * ca2cf2 * v4
            - 6.0 * v5
            + 7.0 * ca2 * v5
            - 23.0 * ca4 * v5
            - 12.0 * ca * cf * v5
            + 12.0 * ca3cf * v5
            + v6
            - ca2 * v6
            + 5.0 * ca4 * v6
            + 2.0 * ca * cf * v6
            - 2.0 * ca3cf * v6
            + 12.0 * v * w
            - 6.0 * ca2 * v * w
            + ca4 * v * w
            + 8.0 * ca * cf * v * w
            - 8.0 * ca2cf2 * v * w
            - 44.0 * v2 * w
            + 29.0 * ca2 * v2 * w
            - 8.0 * ca4 * v2 * w
            - 40.0 * ca * cf * v2 * w
            + 8.0 * ca3cf * v2 * w
            + 24.0 * ca2cf2 * v2 * w
            + 64.0 * v3 * w
            - 56.0 * ca2 * v3 * w
            + 18.0 * ca4 * v3 * w
            + 80.0 * ca * cf * v3 * w
            - 28.0 * ca3cf * v3 * w
            - 24.0 * ca2cf2 * v3 * w
            - 48.0 * v4 * w
            + 54.0 * ca2 * v4 * w
            - 16.0 * ca4 * v4 * w
            - 80.0 * ca * cf * v4 * w
            + 36.0 * ca3cf * v4 * w
            + 8.0 * ca2cf2 * v4 * w
            + 20.0 * v5 * w
            - 26.0 * ca2 * v5 * w
            + 5.0 * ca4 * v5 * w
            + 40.0 * ca * cf * v5 * w
            - 20.0 * ca3cf * v5 * w
            - 4.0 * v6 * w
            + 5.0 * ca2 * v6 * w
            - 8.0 * ca * cf * v6 * w
            + 4.0 * ca3cf * v6 * w
            + 19.0 * v2 * w2
            - 9.0 * ca2 * v2 * w2
            + 3.0 * ca4 * v2 * w2
            + 12.0 * ca * cf * v2 * w2
            - 4.0 * ca3cf * v2 * w2
            - 12.0 * ca2cf2 * v2 * w2
            - 52.0 * v3 * w2
            + 38.0 * ca2 * v3 * w2
            - 6.0 * ca4 * v3 * w2
            - 56.0 * ca * cf * v3 * w2
            + 24.0 * ca3cf * v3 * w2
            + 16.0 * ca2cf2 * v3 * w2
            + 54.0 * v4 * w2
            - 60.0 * ca2 * v4 * w2
            - 10.0 * ca4 * v4 * w2
            + 84.0 * ca * cf * v4 * w2
            - 36.0 * ca3cf * v4 * w2
            - 8.0 * ca2cf2 * v4 * w2
            - 28.0 * v5 * w2
            + 42.0 * ca2 * v5 * w2
            + 26.0 * ca4 * v5 * w2
            - 56.0 * ca * cf * v5 * w2
            + 24.0 * ca3cf * v5 * w2
            + 7.0 * v6 * w2
            - 11.0 * ca2 * v6 * w2
            - 13.0 * ca4 * v6 * w2
            + 14.0 * ca * cf * v6 * w2
            - 6.0 * ca3cf * v6 * w2
            + 16.0 * v3 * w3
            - 12.0 * ca2 * v3 * w3
            + 6.0 * ca4 * v3 * w3
            + 16.0 * ca * cf * v3 * w3
            - 4.0 * ca3cf * v3 * w3
            - 8.0 * ca2cf2 * v3 * w3
            - 32.0 * v4 * w3
            + 38.0 * ca2 * v4 * w3
            - 48.0 * ca * cf * v4 * w3
            + 20.0 * ca3cf * v4 * w3
            + 8.0 * ca2cf2 * v4 * w3
            + 24.0 * v5 * w3
            - 40.0 * ca2 * v5 * w3
            - 22.0 * ca4 * v5 * w3
            + 48.0 * ca * cf * v5 * w3
            - 24.0 * ca3cf * v5 * w3
            - 8.0 * v6 * w3
            + 14.0 * ca2 * v6 * w3
            + 16.0 * ca4 * v6 * w3
            - 16.0 * ca * cf * v6 * w3
            + 8.0 * ca3cf * v6 * w3
            + 9.0 * v4 * w4
            - 12.0 * ca2 * v4 * w4
            - 3.0 * ca4 * v4 * w4
            + 14.0 * ca * cf * v4 * w4
            - 6.0 * ca3cf * v4 * w4
            - 4.0 * ca2cf2 * v4 * w4
            - 14.0 * v5 * w4
            + 23.0 * ca2 * v5 * w4
            + 21.0 * ca4 * v5 * w4
            - 28.0 * ca * cf * v5 * w4
            + 12.0 * ca3cf * v5 * w4
            + 7.0 * v6 * w4
            - 11.0 * ca2 * v6 * w4
            - 13.0 * ca4 * v6 * w4
            + 14.0 * ca * cf * v6 * w4
            - 6.0 * ca3cf * v6 * w4
            + 4.0 * v5 * w5
            - 6.0 * ca2 * v5 * w5
            - 7.0 * ca4 * v5 * w5
            + 8.0 * ca * cf * v5 * w5
            - 4.0 * ca3cf * v5 * w5
            - 4.0 * v6 * w5
            + 5.0 * ca2 * v6 * w5
            - 8.0 * ca * cf * v6 * w5
            + 4.0 * ca3cf * v6 * w5
            + v6 * w6
            - ca2 * v6 * w6
            + 5.0 * ca4 * v6 * w6
            + 2.0 * ca * cf * v6 * w6
            - 2.0 * ca3cf * v6 * w6))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v + v * w).powi(3))
}

/// `STRUV16(W,V,X3,S)`, term 6: the `lmss` piece.
fn gg_to_qqbar_quark_frag_term6(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7);
    let (w2, w3, w4, w5, w6) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6);
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let cacf2 = ca * cf.powi(2);
    let ca2cf = ca2 * cf;
    let lmss = pre.lmss;

    (4.0 * cf
        * lmss
        * (4.0 * cacf2 - 8.0 * ca3 * v + 4.0 * cf * v - 4.0 * ca2cf * v - 12.0 * cacf2 * v
            + 48.0 * ca3 * v2
            - 14.0 * cf * v2
            + 14.0 * ca2cf * v2
            + 14.0 * cacf2 * v2
            - 128.0 * ca3 * v3
            + 20.0 * cf * v3
            - 20.0 * ca2cf * v3
            - 8.0 * cacf2 * v3
            + 192.0 * ca3 * v4
            - 15.0 * cf * v4
            + 15.0 * ca2cf * v4
            + 2.0 * cacf2 * v4
            - 168.0 * ca3 * v5
            + 6.0 * cf * v5
            - 6.0 * ca2cf * v5
            + 80.0 * ca3 * v6
            - cf * v6
            + ca2cf * v6
            - 16.0 * ca3 * v7
            + 12.0 * cacf2 * v * w
            - 36.0 * ca3 * v2 * w
            + 6.0 * cf * v2 * w
            - 14.0 * ca2cf * v2 * w
            - 42.0 * cacf2 * v2 * w
            + 196.0 * ca3 * v3 * w
            - 20.0 * cf * v3 * w
            + 58.0 * ca2cf * v3 * w
            + 56.0 * cacf2 * v3 * w
            - 444.0 * ca3 * v4 * w
            + 25.0 * cf * v4 * w
            - 99.0 * ca2cf * v4 * w
            - 34.0 * cacf2 * v4 * w
            + 516.0 * ca3 * v5 * w
            - 14.0 * cf * v5 * w
            + 88.0 * ca2cf * v5 * w
            + 8.0 * cacf2 * v5 * w
            - 304.0 * ca3 * v6 * w
            + 3.0 * cf * v6 * w
            - 41.0 * ca2cf * v6 * w
            + 72.0 * ca3 * v7 * w
            + 8.0 * ca2cf * v7 * w
            + 12.0 * cacf2 * v2 * w2
            - 84.0 * ca3 * v3 * w2
            + 8.0 * cf * v3 * w2
            - 14.0 * ca2cf * v3 * w2
            - 36.0 * cacf2 * v3 * w2
            + 376.0 * ca3 * v4 * w2
            - 17.0 * cf * v4 * w2
            + 51.0 * ca2cf * v4 * w2
            + 42.0 * cacf2 * v4 * w2
            - 652.0 * ca3 * v5 * w2
            + 14.0 * cf * v5 * w2
            - 80.0 * ca2cf * v5 * w2
            - 16.0 * cacf2 * v5 * w2
            + 512.0 * ca3 * v6 * w2
            - 4.0 * cf * v6 * w2
            + 58.0 * ca2cf * v6 * w2
            - 152.0 * ca3 * v7 * w2
            - 16.0 * ca2cf * v7 * w2
            + 4.0 * cacf2 * v3 * w3
            - 116.0 * ca3 * v4 * w3
            + 7.0 * cf * v4 * w3
            - 7.0 * ca2cf * v4 * w3
            - 10.0 * cacf2 * v4 * w3
            + 404.0 * ca3 * v5 * w3
            - 10.0 * cf * v5 * w3
            + 16.0 * ca2cf * v5 * w3
            + 8.0 * cacf2 * v5 * w3
            - 480.0 * ca3 * v6 * w3
            + 4.0 * cf * v6 * w3
            - 18.0 * ca2cf * v6 * w3
            + 192.0 * ca3 * v7 * w3
            + 8.0 * ca2cf * v7 * w3
            - 104.0 * ca3 * v5 * w4
            + 4.0 * cf * v5 * w4
            - 2.0 * ca2cf * v5 * w4
            + 248.0 * ca3 * v6 * w4
            - 3.0 * cf * v6 * w4
            + ca2cf * v6 * w4
            - 152.0 * ca3 * v7 * w4
            - 56.0 * ca3 * v6 * w5
            + cf * v6 * w5
            - ca2cf * v6 * w5
            + 72.0 * ca3 * v7 * w5
            - 16.0 * ca3 * v7 * w6))
        / ((1.0 - v) * v2 * w * (1.0 - v + v * w).powi(3))
}

/// `STRUV16(W,V,X3,S)`, part B: terms 5-6.
fn gg_to_qqbar_quark_frag_part_b(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    gg_to_qqbar_quark_frag_term5(w, v, ctx, pre) + gg_to_qqbar_quark_frag_term6(w, v, ctx, pre)
}

/// `STRUV16(W,V,X3,S)`, term 7: the pure (no-log) `CF`-polynomial piece.
fn gg_to_qqbar_quark_frag_term7(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7, v8, v9) = (
        pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8, pre.v9,
    );
    let (w2, w3, w4, w5, w6, w7, w8) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8);
    let ca2 = pre.ca2;
    let ca4 = pre.ca4;

    (cf * (5.0 - 30.0 * ca2 + 25.0 * ca4 - 22.0 * v + 140.0 * ca2 * v - 150.0 * ca4 * v
        + 32.0 * v2
        - 312.0 * ca2 * v2
        + 536.0 * ca4 * v2
        - 8.0 * v3
        + 384.0 * ca2 * v3
        - 1208.0 * ca4 * v3
        - 23.0 * v4
        - 230.0 * ca2 * v4
        + 1661.0 * ca4 * v4
        + 22.0 * v5
        + 36.0 * ca2 * v5
        - 1370.0 * ca4 * v5
        - 6.0 * v6
        + 12.0 * ca2 * v6
        + 634.0 * ca4 * v6
        - 128.0 * ca4 * v7
        - 10.0 * w
        + 48.0 * ca2 * w
        - 38.0 * ca4 * w
        + 39.0 * v * w
        - 252.0 * ca2 * v * w
        + 249.0 * ca4 * v * w
        - 61.0 * v2 * w
        + 538.0 * ca2 * v2 * w
        - 797.0 * ca4 * v2 * w
        + 47.0 * v3 * w
        - 632.0 * ca2 * v3 * w
        + 1677.0 * ca4 * v3 * w
        - 27.0 * v4 * w
        + 312.0 * ca2 * v4 * w
        - 2033.0 * ca4 * v4 * w
        + 34.0 * v5 * w
        + 56.0 * ca2 * v5 * w
        + 1006.0 * ca4 * v5 * w
        - 34.0 * v6 * w
        - 34.0 * ca2 * v6 * w
        + 392.0 * ca4 * v6 * w
        + 12.0 * v7 * w
        - 36.0 * ca2 * v7 * w
        - 712.0 * ca4 * v7 * w
        + 256.0 * ca4 * v8 * w
        - 10.0 * v * w2
        + 48.0 * ca2 * v * w2
        - 38.0 * ca4 * v * w2
        - 6.0 * v2 * w2
        - 18.0 * ca2 * v2 * w2
        + 60.0 * ca4 * v2 * w2
        + 48.0 * v3 * w2
        - 318.0 * ca2 * v3 * w2
        + 114.0 * ca4 * v3 * w2
        - v4 * w2
        + 786.0 * ca2 * v4 * w2
        - 905.0 * ca4 * v4 * w2
        - 104.0 * v5 * w2
        - 856.0 * ca2 * v5 * w2
        + 2512.0 * ca4 * v5 * w2
        + 103.0 * v6 * w2
        + 272.0 * ca2 * v6 * w2
        - 3379.0 * ca4 * v6 * w2
        - 24.0 * v7 * w2
        + 50.0 * ca2 * v7 * w2
        + 2242.0 * ca4 * v7 * w2
        - 6.0 * v8 * w2
        + 36.0 * ca2 * v8 * w2
        - 478.0 * ca4 * v8 * w2
        - 128.0 * ca4 * v9 * w2
        + 20.0 * v2 * w3
        - 96.0 * ca2 * v2 * w3
        + 76.0 * ca4 * v2 * w3
        - 78.0 * v3 * w3
        + 504.0 * ca2 * v3 * w3
        - 498.0 * ca4 * v3 * w3
        + 40.0 * v4 * w3
        - 756.0 * ca2 * v4 * w3
        + 1316.0 * ca4 * v4 * w3
        + 105.0 * v5 * w3
        + 378.0 * ca2 * v5 * w3
        - 2283.0 * ca4 * v5 * w3
        - 117.0 * v6 * w3
        + 380.0 * ca2 * v6 * w3
        + 2573.0 * ca4 * v6 * w3
        + 4.0 * v7 * w3
        - 434.0 * ca2 * v7 * w3
        - 1454.0 * ca4 * v7 * w3
        + 26.0 * v8 * w3
        - 28.0 * ca2 * v8 * w3
        - 222.0 * ca4 * v8 * w3
        - 12.0 * ca2 * v9 * w3
        + 556.0 * ca4 * v9 * w3
        + 20.0 * v3 * w4
        - 96.0 * ca2 * v3 * w4
        + 76.0 * ca4 * v3 * w4
        - 3.0 * v4 * w4
        + 126.0 * ca2 * v4 * w4
        - 195.0 * ca4 * v4 * w4
        - 78.0 * v5 * w4
        + 228.0 * ca2 * v5 * w4
        + 106.0 * ca4 * v5 * w4
        + 75.0 * v6 * w4
        - 786.0 * ca2 * v6 * w4
        + 31.0 * ca4 * v6 * w4
        + 28.0 * v7 * w4
        + 692.0 * ca2 * v7 * w4
        - 304.0 * ca4 * v7 * w4
        - 48.0 * v8 * w4
        + 12.0 * ca2 * v8 * w4
        + 1140.0 * ca4 * v8 * w4
        + 24.0 * ca2 * v9 * w4
        - 1080.0 * ca4 * v9 * w4
        - 10.0 * v4 * w5
        + 48.0 * ca2 * v4 * w5
        - 38.0 * ca4 * v4 * w5
        + 39.0 * v5 * w5
        - 252.0 * ca2 * v5 * w5
        + 249.0 * ca4 * v5 * w5
        - 25.0 * v6 * w5
        + 454.0 * ca2 * v6 * w5
        - 453.0 * ca4 * v6 * w5
        - 32.0 * v7 * w5
        - 434.0 * ca2 * v7 * w5
        + 534.0 * ca4 * v7 * w5
        + 48.0 * v8 * w5
        - 48.0 * ca2 * v8 * w5
        - 1168.0 * ca4 * v8 * w5
        - 12.0 * ca2 * v9 * w5
        + 1292.0 * ca4 * v9 * w5
        - 10.0 * v5 * w6
        + 48.0 * ca2 * v5 * w6
        - 38.0 * ca4 * v5 * w6
        + 4.0 * v6 * w6
        - 78.0 * ca2 * v6 * w6
        + 110.0 * ca4 * v6 * w6
        + 12.0 * v7 * w6
        + 110.0 * ca2 * v7 * w6
        - 158.0 * ca4 * v7 * w6
        - 26.0 * v8 * w6
        + 40.0 * ca2 * v8 * w6
        + 626.0 * ca4 * v8 * w6
        - 1056.0 * ca4 * v9 * w6
        + 6.0 * v8 * w7
        - 12.0 * ca2 * v8 * w7
        - 154.0 * ca4 * v8 * w7
        + 544.0 * ca4 * v9 * w7
        - 128.0 * ca4 * v9 * w8))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(3))
}

/// `STRUV16(W,V,X3,S)`, term 8: the `l1w` piece.
fn gg_to_qqbar_quark_frag_term8(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7, v8, v9) = (
        pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8, pre.v9,
    );
    let (w2, w3, w4, w5, w6, w7, w8) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8);
    let ca2 = pre.ca2;
    let ca4 = pre.ca4;
    let l1w = pre.l1w;

    -(2.0
        * cf
        * l1w
        * (3.0 - 22.0 * ca2 + 19.0 * ca4 - 12.0 * v + 108.0 * ca2 * v - 138.0 * ca4 * v
            + 20.0 * v2
            - 238.0 * ca2 * v2
            + 472.0 * ca4 * v2
            - 16.0 * v3
            + 288.0 * ca2 * v3
            - 932.0 * ca4 * v3
            + 3.0 * v4
            - 186.0 * ca2 * v4
            + 1123.0 * ca4 * v4
            + 4.0 * v5
            + 52.0 * ca2 * v5
            - 826.0 * ca4 * v5
            - 2.0 * v6
            - 2.0 * ca2 * v6
            + 346.0 * ca4 * v6
            - 64.0 * ca4 * v7
            - 2.0 * w
            + 12.0 * ca2 * w
            - 10.0 * ca4 * w
            + 11.0 * v * w
            - 74.0 * ca2 * v * w
            + 79.0 * ca4 * v * w
            - 16.0 * v2 * w
            + 182.0 * ca2 * v2 * w
            - 378.0 * ca4 * v2 * w
            - 2.0 * v3 * w
            - 226.0 * ca2 * v3 * w
            + 976.0 * ca4 * v3 * w
            + 30.0 * v4 * w
            + 82.0 * ca2 * v4 * w
            - 1244.0 * ca4 * v4 * w
            - 29.0 * v5 * w
            + 76.0 * ca2 * v5 * w
            + 613.0 * ca4 * v5 * w
            + 4.0 * v6 * w
            - 44.0 * ca2 * v6 * w
            + 216.0 * ca4 * v6 * w
            + 4.0 * v7 * w
            - 8.0 * ca2 * v7 * w
            - 380.0 * ca4 * v7 * w
            + 128.0 * ca4 * v8 * w
            - 2.0 * v * w2
            + 12.0 * ca2 * v * w2
            - 10.0 * ca4 * v * w2
            - 4.0 * v2 * w2
            + 32.0 * ca2 * v2 * w2
            - 24.0 * ca4 * v2 * w2
            + 30.0 * v3 * w2
            - 210.0 * ca2 * v3 * w2
            + 200.0 * ca4 * v3 * w2
            - 44.0 * v4 * w2
            + 452.0 * ca2 * v4 * w2
            - 840.0 * ca4 * v4 * w2
            + 8.0 * v5 * w2
            - 498.0 * ca2 * v5 * w2
            + 1960.0 * ca4 * v5 * w2
            + 38.0 * v6 * w2
            + 194.0 * ca2 * v6 * w2
            - 2350.0 * ca4 * v6 * w2
            - 24.0 * v7 * w2
            + 1390.0 * ca4 * v7 * w2
            - 2.0 * v8 * w2
            + 18.0 * ca2 * v8 * w2
            - 262.0 * ca4 * v8 * w2
            - 64.0 * ca4 * v9 * w2
            + 4.0 * v2 * w3
            - 24.0 * ca2 * v2 * w3
            + 20.0 * ca4 * v2 * w3
            - 22.0 * v3 * w3
            + 152.0 * ca2 * v3 * w3
            - 174.0 * ca4 * v3 * w3
            + 20.0 * v4 * w3
            - 262.0 * ca2 * v4 * w3
            + 632.0 * ca4 * v4 * w3
            + 36.0 * v5 * w3
            + 192.0 * ca2 * v5 * w3
            - 1404.0 * ca4 * v5 * w3
            - 82.0 * v6 * w3
            + 104.0 * ca2 * v6 * w3
            + 1584.0 * ca4 * v6 * w3
            + 34.0 * v7 * w3
            - 166.0 * ca2 * v7 * w3
            - 672.0 * ca4 * v7 * w3
            + 14.0 * v8 * w3
            - 12.0 * ca2 * v8 * w3
            - 262.0 * ca4 * v8 * w3
            - 8.0 * ca2 * v9 * w3
            + 296.0 * ca4 * v9 * w3
            + 4.0 * v3 * w4
            - 24.0 * ca2 * v3 * w4
            + 20.0 * ca4 * v3 * w4
            - v4 * w4
            + 2.0 * ca2 * v4 * w4
            - 9.0 * ca4 * v4 * w4
            - 30.0 * v5 * w4
            + 124.0 * ca2 * v5 * w4
            - 72.0 * ca4 * v5 * w4
            + 54.0 * v6 * w4
            - 306.0 * ca2 * v6 * w4
            + 266.0 * ca4 * v6 * w4
            - 8.0 * v7 * w4
            + 270.0 * ca2 * v7 * w4
            - 608.0 * ca4 * v7 * w4
            - 32.0 * v8 * w4
            + 6.0 * ca2 * v8 * w4
            + 836.0 * ca4 * v8 * w4
            + 16.0 * ca2 * v9 * w4
            - 528.0 * ca4 * v9 * w4
            - 2.0 * v4 * w5
            + 12.0 * ca2 * v4 * w5
            - 10.0 * ca4 * v4 * w5
            + 11.0 * v5 * w5
            - 82.0 * ca2 * v5 * w5
            + 111.0 * ca4 * v5 * w5
            - 14.0 * v6 * w5
            + 138.0 * ca2 * v6 * w5
            - 244.0 * ca4 * v6 * w5
            - 10.0 * v7 * w5
            - 134.0 * ca2 * v7 * w5
            + 348.0 * ca4 * v7 * w5
            + 32.0 * v8 * w5
            - 30.0 * ca2 * v8 * w5
            - 556.0 * ca4 * v8 * w5
            - 8.0 * ca2 * v9 * w5
            + 520.0 * ca4 * v9 * w5
            - 2.0 * v5 * w6
            + 12.0 * ca2 * v5 * w6
            - 10.0 * ca4 * v5 * w6
            + 2.0 * v6 * w6
            - 12.0 * ca2 * v6 * w6
            + 14.0 * ca4 * v6 * w6
            + 4.0 * v7 * w6
            + 18.0 * ca2 * v7 * w6
            + 26.0 * ca4 * v7 * w6
            - 14.0 * v8 * w6
            + 20.0 * ca2 * v8 * w6
            + 158.0 * ca4 * v8 * w6
            - 384.0 * ca4 * v9 * w6
            + 4.0 * ca2 * v7 * w7
            - 16.0 * ca4 * v7 * w7
            + 2.0 * v8 * w7
            - 2.0 * ca2 * v8 * w7
            - 42.0 * ca4 * v8 * w7
            + 224.0 * ca4 * v9 * w7
            - 64.0 * ca4 * v9 * w8))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(3))
}

/// `STRUV16(W,V,X3,S)`, term 9: the `lv` piece.
fn gg_to_qqbar_quark_frag_term9(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let (v2, v3, v4, v5, v6, v7, v8, v9) = (
        pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8, pre.v9,
    );
    let (w2, w3, w4, w5, w6, w7, w8) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8);
    let ca2 = pre.ca2;
    let ca4 = pre.ca4;
    let lv = pre.lv;

    -(2.0
        * cf
        * lv
        * (3.0 - 26.0 * ca2 + 23.0 * ca4 - 14.0 * v + 122.0 * ca2 * v - 164.0 * ca4 * v
            + 26.0 * v2
            - 260.0 * ca2 * v2
            + 546.0 * ca4 * v2
            - 24.0 * v3
            + 312.0 * ca2 * v3
            - 1048.0 * ca4 * v3
            + 11.0 * v4
            - 206.0 * ca2 * v4
            + 1227.0 * ca4 * v4
            - 2.0 * v5
            + 62.0 * ca2 * v5
            - 876.0 * ca4 * v5
            - 4.0 * ca2 * v6
            + 356.0 * ca4 * v6
            - 64.0 * ca4 * v7
            - 2.0 * w
            + 12.0 * ca2 * w
            - 10.0 * ca4 * w
            + 11.0 * v * w
            - 78.0 * ca2 * v * w
            + 83.0 * ca4 * v * w
            - 18.0 * v2 * w
            + 192.0 * ca2 * v2 * w
            - 414.0 * ca4 * v2 * w
            + 2.0 * v3 * w
            - 246.0 * ca2 * v3 * w
            + 1084.0 * ca4 * v3 * w
            + 22.0 * v4 * w
            + 106.0 * ca2 * v4 * w
            - 1388.0 * ca4 * v4 * w
            - 21.0 * v5 * w
            + 72.0 * ca2 * v5 * w
            + 697.0 * ca4 * v5 * w
            + 6.0 * v6 * w
            - 54.0 * ca2 * v6 * w
            + 204.0 * ca4 * v6 * w
            - 4.0 * ca2 * v7 * w
            - 384.0 * ca4 * v7 * w
            + 128.0 * ca4 * v8 * w
            - 2.0 * v * w2
            + 12.0 * ca2 * v * w2
            - 10.0 * ca4 * v * w2
            - 4.0 * v2 * w2
            + 36.0 * ca2 * v2 * w2
            - 16.0 * ca4 * v2 * w2
            + 32.0 * v3 * w2
            - 216.0 * ca2 * v3 * w2
            + 160.0 * ca4 * v3 * w2
            - 46.0 * v4 * w2
            + 462.0 * ca2 * v4 * w2
            - 784.0 * ca4 * v4 * w2
            + 12.0 * v5 * w2
            - 542.0 * ca2 * v5 * w2
            + 1974.0 * ca4 * v5 * w2
            + 18.0 * v6 * w2
            + 242.0 * ca2 * v6 * w2
            - 2448.0 * ca4 * v6 * w2
            - 10.0 * v7 * w2
            - 10.0 * ca2 * v7 * w2
            + 1472.0 * ca4 * v7 * w2
            + 16.0 * ca2 * v8 * w2
            - 284.0 * ca4 * v8 * w2
            - 64.0 * ca4 * v9 * w2
            + 4.0 * v2 * w3
            - 24.0 * ca2 * v2 * w3
            + 20.0 * ca4 * v2 * w3
            - 22.0 * v3 * w3
            + 156.0 * ca2 * v3 * w3
            - 166.0 * ca4 * v3 * w3
            + 22.0 * v4 * w3
            - 280.0 * ca2 * v4 * w3
            + 658.0 * ca4 * v4 * w3
            + 28.0 * v5 * w3
            + 256.0 * ca2 * v5 * w3
            - 1564.0 * ca4 * v5 * w3
            - 54.0 * v6 * w3
            + 44.0 * ca2 * v6 * w3
            + 1878.0 * ca4 * v6 * w3
            + 18.0 * v7 * w3
            - 170.0 * ca2 * v7 * w3
            - 916.0 * ca4 * v7 * w3
            + 4.0 * v8 * w3
            - 2.0 * ca2 * v8 * w3
            - 202.0 * ca4 * v8 * w3
            - 8.0 * ca2 * v9 * w3
            + 312.0 * ca4 * v9 * w3
            + 4.0 * v3 * w4
            - 24.0 * ca2 * v3 * w4
            + 20.0 * ca4 * v3 * w4
            - v4 * w4
            + 6.0 * ca2 * v4 * w4
            - 37.0 * ca4 * v4 * w4
            - 28.0 * v5 * w4
            + 94.0 * ca2 * v5 * w4
            + 62.0 * ca4 * v5 * w4
            + 40.0 * v6 * w4
            - 284.0 * ca2 * v6 * w4
            + 28.0 * ca4 * v6 * w4
            - 4.0 * v7 * w4
            + 306.0 * ca2 * v7 * w4
            - 410.0 * ca4 * v7 * w4
            - 12.0 * v8 * w4
            - 14.0 * ca2 * v8 * w4
            + 846.0 * ca4 * v8 * w4
            + 16.0 * ca2 * v9 * w4
            - 608.0 * ca4 * v9 * w4
            - 2.0 * v4 * w5
            + 12.0 * ca2 * v4 * w5
            - 10.0 * ca4 * v4 * w5
            + 11.0 * v5 * w5
            - 78.0 * ca2 * v5 * w5
            + 83.0 * ca4 * v5 * w5
            - 12.0 * v6 * w5
            + 144.0 * ca2 * v6 * w5
            - 212.0 * ca4 * v6 * w5
            - 6.0 * v7 * w5
            - 178.0 * ca2 * v7 * w5
            + 368.0 * ca4 * v7 * w5
            + 12.0 * v8 * w5
            - 10.0 * ca2 * v8 * w5
            - 690.0 * ca4 * v8 * w5
            - 8.0 * ca2 * v9 * w5
            + 664.0 * ca4 * v9 * w5
            - 2.0 * v5 * w6
            + 12.0 * ca2 * v5 * w6
            - 10.0 * ca4 * v5 * w6
            + 2.0 * v6 * w6
            - 16.0 * ca2 * v6 * w6
            + 30.0 * ca4 * v6 * w6
            + 2.0 * v7 * w6
            + 40.0 * ca2 * v7 * w6
            - 42.0 * ca4 * v7 * w6
            - 4.0 * v8 * w6
            + 10.0 * ca2 * v8 * w6
            + 266.0 * ca4 * v8 * w6
            - 496.0 * ca4 * v9 * w6
            - 64.0 * ca4 * v8 * w7
            + 256.0 * ca4 * v9 * w7
            - 64.0 * ca4 * v9 * w8))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(3))
}

/// `STRUV16(W,V,X3,S)`, part C: terms 7-9.
fn gg_to_qqbar_quark_frag_part_c(w: f64, v: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    gg_to_qqbar_quark_frag_term7(w, v, ctx, pre)
        + gg_to_qqbar_quark_frag_term8(w, v, ctx, pre)
        + gg_to_qqbar_quark_frag_term9(w, v, ctx, pre)
}

/// `STRUV16(W,V,X3,S)`.
#[must_use]
pub fn gg_to_qqbar_quark_frag(
    w: f64,
    v: f64,
    _x3: f64,
    _s: f64,
    ctx: &MeContext,
    pre: &Precalc,
) -> f64 {
    gg_to_qqbar_quark_frag_part_a(w, v, ctx, pre)
        + gg_to_qqbar_quark_frag_part_b(w, v, ctx, pre)
        + gg_to_qqbar_quark_frag_part_c(w, v, ctx, pre)
}

/// `STRU(XUHA,...,XGPROA,XUHB,...,XGPROB,XDUP,...,XDGP,GPPV,GPPC)`: combines
/// parton densities of hadrons A/B with fragmentation functions into the
/// flavor-summed weight for each of the 16 channels. Confirmed byte-for-byte
/// identical to `polarized/me-pol-ms.f`'s `STRU` and the unpolarized
/// package's `STRU` -- the flavor combinatorics don't depend on whether the
/// PDFs are polarized, only the numeric values fed in do. `GPPV` uses A as
/// the "unintegrated" (v-side) hadron and B as the "collinear" (w-side)
/// one; `GPPC` swaps A and B (Fortran's convention for the two `DPLUS`
/// terms).
///
/// Bottom-quark densities (`PartonDensities::bottom`, always zero here)
/// are never referenced, matching the Fortran subroutine's argument list.
#[must_use]
pub fn stru(
    a: &crate::pdfs::PartonDensities,
    b: &crate::pdfs::PartonDensities,
    ff: &crate::pdfs::FragmentationFunctions,
) -> ([f64; 16], [f64; 16]) {
    let (xuha, xubha, xdha, xdbha, xsha, xcha, xgproa) =
        (a.up, a.upb, a.down, a.downb, a.strange, a.charm, a.gluon);
    let (xuhb, xubhb, xdhb, xdbhb, xshb, xchb, xgprob) =
        (b.up, b.upb, b.down, b.downb, b.strange, b.charm, b.gluon);
    let (xdup, xdubp, xddp, xddbp, xdsp, xdsbp, xdcp, xdcbp, xdgp) =
        (ff.u, ff.ub, ff.d, ff.db, ff.s, ff.sb, ff.c, ff.cb, ff.g);

    let mut gppv = [0.0_f64; 16];
    let mut gppc = [0.0_f64; 16];

    gppv[0] = xuha * (xdhb + xshb + xchb) * xdup
        + xdha * (xuhb + xshb + xchb) * xddp
        + xsha * (xuhb + xdhb + xchb) * xdsp
        + xcha * (xuhb + xdhb + xshb) * xdcp
        + xubha * (xdbhb + xshb + xchb) * xdubp
        + xdbha * (xubhb + xshb + xchb) * xddbp
        + xsha * (xubhb + xdbhb + xchb) * xdsbp
        + xcha * (xubhb + xdbhb + xshb) * xdcbp;
    gppc[0] = xuhb * (xdha + xsha + xcha) * xdup
        + xdhb * (xuha + xsha + xcha) * xddp
        + xshb * (xuha + xdha + xcha) * xdsp
        + xchb * (xuha + xdha + xsha) * xdcp
        + xubhb * (xdbha + xsha + xcha) * xdubp
        + xdbhb * (xubha + xsha + xcha) * xddbp
        + xshb * (xubha + xdbha + xcha) * xdsbp
        + xchb * (xubha + xdbha + xsha) * xdcbp;

    gppv[1] = (xuha * (xdhb + xshb + xchb)
        + xdha * (xshb + xchb)
        + xsha * xchb
        + xubha * (xdbhb + xshb + xchb)
        + xdbha * (xshb + xchb)
        + xsha * xchb)
        * xdgp;
    gppc[1] = (xuhb * (xdha + xsha + xcha)
        + xdhb * (xsha + xcha)
        + xshb * xcha
        + xubhb * (xdbha + xsha + xcha)
        + xdbhb * (xsha + xcha)
        + xshb * xcha)
        * xdgp;

    gppv[2] = xuha * (xdbhb + xshb + xchb) * xdup
        + xdha * (xubhb + xshb + xchb) * xddp
        + xsha * (xubhb + xdbhb + xchb) * xdsp
        + xcha * (xubhb + xdbhb + xshb) * xdcp
        + xubha * (xdhb + xshb + xchb) * xdubp
        + xdbha * (xuhb + xshb + xchb) * xddbp
        + xsha * (xuhb + xdhb + xchb) * xdsbp
        + xcha * (xuhb + xdhb + xshb) * xdcbp;
    gppc[2] = xuhb * (xdbha + xsha + xcha) * xdup
        + xdhb * (xubha + xsha + xcha) * xddp
        + xshb * (xubha + xdbha + xcha) * xdsp
        + xchb * (xubha + xdbha + xsha) * xdcp
        + xubhb * (xdha + xsha + xcha) * xdubp
        + xdbhb * (xuha + xsha + xcha) * xddbp
        + xshb * (xuha + xdha + xcha) * xdsbp
        + xchb * (xuha + xdha + xsha) * xdcbp;

    gppv[3] = (xuha * (xdbhb + xshb + xchb)
        + xdha * (xubhb + xshb + xchb)
        + xsha * (xubhb + xdbhb + xchb)
        + xcha * (xubhb + xdbhb + xshb))
        * xdgp;
    gppc[3] = (xuhb * (xdbha + xsha + xcha)
        + xdhb * (xubha + xsha + xcha)
        + xshb * (xubha + xdbha + xcha)
        + xchb * (xubha + xdbha + xsha))
        * xdgp;

    gppv[4] = (xdha * xdbhb + xsha * xshb + xcha * xchb) * xdup
        + (xuha * xubhb + xsha * xshb + xcha * xchb) * xddp
        + (xuha * xubhb + xdha * xdbhb + xcha * xchb) * xdsp
        + (xuha * xubhb + xdha * xdbhb + xsha * xshb) * xdcp
        + (xdbha * xdhb + xsha * xshb + xcha * xchb) * xdubp
        + (xubha * xuhb + xsha * xshb + xcha * xchb) * xddbp
        + (xubha * xuhb + xdbha * xdhb + xcha * xchb) * xdsbp
        + (xubha * xuhb + xdbha * xdhb + xsha * xshb) * xdcbp;
    gppc[4] = (xdhb * xdbha + xshb * xsha + xchb * xcha) * xdup
        + (xuhb * xubha + xshb * xsha + xchb * xcha) * xddp
        + (xuhb * xubha + xdhb * xdbha + xchb * xcha) * xdsp
        + (xuhb * xubha + xdhb * xdbha + xshb * xsha) * xdcp
        + (xdbhb * xdha + xshb * xsha + xchb * xcha) * xdubp
        + (xubhb * xuha + xshb * xsha + xchb * xcha) * xddbp
        + (xubhb * xuha + xdbhb * xdha + xchb * xcha) * xdsbp
        + (xubhb * xuha + xdbhb * xdha + xshb * xsha) * xdcbp;

    gppv[5] = xuha * xuhb * xdup
        + xdha * xdhb * xddp
        + xsha * xshb * xdsp
        + xcha * xchb * xdcp
        + xubha * xubhb * xdubp
        + xdbha * xdbhb * xddbp
        + xsha * xshb * xdsbp
        + xcha * xchb * xdcbp;
    gppv[5] /= 2.0;
    gppc[5] = xuhb * xuha * xdup
        + xdhb * xdha * xddp
        + xshb * xsha * xdsp
        + xchb * xcha * xdcp
        + xubhb * xubha * xdubp
        + xdbhb * xdbha * xddbp
        + xshb * xsha * xdsbp
        + xchb * xcha * xdcbp;
    gppc[5] /= 2.0;

    gppv[6] = (xuha * xuhb
        + xdha * xdhb
        + xsha * xshb
        + xcha * xchb
        + xubha * xubhb
        + xdbha * xdbhb
        + xsha * xshb
        + xcha * xchb)
        * xdgp;
    gppv[6] /= 2.0;
    gppc[6] = (xuhb * xuha
        + xdhb * xdha
        + xshb * xsha
        + xchb * xcha
        + xubhb * xubha
        + xdbhb * xdbha
        + xshb * xsha
        + xchb * xcha)
        * xdgp;
    gppc[6] /= 2.0;

    gppv[7] = ((xdha + xsha + xcha) * xdup
        + (xuha + xsha + xcha) * xddp
        + (xuha + xdha + xcha) * xdsp
        + (xuha + xdha + xsha) * xdcp
        + (xdbha + xsha + xcha) * xdubp
        + (xubha + xsha + xcha) * xddbp
        + (xubha + xdbha + xcha) * xdsbp
        + (xubha + xdbha + xsha) * xdcbp)
        * xgprob;
    gppc[7] = ((xdhb + xshb + xchb) * xdup
        + (xuhb + xshb + xchb) * xddp
        + (xuhb + xdhb + xchb) * xdsp
        + (xuhb + xdhb + xshb) * xdcp
        + (xdbhb + xshb + xchb) * xdubp
        + (xubhb + xshb + xchb) * xddbp
        + (xubhb + xdbhb + xchb) * xdsbp
        + (xubhb + xdbhb + xshb) * xdcbp)
        * xgproa;

    gppv[8] = ((xdha + xsha + xcha) * xdubp
        + (xuha + xsha + xcha) * xddbp
        + (xuha + xdha + xcha) * xdsbp
        + (xuha + xdha + xsha) * xdcbp
        + (xdbha + xsha + xcha) * xdup
        + (xubha + xsha + xcha) * xddp
        + (xubha + xdbha + xcha) * xdsp
        + (xubha + xdbha + xsha) * xdcp)
        * xgprob;
    gppc[8] = ((xdhb + xshb + xchb) * xdubp
        + (xuhb + xshb + xchb) * xddbp
        + (xuhb + xdhb + xchb) * xdsbp
        + (xuhb + xdhb + xshb) * xdcbp
        + (xdbhb + xshb + xchb) * xdup
        + (xubhb + xshb + xchb) * xddp
        + (xubhb + xdbhb + xchb) * xdsp
        + (xubhb + xdbhb + xshb) * xdcp)
        * xgproa;

    gppv[9] = (xuha * xdubp
        + xdha * xddbp
        + xsha * xdsbp
        + xcha * xdcbp
        + xubha * xdup
        + xdbha * xddp
        + xsha * xdsp
        + xcha * xdcp)
        * xgprob;
    gppc[9] = (xuhb * xdubp
        + xdhb * xddbp
        + xshb * xdsbp
        + xchb * xdcbp
        + xubhb * xdup
        + xdbhb * xddp
        + xshb * xdsp
        + xchb * xdcp)
        * xgproa;

    gppv[10] = xuha * xubhb * xdup
        + xdha * xdbhb * xddp
        + xsha * xshb * xdsp
        + xcha * xchb * xdcp
        + xubha * xuhb * xdubp
        + xdbha * xdhb * xddbp
        + xsha * xshb * xdsbp
        + xcha * xchb * xdcbp;
    gppc[10] = xuhb * xubha * xdup
        + xdhb * xdbha * xddp
        + xshb * xsha * xdsp
        + xchb * xcha * xdcp
        + xubhb * xuha * xdubp
        + xdbhb * xdha * xddbp
        + xshb * xsha * xdsbp
        + xchb * xcha * xdcbp;

    gppv[11] = (xuha * xubhb + xdha * xdbhb + xsha * xshb + xcha * xchb) * xdgp;
    gppc[11] = (xuhb * xubha + xdhb * xdbha + xshb * xsha + xchb * xcha) * xdgp;

    gppv[12] = (xuha * xdup
        + xubha * xdubp
        + xdha * xddp
        + xdbha * xddbp
        + xsha * xdsp
        + xsha * xdsbp
        + xcha * xdcp
        + xcha * xdcbp)
        * xgprob;
    gppc[12] = (xuhb * xdup
        + xubhb * xdubp
        + xdhb * xddp
        + xdbhb * xddbp
        + xshb * xdsp
        + xshb * xdsbp
        + xchb * xdcp
        + xchb * xdcbp)
        * xgproa;

    gppv[13] = (xuha + xubha + xdha + xdbha + 2.0 * xsha + 2.0 * xcha) * xgprob * xdgp;
    gppc[13] = (xuhb + xubhb + xdhb + xdbhb + 2.0 * xshb + 2.0 * xchb) * xgproa * xdgp;

    gppv[14] = xgproa * xgprob * xdgp / 2.0;
    gppc[14] = xgprob * xgproa * xdgp / 2.0;

    gppv[15] = xgproa * xgprob * (xdup + xdubp + xddp + xddbp + xdsp + xdsbp + xdcp + xdcbp) / 2.0;
    gppc[15] = xgprob * xgproa * (xdup + xdubp + xddp + xddbp + xdsp + xdsbp + xdcp + xdcbp) / 2.0;

    (gppv, gppc)
}
