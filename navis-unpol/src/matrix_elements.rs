//! Unpolarized partonic matrix elements.

/// Scale/color context threaded through the matrix elements.
#[derive(Debug, Clone, Copy)]
pub struct MeContext {
    pub ca: f64,
    pub cf: f64,
    pub nf: f64,
    pub q2fac: f64,
    pub q2mu: f64,
    pub q2frag: f64,
}

/// `FBOR(V,SHD,F0)`: Born cross sections (including the `1/v/(1-v)` phase
/// space factor), one entry per of the 16 channels.
#[must_use]
pub fn fbor(v: f64, shd: f64, nc: f64, cf: f64) -> [f64; 16] {
    let v2 = v.powi(2);
    let v3 = v.powi(3);
    let v4 = v.powi(4);
    let vc = nc * nc - 1.0;
    let nc2 = nc * nc;
    let vm = 1.0 - v;
    let vm2 = vm.powi(2);
    let prelo = std::f64::consts::PI / shd / v / vm;

    let mut f0 = [0.0_f64; 16];

    f0[0] = cf / nc * prelo * (v2 + 1.0) / vm2;
    f0[1] = 0.0;
    f0[2] = cf / nc * prelo * (v2 + 1.0) / vm2;
    f0[3] = 0.0;
    f0[4] = cf / nc * prelo * (2.0 * v2 - 2.0 * v + 1.0);
    f0[5] = 2.0 * cf / nc2
        * prelo
        * (nc * v4 - 2.0 * nc * v3 + 4.0 * nc * v2 + v2 - (3.0 * nc + 1.0) * v + nc)
        / v2
        / vm2;
    f0[6] = 0.0;
    f0[7] = 0.0;
    f0[8] = 0.0;
    f0[9] = 0.0;
    f0[10] = 2.0 * cf / nc2
        * prelo
        * (nc * v4 - (3.0 * nc + 1.0) * v3 + (4.0 * nc + 1.0) * v2 - 2.0 * nc * v + nc)
        / vm2;
    f0[11] = cf / nc2
        * prelo
        * (2.0 * v2 - 2.0 * v + 1.0)
        * (2.0 * nc2 * v2 - 2.0 * nc2 * v + nc2 - 1.0)
        / v
        / vm;
    f0[12] = 1.0 / (2.0 * nc2) * prelo * (v2 + 1.0) * ((nc2 - 1.0) * v2 + 2.0 * v + (nc2 - 1.0))
        / v
        / vm2;
    f0[13] = 1.0 / (2.0 * nc2)
        * prelo
        * (v2 - 2.0 * v + 2.0)
        * ((nc2 - 1.0) * v2 - 2.0 * nc2 * v + 2.0 * nc2)
        / v2
        / vm;
    f0[14] = 4.0 * nc2 / vc * prelo * (3.0 - v * vm + v / vm2 + vm / v2);
    f0[15] =
        1.0 / (2.0 * nc) / vc * prelo * (v2 + vm2) * (2.0 * nc2 * (v2 - v) + nc2 - 1.0) / v / vm;

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
    let l1v = (1.0 - v).ln();
    let lv = v.ln();
    let l1w = (1.0 - w).ln();
    let lw = w.ln();
    let lvw = ((1.0 - v) / (1.0 - v * w)).ln() / (1.0 - w);
    let l1vw = (1.0 - v + v * w).ln() / (1.0 - w);

    let lmu = (ctx.q2mu / s).ln();
    let lms = (ctx.q2fac / s).ln();
    let lmss = (ctx.q2frag / s).ln();

    let cacf = ctx.ca * ctx.cf;
    let ca2 = ctx.ca.powi(2);
    let ca4 = ctx.ca.powi(4);

    Precalc {
        l1v,
        lv,
        l1w,
        lw,
        lvw,
        l1vw,
        lmu,
        lms,
        lmss,
        cacf,
        ca2,
        ca4,
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
    let rresc = struv(j0, w, v, x3, sh, ctx, pre) + avgo(w, v);
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
    let rrescc = struv(j0, wx, vx, x3, sh, ctx, pre) + avgo(wx, vx);
    rrescc / sh
}

/// `AVWPL(W,V,S)`: all `1/(1-W)+` pieces, selected by channel `j0`
/// (1-indexed, matching Fortran `J0`).
#[must_use]
pub fn avwpl(j0: usize, _w: f64, v: f64, s: f64, ctx: &MeContext) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let m = ctx.q2fac.sqrt();
    let mp = ctx.q2frag.sqrt();

    let l1v = (1.0 - v).ln();
    let lv = v.ln();
    let lms = (m.powi(2) / s).ln();
    let lmss = (mp.powi(2) / s).ln();
    let nf = 2.0 * (ctx.nf / 2.0);

    match j0 {
        1 => {
            (-6.0 * ca * cf.powi(2) * (1.0 + v.powi(2))) / ((1.0 - v).powi(2) * v)
                - (8.0 * (1.0 + ca.powi(2)) * cf * l1v * (1.0 + v.powi(2)))
                    / ((1.0 - v).powi(2) * v)
                - (16.0 * ca * cf.powi(2) * lms * (1.0 + v.powi(2))) / ((1.0 - v).powi(2) * v)
                - (8.0 * ca * cf.powi(2) * lmss * (1.0 + v.powi(2))) / ((1.0 - v).powi(2) * v)
                - (4.0 * (11.0 - 7.0 * ca.powi(2)) * cf * lv * (1.0 + v.powi(2)))
                    / ((1.0 - v).powi(2) * v)
        }
        2 => 0.0,
        3 => {
            (-6.0 * ca * cf.powi(2) * (1.0 + v.powi(2))) / ((1.0 - v).powi(2) * v)
                - (8.0 * (1.0 + ca.powi(2)) * cf * l1v * (1.0 + v.powi(2)))
                    / ((1.0 - v).powi(2) * v)
                - (16.0 * ca * cf.powi(2) * lms * (1.0 + v.powi(2))) / ((1.0 - v).powi(2) * v)
                - (8.0 * ca * cf.powi(2) * lmss * (1.0 + v.powi(2))) / ((1.0 - v).powi(2) * v)
                + (4.0 * (5.0 + 3.0 * ca.powi(2)) * cf * lv * (1.0 + v.powi(2)))
                    / ((1.0 - v).powi(2) * v)
        }
        4 => 0.0,
        5 => {
            (-6.0 * ca * cf.powi(2) * (1.0 - 2.0 * v + 2.0 * v.powi(2))) / v
                - (8.0 * (3.0 - ca.powi(2)) * cf * l1v * (1.0 - 2.0 * v + 2.0 * v.powi(2))) / v
                - (16.0 * ca * cf.powi(2) * lms * (1.0 - 2.0 * v + 2.0 * v.powi(2))) / v
                - (8.0 * ca * cf.powi(2) * lmss * (1.0 - 2.0 * v + 2.0 * v.powi(2))) / v
                + (4.0 * (5.0 + 3.0 * ca.powi(2)) * cf * lv * (1.0 - 2.0 * v + 2.0 * v.powi(2))) / v
        }
        6 => {
            (-12.0
                * cf.powi(2)
                * (ca - v - 3.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2) - 2.0 * ca * v.powi(3)
                    + ca * v.powi(4)))
                / ((1.0 - v).powi(2) * v.powi(3))
                - (32.0
                    * cf.powi(2)
                    * lms
                    * (ca - v - 3.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2)
                        - 2.0 * ca * v.powi(3)
                        + ca * v.powi(4)))
                    / ((1.0 - v).powi(2) * v.powi(3))
                - (16.0
                    * cf.powi(2)
                    * lmss
                    * (ca - v - 3.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2)
                        - 2.0 * ca * v.powi(3)
                        + ca * v.powi(4)))
                    / ((1.0 - v).powi(2) * v.powi(3))
                - (16.0
                    * cf
                    * l1v
                    * (3.0 * ca - ca.powi(3) - v - 9.0 * ca * v - ca.powi(2) * v
                        + 3.0 * ca.powi(3) * v
                        + v.powi(2)
                        + 11.0 * ca * v.powi(2)
                        + ca.powi(2) * v.powi(2)
                        - 3.0 * ca.powi(3) * v.powi(2)
                        - 6.0 * ca * v.powi(3)
                        + 2.0 * ca.powi(3) * v.powi(3)
                        + 2.0 * ca * v.powi(4)))
                    / (ca * (1.0 - v).powi(2) * v.powi(3))
                - (8.0
                    * cf
                    * lv
                    * (7.0 * ca - 3.0 * ca.powi(3) - 7.0 * v - 21.0 * ca * v
                        + 3.0 * ca.powi(2) * v
                        + 9.0 * ca.powi(3) * v
                        + 7.0 * v.powi(2)
                        + 30.0 * ca * v.powi(2)
                        - 3.0 * ca.powi(2) * v.powi(2)
                        - 14.0 * ca.powi(3) * v.powi(2)
                        - 14.0 * ca * v.powi(3)
                        + 6.0 * ca.powi(3) * v.powi(3)
                        + 9.0 * ca * v.powi(4)
                        - 5.0 * ca.powi(3) * v.powi(4)))
                    / (ca * (1.0 - v).powi(2) * v.powi(3))
        }
        7 => 0.0,
        8 => 0.0,
        9 => 0.0,
        10 => 0.0,
        11 => {
            (-12.0
                * cf.powi(2)
                * (ca - 2.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2)
                    - v.powi(3)
                    - 3.0 * ca * v.powi(3)
                    + ca * v.powi(4)))
                / ((1.0 - v).powi(2) * v)
                - (32.0
                    * cf.powi(2)
                    * lms
                    * (ca - 2.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2)
                        - v.powi(3)
                        - 3.0 * ca * v.powi(3)
                        + ca * v.powi(4)))
                    / ((1.0 - v).powi(2) * v)
                - (16.0
                    * cf.powi(2)
                    * lmss
                    * (ca - 2.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2)
                        - v.powi(3)
                        - 3.0 * ca * v.powi(3)
                        + ca * v.powi(4)))
                    / ((1.0 - v).powi(2) * v)
                - (16.0
                    * cf
                    * l1v
                    * (2.0 * ca - 6.0 * ca * v
                        + 2.0 * ca.powi(3) * v
                        + v.powi(2)
                        + 11.0 * ca * v.powi(2)
                        + ca.powi(2) * v.powi(2)
                        - 3.0 * ca.powi(3) * v.powi(2)
                        - v.powi(3)
                        - 9.0 * ca * v.powi(3)
                        - ca.powi(2) * v.powi(3)
                        + 3.0 * ca.powi(3) * v.powi(3)
                        + 3.0 * ca * v.powi(4)
                        - ca.powi(3) * v.powi(4)))
                    / (ca * (1.0 - v).powi(2) * v)
                + (8.0
                    * cf
                    * lv
                    * (5.0 * ca + 3.0 * ca.powi(3) - 10.0 * ca * v - 6.0 * ca.powi(3) * v
                        + v.powi(2)
                        + 20.0 * ca * v.powi(2)
                        + 7.0 * ca.powi(2) * v.powi(2)
                        + 12.0 * ca.powi(3) * v.powi(2)
                        - v.powi(3)
                        - 15.0 * ca * v.powi(3)
                        - 7.0 * ca.powi(2) * v.powi(3)
                        - 9.0 * ca.powi(3) * v.powi(3)
                        + 5.0 * ca * v.powi(4)
                        + 3.0 * ca.powi(3) * v.powi(4)))
                    / (ca * (1.0 - v).powi(2) * v)
        }
        12 => {
            (-16.0
                * ca.powi(2)
                * cf
                * lmss
                * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                * (cf - ca * v + ca * v.powi(2)))
                / ((1.0 - v) * v.powi(2))
                + (22.0
                    * ca
                    * cf
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (-2.0 * ca * cf + 2.0 * ca.powi(2) * v - 2.0 * ca.powi(2) * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(2))
                + (16.0
                    * cf.powi(2)
                    * lms
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (-2.0 * ca * cf + 2.0 * ca.powi(2) * v - 2.0 * ca.powi(2) * v.powi(2)))
                    / ((1.0 - v) * v.powi(2))
                - (4.0
                    * cf
                    * nf
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (-2.0 * ca * cf + 2.0 * ca.powi(2) * v - 2.0 * ca.powi(2) * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(2))
                - (8.0
                    * cf
                    * l1v
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (1.0 + ca.powi(4) + 2.0 * ca.powi(2) * v
                        - 2.0 * ca.powi(4) * v
                        - 2.0 * ca.powi(2) * v.powi(2)))
                    / (ca * (1.0 - v) * v.powi(2))
                + (8.0
                    * cf
                    * lv
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (1.0 - 5.0 * ca.powi(2) + 4.0 * ca.powi(4) + 2.0 * ca.powi(2) * v
                        - 8.0 * ca.powi(4) * v
                        - 2.0 * ca.powi(2) * v.powi(2)
                        + 6.0 * ca.powi(4) * v.powi(2)))
                    / (ca * (1.0 - v) * v.powi(2))
        }
        13 => {
            (22.0
                * ca
                * cf
                * (1.0 + v.powi(2))
                * (-2.0 * ca * cf - 2.0 * v + v.powi(2) - ca.powi(2) * v.powi(2)))
                / (3.0 * (1.0 - v).powi(2) * v.powi(2))
                - (4.0
                    * (1.0 - 3.0 * ca.powi(2))
                    * cf
                    * lms
                    * (1.0 + v.powi(2))
                    * (-2.0 * ca * cf - 2.0 * v + v.powi(2) - ca.powi(2) * v.powi(2)))
                    / (ca * (1.0 - v).powi(2) * v.powi(2))
                - (4.0
                    * cf
                    * nf
                    * (1.0 + v.powi(2))
                    * (-2.0 * ca * cf - 2.0 * v + v.powi(2) - ca.powi(2) * v.powi(2)))
                    / (3.0 * (1.0 - v).powi(2) * v.powi(2))
                - (8.0
                    * cf.powi(2)
                    * lmss
                    * (1.0 + v.powi(2))
                    * (2.0 * ca * cf + 2.0 * v - v.powi(2) + ca.powi(2) * v.powi(2)))
                    / ((1.0 - v).powi(2) * v.powi(2))
                + (8.0
                    * cf
                    * l1v
                    * (1.0 + v.powi(2))
                    * (1.0 + 2.0 * ca.powi(2) - ca.powi(4) - 2.0 * v - 6.0 * ca.powi(2) * v
                        + v.powi(2)
                        + 2.0 * ca.powi(2) * v.powi(2)
                        - ca.powi(4) * v.powi(2)))
                    / (ca * (1.0 - v).powi(2) * v.powi(2))
                + (8.0
                    * cf
                    * lv
                    * (1.0 + v.powi(2))
                    * (1.0 - 5.0 * ca.powi(2) + 4.0 * ca.powi(4) - 2.0 * v
                        + 8.0 * ca.powi(2) * v
                        + v.powi(2)
                        - 5.0 * ca.powi(2) * v.powi(2)
                        + 2.0 * ca.powi(4) * v.powi(2)))
                    / (ca * (1.0 - v).powi(2) * v.powi(2))
        }
        14 => {
            (-16.0 * ca.powi(3) * cf * l1v * (1.0 - v) * (2.0 - 2.0 * v + v.powi(2))) / v.powi(3)
                - (6.0
                    * cf.powi(2)
                    * (2.0 - 2.0 * v + v.powi(2))
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / ((1.0 - v) * v.powi(3))
                + (4.0
                    * (1.0 - 3.0 * ca.powi(2))
                    * cf
                    * lms
                    * (2.0 - 2.0 * v + v.powi(2))
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / (ca * (1.0 - v) * v.powi(3))
                - (8.0
                    * ca
                    * cf
                    * lmss
                    * (2.0 - 2.0 * v + v.powi(2))
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / ((1.0 - v) * v.powi(3))
                - (4.0
                    * cf
                    * lv
                    * (2.0 - 2.0 * v + v.powi(2))
                    * (2.0 * ca.powi(2) - 18.0 * ca.powi(4) - 2.0 * ca.powi(2) * v
                        + 18.0 * ca.powi(4) * v
                        - v.powi(2)
                        + 6.0 * ca.powi(2) * v.powi(2)
                        - 9.0 * ca.powi(4) * v.powi(2)))
                    / (ca * (1.0 - v) * v.powi(3))
        }
        15 => {
            (-256.0 * ca.powi(3) * l1v * (1.0 - v + v.powi(2)).powi(2)) / v.powi(3)
                - (704.0 * ca.powi(3) * (1.0 - v + v.powi(2)).powi(3))
                    / (3.0 * (1.0 - v).powi(2) * v.powi(3))
                - (512.0 * ca.powi(3) * lms * (1.0 - v + v.powi(2)).powi(3))
                    / ((1.0 - v).powi(2) * v.powi(3))
                - (256.0 * ca.powi(3) * lmss * (1.0 - v + v.powi(2)).powi(3))
                    / ((1.0 - v).powi(2) * v.powi(3))
                + (128.0 * ca.powi(3) * nf * (1.0 - v + v.powi(2)).powi(3))
                    / (9.0 * (1.0 - v).powi(2) * v.powi(3))
                + (256.0
                    * ca.powi(3)
                    * lv
                    * (1.0 - v + v.powi(2)).powi(2)
                    * (5.0 - 5.0 * v + 4.0 * v.powi(2)))
                    / ((1.0 - v).powi(2) * v.powi(3))
        }
        16 => {
            (-16.0 * ca.powi(3) * cf * l1v * (1.0 - v) * (1.0 - 2.0 * v + 2.0 * v.powi(2)))
                / v.powi(2)
                - (16.0
                    * ca
                    * cf.powi(2)
                    * lmss
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (cf - ca * v + ca * v.powi(2)))
                    / ((1.0 - v) * v.powi(2))
                + (6.0
                    * cf.powi(2)
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (-2.0 * ca * cf + 2.0 * ca.powi(2) * v - 2.0 * ca.powi(2) * v.powi(2)))
                    / ((1.0 - v) * v.powi(2))
                + (4.0
                    * cf
                    * lv
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (1.0 - 10.0 * ca.powi(2) + 9.0 * ca.powi(4) + 2.0 * ca.powi(2) * v
                        - 18.0 * ca.powi(4) * v
                        - 2.0 * ca.powi(2) * v.powi(2)
                        + 14.0 * ca.powi(4) * v.powi(2)))
                    / (ca * (1.0 - v) * v.powi(2))
                + (8.0
                    * ca
                    * cf
                    * lms
                    * (-4.0 * ca * cf - 6.0 * v + 10.0 * ca.powi(2) * v + 8.0 * v.powi(2)
                        - 24.0 * ca.powi(2) * v.powi(2)
                        - 4.0 * v.powi(3)
                        + 32.0 * ca.powi(2) * v.powi(3)
                        - 24.0 * ca.powi(2) * v.powi(4)
                        + 8.0 * ca.powi(2) * v.powi(5)))
                    / ((1.0 - v).powi(2) * v.powi(2))
        }
        _ => unreachable!("j0 must be in 1..=16, got {j0}"),
    }
}

/// `AVDEL(V,S)`: the `delta(1-W)` term, selected by channel `j0`.
#[must_use]
pub fn avdel(j0: usize, v: f64, s: f64, ctx: &MeContext) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;
    let m = ctx.q2fac.sqrt();
    let mp = ctx.q2frag.sqrt();
    let pi = std::f64::consts::PI;
    let pi2 = pi.powi(2);

    let l1v = (1.0 - v).ln();
    let lv = v.ln();
    let lmu = (ctx.q2mu / s).ln();
    let lms = (m.powi(2) / s).ln();
    let lmss = (mp.powi(2) / s).ln();
    let nf = 2.0 * (ctx.nf / 2.0);

    match j0 {
        1 => {
            (-12.0 * ca * cf.powi(2) * lms * (1.0 + v.powi(2))) / ((1.0 - v).powi(2) * v)
                + (8.0 * ca * cf.powi(2) * l1v * lms * (1.0 + v.powi(2))) / ((1.0 - v).powi(2) * v)
                - (6.0 * ca * cf.powi(2) * lmss * (1.0 + v.powi(2))) / ((1.0 - v).powi(2) * v)
                + (44.0 * ca.powi(2) * cf * lmu * (1.0 + v.powi(2))) / (3.0 * (1.0 - v).powi(2) * v)
                - (8.0 * ca * cf.powi(2) * lms * lv * (1.0 + v.powi(2))) / ((1.0 - v).powi(2) * v)
                - (8.0 * ca * cf.powi(2) * lmss * lv * (1.0 + v.powi(2))) / ((1.0 - v).powi(2) * v)
                - (40.0 * ca * cf * nf * (1.0 + v.powi(2))) / (9.0 * (1.0 - v).powi(2) * v)
                + (8.0 * ca * cf * l1v * nf * (1.0 + v.powi(2))) / (3.0 * (1.0 - v).powi(2) * v)
                - (8.0 * ca * cf * lmu * nf * (1.0 + v.powi(2))) / (3.0 * (1.0 - v).powi(2) * v)
                - (2.0
                    * cf
                    * lv.powi(2)
                    * (16.0 - 9.0 * ca.powi(2) + 20.0 * v.powi(2) - 11.0 * ca.powi(2) * v.powi(2)))
                    / ((1.0 - v).powi(2) * v)
                + (4.0
                    * cf
                    * l1v
                    * lv
                    * (5.0 - 4.0 * ca.powi(2) + 9.0 * v.powi(2) - 6.0 * ca.powi(2) * v.powi(2)))
                    / ((1.0 - v).powi(2) * v)
                - (4.0
                    * cf
                    * l1v
                    * (3.0 + 5.0 * ca.powi(2) - 3.0 * ca.powi(2) * v
                        + 15.0 * v.powi(2)
                        + 2.0 * ca.powi(2) * v.powi(2)))
                    / (3.0 * (1.0 - v).powi(2) * v)
                - (cf
                    * lv
                    * (5.0 - ca.powi(2) - 8.0 * v + 4.0 * ca.powi(2) * v - 3.0 * v.powi(2)
                        + 3.0 * ca.powi(2) * v.powi(2)))
                    / ((1.0 - v).powi(2) * v)
                + (2.0
                    * cf
                    * l1v.powi(2)
                    * (5.0 + 2.0 * ca.powi(2) - 3.0 * v.powi(2) + 4.0 * ca.powi(2) * v.powi(2)))
                    / ((1.0 - v).powi(2) * v)
                + (cf
                    * (225.0
                        + 115.0 * ca.powi(2)
                        + 42.0 * pi2
                        + 12.0 * ca.powi(2) * pi2
                        + 225.0 * v.powi(2)
                        + 115.0 * ca.powi(2) * v.powi(2)
                        - 30.0 * pi2 * v.powi(2)
                        + 48.0 * ca.powi(2) * pi2 * v.powi(2)))
                    / (9.0 * (1.0 - v).powi(2) * v)
        }
        2 => 0.0,
        3 => {
            (-12.0 * ca * cf.powi(2) * lms * (1.0 + v.powi(2))) / ((1.0 - v).powi(2) * v)
                + (8.0 * ca * cf.powi(2) * l1v * lms * (1.0 + v.powi(2))) / ((1.0 - v).powi(2) * v)
                - (6.0 * ca * cf.powi(2) * lmss * (1.0 + v.powi(2))) / ((1.0 - v).powi(2) * v)
                + (44.0 * ca.powi(2) * cf * lmu * (1.0 + v.powi(2))) / (3.0 * (1.0 - v).powi(2) * v)
                - (8.0 * ca * cf.powi(2) * lms * lv * (1.0 + v.powi(2))) / ((1.0 - v).powi(2) * v)
                - (8.0 * ca * cf.powi(2) * lmss * lv * (1.0 + v.powi(2))) / ((1.0 - v).powi(2) * v)
                - (40.0 * ca * cf * nf * (1.0 + v.powi(2))) / (9.0 * (1.0 - v).powi(2) * v)
                + (8.0 * ca * cf * l1v * nf * (1.0 + v.powi(2))) / (3.0 * (1.0 - v).powi(2) * v)
                - (8.0 * ca * cf * lmu * nf * (1.0 + v.powi(2))) / (3.0 * (1.0 - v).powi(2) * v)
                + (cf
                    * lv
                    * (11.0 - 3.0 * ca.powi(2) - 8.0 * v + 3.0 * v.powi(2)
                        - 3.0 * ca.powi(2) * v.powi(2)))
                    / ((1.0 - v).powi(2) * v)
                - (2.0
                    * cf
                    * l1v.powi(2)
                    * (3.0 - 4.0 * ca.powi(2) - 5.0 * v.powi(2) - 2.0 * ca.powi(2) * v.powi(2)))
                    / ((1.0 - v).powi(2) * v)
                + (4.0
                    * cf
                    * lv.powi(2)
                    * (6.0 + ca.powi(2) + 8.0 * v.powi(2) + ca.powi(2) * v.powi(2)))
                    / ((1.0 - v).powi(2) * v)
                - (4.0
                    * cf
                    * l1v
                    * lv
                    * (7.0 + ca.powi(2) + 11.0 * v.powi(2) + ca.powi(2) * v.powi(2)))
                    / ((1.0 - v).powi(2) * v)
                - (4.0
                    * cf
                    * l1v
                    * (15.0 + 2.0 * ca.powi(2) - 3.0 * ca.powi(2) * v
                        + 3.0 * v.powi(2)
                        + 5.0 * ca.powi(2) * v.powi(2)))
                    / (3.0 * (1.0 - v).powi(2) * v)
                + (cf
                    * (225.0 + 115.0 * ca.powi(2) - 30.0 * pi2
                        + 30.0 * ca.powi(2) * pi2
                        + 225.0 * v.powi(2)
                        + 115.0 * ca.powi(2) * v.powi(2)
                        + 42.0 * pi2 * v.powi(2)
                        + 30.0 * ca.powi(2) * pi2 * v.powi(2)))
                    / (9.0 * (1.0 - v).powi(2) * v)
        }
        4 => 0.0,
        5 => {
            -4.0 * (2.0 - ca.powi(2)) * cf * l1v
                - (12.0 * ca * cf.powi(2) * lms * (1.0 - 2.0 * v + 2.0 * v.powi(2))) / v
                + (8.0 * ca * cf.powi(2) * l1v * lms * (1.0 - 2.0 * v + 2.0 * v.powi(2))) / v
                - (6.0 * ca * cf.powi(2) * lmss * (1.0 - 2.0 * v + 2.0 * v.powi(2))) / v
                + (44.0 * ca.powi(2) * cf * lmu * (1.0 - 2.0 * v + 2.0 * v.powi(2))) / (3.0 * v)
                - (4.0 * (7.0 - ca.powi(2)) * cf * l1v * lv * (1.0 - 2.0 * v + 2.0 * v.powi(2))) / v
                - (8.0 * ca * cf.powi(2) * lms * lv * (1.0 - 2.0 * v + 2.0 * v.powi(2))) / v
                - (8.0 * ca * cf.powi(2) * lmss * lv * (1.0 - 2.0 * v + 2.0 * v.powi(2))) / v
                - (40.0 * ca * cf * nf * (1.0 - 2.0 * v + 2.0 * v.powi(2))) / (9.0 * v)
                - (8.0 * ca * cf * lmu * nf * (1.0 - 2.0 * v + 2.0 * v.powi(2))) / (3.0 * v)
                + (cf
                    * (225.0 + 115.0 * ca.powi(2) - 30.0 * pi2 - 6.0 * ca.powi(2) * pi2)
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2)))
                    / (9.0 * v)
                + (cf
                    * lv
                    * (11.0 - 3.0 * ca.powi(2) - 14.0 * v + 6.0 * ca.powi(2) * v + 6.0 * v.powi(2)
                        - 6.0 * ca.powi(2) * v.powi(2)))
                    / v
                - (2.0
                    * cf
                    * l1v.powi(2)
                    * (3.0 - 2.0 * ca.powi(2) - 6.0 * v + 4.0 * ca.powi(2) * v + 2.0 * v.powi(2)
                        - 2.0 * ca.powi(2) * v.powi(2)))
                    / v
                + (4.0
                    * cf
                    * lv.powi(2)
                    * (6.0 + ca.powi(2) - 12.0 * v - 2.0 * ca.powi(2) * v
                        + 14.0 * v.powi(2)
                        + 2.0 * ca.powi(2) * v.powi(2)))
                    / v
        }
        6 => {
            (-8.0 * cf * l1v * nf * (1.0 - v - ca * v - ca * v.powi(3)))
                / (3.0 * (1.0 - v).powi(2) * v.powi(2))
                + (8.0
                    * cf
                    * lv
                    * nf
                    * (2.0 * ca - v - 4.0 * ca * v + 3.0 * ca * v.powi(2) - ca * v.powi(3)))
                    / (3.0 * (1.0 - v) * v.powi(3))
                + (4.0
                    * cf
                    * l1v
                    * (6.0 - 6.0 * ca + 2.0 * ca.powi(2) + 3.0 * ca.powi(3) + 9.0 * ca * v
                        - 2.0 * ca.powi(2) * v
                        - 11.0 * ca.powi(3) * v
                        - 6.0 * v.powi(2)
                        - 6.0 * ca * v.powi(2)
                        + 6.0 * ca.powi(3) * v.powi(2)
                        - 15.0 * ca * v.powi(3)
                        - 2.0 * ca.powi(3) * v.powi(3)))
                    / (3.0 * ca * (1.0 - v).powi(2) * v.powi(2))
                - (24.0
                    * cf.powi(2)
                    * lms
                    * (ca - v - 3.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2)
                        - 2.0 * ca * v.powi(3)
                        + ca * v.powi(4)))
                    / ((1.0 - v).powi(2) * v.powi(3))
                + (16.0
                    * cf.powi(2)
                    * l1v
                    * lms
                    * (ca - v - 3.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2)
                        - 2.0 * ca * v.powi(3)
                        + ca * v.powi(4)))
                    / ((1.0 - v).powi(2) * v.powi(3))
                - (12.0
                    * cf.powi(2)
                    * lmss
                    * (ca - v - 3.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2)
                        - 2.0 * ca * v.powi(3)
                        + ca * v.powi(4)))
                    / ((1.0 - v).powi(2) * v.powi(3))
                + (88.0
                    * ca
                    * cf
                    * lmu
                    * (ca - v - 3.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2)
                        - 2.0 * ca * v.powi(3)
                        + ca * v.powi(4)))
                    / (3.0 * (1.0 - v).powi(2) * v.powi(3))
                - (16.0
                    * cf.powi(2)
                    * lms
                    * lv
                    * (ca - v - 3.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2)
                        - 2.0 * ca * v.powi(3)
                        + ca * v.powi(4)))
                    / ((1.0 - v).powi(2) * v.powi(3))
                - (16.0
                    * cf.powi(2)
                    * lmss
                    * lv
                    * (ca - v - 3.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2)
                        - 2.0 * ca * v.powi(3)
                        + ca * v.powi(4)))
                    / ((1.0 - v).powi(2) * v.powi(3))
                - (80.0
                    * cf
                    * nf
                    * (ca - v - 3.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2)
                        - 2.0 * ca * v.powi(3)
                        + ca * v.powi(4)))
                    / (9.0 * (1.0 - v).powi(2) * v.powi(3))
                - (16.0
                    * cf
                    * lmu
                    * nf
                    * (ca - v - 3.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2)
                        - 2.0 * ca * v.powi(3)
                        + ca * v.powi(4)))
                    / (3.0 * (1.0 - v).powi(2) * v.powi(3))
                - (2.0
                    * cf
                    * lv.powi(2)
                    * (16.0 * ca - 8.0 * ca.powi(3) - 17.0 * v - 56.0 * ca * v
                        + 6.0 * ca.powi(2) * v
                        + 26.0 * ca.powi(3) * v
                        + 19.0 * v.powi(2)
                        + 92.0 * ca * v.powi(2)
                        - 6.0 * ca.powi(2) * v.powi(2)
                        - 42.0 * ca.powi(3) * v.powi(2)
                        - 4.0 * v.powi(3)
                        - 48.0 * ca * v.powi(3)
                        + 20.0 * ca.powi(3) * v.powi(3)
                        + 2.0 * v.powi(4)
                        + 32.0 * ca * v.powi(4)
                        - 16.0 * ca.powi(3) * v.powi(4)))
                    / (ca * (1.0 - v).powi(2) * v.powi(3))
                + (4.0
                    * cf
                    * l1v
                    * lv
                    * (6.0 * ca - 2.0 * ca.powi(3) - 5.0 * v - 22.0 * ca * v
                        + 2.0 * ca.powi(2) * v
                        + 8.0 * ca.powi(3) * v
                        + 7.0 * v.powi(2)
                        + 36.0 * ca * v.powi(2)
                        - 2.0 * ca.powi(2) * v.powi(2)
                        - 16.0 * ca.powi(3) * v.powi(2)
                        - 4.0 * v.powi(3)
                        - 20.0 * ca * v.powi(3)
                        + 8.0 * ca.powi(3) * v.powi(3)
                        + 2.0 * v.powi(4)
                        + 14.0 * ca * v.powi(4)
                        - 8.0 * ca.powi(3) * v.powi(4)))
                    / (ca * (1.0 - v).powi(2) * v.powi(3))
                - (2.0
                    * cf
                    * l1v.powi(2)
                    * (2.0 * ca - 2.0 * ca.powi(3) + v - 10.0 * ca * v
                        + 4.0 * ca.powi(2) * v
                        + 8.0 * ca.powi(3) * v
                        + v.powi(2)
                        + 12.0 * ca * v.powi(2)
                        - 4.0 * ca.powi(2) * v.powi(2)
                        - 14.0 * ca.powi(3) * v.powi(2)
                        - 4.0 * v.powi(3)
                        - 12.0 * ca * v.powi(3)
                        + 8.0 * ca.powi(3) * v.powi(3)
                        + 2.0 * v.powi(4)
                        + 6.0 * ca * v.powi(4)
                        - 6.0 * ca.powi(3) * v.powi(4)))
                    / (ca * (1.0 - v).powi(2) * v.powi(3))
                - (2.0
                    * cf
                    * lv
                    * (27.0 * ca + 17.0 * ca.powi(3)
                        - 15.0 * v
                        - 105.0 * ca * v
                        - 13.0 * ca.powi(2) * v
                        - 45.0 * ca.powi(3) * v
                        + 27.0 * v.powi(2)
                        + 162.0 * ca * v.powi(2)
                        + 13.0 * ca.powi(2) * v.powi(2)
                        + 46.0 * ca.powi(3) * v.powi(2)
                        - 12.0 * v.powi(3)
                        - 114.0 * ca * v.powi(3)
                        - 22.0 * ca.powi(3) * v.powi(3)
                        + 21.0 * ca * v.powi(4)
                        + 13.0 * ca.powi(3) * v.powi(4)))
                    / (3.0 * ca * (1.0 - v).powi(2) * v.powi(3))
                + (2.0
                    * cf
                    * (225.0 * ca + 115.0 * ca.powi(3) + 6.0 * ca * pi2 + 30.0 * ca.powi(3) * pi2
                        - 225.0 * v
                        - 675.0 * ca * v
                        - 115.0 * ca.powi(2) * v
                        - 345.0 * ca.powi(3) * v
                        - 15.0 * pi2 * v
                        + 18.0 * ca * pi2 * v
                        - 30.0 * ca.powi(2) * pi2 * v
                        - 108.0 * ca.powi(3) * pi2 * v
                        + 225.0 * v.powi(2)
                        + 900.0 * ca * v.powi(2)
                        + 115.0 * ca.powi(2) * v.powi(2)
                        + 460.0 * ca.powi(3) * v.powi(2)
                        - 3.0 * pi2 * v.powi(2)
                        - 48.0 * ca * pi2 * v.powi(2)
                        + 30.0 * ca.powi(2) * pi2 * v.powi(2)
                        + 156.0 * ca.powi(3) * pi2 * v.powi(2)
                        - 450.0 * ca * v.powi(3)
                        - 230.0 * ca.powi(3) * v.powi(3)
                        + 36.0 * pi2 * v.powi(3)
                        + 60.0 * ca * pi2 * v.powi(3)
                        - 96.0 * ca.powi(3) * pi2 * v.powi(3)
                        + 225.0 * ca * v.powi(4)
                        + 115.0 * ca.powi(3) * v.powi(4)
                        - 18.0 * pi2 * v.powi(4)
                        - 30.0 * ca * pi2 * v.powi(4)
                        + 48.0 * ca.powi(3) * pi2 * v.powi(4)))
                    / (9.0 * ca * (1.0 - v).powi(2) * v.powi(3))
        }
        7..=10 => 0.0,
        11 => {
            (8.0 * cf * l1v * nf * (ca + v.powi(2) + ca * v.powi(2) - v.powi(3)))
                / (3.0 * (1.0 - v).powi(2) * v)
                - (4.0
                    * cf
                    * l1v
                    * (15.0 * ca + 2.0 * ca.powi(3) + 6.0 * v + 6.0 * ca * v
                        - 6.0 * ca.powi(3) * v
                        - 9.0 * ca * v.powi(2)
                        + 2.0 * ca.powi(2) * v.powi(2)
                        + 11.0 * ca.powi(3) * v.powi(2)
                        - 6.0 * v.powi(3)
                        + 6.0 * ca * v.powi(3)
                        - 2.0 * ca.powi(2) * v.powi(3)
                        - 3.0 * ca.powi(3) * v.powi(3)))
                    / (3.0 * ca * (1.0 - v).powi(2) * v)
                - (24.0
                    * cf.powi(2)
                    * lms
                    * (ca - 2.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2)
                        - v.powi(3)
                        - 3.0 * ca * v.powi(3)
                        + ca * v.powi(4)))
                    / ((1.0 - v).powi(2) * v)
                + (16.0
                    * cf.powi(2)
                    * l1v
                    * lms
                    * (ca - 2.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2)
                        - v.powi(3)
                        - 3.0 * ca * v.powi(3)
                        + ca * v.powi(4)))
                    / ((1.0 - v).powi(2) * v)
                - (12.0
                    * cf.powi(2)
                    * lmss
                    * (ca - 2.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2)
                        - v.powi(3)
                        - 3.0 * ca * v.powi(3)
                        + ca * v.powi(4)))
                    / ((1.0 - v).powi(2) * v)
                + (88.0
                    * ca
                    * cf
                    * lmu
                    * (ca - 2.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2)
                        - v.powi(3)
                        - 3.0 * ca * v.powi(3)
                        + ca * v.powi(4)))
                    / (3.0 * (1.0 - v).powi(2) * v)
                - (16.0
                    * cf.powi(2)
                    * lms
                    * lv
                    * (ca - 2.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2)
                        - v.powi(3)
                        - 3.0 * ca * v.powi(3)
                        + ca * v.powi(4)))
                    / ((1.0 - v).powi(2) * v)
                - (16.0
                    * cf.powi(2)
                    * lmss
                    * lv
                    * (ca - 2.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2)
                        - v.powi(3)
                        - 3.0 * ca * v.powi(3)
                        + ca * v.powi(4)))
                    / ((1.0 - v).powi(2) * v)
                - (80.0
                    * cf
                    * nf
                    * (ca - 2.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2)
                        - v.powi(3)
                        - 3.0 * ca * v.powi(3)
                        + ca * v.powi(4)))
                    / (9.0 * (1.0 - v).powi(2) * v)
                - (16.0
                    * cf
                    * lmu
                    * nf
                    * (ca - 2.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2)
                        - v.powi(3)
                        - 3.0 * ca * v.powi(3)
                        + ca * v.powi(4)))
                    / (3.0 * (1.0 - v).powi(2) * v)
                + (2.0
                    * cf
                    * lv
                    * (11.0 * ca - 3.0 * ca.powi(3) - 22.0 * ca * v
                        + 6.0 * ca.powi(3) * v
                        + 3.0 * v.powi(2)
                        + 24.0 * ca * v.powi(2)
                        - 3.0 * ca.powi(2) * v.powi(2)
                        - 12.0 * ca.powi(3) * v.powi(2)
                        - 3.0 * v.powi(3)
                        - 13.0 * ca * v.powi(3)
                        + 3.0 * ca.powi(2) * v.powi(3)
                        + 9.0 * ca.powi(3) * v.powi(3)
                        + 3.0 * ca * v.powi(4)
                        - 3.0 * ca.powi(3) * v.powi(4)))
                    / (ca * (1.0 - v).powi(2) * v)
                - (2.0
                    * cf
                    * l1v.powi(2)
                    * (2.0 + 6.0 * ca - 6.0 * ca.powi(3) - 4.0 * v - 12.0 * ca * v
                        + 8.0 * ca.powi(3) * v
                        + v.powi(2)
                        + 12.0 * ca * v.powi(2)
                        - 4.0 * ca.powi(2) * v.powi(2)
                        - 14.0 * ca.powi(3) * v.powi(2)
                        + v.powi(3)
                        - 10.0 * ca * v.powi(3)
                        + 4.0 * ca.powi(2) * v.powi(3)
                        + 8.0 * ca.powi(3) * v.powi(3)
                        + 2.0 * ca * v.powi(4)
                        - 2.0 * ca.powi(3) * v.powi(4)))
                    / (ca * (1.0 - v).powi(2) * v)
                - (8.0
                    * cf
                    * l1v
                    * lv
                    * (7.0 * ca - 14.0 * ca * v
                        + 2.0 * ca.powi(3) * v
                        + 4.0 * v.powi(2)
                        + 30.0 * ca * v.powi(2)
                        + 4.0 * ca.powi(2) * v.powi(2)
                        - 3.0 * ca.powi(3) * v.powi(2)
                        - 4.0 * v.powi(3)
                        - 21.0 * ca * v.powi(3)
                        - 4.0 * ca.powi(2) * v.powi(3)
                        + 3.0 * ca.powi(3) * v.powi(3)
                        + 7.0 * ca * v.powi(4)
                        - ca.powi(3) * v.powi(4)))
                    / (ca * (1.0 - v).powi(2) * v)
                + (8.0
                    * cf
                    * lv.powi(2)
                    * (6.0 * ca + ca.powi(3) - 12.0 * ca * v - 2.0 * ca.powi(3) * v
                        + 3.0 * v.powi(2)
                        + 26.0 * ca * v.powi(2)
                        + 5.0 * ca.powi(2) * v.powi(2)
                        + 4.0 * ca.powi(3) * v.powi(2)
                        - 3.0 * v.powi(3)
                        - 20.0 * ca * v.powi(3)
                        - 5.0 * ca.powi(2) * v.powi(3)
                        - 3.0 * ca.powi(3) * v.powi(3)
                        + 7.0 * ca * v.powi(4)
                        + ca.powi(3) * v.powi(4)))
                    / (ca * (1.0 - v).powi(2) * v)
                + (2.0
                    * cf
                    * (225.0 * ca + 115.0 * ca.powi(3) - 30.0 * ca * pi2 + 12.0 * ca.powi(3) * pi2
                        - 450.0 * ca * v
                        - 230.0 * ca.powi(3) * v
                        + 60.0 * ca * pi2 * v
                        + 12.0 * ca.powi(3) * pi2 * v
                        + 225.0 * v.powi(2)
                        + 900.0 * ca * v.powi(2)
                        + 115.0 * ca.powi(2) * v.powi(2)
                        + 460.0 * ca.powi(3) * v.powi(2)
                        - 12.0 * pi2 * v.powi(2)
                        - 84.0 * ca * pi2 * v.powi(2)
                        + 12.0 * ca.powi(2) * pi2 * v.powi(2)
                        - 6.0 * ca.powi(3) * pi2 * v.powi(2)
                        - 225.0 * v.powi(3)
                        - 675.0 * ca * v.powi(3)
                        - 115.0 * ca.powi(2) * v.powi(3)
                        - 345.0 * ca.powi(3) * v.powi(3)
                        + 12.0 * pi2 * v.powi(3)
                        + 90.0 * ca * pi2 * v.powi(3)
                        - 12.0 * ca.powi(2) * pi2 * v.powi(3)
                        + 18.0 * ca.powi(3) * pi2 * v.powi(3)
                        + 225.0 * ca * v.powi(4)
                        + 115.0 * ca.powi(3) * v.powi(4)
                        - 30.0 * ca * pi2 * v.powi(4)
                        - 6.0 * ca.powi(3) * pi2 * v.powi(4)))
                    / (9.0 * ca * (1.0 - v).powi(2) * v)
        }
        12 => {
            (2.0 * cf
                * l1v
                * (2.0 + 2.0 * ca.powi(2) + v - 5.0 * ca.powi(2) * v)
                * (1.0 - ca.powi(2) * v))
                / (ca * (1.0 - v) * v)
                + (16.0
                    * ca
                    * cf.powi(2)
                    * l1v
                    * lms
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (cf - ca * v + ca * v.powi(2)))
                    / ((1.0 - v) * v.powi(2))
                - (44.0
                    * ca.powi(2)
                    * cf
                    * lmss
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (cf - ca * v + ca * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(2))
                + (88.0
                    * ca.powi(2)
                    * cf
                    * lmu
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (cf - ca * v + ca * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(2))
                - (16.0
                    * ca
                    * cf.powi(2)
                    * lms
                    * lv
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (cf - ca * v + ca * v.powi(2)))
                    / ((1.0 - v) * v.powi(2))
                - (16.0
                    * ca.powi(2)
                    * cf
                    * lmss
                    * lv
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (cf - ca * v + ca * v.powi(2)))
                    / ((1.0 - v) * v.powi(2))
                + (8.0
                    * ca
                    * cf
                    * lmss
                    * nf
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (cf - ca * v + ca * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(2))
                - (16.0
                    * ca
                    * cf
                    * lmu
                    * nf
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (cf - ca * v + ca * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(2))
                + (12.0
                    * cf.powi(2)
                    * lms
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (-2.0 * ca * cf + 2.0 * ca.powi(2) * v - 2.0 * ca.powi(2) * v.powi(2)))
                    / ((1.0 - v) * v.powi(2))
                + (20.0
                    * cf
                    * nf
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (-2.0 * ca * cf + 2.0 * ca.powi(2) * v - 2.0 * ca.powi(2) * v.powi(2)))
                    / (9.0 * (1.0 - v) * v.powi(2))
                - (4.0
                    * cf
                    * lv
                    * nf
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (-2.0 * ca * cf + 2.0 * ca.powi(2) * v - 2.0 * ca.powi(2) * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(2))
                - (4.0
                    * cf
                    * l1v
                    * lv
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (1.0 - 4.0 * ca.powi(2) + 3.0 * ca.powi(4) + 2.0 * ca.powi(2) * v
                        - 6.0 * ca.powi(4) * v
                        - 2.0 * ca.powi(2) * v.powi(2)
                        + 2.0 * ca.powi(4) * v.powi(2)))
                    / (ca * (1.0 - v) * v.powi(2))
                + (2.0
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
                + (2.0
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
                + (2.0
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
                - (2.0
                    * cf
                    * (63.0 - 59.0 * ca.powi(2) - 4.0 * ca.powi(4) - 9.0 * pi2
                        + 12.0 * ca.powi(2) * pi2
                        - 3.0 * ca.powi(4) * pi2
                        - 126.0 * v
                        + 253.0 * ca.powi(2) * v
                        + 25.0 * ca.powi(4) * v
                        + 18.0 * pi2 * v
                        - 42.0 * ca.powi(2) * pi2 * v
                        + 12.0 * ca.powi(4) * pi2 * v
                        + 126.0 * v.powi(2)
                        - 541.0 * ca.powi(2) * v.powi(2)
                        - 77.0 * ca.powi(4) * v.powi(2)
                        - 18.0 * pi2 * v.powi(2)
                        + 78.0 * ca.powi(2) * pi2 * v.powi(2)
                        - 24.0 * ca.powi(4) * pi2 * v.powi(2)
                        + 576.0 * ca.powi(2) * v.powi(3)
                        + 104.0 * ca.powi(4) * v.powi(3)
                        - 72.0 * ca.powi(2) * pi2 * v.powi(3)
                        + 24.0 * ca.powi(4) * pi2 * v.powi(3)
                        - 288.0 * ca.powi(2) * v.powi(4)
                        - 52.0 * ca.powi(4) * v.powi(4)
                        + 36.0 * ca.powi(2) * pi2 * v.powi(4)
                        - 12.0 * ca.powi(4) * pi2 * v.powi(4)))
                    / (9.0 * ca * (1.0 - v) * v.powi(2))
        }
        13 => {
            -((9.0 - 31.0 * ca.powi(2))
                * cf
                * lms
                * (1.0 + v.powi(2))
                * (-2.0 * ca * cf - 2.0 * v + v.powi(2) - ca.powi(2) * v.powi(2)))
                / (3.0 * ca * (1.0 - v).powi(2) * v.powi(2))
                + (20.0
                    * cf
                    * nf
                    * (1.0 + v.powi(2))
                    * (-2.0 * ca * cf - 2.0 * v + v.powi(2) - ca.powi(2) * v.powi(2)))
                    / (9.0 * (1.0 - v).powi(2) * v.powi(2))
                - (4.0
                    * cf
                    * lms
                    * nf
                    * (1.0 + v.powi(2))
                    * (-2.0 * ca * cf - 2.0 * v + v.powi(2) - ca.powi(2) * v.powi(2)))
                    / (3.0 * (1.0 - v).powi(2) * v.powi(2))
                - (4.0
                    * cf
                    * lv
                    * nf
                    * (1.0 + v.powi(2))
                    * (-2.0 * ca * cf - 2.0 * v + v.powi(2) - ca.powi(2) * v.powi(2)))
                    / (3.0 * (1.0 - v).powi(2) * v.powi(2))
                + (8.0
                    * ca
                    * cf
                    * l1v
                    * lms
                    * (1.0 + v.powi(2))
                    * (2.0 * ca * cf + 2.0 * v - v.powi(2) + ca.powi(2) * v.powi(2)))
                    / ((1.0 - v).powi(2) * v.powi(2))
                - (6.0
                    * cf.powi(2)
                    * lmss
                    * (1.0 + v.powi(2))
                    * (2.0 * ca * cf + 2.0 * v - v.powi(2) + ca.powi(2) * v.powi(2)))
                    / ((1.0 - v).powi(2) * v.powi(2))
                + (44.0
                    * ca
                    * cf
                    * lmu
                    * (1.0 + v.powi(2))
                    * (2.0 * ca * cf + 2.0 * v - v.powi(2) + ca.powi(2) * v.powi(2)))
                    / (3.0 * (1.0 - v).powi(2) * v.powi(2))
                - (8.0
                    * ca
                    * cf
                    * lms
                    * lv
                    * (1.0 + v.powi(2))
                    * (2.0 * ca * cf + 2.0 * v - v.powi(2) + ca.powi(2) * v.powi(2)))
                    / ((1.0 - v).powi(2) * v.powi(2))
                - (8.0
                    * cf.powi(2)
                    * lmss
                    * lv
                    * (1.0 + v.powi(2))
                    * (2.0 * ca * cf + 2.0 * v - v.powi(2) + ca.powi(2) * v.powi(2)))
                    / ((1.0 - v).powi(2) * v.powi(2))
                - (8.0
                    * cf
                    * lmu
                    * nf
                    * (1.0 + v.powi(2))
                    * (2.0 * ca * cf + 2.0 * v - v.powi(2) + ca.powi(2) * v.powi(2)))
                    / (3.0 * (1.0 - v).powi(2) * v.powi(2))
                + (2.0
                    * cf
                    * l1v
                    * (4.0 - ca.powi(2) + ca.powi(4) - 8.0 * v - 10.0 * ca.powi(2) * v
                        + 10.0 * ca.powi(4) * v
                        + 4.0 * v.powi(2)
                        - ca.powi(2) * v.powi(2)
                        + ca.powi(4) * v.powi(2)))
                    / (ca * (1.0 - v).powi(2) * v)
                + (2.0
                    * cf
                    * lv
                    * (9.0 - 7.0 * ca.powi(2) - 2.0 * ca.powi(4) - 24.0 * v + 5.0 * ca.powi(2) * v
                        - 3.0 * ca.powi(4) * v
                        + 21.0 * v.powi(2)
                        + 19.0 * ca.powi(2) * v.powi(2)
                        - 28.0 * ca.powi(4) * v.powi(2)
                        - 6.0 * v.powi(3)
                        - 28.0 * ca.powi(2) * v.powi(3)
                        + 11.0 * ca.powi(2) * v.powi(4)
                        - 11.0 * ca.powi(4) * v.powi(4)))
                    / (3.0 * ca * (1.0 - v).powi(2) * v.powi(2))
                + (4.0
                    * cf
                    * l1v
                    * lv
                    * (4.0 * ca.powi(2) - 4.0 * ca.powi(4) + 2.0 * v
                        - 9.0 * ca.powi(2) * v
                        - 5.0 * v.powi(2)
                        + 5.0 * ca.powi(2) * v.powi(2)
                        - 5.0 * ca.powi(4) * v.powi(2)
                        + 4.0 * v.powi(3)
                        - 6.0 * ca.powi(2) * v.powi(3)
                        - 2.0 * ca.powi(4) * v.powi(3)
                        - v.powi(4)
                        + 2.0 * ca.powi(2) * v.powi(4)
                        - ca.powi(4) * v.powi(4)))
                    / (ca * (1.0 - v).powi(2) * v.powi(2))
                + (2.0
                    * cf
                    * l1v.powi(2)
                    * (3.0 + ca.powi(4) - 10.0 * v - ca.powi(2) * v
                        + 2.0 * ca.powi(4) * v
                        + 14.0 * v.powi(2)
                        + 2.0 * ca.powi(2) * v.powi(2)
                        + 2.0 * ca.powi(4) * v.powi(2)
                        - 10.0 * v.powi(3)
                        - ca.powi(2) * v.powi(3)
                        + 2.0 * ca.powi(4) * v.powi(3)
                        + 3.0 * v.powi(4)
                        + ca.powi(4) * v.powi(4)))
                    / (ca * (1.0 - v).powi(2) * v.powi(2))
                + (2.0
                    * cf
                    * lv.powi(2)
                    * (2.0 - 12.0 * ca.powi(2) + 10.0 * ca.powi(4) - 6.0 * v
                        + 21.0 * ca.powi(2) * v
                        + 9.0 * v.powi(2)
                        - 21.0 * ca.powi(2) * v.powi(2)
                        + 13.0 * ca.powi(4) * v.powi(2)
                        - 8.0 * v.powi(3)
                        + 18.0 * ca.powi(2) * v.powi(3)
                        + 2.0 * ca.powi(4) * v.powi(3)
                        + 3.0 * v.powi(4)
                        - 10.0 * ca.powi(2) * v.powi(4)
                        + 3.0 * ca.powi(4) * v.powi(4)))
                    / (ca * (1.0 - v).powi(2) * v.powi(2))
                - (2.0
                    * cf
                    * (63.0 - 59.0 * ca.powi(2) - 4.0 * ca.powi(4) - 9.0 * pi2
                        + 12.0 * ca.powi(2) * pi2
                        - 3.0 * ca.powi(4) * pi2
                        - 126.0 * v
                        - 17.0 * ca.powi(2) * v
                        - 9.0 * ca.powi(4) * v
                        + 36.0 * pi2 * v
                        - 15.0 * ca.powi(2) * pi2 * v
                        + 126.0 * v.powi(2)
                        - 136.0 * ca.powi(2) * v.powi(2)
                        - 26.0 * ca.powi(4) * v.powi(2)
                        - 63.0 * pi2 * v.powi(2)
                        - 3.0 * ca.powi(2) * pi2 * v.powi(2)
                        - 15.0 * ca.powi(4) * pi2 * v.powi(2)
                        - 126.0 * v.powi(3)
                        - 17.0 * ca.powi(2) * v.powi(3)
                        - 9.0 * ca.powi(4) * v.powi(3)
                        + 54.0 * pi2 * v.powi(3)
                        + 12.0 * ca.powi(2) * pi2 * v.powi(3)
                        - 18.0 * ca.powi(4) * pi2 * v.powi(3)
                        + 63.0 * v.powi(4)
                        - 59.0 * ca.powi(2) * v.powi(4)
                        - 4.0 * ca.powi(4) * v.powi(4)
                        - 18.0 * pi2 * v.powi(4)
                        - 6.0 * ca.powi(2) * pi2 * v.powi(4)
                        - 12.0 * ca.powi(4) * pi2 * v.powi(4)))
                    / (9.0 * ca * (1.0 - v).powi(2) * v.powi(2))
        }
        14 => {
            (-2.0
                * cf
                * l1v
                * (ca.powi(2) - v)
                * (1.0 - 5.0 * ca.powi(2) + 2.0 * v + 2.0 * ca.powi(2) * v))
                / (ca * (1.0 - v) * v.powi(2))
                + ((9.0 - 31.0 * ca.powi(2))
                    * cf
                    * lms
                    * (2.0 - 2.0 * v + v.powi(2))
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / (3.0 * ca * (1.0 - v) * v.powi(3))
                + (8.0
                    * ca
                    * cf
                    * l1v
                    * lms
                    * (2.0 - 2.0 * v + v.powi(2))
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / ((1.0 - v) * v.powi(3))
                - (22.0
                    * ca
                    * cf
                    * lmss
                    * (2.0 - 2.0 * v + v.powi(2))
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(3))
                + (44.0
                    * ca
                    * cf
                    * lmu
                    * (2.0 - 2.0 * v + v.powi(2))
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(3))
                - (8.0
                    * ca
                    * cf
                    * lms
                    * lv
                    * (2.0 - 2.0 * v + v.powi(2))
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / ((1.0 - v) * v.powi(3))
                - (8.0
                    * ca
                    * cf
                    * lmss
                    * lv
                    * (2.0 - 2.0 * v + v.powi(2))
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / ((1.0 - v) * v.powi(3))
                + (4.0
                    * cf
                    * lms
                    * nf
                    * (2.0 - 2.0 * v + v.powi(2))
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(3))
                + (4.0
                    * cf
                    * lmss
                    * nf
                    * (2.0 - 2.0 * v + v.powi(2))
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(3))
                - (8.0
                    * cf
                    * lmu
                    * nf
                    * (2.0 - 2.0 * v + v.powi(2))
                    * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2)
                        + ca.powi(2) * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(3))
                + (2.0
                    * cf
                    * l1v.powi(2)
                    * (8.0 * ca.powi(4) - ca.powi(2) * v - 20.0 * ca.powi(4) * v + v.powi(2)
                        - ca.powi(2) * v.powi(2)
                        + 21.0 * ca.powi(4) * v.powi(2)
                        - 2.0 * v.powi(3)
                        - 10.0 * ca.powi(4) * v.powi(3)
                        + 2.0 * v.powi(4)
                        + 2.0 * ca.powi(4) * v.powi(4)))
                    / (ca * (1.0 - v) * v.powi(3))
                - (cf
                    * lv
                    * (12.0 * ca.powi(2) - 12.0 * ca.powi(4) - 24.0 * ca.powi(2) * v
                        + 24.0 * ca.powi(4) * v
                        - 2.0 * v.powi(2)
                        - 4.0 * ca.powi(2) * v.powi(2)
                        - 2.0 * ca.powi(4) * v.powi(2)
                        + 2.0 * v.powi(3)
                        + 16.0 * ca.powi(2) * v.powi(3)
                        - 10.0 * ca.powi(4) * v.powi(3)
                        + 3.0 * v.powi(4)
                        - 6.0 * ca.powi(2) * v.powi(4)
                        + 3.0 * ca.powi(4) * v.powi(4)))
                    / (ca * (1.0 - v) * v.powi(3))
                - (4.0
                    * cf
                    * l1v
                    * lv
                    * (16.0 * ca.powi(4) - ca.powi(2) * v - 36.0 * ca.powi(4) * v
                        + v.powi(2)
                        + 3.0 * ca.powi(2) * v.powi(2)
                        + 37.0 * ca.powi(4) * v.powi(2)
                        - 2.0 * v.powi(3)
                        - 4.0 * ca.powi(2) * v.powi(3)
                        - 18.0 * ca.powi(4) * v.powi(3)
                        + 2.0 * v.powi(4)
                        + 2.0 * ca.powi(2) * v.powi(4)
                        + 4.0 * ca.powi(4) * v.powi(4)))
                    / (ca * (1.0 - v) * v.powi(3))
                + (2.0
                    * cf
                    * lv.powi(2)
                    * (48.0 * ca.powi(4) - 96.0 * ca.powi(4) * v + 2.0 * v.powi(2)
                        - 5.0 * ca.powi(2) * v.powi(2)
                        + 94.0 * ca.powi(4) * v.powi(2)
                        - 2.0 * v.powi(3)
                        + 5.0 * ca.powi(2) * v.powi(3)
                        - 46.0 * ca.powi(4) * v.powi(3)
                        + 3.0 * v.powi(4)
                        - 2.0 * ca.powi(2) * v.powi(4)
                        + 11.0 * ca.powi(4) * v.powi(4)))
                    / (ca * (1.0 - v) * v.powi(3))
                + (cf
                    * (108.0 * ca.powi(2) - 60.0 * ca.powi(4)
                        + 8.0 * ca.powi(2) * pi2
                        + 40.0 * ca.powi(4) * pi2
                        - 216.0 * ca.powi(2) * v
                        + 120.0 * ca.powi(4) * v
                        - 22.0 * ca.powi(2) * pi2 * v
                        - 104.0 * ca.powi(4) * pi2 * v
                        - 42.0 * v.powi(2)
                        + 240.0 * ca.powi(2) * v.powi(2)
                        - 138.0 * ca.powi(4) * v.powi(2)
                        + 2.0 * pi2 * v.powi(2)
                        + 14.0 * ca.powi(2) * pi2 * v.powi(2)
                        + 110.0 * ca.powi(4) * pi2 * v.powi(2)
                        + 42.0 * v.powi(3)
                        - 132.0 * ca.powi(2) * v.powi(3)
                        + 78.0 * ca.powi(4) * v.powi(3)
                        - 8.0 * pi2 * v.powi(3)
                        - 12.0 * ca.powi(2) * pi2 * v.powi(3)
                        - 52.0 * ca.powi(4) * pi2 * v.powi(3)
                        - 21.0 * v.powi(4)
                        + 42.0 * ca.powi(2) * v.powi(4)
                        - 21.0 * ca.powi(4) * v.powi(4)
                        + 10.0 * pi2 * v.powi(4)
                        + 4.0 * ca.powi(2) * pi2 * v.powi(4)
                        + 10.0 * ca.powi(4) * pi2 * v.powi(4)))
                    / (3.0 * ca * (1.0 - v) * v.powi(3))
        }
        15 => {
            (-16.0 * ca.powi(3) * lv.powi(2) * nf * (1.0 - v - v.powi(2))) / (3.0 * (1.0 - v) * v)
                + (16.0 * ca.powi(3) * l1v.powi(2) * nf * (1.0 - 3.0 * v + v.powi(2)))
                    / (3.0 * v.powi(2))
                - (1408.0 * ca.powi(3) * lms * (1.0 - v + v.powi(2)).powi(3))
                    / (3.0 * (1.0 - v).powi(2) * v.powi(3))
                + (256.0 * ca.powi(3) * l1v * lms * (1.0 - v + v.powi(2)).powi(3))
                    / ((1.0 - v).powi(2) * v.powi(3))
                - (704.0 * ca.powi(3) * lmss * (1.0 - v + v.powi(2)).powi(3))
                    / (3.0 * (1.0 - v).powi(2) * v.powi(3))
                + (1408.0 * ca.powi(3) * lmu * (1.0 - v + v.powi(2)).powi(3))
                    / (3.0 * (1.0 - v).powi(2) * v.powi(3))
                - (256.0 * ca.powi(3) * lms * lv * (1.0 - v + v.powi(2)).powi(3))
                    / ((1.0 - v).powi(2) * v.powi(3))
                - (256.0 * ca.powi(3) * lmss * lv * (1.0 - v + v.powi(2)).powi(3))
                    / ((1.0 - v).powi(2) * v.powi(3))
                + (256.0 * ca.powi(3) * lms * nf * (1.0 - v + v.powi(2)).powi(3))
                    / (9.0 * (1.0 - v).powi(2) * v.powi(3))
                + (128.0 * ca.powi(3) * lmss * nf * (1.0 - v + v.powi(2)).powi(3))
                    / (9.0 * (1.0 - v).powi(2) * v.powi(3))
                - (256.0 * ca.powi(3) * lmu * nf * (1.0 - v + v.powi(2)).powi(3))
                    / (9.0 * (1.0 - v).powi(2) * v.powi(3))
                - (32.0 * ca.powi(3) * l1v * lv * nf * (1.0 - 2.0 * v + 2.0 * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(2))
                - (32.0
                    * ca.powi(3)
                    * l1v
                    * nf
                    * (1.0 - v + v.powi(2))
                    * (5.0 - 2.0 * v + 5.0 * v.powi(2)))
                    / (9.0 * (1.0 - v).powi(2) * v.powi(2))
                + (64.0
                    * ca.powi(3)
                    * l1v
                    * (1.0 - v + v.powi(2))
                    * (7.0 + 8.0 * v + 7.0 * v.powi(2)))
                    / (3.0 * (1.0 - v).powi(2) * v.powi(2))
                + (64.0
                    * ca.powi(3)
                    * lv
                    * (1.0 - v + v.powi(2))
                    * (11.0 - 22.0 * v - 4.0 * v.powi(2) + 15.0 * v.powi(3) - 11.0 * v.powi(4)))
                    / (3.0 * (1.0 - v).powi(2) * v.powi(3))
                - (32.0
                    * ca.powi(3)
                    * lv
                    * nf
                    * (1.0 - v + v.powi(2))
                    * (4.0 - 8.0 * v + v.powi(2) + 3.0 * v.powi(3) - 4.0 * v.powi(4)))
                    / (9.0 * (1.0 - v).powi(2) * v.powi(3))
                + (64.0
                    * ca.powi(3)
                    * l1v.powi(2)
                    * (2.0 - 7.0 * v + 14.0 * v.powi(2) - 16.0 * v.powi(3) + 14.0 * v.powi(4)
                        - 7.0 * v.powi(5)
                        + 2.0 * v.powi(6)))
                    / ((1.0 - v).powi(2) * v.powi(3))
                - (64.0
                    * ca.powi(3)
                    * l1v
                    * lv
                    * (8.0 - 26.0 * v + 47.0 * v.powi(2) - 50.0 * v.powi(3) + 37.0 * v.powi(4)
                        - 16.0 * v.powi(5)
                        + 4.0 * v.powi(6)))
                    / ((1.0 - v).powi(2) * v.powi(3))
                + (64.0
                    * ca.powi(3)
                    * lv.powi(2)
                    * (12.0 - 36.0 * v + 66.0 * v.powi(2) - 72.0 * v.powi(3) + 57.0 * v.powi(4)
                        - 27.0 * v.powi(5)
                        + 8.0 * v.powi(6)))
                    / ((1.0 - v).powi(2) * v.powi(3))
                + (16.0
                    * ca.powi(3)
                    * nf
                    * (40.0 - 120.0 * v + 9.0 * pi2 * v + 294.0 * v.powi(2)
                        - 27.0 * pi2 * v.powi(2)
                        - 388.0 * v.powi(3)
                        + 36.0 * pi2 * v.powi(3)
                        + 294.0 * v.powi(4)
                        - 18.0 * pi2 * v.powi(4)
                        - 120.0 * v.powi(5)
                        + 40.0 * v.powi(6)))
                    / (27.0 * (1.0 - v).powi(2) * v.powi(3))
                - (32.0
                    * ca.powi(3)
                    * (134.0 - 24.0 * pi2 - 402.0 * v + 90.0 * pi2 * v + 831.0 * v.powi(2)
                        - 171.0 * pi2 * v.powi(2)
                        - 992.0 * v.powi(3)
                        + 186.0 * pi2 * v.powi(3)
                        + 831.0 * v.powi(4)
                        - 153.0 * pi2 * v.powi(4)
                        - 402.0 * v.powi(5)
                        + 72.0 * pi2 * v.powi(5)
                        + 134.0 * v.powi(6)
                        - 24.0 * pi2 * v.powi(6)))
                    / (9.0 * (1.0 - v).powi(2) * v.powi(3))
        }
        16 => {
            (2.0 * cf
                * l1v
                * (2.0 + 2.0 * ca.powi(2) + v - 5.0 * ca.powi(2) * v)
                * (1.0 - ca.powi(2) * v))
                / (ca * (1.0 - v) * v)
                + (16.0
                    * ca
                    * cf
                    * l1v
                    * lv
                    * (1.0 + ca - ca * v)
                    * (1.0 - ca + ca * v)
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2)))
                    / ((1.0 - v) * v.powi(2))
                + (16.0
                    * ca.powi(2)
                    * cf
                    * l1v
                    * lms
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (cf - ca * v + ca * v.powi(2)))
                    / ((1.0 - v) * v.powi(2))
                - (12.0
                    * ca
                    * cf.powi(2)
                    * lmss
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (cf - ca * v + ca * v.powi(2)))
                    / ((1.0 - v) * v.powi(2))
                + (88.0
                    * ca.powi(2)
                    * cf
                    * lmu
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (cf - ca * v + ca * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(2))
                - (16.0
                    * ca.powi(2)
                    * cf
                    * lms
                    * lv
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (cf - ca * v + ca * v.powi(2)))
                    / ((1.0 - v) * v.powi(2))
                - (16.0
                    * ca
                    * cf.powi(2)
                    * lmss
                    * lv
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (cf - ca * v + ca * v.powi(2)))
                    / ((1.0 - v) * v.powi(2))
                - (16.0
                    * ca
                    * cf
                    * lmu
                    * nf
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (cf - ca * v + ca * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(2))
                + (44.0
                    * ca
                    * cf
                    * lms
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (-2.0 * ca * cf + 2.0 * ca.powi(2) * v - 2.0 * ca.powi(2) * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(2))
                - (8.0
                    * cf
                    * lms
                    * nf
                    * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                    * (-2.0 * ca * cf + 2.0 * ca.powi(2) * v - 2.0 * ca.powi(2) * v.powi(2)))
                    / (3.0 * (1.0 - v) * v.powi(2))
                + (cf
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
                    / (ca * (1.0 - v) * v.powi(2))
                + (2.0
                    * cf
                    * l1v.powi(2)
                    * (2.0 + 2.0 * ca.powi(4) - 2.0 * v - 10.0 * ca.powi(4) * v + v.powi(2)
                        - ca.powi(2) * v.powi(2)
                        + 21.0 * ca.powi(4) * v.powi(2)
                        - ca.powi(2) * v.powi(3)
                        - 20.0 * ca.powi(4) * v.powi(3)
                        + 8.0 * ca.powi(4) * v.powi(4)))
                    / (ca * (1.0 - v) * v.powi(2))
                + (2.0
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
                - (cf
                    * (21.0 - 42.0 * ca.powi(2) + 21.0 * ca.powi(4) - 4.0 * pi2
                        + 8.0 * ca.powi(2) * pi2
                        - 4.0 * ca.powi(4) * pi2
                        - 42.0 * v
                        + 132.0 * ca.powi(2) * v
                        - 78.0 * ca.powi(4) * v
                        + 8.0 * pi2 * v
                        - 24.0 * ca.powi(2) * pi2 * v
                        + 16.0 * ca.powi(4) * pi2 * v
                        + 42.0 * v.powi(2)
                        - 240.0 * ca.powi(2) * v.powi(2)
                        + 138.0 * ca.powi(4) * v.powi(2)
                        - 8.0 * pi2 * v.powi(2)
                        + 40.0 * ca.powi(2) * pi2 * v.powi(2)
                        - 32.0 * ca.powi(4) * pi2 * v.powi(2)
                        + 216.0 * ca.powi(2) * v.powi(3)
                        - 120.0 * ca.powi(4) * v.powi(3)
                        - 32.0 * ca.powi(2) * pi2 * v.powi(3)
                        + 32.0 * ca.powi(4) * pi2 * v.powi(3)
                        - 108.0 * ca.powi(2) * v.powi(4)
                        + 60.0 * ca.powi(4) * v.powi(4)
                        + 16.0 * ca.powi(2) * pi2 * v.powi(4)
                        - 16.0 * ca.powi(4) * pi2 * v.powi(4)))
                    / (3.0 * ca * (1.0 - v) * v.powi(2))
        }
        _ => unreachable!("j0 must be in 1..=16, got {j0}"),
    }
}

/// `AVLO(W,V,S)`: the `log(1-W)/(1-W)+` term.
#[must_use]
pub fn avlo(j0: usize, _w: f64, v: f64, _s: f64, ctx: &MeContext) -> f64 {
    let ca = ctx.ca;
    let cf = ctx.cf;

    match j0 {
        1 | 3 => (40.0 * ca * cf.powi(2) * (1.0 + v.powi(2))) / ((-1.0 + v).powi(2) * v),
        2 | 4 | 7..=10 => 0.0,
        5 => (40.0 * ca * cf.powi(2) * (1.0 - 2.0 * v + 2.0 * v.powi(2))) / v,
        6 => {
            (80.0
                * cf.powi(2)
                * (ca - v - 3.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2) - 2.0 * ca * v.powi(3)
                    + ca * v.powi(4)))
                / ((-1.0 + v).powi(2) * v.powi(3))
        }
        11 => {
            (80.0
                * cf.powi(2)
                * (ca - 2.0 * ca * v + v.powi(2) + 4.0 * ca * v.powi(2)
                    - v.powi(3)
                    - 3.0 * ca * v.powi(3)
                    + ca * v.powi(4)))
                / ((-1.0 + v).powi(2) * v)
        }
        12 => {
            (-16.0
                * (-2.0 + 3.0 * ca.powi(2))
                * cf
                * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                * (cf - ca * v + ca * v.powi(2)))
                / ((-1.0 + v) * v.powi(2))
        }
        13 => {
            (8.0 * (-2.0 + 3.0 * ca.powi(2))
                * cf
                * (1.0 + v.powi(2))
                * (2.0 * ca * cf + 2.0 * v - v.powi(2) + ca.powi(2) * v.powi(2)))
                / (ca * (-1.0 + v).powi(2) * v.powi(2))
        }
        14 => {
            (-4.0
                * (-1.0 + 3.0 * ca)
                * (1.0 + 3.0 * ca)
                * cf
                * (2.0 - 2.0 * v + v.powi(2))
                * (2.0 * ca.powi(2) - 2.0 * ca.powi(2) * v - v.powi(2) + ca.powi(2) * v.powi(2)))
                / (ca * (-1.0 + v) * v.powi(3))
        }
        15 => {
            (1280.0 * ca.powi(3) * (1.0 - v + v.powi(2)).powi(3)) / ((-1.0 + v).powi(2) * v.powi(3))
        }
        16 => {
            (-8.0
                * (-1.0 + 3.0 * ca)
                * (1.0 + 3.0 * ca)
                * cf
                * (1.0 - 2.0 * v + 2.0 * v.powi(2))
                * (cf - ca * v + ca * v.powi(2)))
                / ((-1.0 + v) * v.powi(2))
        }
        _ => unreachable!("j0 must be in 1..=16, got {j0}"),
    }
}

/// `AVGO(W,V)`: always zero in the Fortran source (unused/placeholder).
#[must_use]
pub fn avgo(_w: f64, _v: f64) -> f64 {
    0.0
}

/// `STRUV(W,V,X3,S)`: dispatches to `STRUV1..16` by channel `j0`.
#[must_use]
pub fn struv(j0: usize, w: f64, v: f64, x3: f64, s: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    match j0 {
        1 => struv1(w, v, x3, s, ctx, pre),
        2 => struv2(w, v, x3, s, ctx, pre),
        3 => struv3(w, v, x3, s, ctx, pre),
        4 => struv4(w, v, x3, s, ctx, pre),
        5 => struv5(w, v, x3, s, ctx, pre),
        6 => struv6(w, v, x3, s, ctx, pre),
        7 => struv7(w, v, x3, s, ctx, pre),
        8 => struv8(w, v, x3, s, ctx, pre),
        9 => struv9(w, v, x3, s, ctx, pre),
        10 => struv10(w, v, x3, s, ctx, pre),
        11 => struv11(w, v, x3, s, ctx, pre),
        12 => struv12(w, v, x3, s, ctx, pre),
        13 => struv13(w, v, x3, s, ctx, pre),
        14 => struv14(w, v, x3, s, ctx, pre),
        15 => struv15(w, v, x3, s, ctx, pre),
        16 => struv16(w, v, x3, s, ctx, pre),
        _ => unreachable!("j0 must be in 1..=16, got {j0}"),
    }
}

/// `STRUV1(W,V,X3,S)`. `x3`/`s` are unused, matching the Fortran (the
/// function's value depends only on `w`, `v`, and the precomputed powers
/// and logs in `pre`).
#[must_use]
pub fn struv1(w: f64, v: f64, _x3: f64, _s: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let cf = ctx.cf;
    let ca = ctx.ca;
    let (v2, v3, v4, v5, v6, v7) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7);
    let (w2, w3, w4, w5, w6, w7, w8) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8);
    let ca2 = pre.ca2;
    let cacf = ca * cf; // CA*CF appears literally, not via COMMON /PRECOLOR/ CACF here
    let (l1v, lv, l1w, lw, lvw, l1vw, lms, lmss) = (
        pre.l1v, pre.lv, pre.l1w, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.lmss,
    );

    let part1 = (-4.0
        * cf
        * l1v
        * (2.0 - 4.0 * ca2 - 4.0 * v + ca2 * v - ca2 * v2 + 4.0 * v * w
            - 3.0 * ca2 * v * w
            - 4.0 * v2 * w
            + 3.0 * ca2 * v2 * w
            + 2.0 * v2 * w2))
        / ((1.0 - v).powi(2) * (1.0 - v * w));

    let part2 = (4.0
        * ca
        * cf.powi(2)
        * lmss
        * (4.0 - 5.0 * v + 2.0 * v2 - v3 + 9.0 * v * w - 8.0 * v2 * w
            + 3.0 * v3 * w
            + 6.0 * v2 * w2
            - 4.0 * v3 * w2
            + 2.0 * v3 * w3))
        / ((1.0 - v).powi(2) * (1.0 - v + v * w));

    let part3 = (4.0
        * cf
        * lvw
        * (2.0 * cacf - v2 + ca2 * v2 + 5.0 * w + 2.0 * ca2 * w
            - v * w
            - 4.0 * ca2 * v * w
            - v2 * w
            + 3.0 * ca2 * v2 * w
            + v3 * w
            - ca2 * v3 * w
            + v * w2
            + 4.0 * ca2 * v * w2
            - 2.0 * v2 * w2
            - 3.0 * ca2 * v2 * w2
            - 3.0 * v3 * w2
            + 3.0 * ca2 * v3 * w2
            + 4.0 * ca2 * v2 * w3
            + 4.0 * v3 * w3
            - 4.0 * ca2 * v3 * w3
            - 2.0 * v3 * w4
            + 2.0 * ca2 * v3 * w4))
        / ((1.0 - v).powi(2) * v * w);

    let part4 = -(4.0
        * cf
        * l1vw
        * (2.0 * ca2 - 4.0 * cacf + 3.0 * v - 6.0 * ca2 * v + 12.0 * cacf * v - 7.0 * v2
            + 7.0 * ca2 * v2
            - 14.0 * cacf * v2
            + 5.0 * v3
            - 4.0 * ca2 * v3
            + 8.0 * cacf * v3
            - v4
            + ca2 * v4
            - 2.0 * cacf * v4
            - 3.0 * v * w
            + 6.0 * ca2 * v * w
            - 12.0 * cacf * v * w
            + 10.0 * v2 * w
            - 13.0 * ca2 * v2 * w
            + 28.0 * cacf * v2 * w
            - 11.0 * v3 * w
            + 11.0 * ca2 * v3 * w
            - 24.0 * cacf * v3 * w
            + 4.0 * v4 * w
            - 4.0 * ca2 * v4 * w
            + 8.0 * cacf * v4 * w
            - 7.0 * v2 * w2
            + 8.0 * ca2 * v2 * w2
            - 18.0 * cacf * v2 * w2
            + 12.0 * v3 * w2
            - 13.0 * ca2 * v3 * w2
            + 28.0 * cacf * v3 * w2
            - 7.0 * v4 * w2
            + 7.0 * ca2 * v4 * w2
            - 14.0 * cacf * v4 * w2
            - 6.0 * v3 * w3
            + 6.0 * ca2 * v3 * w3
            - 12.0 * cacf * v3 * w3
            + 6.0 * v4 * w3
            - 6.0 * ca2 * v4 * w3
            + 12.0 * cacf * v4 * w3
            - 2.0 * v4 * w4
            + 2.0 * ca2 * v4 * w4
            - 4.0 * cacf * v4 * w4))
        / ((1.0 - v).powi(2) * v * (1.0 - v + v * w));

    let part5 = -(2.0
        * cf
        * lms
        * (4.0 * cacf + 4.0 * cacf * v2 - 2.0 * cacf * w + 2.0 * v * w
            - 18.0 * cacf * v * w
            - 4.0 * cacf * v2 * w
            + 2.0 * v3 * w
            - 8.0 * cacf * v3 * w
            + 6.0 * cacf * v * w2
            - 5.0 * v2 * w2
            + ca2 * v2 * w2
            + 18.0 * cacf * v2 * w2
            - 4.0 * v3 * w2
            + 16.0 * cacf * v3 * w2
            - v4 * w2
            + ca2 * v4 * w2
            + 8.0 * cacf * v4 * w2
            - 6.0 * cacf * v2 * w3
            + 8.0 * v3 * w3
            - 2.0 * ca2 * v3 * w3
            - 6.0 * cacf * v3 * w3
            + 2.0 * v4 * w3
            - 2.0 * ca2 * v4 * w3
            - 16.0 * cacf * v4 * w3
            + 2.0 * v5 * w3
            - 4.0 * cacf * v5 * w3
            + 2.0 * cacf * v3 * w4
            - 7.0 * v4 * w4
            + 3.0 * ca2 * v4 * w4
            + 2.0 * cacf * v4 * w4
            - 4.0 * v5 * w4
            + 4.0 * cacf * v5 * w4
            - v6 * w4
            + ca2 * v6 * w4
            + 6.0 * v5 * w5
            - 2.0 * ca2 * v5 * w5
            + 2.0 * v6 * w5
            - 2.0 * ca2 * v6 * w5
            - 2.0 * v6 * w6
            + 2.0 * ca2 * v6 * w6))
        / ((1.0 - v).powi(2) * v * w * (1.0 - v * w).powi(3))
        - (2.0
            * cf
            * l1w
            * (2.0 - 2.0 * ca2 - 2.0 * v + 2.0 * ca2 * v + 2.0 * v2 - 2.0 * ca2 * v2 - 2.0 * v3
                + 2.0 * ca2 * v3
                - w
                + ca2 * w
                - 24.0 * v * w
                + 18.0 * ca2 * v * w
                + 34.0 * v2 * w
                - 22.0 * ca2 * v2 * w
                - 12.0 * v3 * w
                + 6.0 * ca2 * v3 * w
                + 7.0 * v4 * w
                - 7.0 * ca2 * v4 * w
                + 2.0 * v * w2
                - 2.0 * ca2 * v * w2
                + 7.0 * v2 * w2
                - 3.0 * ca2 * v2 * w2
                - 27.0 * v3 * w2
                + 17.0 * ca2 * v3 * w2
                - 2.0 * v4 * w2
                + 4.0 * ca2 * v4 * w2
                - 8.0 * v5 * w2
                + 12.0 * ca2 * v5 * w2
                + 39.0 * v3 * w3
                - 31.0 * ca2 * v3 * w3
                - 16.0 * v4 * w3
                + 14.0 * ca2 * v4 * w3
                + 22.0 * v5 * w3
                - 28.0 * ca2 * v5 * w3
                + 7.0 * v6 * w3
                - 7.0 * ca2 * v6 * w3
                - 2.0 * v3 * w4
                + 2.0 * ca2 * v3 * w4
                - 19.0 * v4 * w4
                + 11.0 * ca2 * v4 * w4
                + 11.0 * v5 * w4
                + 3.0 * ca2 * v5 * w4
                - 24.0 * v6 * w4
                + 18.0 * ca2 * v6 * w4
                - 2.0 * v7 * w4
                + 2.0 * ca2 * v7 * w4
                + v4 * w5
                - ca2 * v4 * w5
                - 13.0 * v5 * w5
                + 7.0 * ca2 * v5 * w5
                + 18.0 * v6 * w5
                - 12.0 * ca2 * v6 * w5
                + 6.0 * v7 * w5
                - 6.0 * ca2 * v7 * w5
                - 8.0 * v7 * w6
                + 8.0 * ca2 * v7 * w6
                + 4.0 * v7 * w7
                - 4.0 * ca2 * v7 * w7))
            / ((1.0 - v).powi(2) * v * w * (1.0 - v * w).powi(3) * (1.0 - v + v * w));

    let part6 = -(2.0
        * cf
        * lv
        * (2.0 - 2.0 * ca2 - 2.0 * v + 2.0 * ca2 * v + 2.0 * v2 - 2.0 * ca2 * v2 - 2.0 * v3
            + 2.0 * ca2 * v3
            - w
            + ca2 * w
            - 44.0 * v * w
            + 26.0 * ca2 * v * w
            + 54.0 * v2 * w
            - 32.0 * ca2 * v2 * w
            - 20.0 * v3 * w
            + 10.0 * ca2 * v3 * w
            + 15.0 * v4 * w
            - 9.0 * ca2 * v4 * w
            + 2.0 * v * w2
            - 2.0 * ca2 * v * w2
            + 11.0 * v2 * w2
            - 5.0 * ca2 * v2 * w2
            - 35.0 * v3 * w2
            + 23.0 * ca2 * v3 * w2
            - 10.0 * v4 * w2
            + 4.0 * ca2 * v4 * w2
            - 24.0 * v5 * w2
            + 16.0 * ca2 * v5 * w2
            + 71.0 * v3 * w3
            - 45.0 * ca2 * v3 * w3
            - 40.0 * v4 * w3
            + 26.0 * ca2 * v4 * w3
            + 62.0 * v5 * w3
            - 40.0 * ca2 * v5 * w3
            + 15.0 * v6 * w3
            - 9.0 * ca2 * v6 * w3
            - 2.0 * v3 * w4
            + 2.0 * ca2 * v3 * w4
            - 19.0 * v4 * w4
            + 13.0 * ca2 * v4 * w4
            + 3.0 * v5 * w4
            + ca2 * v5 * w4
            - 48.0 * v6 * w4
            + 26.0 * ca2 * v6 * w4
            - 2.0 * v7 * w4
            + 2.0 * ca2 * v7 * w4
            + v4 * w5
            - ca2 * v4 * w5
            - 25.0 * v5 * w5
            + 13.0 * ca2 * v5 * w5
            + 38.0 * v6 * w5
            - 18.0 * ca2 * v6 * w5
            + 6.0 * v7 * w5
            - 6.0 * ca2 * v7 * w5
            - 4.0 * v6 * w6
            - 8.0 * v7 * w6
            + 8.0 * ca2 * v7 * w6
            + 4.0 * v7 * w7
            - 4.0 * ca2 * v7 * w7))
        / ((1.0 - v).powi(2) * v * w * (1.0 - v * w).powi(3) * (1.0 - v + v * w));

    let part7 = -(cf
        * (6.0 - 6.0 * ca2 + 4.0 * cacf - 14.0 * v + 14.0 * ca2 * v - 12.0 * cacf * v + 14.0 * v2
            - 14.0 * ca2 * v2
            + 12.0 * cacf * v2
            - 6.0 * v3
            + 6.0 * ca2 * v3
            - 4.0 * cacf * v3
            + 4.0 * w
            - 4.0 * cacf * w
            - 25.0 * v * w
            + 5.0 * ca2 * v * w
            + 4.0 * cacf * v * w
            + 70.0 * v2 * w
            - 30.0 * ca2 * v2 * w
            + 24.0 * cacf * v2 * w
            - 83.0 * v3 * w
            + 47.0 * ca2 * v3 * w
            - 44.0 * cacf * v3 * w
            + 38.0 * v4 * w
            - 26.0 * ca2 * v4 * w
            + 20.0 * cacf * v4 * w
            - 8.0 * v * w2
            + 8.0 * cacf * v * w2
            - 2.0 * v2 * w2
            + 10.0 * ca2 * v2 * w2
            - 36.0 * cacf * v2 * w2
            + v3 * w2
            - 13.0 * ca2 * v3 * w2
            + 20.0 * cacf * v3 * w2
            + 39.0 * v4 * w2
            - 11.0 * ca2 * v4 * w2
            + 40.0 * cacf * v4 * w2
            - 40.0 * v5 * w2
            + 24.0 * ca2 * v5 * w2
            - 32.0 * cacf * v5 * w2
            + 48.0 * v3 * w3
            - 8.0 * ca2 * v3 * w3
            + 36.0 * cacf * v3 * w3
            - 103.0 * v4 * w3
            + 15.0 * ca2 * v4 * w3
            - 64.0 * cacf * v4 * w3
            + 65.0 * v5 * w3
            - 5.0 * ca2 * v5 * w3
            + 8.0 * cacf * v5 * w3
            + 6.0 * v6 * w3
            - 18.0 * ca2 * v6 * w3
            + 20.0 * cacf * v6 * w3
            + 8.0 * v3 * w4
            - 8.0 * cacf * v3 * w4
            - 20.0 * v4 * w4
            + 12.0 * ca2 * v4 * w4
            + 5.0 * v5 * w4
            - 17.0 * ca2 * v5 * w4
            + 32.0 * cacf * v5 * w4
            - 13.0 * v6 * w4
            + 25.0 * ca2 * v6 * w4
            - 20.0 * cacf * v6 * w4
            - 6.0 * v7 * w4
            + 6.0 * ca2 * v7 * w4
            - 4.0 * cacf * v7 * w4
            - 4.0 * v4 * w5
            + 4.0 * cacf * v4 * w5
            + 13.0 * v5 * w5
            - ca2 * v5 * w5
            - 8.0 * cacf * v5 * w5
            + 9.0 * v6 * w5
            - 17.0 * ca2 * v6 * w5
            + 10.0 * v7 * w5
            - 10.0 * ca2 * v7 * w5
            + 4.0 * cacf * v7 * w5
            - 8.0 * v6 * w6
            + 8.0 * ca2 * v6 * w6
            - 8.0 * v7 * w6
            + 8.0 * ca2 * v7 * w6
            + 4.0 * v7 * w7
            - 4.0 * ca2 * v7 * w7))
        / ((1.0 - v).powi(2) * v * w * (1.0 - v * w).powi(3) * (1.0 - v + v * w));

    let part8 = -(2.0
        * cf
        * lw
        * (3.0 - 3.0 * ca2 + 2.0 * cacf - 3.0 * v + 3.0 * ca2 * v - 2.0 * cacf * v + 3.0 * v2
            - 3.0 * ca2 * v2
            + 2.0 * cacf * v2
            - 3.0 * v3
            + 3.0 * ca2 * v3
            - 2.0 * cacf * v3
            + 14.0 * w
            - 8.0 * ca2 * w
            + 8.0 * cacf * w
            - 48.0 * v * w
            + 30.0 * ca2 * v * w
            - 28.0 * cacf * v * w
            + 57.0 * v2 * w
            - 37.0 * ca2 * v2 * w
            + 34.0 * cacf * v2 * w
            - 36.0 * v3 * w
            + 22.0 * ca2 * v3 * w
            - 20.0 * cacf * v3 * w
            + 19.0 * v4 * w
            - 13.0 * ca2 * v4 * w
            + 10.0 * cacf * v4 * w
            + w2
            - ca2 * w2
            + 2.0 * cacf * w2
            - v * w2
            + ca2 * v * w2
            - 2.0 * cacf * v * w2
            + 23.0 * v2 * w2
            - 13.0 * ca2 * v2 * w2
            + 10.0 * cacf * v2 * w2
            - 50.0 * v3 * w2
            + 32.0 * ca2 * v3 * w2
            - 24.0 * cacf * v3 * w2
            + 33.0 * v4 * w2
            - 17.0 * ca2 * v4 * w2
            + 14.0 * cacf * v4 * w2
            - 28.0 * v5 * w2
            + 16.0 * ca2 * v5 * w2
            - 12.0 * cacf * v5 * w2
            - 2.0 * v * w3
            + 2.0 * ca2 * v * w3
            - 4.0 * cacf * v * w3
            - 3.0 * v2 * w3
            + ca2 * v2 * w3
            + 2.0 * cacf * v2 * w3
            + 10.0 * v3 * w3
            - 8.0 * ca2 * v3 * w3
            + 8.0 * cacf * v3 * w3
            + 2.0 * v4 * w3
            - 6.0 * ca2 * v4 * w3
            + 8.0 * cacf * v4 * w3
            + 16.0 * v5 * w3
            - 6.0 * ca2 * v5 * w3
            - 4.0 * cacf * v5 * w3
            + 15.0 * v6 * w3
            - 9.0 * ca2 * v6 * w3
            + 10.0 * cacf * v6 * w3
            - 19.0 * v3 * w4
            + 13.0 * ca2 * v3 * w4
            - 14.0 * cacf * v3 * w4
            + 30.0 * v4 * w4
            - 10.0 * ca2 * v4 * w4
            - 4.0 * cacf * v4 * w4
            - 26.0 * v5 * w4
            + 8.0 * ca2 * v5 * w4
            + 12.0 * cacf * v5 * w4
            - 30.0 * v6 * w4
            + 18.0 * ca2 * v6 * w4
            - 20.0 * cacf * v6 * w4
            - v7 * w4
            + ca2 * v7 * w4
            - 2.0 * cacf * v7 * w4
            + 2.0 * v3 * w5
            - 2.0 * ca2 * v3 * w5
            + 4.0 * cacf * v3 * w5
            - 5.0 * v4 * w5
            - 3.0 * ca2 * v4 * w5
            + 10.0 * cacf * v4 * w5
            - 4.0 * v5 * w5
            + 8.0 * ca2 * v5 * w5
            - 16.0 * cacf * v5 * w5
            + 45.0 * v6 * w5
            - 25.0 * ca2 * v6 * w5
            + 22.0 * cacf * v6 * w5
            + 4.0 * v7 * w5
            - 4.0 * ca2 * v7 * w5
            + 8.0 * cacf * v7 * w5
            - v4 * w6
            + ca2 * v4 * w6
            - 2.0 * cacf * v4 * w6
            + 16.0 * v5 * w6
            - 10.0 * ca2 * v5 * w6
            + 8.0 * cacf * v5 * w6
            - 34.0 * v6 * w6
            + 16.0 * ca2 * v6 * w6
            - 12.0 * cacf * v6 * w6
            - 7.0 * v7 * w6
            + 7.0 * ca2 * v7 * w6
            - 14.0 * cacf * v7 * w6
            + 4.0 * v6 * w7
            + 6.0 * v7 * w7
            - 6.0 * ca2 * v7 * w7
            + 12.0 * cacf * v7 * w7
            - 2.0 * v7 * w8
            + 2.0 * ca2 * v7 * w8
            - 4.0 * cacf * v7 * w8))
        / ((1.0 - v).powi(2) * v * (1.0 - w) * w * (1.0 - v * w).powi(3) * (1.0 - v + v * w));

    part1 + part2 + part3 + part4 + part5 + part6 + part7 + part8
}

/// `STRUV2(W,V,X3,S)`.
#[must_use]
pub fn struv2(w: f64, v: f64, _x3: f64, _s: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let cf = ctx.cf;
    let ca = ctx.ca;
    let (v2, v3, v4, v5, v6, v7, v8) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8);
    let (w2, w3, w4, w5, w6, w7, w8) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8);
    let ca2 = pre.ca2;
    let (l1v, lw, lvw, l1vw, lms, l1w, lv) =
        (pre.l1v, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.l1w, pre.lv);

    let part1 = (16.0
        * ca
        * cf.powi(2)
        * l1vw
        * (1.0 - w)
        * (1.0 - v - v2 + v3 + v * w + 4.0 * v2 * w - 5.0 * v3 * w - 3.0 * v2 * w2
            + 5.0 * v3 * w2
            - v3 * w3))
        / ((1.0 - v) * v * w * (1.0 - v + v * w).powi(2));

    let part2 = -(4.0
        * cf
        * l1v
        * (1.0 - v - v * w)
        * (2.0 * ca2 - 2.0 * ca2 * v + 2.0 * ca2 * v2 - 2.0 * ca2 * v3 - 2.0 * v * w
            + ca2 * v * w
            - 2.0 * ca2 * v2 * w
            + 2.0 * v3 * w
            + 5.0 * ca2 * v3 * w
            + 2.0 * v2 * w2
            + ca2 * v2 * w2
            - 2.0 * v3 * w2
            - 5.0 * ca2 * v3 * w2
            + 2.0 * ca2 * v3 * w3))
        / ((1.0 - v).powi(2) * v2 * w2 * (1.0 - v + v * w));

    let part3 = -(8.0
        * ca
        * cf.powi(2)
        * lms
        * (1.0 + v2 - 2.0 * v2 * w + v2 * w2)
        * (1.0 - 4.0 * v + 6.0 * v2 - 4.0 * v3 + v4 + v * w - 3.0 * v2 * w + 3.0 * v3 * w
            - v4 * w
            + v2 * w2
            - 2.0 * v3 * w2
            + v4 * w2
            + v3 * w3
            - v4 * w3
            + v4 * w4))
        / ((1.0 - v).powi(2) * v2 * w2 * (1.0 - v + v * w).powi(2));

    let part4 = -(4.0
        * cf
        * lw
        * (2.0 - 4.0 * ca2 - 6.0 * v + 6.0 * ca2 * v + 8.0 * v2 - 8.0 * ca2 * v2 - 8.0 * v3
            + 8.0 * ca2 * v3
            + 6.0 * v4
            - 4.0 * ca2 * v4
            - 2.0 * v5
            + 2.0 * ca2 * v5
            + 2.0 * v * w
            + 2.0 * ca2 * v * w
            - 6.0 * v2 * w
            + 2.0 * ca2 * v2 * w
            + 10.0 * v3 * w
            - 14.0 * ca2 * v3 * w
            - 10.0 * v4 * w
            + 6.0 * ca2 * v4 * w
            + 4.0 * v5 * w
            - 4.0 * ca2 * v5 * w
            - 3.0 * v2 * w2
            - ca2 * v2 * w2
            - 5.0 * v3 * w2
            + 13.0 * ca2 * v3 * w2
            + 11.0 * v4 * w2
            - 3.0 * ca2 * v4 * w2
            - 3.0 * v5 * w2
            + 3.0 * ca2 * v5 * w2
            + 5.0 * v3 * w3
            - 5.0 * ca2 * v3 * w3
            - 6.0 * v4 * w3
            - 2.0 * ca2 * v4 * w3
            + v5 * w3
            - ca2 * v5 * w3
            + 2.0 * ca2 * v4 * w4))
        / ((1.0 - v).powi(2) * v2 * w2 * (1.0 - v + v * w));

    let part5 = (4.0
        * cf
        * lvw
        * (1.0 - w)
        * (4.0 * ca2 + 4.0 * ca2 * v2 + 2.0 * v * w
            - 9.0 * ca2 * v * w
            - 6.0 * ca2 * v2 * w
            - 2.0 * v3 * w
            - ca2 * v3 * w
            - 5.0 * v2 * w2
            + 12.0 * ca2 * v2 * w2
            + 2.0 * v3 * w2
            + 3.0 * ca2 * v3 * w2
            - v4 * w2
            + ca2 * v4 * w2
            + 2.0 * v3 * w3
            - 6.0 * ca2 * v3 * w3
            + 2.0 * v4 * w3
            - 2.0 * ca2 * v4 * w3
            - 2.0 * v4 * w4
            + 2.0 * ca2 * v4 * w4))
        / ((1.0 - v).powi(2) * v2 * w2);

    let part6 = (2.0
        * cf
        * lms
        * (2.0 - 6.0 * ca2 - 4.0 * v + 4.0 * v2 - 8.0 * ca2 * v2 - 4.0 * v3 + 2.0 * v4
            - 2.0 * ca2 * v4
            - 2.0 * w
            + 2.0 * ca2 * w
            + 20.0 * ca2 * v * w
            + 4.0 * v2 * w
            + 12.0 * ca2 * v2 * w
            - 4.0 * v3 * w
            + 16.0 * ca2 * v3 * w
            + 6.0 * v4 * w
            + 2.0 * ca2 * v4 * w
            - 4.0 * v5 * w
            + 4.0 * ca2 * v5 * w
            + w2
            - ca2 * w2
            + 2.0 * v * w2
            - 4.0 * ca2 * v * w2
            - 2.0 * v2 * w2
            - 36.0 * ca2 * v2 * w2
            + 2.0 * v3 * w2
            - 24.0 * ca2 * v3 * w2
            - v4 * w2
            - 13.0 * ca2 * v4 * w2
            - 4.0 * ca2 * v5 * w2
            + 2.0 * v6 * w2
            - 2.0 * ca2 * v6 * w2
            - 2.0 * v * w3
            + 2.0 * ca2 * v * w3
            + 2.0 * v2 * w3
            + 2.0 * ca2 * v2 * w3
            - 6.0 * v3 * w3
            + 40.0 * ca2 * v3 * w3
            - 4.0 * v4 * w3
            + 20.0 * ca2 * v4 * w3
            + 6.0 * ca2 * v5 * w3
            - 2.0 * v6 * w3
            + 2.0 * ca2 * v6 * w3
            + v2 * w4
            - ca2 * v2 * w4
            - 2.0 * v3 * w4
            + 11.0 * v4 * w4
            - 27.0 * ca2 * v4 * w4
            + 2.0 * v5 * w4
            - 8.0 * ca2 * v5 * w4
            + 2.0 * v6 * w4
            - 2.0 * ca2 * v6 * w4
            - 6.0 * v5 * w5
            + 10.0 * ca2 * v5 * w5
            - 2.0 * v6 * w5
            + 2.0 * ca2 * v6 * w5
            + 2.0 * v6 * w6
            - 2.0 * ca2 * v6 * w6))
        / ((1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w).powi(2));

    let part7 = -(2.0
        * cf
        * l1w
        * (4.0 - 12.0 * ca2 - 16.0 * v + 32.0 * ca2 * v + 28.0 * v2 - 44.0 * ca2 * v2 - 32.0 * v3
            + 48.0 * ca2 * v3
            + 28.0 * v4
            - 36.0 * ca2 * v4
            - 16.0 * v5
            + 16.0 * ca2 * v5
            + 4.0 * v6
            - 4.0 * ca2 * v6
            - 2.0 * w
            + 2.0 * ca2 * w
            + 18.0 * v * w
            + 8.0 * ca2 * v * w
            - 28.0 * v2 * w
            - 28.0 * ca2 * v2 * w
            + 32.0 * v3 * w
            - 4.0 * ca2 * v3 * w
            - 38.0 * v4 * w
            + 14.0 * ca2 * v4 * w
            + 6.0 * v5 * w
            + 20.0 * ca2 * v5 * w
            + 20.0 * v6 * w
            - 20.0 * ca2 * v6 * w
            - 8.0 * v7 * w
            + 8.0 * ca2 * v7 * w
            + w2
            - ca2 * w2
            - 4.0 * v * w2
            + 2.0 * ca2 * v * w2
            - 17.0 * v2 * w2
            + ca2 * v2 * w2
            + 28.0 * v3 * w2
            + 46.0 * ca2 * v3 * w2
            - 13.0 * v4 * w2
            - 13.0 * ca2 * v4 * w2
            + 72.0 * v5 * w2
            - 88.0 * ca2 * v5 * w2
            - 79.0 * v6 * w2
            + 49.0 * ca2 * v6 * w2
            + 8.0 * v7 * w2
            - 8.0 * ca2 * v7 * w2
            + 4.0 * v8 * w2
            - 4.0 * ca2 * v8 * w2
            + 6.0 * v2 * w3
            - 6.0 * ca2 * v2 * w3
            - 6.0 * v3 * w3
            - 12.0 * ca2 * v3 * w3
            - 32.0 * ca2 * v4 * w3
            - 92.0 * v5 * w3
            + 126.0 * ca2 * v5 * w3
            + 74.0 * v6 * w3
            - 22.0 * ca2 * v6 * w3
            + 30.0 * v7 * w3
            - 18.0 * ca2 * v7 * w3
            - 12.0 * v8 * w3
            + 12.0 * ca2 * v8 * w3
            - 2.0 * v2 * w4
            + 2.0 * ca2 * v2 * w4
            + 6.0 * v3 * w4
            - 2.0 * ca2 * v3 * w4
            - 2.0 * v4 * w4
            + 20.0 * ca2 * v4 * w4
            + 64.0 * v5 * w4
            - 66.0 * ca2 * v5 * w4
            - 8.0 * v6 * w4
            - 40.0 * ca2 * v6 * w4
            - 66.0 * v7 * w4
            + 38.0 * ca2 * v7 * w4
            + 16.0 * v8 * w4
            - 16.0 * ca2 * v8 * w4
            - 4.0 * v4 * w5
            + 4.0 * ca2 * v4 * w5
            - 16.0 * v5 * w5
            + 8.0 * ca2 * v5 * w5
            - 38.0 * v6 * w5
            + 50.0 * ca2 * v6 * w5
            + 50.0 * v7 * w5
            - 22.0 * ca2 * v7 * w5
            - 16.0 * v8 * w5
            + 16.0 * ca2 * v8 * w5
            + v4 * w6
            - ca2 * v4 * w6
            - 2.0 * v5 * w6
            + 23.0 * v6 * w6
            - 17.0 * ca2 * v6 * w6
            - 10.0 * v7 * w6
            - 2.0 * ca2 * v7 * w6
            + 16.0 * v8 * w6
            - 16.0 * ca2 * v8 * w6
            - 4.0 * v7 * w7
            + 4.0 * ca2 * v7 * w7
            - 12.0 * v8 * w7
            + 12.0 * ca2 * v8 * w7
            + 4.0 * v8 * w8
            - 4.0 * ca2 * v8 * w8))
        / ((1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2));

    let part8 = -(2.0
        * cf
        * lv
        * (4.0 - 16.0 * ca2 - 16.0 * v + 44.0 * ca2 * v + 28.0 * v2 - 60.0 * ca2 * v2 - 32.0 * v3
            + 64.0 * ca2 * v3
            + 28.0 * v4
            - 48.0 * ca2 * v4
            - 16.0 * v5
            + 20.0 * ca2 * v5
            + 4.0 * v6
            - 4.0 * ca2 * v6
            - 2.0 * w
            + 2.0 * ca2 * w
            + 22.0 * v * w
            + 14.0 * ca2 * v * w
            - 36.0 * v2 * w
            - 44.0 * ca2 * v2 * w
            + 32.0 * v3 * w
            + 8.0 * ca2 * v3 * w
            - 30.0 * v4 * w
            + 6.0 * ca2 * v4 * w
            + 2.0 * v5 * w
            + 34.0 * ca2 * v5 * w
            + 20.0 * v6 * w
            - 28.0 * ca2 * v6 * w
            - 8.0 * v7 * w
            + 8.0 * ca2 * v7 * w
            + w2
            - ca2 * w2
            - 4.0 * v * w2
            + 2.0 * ca2 * v * w2
            - 29.0 * v2 * w2
            + 3.0 * ca2 * v2 * w2
            + 56.0 * v3 * w2
            + 52.0 * ca2 * v3 * w2
            - 25.0 * v4 * w2
            - 7.0 * ca2 * v4 * w2
            + 60.0 * v5 * w2
            - 114.0 * ca2 * v5 * w2
            - 71.0 * v6 * w2
            + 57.0 * ca2 * v6 * w2
            + 8.0 * v7 * w2
            - 4.0 * ca2 * v7 * w2
            + 4.0 * v8 * w2
            - 4.0 * ca2 * v8 * w2
            + 6.0 * v2 * w3
            - 6.0 * ca2 * v2 * w3
            + 2.0 * v3 * w3
            - 20.0 * ca2 * v3 * w3
            - 32.0 * v4 * w3
            - 40.0 * ca2 * v4 * w3
            - 64.0 * v5 * w3
            + 148.0 * ca2 * v5 * w3
            + 74.0 * v6 * w3
            - 10.0 * ca2 * v6 * w3
            + 26.0 * v7 * w3
            - 28.0 * ca2 * v7 * w3
            - 12.0 * v8 * w3
            + 12.0 * ca2 * v8 * w3
            - 2.0 * v2 * w4
            + 2.0 * ca2 * v2 * w4
            + 6.0 * v3 * w4
            - 2.0 * ca2 * v3 * w4
            + 6.0 * v4 * w4
            + 28.0 * ca2 * v4 * w4
            + 72.0 * v5 * w4
            - 74.0 * ca2 * v5 * w4
            - 28.0 * v6 * w4
            - 70.0 * ca2 * v6 * w4
            - 62.0 * v7 * w4
            + 44.0 * ca2 * v7 * w4
            + 16.0 * v8 * w4
            - 16.0 * ca2 * v8 * w4
            - 4.0 * v4 * w5
            + 4.0 * ca2 * v4 * w5
            - 28.0 * v5 * w5
            + 6.0 * ca2 * v5 * w5
            - 30.0 * v6 * w5
            + 74.0 * ca2 * v6 * w5
            + 54.0 * v7 * w5
            - 16.0 * ca2 * v7 * w5
            - 16.0 * v8 * w5
            + 16.0 * ca2 * v8 * w5
            + v4 * w6
            - ca2 * v4 * w6
            - 2.0 * v5 * w6
            + 27.0 * v6 * w6
            - 23.0 * ca2 * v6 * w6
            - 14.0 * v7 * w6
            - 12.0 * ca2 * v7 * w6
            + 16.0 * v8 * w6
            - 16.0 * ca2 * v8 * w6
            - 4.0 * v7 * w7
            + 8.0 * ca2 * v7 * w7
            - 12.0 * v8 * w7
            + 12.0 * ca2 * v8 * w7
            + 4.0 * v8 * w8
            - 4.0 * ca2 * v8 * w8))
        / ((1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2));

    let part9 = -(2.0
        * cf
        * (2.0 - 6.0 * ca2 - 12.0 * v + 28.0 * ca2 * v + 30.0 * v2 - 54.0 * ca2 * v2 - 40.0 * v3
            + 56.0 * ca2 * v3
            + 30.0 * v4
            - 34.0 * ca2 * v4
            - 12.0 * v5
            + 12.0 * ca2 * v5
            + 2.0 * v6
            - 2.0 * ca2 * v6
            + w
            - ca2 * w
            - v * w
            + ca2 * v * w
            - 2.0 * v2 * w
            - 6.0 * ca2 * v2 * w
            - 4.0 * v3 * w
            + 28.0 * ca2 * v3 * w
            + 23.0 * v4 * w
            - 47.0 * ca2 * v4 * w
            - 31.0 * v5 * w
            + 39.0 * ca2 * v5 * w
            + 18.0 * v6 * w
            - 18.0 * ca2 * v6 * w
            - 4.0 * v7 * w
            + 4.0 * ca2 * v7 * w
            - w2
            + ca2 * w2
            + 9.0 * v * w2
            - 7.0 * ca2 * v * w2
            - 32.0 * v2 * w2
            + 22.0 * ca2 * v2 * w2
            + 34.0 * v3 * w2
            - 38.0 * ca2 * v3 * w2
            - 11.0 * v4 * w2
            + 39.0 * ca2 * v4 * w2
            + 13.0 * v5 * w2
            - 27.0 * ca2 * v5 * w2
            - 14.0 * v6 * w2
            + 12.0 * ca2 * v6 * w2
            + 2.0 * v8 * w2
            - 2.0 * ca2 * v8 * w2
            - 4.0 * v2 * w3
            + 4.0 * ca2 * v2 * w3
            + 58.0 * v3 * w3
            - 28.0 * ca2 * v3 * w3
            - 80.0 * v4 * w3
            + 32.0 * ca2 * v4 * w3
            - 4.0 * v5 * w3
            + 32.0 * v6 * w3
            - 8.0 * ca2 * v6 * w3
            + 4.0 * v7 * w3
            - 6.0 * ca2 * v7 * w3
            - 6.0 * v8 * w3
            + 6.0 * ca2 * v8 * w3
            + 2.0 * v2 * w4
            - 2.0 * ca2 * v2 * w4
            - 16.0 * v3 * w4
            + 12.0 * ca2 * v3 * w4
            - 12.0 * v4 * w4
            + 2.0 * ca2 * v4 * w4
            + 70.0 * v5 * w4
            - 32.0 * ca2 * v5 * w4
            - 44.0 * v6 * w4
            + 18.0 * ca2 * v6 * w4
            - 10.0 * v7 * w4
            + 8.0 * ca2 * v7 * w4
            + 10.0 * v8 * w4
            - 10.0 * ca2 * v8 * w4
            + 3.0 * v4 * w5
            - 3.0 * ca2 * v4 * w5
            - 11.0 * v5 * w5
            + 13.0 * ca2 * v5 * w5
            + 10.0 * v6 * w5
            - 6.0 * ca2 * v6 * w5
            + 10.0 * v7 * w5
            - 8.0 * ca2 * v7 * w5
            - 12.0 * v8 * w5
            + 12.0 * ca2 * v8 * w5
            - v4 * w6
            + ca2 * v4 * w6
            + 7.0 * v5 * w6
            - 5.0 * ca2 * v5 * w6
            - 12.0 * v6 * w6
            + 4.0 * ca2 * v6 * w6
            - 2.0 * v7 * w6
            + 4.0 * ca2 * v7 * w6
            + 10.0 * v8 * w6
            - 10.0 * ca2 * v8 * w6
            + 2.0 * v7 * w7
            - 2.0 * ca2 * v7 * w7
            - 6.0 * v8 * w7
            + 6.0 * ca2 * v8 * w7
            + 2.0 * v8 * w8
            - 2.0 * ca2 * v8 * w8))
        / ((1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2));

    part1 + part2 + part3 + part4 + part5 + part6 + part7 + part8 + part9
}

/// `STRUV3(W,V,X3,S)`.
#[must_use]
pub fn struv3(w: f64, v: f64, _x3: f64, _s: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let cf = ctx.cf;
    let ca = ctx.ca;
    let (v2, v3, v4, v5, v6, v7) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7);
    let (w2, w3, w4, w5, w6, w7, w8) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8);
    let ca2 = pre.ca2;
    let cacf = ca * cf;
    let (l1v, lw, lvw, l1vw, lms, l1w, lv) =
        (pre.l1v, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.l1w, pre.lv);

    let part1 = (4.0
        * cf
        * l1v
        * (6.0 + 2.0 * ca2 - 4.0 * v + ca2 * v + ca2 * v2 + 4.0 * v * w + ca2 * v * w
            - 4.0 * v2 * w
            - ca2 * v2 * w
            - 2.0 * v2 * w2))
        / ((1.0 - v).powi(2) * (1.0 - v * w));

    let part2 = (4.0
        * ca
        * cf.powi(2)
        * pre.lmss
        * (4.0 - 5.0 * v + 2.0 * v2 - v3 + 9.0 * v * w - 8.0 * v2 * w
            + 3.0 * v3 * w
            + 6.0 * v2 * w2
            - 4.0 * v3 * w2
            + 2.0 * v3 * w3))
        / ((1.0 - v).powi(2) * (1.0 - v + v * w));

    let part3 = (4.0
        * cf
        * lvw
        * (2.0 * cacf - v2 + ca2 * v2 - 3.0 * w + 4.0 * ca2 * w + 7.0 * v * w
            - 6.0 * ca2 * v * w
            - v2 * w
            + 3.0 * ca2 * v2 * w
            + v3 * w
            - ca2 * v3 * w
            - 7.0 * v * w2
            + 6.0 * ca2 * v * w2
            + 6.0 * v2 * w2
            - 5.0 * ca2 * v2 * w2
            - 3.0 * v3 * w2
            + 3.0 * ca2 * v3 * w2
            + 4.0 * ca2 * v2 * w3
            + 4.0 * v3 * w3
            - 4.0 * ca2 * v3 * w3
            - 2.0 * v3 * w4
            + 2.0 * ca2 * v3 * w4))
        / ((1.0 - v).powi(2) * v * w);

    let part4 = (4.0
        * cf
        * l1vw
        * (4.0 - 3.0 * ca2 + 4.0 * cacf - 11.0 * v + 8.0 * ca2 * v - 12.0 * cacf * v + 11.0 * v2
            - 8.0 * ca2 * v2
            + 14.0 * cacf * v2
            - 5.0 * v3
            + 4.0 * ca2 * v3
            - 8.0 * cacf * v3
            + v4
            - ca2 * v4
            + 2.0 * cacf * v4
            + 11.0 * v * w
            - 8.0 * ca2 * v * w
            + 12.0 * cacf * v * w
            - 22.0 * v2 * w
            + 16.0 * ca2 * v2 * w
            - 28.0 * cacf * v2 * w
            + 15.0 * v3 * w
            - 12.0 * ca2 * v3 * w
            + 24.0 * cacf * v3 * w
            - 4.0 * v4 * w
            + 4.0 * ca2 * v4 * w
            - 8.0 * cacf * v4 * w
            + 11.0 * v2 * w2
            - 9.0 * ca2 * v2 * w2
            + 18.0 * cacf * v2 * w2
            - 16.0 * v3 * w2
            + 14.0 * ca2 * v3 * w2
            - 28.0 * cacf * v3 * w2
            + 7.0 * v4 * w2
            - 7.0 * ca2 * v4 * w2
            + 14.0 * cacf * v4 * w2
            + 6.0 * v3 * w3
            - 6.0 * ca2 * v3 * w3
            + 12.0 * cacf * v3 * w3
            - 6.0 * v4 * w3
            + 6.0 * ca2 * v4 * w3
            - 12.0 * cacf * v4 * w3
            + 2.0 * v4 * w4
            - 2.0 * ca2 * v4 * w4
            + 4.0 * cacf * v4 * w4))
        / ((1.0 - v).powi(2) * v * (1.0 - v + v * w));

    let part5 = -(2.0
        * cf
        * lms
        * (4.0 * cacf + 4.0 * cacf * v2 - 2.0 * cacf * w + 2.0 * v * w
            - 18.0 * cacf * v * w
            - 4.0 * cacf * v2 * w
            + 2.0 * v3 * w
            - 8.0 * cacf * v3 * w
            + 6.0 * cacf * v * w2
            - 5.0 * v2 * w2
            + ca2 * v2 * w2
            + 18.0 * cacf * v2 * w2
            - 4.0 * v3 * w2
            + 16.0 * cacf * v3 * w2
            - v4 * w2
            + ca2 * v4 * w2
            + 8.0 * cacf * v4 * w2
            - 6.0 * cacf * v2 * w3
            + 8.0 * v3 * w3
            - 2.0 * ca2 * v3 * w3
            - 6.0 * cacf * v3 * w3
            + 2.0 * v4 * w3
            - 2.0 * ca2 * v4 * w3
            - 16.0 * cacf * v4 * w3
            + 2.0 * v5 * w3
            - 4.0 * cacf * v5 * w3
            + 2.0 * cacf * v3 * w4
            - 7.0 * v4 * w4
            + 3.0 * ca2 * v4 * w4
            + 2.0 * cacf * v4 * w4
            - 4.0 * v5 * w4
            + 4.0 * cacf * v5 * w4
            - v6 * w4
            + ca2 * v6 * w4
            + 6.0 * v5 * w5
            - 2.0 * ca2 * v5 * w5
            + 2.0 * v6 * w5
            - 2.0 * ca2 * v6 * w5
            - 2.0 * v6 * w6
            + 2.0 * ca2 * v6 * w6))
        / ((1.0 - v).powi(2) * v * w * (1.0 - v * w).powi(3));

    let part6 = -(2.0
        * cf
        * l1w
        * (2.0 - 2.0 * ca2 - 2.0 * v + 2.0 * ca2 * v + 2.0 * v2 - 2.0 * ca2 * v2 - 2.0 * v3
            + 2.0 * ca2 * v3
            - w
            + ca2 * w
            - 8.0 * v * w
            + 14.0 * ca2 * v * w
            + 2.0 * v2 * w
            - 14.0 * ca2 * v2 * w
            + 4.0 * v3 * w
            + 2.0 * ca2 * v3 * w
            + 7.0 * v4 * w
            - 7.0 * ca2 * v4 * w
            + 2.0 * v * w2
            - 2.0 * ca2 * v * w2
            + 7.0 * v2 * w2
            - 3.0 * ca2 * v2 * w2
            - 11.0 * v3 * w2
            + 13.0 * ca2 * v3 * w2
            - 18.0 * v4 * w2
            + 8.0 * ca2 * v4 * w2
            - 8.0 * v5 * w2
            + 12.0 * ca2 * v5 * w2
            + 7.0 * v3 * w3
            - 23.0 * ca2 * v3 * w3
            + 32.0 * v4 * w3
            + 2.0 * ca2 * v4 * w3
            + 6.0 * v5 * w3
            - 24.0 * ca2 * v5 * w3
            + 7.0 * v6 * w3
            - 7.0 * ca2 * v6 * w3
            - 2.0 * v3 * w4
            + 2.0 * ca2 * v3 * w4
            - 19.0 * v4 * w4
            + 11.0 * ca2 * v4 * w4
            - 5.0 * v5 * w4
            + 7.0 * ca2 * v5 * w4
            - 8.0 * v6 * w4
            + 14.0 * ca2 * v6 * w4
            - 2.0 * v7 * w4
            + 2.0 * ca2 * v7 * w4
            + v4 * w5
            - ca2 * v4 * w5
            + 3.0 * v5 * w5
            + 3.0 * ca2 * v5 * w5
            + 2.0 * v6 * w5
            - 8.0 * ca2 * v6 * w5
            + 6.0 * v7 * w5
            - 6.0 * ca2 * v7 * w5
            - 8.0 * v7 * w6
            + 8.0 * ca2 * v7 * w6
            + 4.0 * v7 * w7
            - 4.0 * ca2 * v7 * w7))
        / ((1.0 - v).powi(2) * v * w * (1.0 - v * w).powi(3) * (1.0 - v + v * w));

    let part7 = -(2.0
        * cf
        * lv
        * (2.0 - 2.0 * ca2 - 2.0 * v + 2.0 * ca2 * v + 2.0 * v2 - 2.0 * ca2 * v2 - 2.0 * v3
            + 2.0 * ca2 * v3
            - w
            + ca2 * w
            + 20.0 * v * w
            + 10.0 * ca2 * v * w
            - 26.0 * v2 * w
            - 12.0 * ca2 * v2 * w
            + 12.0 * v3 * w
            + 2.0 * ca2 * v3 * w
            - v4 * w
            - 5.0 * ca2 * v4 * w
            + 2.0 * v * w2
            - 2.0 * ca2 * v * w2
            - 5.0 * v2 * w2
            - ca2 * v2 * w2
            + 13.0 * v3 * w2
            + 11.0 * ca2 * v3 * w2
            - 10.0 * v4 * w2
            + 4.0 * ca2 * v4 * w2
            + 8.0 * v5 * w2
            + 8.0 * ca2 * v5 * w2
            - 41.0 * v3 * w3
            - 17.0 * ca2 * v3 * w3
            + 56.0 * v4 * w3
            + 2.0 * ca2 * v4 * w3
            - 34.0 * v5 * w3
            - 16.0 * ca2 * v5 * w3
            - v6 * w3
            - 5.0 * ca2 * v6 * w3
            - 2.0 * v3 * w4
            + 2.0 * ca2 * v3 * w4
            - 3.0 * v4 * w4
            + 9.0 * ca2 * v4 * w4
            - 13.0 * v5 * w4
            + 5.0 * ca2 * v5 * w4
            + 16.0 * v6 * w4
            + 10.0 * ca2 * v6 * w4
            - 2.0 * v7 * w4
            + 2.0 * ca2 * v7 * w4
            + v4 * w5
            - ca2 * v4 * w5
            + 23.0 * v5 * w5
            + ca2 * v5 * w5
            - 10.0 * v6 * w5
            - 6.0 * ca2 * v6 * w5
            + 6.0 * v7 * w5
            - 6.0 * ca2 * v7 * w5
            - 4.0 * v6 * w6
            - 8.0 * v7 * w6
            + 8.0 * ca2 * v7 * w6
            + 4.0 * v7 * w7
            - 4.0 * ca2 * v7 * w7))
        / ((1.0 - v).powi(2) * v * w * (1.0 - v * w).powi(3) * (1.0 - v + v * w));

    let part8 = -(cf
        * (6.0 - 6.0 * ca2 + 4.0 * cacf - 14.0 * v + 14.0 * ca2 * v - 12.0 * cacf * v + 14.0 * v2
            - 14.0 * ca2 * v2
            + 12.0 * cacf * v2
            - 6.0 * v3
            + 6.0 * ca2 * v3
            - 4.0 * cacf * v3
            - 12.0 * w
            + 4.0 * ca2 * w
            - 4.0 * cacf * w
            + 39.0 * v * w
            - 11.0 * ca2 * v * w
            + 4.0 * cacf * v * w
            - 42.0 * v2 * w
            - 2.0 * ca2 * v2 * w
            + 24.0 * cacf * v2 * w
            + 13.0 * v3 * w
            + 23.0 * ca2 * v3 * w
            - 44.0 * cacf * v3 * w
            + 6.0 * v4 * w
            - 18.0 * ca2 * v4 * w
            + 20.0 * cacf * v4 * w
            + 24.0 * v * w2
            - 8.0 * ca2 * v * w2
            + 8.0 * cacf * v * w2
            - 34.0 * v2 * w2
            + 18.0 * ca2 * v2 * w2
            - 36.0 * cacf * v2 * w2
            - 31.0 * v3 * w2
            - 5.0 * ca2 * v3 * w2
            + 20.0 * cacf * v3 * w2
            + 71.0 * v4 * w2
            - 19.0 * ca2 * v4 * w2
            + 40.0 * cacf * v4 * w2
            - 40.0 * v5 * w2
            + 24.0 * ca2 * v5 * w2
            - 32.0 * cacf * v5 * w2
            + 16.0 * v3 * w3
            + 36.0 * cacf * v3 * w3
            - 7.0 * v4 * w3
            - 9.0 * ca2 * v4 * w3
            - 64.0 * cacf * v4 * w3
            - 31.0 * v5 * w3
            + 19.0 * ca2 * v5 * w3
            + 8.0 * cacf * v5 * w3
            + 38.0 * v6 * w3
            - 26.0 * ca2 * v6 * w3
            + 20.0 * cacf * v6 * w3
            - 24.0 * v3 * w4
            + 8.0 * ca2 * v3 * w4
            - 8.0 * cacf * v3 * w4
            + 12.0 * v4 * w4
            + 4.0 * ca2 * v4 * w4
            + 37.0 * v5 * w4
            - 25.0 * ca2 * v5 * w4
            + 32.0 * cacf * v5 * w4
            - 45.0 * v6 * w4
            + 33.0 * ca2 * v6 * w4
            - 20.0 * cacf * v6 * w4
            - 6.0 * v7 * w4
            + 6.0 * ca2 * v7 * w4
            - 4.0 * cacf * v7 * w4
            + 12.0 * v4 * w5
            - 4.0 * ca2 * v4 * w5
            + 4.0 * cacf * v4 * w5
            - 19.0 * v5 * w5
            + 7.0 * ca2 * v5 * w5
            - 8.0 * cacf * v5 * w5
            + 25.0 * v6 * w5
            - 21.0 * ca2 * v6 * w5
            + 10.0 * v7 * w5
            - 10.0 * ca2 * v7 * w5
            + 4.0 * cacf * v7 * w5
            - 8.0 * v6 * w6
            + 8.0 * ca2 * v6 * w6
            - 8.0 * v7 * w6
            + 8.0 * ca2 * v7 * w6
            + 4.0 * v7 * w7
            - 4.0 * ca2 * v7 * w7))
        / ((1.0 - v).powi(2) * v * w * (1.0 - v * w).powi(3) * (1.0 - v + v * w));

    let part9 = -(2.0
        * cf
        * lw
        * (3.0 - 3.0 * ca2 + 2.0 * cacf - 3.0 * v + 3.0 * ca2 * v - 2.0 * cacf * v + 3.0 * v2
            - 3.0 * ca2 * v2
            + 2.0 * cacf * v2
            - 3.0 * v3
            + 3.0 * ca2 * v3
            - 2.0 * cacf * v3
            - 10.0 * w
            - 2.0 * ca2 * w
            + 8.0 * cacf * w
            + 16.0 * v * w
            + 14.0 * ca2 * v * w
            - 28.0 * cacf * v * w
            - 15.0 * v2 * w
            - 19.0 * ca2 * v2 * w
            + 34.0 * cacf * v2 * w
            + 12.0 * v3 * w
            + 10.0 * ca2 * v3 * w
            - 20.0 * cacf * v3 * w
            + 3.0 * v4 * w
            - 9.0 * ca2 * v4 * w
            + 10.0 * cacf * v4 * w
            + w2
            - ca2 * w2
            + 2.0 * cacf * w2
            + 7.0 * v * w2
            - ca2 * v * w2
            - 2.0 * cacf * v * w2
            - 17.0 * v2 * w2
            - 3.0 * ca2 * v2 * w2
            + 10.0 * cacf * v2 * w2
            + 14.0 * v3 * w2
            + 16.0 * ca2 * v3 * w2
            - 24.0 * cacf * v3 * w2
            - 31.0 * v4 * w2
            - ca2 * v4 * w2
            + 14.0 * cacf * v4 * w2
            + 4.0 * v5 * w2
            + 8.0 * ca2 * v5 * w2
            - 12.0 * cacf * v5 * w2
            - 2.0 * v * w3
            + 2.0 * ca2 * v * w3
            - 4.0 * cacf * v * w3
            - 3.0 * v2 * w3
            + ca2 * v2 * w3
            + 2.0 * cacf * v2 * w3
            + 10.0 * v3 * w3
            - 8.0 * ca2 * v3 * w3
            + 8.0 * cacf * v3 * w3
            + 34.0 * v4 * w3
            - 14.0 * ca2 * v4 * w3
            + 8.0 * cacf * v4 * w3
            - 2.0 * ca2 * v5 * w3
            - 4.0 * cacf * v5 * w3
            - v6 * w3
            - 5.0 * ca2 * v6 * w3
            + 10.0 * cacf * v6 * w3
            + 13.0 * v3 * w4
            + 5.0 * ca2 * v3 * w4
            - 14.0 * cacf * v3 * w4
            - 66.0 * v4 * w4
            + 14.0 * ca2 * v4 * w4
            - 4.0 * cacf * v4 * w4
            + 6.0 * v5 * w4
            + 12.0 * cacf * v5 * w4
            + 2.0 * v6 * w4
            + 10.0 * ca2 * v6 * w4
            - 20.0 * cacf * v6 * w4
            - v7 * w4
            + ca2 * v7 * w4
            - 2.0 * cacf * v7 * w4
            + 2.0 * v3 * w5
            - 2.0 * ca2 * v3 * w5
            + 4.0 * cacf * v3 * w5
            + 19.0 * v4 * w5
            - 9.0 * ca2 * v4 * w5
            + 10.0 * cacf * v4 * w5
            + 28.0 * v5 * w5
            - 16.0 * cacf * v5 * w5
            - 11.0 * v6 * w5
            - 11.0 * ca2 * v6 * w5
            + 22.0 * cacf * v6 * w5
            + 4.0 * v7 * w5
            - 4.0 * ca2 * v7 * w5
            + 8.0 * cacf * v7 * w5
            - v4 * w6
            + ca2 * v4 * w6
            - 2.0 * cacf * v4 * w6
            - 24.0 * v5 * w6
            + 8.0 * cacf * v5 * w6
            + 6.0 * v6 * w6
            + 6.0 * ca2 * v6 * w6
            - 12.0 * cacf * v6 * w6
            - 7.0 * v7 * w6
            + 7.0 * ca2 * v7 * w6
            - 14.0 * cacf * v7 * w6
            + 4.0 * v6 * w7
            + 6.0 * v7 * w7
            - 6.0 * ca2 * v7 * w7
            + 12.0 * cacf * v7 * w7
            - 2.0 * v7 * w8
            + 2.0 * ca2 * v7 * w8
            - 4.0 * cacf * v7 * w8))
        / ((1.0 - v).powi(2) * v * (1.0 - w) * w * (1.0 - v * w).powi(3) * (1.0 - v + v * w));

    part1 + part2 + part3 + part4 + part5 + part6 + part7 + part8 + part9
}

/// `STRUV4(W,V,X3,S)`.
#[must_use]
pub fn struv4(w: f64, v: f64, _x3: f64, _s: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let cf = ctx.cf;
    let ca = ctx.ca;
    let (v2, v3, v4, v5, v6, v7, v8) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8);
    let (w2, w3, w4, w5, w6, w7, w8) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8);
    let ca2 = pre.ca2;
    let (l1v, lw, lvw, l1vw, lms, l1w, lv) =
        (pre.l1v, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.l1w, pre.lv);

    let part1 = (8.0
        * cf
        * l1vw
        * (1.0 - w)
        * (3.0 - 3.0 * v - 3.0 * v2 + 3.0 * v3 + 3.0 * v * w + 4.0 * v2 * w + 2.0 * ca2 * v2 * w
            - 7.0 * v3 * w
            - 2.0 * ca2 * v3 * w
            - v2 * w2
            - 2.0 * ca2 * v2 * w2
            + 7.0 * v3 * w2
            + 2.0 * ca2 * v3 * w2
            - 3.0 * v3 * w3))
        / ((1.0 - v) * v * w * (1.0 - v + v * w).powi(2));

    let part2 = -(8.0
        * cf
        * l1v
        * (1.0 - v - v * w)
        * (ca2 - ca2 * v + ca2 * v2 - ca2 * v3 + v * w - ca2 * v2 * w - v3 * w
            + 3.0 * ca2 * v3 * w
            - v2 * w2
            + ca2 * v2 * w2
            + v3 * w2
            - 3.0 * ca2 * v3 * w2
            + ca2 * v3 * w3))
        / ((1.0 - v).powi(2) * v2 * w2 * (1.0 - v + v * w));

    let part3 = -(8.0
        * ca
        * cf.powi(2)
        * lms
        * (1.0 + v2 - 2.0 * v2 * w + v2 * w2)
        * (1.0 - 4.0 * v + 6.0 * v2 - 4.0 * v3 + v4 + v * w - 3.0 * v2 * w + 3.0 * v3 * w
            - v4 * w
            + v2 * w2
            - 2.0 * v3 * w2
            + v4 * w2
            + v3 * w3
            - v4 * w3
            + v4 * w4))
        / ((1.0 - v).powi(2) * v2 * w2 * (1.0 - v + v * w).powi(2));

    let part4 = -(4.0
        * cf
        * lw
        * (2.0 - 4.0 * ca2 - 6.0 * v + 6.0 * ca2 * v + 8.0 * v2 - 8.0 * ca2 * v2 - 8.0 * v3
            + 8.0 * ca2 * v3
            + 6.0 * v4
            - 4.0 * ca2 * v4
            - 2.0 * v5
            + 2.0 * ca2 * v5
            + 2.0 * v * w
            + 2.0 * ca2 * v * w
            - 6.0 * v2 * w
            + 2.0 * ca2 * v2 * w
            + 10.0 * v3 * w
            - 14.0 * ca2 * v3 * w
            - 10.0 * v4 * w
            + 6.0 * ca2 * v4 * w
            + 4.0 * v5 * w
            - 4.0 * ca2 * v5 * w
            + 5.0 * v2 * w2
            - 3.0 * ca2 * v2 * w2
            - 5.0 * v3 * w2
            + 13.0 * ca2 * v3 * w2
            + 3.0 * v4 * w2
            - ca2 * v4 * w2
            - 3.0 * v5 * w2
            + 3.0 * ca2 * v5 * w2
            - 3.0 * v3 * w3
            - 3.0 * ca2 * v3 * w3
            + 2.0 * v4 * w3
            - 4.0 * ca2 * v4 * w3
            + v5 * w3
            - ca2 * v5 * w3
            + 2.0 * ca2 * v4 * w4))
        / ((1.0 - v).powi(2) * v2 * w2 * (1.0 - v + v * w));

    let part5 = (4.0
        * cf
        * lvw
        * (1.0 - w)
        * (4.0 * ca2 + 4.0 * ca2 * v2 - 2.0 * v * w - 8.0 * ca2 * v * w - 6.0 * ca2 * v2 * w
            + 2.0 * v3 * w
            - 2.0 * ca2 * v3 * w
            - v2 * w2
            + 11.0 * ca2 * v2 * w2
            - 2.0 * v3 * w2
            + 4.0 * ca2 * v3 * w2
            - v4 * w2
            + ca2 * v4 * w2
            + 2.0 * v3 * w3
            - 6.0 * ca2 * v3 * w3
            + 2.0 * v4 * w3
            - 2.0 * ca2 * v4 * w3
            - 2.0 * v4 * w4
            + 2.0 * ca2 * v4 * w4))
        / ((1.0 - v).powi(2) * v2 * w2);

    let part6 = (2.0
        * cf
        * lms
        * (2.0 - 6.0 * ca2 - 4.0 * v + 4.0 * v2 - 8.0 * ca2 * v2 - 4.0 * v3 + 2.0 * v4
            - 2.0 * ca2 * v4
            - 2.0 * w
            + 2.0 * ca2 * w
            + 20.0 * ca2 * v * w
            + 4.0 * v2 * w
            + 12.0 * ca2 * v2 * w
            - 4.0 * v3 * w
            + 16.0 * ca2 * v3 * w
            + 6.0 * v4 * w
            + 2.0 * ca2 * v4 * w
            - 4.0 * v5 * w
            + 4.0 * ca2 * v5 * w
            + w2
            - ca2 * w2
            + 2.0 * v * w2
            - 4.0 * ca2 * v * w2
            - 2.0 * v2 * w2
            - 36.0 * ca2 * v2 * w2
            + 2.0 * v3 * w2
            - 24.0 * ca2 * v3 * w2
            - v4 * w2
            - 13.0 * ca2 * v4 * w2
            - 4.0 * ca2 * v5 * w2
            + 2.0 * v6 * w2
            - 2.0 * ca2 * v6 * w2
            - 2.0 * v * w3
            + 2.0 * ca2 * v * w3
            + 2.0 * v2 * w3
            + 2.0 * ca2 * v2 * w3
            - 6.0 * v3 * w3
            + 40.0 * ca2 * v3 * w3
            - 4.0 * v4 * w3
            + 20.0 * ca2 * v4 * w3
            + 6.0 * ca2 * v5 * w3
            - 2.0 * v6 * w3
            + 2.0 * ca2 * v6 * w3
            + v2 * w4
            - ca2 * v2 * w4
            - 2.0 * v3 * w4
            + 11.0 * v4 * w4
            - 27.0 * ca2 * v4 * w4
            + 2.0 * v5 * w4
            - 8.0 * ca2 * v5 * w4
            + 2.0 * v6 * w4
            - 2.0 * ca2 * v6 * w4
            - 6.0 * v5 * w5
            + 10.0 * ca2 * v5 * w5
            - 2.0 * v6 * w5
            + 2.0 * ca2 * v6 * w5
            + 2.0 * v6 * w6
            - 2.0 * ca2 * v6 * w6))
        / ((1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w).powi(2));

    let part7 = -(2.0
        * cf
        * l1w
        * (4.0 - 12.0 * ca2 - 16.0 * v + 32.0 * ca2 * v + 28.0 * v2 - 44.0 * ca2 * v2 - 32.0 * v3
            + 48.0 * ca2 * v3
            + 28.0 * v4
            - 36.0 * ca2 * v4
            - 16.0 * v5
            + 16.0 * ca2 * v5
            + 4.0 * v6
            - 4.0 * ca2 * v6
            - 2.0 * w
            + 2.0 * ca2 * w
            - 6.0 * v * w
            + 14.0 * ca2 * v * w
            + 20.0 * v2 * w
            - 40.0 * ca2 * v2 * w
            - 32.0 * v3 * w
            + 12.0 * ca2 * v3 * w
            + 42.0 * v4 * w
            - 6.0 * ca2 * v4 * w
            - 34.0 * v5 * w
            + 30.0 * ca2 * v5 * w
            + 20.0 * v6 * w
            - 20.0 * ca2 * v6 * w
            - 8.0 * v7 * w
            + 8.0 * ca2 * v7 * w
            + w2
            - ca2 * w2
            - 4.0 * v * w2
            + 2.0 * ca2 * v * w2
            + 23.0 * v2 * w2
            - 9.0 * ca2 * v2 * w2
            - 12.0 * v3 * w2
            + 56.0 * ca2 * v3 * w2
            - 37.0 * v4 * w2
            - 7.0 * ca2 * v4 * w2
            + 16.0 * v5 * w2
            - 74.0 * ca2 * v5 * w2
            + v6 * w2
            + 29.0 * ca2 * v6 * w2
            + 8.0 * v7 * w2
            - 8.0 * ca2 * v7 * w2
            + 4.0 * v8 * w2
            - 4.0 * ca2 * v8 * w2
            + 6.0 * v2 * w3
            - 6.0 * ca2 * v2 * w3
            - 38.0 * v3 * w3
            - 4.0 * ca2 * v3 * w3
            + 64.0 * v4 * w3
            - 48.0 * ca2 * v4 * w3
            + 44.0 * v5 * w3
            + 92.0 * ca2 * v5 * w3
            - 54.0 * v6 * w3
            + 10.0 * ca2 * v6 * w3
            - 10.0 * v7 * w3
            - 8.0 * ca2 * v7 * w3
            - 12.0 * v8 * w3
            + 12.0 * ca2 * v8 * w3
            - 2.0 * v2 * w4
            + 2.0 * ca2 * v2 * w4
            + 6.0 * v3 * w4
            - 2.0 * ca2 * v3 * w4
            - 2.0 * v4 * w4
            + 20.0 * ca2 * v4 * w4
            - 96.0 * v5 * w4
            - 26.0 * ca2 * v5 * w4
            + 48.0 * v6 * w4
            - 54.0 * ca2 * v6 * w4
            + 38.0 * v7 * w4
            + 12.0 * ca2 * v7 * w4
            + 16.0 * v8 * w4
            - 16.0 * ca2 * v8 * w4
            - 4.0 * v4 * w5
            + 4.0 * ca2 * v4 * w5
            + 40.0 * v5 * w5
            - 6.0 * ca2 * v5 * w5
            + 10.0 * v6 * w5
            + 38.0 * ca2 * v6 * w5
            - 54.0 * v7 * w5
            + 4.0 * ca2 * v7 * w5
            - 16.0 * v8 * w5
            + 16.0 * ca2 * v8 * w5
            + v4 * w6
            - ca2 * v4 * w6
            - 2.0 * v5 * w6
            - 17.0 * v6 * w6
            - 7.0 * ca2 * v6 * w6
            + 30.0 * v7 * w6
            - 12.0 * ca2 * v7 * w6
            + 16.0 * v8 * w6
            - 16.0 * ca2 * v8 * w6
            - 4.0 * v7 * w7
            + 4.0 * ca2 * v7 * w7
            - 12.0 * v8 * w7
            + 12.0 * ca2 * v8 * w7
            + 4.0 * v8 * w8
            - 4.0 * ca2 * v8 * w8))
        / ((1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2));

    let part8 = -(2.0
        * cf
        * lv
        * (4.0 - 16.0 * ca2 - 16.0 * v + 44.0 * ca2 * v + 28.0 * v2 - 60.0 * ca2 * v2 - 32.0 * v3
            + 64.0 * ca2 * v3
            + 28.0 * v4
            - 48.0 * ca2 * v4
            - 16.0 * v5
            + 20.0 * ca2 * v5
            + 4.0 * v6
            - 4.0 * ca2 * v6
            - 2.0 * w
            + 2.0 * ca2 * w
            - 10.0 * v * w
            + 22.0 * ca2 * v * w
            + 28.0 * v2 * w
            - 60.0 * ca2 * v2 * w
            - 32.0 * v3 * w
            + 24.0 * ca2 * v3 * w
            + 34.0 * v4 * w
            - 10.0 * ca2 * v4 * w
            - 30.0 * v5 * w
            + 42.0 * ca2 * v5 * w
            + 20.0 * v6 * w
            - 28.0 * ca2 * v6 * w
            - 8.0 * v7 * w
            + 8.0 * ca2 * v7 * w
            + w2
            - ca2 * w2
            - 4.0 * v * w2
            + 2.0 * ca2 * v * w2
            + 35.0 * v2 * w2
            - 13.0 * ca2 * v2 * w2
            - 40.0 * v3 * w2
            + 76.0 * ca2 * v3 * w2
            - 25.0 * v4 * w2
            - 7.0 * ca2 * v4 * w2
            + 28.0 * v5 * w2
            - 106.0 * ca2 * v5 * w2
            - 7.0 * v6 * w2
            + 41.0 * ca2 * v6 * w2
            + 8.0 * v7 * w2
            - 4.0 * ca2 * v7 * w2
            + 4.0 * v8 * w2
            - 4.0 * ca2 * v8 * w2
            + 6.0 * v2 * w3
            - 6.0 * ca2 * v2 * w3
            - 46.0 * v3 * w3
            - 8.0 * ca2 * v3 * w3
            + 96.0 * v4 * w3
            - 72.0 * ca2 * v4 * w3
            + 16.0 * v5 * w3
            + 128.0 * ca2 * v5 * w3
            - 54.0 * v6 * w3
            + 22.0 * ca2 * v6 * w3
            - 6.0 * v7 * w3
            - 20.0 * ca2 * v7 * w3
            - 12.0 * v8 * w3
            + 12.0 * ca2 * v8 * w3
            - 2.0 * v2 * w4
            + 2.0 * ca2 * v2 * w4
            + 6.0 * v3 * w4
            - 2.0 * ca2 * v3 * w4
            - 10.0 * v4 * w4
            + 32.0 * ca2 * v4 * w4
            - 104.0 * v5 * w4
            - 30.0 * ca2 * v5 * w4
            + 68.0 * v6 * w4
            - 94.0 * ca2 * v6 * w4
            + 34.0 * v7 * w4
            + 20.0 * ca2 * v7 * w4
            + 16.0 * v8 * w4
            - 16.0 * ca2 * v8 * w4
            - 4.0 * v4 * w5
            + 4.0 * ca2 * v4 * w5
            + 52.0 * v5 * w5
            - 14.0 * ca2 * v5 * w5
            + 2.0 * v6 * w5
            + 66.0 * ca2 * v6 * w5
            - 58.0 * v7 * w5
            + 12.0 * ca2 * v7 * w5
            - 16.0 * v8 * w5
            + 16.0 * ca2 * v8 * w5
            + v4 * w6
            - ca2 * v4 * w6
            - 2.0 * v5 * w6
            - 21.0 * v6 * w6
            - 11.0 * ca2 * v6 * w6
            + 34.0 * v7 * w6
            - 24.0 * ca2 * v7 * w6
            + 16.0 * v8 * w6
            - 16.0 * ca2 * v8 * w6
            - 4.0 * v7 * w7
            + 8.0 * ca2 * v7 * w7
            - 12.0 * v8 * w7
            + 12.0 * ca2 * v8 * w7
            + 4.0 * v8 * w8
            - 4.0 * ca2 * v8 * w8))
        / ((1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2));

    let part9 = -(2.0
        * cf
        * (2.0 - 6.0 * ca2 - 12.0 * v + 28.0 * ca2 * v + 30.0 * v2 - 54.0 * ca2 * v2 - 40.0 * v3
            + 56.0 * ca2 * v3
            + 30.0 * v4
            - 34.0 * ca2 * v4
            - 12.0 * v5
            + 12.0 * ca2 * v5
            + 2.0 * v6
            - 2.0 * ca2 * v6
            + w
            - ca2 * w
            - v * w
            + ca2 * v * w
            - 2.0 * v2 * w
            - 6.0 * ca2 * v2 * w
            - 4.0 * v3 * w
            + 28.0 * ca2 * v3 * w
            + 23.0 * v4 * w
            - 47.0 * ca2 * v4 * w
            - 31.0 * v5 * w
            + 39.0 * ca2 * v5 * w
            + 18.0 * v6 * w
            - 18.0 * ca2 * v6 * w
            - 4.0 * v7 * w
            + 4.0 * ca2 * v7 * w
            - w2
            + ca2 * w2
            + v * w2
            - 5.0 * ca2 * v * w2
            + 8.0 * v2 * w2
            + 12.0 * ca2 * v2 * w2
            + 2.0 * v3 * w2
            - 30.0 * ca2 * v3 * w2
            - 43.0 * v4 * w2
            + 47.0 * ca2 * v4 * w2
            + 53.0 * v5 * w2
            - 37.0 * ca2 * v5 * w2
            - 22.0 * v6 * w2
            + 14.0 * ca2 * v6 * w2
            + 2.0 * v8 * w2
            - 2.0 * ca2 * v8 * w2
            - 4.0 * v2 * w3
            + 4.0 * ca2 * v2 * w3
            - 14.0 * v3 * w3
            - 10.0 * ca2 * v3 * w3
            + 48.0 * v4 * w3
            - 36.0 * v5 * w3
            + 8.0 * ca2 * v5 * w3
            + 12.0 * v7 * w3
            - 8.0 * ca2 * v7 * w3
            - 6.0 * v8 * w3
            + 6.0 * ca2 * v8 * w3
            + 2.0 * v2 * w4
            - 2.0 * ca2 * v2 * w4
            + 8.0 * ca2 * v3 * w4
            - 4.0 * v4 * w4
            - 2.0 * v5 * w4
            - 14.0 * ca2 * v5 * w4
            + 12.0 * v6 * w4
            + 4.0 * ca2 * v6 * w4
            - 18.0 * v7 * w4
            + 10.0 * ca2 * v7 * w4
            + 10.0 * v8 * w4
            - 10.0 * ca2 * v8 * w4
            + 3.0 * v4 * w5
            - 3.0 * ca2 * v4 * w5
            - 3.0 * v5 * w5
            + 11.0 * ca2 * v5 * w5
            - 6.0 * v6 * w5
            - 2.0 * ca2 * v6 * w5
            + 18.0 * v7 * w5
            - 10.0 * ca2 * v7 * w5
            - 12.0 * v8 * w5
            + 12.0 * ca2 * v8 * w5
            - v4 * w6
            + ca2 * v4 * w6
            - v5 * w6
            - 3.0 * ca2 * v5 * w6
            + 4.0 * v6 * w6
            - 10.0 * v7 * w6
            + 6.0 * ca2 * v7 * w6
            + 10.0 * v8 * w6
            - 10.0 * ca2 * v8 * w6
            + 2.0 * v7 * w7
            - 2.0 * ca2 * v7 * w7
            - 6.0 * v8 * w7
            + 6.0 * ca2 * v8 * w7
            + 2.0 * v8 * w8
            - 2.0 * ca2 * v8 * w8))
        / ((1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2));

    part1 + part2 + part3 + part4 + part5 + part6 + part7 + part8 + part9
}

/// `STRUV5(W,V,X3,S)`.
#[must_use]
pub fn struv5(w: f64, v: f64, _x3: f64, _s: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let cf = ctx.cf;
    let ca = ctx.ca;
    let (v2, v3, v4, v5, v6, v7) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7);
    let (w2, w3, w4, w5, w6, w7, w8) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8);
    let ca2 = pre.ca2;
    let cacf = ca * cf;
    let (l1v, lw, lvw, l1vw, lms, l1w, lv, lmss) = (
        pre.l1v, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.l1w, pre.lv, pre.lmss,
    );

    let part1 = (-4.0
        * cf
        * l1v
        * (10.0 - 2.0 * ca2 - 20.0 * v + 5.0 * ca2 * v + 14.0 * v2 - 4.0 * ca2 * v2 - 8.0 * v * w
            + ca2 * v * w
            + 2.0 * v2 * w2))
        / (1.0 - v + v * w);

    let part2 = -(4.0
        * cf
        * lvw
        * (4.0 - 3.0 * ca2 - 5.0 * v + 4.0 * ca2 * v + 2.0 * v2 - 2.0 * ca2 * v2 - 7.0 * v * w
            + 5.0 * ca2 * v * w
            + 6.0 * v2 * w
            - 4.0 * ca2 * v2 * w
            - 2.0 * v3 * w
            + 2.0 * ca2 * v3 * w
            + 4.0 * v2 * w2
            - 4.0 * ca2 * v2 * w2
            - 2.0 * v3 * w3
            + 2.0 * ca2 * v3 * w3))
        / ((1.0 - v) * v);

    let part3 = -(4.0
        * ca
        * cf.powi(2)
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

    let part4 = (4.0
        * cf
        * l1vw
        * (4.0 - 5.0 * ca2 + 4.0 * cacf - 14.0 * v + 18.0 * ca2 * v - 12.0 * cacf * v + 20.0 * v2
            - 28.0 * ca2 * v2
            + 14.0 * cacf * v2
            - 14.0 * v3
            + 22.0 * ca2 * v3
            - 8.0 * cacf * v3
            + 4.0 * v4
            - 7.0 * ca2 * v4
            + 2.0 * cacf * v4
            + 6.0 * v * w
            - 8.0 * ca2 * v * w
            + 4.0 * cacf * v * w
            - 16.0 * v2 * w
            + 22.0 * ca2 * v2 * w
            - 12.0 * cacf * v2 * w
            + 18.0 * v3 * w
            - 24.0 * ca2 * v3 * w
            + 12.0 * cacf * v3 * w
            - 8.0 * v4 * w
            + 10.0 * ca2 * v4 * w
            - 4.0 * cacf * v4 * w
            - 4.0 * v2 * w2
            - 2.0 * ca2 * v2 * w2
            + 6.0 * cacf * v2 * w2
            + 6.0 * v3 * w2
            + 2.0 * ca2 * v3 * w2
            - 8.0 * cacf * v3 * w2
            - 4.0 * ca2 * v4 * w2
            + 4.0 * cacf * v4 * w2
            - 10.0 * v3 * w3
            + 4.0 * cacf * v3 * w3
            + 8.0 * v4 * w3
            + 6.0 * ca2 * v4 * w3
            - 4.0 * cacf * v4 * w3
            - 4.0 * v4 * w4
            - 5.0 * ca2 * v4 * w4
            + 2.0 * cacf * v4 * w4))
        / (v * (1.0 - v + v * w).powi(3));

    let part5 = -(2.0
        * cf
        * lmss
        * (2.0 * cacf + 2.0 * v - 2.0 * ca2 * v - 8.0 * cacf * v - 9.0 * v2
            + 9.0 * ca2 * v2
            + 14.0 * cacf * v2
            + 18.0 * v3
            - 18.0 * ca2 * v3
            - 12.0 * cacf * v3
            - 19.0 * v4
            + 19.0 * ca2 * v4
            + 4.0 * cacf * v4
            + 10.0 * v5
            - 10.0 * ca2 * v5
            - 2.0 * v6
            + 2.0 * ca2 * v6
            - 2.0 * v * w
            + 12.0 * cacf * v * w
            + 14.0 * v2 * w
            - 4.0 * ca2 * v2 * w
            - 54.0 * cacf * v2 * w
            - 40.0 * v3 * w
            + 18.0 * ca2 * v3 * w
            + 94.0 * cacf * v3 * w
            + 54.0 * v4 * w
            - 28.0 * ca2 * v4 * w
            - 82.0 * cacf * v4 * w
            - 34.0 * v5 * w
            + 18.0 * ca2 * v5 * w
            + 38.0 * cacf * v5 * w
            + 8.0 * v6 * w
            - 4.0 * ca2 * v6 * w
            - 8.0 * cacf * v6 * w
            - 5.0 * v2 * w2
            + ca2 * v2 * w2
            + 12.0 * cacf * v2 * w2
            + 30.0 * v3 * w2
            - 10.0 * ca2 * v3 * w2
            - 46.0 * cacf * v3 * w2
            - 58.0 * v4 * w2
            + 22.0 * ca2 * v4 * w2
            + 74.0 * cacf * v4 * w2
            + 48.0 * v5 * w2
            - 20.0 * ca2 * v5 * w2
            - 54.0 * cacf * v5 * w2
            - 14.0 * v6 * w2
            + 6.0 * ca2 * v6 * w2
            + 16.0 * cacf * v6 * w2
            - 8.0 * v3 * w3
            + 2.0 * ca2 * v3 * w3
            + 4.0 * cacf * v3 * w3
            + 30.0 * v4 * w3
            - 12.0 * ca2 * v4 * w3
            - 14.0 * cacf * v4 * w3
            - 40.0 * v5 * w3
            + 20.0 * ca2 * v5 * w3
            + 14.0 * cacf * v5 * w3
            + 16.0 * v6 * w3
            - 8.0 * ca2 * v6 * w3
            - 8.0 * cacf * v6 * w3
            - 7.0 * v4 * w4
            + 3.0 * ca2 * v4 * w4
            + 2.0 * cacf * v4 * w4
            + 22.0 * v5 * w4
            - 10.0 * ca2 * v5 * w4
            + 2.0 * cacf * v5 * w4
            - 14.0 * v6 * w4
            + 6.0 * ca2 * v6 * w4
            - 6.0 * v5 * w5
            + 2.0 * ca2 * v5 * w5
            + 8.0 * v6 * w5
            - 4.0 * ca2 * v6 * w5
            - 2.0 * v6 * w6
            + 2.0 * ca2 * v6 * w6))
        / ((1.0 - v) * v * w * (1.0 - v + v * w).powi(3));

    let part6 = -(cf
        * (4.0 - 4.0 * ca2 + 4.0 * cacf - 16.0 * v + 16.0 * ca2 * v - 16.0 * cacf * v + 24.0 * v2
            - 24.0 * ca2 * v2
            + 24.0 * cacf * v2
            - 16.0 * v3
            + 16.0 * ca2 * v3
            - 16.0 * cacf * v3
            + 4.0 * v4
            - 4.0 * ca2 * v4
            + 4.0 * cacf * v4
            - 12.0 * w
            + 4.0 * ca2 * w
            - 4.0 * cacf * w
            + 39.0 * v * w
            - 15.0 * ca2 * v * w
            + 24.0 * cacf * v * w
            - 28.0 * v2 * w
            + 4.0 * ca2 * v2 * w
            - 36.0 * cacf * v2 * w
            - 36.0 * v3 * w
            + 36.0 * ca2 * v3 * w
            + 4.0 * cacf * v3 * w
            + 66.0 * v4 * w
            - 42.0 * ca2 * v4 * w
            + 24.0 * cacf * v4 * w
            - 35.0 * v5 * w
            + 11.0 * ca2 * v5 * w
            - 12.0 * cacf * v5 * w
            + 6.0 * v6 * w
            + 2.0 * ca2 * v6 * w
            - 24.0 * v * w2
            + 8.0 * ca2 * v * w2
            - 8.0 * cacf * v * w2
            + 102.0 * v2 * w2
            - 26.0 * ca2 * v2 * w2
            + 12.0 * cacf * v2 * w2
            - 131.0 * v3 * w2
            + 27.0 * ca2 * v3 * w2
            + 40.0 * cacf * v3 * w2
            + 63.0 * v4 * w2
            - 23.0 * ca2 * v4 * w2
            - 80.0 * cacf * v4 * w2
            - 27.0 * v5 * w2
            + 27.0 * ca2 * v5 * w2
            + 32.0 * cacf * v5 * w2
            + 23.0 * v6 * w2
            - 11.0 * ca2 * v6 * w2
            + 4.0 * cacf * v6 * w2
            - 6.0 * v7 * w2
            - 2.0 * ca2 * v7 * w2
            - 20.0 * v3 * w3
            - 36.0 * cacf * v3 * w3
            + 53.0 * v4 * w3
            - 9.0 * ca2 * v4 * w3
            + 80.0 * cacf * v4 * w3
            - 4.0 * ca2 * v5 * w3
            - 32.0 * cacf * v5 * w3
            - 45.0 * v6 * w3
            + 9.0 * ca2 * v6 * w3
            - 12.0 * cacf * v6 * w3
            + 12.0 * v7 * w3
            + 4.0 * ca2 * v7 * w3
            + 24.0 * v3 * w4
            - 8.0 * ca2 * v3 * w4
            + 8.0 * cacf * v3 * w4
            - 82.0 * v4 * w4
            + 38.0 * ca2 * v4 * w4
            - 32.0 * cacf * v4 * w4
            + 47.0 * v5 * w4
            - 23.0 * ca2 * v5 * w4
            + 16.0 * cacf * v5 * w4
            + 21.0 * v6 * w4
            - ca2 * v6 * w4
            + 12.0 * cacf * v6 * w4
            - 6.0 * v7 * w4
            - 10.0 * ca2 * v7 * w4
            + 12.0 * v4 * w5
            - 4.0 * ca2 * v4 * w5
            + 4.0 * cacf * v4 * w5
            - 11.0 * v5 * w5
            - ca2 * v5 * w5
            - 4.0 * cacf * v5 * w5
            - 5.0 * v6 * w5
            - 7.0 * ca2 * v6 * w5
            - 4.0 * cacf * v6 * w5
            + 16.0 * ca2 * v7 * w5
            + 8.0 * ca2 * v6 * w6
            - 8.0 * ca2 * v7 * w6))
        / ((1.0 - v) * v * w * (1.0 - v * w) * (1.0 - v + v * w).powi(3));

    let part7 = -(2.0
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
        / ((1.0 - v) * v * w * (1.0 - v * w) * (1.0 - v + v * w).powi(3));

    let part8 = -(2.0
        * cf
        * lv
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
        / ((1.0 - v) * v * w * (1.0 - v * w) * (1.0 - v + v * w).powi(3));

    let part9 = -(2.0
        * cf
        * lw
        * (3.0 - 3.0 * ca2 + 2.0 * cacf - 18.0 * v + 18.0 * ca2 * v - 12.0 * cacf * v + 48.0 * v2
            - 48.0 * ca2 * v2
            + 32.0 * cacf * v2
            - 72.0 * v3
            + 72.0 * ca2 * v3
            - 48.0 * cacf * v3
            + 63.0 * v4
            - 63.0 * ca2 * v4
            + 42.0 * cacf * v4
            - 30.0 * v5
            + 30.0 * ca2 * v5
            - 20.0 * cacf * v5
            + 6.0 * v6
            - 6.0 * ca2 * v6
            + 4.0 * cacf * v6
            - 10.0 * w
            - 2.0 * ca2 * w
            + 8.0 * cacf * w
            + 54.0 * v * w
            - 28.0 * cacf * v * w
            - 129.0 * v2 * w
            + 23.0 * ca2 * v2 * w
            + 34.0 * cacf * v2 * w
            + 173.0 * v3 * w
            - 55.0 * ca2 * v3 * w
            - 10.0 * cacf * v3 * w
            - 129.0 * v4 * w
            + 51.0 * ca2 * v4 * w
            - 10.0 * cacf * v4 * w
            + 39.0 * v5 * w
            - 11.0 * ca2 * v5 * w
            + 2.0 * cacf * v5 * w
            + 8.0 * v6 * w
            - 12.0 * ca2 * v6 * w
            + 8.0 * cacf * v6 * w
            - 6.0 * v7 * w
            + 6.0 * ca2 * v7 * w
            - 4.0 * cacf * v7 * w
            + w2
            - ca2 * w2
            + 2.0 * cacf * w2
            - 14.0 * v * w2
            + 8.0 * ca2 * v * w2
            - 12.0 * cacf * v * w2
            + 46.0 * v2 * w2
            - 30.0 * ca2 * v2 * w2
            + 40.0 * cacf * v2 * w2
            - 69.0 * v3 * w2
            + 49.0 * ca2 * v3 * w2
            - 66.0 * cacf * v3 * w2
            + 30.0 * v4 * w2
            - 22.0 * ca2 * v4 * w2
            + 48.0 * cacf * v4 * w2
            + 49.0 * v5 * w2
            - 35.0 * ca2 * v5 * w2
            + 2.0 * cacf * v5 * w2
            - 65.0 * v6 * w2
            + 49.0 * ca2 * v6 * w2
            - 26.0 * cacf * v6 * w2
            + 22.0 * v7 * w2
            - 18.0 * ca2 * v7 * w2
            + 12.0 * cacf * v7 * w2
            + 2.0 * v * w3
            - 2.0 * ca2 * v * w3
            + 4.0 * cacf * v * w3
            - 15.0 * v2 * w3
            + 13.0 * ca2 * v2 * w3
            - 22.0 * cacf * v2 * w3
            + 35.0 * v3 * w3
            - 27.0 * ca2 * v3 * w3
            + 42.0 * cacf * v3 * w3
            + 4.0 * v4 * w3
            + 4.0 * ca2 * v4 * w3
            - 20.0 * cacf * v4 * w3
            - 102.0 * v5 * w3
            + 52.0 * ca2 * v5 * w3
            - 28.0 * cacf * v5 * w3
            + 114.0 * v6 * w3
            - 66.0 * ca2 * v6 * w3
            + 44.0 * cacf * v6 * w3
            - 38.0 * v7 * w3
            + 26.0 * ca2 * v7 * w3
            - 20.0 * cacf * v7 * w3
            - 13.0 * v3 * w4
            - 5.0 * ca2 * v3 * w4
            + 14.0 * cacf * v3 * w4
            - 14.0 * v4 * w4
            + 34.0 * ca2 * v4 * w4
            - 60.0 * cacf * v4 * w4
            + 114.0 * v5 * w4
            - 72.0 * ca2 * v5 * w4
            + 84.0 * cacf * v5 * w4
            - 132.0 * v6 * w4
            + 72.0 * ca2 * v6 * w4
            - 64.0 * cacf * v6 * w4
            + 46.0 * v7 * w4
            - 30.0 * ca2 * v7 * w4
            + 28.0 * cacf * v7 * w4
            - 2.0 * v3 * w5
            + 2.0 * ca2 * v3 * w5
            - 4.0 * cacf * v3 * w5
            + 27.0 * v4 * w5
            - 17.0 * ca2 * v4 * w5
            + 26.0 * cacf * v4 * w5
            - 97.0 * v5 * w5
            + 39.0 * ca2 * v5 * w5
            - 38.0 * cacf * v5 * w5
            + 110.0 * v6 * w5
            - 46.0 * ca2 * v6 * w5
            + 36.0 * cacf * v6 * w5
            - 42.0 * v7 * w5
            + 26.0 * ca2 * v7 * w5
            - 28.0 * cacf * v7 * w5
            - v4 * w6
            + ca2 * v4 * w6
            - 2.0 * cacf * v4 * w6
            + 27.0 * v5 * w6
            - 3.0 * ca2 * v5 * w6
            - 2.0 * cacf * v5 * w6
            - 45.0 * v6 * w6
            + 9.0 * ca2 * v6 * w6
            - 2.0 * cacf * v6 * w6
            + 26.0 * v7 * w6
            - 14.0 * ca2 * v7 * w6
            + 20.0 * cacf * v7 * w6
            + 4.0 * v6 * w7
            - 10.0 * v7 * w7
            + 6.0 * ca2 * v7 * w7
            - 12.0 * cacf * v7 * w7
            + 2.0 * v7 * w8
            - 2.0 * ca2 * v7 * w8
            + 4.0 * cacf * v7 * w8))
        / ((1.0 - v) * v * (1.0 - w) * w * (1.0 - v * w) * (1.0 - v + v * w).powi(3));

    part1 + part2 + part3 + part4 + part5 + part6 + part7 + part8 + part9
}

/// `STRUV6(W,V,X3,S)`.
#[must_use]
pub fn struv6(w: f64, v: f64, _x3: f64, _s: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let cf = ctx.cf;
    let ca = ctx.ca;
    let (v2, v3, v4, v5, v6, v7, v8, v9) = (
        pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8, pre.v9,
    );
    let (w2, w3, w4, w5, w6, w7, w8, w9) = (
        pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8, pre.w9,
    );
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let cacf = ca * cf;
    let ca2cf = ca2 * cf;
    let (l1v, lw, lvw, l1vw, lms, l1w, lv, lmss) = (
        pre.l1v, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.l1w, pre.lv, pre.lmss,
    );

    let part1 = (4.0
        * cf
        * l1v
        * (4.0 * ca - 12.0 * ca * v + 14.0 * ca * v2 - 8.0 * ca * v3 + 2.0 * ca * v4
            - 12.0 * ca * w
            + 4.0 * ca3 * w
            + 32.0 * ca * v * w
            + ca2 * v * w
            - 14.0 * ca3 * v * w
            + 4.0 * v2 * w
            - 32.0 * ca * v2 * w
            + 19.0 * ca3 * v2 * w
            - 7.0 * v3 * w
            + 14.0 * ca * v3 * w
            - 2.0 * ca2 * v3 * w
            - 12.0 * ca3 * v3 * w
            + 4.0 * v4 * w
            + 2.0 * ca2 * v4 * w
            + 3.0 * ca3 * v4 * w
            - v5 * w
            - 2.0 * ca * v5 * w
            - ca2 * v5 * w
            + 12.0 * ca * v * w2
            - 4.0 * ca3 * v * w2
            - 5.0 * v2 * w2
            - 32.0 * ca * v2 * w2
            - 4.0 * ca2 * v2 * w2
            + 13.0 * ca3 * v2 * w2
            + 8.0 * v3 * w2
            + 34.0 * ca * v3 * w2
            + 5.0 * ca2 * v3 * w2
            - 13.0 * ca3 * v3 * w2
            - 5.0 * v4 * w2
            - 20.0 * ca * v4 * w2
            - 3.0 * ca2 * v4 * w2
            + 10.0 * ca3 * v4 * w2
            + 3.0 * v5 * w2
            + 8.0 * ca * v5 * w2
            + ca2 * v5 * w2
            - 2.0 * ca3 * v5 * w2
            - v6 * w2
            + ca2 * v6 * w2
            + v3 * w3
            - 4.0 * ca * v3 * w3
            - ca2 * v3 * w3
            + ca3 * v3 * w3
            + 4.0 * ca * v4 * w3
            + 2.0 * ca2 * v4 * w3
            + ca3 * v4 * w3
            - 3.0 * v5 * w3
            + ca2 * v5 * w3
            - 2.0 * ca3 * v5 * w3
            + 2.0 * v6 * w3
            - 2.0 * ca2 * v6 * w3
            + v5 * w4
            - 2.0 * ca * v5 * w4
            - ca2 * v5 * w4
            - v6 * w4
            + ca2 * v6 * w4))
        / (ca * (1.0 - v).powi(2) * v3 * w2 * (1.0 - v * w));

    let part2 = -(4.0
        * cf
        * lvw
        * (4.0 * ca - 4.0 * ca3 - 8.0 * ca * v + 8.0 * ca3 * v + 6.0 * ca * v2
            - 6.0 * ca3 * v2
            - 2.0 * ca * v3
            + 2.0 * ca3 * v3
            - v * w
            - 8.0 * ca * v * w
            + 6.0 * ca2 * v * w
            + 6.0 * ca3 * v * w
            + 3.0 * v2 * w
            + 15.0 * ca * v2 * w
            - 11.0 * ca2 * v2 * w
            - 11.0 * ca3 * v2 * w
            - 3.0 * v3 * w
            - 8.0 * ca * v3 * w
            + 8.0 * ca2 * v3 * w
            + 6.0 * ca3 * v3 * w
            + v4 * w
            + 3.0 * ca * v4 * w
            - 3.0 * ca2 * v4 * w
            - 3.0 * ca3 * v4 * w
            - ca * v2 * w2
            + 5.0 * ca2 * v2 * w2
            - 5.0 * ca3 * v2 * w2
            - 4.0 * ca * v3 * w2
            - 11.0 * ca2 * v3 * w2
            + 7.0 * ca3 * v3 * w2
            + 2.0 * ca * v4 * w2
            + 6.0 * ca2 * v4 * w2
            - 3.0 * ca3 * v4 * w2
            - ca * v5 * w2
            + ca3 * v5 * w2
            - v3 * w3
            - 2.0 * ca * v3 * w3
            + 3.0 * ca2 * v3 * w3
            - 3.0 * ca3 * v3 * w3
            + v4 * w3
            + 3.0 * ca * v4 * w3
            - 3.0 * ca2 * v4 * w3
            + 2.0 * ca3 * v4 * w3
            + 3.0 * ca * v5 * w3
            - 3.0 * ca3 * v5 * w3
            - 4.0 * ca3 * v4 * w4
            - 4.0 * ca * v5 * w4
            + 4.0 * ca3 * v5 * w4
            + 2.0 * ca * v5 * w5
            - 2.0 * ca3 * v5 * w5))
        / (ca * (1.0 - v).powi(2) * v3 * w2);

    let part3 = -(8.0
        * cf.powi(2)
        * lmss
        * (2.0 * ca - 10.0 * ca * v + 21.0 * ca * v2 - 24.0 * ca * v3 + 16.0 * ca * v4
            - 6.0 * ca * v5
            + ca * v6
            + 2.0 * ca * w
            - 2.0 * v * w
            - 6.0 * ca * v * w
            + 8.0 * v2 * w
            + 3.0 * ca * v2 * w
            - 13.0 * v3 * w
            + 9.0 * ca * v3 * w
            + 11.0 * v4 * w
            - 15.0 * ca * v4 * w
            - 5.0 * v5 * w
            + 9.0 * ca * v5 * w
            + v6 * w
            - 2.0 * ca * v6 * w
            + 2.0 * ca * v * w2
            - 2.0 * v2 * w2
            - 6.0 * ca * v2 * w2
            + 9.0 * v3 * w2
            + 3.0 * ca * v3 * w2
            - 15.0 * v4 * w2
            + 4.0 * ca * v4 * w2
            + 11.0 * v5 * w2
            - 5.0 * ca * v5 * w2
            - 3.0 * v6 * w2
            + 2.0 * ca * v6 * w2
            + 4.0 * v4 * w3
            - 5.0 * ca * v4 * w3
            - 7.0 * v5 * w3
            + 5.0 * ca * v5 * w3
            + 3.0 * v6 * w3
            - 2.0 * ca * v6 * w3
            + v5 * w4
            - 3.0 * ca * v5 * w4
            - v6 * w4
            + 2.0 * ca * v6 * w4
            - ca * v6 * w5))
        / ((1.0 - v).powi(2) * v3 * w2 * (1.0 - v + v * w));

    let part4 = (4.0
        * cf
        * l1vw
        * (4.0 * ca - 4.0 * ca3 + 8.0 * ca2cf - 20.0 * ca * v + 20.0 * ca3 * v - 40.0 * ca2cf * v
            + 42.0 * ca * v2
            - 42.0 * ca3 * v2
            + 84.0 * ca2cf * v2
            - 48.0 * ca * v3
            + 48.0 * ca3 * v3
            - 96.0 * ca2cf * v3
            + 32.0 * ca * v4
            - 32.0 * ca3 * v4
            + 64.0 * ca2cf * v4
            - 12.0 * ca * v5
            + 12.0 * ca3 * v5
            - 24.0 * ca2cf * v5
            + 2.0 * ca * v6
            - 2.0 * ca3 * v6
            + 4.0 * ca2cf * v6
            - 4.0 * v * w
            + 4.0 * ca * v * w
            + 4.0 * ca2 * v * w
            - 8.0 * ca3 * v * w
            - 8.0 * cacf * v * w
            + 16.0 * ca2cf * v * w
            + 16.0 * v2 * w
            - 22.0 * ca * v2 * w
            - 16.0 * ca2 * v2 * w
            + 35.0 * ca3 * v2 * w
            + 32.0 * cacf * v2 * w
            - 72.0 * ca2cf * v2 * w
            - 26.0 * v3 * w
            + 48.0 * ca * v3 * w
            + 26.0 * ca2 * v3 * w
            - 63.0 * ca3 * v3 * w
            - 52.0 * cacf * v3 * w
            + 132.0 * ca2cf * v3 * w
            + 22.0 * v4 * w
            - 52.0 * ca * v4 * w
            - 22.0 * ca2 * v4 * w
            + 59.0 * ca3 * v4 * w
            + 44.0 * cacf * v4 * w
            - 124.0 * ca2cf * v4 * w
            - 10.0 * v5 * w
            + 28.0 * ca * v5 * w
            + 10.0 * ca2 * v5 * w
            - 29.0 * ca3 * v5 * w
            - 20.0 * cacf * v5 * w
            + 60.0 * ca2cf * v5 * w
            + 2.0 * v6 * w
            - 6.0 * ca * v6 * w
            - 2.0 * ca2 * v6 * w
            + 6.0 * ca3 * v6 * w
            + 4.0 * cacf * v6 * w
            - 12.0 * ca2cf * v6 * w
            - 12.0 * v2 * w2
            + 4.0 * ca * v2 * w2
            + 12.0 * ca2 * v2 * w2
            - 9.0 * ca3 * v2 * w2
            - 24.0 * cacf * v2 * w2
            + 20.0 * ca2cf * v2 * w2
            + 40.0 * v3 * w2
            - 24.0 * ca * v3 * w2
            - 40.0 * ca2 * v3 * w2
            + 33.0 * ca3 * v3 * w2
            + 80.0 * cacf * v3 * w2
            - 72.0 * ca2cf * v3 * w2
            - 52.0 * v4 * w2
            + 44.0 * ca * v4 * w2
            + 52.0 * ca2 * v4 * w2
            - 47.0 * ca3 * v4 * w2
            - 104.0 * cacf * v4 * w2
            + 100.0 * ca2cf * v4 * w2
            + 32.0 * v5 * w2
            - 32.0 * ca * v5 * w2
            - 32.0 * ca2 * v5 * w2
            + 31.0 * ca3 * v5 * w2
            + 64.0 * cacf * v5 * w2
            - 64.0 * ca2cf * v5 * w2
            - 8.0 * v6 * w2
            + 8.0 * ca * v6 * w2
            + 8.0 * ca2 * v6 * w2
            - 8.0 * ca3 * v6 * w2
            - 16.0 * cacf * v6 * w2
            + 16.0 * ca2cf * v6 * w2
            - 14.0 * v3 * w3
            + 8.0 * ca * v3 * w3
            + 14.0 * ca2 * v3 * w3
            - 10.0 * ca3 * v3 * w3
            - 28.0 * cacf * v3 * w3
            + 20.0 * ca2cf * v3 * w3
            + 38.0 * v4 * w3
            - 24.0 * ca * v4 * w3
            - 38.0 * ca2 * v4 * w3
            + 25.0 * ca3 * v4 * w3
            + 76.0 * cacf * v4 * w3
            - 52.0 * ca2cf * v4 * w3
            - 36.0 * v5 * w3
            + 24.0 * ca * v5 * w3
            + 36.0 * ca2 * v5 * w3
            - 23.0 * ca3 * v5 * w3
            - 72.0 * cacf * v5 * w3
            + 48.0 * ca2cf * v5 * w3
            + 12.0 * v6 * w3
            - 8.0 * ca * v6 * w3
            - 12.0 * ca2 * v6 * w3
            + 8.0 * ca3 * v6 * w3
            + 24.0 * cacf * v6 * w3
            - 16.0 * ca2cf * v6 * w3
            - 8.0 * v4 * w4
            + 8.0 * ca * v4 * w4
            + 8.0 * ca2 * v4 * w4
            - 9.0 * ca3 * v4 * w4
            - 16.0 * cacf * v4 * w4
            + 20.0 * ca2cf * v4 * w4
            + 16.0 * v5 * w4
            - 14.0 * ca * v5 * w4
            - 16.0 * ca2 * v5 * w4
            + 15.0 * ca3 * v5 * w4
            + 32.0 * cacf * v5 * w4
            - 32.0 * ca2cf * v5 * w4
            - 8.0 * v6 * w4
            + 8.0 * ca * v6 * w4
            + 8.0 * ca2 * v6 * w4
            - 8.0 * ca3 * v6 * w4
            - 16.0 * cacf * v6 * w4
            + 16.0 * ca2cf * v6 * w4
            - 2.0 * v5 * w5
            + 6.0 * ca * v5 * w5
            + 2.0 * ca2 * v5 * w5
            - 6.0 * ca3 * v5 * w5
            - 4.0 * cacf * v5 * w5
            + 12.0 * ca2cf * v5 * w5
            + 2.0 * v6 * w5
            - 6.0 * ca * v6 * w5
            - 2.0 * ca2 * v6 * w5
            + 6.0 * ca3 * v6 * w5
            + 4.0 * cacf * v6 * w5
            - 12.0 * ca2cf * v6 * w5
            + 2.0 * ca * v6 * w6
            - 2.0 * ca3 * v6 * w6
            + 4.0 * ca2cf * v6 * w6))
        / (ca * (1.0 - v).powi(2) * v3 * w2 * (1.0 - v + v * w));

    let part5 = -(2.0
        * cf
        * lms
        * (8.0 * ca2 + 8.0 * cacf - 24.0 * ca2 * v - 16.0 * cacf * v - 4.0 * v2
            + 32.0 * ca2 * v2
            + 12.0 * cacf * v2
            + 8.0 * v3
            - 24.0 * ca2 * v3
            - 4.0 * cacf * v3
            - 6.0 * v4
            + 10.0 * ca2 * v4
            + 2.0 * v5
            - 2.0 * ca2 * v5
            - 8.0 * ca2 * w
            + 12.0 * cacf * w
            - 12.0 * cf * v * w
            - 60.0 * cacf * v * w
            + 3.0 * v2 * w
            + 41.0 * ca2 * v2 * w
            + 20.0 * cf * v2 * w
            + 84.0 * cacf * v2 * w
            + 4.0 * v3 * w
            - 72.0 * ca2 * v3 * w
            - 12.0 * cf * v3 * w
            - 48.0 * cacf * v3 * w
            - 19.0 * v4 * w
            + 63.0 * ca2 * v4 * w
            + 4.0 * cf * v4 * w
            + 16.0 * cacf * v4 * w
            + 16.0 * v5 * w
            - 28.0 * ca2 * v5 * w
            - 6.0 * v6 * w
            + 6.0 * ca2 * v6 * w
            + 4.0 * ca2 * w2
            - 4.0 * cacf * w2
            + 12.0 * ca2 * v * w2
            + 4.0 * cf * v * w2
            - 24.0 * cacf * v * w2
            - 2.0 * v2 * w2
            - 32.0 * ca2 * v2 * w2
            + 32.0 * cf * v2 * w2
            + 116.0 * cacf * v2 * w2
            - 4.0 * v3 * w2
            + 10.0 * ca2 * v3 * w2
            - 48.0 * cf * v3 * w2
            - 156.0 * cacf * v3 * w2
            + 11.0 * v4 * w2
            + 27.0 * ca2 * v4 * w2
            + 16.0 * cf * v4 * w2
            + 72.0 * cacf * v4 * w2
            + 9.0 * v5 * w2
            - 43.0 * ca2 * v5 * w2
            - 4.0 * cf * v5 * w2
            - 28.0 * cacf * v5 * w2
            - 12.0 * v6 * w2
            + 24.0 * ca2 * v6 * w2
            + 6.0 * v7 * w2
            - 6.0 * ca2 * v7 * w2
            - 12.0 * ca2 * v * w3
            + 12.0 * cacf * v * w3
            + 12.0 * ca2 * v2 * w3
            - 12.0 * cf * v2 * w3
            + 6.0 * v3 * w3
            + 16.0 * ca2 * v3 * w3
            - 24.0 * cf * v3 * w3
            - 68.0 * cacf * v3 * w3
            - 8.0 * v4 * w3
            - 32.0 * ca2 * v4 * w3
            + 40.0 * cf * v4 * w3
            + 104.0 * cacf * v4 * w3
            - 15.0 * v5 * w3
            + 25.0 * ca2 * v5 * w3
            - 4.0 * cf * v5 * w3
            - 28.0 * cacf * v5 * w3
            + 5.0 * v6 * w3
            - ca2 * v6 * w3
            + 20.0 * cacf * v6 * w3
            - 4.0 * ca2 * v7 * w3
            - 2.0 * v8 * w3
            + 2.0 * ca2 * v8 * w3
            + 12.0 * ca2 * v2 * w4
            - 12.0 * cacf * v2 * w4
            - 28.0 * ca2 * v3 * w4
            + 12.0 * cf * v3 * w4
            + 24.0 * cacf * v3 * w4
            - 6.0 * v4 * w4
            + 24.0 * ca2 * v4 * w4
            - 12.0 * cacf * v4 * w4
            + 18.0 * v5 * w4
            - 8.0 * ca2 * v5 * w4
            - 12.0 * cf * v5 * w4
            - 8.0 * cacf * v5 * w4
            + 3.0 * v6 * w4
            - 13.0 * ca2 * v6 * w4
            - 8.0 * cacf * v6 * w4
            - v7 * w4
            + 7.0 * ca2 * v7 * w4
            - 8.0 * cacf * v7 * w4
            + 2.0 * v8 * w4
            - 2.0 * ca2 * v8 * w4
            - 4.0 * ca2 * v3 * w5
            + 4.0 * cacf * v3 * w5
            + 12.0 * ca2 * v4 * w5
            - 4.0 * cf * v4 * w5
            - 12.0 * cacf * v4 * w5
            + 2.0 * v5 * w5
            - 16.0 * ca2 * v5 * w5
            + 4.0 * cf * v5 * w5
            + 16.0 * cacf * v5 * w5
            - 13.0 * v6 * w5
            + 17.0 * ca2 * v6 * w5
            - 12.0 * cacf * v6 * w5
            - v7 * w5
            - 5.0 * ca2 * v7 * w5
            + 8.0 * cacf * v7 * w5
            - 2.0 * v8 * w5
            + 2.0 * ca2 * v8 * w5
            + 6.0 * v7 * w6
            - 2.0 * ca2 * v7 * w6
            + 2.0 * v8 * w6
            - 2.0 * ca2 * v8 * w6
            - 2.0 * v8 * w7
            + 2.0 * ca2 * v8 * w7))
        / ((1.0 - v).powi(2) * v3 * w2 * (1.0 - v * w).powi(3));

    let part6 = -(2.0
        * cf
        * l1w
        * (16.0 * ca - 16.0 * ca3 - 64.0 * ca * v + 64.0 * ca3 * v + 112.0 * ca * v2
            - 112.0 * ca3 * v2
            - 112.0 * ca * v3
            + 112.0 * ca3 * v3
            + 68.0 * ca * v4
            - 68.0 * ca3 * v4
            - 24.0 * ca * v5
            + 24.0 * ca3 * v5
            + 4.0 * ca * v6
            - 4.0 * ca3 * v6
            + 18.0 * ca * w
            - 10.0 * ca3 * w
            - 18.0 * v * w
            - 120.0 * ca * v * w
            + 12.0 * ca2 * v * w
            + 76.0 * ca3 * v * w
            + 64.0 * v2 * w
            + 303.0 * ca * v2 * w
            - 34.0 * ca2 * v2 * w
            - 213.0 * ca3 * v2 * w
            - 92.0 * v3 * w
            - 397.0 * ca * v3 * w
            + 38.0 * ca2 * v3 * w
            + 311.0 * ca3 * v3 * w
            + 68.0 * v4 * w
            + 323.0 * ca * v4 * w
            - 22.0 * ca2 * v4 * w
            - 285.0 * ca3 * v4 * w
            - 26.0 * v5 * w
            - 179.0 * ca * v5 * w
            + 6.0 * ca2 * v5 * w
            + 173.0 * ca3 * v5 * w
            + 4.0 * v6 * w
            + 64.0 * ca * v6 * w
            - 64.0 * ca3 * v6 * w
            - 12.0 * ca * v7 * w
            + 12.0 * ca3 * v7 * w
            - 2.0 * ca * w2
            - 2.0 * ca3 * w2
            + 2.0 * v * w2
            - 28.0 * ca * v * w2
            - 2.0 * ca2 * v * w2
            + 28.0 * ca3 * v * w2
            + 30.0 * v2 * w2
            + 182.0 * ca * v2 * w2
            - 26.0 * ca2 * v2 * w2
            - 110.0 * ca3 * v2 * w2
            - 114.0 * v3 * w2
            - 451.0 * ca * v3 * w2
            + 82.0 * ca2 * v3 * w2
            + 245.0 * ca3 * v3 * w2
            + 158.0 * v4 * w2
            + 515.0 * ca * v4 * w2
            - 86.0 * ca2 * v4 * w2
            - 281.0 * ca3 * v4 * w2
            - 116.0 * v5 * w2
            - 313.0 * ca * v5 * w2
            + 48.0 * ca2 * v5 * w2
            + 199.0 * ca3 * v5 * w2
            + 48.0 * v6 * w2
            + 137.0 * ca * v6 * w2
            - 16.0 * ca2 * v6 * w2
            - 119.0 * ca3 * v6 * w2
            - 8.0 * v7 * w2
            - 48.0 * ca * v7 * w2
            + 48.0 * ca3 * v7 * w2
            + 12.0 * ca * v8 * w2
            - 12.0 * ca3 * v8 * w2
            + 4.0 * ca * v * w3
            + 4.0 * ca3 * v * w3
            - 4.0 * v2 * w3
            - 18.0 * ca * v2 * w3
            + 4.0 * ca2 * v2 * w3
            - 18.0 * ca3 * v2 * w3
            + 10.0 * v3 * w3
            + 8.0 * ca * v3 * w3
            - 6.0 * ca2 * v3 * w3
            + 28.0 * ca3 * v3 * w3
            + 20.0 * v4 * w3
            + 120.0 * ca * v4 * w3
            - 32.0 * ca2 * v4 * w3
            - 52.0 * ca3 * v4 * w3
            - 42.0 * v5 * w3
            - 212.0 * ca * v5 * w3
            + 44.0 * ca2 * v5 * w3
            + 52.0 * ca3 * v5 * w3
            + 32.0 * v6 * w3
            + 89.0 * ca * v6 * w3
            - 26.0 * ca2 * v6 * w3
            + 9.0 * ca3 * v6 * w3
            - 20.0 * v7 * w3
            - 15.0 * ca * v7 * w3
            + 16.0 * ca2 * v7 * w3
            + ca3 * v7 * w3
            + 4.0 * v8 * w3
            - 4.0 * ca * v9 * w3
            + 4.0 * ca3 * v9 * w3
            + 42.0 * ca * v3 * w4
            - 14.0 * ca3 * v3 * w4
            - 42.0 * v4 * w4
            - 156.0 * ca * v4 * w4
            + 38.0 * ca2 * v4 * w4
            + 56.0 * ca3 * v4 * w4
            + 58.0 * v5 * w4
            + 210.0 * ca * v5 * w4
            - 42.0 * ca2 * v5 * w4
            - 94.0 * ca3 * v5 * w4
            - 34.0 * v6 * w4
            - 72.0 * ca * v6 * w4
            + 22.0 * ca2 * v6 * w4
            + 48.0 * ca3 * v6 * w4
            + 20.0 * v7 * w4
            + 21.0 * ca * v7 * w4
            - 12.0 * ca2 * v7 * w4
            - 47.0 * ca3 * v7 * w4
            - 2.0 * v8 * w4
            - ca * v8 * w4
            - 6.0 * ca2 * v8 * w4
            + 7.0 * ca3 * v8 * w4
            + 8.0 * ca * v9 * w4
            - 8.0 * ca3 * v9 * w4
            - 4.0 * ca * v3 * w5
            - 4.0 * ca3 * v3 * w5
            + 4.0 * v4 * w5
            - 8.0 * ca * v4 * w5
            - 4.0 * ca2 * v4 * w5
            + 20.0 * ca3 * v4 * w5
            + 20.0 * v5 * w5
            + 64.0 * ca * v5 * w5
            - 14.0 * ca2 * v5 * w5
            - 40.0 * ca3 * v5 * w5
            - 32.0 * v6 * w5
            - 113.0 * ca * v6 * w5
            + 18.0 * ca2 * v6 * w5
            + 47.0 * ca3 * v6 * w5
            + 12.0 * v7 * w5
            + 57.0 * ca * v7 * w5
            - 12.0 * ca2 * v7 * w5
            - 7.0 * ca3 * v7 * w5
            - 4.0 * v8 * w5
            - 24.0 * ca * v8 * w5
            + 12.0 * ca2 * v8 * w5
            + 12.0 * ca3 * v8 * w5
            - 8.0 * ca * v9 * w5
            + 8.0 * ca3 * v9 * w5
            + 2.0 * ca * v4 * w6
            + 2.0 * ca3 * v4 * w6
            - 2.0 * v5 * w6
            - 6.0 * ca * v5 * w6
            + 2.0 * ca2 * v5 * w6
            - 6.0 * ca3 * v5 * w6
            + 4.0 * v6 * w6
            + 6.0 * ca * v6 * w6
            - 4.0 * ca2 * v6 * w6
            + 6.0 * ca3 * v6 * w6
            - 13.0 * ca * v7 * w6
            + 8.0 * ca2 * v7 * w6
            + 3.0 * ca3 * v7 * w6
            - 2.0 * v8 * w6
            + 15.0 * ca * v8 * w6
            - 6.0 * ca2 * v8 * w6
            - 9.0 * ca3 * v8 * w6
            + 8.0 * ca * v9 * w6
            - 8.0 * ca3 * v9 * w6
            - 4.0 * v7 * w7
            + 4.0 * v8 * w7
            - 8.0 * ca * v9 * w7
            + 8.0 * ca3 * v9 * w7
            + 4.0 * ca * v9 * w8
            - 4.0 * ca3 * v9 * w8))
        / (ca * (1.0 - v).powi(2) * v3 * w2 * (1.0 - v * w).powi(3) * (1.0 - v + v * w));

    let part7 = -(2.0
        * cf
        * lv
        * (24.0 * ca - 16.0 * ca3 - 96.0 * ca * v + 64.0 * ca3 * v + 164.0 * ca * v2
            - 112.0 * ca3 * v2
            - 156.0 * ca * v3
            + 112.0 * ca3 * v3
            + 88.0 * ca * v4
            - 68.0 * ca3 * v4
            - 28.0 * ca * v5
            + 24.0 * ca3 * v5
            + 4.0 * ca * v6
            - 4.0 * ca3 * v6
            + 26.0 * ca * w
            - 2.0 * ca3 * w
            - 26.0 * v * w
            - 152.0 * ca * v * w
            + 10.0 * ca2 * v * w
            + 40.0 * ca3 * v * w
            + 88.0 * v2 * w
            + 367.0 * ca * v2 * w
            - 32.0 * ca2 * v2 * w
            - 147.0 * ca3 * v2 * w
            - 122.0 * v3 * w
            - 485.0 * ca * v3 * w
            + 42.0 * ca2 * v3 * w
            + 249.0 * ca3 * v3 * w
            + 90.0 * v4 * w
            + 407.0 * ca * v4 * w
            - 30.0 * ca2 * v4 * w
            - 255.0 * ca3 * v4 * w
            - 36.0 * v5 * w
            - 227.0 * ca * v5 * w
            + 12.0 * ca2 * v5 * w
            + 167.0 * ca3 * v5 * w
            + 6.0 * v6 * w
            + 76.0 * ca * v6 * w
            - 2.0 * ca2 * v6 * w
            - 64.0 * ca3 * v6 * w
            - 12.0 * ca * v7 * w
            + 12.0 * ca3 * v7 * w
            - 2.0 * ca * w2
            - 2.0 * ca3 * w2
            + 2.0 * v * w2
            - 44.0 * ca * v * w2
            - 2.0 * ca2 * v * w2
            + 12.0 * ca3 * v * w2
            + 48.0 * v2 * w2
            + 222.0 * ca * v2 * w2
            - 16.0 * ca2 * v2 * w2
            - 32.0 * ca3 * v2 * w2
            - 164.0 * v3 * w2
            - 487.0 * ca * v3 * w2
            + 60.0 * ca2 * v3 * w2
            + 99.0 * ca3 * v3 * w2
            + 210.0 * v4 * w2
            + 511.0 * ca * v4 * w2
            - 74.0 * ca2 * v4 * w2
            - 135.0 * ca3 * v4 * w2
            - 152.0 * v5 * w2
            - 317.0 * ca * v5 * w2
            + 52.0 * ca2 * v5 * w2
            + 121.0 * ca3 * v5 * w2
            + 70.0 * v6 * w2
            + 169.0 * ca * v6 * w2
            - 26.0 * ca2 * v6 * w2
            - 103.0 * ca3 * v6 * w2
            - 14.0 * v7 * w2
            - 60.0 * ca * v7 * w2
            + 6.0 * ca2 * v7 * w2
            + 48.0 * ca3 * v7 * w2
            + 12.0 * ca * v8 * w2
            - 12.0 * ca3 * v8 * w2
            + 4.0 * ca * v * w3
            + 4.0 * ca3 * v * w3
            - 4.0 * v2 * w3
            - 18.0 * ca * v2 * w3
            + 4.0 * ca2 * v2 * w3
            - 18.0 * ca3 * v2 * w3
            + 10.0 * v3 * w3
            - 10.0 * ca2 * v3 * w3
            + 8.0 * ca3 * v3 * w3
            + 30.0 * v4 * w3
            + 108.0 * ca * v4 * w3
            - 10.0 * ca2 * v4 * w3
            + 16.0 * ca3 * v4 * w3
            - 40.0 * v5 * w3
            - 140.0 * ca * v5 * w3
            + 16.0 * ca2 * v5 * w3
            - 40.0 * ca3 * v5 * w3
            + 22.0 * v6 * w3
            + 5.0 * ca * v6 * w3
            - 6.0 * ca2 * v6 * w3
            + 75.0 * ca3 * v6 * w3
            - 28.0 * v7 * w3
            + 15.0 * ca * v7 * w3
            + 12.0 * ca2 * v7 * w3
            - 13.0 * ca3 * v7 * w3
            + 10.0 * v8 * w3
            + 4.0 * ca * v8 * w3
            - 6.0 * ca2 * v8 * w3
            - 4.0 * ca * v9 * w3
            + 4.0 * ca3 * v9 * w3
            + 58.0 * ca * v3 * w4
            + 2.0 * ca3 * v3 * w4
            - 58.0 * v4 * w4
            - 172.0 * ca * v4 * w4
            + 26.0 * ca2 * v4 * w4
            + 8.0 * ca3 * v4 * w4
            + 74.0 * v5 * w4
            + 226.0 * ca * v5 * w4
            - 30.0 * ca2 * v5 * w4
            - 54.0 * ca3 * v5 * w4
            - 56.0 * v6 * w4
            - 92.0 * ca * v6 * w4
            + 20.0 * ca2 * v6 * w4
            + 38.0 * ca3 * v6 * w4
            + 54.0 * v7 * w4
            + 85.0 * ca * v7 * w4
            - 26.0 * ca2 * v7 * w4
            - 65.0 * ca3 * v7 * w4
            - 12.0 * v8 * w4
            - 5.0 * ca * v8 * w4
            + 8.0 * ca2 * v8 * w4
            + 11.0 * ca3 * v8 * w4
            - 2.0 * v9 * w4
            + 8.0 * ca * v9 * w4
            + 2.0 * ca2 * v9 * w4
            - 8.0 * ca3 * v9 * w4
            - 4.0 * ca * v3 * w5
            - 4.0 * ca3 * v3 * w5
            + 4.0 * v4 * w5
            - 16.0 * ca * v4 * w5
            - 4.0 * ca2 * v4 * w5
            + 12.0 * ca3 * v4 * w5
            + 24.0 * v5 * w5
            + 72.0 * ca * v5 * w5
            - 8.0 * ca2 * v5 * w5
            - 16.0 * ca3 * v5 * w5
            - 20.0 * v6 * w5
            - 97.0 * ca * v6 * w5
            + 8.0 * ca2 * v6 * w5
            + 17.0 * ca3 * v6 * w5
            - 6.0 * v7 * w5
            + 25.0 * ca * v7 * w5
            + 6.0 * ca2 * v7 * w5
            + 15.0 * ca3 * v7 * w5
            - 8.0 * v8 * w5
            - 40.0 * ca * v8 * w5
            + 4.0 * ca2 * v8 * w5
            + 12.0 * ca3 * v8 * w5
            + 6.0 * v9 * w5
            - 8.0 * ca * v9 * w5
            - 6.0 * ca2 * v9 * w5
            + 8.0 * ca3 * v9 * w5
            + 2.0 * ca * v4 * w6
            + 2.0 * ca3 * v4 * w6
            - 2.0 * v5 * w6
            - 6.0 * ca * v5 * w6
            + 2.0 * ca2 * v5 * w6
            - 6.0 * ca3 * v5 * w6
            + 2.0 * v6 * w6
            + 6.0 * ca * v6 * w6
            - 2.0 * ca2 * v6 * w6
            + 8.0 * ca3 * v6 * w6
            - 6.0 * v7 * w6
            - 25.0 * ca * v7 * w6
            + 2.0 * ca2 * v7 * w6
            + 5.0 * ca3 * v7 * w6
            + 12.0 * v8 * w6
            + 35.0 * ca * v8 * w6
            - 8.0 * ca2 * v8 * w6
            - 13.0 * ca3 * v8 * w6
            - 6.0 * v9 * w6
            + 8.0 * ca * v9 * w6
            + 6.0 * ca2 * v9 * w6
            - 8.0 * ca3 * v9 * w6
            - 2.0 * v8 * w7
            - 4.0 * ca * v8 * w7
            + 2.0 * ca2 * v8 * w7
            + 2.0 * v9 * w7
            - 8.0 * ca * v9 * w7
            - 2.0 * ca2 * v9 * w7
            + 8.0 * ca3 * v9 * w7
            + 4.0 * ca * v9 * w8
            - 4.0 * ca3 * v9 * w8))
        / (ca * (1.0 - v).powi(2) * v3 * w2 * (1.0 - v * w).powi(3) * (1.0 - v + v * w));

    let part8 = -(cf
        * (8.0 * ca * v2 - 8.0 * ca3 * v2 - 24.0 * ca * v3 + 24.0 * ca3 * v3 + 28.0 * ca * v4
            - 28.0 * ca3 * v4
            - 16.0 * ca * v5
            + 16.0 * ca3 * v5
            + 4.0 * ca * v6
            - 4.0 * ca3 * v6
            + 6.0 * ca * w
            - 14.0 * ca3 * w
            - 24.0 * v * w
            - 24.0 * ca * v * w
            + 24.0 * ca2 * v * w
            + 56.0 * ca3 * v * w
            - 40.0 * cacf * v * w
            + 76.0 * v2 * w
            + 39.0 * ca * v2 * w
            - 76.0 * ca2 * v2 * w
            - 91.0 * ca3 * v2 * w
            + 136.0 * cacf * v2 * w
            + 8.0 * ca2cf * v2 * w
            - 108.0 * v3 * w
            - 45.0 * ca * v3 * w
            + 108.0 * ca2 * v3 * w
            + 89.0 * ca3 * v3 * w
            - 192.0 * cacf * v3 * w
            - 24.0 * ca2cf * v3 * w
            + 92.0 * v4 * w
            + 63.0 * ca * v4 * w
            - 92.0 * ca2 * v4 * w
            - 83.0 * ca3 * v4 * w
            + 144.0 * cacf * v4 * w
            + 24.0 * ca2cf * v4 * w
            - 44.0 * v5 * w
            - 67.0 * ca * v5 * w
            + 44.0 * ca2 * v5 * w
            + 71.0 * ca3 * v5 * w
            - 56.0 * cacf * v5 * w
            - 8.0 * ca2cf * v5 * w
            + 8.0 * v6 * w
            + 40.0 * ca * v6 * w
            - 8.0 * ca2 * v6 * w
            - 40.0 * ca3 * v6 * w
            + 8.0 * cacf * v6 * w
            - 12.0 * ca * v7 * w
            + 12.0 * ca3 * v7 * w
            + 8.0 * ca * w2
            + 8.0 * ca3 * w2
            - 2.0 * v * w2
            - 76.0 * ca * v * w2
            - 10.0 * ca2 * v * w2
            + 4.0 * ca3 * v * w2
            + 8.0 * cacf * v * w2
            + 64.0 * v2 * w2
            + 220.0 * ca * v2 * w2
            - 12.0 * ca2 * v2 * w2
            - 108.0 * ca3 * v2 * w2
            + 56.0 * cacf * v2 * w2
            - 8.0 * ca2cf * v2 * w2
            - 182.0 * v3 * w2
            - 275.0 * ca * v3 * w2
            + 102.0 * ca2 * v3 * w2
            + 223.0 * ca3 * v3 * w2
            - 232.0 * cacf * v3 * w2
            + 8.0 * ca2cf * v3 * w2
            + 220.0 * v4 * w2
            + 179.0 * ca * v4 * w2
            - 172.0 * ca2 * v4 * w2
            - 207.0 * ca3 * v4 * w2
            + 344.0 * cacf * v4 * w2
            + 48.0 * ca2cf * v4 * w2
            - 180.0 * v5 * w2
            - 99.0 * ca * v5 * w2
            + 176.0 * ca2 * v5 * w2
            + 127.0 * ca3 * v5 * w2
            - 296.0 * cacf * v5 * w2
            - 88.0 * ca2cf * v5 * w2
            + 104.0 * v6 * w2
            + 59.0 * ca * v6 * w2
            - 108.0 * ca2 * v6 * w2
            - 63.0 * ca3 * v6 * w2
            + 144.0 * cacf * v6 * w2
            + 40.0 * ca2cf * v6 * w2
            - 24.0 * v7 * w2
            - 24.0 * ca * v7 * w2
            + 24.0 * ca2 * v7 * w2
            + 24.0 * ca3 * v7 * w2
            - 24.0 * cacf * v7 * w2
            + 12.0 * ca * v8 * w2
            - 12.0 * ca3 * v8 * w2
            - 16.0 * ca * v * w3
            - 16.0 * ca3 * v * w3
            + 4.0 * v2 * w3
            + 136.0 * ca * v2 * w3
            + 20.0 * ca2 * v2 * w3
            + 56.0 * ca3 * v2 * w3
            - 16.0 * cacf * v2 * w3
            - 34.0 * v3 * w3
            - 406.0 * ca * v3 * w3
            - 82.0 * ca2 * v3 * w3
            - 10.0 * ca3 * v3 * w3
            + 56.0 * cacf * v3 * w3
            + 16.0 * ca2cf * v3 * w3
            + 102.0 * v4 * w3
            + 482.0 * ca * v4 * w3
            + 90.0 * ca2 * v4 * w3
            - 122.0 * ca3 * v4 * w3
            - 32.0 * cacf * v4 * w3
            - 72.0 * ca2cf * v4 * w3
            - 70.0 * v5 * w3
            - 216.0 * ca * v5 * w3
            - 42.0 * ca2 * v5 * w3
            + 136.0 * ca3 * v5 * w3
            - 24.0 * cacf * v5 * w3
            + 40.0 * ca2cf * v5 * w3
            + 28.0 * v6 * w3
            + 43.0 * ca * v6 * w3
            - 36.0 * ca2 * v6 * w3
            - 71.0 * ca3 * v6 * w3
            + 104.0 * cacf * v6 * w3
            + 80.0 * ca2cf * v6 * w3
            - 54.0 * v7 * w3
            - 21.0 * ca * v7 * w3
            + 74.0 * ca2 * v7 * w3
            + 25.0 * ca3 * v7 * w3
            - 112.0 * cacf * v7 * w3
            - 64.0 * ca2cf * v7 * w3
            + 24.0 * v8 * w3
            - 8.0 * ca * v8 * w3
            - 24.0 * ca2 * v8 * w3
            + 8.0 * ca3 * v8 * w3
            + 24.0 * cacf * v8 * w3
            - 4.0 * ca * v9 * w3
            + 4.0 * ca3 * v9 * w3
            - 12.0 * ca * v3 * w4
            - 52.0 * ca3 * v3 * w4
            - 42.0 * v4 * w4
            + 138.0 * ca * v4 * w4
            + 78.0 * ca2 * v4 * w4
            + 118.0 * ca3 * v4 * w4
            - 104.0 * cacf * v4 * w4
            + 32.0 * v5 * w4
            - 226.0 * ca * v5 * w4
            - 156.0 * ca2 * v5 * w4
            - 70.0 * ca3 * v5 * w4
            + 192.0 * cacf * v5 * w4
            + 72.0 * ca2cf * v5 * w4
            - 44.0 * v6 * w4
            + 64.0 * ca * v6 * w4
            + 168.0 * ca2 * v6 * w4
            - 208.0 * cacf * v6 * w4
            - 128.0 * ca2cf * v6 * w4
            + 88.0 * v7 * w4
            + 35.0 * ca * v7 * w4
            - 108.0 * ca2 * v7 * w4
            + 25.0 * ca3 * v7 * w4
            + 120.0 * cacf * v7 * w4
            + 16.0 * ca2cf * v7 * w4
            - 26.0 * v8 * w4
            + 9.0 * ca * v8 * w4
            + 10.0 * ca2 * v8 * w4
            - 29.0 * ca3 * v8 * w4
            + 8.0 * cacf * v8 * w4
            + 40.0 * ca2cf * v8 * w4
            - 8.0 * v9 * w4
            + 8.0 * ca * v9 * w4
            + 8.0 * ca2 * v9 * w4
            - 8.0 * ca3 * v9 * w4
            - 8.0 * cacf * v9 * w4
            + 16.0 * ca * v3 * w5
            + 16.0 * ca3 * v3 * w5
            - 4.0 * v4 * w5
            - 110.0 * ca * v4 * w5
            - 20.0 * ca2 * v4 * w5
            - 10.0 * ca3 * v4 * w5
            + 16.0 * cacf * v4 * w5
            + 50.0 * v5 * w5
            + 182.0 * ca * v5 * w5
            + 18.0 * ca2 * v5 * w5
            - 54.0 * ca3 * v5 * w5
            + 16.0 * cacf * v5 * w5
            - 16.0 * ca2cf * v5 * w5
            - 34.0 * v6 * w5
            - 89.0 * ca * v6 * w5
            + 2.0 * ca2 * v6 * w5
            + 85.0 * ca3 * v6 * w5
            - 32.0 * cacf * v6 * w5
            - 24.0 * v7 * w5
            - 7.0 * ca * v7 * w5
            - 20.0 * ca2 * v7 * w5
            - 53.0 * ca3 * v7 * w5
            + 40.0 * cacf * v7 * w5
            + 64.0 * ca2cf * v7 * w5
            - 8.0 * v8 * w5
            - 2.0 * ca * v8 * w5
            + 40.0 * ca2 * v8 * w5
            + 26.0 * ca3 * v8 * w5
            - 56.0 * cacf * v8 * w5
            - 40.0 * ca2cf * v8 * w5
            + 20.0 * v9 * w5
            - 16.0 * ca * v9 * w5
            - 20.0 * ca2 * v9 * w5
            + 16.0 * ca3 * v9 * w5
            + 16.0 * cacf * v9 * w5
            - 8.0 * ca2cf * v9 * w5
            - 8.0 * ca * v4 * w6
            - 8.0 * ca3 * v4 * w6
            + 2.0 * v5 * w6
            + 56.0 * ca * v5 * w6
            + 10.0 * ca2 * v5 * w6
            + 16.0 * ca3 * v5 * w6
            - 8.0 * cacf * v5 * w6
            - 14.0 * v6 * w6
            - 110.0 * ca * v6 * w6
            - 26.0 * ca2 * v6 * w6
            - 2.0 * ca3 * v6 * w6
            + 16.0 * cacf * v6 * w6
            + 8.0 * ca2cf * v6 * w6
            + 14.0 * v7 * w6
            + 89.0 * ca * v7 * w6
            + 30.0 * ca2 * v7 * w6
            - 13.0 * ca3 * v7 * w6
            - 24.0 * cacf * v7 * w6
            - 16.0 * ca2cf * v7 * w6
            + 18.0 * v8 * w6
            - 15.0 * ca * v8 * w6
            - 34.0 * ca2 * v8 * w6
            - 5.0 * ca3 * v8 * w6
            + 32.0 * cacf * v8 * w6
            - 20.0 * v9 * w6
            + 16.0 * ca * v9 * w6
            + 20.0 * ca2 * v9 * w6
            - 16.0 * ca3 * v9 * w6
            - 16.0 * cacf * v9 * w6
            + 8.0 * ca2cf * v9 * w6
            - 8.0 * v8 * w7
            - 8.0 * ca * v8 * w7
            + 8.0 * ca2 * v8 * w7
            + 8.0 * ca3 * v8 * w7
            - 8.0 * cacf * v8 * w7
            + 8.0 * v9 * w7
            - 8.0 * ca * v9 * w7
            - 8.0 * ca2 * v9 * w7
            + 8.0 * ca3 * v9 * w7
            + 8.0 * cacf * v9 * w7
            + 4.0 * ca * v9 * w8
            - 4.0 * ca3 * v9 * w8))
        / (ca * (1.0 - v).powi(2) * v3 * w2 * (1.0 - v * w).powi(3) * (1.0 - v + v * w));

    let part9 = -(4.0
        * cf
        * lw
        * (8.0 * ca - 12.0 * ca3 + 8.0 * ca2cf - 32.0 * ca * v + 48.0 * ca3 * v
            - 32.0 * ca2cf * v
            + 58.0 * ca * v2
            - 84.0 * ca3 * v2
            + 56.0 * ca2cf * v2
            - 62.0 * ca * v3
            + 84.0 * ca3 * v3
            - 56.0 * ca2cf * v3
            + 41.0 * ca * v4
            - 51.0 * ca3 * v4
            + 34.0 * ca2cf * v4
            - 16.0 * ca * v5
            + 18.0 * ca3 * v5
            - 12.0 * ca2cf * v5
            + 3.0 * ca * v6
            - 3.0 * ca3 * v6
            + 2.0 * ca2cf * v6
            + ca * w
            + 3.0 * ca3 * w
            + 2.0 * ca2cf * w
            - 10.0 * v * w
            - 20.0 * ca * v * w
            + 10.0 * ca2 * v * w
            + 12.0 * ca3 * v * w
            - 10.0 * cacf * v * w
            - 24.0 * ca2cf * v * w
            + 32.0 * v2 * w
            + 64.0 * ca * v2 * w
            - 32.0 * ca2 * v2 * w
            - 74.0 * ca3 * v2 * w
            + 32.0 * cacf * v2 * w
            + 74.0 * ca2cf * v2 * w
            - 42.0 * v3 * w
            - 99.0 * ca * v3 * w
            + 42.0 * ca2 * v3 * w
            + 141.0 * ca3 * v3 * w
            - 42.0 * cacf * v3 * w
            - 112.0 * ca2cf * v3 * w
            + 30.0 * v4 * w
            + 108.0 * ca * v4 * w
            - 30.0 * ca2 * v4 * w
            - 156.0 * ca3 * v4 * w
            + 30.0 * cacf * v4 * w
            + 110.0 * ca2cf * v4 * w
            - 12.0 * v5 * w
            - 84.0 * ca * v5 * w
            + 12.0 * ca2 * v5 * w
            + 110.0 * ca3 * v5 * w
            - 12.0 * cacf * v5 * w
            - 74.0 * ca2cf * v5 * w
            + 2.0 * v6 * w
            + 39.0 * ca * v6 * w
            - 2.0 * ca2 * v6 * w
            - 45.0 * ca3 * v6 * w
            + 2.0 * cacf * v6 * w
            + 30.0 * ca2cf * v6 * w
            - 9.0 * ca * v7 * w
            + 9.0 * ca3 * v7 * w
            - 6.0 * ca2cf * v7 * w
            - 10.0 * ca * v * w2
            - 3.0 * ca2 * v * w2
            - 4.0 * ca3 * v * w2
            - 4.0 * ca2cf * v * w2
            + 11.0 * v2 * w2
            + 54.0 * ca * v2 * w2
            + ca2 * v2 * w2
            + 5.0 * ca3 * v2 * w2
            + 8.0 * cacf * v2 * w2
            + 30.0 * ca2cf * v2 * w2
            - 38.0 * v3 * w2
            - 98.0 * ca * v3 * w2
            + 22.0 * ca2 * v3 * w2
            + 13.0 * ca3 * v3 * w2
            - 30.0 * cacf * v3 * w2
            - 70.0 * ca2cf * v3 * w2
            + 52.0 * v4 * w2
            + 58.0 * ca * v4 * w2
            - 44.0 * ca2 * v4 * w2
            - 14.0 * ca3 * v4 * w2
            + 44.0 * cacf * v4 * w2
            + 62.0 * ca2cf * v4 * w2
            - 44.0 * v5 * w2
            - 11.0 * ca * v5 * w2
            + 43.0 * ca2 * v5 * w2
            + 16.0 * ca3 * v5 * w2
            - 40.0 * cacf * v5 * w2
            - 32.0 * ca2cf * v5 * w2
            + 25.0 * v6 * w2
            + 22.0 * ca * v6 * w2
            - 25.0 * ca2 * v6 * w2
            - 37.0 * ca3 * v6 * w2
            + 24.0 * cacf * v6 * w2
            + 28.0 * ca2cf * v6 * w2
            - 6.0 * v7 * w2
            - 21.0 * ca * v7 * w2
            + 6.0 * ca2 * v7 * w2
            + 27.0 * ca3 * v7 * w2
            - 6.0 * cacf * v7 * w2
            - 18.0 * ca2cf * v7 * w2
            + 9.0 * ca * v8 * w2
            - 9.0 * ca3 * v8 * w2
            + 6.0 * ca2cf * v8 * w2
            + ca * w3
            - ca3 * w3
            + 2.0 * ca2cf * w3
            - v * w3
            - 4.0 * ca * v * w3
            + ca2 * v * w3
            + 4.0 * ca3 * v * w3
            - 2.0 * cacf * v * w3
            - 8.0 * ca2cf * v * w3
            + v2 * w3
            + 23.0 * ca * v2 * w3
            + ca2 * v2 * w3
            - 10.0 * ca3 * v2 * w3
            + 4.0 * cacf * v2 * w3
            + 14.0 * ca2cf * v2 * w3
            + 9.0 * v3 * w3
            - 77.0 * ca * v3 * w3
            - 23.0 * ca2 * v3 * w3
            + 8.0 * ca3 * v3 * w3
            + 10.0 * cacf * v3 * w3
            - 18.0 * ca2cf * v3 * w3
            - 18.0 * v4 * w3
            + 142.0 * ca * v4 * w3
            + 44.0 * ca2 * v4 * w3
            - 16.0 * ca3 * v4 * w3
            - 28.0 * cacf * v4 * w3
            + 40.0 * ca2cf * v4 * w3
            + 27.0 * v5 * w3
            - 84.0 * ca * v5 * w3
            - 43.0 * ca2 * v5 * w3
            - 18.0 * ca3 * v5 * w3
            + 36.0 * cacf * v5 * w3
            - 22.0 * ca2cf * v5 * w3
            - 17.0 * v6 * w3
            - 32.0 * ca * v6 * w3
            + 19.0 * ca2 * v6 * w3
            + 72.0 * ca3 * v6 * w3
            - 20.0 * cacf * v6 * w3
            - 30.0 * ca2cf * v6 * w3
            - 7.0 * v7 * w3
            + 34.0 * ca * v7 * w3
            + 7.0 * ca2 * v7 * w3
            - 42.0 * ca3 * v7 * w3
            - 6.0 * cacf * v7 * w3
            + 24.0 * ca2cf * v7 * w3
            + 6.0 * v8 * w3
            - 11.0 * ca * v8 * w3
            - 6.0 * ca2 * v8 * w3
            + 9.0 * ca3 * v8 * w3
            + 6.0 * cacf * v8 * w3
            - 6.0 * ca2cf * v8 * w3
            - 3.0 * ca * v9 * w3
            + 3.0 * ca3 * v9 * w3
            - 2.0 * ca2cf * v9 * w3
            - 2.0 * ca * v * w4
            + 2.0 * ca3 * v * w4
            - 4.0 * ca2cf * v * w4
            + 2.0 * v2 * w4
            + 9.0 * ca * v2 * w4
            - 2.0 * ca2 * v2 * w4
            - 9.0 * ca3 * v2 * w4
            + 4.0 * cacf * v2 * w4
            + 18.0 * ca2cf * v2 * w4
            - 5.0 * v3 * w4
            - 15.0 * ca * v3 * w4
            + 7.0 * ca2 * v3 * w4
            + 21.0 * ca3 * v3 * w4
            - 10.0 * cacf * v3 * w4
            - 30.0 * ca2cf * v3 * w4
            - v4 * w4
            + 25.0 * ca * v4 * w4
            + ca2 * v4 * w4
            - 17.0 * ca3 * v4 * w4
            + 14.0 * ca2cf * v4 * w4
            - 92.0 * ca * v5 * w4
            - 13.0 * ca2 * v5 * w4
            + 47.0 * ca3 * v5 * w4
            + 10.0 * cacf * v5 * w4
            - 22.0 * ca2cf * v5 * w4
            - 19.0 * v6 * w4
            + 112.0 * ca * v6 * w4
            + 31.0 * ca2 * v6 * w4
            - 69.0 * ca3 * v6 * w4
            - 26.0 * cacf * v6 * w4
            + 44.0 * ca2cf * v6 * w4
            + 39.0 * v7 * w4
            - 26.0 * ca * v7 * w4
            - 40.0 * ca2 * v7 * w4
            + 13.0 * ca3 * v7 * w4
            + 38.0 * cacf * v7 * w4
            - 14.0 * ca2cf * v7 * w4
            - 14.0 * v8 * w4
            - ca * v8 * w4
            + 14.0 * ca2 * v8 * w4
            + 8.0 * ca3 * v8 * w4
            - 14.0 * cacf * v8 * w4
            - 2.0 * ca2cf * v8 * w4
            - 2.0 * v9 * w4
            + 9.0 * ca * v9 * w4
            + 2.0 * ca2 * v9 * w4
            - 9.0 * ca3 * v9 * w4
            - 2.0 * cacf * v9 * w4
            + 6.0 * ca2cf * v9 * w4
            - 3.0 * ca * v3 * w5
            + 3.0 * ca3 * v3 * w5
            - 6.0 * ca2cf * v3 * w5
            + 3.0 * v4 * w5
            - 8.0 * ca * v4 * w5
            - 5.0 * ca2 * v4 * w5
            - 8.0 * ca3 * v4 * w5
            + 6.0 * cacf * v4 * w5
            + 16.0 * ca2cf * v4 * w5
            - 6.0 * v5 * w5
            + 33.0 * ca * v5 * w5
            + 10.0 * ca2 * v5 * w5
            - 5.0 * ca3 * v5 * w5
            - 10.0 * cacf * v5 * w5
            - 10.0 * ca2cf * v5 * w5
            + 33.0 * v6 * w5
            - 6.0 * ca * v6 * w5
            - 27.0 * ca2 * v6 * w5
            + 2.0 * ca3 * v6 * w5
            + 24.0 * cacf * v6 * w5
            - 4.0 * ca2cf * v6 * w5
            - 42.0 * v7 * w5
            - 38.0 * ca * v7 * w5
            + 34.0 * ca2 * v7 * w5
            + 28.0 * ca3 * v7 * w5
            - 32.0 * cacf * v7 * w5
            - 6.0 * ca2cf * v7 * w5
            + 4.0 * v8 * w5
            + 10.0 * ca * v8 * w5
            - 4.0 * ca2 * v8 * w5
            - 16.0 * ca3 * v8 * w5
            + 4.0 * cacf * v8 * w5
            + 4.0 * ca2cf * v8 * w5
            + 8.0 * v9 * w5
            - 11.0 * ca * v9 * w5
            - 8.0 * ca2 * v9 * w5
            + 11.0 * ca3 * v9 * w5
            + 8.0 * cacf * v9 * w5
            - 8.0 * ca2cf * v9 * w5
            + 2.0 * ca * v3 * w6
            - 2.0 * ca3 * v3 * w6
            + 4.0 * ca2cf * v3 * w6
            - 2.0 * v4 * w6
            - 5.0 * ca * v4 * w6
            + 2.0 * ca2 * v4 * w6
            + 5.0 * ca3 * v4 * w6
            - 4.0 * cacf * v4 * w6
            - 10.0 * ca2cf * v4 * w6
            + 3.0 * v5 * w6
            + 13.0 * ca * v5 * w6
            - 5.0 * ca3 * v5 * w6
            + 2.0 * cacf * v5 * w6
            + 10.0 * ca2cf * v5 * w6
            - 14.0 * v6 * w6
            - 32.0 * ca * v6 * w6
            + 4.0 * ca2 * v6 * w6
            + 11.0 * ca3 * v6 * w6
            - 2.0 * cacf * v6 * w6
            - 2.0 * ca2cf * v6 * w6
            + 9.0 * v7 * w6
            + 28.0 * ca * v7 * w6
            - 6.0 * ca2 * v7 * w6
            - 16.0 * ca3 * v7 * w6
            + 4.0 * cacf * v7 * w6
            + 16.0 * v8 * w6
            + 7.0 * ca * v8 * w6
            - 12.0 * ca2 * v8 * w6
            + 2.0 * ca3 * v8 * w6
            + 12.0 * cacf * v8 * w6
            + 4.0 * ca2cf * v8 * w6
            - 12.0 * v9 * w6
            + 8.0 * ca * v9 * w6
            + 12.0 * ca2 * v9 * w6
            - 8.0 * ca3 * v9 * w6
            - 12.0 * cacf * v9 * w6
            + 8.0 * ca2cf * v9 * w6
            - ca * v4 * w7
            + ca3 * v4 * w7
            - 2.0 * ca2cf * v4 * w7
            + v5 * w7
            + 3.0 * ca * v5 * w7
            - ca2 * v5 * w7
            - 3.0 * ca3 * v5 * w7
            + 2.0 * cacf * v5 * w7
            + 6.0 * ca2cf * v5 * w7
            - 4.0 * ca * v6 * w7
            + 3.0 * ca3 * v6 * w7
            - 2.0 * cacf * v6 * w7
            - 8.0 * ca2cf * v6 * w7
            + 7.0 * v7 * w7
            + 10.0 * ca * v7 * w7
            - ca2 * v7 * w7
            - 5.0 * ca3 * v7 * w7
            + 2.0 * cacf * v7 * w7
            + 8.0 * ca2cf * v7 * w7
            - 16.0 * v8 * w7
            - 16.0 * ca * v8 * w7
            + 10.0 * ca2 * v8 * w7
            + 6.0 * ca3 * v8 * w7
            - 10.0 * cacf * v8 * w7
            - 6.0 * ca2cf * v8 * w7
            + 8.0 * v9 * w7
            - 5.0 * ca * v9 * w7
            - 8.0 * ca2 * v9 * w7
            + 5.0 * ca3 * v9 * w7
            + 8.0 * cacf * v9 * w7
            - 8.0 * ca2cf * v9 * w7
            - 2.0 * v7 * w8
            + 4.0 * v8 * w8
            + 2.0 * ca * v8 * w8
            - 2.0 * ca2 * v8 * w8
            + 2.0 * cacf * v8 * w8
            - 2.0 * v9 * w8
            + 3.0 * ca * v9 * w8
            + 2.0 * ca2 * v9 * w8
            - 3.0 * ca3 * v9 * w8
            - 2.0 * cacf * v9 * w8
            + 6.0 * ca2cf * v9 * w8
            - ca * v9 * w9
            + ca3 * v9 * w9
            - 2.0 * ca2cf * v9 * w9))
        / (ca
            * (1.0 - v).powi(2)
            * v3
            * (1.0 - w)
            * w2
            * (1.0 - v * w).powi(3)
            * (1.0 - v + v * w));

    part1 + part2 + part3 + part4 + part5 + part6 + part7 + part8 + part9
}

/// `STRUV7(W,V,X3,S)`.
#[must_use]
pub fn struv7(w: f64, v: f64, _x3: f64, _s: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let cf = ctx.cf;
    let ca = ctx.ca;
    let (v2, v3, v4, v5, v6, v7, v8) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8);
    let (w2, w3, w4, w5, w6, w7, w8) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8);
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let (l1v, lw, lvw, l1vw, lms, l1w, lv, lmss) = (
        pre.l1v, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.l1w, pre.lv, pre.lmss,
    );

    let part1 = (16.0
        * ca
        * cf.powi(2)
        * l1vw
        * (1.0 - w)
        * (1.0 - v - v2 + v3 + v * w + 4.0 * v2 * w - 5.0 * v3 * w - 3.0 * v2 * w2
            + 5.0 * v3 * w2
            - v3 * w3))
        / ((1.0 - v) * v * w * (1.0 - v + v * w).powi(2));

    let part2 = -(8.0
        * cf.powi(2)
        * lmss
        * (1.0 + v2 - 2.0 * v2 * w + v2 * w2)
        * (ca - 4.0 * ca * v + 6.0 * ca * v2 - 4.0 * ca * v3 + ca * v4 - v * w
            + ca * v * w
            + 3.0 * v2 * w
            - 3.0 * ca * v2 * w
            - 3.0 * v3 * w
            + 3.0 * ca * v3 * w
            + v4 * w
            - ca * v4 * w
            - 2.0 * v2 * w2
            + ca * v2 * w2
            + 4.0 * v3 * w2
            - 2.0 * ca * v3 * w2
            - 2.0 * v4 * w2
            + ca * v4 * w2
            - v3 * w3
            + ca * v3 * w3
            + v4 * w3
            - ca * v4 * w3
            + ca * v4 * w4))
        / ((1.0 - v).powi(2) * v2 * w2 * (1.0 - v + v * w).powi(2));

    let part3 = (4.0
        * cf
        * lvw
        * (1.0 - w)
        * (4.0 * ca2 + 4.0 * ca2 * v2 + 2.0 * v * w
            - 9.0 * ca2 * v * w
            - 6.0 * ca2 * v2 * w
            - 2.0 * v3 * w
            - ca2 * v3 * w
            - 5.0 * v2 * w2
            + 12.0 * ca2 * v2 * w2
            + 2.0 * v3 * w2
            + 3.0 * ca2 * v3 * w2
            - v4 * w2
            + ca2 * v4 * w2
            + 2.0 * v3 * w3
            - 6.0 * ca2 * v3 * w3
            + 2.0 * v4 * w3
            - 2.0 * ca2 * v4 * w3
            - 2.0 * v4 * w4
            + 2.0 * ca2 * v4 * w4))
        / ((1.0 - v).powi(2) * v2 * w2);

    let part4 = -(4.0
        * cf
        * lw
        * (2.0 * ca - 4.0 * ca3 - 6.0 * ca * v + 6.0 * ca3 * v + 8.0 * ca * v2
            - 8.0 * ca3 * v2
            - 8.0 * ca * v3
            + 8.0 * ca3 * v3
            + 6.0 * ca * v4
            - 4.0 * ca3 * v4
            - 2.0 * ca * v5
            + 2.0 * ca3 * v5
            + w
            - ca2 * w
            - 3.0 * v * w
            + 3.0 * ca2 * v * w
            + 6.0 * ca3 * v * w
            + 4.0 * v2 * w
            - 4.0 * ca2 * v2 * w
            - 4.0 * ca3 * v2 * w
            - 4.0 * v3 * w
            + 2.0 * ca * v3 * w
            + 4.0 * ca2 * v3 * w
            - 6.0 * ca3 * v3 * w
            + 3.0 * v4 * w
            - 2.0 * ca * v4 * w
            - 3.0 * ca2 * v4 * w
            - 2.0 * ca3 * v4 * w
            - v5 * w
            - 2.0 * ca * v5 * w
            + ca2 * v5 * w
            + 2.0 * ca * v6 * w
            - 2.0 * ca3 * v6 * w
            - v * w2
            - ca2 * v * w2
            - v2 * w2
            - 5.0 * ca * v2 * w2
            + ca2 * v2 * w2
            - 3.0 * ca3 * v2 * w2
            + 4.0 * v3 * w2
            + ca * v3 * w2
            - 4.0 * ca2 * v3 * w2
            + 11.0 * ca3 * v3 * w2
            - 4.0 * v4 * w2
            + ca * v4 * w2
            + 4.0 * ca2 * v4 * w2
            + 11.0 * ca3 * v4 * w2
            + v5 * w2
            + 7.0 * ca * v5 * w2
            + ca2 * v5 * w2
            - 3.0 * ca3 * v5 * w2
            + v6 * w2
            - 4.0 * ca * v6 * w2
            - ca2 * v6 * w2
            + 4.0 * ca3 * v6 * w2
            + v2 * w3
            + ca2 * v2 * w3
            + 8.0 * ca * v3 * w3
            + 2.0 * ca2 * v3 * w3
            - 4.0 * ca3 * v3 * w3
            - ca * v4 * w3
            + 2.0 * ca2 * v4 * w3
            - 15.0 * ca3 * v4 * w3
            + 2.0 * v5 * w3
            - 10.0 * ca * v5 * w3
            - 8.0 * ca2 * v5 * w3
            + 2.0 * ca3 * v5 * w3
            - 3.0 * v6 * w3
            + 3.0 * ca * v6 * w3
            + 3.0 * ca2 * v6 * w3
            - 3.0 * ca3 * v6 * w3
            - v3 * w4
            - ca2 * v3 * w4
            + v4 * w4
            - 5.0 * ca * v4 * w4
            - 5.0 * ca2 * v4 * w4
            + 7.0 * ca3 * v4 * w4
            - 3.0 * v5 * w4
            + 6.0 * ca * v5 * w4
            + 9.0 * ca2 * v5 * w4
            + 2.0 * ca3 * v5 * w4
            + 3.0 * v6 * w4
            - ca * v6 * w4
            - 3.0 * ca2 * v6 * w4
            + ca3 * v6 * w4
            + 2.0 * ca2 * v4 * w5
            + v5 * w5
            - 3.0 * ca2 * v5 * w5
            - 2.0 * ca3 * v5 * w5
            - v6 * w5
            + ca2 * v6 * w5))
        / (ca * (1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w) * (1.0 - v + v * w));

    let part5 = -(4.0
        * cf
        * l1v
        * (2.0 * ca3 - 4.0 * ca3 * v + 4.0 * ca3 * v2 - 4.0 * ca3 * v3 + 2.0 * ca3 * v4
            - w
            - ca2 * w
            + v * w
            - 2.0 * ca * v * w
            + ca2 * v * w
            - 3.0 * ca3 * v * w
            + 2.0 * ca * v2 * w
            + 3.0 * ca3 * v2 * w
            + 2.0 * ca * v3 * w
            + ca3 * v3 * w
            + v4 * w
            - 2.0 * ca * v4 * w
            + ca2 * v4 * w
            + ca3 * v4 * w
            - v5 * w
            - ca2 * v5 * w
            - 2.0 * ca3 * v5 * w
            + v * w2
            + ca2 * v * w2
            + v2 * w2
            + 6.0 * ca * v2 * w2
            - ca2 * v2 * w2
            + ca3 * v2 * w2
            - 2.0 * v3 * w2
            - 6.0 * ca * v3 * w2
            + 2.0 * ca2 * v3 * w2
            - 3.0 * ca3 * v3 * w2
            - 2.0 * v4 * w2
            - 2.0 * ca * v4 * w2
            - 6.0 * ca2 * v4 * w2
            - 5.0 * ca3 * v4 * w2
            + v5 * w2
            + 2.0 * ca * v5 * w2
            + 5.0 * ca2 * v5 * w2
            + 3.0 * ca3 * v5 * w2
            + v6 * w2
            - ca2 * v6 * w2
            - v2 * w3
            - ca2 * v2 * w3
            - 6.0 * ca * v3 * w3
            - 2.0 * ca2 * v3 * w3
            + ca3 * v3 * w3
            + 2.0 * v4 * w3
            + 6.0 * ca * v4 * w3
            + 8.0 * ca2 * v4 * w3
            + 7.0 * ca3 * v4 * w3
            + 2.0 * v5 * w3
            - 8.0 * ca2 * v5 * w3
            - 3.0 * v6 * w3
            + 3.0 * ca2 * v6 * w3
            + v3 * w4
            + ca2 * v3 * w4
            - v4 * w4
            + 2.0 * ca * v4 * w4
            - 3.0 * ca2 * v4 * w4
            - 3.0 * ca3 * v4 * w4
            - 3.0 * v5 * w4
            - 2.0 * ca * v5 * w4
            + 5.0 * ca2 * v5 * w4
            - 3.0 * ca3 * v5 * w4
            + 3.0 * v6 * w4
            - 3.0 * ca2 * v6 * w4
            + v5 * w5
            - ca2 * v5 * w5
            + 2.0 * ca3 * v5 * w5
            - v6 * w5
            + ca2 * v6 * w5))
        / (ca * (1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w) * (1.0 - v + v * w));

    let part6 = (2.0
        * cf
        * lms
        * (2.0 - 6.0 * ca2 - 4.0 * v + 4.0 * v2 - 8.0 * ca2 * v2 - 4.0 * v3 + 2.0 * v4
            - 2.0 * ca2 * v4
            - 2.0 * w
            + 2.0 * ca2 * w
            + 20.0 * ca2 * v * w
            + 4.0 * v2 * w
            + 12.0 * ca2 * v2 * w
            - 4.0 * v3 * w
            + 16.0 * ca2 * v3 * w
            + 6.0 * v4 * w
            + 2.0 * ca2 * v4 * w
            - 4.0 * v5 * w
            + 4.0 * ca2 * v5 * w
            + w2
            - ca2 * w2
            + 2.0 * v * w2
            - 4.0 * ca2 * v * w2
            - 2.0 * v2 * w2
            - 36.0 * ca2 * v2 * w2
            + 2.0 * v3 * w2
            - 24.0 * ca2 * v3 * w2
            - v4 * w2
            - 13.0 * ca2 * v4 * w2
            - 4.0 * ca2 * v5 * w2
            + 2.0 * v6 * w2
            - 2.0 * ca2 * v6 * w2
            - 2.0 * v * w3
            + 2.0 * ca2 * v * w3
            + 2.0 * v2 * w3
            + 2.0 * ca2 * v2 * w3
            - 6.0 * v3 * w3
            + 40.0 * ca2 * v3 * w3
            - 4.0 * v4 * w3
            + 20.0 * ca2 * v4 * w3
            + 6.0 * ca2 * v5 * w3
            - 2.0 * v6 * w3
            + 2.0 * ca2 * v6 * w3
            + v2 * w4
            - ca2 * v2 * w4
            - 2.0 * v3 * w4
            + 11.0 * v4 * w4
            - 27.0 * ca2 * v4 * w4
            + 2.0 * v5 * w4
            - 8.0 * ca2 * v5 * w4
            + 2.0 * v6 * w4
            - 2.0 * ca2 * v6 * w4
            - 6.0 * v5 * w5
            + 10.0 * ca2 * v5 * w5
            - 2.0 * v6 * w5
            + 2.0 * ca2 * v6 * w5
            + 2.0 * v6 * w6
            - 2.0 * ca2 * v6 * w6))
        / ((1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w).powi(2));

    let part7 = -(2.0
        * cf
        * l1w
        * (4.0 * ca - 12.0 * ca3 - 16.0 * ca * v + 32.0 * ca3 * v + 28.0 * ca * v2
            - 44.0 * ca3 * v2
            - 32.0 * ca * v3
            + 48.0 * ca3 * v3
            + 28.0 * ca * v4
            - 36.0 * ca3 * v4
            - 16.0 * ca * v5
            + 16.0 * ca3 * v5
            + 4.0 * ca * v6
            - 4.0 * ca3 * v6
            - 2.0 * w
            - 2.0 * ca * w
            + 2.0 * ca2 * w
            + 2.0 * ca3 * w
            + 6.0 * v * w
            + 18.0 * ca * v * w
            - 2.0 * ca2 * v * w
            + 8.0 * ca3 * v * w
            - 8.0 * v2 * w
            - 28.0 * ca * v2 * w
            - 4.0 * ca2 * v2 * w
            - 28.0 * ca3 * v2 * w
            + 8.0 * v3 * w
            + 32.0 * ca * v3 * w
            + 8.0 * ca2 * v3 * w
            - 4.0 * ca3 * v3 * w
            - 6.0 * v4 * w
            - 38.0 * ca * v4 * w
            - 10.0 * ca2 * v4 * w
            + 14.0 * ca3 * v4 * w
            + 2.0 * v5 * w
            + 6.0 * ca * v5 * w
            + 10.0 * ca2 * v5 * w
            + 20.0 * ca3 * v5 * w
            + 20.0 * ca * v6 * w
            - 4.0 * ca2 * v6 * w
            - 20.0 * ca3 * v6 * w
            - 8.0 * ca * v7 * w
            + 8.0 * ca3 * v7 * w
            + ca * w2
            - ca3 * w2
            + 2.0 * v * w2
            - 4.0 * ca * v * w2
            + 2.0 * ca2 * v * w2
            + 2.0 * ca3 * v * w2
            - 6.0 * v2 * w2
            - 17.0 * ca * v2 * w2
            - 2.0 * ca2 * v2 * w2
            + ca3 * v2 * w2
            + 4.0 * v3 * w2
            + 28.0 * ca * v3 * w2
            + 46.0 * ca3 * v3 * w2
            - 4.0 * v4 * w2
            - 13.0 * ca * v4 * w2
            + 12.0 * ca2 * v4 * w2
            - 13.0 * ca3 * v4 * w2
            + 10.0 * v5 * w2
            + 72.0 * ca * v5 * w2
            - 22.0 * ca2 * v5 * w2
            - 88.0 * ca3 * v5 * w2
            - 6.0 * v6 * w2
            - 79.0 * ca * v6 * w2
            + 6.0 * ca2 * v6 * w2
            + 49.0 * ca3 * v6 * w2
            + 8.0 * ca * v7 * w2
            + 4.0 * ca2 * v7 * w2
            - 8.0 * ca3 * v7 * w2
            + 4.0 * ca * v8 * w2
            - 4.0 * ca3 * v8 * w2
            + 6.0 * ca * v2 * w3
            - 4.0 * ca2 * v2 * w3
            - 6.0 * ca3 * v2 * w3
            - 6.0 * ca * v3 * w3
            + 4.0 * ca2 * v3 * w3
            - 12.0 * ca3 * v3 * w3
            + 12.0 * v4 * w3
            - 12.0 * ca2 * v4 * w3
            - 32.0 * ca3 * v4 * w3
            - 24.0 * v5 * w3
            - 92.0 * ca * v5 * w3
            + 24.0 * ca2 * v5 * w3
            + 126.0 * ca3 * v5 * w3
            + 8.0 * v6 * w3
            + 74.0 * ca * v6 * w3
            + 4.0 * ca2 * v6 * w3
            - 22.0 * ca3 * v6 * w3
            + 4.0 * v7 * w3
            + 30.0 * ca * v7 * w3
            - 16.0 * ca2 * v7 * w3
            - 18.0 * ca3 * v7 * w3
            - 12.0 * ca * v8 * w3
            + 12.0 * ca3 * v8 * w3
            - 2.0 * ca * v2 * w4
            + 2.0 * ca3 * v2 * w4
            + 6.0 * ca * v3 * w4
            - 2.0 * ca3 * v3 * w4
            - 8.0 * v4 * w4
            - 2.0 * ca * v4 * w4
            + 8.0 * ca2 * v4 * w4
            + 20.0 * ca3 * v4 * w4
            + 12.0 * v5 * w4
            + 64.0 * ca * v5 * w4
            - 20.0 * ca2 * v5 * w4
            - 66.0 * ca3 * v5 * w4
            + 8.0 * v6 * w4
            - 8.0 * ca * v6 * w4
            - 16.0 * ca2 * v6 * w4
            - 40.0 * ca3 * v6 * w4
            - 12.0 * v7 * w4
            - 66.0 * ca * v7 * w4
            + 28.0 * ca2 * v7 * w4
            + 38.0 * ca3 * v7 * w4
            + 16.0 * ca * v8 * w4
            - 16.0 * ca3 * v8 * w4
            + 2.0 * v4 * w5
            - 4.0 * ca * v4 * w5
            - 2.0 * ca2 * v4 * w5
            + 4.0 * ca3 * v4 * w5
            + 2.0 * v5 * w5
            - 16.0 * ca * v5 * w5
            + 10.0 * ca2 * v5 * w5
            + 8.0 * ca3 * v5 * w5
            - 16.0 * v6 * w5
            - 38.0 * ca * v6 * w5
            + 20.0 * ca2 * v6 * w5
            + 50.0 * ca3 * v6 * w5
            + 12.0 * v7 * w5
            + 50.0 * ca * v7 * w5
            - 28.0 * ca2 * v7 * w5
            - 22.0 * ca3 * v7 * w5
            - 16.0 * ca * v8 * w5
            + 16.0 * ca3 * v8 * w5
            + ca * v4 * w6
            - ca3 * v4 * w6
            - 2.0 * v5 * w6
            - 2.0 * ca * v5 * w6
            - 2.0 * ca2 * v5 * w6
            + 6.0 * v6 * w6
            + 23.0 * ca * v6 * w6
            - 14.0 * ca2 * v6 * w6
            - 17.0 * ca3 * v6 * w6
            - 4.0 * v7 * w6
            - 10.0 * ca * v7 * w6
            + 16.0 * ca2 * v7 * w6
            - 2.0 * ca3 * v7 * w6
            + 16.0 * ca * v8 * w6
            - 16.0 * ca3 * v8 * w6
            + 4.0 * ca2 * v6 * w7
            - 4.0 * ca * v7 * w7
            - 4.0 * ca2 * v7 * w7
            + 4.0 * ca3 * v7 * w7
            - 12.0 * ca * v8 * w7
            + 12.0 * ca3 * v8 * w7
            + 4.0 * ca * v8 * w8
            - 4.0 * ca3 * v8 * w8))
        / (ca * (1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2));

    let part8 = -(2.0
        * cf
        * lv
        * (4.0 * ca - 16.0 * ca3 - 16.0 * ca * v + 44.0 * ca3 * v + 28.0 * ca * v2
            - 60.0 * ca3 * v2
            - 32.0 * ca * v3
            + 64.0 * ca3 * v3
            + 28.0 * ca * v4
            - 48.0 * ca3 * v4
            - 16.0 * ca * v5
            + 20.0 * ca3 * v5
            + 4.0 * ca * v6
            - 4.0 * ca3 * v6
            - 2.0 * ca * w
            + 2.0 * ca3 * w
            - 2.0 * v * w
            + 22.0 * ca * v * w
            + 6.0 * ca2 * v * w
            + 14.0 * ca3 * v * w
            + 6.0 * v2 * w
            - 36.0 * ca * v2 * w
            - 18.0 * ca2 * v2 * w
            - 44.0 * ca3 * v2 * w
            - 8.0 * v3 * w
            + 32.0 * ca * v3 * w
            + 24.0 * ca2 * v3 * w
            + 8.0 * ca3 * v3 * w
            + 8.0 * v4 * w
            - 30.0 * ca * v4 * w
            - 24.0 * ca2 * v4 * w
            + 6.0 * ca3 * v4 * w
            - 6.0 * v5 * w
            + 2.0 * ca * v5 * w
            + 18.0 * ca2 * v5 * w
            + 34.0 * ca3 * v5 * w
            + 2.0 * v6 * w
            + 20.0 * ca * v6 * w
            - 6.0 * ca2 * v6 * w
            - 28.0 * ca3 * v6 * w
            - 8.0 * ca * v7 * w
            + 8.0 * ca3 * v7 * w
            + ca * w2
            - ca3 * w2
            - 4.0 * ca * v * w2
            + 2.0 * ca3 * v * w2
            - 4.0 * v2 * w2
            - 29.0 * ca * v2 * w2
            + 3.0 * ca3 * v2 * w2
            + 8.0 * v3 * w2
            + 56.0 * ca * v3 * w2
            - 4.0 * ca2 * v3 * w2
            + 52.0 * ca3 * v3 * w2
            - 12.0 * v4 * w2
            - 25.0 * ca * v4 * w2
            + 20.0 * ca2 * v4 * w2
            - 7.0 * ca3 * v4 * w2
            + 12.0 * v5 * w2
            + 60.0 * ca * v5 * w2
            - 20.0 * ca2 * v5 * w2
            - 114.0 * ca3 * v5 * w2
            - 71.0 * ca * v6 * w2
            - 4.0 * ca2 * v6 * w2
            + 57.0 * ca3 * v6 * w2
            - 4.0 * v7 * w2
            + 8.0 * ca * v7 * w2
            + 8.0 * ca2 * v7 * w2
            - 4.0 * ca3 * v7 * w2
            + 4.0 * ca * v8 * w2
            - 4.0 * ca3 * v8 * w2
            + 6.0 * ca * v2 * w3
            - 6.0 * ca3 * v2 * w3
            + 2.0 * v3 * w3
            + 2.0 * ca * v3 * w3
            - 2.0 * ca2 * v3 * w3
            - 20.0 * ca3 * v3 * w3
            + 2.0 * v4 * w3
            - 32.0 * ca * v4 * w3
            - 2.0 * ca2 * v4 * w3
            - 40.0 * ca3 * v4 * w3
            - 4.0 * v5 * w3
            - 64.0 * ca * v5 * w3
            - 12.0 * ca2 * v5 * w3
            + 148.0 * ca3 * v5 * w3
            - 16.0 * v6 * w3
            + 74.0 * ca * v6 * w3
            + 40.0 * ca2 * v6 * w3
            - 10.0 * ca3 * v6 * w3
            + 14.0 * v7 * w3
            + 26.0 * ca * v7 * w3
            - 22.0 * ca2 * v7 * w3
            - 28.0 * ca3 * v7 * w3
            + 2.0 * v8 * w3
            - 12.0 * ca * v8 * w3
            - 2.0 * ca2 * v8 * w3
            + 12.0 * ca3 * v8 * w3
            - 2.0 * ca * v2 * w4
            + 2.0 * ca3 * v2 * w4
            + 6.0 * ca * v3 * w4
            - 2.0 * ca3 * v3 * w4
            + 6.0 * ca * v4 * w4
            + 28.0 * ca3 * v4 * w4
            - 4.0 * v5 * w4
            + 72.0 * ca * v5 * w4
            + 20.0 * ca2 * v5 * w4
            - 74.0 * ca3 * v5 * w4
            + 28.0 * v6 * w4
            - 28.0 * ca * v6 * w4
            - 44.0 * ca2 * v6 * w4
            - 70.0 * ca3 * v6 * w4
            - 16.0 * v7 * w4
            - 62.0 * ca * v7 * w4
            + 16.0 * ca2 * v7 * w4
            + 44.0 * ca3 * v7 * w4
            - 8.0 * v8 * w4
            + 16.0 * ca * v8 * w4
            + 8.0 * ca2 * v8 * w4
            - 16.0 * ca3 * v8 * w4
            - 4.0 * ca * v4 * w5
            + 4.0 * ca3 * v4 * w5
            + 2.0 * v5 * w5
            - 28.0 * ca * v5 * w5
            - 6.0 * ca2 * v5 * w5
            + 6.0 * ca3 * v5 * w5
            - 18.0 * v6 * w5
            - 30.0 * ca * v6 * w5
            + 14.0 * ca2 * v6 * w5
            + 74.0 * ca3 * v6 * w5
            + 4.0 * v7 * w5
            + 54.0 * ca * v7 * w5
            + 4.0 * ca2 * v7 * w5
            - 16.0 * ca3 * v7 * w5
            + 12.0 * v8 * w5
            - 16.0 * ca * v8 * w5
            - 12.0 * ca2 * v8 * w5
            + 16.0 * ca3 * v8 * w5
            + ca * v4 * w6
            - ca3 * v4 * w6
            - 2.0 * ca * v5 * w6
            + 4.0 * v6 * w6
            + 27.0 * ca * v6 * w6
            - 23.0 * ca3 * v6 * w6
            + 4.0 * v7 * w6
            - 14.0 * ca * v7 * w6
            - 8.0 * ca2 * v7 * w6
            - 12.0 * ca3 * v7 * w6
            - 8.0 * v8 * w6
            + 16.0 * ca * v8 * w6
            + 8.0 * ca2 * v8 * w6
            - 16.0 * ca3 * v8 * w6
            - 2.0 * v7 * w7
            - 4.0 * ca * v7 * w7
            + 2.0 * ca2 * v7 * w7
            + 8.0 * ca3 * v7 * w7
            + 2.0 * v8 * w7
            - 12.0 * ca * v8 * w7
            - 2.0 * ca2 * v8 * w7
            + 12.0 * ca3 * v8 * w7
            + 4.0 * ca * v8 * w8
            - 4.0 * ca3 * v8 * w8))
        / (ca * (1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2));

    let part9 = -(2.0
        * cf
        * (2.0 * ca - 6.0 * ca3 - 12.0 * ca * v + 28.0 * ca3 * v + 30.0 * ca * v2
            - 54.0 * ca3 * v2
            - 40.0 * ca * v3
            + 56.0 * ca3 * v3
            + 30.0 * ca * v4
            - 34.0 * ca3 * v4
            - 12.0 * ca * v5
            + 12.0 * ca3 * v5
            + 2.0 * ca * v6
            - 2.0 * ca3 * v6
            + ca * w
            - ca3 * w
            - 2.0 * v * w
            - ca * v * w
            + 2.0 * ca2 * v * w
            + ca3 * v * w
            + 10.0 * v2 * w
            - 2.0 * ca * v2 * w
            - 10.0 * ca2 * v2 * w
            - 6.0 * ca3 * v2 * w
            - 20.0 * v3 * w
            - 4.0 * ca * v3 * w
            + 20.0 * ca2 * v3 * w
            + 28.0 * ca3 * v3 * w
            + 20.0 * v4 * w
            + 23.0 * ca * v4 * w
            - 20.0 * ca2 * v4 * w
            - 47.0 * ca3 * v4 * w
            - 10.0 * v5 * w
            - 31.0 * ca * v5 * w
            + 10.0 * ca2 * v5 * w
            + 39.0 * ca3 * v5 * w
            + 2.0 * v6 * w
            + 18.0 * ca * v6 * w
            - 2.0 * ca2 * v6 * w
            - 18.0 * ca3 * v6 * w
            - 4.0 * ca * v7 * w
            + 4.0 * ca3 * v7 * w
            - ca * w2
            + ca3 * w2
            + 9.0 * ca * v * w2
            - 7.0 * ca3 * v * w2
            - 4.0 * v2 * w2
            - 32.0 * ca * v2 * w2
            + 4.0 * ca2 * v2 * w2
            + 22.0 * ca3 * v2 * w2
            + 12.0 * v3 * w2
            + 34.0 * ca * v3 * w2
            - 12.0 * ca2 * v3 * w2
            - 38.0 * ca3 * v3 * w2
            - 8.0 * v4 * w2
            - 11.0 * ca * v4 * w2
            + 8.0 * ca2 * v4 * w2
            + 39.0 * ca3 * v4 * w2
            - 8.0 * v5 * w2
            + 13.0 * ca * v5 * w2
            + 8.0 * ca2 * v5 * w2
            - 27.0 * ca3 * v5 * w2
            + 12.0 * v6 * w2
            - 14.0 * ca * v6 * w2
            - 12.0 * ca2 * v6 * w2
            + 12.0 * ca3 * v6 * w2
            - 4.0 * v7 * w2
            + 4.0 * ca2 * v7 * w2
            + 2.0 * ca * v8 * w2
            - 2.0 * ca3 * v8 * w2
            - 4.0 * ca * v2 * w3
            + 4.0 * ca3 * v2 * w3
            + 2.0 * v3 * w3
            + 58.0 * ca * v3 * w3
            - 2.0 * ca2 * v3 * w3
            - 28.0 * ca3 * v3 * w3
            - 18.0 * v4 * w3
            - 80.0 * ca * v4 * w3
            + 18.0 * ca2 * v4 * w3
            + 32.0 * ca3 * v4 * w3
            + 40.0 * v5 * w3
            - 4.0 * ca * v5 * w3
            - 40.0 * ca2 * v5 * w3
            - 32.0 * v6 * w3
            + 32.0 * ca * v6 * w3
            + 32.0 * ca2 * v6 * w3
            - 8.0 * ca3 * v6 * w3
            + 6.0 * v7 * w3
            + 4.0 * ca * v7 * w3
            - 6.0 * ca2 * v7 * w3
            - 6.0 * ca3 * v7 * w3
            + 2.0 * v8 * w3
            - 6.0 * ca * v8 * w3
            - 2.0 * ca2 * v8 * w3
            + 6.0 * ca3 * v8 * w3
            + 2.0 * ca * v2 * w4
            - 2.0 * ca3 * v2 * w4
            - 16.0 * ca * v3 * w4
            + 12.0 * ca3 * v3 * w4
            + 8.0 * v4 * w4
            - 12.0 * ca * v4 * w4
            - 8.0 * ca2 * v4 * w4
            + 2.0 * ca3 * v4 * w4
            - 24.0 * v5 * w4
            + 70.0 * ca * v5 * w4
            + 24.0 * ca2 * v5 * w4
            - 32.0 * ca3 * v5 * w4
            + 16.0 * v6 * w4
            - 44.0 * ca * v6 * w4
            - 16.0 * ca2 * v6 * w4
            + 18.0 * ca3 * v6 * w4
            + 8.0 * v7 * w4
            - 10.0 * ca * v7 * w4
            - 8.0 * ca2 * v7 * w4
            + 8.0 * ca3 * v7 * w4
            - 8.0 * v8 * w4
            + 10.0 * ca * v8 * w4
            + 8.0 * ca2 * v8 * w4
            - 10.0 * ca3 * v8 * w4
            + 3.0 * ca * v4 * w5
            - 3.0 * ca3 * v4 * w5
            + 2.0 * v5 * w5
            - 11.0 * ca * v5 * w5
            - 2.0 * ca2 * v5 * w5
            + 13.0 * ca3 * v5 * w5
            + 6.0 * v6 * w5
            + 10.0 * ca * v6 * w5
            - 6.0 * ca2 * v6 * w5
            - 6.0 * ca3 * v6 * w5
            - 20.0 * v7 * w5
            + 10.0 * ca * v7 * w5
            + 20.0 * ca2 * v7 * w5
            - 8.0 * ca3 * v7 * w5
            + 12.0 * v8 * w5
            - 12.0 * ca * v8 * w5
            - 12.0 * ca2 * v8 * w5
            + 12.0 * ca3 * v8 * w5
            - ca * v4 * w6
            + ca3 * v4 * w6
            + 7.0 * ca * v5 * w6
            - 5.0 * ca3 * v5 * w6
            - 4.0 * v6 * w6
            - 12.0 * ca * v6 * w6
            + 4.0 * ca2 * v6 * w6
            + 4.0 * ca3 * v6 * w6
            + 12.0 * v7 * w6
            - 2.0 * ca * v7 * w6
            - 12.0 * ca2 * v7 * w6
            + 4.0 * ca3 * v7 * w6
            - 8.0 * v8 * w6
            + 10.0 * ca * v8 * w6
            + 8.0 * ca2 * v8 * w6
            - 10.0 * ca3 * v8 * w6
            - 2.0 * v7 * w7
            + 2.0 * ca * v7 * w7
            + 2.0 * ca2 * v7 * w7
            - 2.0 * ca3 * v7 * w7
            + 2.0 * v8 * w7
            - 6.0 * ca * v8 * w7
            - 2.0 * ca2 * v8 * w7
            + 6.0 * ca3 * v8 * w7
            + 2.0 * ca * v8 * w8
            - 2.0 * ca3 * v8 * w8))
        / (ca * (1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2));

    part1 + part2 + part3 + part4 + part5 + part6 + part7 + part8 + part9
}

/// `STRUV8(W,V,X3,S)`.
#[must_use]
pub fn struv8(w: f64, v: f64, _x3: f64, _s: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let cf = ctx.cf;
    let ca = ctx.ca;
    let (v2, v3, v4, v5, v6, v7, v8) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8);
    let (w2, w3, w4, w5, w6, w7, w8) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8);
    let ca2 = pre.ca2;
    let cacf = ca * cf;
    let (l1v, lw, lvw, l1vw, lms, l1w, lv, lmss) = (
        pre.l1v, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.l1w, pre.lv, pre.lmss,
    );

    let part1 = (-2.0
        * cf
        * lmss
        * (2.0 - 4.0 * v + 2.0 * v2 + 2.0 * v * w - 2.0 * v2 * w + v2 * w2)
        * (1.0 - 2.0 * v + 2.0 * v2 + 2.0 * v * w - 4.0 * v2 * w + 2.0 * v2 * w2)
        * (2.0 * ca2 - 4.0 * ca2 * v + 2.0 * ca2 * v2 + 2.0 * ca2 * v * w
            - 2.0 * ca2 * v2 * w
            - v2 * w2
            + ca2 * v2 * w2))
        / ((1.0 - v) * v2 * w2 * (1.0 - v + v * w).powi(2));

    let part2 = -(4.0
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
        / (v * w * (1.0 - v + v * w).powi(2));

    let part3 = -(8.0
        * cf
        * l1v
        * (ca2 - 2.0 * ca2 * v + 2.0 * ca2 * v2 - 3.0 * v * w + ca2 * v * w + 6.0 * v2 * w
            - 3.0 * ca2 * v2 * w
            - 8.0 * v3 * w
            - 2.0 * v2 * w2
            + 2.0 * ca2 * v2 * w2
            + 10.0 * v3 * w2
            - ca2 * v3 * w2
            + 2.0 * ca2 * v4 * w2
            - 5.0 * v3 * w3
            - 2.0 * ca2 * v4 * w3
            + ca2 * v4 * w4))
        / (v2 * w2 * (1.0 - v * w));

    let part4 = (8.0
        * cf
        * lvw
        * (1.0 - w)
        * (2.0 * cacf + 2.0 * v - 2.0 * ca2 * v - 2.0 * v2
            + 2.0 * ca2 * v2
            + 2.0 * v * w
            + ca2 * v * w
            - 5.0 * v2 * w
            - 4.0 * ca2 * v2 * w
            + 4.0 * v3 * w
            + 2.0 * ca2 * v3 * w
            + 3.0 * ca2 * v2 * w2
            + v3 * w2
            - 4.0 * ca2 * v3 * w2
            - 2.0 * v4 * w2
            + 2.0 * ca2 * v4 * w2
            - v3 * w3
            + ca2 * v3 * w3
            + 2.0 * v4 * w3
            - 2.0 * ca2 * v4 * w3
            - v4 * w4
            + ca2 * v4 * w4))
        / ((1.0 - v) * v2 * w2);

    let part5 = -(4.0
        * cf
        * lw
        * (2.0 - 4.0 * ca2 - 4.0 * v + 14.0 * ca2 * v + 4.0 * v2 - 24.0 * ca2 * v2
            + 20.0 * ca2 * v3
            - 8.0 * ca2 * v4
            - 2.0 * v * w
            - 2.0 * ca2 * v * w
            + 2.0 * v2 * w
            + 10.0 * ca2 * v2 * w
            - 4.0 * v3 * w
            - 4.0 * ca2 * v3 * w
            - 8.0 * ca2 * v4 * w
            + 8.0 * ca2 * v5 * w
            + 5.0 * v2 * w2
            - 3.0 * ca2 * v2 * w2
            - 10.0 * v3 * w2
            - 4.0 * ca2 * v3 * w2
            + 8.0 * v4 * w2
            + 16.0 * ca2 * v4 * w2
            - 12.0 * ca2 * v5 * w2
            + 3.0 * v3 * w3
            + 3.0 * ca2 * v3 * w3
            - 4.0 * v4 * w3
            - 10.0 * ca2 * v4 * w3
            + 8.0 * ca2 * v5 * w3
            + 2.0 * ca2 * v4 * w4
            - 2.0 * ca2 * v5 * w4))
        / ((1.0 - v) * v2 * w2 * (1.0 - v * w));

    let part6 = (2.0
        * cf
        * lms
        * (4.0 - 4.0 * ca2 - 8.0 * v + 12.0 * ca2 * v + 8.0 * v2 - 20.0 * ca2 * v2
            + 16.0 * ca2 * v3
            - 8.0 * ca2 * v4
            - 2.0 * w
            + 2.0 * ca2 * w
            - 2.0 * v * w
            - 2.0 * ca2 * v * w
            + 4.0 * v2 * w
            - 12.0 * v3 * w
            + 20.0 * ca2 * v3 * w
            - 24.0 * ca2 * v4 * w
            + 16.0 * ca2 * v5 * w
            + w2
            - ca2 * w2
            + 2.0 * v * w2
            + 2.0 * ca2 * v2 * w2
            + 4.0 * v3 * w2
            - 16.0 * ca2 * v3 * w2
            + 8.0 * v4 * w2
            + 8.0 * ca2 * v4 * w2
            - 8.0 * ca2 * v6 * w2
            - 2.0 * v * w3
            + 2.0 * ca2 * v * w3
            + 2.0 * v2 * w3
            - 6.0 * ca2 * v2 * w3
            - 4.0 * v3 * w3
            + 12.0 * ca2 * v3 * w3
            - 4.0 * v4 * w3
            - 4.0 * v5 * w3
            - 4.0 * ca2 * v5 * w3
            + 8.0 * ca2 * v6 * w3
            + v2 * w4
            - ca2 * v2 * w4
            - 2.0 * v3 * w4
            + 4.0 * ca2 * v3 * w4
            + 6.0 * v4 * w4
            - 12.0 * ca2 * v4 * w4
            + 8.0 * ca2 * v5 * w4
            + 4.0 * v6 * w4
            - 8.0 * ca2 * v6 * w4
            - 2.0 * v5 * w5
            + 2.0 * ca2 * v5 * w5
            - 4.0 * v6 * w5
            + 4.0 * ca2 * v6 * w5
            + 2.0 * v6 * w6
            - 2.0 * ca2 * v6 * w6))
        / ((1.0 - v) * v2 * w2 * (1.0 - v * w).powi(2));

    let part7 = -(2.0
        * cf
        * l1w
        * (4.0 - 12.0 * ca2 - 16.0 * v + 64.0 * ca2 * v + 28.0 * v2
            - 156.0 * ca2 * v2
            - 24.0 * v3
            + 216.0 * ca2 * v3
            + 8.0 * v4
            - 176.0 * ca2 * v4
            + 80.0 * ca2 * v5
            - 16.0 * ca2 * v6
            - 2.0 * w
            + 2.0 * ca2 * w
            + 22.0 * v * w
            - 30.0 * ca2 * v * w
            - 78.0 * v2 * w
            + 114.0 * ca2 * v2 * w
            + 150.0 * v3 * w
            - 178.0 * ca2 * v3 * w
            - 168.0 * v4 * w
            + 84.0 * ca2 * v4 * w
            + 108.0 * v5 * w
            + 72.0 * ca2 * v5 * w
            - 32.0 * v6 * w
            - 96.0 * ca2 * v6 * w
            + 32.0 * ca2 * v7 * w
            + w2
            - ca2 * w2
            - 4.0 * v * w2
            + 6.0 * ca2 * v * w2
            + 23.0 * v2 * w2
            - 23.0 * ca2 * v2 * w2
            - 98.0 * v3 * w2
            + 12.0 * ca2 * v3 * w2
            + 178.0 * v4 * w2
            + 138.0 * ca2 * v4 * w2
            - 124.0 * v5 * w2
            - 292.0 * ca2 * v5 * w2
            - 4.0 * v6 * w2
            + 204.0 * ca2 * v6 * w2
            + 32.0 * v7 * w2
            - 32.0 * ca2 * v7 * w2
            - 16.0 * ca2 * v8 * w2
            + 6.0 * v2 * w3
            - 6.0 * ca2 * v2 * w3
            + 2.0 * v3 * w3
            + 40.0 * ca2 * v3 * w3
            - 36.0 * v4 * w3
            - 158.0 * ca2 * v4 * w3
            - 40.0 * v5 * w3
            + 260.0 * ca2 * v5 * w3
            + 172.0 * v6 * w3
            - 132.0 * ca2 * v6 * w3
            - 116.0 * v7 * w3
            - 40.0 * ca2 * v7 * w3
            + 48.0 * ca2 * v8 * w3
            - 2.0 * v2 * w4
            + 2.0 * ca2 * v2 * w4
            + 6.0 * v3 * w4
            - 10.0 * ca2 * v3 * w4
            - 2.0 * v4 * w4
            + 40.0 * ca2 * v4 * w4
            + 84.0 * v5 * w4
            - 74.0 * ca2 * v5 * w4
            - 222.0 * v6 * w4
            - 2.0 * ca2 * v6 * w4
            + 144.0 * v7 * w4
            + 92.0 * ca2 * v7 * w4
            + 8.0 * v8 * w4
            - 64.0 * ca2 * v8 * w4
            - 4.0 * v4 * w5
            + 4.0 * ca2 * v4 * w5
            - 24.0 * v5 * w5
            - 10.0 * ca2 * v5 * w5
            + 106.0 * v6 * w5
            + 44.0 * ca2 * v6 * w5
            - 70.0 * v7 * w5
            - 78.0 * ca2 * v7 * w5
            - 24.0 * v8 * w5
            + 56.0 * ca2 * v8 * w5
            + v4 * w6
            - ca2 * v4 * w6
            - 2.0 * v5 * w6
            + 4.0 * ca2 * v5 * w6
            - 17.0 * v6 * w6
            - 13.0 * ca2 * v6 * w6
            + 6.0 * v7 * w6
            + 30.0 * ca2 * v7 * w6
            + 28.0 * v8 * w6
            - 36.0 * ca2 * v8 * w6
            + 4.0 * v7 * w7
            - 4.0 * ca2 * v7 * w7
            - 16.0 * v8 * w7
            + 16.0 * ca2 * v8 * w7
            + 4.0 * v8 * w8
            - 4.0 * ca2 * v8 * w8))
        / ((1.0 - v) * v2 * w2 * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2));

    let part8 = -(2.0
        * cf
        * lv
        * (4.0 - 16.0 * ca2 - 16.0 * v + 84.0 * ca2 * v + 28.0 * v2
            - 200.0 * ca2 * v2
            - 24.0 * v3
            + 268.0 * ca2 * v3
            + 8.0 * v4
            - 208.0 * ca2 * v4
            + 88.0 * ca2 * v5
            - 16.0 * ca2 * v6
            - 2.0 * w
            + 2.0 * ca2 * w
            + 26.0 * v * w
            - 38.0 * ca2 * v * w
            - 98.0 * v2 * w
            + 150.0 * ca2 * v2 * w
            + 186.0 * v3 * w
            - 238.0 * ca2 * v3 * w
            - 196.0 * v4 * w
            + 120.0 * ca2 * v4 * w
            + 116.0 * v5 * w
            + 76.0 * ca2 * v5 * w
            - 32.0 * v6 * w
            - 104.0 * ca2 * v6 * w
            + 32.0 * ca2 * v7 * w
            + w2
            - ca2 * w2
            - 4.0 * v * w2
            + 6.0 * ca2 * v * w2
            + 35.0 * v2 * w2
            - 27.0 * ca2 * v2 * w2
            - 142.0 * v3 * w2
            + 16.0 * ca2 * v3 * w2
            + 230.0 * v4 * w2
            + 178.0 * ca2 * v4 * w2
            - 144.0 * v5 * w2
            - 380.0 * ca2 * v5 * w2
            - 4.0 * v6 * w2
            + 260.0 * ca2 * v6 * w2
            + 32.0 * v7 * w2
            - 40.0 * ca2 * v7 * w2
            - 16.0 * ca2 * v8 * w2
            + 6.0 * v2 * w3
            - 6.0 * ca2 * v2 * w3
            + 10.0 * v3 * w3
            + 44.0 * ca2 * v3 * w3
            - 44.0 * v4 * w3
            - 202.0 * ca2 * v4 * w3
            - 60.0 * v5 * w3
            + 360.0 * ca2 * v5 * w3
            + 200.0 * v6 * w3
            - 196.0 * ca2 * v6 * w3
            - 124.0 * v7 * w3
            - 44.0 * ca2 * v7 * w3
            + 56.0 * ca2 * v8 * w3
            - 2.0 * v2 * w4
            + 2.0 * ca2 * v2 * w4
            + 6.0 * v3 * w4
            - 10.0 * ca2 * v3 * w4
            - 10.0 * v4 * w4
            + 52.0 * ca2 * v4 * w4
            + 124.0 * v5 * w4
            - 118.0 * ca2 * v5 * w4
            - 274.0 * v6 * w4
            + 18.0 * ca2 * v6 * w4
            + 164.0 * v7 * w4
            + 128.0 * ca2 * v7 * w4
            + 8.0 * v8 * w4
            - 88.0 * ca2 * v8 * w4
            - 4.0 * v4 * w5
            + 4.0 * ca2 * v4 * w5
            - 36.0 * v5 * w5
            - 2.0 * ca2 * v5 * w5
            + 134.0 * v6 * w5
            + 48.0 * ca2 * v6 * w5
            - 86.0 * v7 * w5
            - 118.0 * ca2 * v7 * w5
            - 24.0 * v8 * w5
            + 84.0 * ca2 * v8 * w5
            + v4 * w6
            - ca2 * v4 * w6
            - 2.0 * v5 * w6
            + 4.0 * ca2 * v5 * w6
            - 21.0 * v6 * w6
            - 17.0 * ca2 * v6 * w6
            + 10.0 * v7 * w6
            + 50.0 * ca2 * v7 * w6
            + 28.0 * v8 * w6
            - 52.0 * ca2 * v8 * w6
            + 4.0 * v7 * w7
            - 8.0 * ca2 * v7 * w7
            - 16.0 * v8 * w7
            + 20.0 * ca2 * v8 * w7
            + 4.0 * v8 * w8
            - 4.0 * ca2 * v8 * w8))
        / ((1.0 - v) * v2 * w2 * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2));

    let part9 = -(2.0
        * cf
        * (4.0 * v - 12.0 * ca2 * v - 12.0 * v2 + 52.0 * ca2 * v2 + 12.0 * v3
            - 92.0 * ca2 * v3
            - 4.0 * v4
            + 84.0 * ca2 * v4
            - 40.0 * ca2 * v5
            + 8.0 * ca2 * v6
            + w
            - ca2 * w
            - 9.0 * v * w
            + 17.0 * ca2 * v * w
            + 27.0 * v2 * w
            - 75.0 * ca2 * v2 * w
            - 29.0 * v3 * w
            + 125.0 * ca2 * v3 * w
            + 8.0 * v4 * w
            - 72.0 * ca2 * v4 * w
            + 2.0 * v5 * w
            - 26.0 * ca2 * v5 * w
            + 48.0 * ca2 * v6 * w
            - 16.0 * ca2 * v7 * w
            - w2
            + ca2 * w2
            + 7.0 * v * w2
            - 3.0 * ca2 * v * w2
            - 17.0 * v2 * w2
            + 11.0 * ca2 * v2 * w2
            + v3 * w2
            + 7.0 * ca2 * v3 * w2
            + 34.0 * v4 * w2
            - 112.0 * ca2 * v4 * w2
            - 28.0 * v5 * w2
            + 204.0 * ca2 * v5 * w2
            + 4.0 * v6 * w2
            - 140.0 * ca2 * v6 * w2
            + 24.0 * ca2 * v7 * w2
            + 8.0 * ca2 * v8 * w2
            - 4.0 * v2 * w3
            + 4.0 * ca2 * v2 * w3
            + 34.0 * v3 * w3
            - 24.0 * ca2 * v3 * w3
            - 76.0 * v4 * w3
            + 86.0 * ca2 * v4 * w3
            + 68.0 * v5 * w3
            - 144.0 * ca2 * v5 * w3
            - 32.0 * v6 * w3
            + 80.0 * ca2 * v6 * w3
            + 10.0 * v7 * w3
            + 30.0 * ca2 * v7 * w3
            - 32.0 * ca2 * v8 * w3
            + 2.0 * v2 * w4
            - 2.0 * ca2 * v2 * w4
            - 12.0 * v3 * w4
            + 4.0 * ca2 * v3 * w4
            + 27.0 * v4 * w4
            - 5.0 * ca2 * v4 * w4
            - 36.0 * v5 * w4
            + 20.0 * ca2 * v5 * w4
            + 36.0 * v6 * w4
            + 10.0 * ca2 * v6 * w4
            - 8.0 * v7 * w4
            - 84.0 * ca2 * v7 * w4
            - 8.0 * v8 * w4
            + 56.0 * ca2 * v8 * w4
            + 3.0 * v4 * w5
            - 3.0 * ca2 * v4 * w5
            - 5.0 * v5 * w5
            - ca2 * v5 * w5
            - 3.0 * v6 * w5
            - 15.0 * ca2 * v6 * w5
            - 19.0 * v7 * w5
            + 79.0 * ca2 * v7 * w5
            + 24.0 * v8 * w5
            - 60.0 * ca2 * v8 * w5
            - v4 * w6
            + ca2 * v4 * w6
            + 5.0 * v5 * w6
            - ca2 * v5 * w6
            - 6.0 * v6 * w6
            + 6.0 * ca2 * v6 * w6
            + 23.0 * v7 * w6
            - 43.0 * ca2 * v7 * w6
            - 26.0 * v8 * w6
            + 42.0 * ca2 * v8 * w6
            - 6.0 * v7 * w7
            + 10.0 * ca2 * v7 * w7
            + 12.0 * v8 * w7
            - 16.0 * ca2 * v8 * w7
            - 2.0 * v8 * w8
            + 2.0 * ca2 * v8 * w8))
        / ((1.0 - v) * v2 * w2 * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2));

    part1 + part2 + part3 + part4 + part5 + part6 + part7 + part8 + part9
}

/// `STRUV9(W,V,X3,S)`.
#[must_use]
pub fn struv9(w: f64, v: f64, _x3: f64, _s: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let cf = ctx.cf;
    let ca = ctx.ca;
    let (v2, v3, v4, v5, v6, v7, v8) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8);
    let (w2, w3, w4, w5, w6, w7, w8) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8);
    let ca2 = pre.ca2;
    let (l1v, lw, lvw, l1vw, lms, l1w, lv, lmss) = (
        pre.l1v, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.l1w, pre.lv, pre.lmss,
    );

    let part1 = (-2.0
        * cf
        * lmss
        * (2.0 - 4.0 * v + 2.0 * v2 + 2.0 * v * w - 2.0 * v2 * w + v2 * w2)
        * (1.0 - 2.0 * v + 2.0 * v2 + 2.0 * v * w - 4.0 * v2 * w + 2.0 * v2 * w2)
        * (2.0 * ca2 - 4.0 * ca2 * v + 2.0 * ca2 * v2 + 2.0 * ca2 * v * w
            - 2.0 * ca2 * v2 * w
            - v2 * w2
            + ca2 * v2 * w2))
        / ((1.0 - v) * v2 * w2 * (1.0 - v + v * w).powi(2));

    let part2 = (4.0
        * cf
        * l1vw
        * (1.0 - w)
        * (2.0 - ca2 - 8.0 * v + 2.0 * ca2 * v + 10.0 * v2 - ca2 * v2 - 4.0 * v3 + 7.0 * v * w
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
        / (v * w * (1.0 - v + v * w).powi(2));

    let part3 = (16.0
        * ca
        * cf.powi(2)
        * lvw
        * (1.0 - w)
        * (1.0 - 2.0 * v + 2.0 * v2 + 2.0 * v * w - 7.0 * v2 * w + 4.0 * v3 * w + 4.0 * v2 * w2
            - 5.0 * v3 * w2
            + 2.0 * v4 * w2
            + v3 * w3
            - 2.0 * v4 * w3
            + v4 * w4))
        / ((1.0 - v) * v2 * w2);

    let part4 = -(4.0
        * cf
        * l1v
        * (2.0 * ca2 - 4.0 * ca2 * v + 4.0 * ca2 * v2 + 6.0 * v * w - ca2 * v * w - 12.0 * v2 * w
            + 16.0 * v3 * w
            - 8.0 * ca2 * v3 * w
            + 4.0 * v2 * w2
            + 2.0 * ca2 * v2 * w2
            - 20.0 * v3 * w2
            + 8.0 * ca2 * v3 * w2
            + 4.0 * ca2 * v4 * w2
            + 10.0 * v3 * w3
            - 5.0 * ca2 * v3 * w3
            - 4.0 * ca2 * v4 * w3
            + 2.0 * ca2 * v4 * w4))
        / (v2 * w2 * (1.0 - v * w));

    let part5 = -(4.0
        * cf
        * lw
        * (2.0 - 4.0 * ca2 - 4.0 * v + 14.0 * ca2 * v + 4.0 * v2 - 24.0 * ca2 * v2
            + 20.0 * ca2 * v3
            - 8.0 * ca2 * v4
            - 2.0 * v * w
            - 2.0 * ca2 * v * w
            + 2.0 * v2 * w
            + 10.0 * ca2 * v2 * w
            - 4.0 * v3 * w
            - 4.0 * ca2 * v3 * w
            - 8.0 * ca2 * v4 * w
            + 8.0 * ca2 * v5 * w
            - 3.0 * v2 * w2
            - ca2 * v2 * w2
            + 14.0 * v3 * w2
            - 10.0 * ca2 * v3 * w2
            - 8.0 * v4 * w2
            + 20.0 * ca2 * v4 * w2
            - 12.0 * ca2 * v5 * w2
            - 5.0 * v3 * w3
            + 5.0 * ca2 * v3 * w3
            + 4.0 * v4 * w3
            - 12.0 * ca2 * v4 * w3
            + 8.0 * ca2 * v5 * w3
            + 2.0 * ca2 * v4 * w4
            - 2.0 * ca2 * v5 * w4))
        / ((1.0 - v) * v2 * w2 * (1.0 - v * w));

    let part6 = (2.0
        * cf
        * lms
        * (4.0 - 4.0 * ca2 - 8.0 * v + 12.0 * ca2 * v + 8.0 * v2 - 20.0 * ca2 * v2
            + 16.0 * ca2 * v3
            - 8.0 * ca2 * v4
            - 2.0 * w
            + 2.0 * ca2 * w
            - 2.0 * v * w
            - 2.0 * ca2 * v * w
            + 4.0 * v2 * w
            - 12.0 * v3 * w
            + 20.0 * ca2 * v3 * w
            - 24.0 * ca2 * v4 * w
            + 16.0 * ca2 * v5 * w
            + w2
            - ca2 * w2
            + 2.0 * v * w2
            + 2.0 * ca2 * v2 * w2
            + 4.0 * v3 * w2
            - 16.0 * ca2 * v3 * w2
            + 8.0 * v4 * w2
            + 8.0 * ca2 * v4 * w2
            - 8.0 * ca2 * v6 * w2
            - 2.0 * v * w3
            + 2.0 * ca2 * v * w3
            + 2.0 * v2 * w3
            - 6.0 * ca2 * v2 * w3
            - 4.0 * v3 * w3
            + 12.0 * ca2 * v3 * w3
            - 4.0 * v4 * w3
            - 4.0 * v5 * w3
            - 4.0 * ca2 * v5 * w3
            + 8.0 * ca2 * v6 * w3
            + v2 * w4
            - ca2 * v2 * w4
            - 2.0 * v3 * w4
            + 4.0 * ca2 * v3 * w4
            + 6.0 * v4 * w4
            - 12.0 * ca2 * v4 * w4
            + 8.0 * ca2 * v5 * w4
            + 4.0 * v6 * w4
            - 8.0 * ca2 * v6 * w4
            - 2.0 * v5 * w5
            + 2.0 * ca2 * v5 * w5
            - 4.0 * v6 * w5
            + 4.0 * ca2 * v6 * w5
            + 2.0 * v6 * w6
            - 2.0 * ca2 * v6 * w6))
        / ((1.0 - v) * v2 * w2 * (1.0 - v * w).powi(2));

    let part7 = -(2.0
        * cf
        * l1w
        * (4.0 - 12.0 * ca2 - 16.0 * v + 64.0 * ca2 * v + 28.0 * v2
            - 156.0 * ca2 * v2
            - 24.0 * v3
            + 216.0 * ca2 * v3
            + 8.0 * v4
            - 176.0 * ca2 * v4
            + 80.0 * ca2 * v5
            - 16.0 * ca2 * v6
            - 2.0 * w
            + 2.0 * ca2 * w
            - 2.0 * v * w
            - 24.0 * ca2 * v * w
            + 42.0 * v2 * w
            + 84.0 * ca2 * v2 * w
            - 130.0 * v3 * w
            - 108.0 * ca2 * v3 * w
            + 192.0 * v4 * w
            - 6.0 * ca2 * v4 * w
            - 132.0 * v5 * w
            + 132.0 * ca2 * v5 * w
            + 32.0 * v6 * w
            - 112.0 * ca2 * v6 * w
            + 32.0 * ca2 * v7 * w
            + w2
            - ca2 * w2
            - 4.0 * v * w2
            + 6.0 * ca2 * v * w2
            - 17.0 * v2 * w2
            - 13.0 * ca2 * v2 * w2
            + 102.0 * v3 * w2
            - 38.0 * ca2 * v3 * w2
            - 198.0 * v4 * w2
            + 232.0 * ca2 * v4 * w2
            + 124.0 * v5 * w2
            - 354.0 * ca2 * v5 * w2
            + 28.0 * v6 * w2
            + 196.0 * ca2 * v6 * w2
            - 32.0 * v7 * w2
            - 16.0 * ca2 * v7 * w2
            - 16.0 * ca2 * v8 * w2
            + 6.0 * v2 * w3
            - 6.0 * ca2 * v2 * w3
            - 30.0 * v3 * w3
            + 48.0 * ca2 * v3 * w3
            + 60.0 * v4 * w3
            - 182.0 * ca2 * v4 * w3
            + 32.0 * v5 * w3
            + 242.0 * ca2 * v5 * w3
            - 172.0 * v6 * w3
            - 46.0 * ca2 * v6 * w3
            + 92.0 * v7 * w3
            - 92.0 * ca2 * v7 * w3
            + 48.0 * ca2 * v8 * w3
            - 2.0 * v2 * w4
            + 2.0 * ca2 * v2 * w4
            + 6.0 * v3 * w4
            - 10.0 * ca2 * v3 * w4
            - 2.0 * v4 * w4
            + 40.0 * ca2 * v4 * w4
            - 76.0 * v5 * w4
            - 34.0 * ca2 * v5 * w4
            + 202.0 * v6 * w4
            - 108.0 * ca2 * v6 * w4
            - 120.0 * v7 * w4
            + 158.0 * ca2 * v7 * w4
            + 8.0 * v8 * w4
            - 64.0 * ca2 * v8 * w4
            - 4.0 * v4 * w5
            + 4.0 * ca2 * v4 * w5
            + 32.0 * v5 * w5
            - 24.0 * ca2 * v5 * w5
            - 110.0 * v6 * w5
            + 98.0 * ca2 * v6 * w5
            + 90.0 * v7 * w5
            - 118.0 * ca2 * v7 * w5
            - 24.0 * v8 * w5
            + 56.0 * ca2 * v8 * w5
            + v4 * w6
            - ca2 * v4 * w6
            - 2.0 * v5 * w6
            + 4.0 * ca2 * v5 * w6
            + 23.0 * v6 * w6
            - 23.0 * ca2 * v6 * w6
            - 34.0 * v7 * w6
            + 40.0 * ca2 * v7 * w6
            + 28.0 * v8 * w6
            - 36.0 * ca2 * v8 * w6
            + 4.0 * v7 * w7
            - 4.0 * ca2 * v7 * w7
            - 16.0 * v8 * w7
            + 16.0 * ca2 * v8 * w7
            + 4.0 * v8 * w8
            - 4.0 * ca2 * v8 * w8))
        / ((1.0 - v) * v2 * w2 * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2));

    let part8 = -(2.0
        * cf
        * lv
        * (4.0 - 16.0 * ca2 - 16.0 * v + 84.0 * ca2 * v + 28.0 * v2
            - 200.0 * ca2 * v2
            - 24.0 * v3
            + 268.0 * ca2 * v3
            + 8.0 * v4
            - 208.0 * ca2 * v4
            + 88.0 * ca2 * v5
            - 16.0 * ca2 * v6
            - 2.0 * w
            + 2.0 * ca2 * w
            - 6.0 * v * w
            - 30.0 * ca2 * v * w
            + 62.0 * v2 * w
            + 110.0 * ca2 * v2 * w
            - 166.0 * v3 * w
            - 150.0 * ca2 * v3 * w
            + 220.0 * v4 * w
            + 16.0 * ca2 * v4 * w
            - 140.0 * v5 * w
            + 140.0 * ca2 * v5 * w
            + 32.0 * v6 * w
            - 120.0 * ca2 * v6 * w
            + 32.0 * ca2 * v7 * w
            + w2
            - ca2 * w2
            - 4.0 * v * w2
            + 6.0 * ca2 * v * w2
            - 29.0 * v2 * w2
            - 11.0 * ca2 * v2 * w2
            + 146.0 * v3 * w2
            - 56.0 * ca2 * v3 * w2
            - 250.0 * v4 * w2
            + 298.0 * ca2 * v4 * w2
            + 144.0 * v5 * w2
            - 452.0 * ca2 * v5 * w2
            + 28.0 * v6 * w2
            + 252.0 * ca2 * v6 * w2
            - 32.0 * v7 * w2
            - 24.0 * ca2 * v7 * w2
            - 16.0 * ca2 * v8 * w2
            + 6.0 * v2 * w3
            - 6.0 * ca2 * v2 * w3
            - 38.0 * v3 * w3
            + 56.0 * ca2 * v3 * w3
            + 68.0 * v4 * w3
            - 230.0 * ca2 * v4 * w3
            + 52.0 * v5 * w3
            + 332.0 * ca2 * v5 * w3
            - 200.0 * v6 * w3
            - 96.0 * ca2 * v6 * w3
            + 100.0 * v7 * w3
            - 100.0 * ca2 * v7 * w3
            + 56.0 * ca2 * v8 * w3
            - 2.0 * v2 * w4
            + 2.0 * ca2 * v2 * w4
            + 6.0 * v3 * w4
            - 10.0 * ca2 * v3 * w4
            + 6.0 * v4 * w4
            + 48.0 * ca2 * v4 * w4
            - 116.0 * v5 * w4
            - 58.0 * ca2 * v5 * w4
            + 254.0 * v6 * w4
            - 114.0 * ca2 * v6 * w4
            - 140.0 * v7 * w4
            + 204.0 * ca2 * v7 * w4
            + 8.0 * v8 * w4
            - 88.0 * ca2 * v8 * w4
            - 4.0 * v4 * w5
            + 4.0 * ca2 * v4 * w5
            + 44.0 * v5 * w5
            - 22.0 * ca2 * v5 * w5
            - 138.0 * v6 * w5
            + 116.0 * ca2 * v6 * w5
            + 106.0 * v7 * w5
            - 166.0 * ca2 * v7 * w5
            - 24.0 * v8 * w5
            + 84.0 * ca2 * v8 * w5
            + v4 * w6
            - ca2 * v4 * w6
            - 2.0 * v5 * w6
            + 4.0 * ca2 * v5 * w6
            + 27.0 * v6 * w6
            - 29.0 * ca2 * v6 * w6
            - 38.0 * v7 * w6
            + 62.0 * ca2 * v7 * w6
            + 28.0 * v8 * w6
            - 52.0 * ca2 * v8 * w6
            + 4.0 * v7 * w7
            - 8.0 * ca2 * v7 * w7
            - 16.0 * v8 * w7
            + 20.0 * ca2 * v8 * w7
            + 4.0 * v8 * w8
            - 4.0 * ca2 * v8 * w8))
        / ((1.0 - v) * v2 * w2 * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2));

    let part9 = -(2.0
        * cf
        * (4.0 * v - 12.0 * ca2 * v - 12.0 * v2 + 52.0 * ca2 * v2 + 12.0 * v3
            - 92.0 * ca2 * v3
            - 4.0 * v4
            + 84.0 * ca2 * v4
            - 40.0 * ca2 * v5
            + 8.0 * ca2 * v6
            + w
            - ca2 * w
            - 9.0 * v * w
            + 17.0 * ca2 * v * w
            + 27.0 * v2 * w
            - 75.0 * ca2 * v2 * w
            - 29.0 * v3 * w
            + 125.0 * ca2 * v3 * w
            + 8.0 * v4 * w
            - 72.0 * ca2 * v4 * w
            + 2.0 * v5 * w
            - 26.0 * ca2 * v5 * w
            + 48.0 * ca2 * v6 * w
            - 16.0 * ca2 * v7 * w
            - w2
            + ca2 * w2
            - v * w2
            - ca2 * v * w2
            - v2 * w2
            + 7.0 * ca2 * v2 * w2
            + 41.0 * v3 * w2
            - 3.0 * ca2 * v3 * w2
            - 94.0 * v4 * w2
            - 80.0 * ca2 * v4 * w2
            + 84.0 * v5 * w2
            + 176.0 * ca2 * v5 * w2
            - 28.0 * v6 * w2
            - 132.0 * ca2 * v6 * w2
            + 24.0 * ca2 * v7 * w2
            + 8.0 * ca2 * v8 * w2
            - 4.0 * v2 * w3
            + 4.0 * ca2 * v2 * w3
            - 38.0 * v3 * w3
            - 6.0 * ca2 * v3 * w3
            + 156.0 * v4 * w3
            + 28.0 * ca2 * v4 * w3
            - 172.0 * v5 * w3
            - 84.0 * ca2 * v5 * w3
            + 48.0 * v6 * w3
            + 60.0 * ca2 * v6 * w3
            + 10.0 * v7 * w3
            + 30.0 * ca2 * v7 * w3
            - 32.0 * ca2 * v8 * w3
            + 2.0 * v2 * w4
            - 2.0 * ca2 * v2 * w4
            + 4.0 * v3 * w4
            - 61.0 * v4 * w4
            + 17.0 * ca2 * v4 * w4
            + 84.0 * v5 * w4
            - 10.0 * ca2 * v5 * w4
            - 12.0 * v6 * w4
            + 22.0 * ca2 * v6 * w4
            - 8.0 * v7 * w4
            - 84.0 * ca2 * v7 * w4
            - 8.0 * v8 * w4
            + 56.0 * ca2 * v8 * w4
            + 3.0 * v4 * w5
            - 3.0 * ca2 * v4 * w5
            + 3.0 * v5 * w5
            - 3.0 * ca2 * v5 * w5
            - 11.0 * v6 * w5
            - 13.0 * ca2 * v6 * w5
            - 19.0 * v7 * w5
            + 79.0 * ca2 * v7 * w5
            + 24.0 * v8 * w5
            - 60.0 * ca2 * v8 * w5
            - v4 * w6
            + ca2 * v4 * w6
            - 3.0 * v5 * w6
            + ca2 * v5 * w6
            + 2.0 * v6 * w6
            + 4.0 * ca2 * v6 * w6
            + 23.0 * v7 * w6
            - 43.0 * ca2 * v7 * w6
            - 26.0 * v8 * w6
            + 42.0 * ca2 * v8 * w6
            - 6.0 * v7 * w7
            + 10.0 * ca2 * v7 * w7
            + 12.0 * v8 * w7
            - 16.0 * ca2 * v8 * w7
            - 2.0 * v8 * w8
            + 2.0 * ca2 * v8 * w8))
        / ((1.0 - v) * v2 * w2 * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2));

    part1 + part2 + part3 + part4 + part5 + part6 + part7 + part8 + part9
}

/// `STRUV10(W,V,X3,S)`.
#[must_use]
pub fn struv10(w: f64, v: f64, _x3: f64, _s: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let cf = ctx.cf;
    let ca = ctx.ca;
    let (v2, v3, v4, v5, v6, v7, v8) = (pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8);
    let (w2, w3, w4, w5, w6, w7, w8) = (pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8);
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let (l1v, lw, lvw, l1vw, lms, l1w, lv, lmss) = (
        pre.l1v, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.l1w, pre.lv, pre.lmss,
    );

    let part1 = (-2.0
        * cf
        * lmss
        * (2.0 - 4.0 * v + 2.0 * v2 + 2.0 * v * w - 2.0 * v2 * w + v2 * w2)
        * (1.0 - 2.0 * v + 2.0 * v2 + 2.0 * v * w - 4.0 * v2 * w + 2.0 * v2 * w2)
        * (2.0 * ca2 - 4.0 * ca2 * v + 2.0 * ca2 * v2 + 2.0 * ca2 * v * w
            - 2.0 * ca2 * v2 * w
            - v2 * w2
            + ca2 * v2 * w2))
        / ((1.0 - v) * v2 * w2 * (1.0 - v + v * w).powi(2));

    let part2 = (4.0
        * cf
        * l1vw
        * (1.0 - w)
        * (2.0 - ca2 - 8.0 * v + 2.0 * ca2 * v + 10.0 * v2 - ca2 * v2 - 4.0 * v3 + 7.0 * v * w
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
        / (v * w * (1.0 - v + v * w).powi(2));

    let part3 = (16.0
        * cf.powi(2)
        * lvw
        * (1.0 - w)
        * (ca - 2.0 * ca * v + 2.0 * ca * v2 + v * w + 2.0 * ca * v * w
            - 2.0 * v2 * w
            - 7.0 * ca * v2 * w
            + 2.0 * v3 * w
            + 4.0 * ca * v3 * w
            + 4.0 * ca * v2 * w2
            - 2.0 * v3 * w2
            - 5.0 * ca * v3 * w2
            + 2.0 * ca * v4 * w2
            + v3 * w3
            + ca * v3 * w3
            - 2.0 * ca * v4 * w3
            + ca * v4 * w4))
        / ((1.0 - v) * v2 * w2);

    let part4 = -(4.0
        * cf
        * lw
        * (2.0 * ca - 4.0 * ca3 - 6.0 * ca * v + 18.0 * ca3 * v + 8.0 * ca * v2
            - 38.0 * ca3 * v2
            - 4.0 * ca * v3
            + 44.0 * ca3 * v3
            - 28.0 * ca3 * v4
            + 8.0 * ca3 * v5
            + w
            - ca2 * w
            - 3.0 * v * w
            + 3.0 * ca2 * v * w
            - 6.0 * ca3 * v * w
            + 4.0 * v2 * w
            - 4.0 * ca2 * v2 * w
            + 26.0 * ca3 * v2 * w
            - 2.0 * v3 * w
            - 2.0 * ca * v3 * w
            + 2.0 * ca2 * v3 * w
            - 38.0 * ca3 * v3 * w
            + 4.0 * ca * v4 * w
            + 16.0 * ca3 * v4 * w
            + 8.0 * ca3 * v5 * w
            - 8.0 * ca3 * v6 * w
            + v * w2
            + ca2 * v * w2
            - 6.0 * v2 * w2
            - 5.0 * ca * v2 * w2
            - 4.0 * ca2 * v2 * w2
            - 3.0 * ca3 * v2 * w2
            + 10.0 * v3 * w2
            + 19.0 * ca * v3 * w2
            + 10.0 * ca2 * v3 * w2
            + ca3 * v3 * w2
            - 8.0 * v4 * w2
            - 26.0 * ca * v4 * w2
            - 12.0 * ca2 * v4 * w2
            + 26.0 * ca3 * v4 * w2
            + 4.0 * v5 * w2
            + 8.0 * ca * v5 * w2
            + 4.0 * ca2 * v5 * w2
            - 40.0 * ca3 * v5 * w2
            + 20.0 * ca3 * v6 * w2
            + v2 * w3
            + ca2 * v2 * w3
            - 4.0 * v3 * w3
            - 8.0 * ca * v3 * w3
            - 6.0 * ca2 * v3 * w3
            + 4.0 * ca3 * v3 * w3
            + 6.0 * v4 * w3
            + 23.0 * ca * v4 * w3
            + 14.0 * ca2 * v4 * w3
            - 27.0 * ca3 * v4 * w3
            - 6.0 * v5 * w3
            - 12.0 * ca * v5 * w3
            - 6.0 * ca2 * v5 * w3
            + 40.0 * ca3 * v5 * w3
            - 20.0 * ca3 * v6 * w3
            + v3 * w4
            + ca2 * v3 * w4
            - 2.0 * v4 * w4
            - 5.0 * ca * v4 * w4
            - 8.0 * ca2 * v4 * w4
            + 7.0 * ca3 * v4 * w4
            + 4.0 * v5 * w4
            + 4.0 * ca * v5 * w4
            + 4.0 * ca2 * v5 * w4
            - 16.0 * ca3 * v5 * w4
            + 10.0 * ca3 * v6 * w4
            + 2.0 * ca2 * v4 * w5
            - v5 * w5
            - ca2 * v5 * w5
            + 2.0 * ca3 * v5 * w5
            - 2.0 * ca3 * v6 * w5))
        / (ca * (1.0 - v) * v2 * w2 * (1.0 - v * w) * (1.0 - v + v * w));

    let part5 = -(4.0
        * cf
        * l1v
        * (2.0 * ca3 - 8.0 * ca3 * v + 14.0 * ca3 * v2 - 12.0 * ca3 * v3
            + 4.0 * ca3 * v4
            + w
            + ca2 * w
            - 5.0 * v * w
            + 6.0 * ca * v * w
            - 3.0 * ca2 * v * w
            + ca3 * v * w
            + 10.0 * v2 * w
            - 24.0 * ca * v2 * w
            + 4.0 * ca2 * v2 * w
            - 4.0 * ca3 * v2 * w
            - 10.0 * v3 * w
            + 46.0 * ca * v3 * w
            - 2.0 * ca2 * v3 * w
            - ca3 * v3 * w
            + 4.0 * v4 * w
            - 44.0 * ca * v4 * w
            + 12.0 * ca3 * v4 * w
            + 16.0 * ca * v5 * w
            - 8.0 * ca3 * v5 * w
            + v * w2
            + ca2 * v * w2
            - 4.0 * v2 * w2
            + 10.0 * ca * v2 * w2
            - 4.0 * ca2 * v2 * w2
            + ca3 * v2 * w2
            + 6.0 * v3 * w2
            - 46.0 * ca * v3 * w2
            + 4.0 * ca2 * v3 * w2
            + 5.0 * ca3 * v3 * w2
            + 72.0 * ca * v4 * w2
            - 18.0 * ca3 * v4 * w2
            - 4.0 * v5 * w2
            - 36.0 * ca * v5 * w2
            + 8.0 * ca3 * v5 * w2
            + 4.0 * ca3 * v6 * w2
            + v2 * w3
            + ca2 * v2 * w3
            - 4.0 * v3 * w3
            + 14.0 * ca * v3 * w3
            - 4.0 * ca2 * v3 * w3
            - 3.0 * ca3 * v3 * w3
            - 44.0 * ca * v4 * w3
            + 2.0 * ca2 * v4 * w3
            + 12.0 * ca3 * v4 * w3
            + 6.0 * v5 * w3
            + 30.0 * ca * v5 * w3
            - 2.0 * ca2 * v5 * w3
            - ca3 * v5 * w3
            - 8.0 * ca3 * v6 * w3
            + v3 * w4
            + ca2 * v3 * w4
            + 10.0 * ca * v4 * w4
            - 3.0 * ca3 * v4 * w4
            - 4.0 * v5 * w4
            - 10.0 * ca * v5 * w4
            + 2.0 * ca2 * v5 * w4
            - 3.0 * ca3 * v5 * w4
            + 6.0 * ca3 * v6 * w4
            + v5 * w5
            - ca2 * v5 * w5
            + 2.0 * ca3 * v5 * w5
            - 2.0 * ca3 * v6 * w5))
        / (ca * (1.0 - v) * v2 * w2 * (1.0 - v * w) * (1.0 - v + v * w));

    let part6 = (2.0
        * cf
        * lms
        * (4.0 * ca - 4.0 * ca3 - 8.0 * ca * v + 12.0 * ca3 * v + 8.0 * ca * v2 - 20.0 * ca3 * v2
            + 16.0 * ca3 * v3
            - 8.0 * ca3 * v4
            - 2.0 * ca * w
            + 2.0 * ca3 * w
            + 2.0 * v * w
            - 2.0 * ca * v * w
            - 2.0 * ca2 * v * w
            - 2.0 * ca3 * v * w
            - 4.0 * v2 * w
            + 4.0 * ca * v2 * w
            + 4.0 * ca2 * v2 * w
            + 4.0 * v3 * w
            - 12.0 * ca * v3 * w
            - 4.0 * ca2 * v3 * w
            + 20.0 * ca3 * v3 * w
            - 24.0 * ca3 * v4 * w
            + 16.0 * ca3 * v5 * w
            + ca * w2
            - ca3 * w2
            + 2.0 * ca * v * w2
            - 4.0 * v2 * w2
            + 4.0 * ca2 * v2 * w2
            + 2.0 * ca3 * v2 * w2
            + 4.0 * v3 * w2
            + 4.0 * ca * v3 * w2
            - 4.0 * ca2 * v3 * w2
            - 16.0 * ca3 * v3 * w2
            - 8.0 * v4 * w2
            + 8.0 * ca * v4 * w2
            + 8.0 * ca2 * v4 * w2
            + 8.0 * ca3 * v4 * w2
            - 8.0 * ca3 * v6 * w2
            - 2.0 * ca * v * w3
            + 2.0 * ca3 * v * w3
            + 2.0 * ca * v2 * w3
            - 6.0 * ca3 * v2 * w3
            + 4.0 * v3 * w3
            - 4.0 * ca * v3 * w3
            - 4.0 * ca2 * v3 * w3
            + 12.0 * ca3 * v3 * w3
            + 4.0 * v4 * w3
            - 4.0 * ca * v4 * w3
            - 4.0 * ca2 * v4 * w3
            + 4.0 * v5 * w3
            - 4.0 * ca * v5 * w3
            - 4.0 * ca2 * v5 * w3
            - 4.0 * ca3 * v5 * w3
            + 8.0 * ca3 * v6 * w3
            + ca * v2 * w4
            - ca3 * v2 * w4
            - 2.0 * ca * v3 * w4
            + 4.0 * ca3 * v3 * w4
            - 4.0 * v4 * w4
            + 6.0 * ca * v4 * w4
            + 4.0 * ca2 * v4 * w4
            - 12.0 * ca3 * v4 * w4
            - 4.0 * v5 * w4
            + 4.0 * ca2 * v5 * w4
            + 8.0 * ca3 * v5 * w4
            + 4.0 * ca * v6 * w4
            - 8.0 * ca3 * v6 * w4
            + 2.0 * v5 * w5
            - 2.0 * ca * v5 * w5
            - 2.0 * ca2 * v5 * w5
            + 2.0 * ca3 * v5 * w5
            - 4.0 * ca * v6 * w5
            + 4.0 * ca3 * v6 * w5
            + 2.0 * ca * v6 * w6
            - 2.0 * ca3 * v6 * w6))
        / (ca * (1.0 - v) * v2 * w2 * (1.0 - v * w).powi(2));

    let part7 = -(2.0
        * cf
        * l1w
        * (4.0 * ca - 12.0 * ca3 - 16.0 * ca * v + 64.0 * ca3 * v + 28.0 * ca * v2
            - 156.0 * ca3 * v2
            - 24.0 * ca * v3
            + 216.0 * ca3 * v3
            + 8.0 * ca * v4
            - 176.0 * ca3 * v4
            + 80.0 * ca3 * v5
            - 16.0 * ca3 * v6
            - 2.0 * w
            - 2.0 * ca * w
            + 2.0 * ca2 * w
            + 2.0 * ca3 * w
            + 10.0 * v * w
            - 2.0 * ca * v * w
            - 14.0 * ca2 * v * w
            - 24.0 * ca3 * v * w
            - 22.0 * v2 * w
            + 42.0 * ca * v2 * w
            + 38.0 * ca2 * v2 * w
            + 84.0 * ca3 * v2 * w
            + 26.0 * v3 * w
            - 130.0 * ca * v3 * w
            - 54.0 * ca2 * v3 * w
            - 108.0 * ca3 * v3 * w
            - 16.0 * v4 * w
            + 192.0 * ca * v4 * w
            + 40.0 * ca2 * v4 * w
            - 6.0 * ca3 * v4 * w
            + 4.0 * v5 * w
            - 132.0 * ca * v5 * w
            - 12.0 * ca2 * v5 * w
            + 132.0 * ca3 * v5 * w
            + 32.0 * ca * v6 * w
            - 112.0 * ca3 * v6 * w
            + 32.0 * ca3 * v7 * w
            + ca * w2
            - ca3 * w2
            - 2.0 * v * w2
            - 4.0 * ca * v * w2
            - 2.0 * ca2 * v * w2
            + 6.0 * ca3 * v * w2
            + 8.0 * v2 * w2
            - 17.0 * ca * v2 * w2
            + 12.0 * ca2 * v2 * w2
            - 13.0 * ca3 * v2 * w2
            - 10.0 * v3 * w2
            + 102.0 * ca * v3 * w2
            - 30.0 * ca2 * v3 * w2
            - 38.0 * ca3 * v3 * w2
            - 4.0 * v4 * w2
            - 198.0 * ca * v4 * w2
            + 52.0 * ca2 * v4 * w2
            + 232.0 * ca3 * v4 * w2
            + 16.0 * v5 * w2
            + 124.0 * ca * v5 * w2
            - 56.0 * ca2 * v5 * w2
            - 354.0 * ca3 * v5 * w2
            - 8.0 * v6 * w2
            + 28.0 * ca * v6 * w2
            + 24.0 * ca2 * v6 * w2
            + 196.0 * ca3 * v6 * w2
            - 32.0 * ca * v7 * w2
            - 16.0 * ca3 * v7 * w2
            - 16.0 * ca3 * v8 * w2
            + 6.0 * ca * v2 * w3
            - 4.0 * ca2 * v2 * w3
            - 6.0 * ca3 * v2 * w3
            - 30.0 * ca * v3 * w3
            + 20.0 * ca2 * v3 * w3
            + 48.0 * ca3 * v3 * w3
            + 12.0 * v4 * w3
            + 60.0 * ca * v4 * w3
            - 52.0 * ca2 * v4 * w3
            - 182.0 * ca3 * v4 * w3
            - 24.0 * v5 * w3
            + 32.0 * ca * v5 * w3
            + 64.0 * ca2 * v5 * w3
            + 242.0 * ca3 * v5 * w3
            + 8.0 * v6 * w3
            - 172.0 * ca * v6 * w3
            - 16.0 * ca2 * v6 * w3
            - 46.0 * ca3 * v6 * w3
            + 4.0 * v7 * w3
            + 92.0 * ca * v7 * w3
            - 12.0 * ca2 * v7 * w3
            - 92.0 * ca3 * v7 * w3
            + 48.0 * ca3 * v8 * w3
            - 2.0 * ca * v2 * w4
            + 2.0 * ca3 * v2 * w4
            + 6.0 * ca * v3 * w4
            - 10.0 * ca3 * v3 * w4
            - 8.0 * v4 * w4
            - 2.0 * ca * v4 * w4
            + 8.0 * ca2 * v4 * w4
            + 40.0 * ca3 * v4 * w4
            + 20.0 * v5 * w4
            - 76.0 * ca * v5 * w4
            - 12.0 * ca2 * v5 * w4
            - 34.0 * ca3 * v5 * w4
            - 4.0 * v6 * w4
            + 202.0 * ca * v6 * w4
            - 28.0 * ca2 * v6 * w4
            - 108.0 * ca3 * v6 * w4
            - 8.0 * v7 * w4
            - 120.0 * ca * v7 * w4
            + 32.0 * ca2 * v7 * w4
            + 158.0 * ca3 * v7 * w4
            + 8.0 * ca * v8 * w4
            - 64.0 * ca3 * v8 * w4
            + 2.0 * v4 * w5
            - 4.0 * ca * v4 * w5
            - 2.0 * ca2 * v4 * w5
            + 4.0 * ca3 * v4 * w5
            - 10.0 * v5 * w5
            + 32.0 * ca * v5 * w5
            - 2.0 * ca2 * v5 * w5
            - 24.0 * ca3 * v5 * w5
            + 2.0 * v6 * w5
            - 110.0 * ca * v6 * w5
            + 38.0 * ca2 * v6 * w5
            + 98.0 * ca3 * v6 * w5
            + 6.0 * v7 * w5
            + 90.0 * ca * v7 * w5
            - 34.0 * ca2 * v7 * w5
            - 118.0 * ca3 * v7 * w5
            - 24.0 * ca * v8 * w5
            + 56.0 * ca3 * v8 * w5
            + ca * v4 * w6
            - ca3 * v4 * w6
            + 2.0 * v5 * w6
            - 2.0 * ca * v5 * w6
            + 2.0 * ca2 * v5 * w6
            + 4.0 * ca3 * v5 * w6
            + 23.0 * ca * v6 * w6
            - 20.0 * ca2 * v6 * w6
            - 23.0 * ca3 * v6 * w6
            - 2.0 * v7 * w6
            - 34.0 * ca * v7 * w6
            + 18.0 * ca2 * v7 * w6
            + 40.0 * ca3 * v7 * w6
            + 28.0 * ca * v8 * w6
            - 36.0 * ca3 * v8 * w6
            + 4.0 * ca2 * v6 * w7
            + 4.0 * ca * v7 * w7
            - 4.0 * ca2 * v7 * w7
            - 4.0 * ca3 * v7 * w7
            - 16.0 * ca * v8 * w7
            + 16.0 * ca3 * v8 * w7
            + 4.0 * ca * v8 * w8
            - 4.0 * ca3 * v8 * w8))
        / (ca * (1.0 - v) * v2 * w2 * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2));

    let part8 = -(2.0
        * cf
        * lv
        * (4.0 * ca - 16.0 * ca3 - 16.0 * ca * v + 84.0 * ca3 * v + 28.0 * ca * v2
            - 200.0 * ca3 * v2
            - 24.0 * ca * v3
            + 268.0 * ca3 * v3
            + 8.0 * ca * v4
            - 208.0 * ca3 * v4
            + 88.0 * ca3 * v5
            - 16.0 * ca3 * v6
            - 2.0 * ca * w
            + 2.0 * ca3 * w
            + 2.0 * v * w
            - 6.0 * ca * v * w
            - 6.0 * ca2 * v * w
            - 30.0 * ca3 * v * w
            - 8.0 * v2 * w
            + 62.0 * ca * v2 * w
            + 24.0 * ca2 * v2 * w
            + 110.0 * ca3 * v2 * w
            + 14.0 * v3 * w
            - 166.0 * ca * v3 * w
            - 42.0 * ca2 * v3 * w
            - 150.0 * ca3 * v3 * w
            - 12.0 * v4 * w
            + 220.0 * ca * v4 * w
            + 36.0 * ca2 * v4 * w
            + 16.0 * ca3 * v4 * w
            + 4.0 * v5 * w
            - 140.0 * ca * v5 * w
            - 12.0 * ca2 * v5 * w
            + 140.0 * ca3 * v5 * w
            + 32.0 * ca * v6 * w
            - 120.0 * ca3 * v6 * w
            + 32.0 * ca3 * v7 * w
            + ca * w2
            - ca3 * w2
            - 4.0 * ca * v * w2
            + 6.0 * ca3 * v * w2
            - 4.0 * v2 * w2
            - 29.0 * ca * v2 * w2
            - 11.0 * ca3 * v2 * w2
            + 16.0 * v3 * w2
            + 146.0 * ca * v3 * w2
            + 4.0 * ca2 * v3 * w2
            - 56.0 * ca3 * v3 * w2
            - 32.0 * v4 * w2
            - 250.0 * ca * v4 * w2
            + 298.0 * ca3 * v4 * w2
            + 36.0 * v5 * w2
            + 144.0 * ca * v5 * w2
            - 20.0 * ca2 * v5 * w2
            - 452.0 * ca3 * v5 * w2
            - 16.0 * v6 * w2
            + 28.0 * ca * v6 * w2
            + 16.0 * ca2 * v6 * w2
            + 252.0 * ca3 * v6 * w2
            - 32.0 * ca * v7 * w2
            - 24.0 * ca3 * v7 * w2
            - 16.0 * ca3 * v8 * w2
            + 6.0 * ca * v2 * w3
            - 6.0 * ca3 * v2 * w3
            - 2.0 * v3 * w3
            - 38.0 * ca * v3 * w3
            + 2.0 * ca2 * v3 * w3
            + 56.0 * ca3 * v3 * w3
            + 12.0 * v4 * w3
            + 68.0 * ca * v4 * w3
            - 12.0 * ca2 * v4 * w3
            - 230.0 * ca3 * v4 * w3
            - 24.0 * v5 * w3
            + 52.0 * ca * v5 * w3
            + 40.0 * ca2 * v5 * w3
            + 332.0 * ca3 * v5 * w3
            + 4.0 * v6 * w3
            - 200.0 * ca * v6 * w3
            - 28.0 * ca2 * v6 * w3
            - 96.0 * ca3 * v6 * w3
            + 12.0 * v7 * w3
            + 100.0 * ca * v7 * w3
            - 4.0 * ca2 * v7 * w3
            - 100.0 * ca3 * v7 * w3
            + 56.0 * ca3 * v8 * w3
            - 2.0 * ca * v2 * w4
            + 2.0 * ca3 * v2 * w4
            + 6.0 * ca * v3 * w4
            - 10.0 * ca3 * v3 * w4
            + 6.0 * ca * v4 * w4
            + 48.0 * ca3 * v4 * w4
            + 4.0 * v5 * w4
            - 116.0 * ca * v5 * w4
            - 20.0 * ca2 * v5 * w4
            - 58.0 * ca3 * v5 * w4
            + 16.0 * v6 * w4
            + 254.0 * ca * v6 * w4
            + 16.0 * ca2 * v6 * w4
            - 114.0 * ca3 * v6 * w4
            - 28.0 * v7 * w4
            - 140.0 * ca * v7 * w4
            + 12.0 * ca2 * v7 * w4
            + 204.0 * ca3 * v7 * w4
            + 8.0 * ca * v8 * w4
            - 88.0 * ca3 * v8 * w4
            - 4.0 * ca * v4 * w5
            + 4.0 * ca3 * v4 * w5
            - 2.0 * v5 * w5
            + 44.0 * ca * v5 * w5
            + 6.0 * ca2 * v5 * w5
            - 22.0 * ca3 * v5 * w5
            - 12.0 * v6 * w5
            - 138.0 * ca * v6 * w5
            - 4.0 * ca2 * v6 * w5
            + 116.0 * ca3 * v6 * w5
            + 26.0 * v7 * w5
            + 106.0 * ca * v7 * w5
            - 14.0 * ca2 * v7 * w5
            - 166.0 * ca3 * v7 * w5
            - 24.0 * ca * v8 * w5
            + 84.0 * ca3 * v8 * w5
            + ca * v4 * w6
            - ca3 * v4 * w6
            - 2.0 * ca * v5 * w6
            + 4.0 * ca3 * v5 * w6
            + 4.0 * v6 * w6
            + 27.0 * ca * v6 * w6
            - 29.0 * ca3 * v6 * w6
            - 12.0 * v7 * w6
            - 38.0 * ca * v7 * w6
            + 8.0 * ca2 * v7 * w6
            + 62.0 * ca3 * v7 * w6
            + 28.0 * ca * v8 * w6
            - 52.0 * ca3 * v8 * w6
            + 2.0 * v7 * w7
            + 4.0 * ca * v7 * w7
            - 2.0 * ca2 * v7 * w7
            - 8.0 * ca3 * v7 * w7
            - 16.0 * ca * v8 * w7
            + 20.0 * ca3 * v8 * w7
            + 4.0 * ca * v8 * w8
            - 4.0 * ca3 * v8 * w8))
        / (ca * (1.0 - v) * v2 * w2 * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2));

    let part9 = -(2.0
        * cf
        * (4.0 * ca * v - 12.0 * ca3 * v - 12.0 * ca * v2 + 52.0 * ca3 * v2 + 12.0 * ca * v3
            - 92.0 * ca3 * v3
            - 4.0 * ca * v4
            + 84.0 * ca3 * v4
            - 40.0 * ca3 * v5
            + 8.0 * ca3 * v6
            + ca * w
            - ca3 * w
            - 9.0 * ca * v * w
            + 17.0 * ca3 * v * w
            + 4.0 * v2 * w
            + 27.0 * ca * v2 * w
            - 4.0 * ca2 * v2 * w
            - 75.0 * ca3 * v2 * w
            - 12.0 * v3 * w
            - 29.0 * ca * v3 * w
            + 12.0 * ca2 * v3 * w
            + 125.0 * ca3 * v3 * w
            + 12.0 * v4 * w
            + 8.0 * ca * v4 * w
            - 12.0 * ca2 * v4 * w
            - 72.0 * ca3 * v4 * w
            - 4.0 * v5 * w
            + 2.0 * ca * v5 * w
            + 4.0 * ca2 * v5 * w
            - 26.0 * ca3 * v5 * w
            + 48.0 * ca3 * v6 * w
            - 16.0 * ca3 * v7 * w
            - ca * w2
            + ca3 * w2
            - ca * v * w2
            - ca3 * v * w2
            - 4.0 * v2 * w2
            - ca * v2 * w2
            + 4.0 * ca2 * v2 * w2
            + 7.0 * ca3 * v2 * w2
            + 12.0 * v3 * w2
            + 41.0 * ca * v3 * w2
            - 12.0 * ca2 * v3 * w2
            - 3.0 * ca3 * v3 * w2
            - 4.0 * v4 * w2
            - 94.0 * ca * v4 * w2
            + 4.0 * ca2 * v4 * w2
            - 80.0 * ca3 * v4 * w2
            - 12.0 * v5 * w2
            + 84.0 * ca * v5 * w2
            + 12.0 * ca2 * v5 * w2
            + 176.0 * ca3 * v5 * w2
            + 8.0 * v6 * w2
            - 28.0 * ca * v6 * w2
            - 8.0 * ca2 * v6 * w2
            - 132.0 * ca3 * v6 * w2
            + 24.0 * ca3 * v7 * w2
            + 8.0 * ca3 * v8 * w2
            - 4.0 * ca * v2 * w3
            + 4.0 * ca3 * v2 * w3
            - 38.0 * ca * v3 * w3
            - 6.0 * ca3 * v3 * w3
            - 16.0 * v4 * w3
            + 156.0 * ca * v4 * w3
            + 16.0 * ca2 * v4 * w3
            + 28.0 * ca3 * v4 * w3
            + 32.0 * v5 * w3
            - 172.0 * ca * v5 * w3
            - 32.0 * ca2 * v5 * w3
            - 84.0 * ca3 * v5 * w3
            - 12.0 * v6 * w3
            + 48.0 * ca * v6 * w3
            + 12.0 * ca2 * v6 * w3
            + 60.0 * ca3 * v6 * w3
            - 4.0 * v7 * w3
            + 10.0 * ca * v7 * w3
            + 4.0 * ca2 * v7 * w3
            + 30.0 * ca3 * v7 * w3
            - 32.0 * ca3 * v8 * w3
            + 2.0 * ca * v2 * w4
            - 2.0 * ca3 * v2 * w4
            + 4.0 * ca * v3 * w4
            + 8.0 * v4 * w4
            - 61.0 * ca * v4 * w4
            - 8.0 * ca2 * v4 * w4
            + 17.0 * ca3 * v4 * w4
            - 16.0 * v5 * w4
            + 84.0 * ca * v5 * w4
            + 16.0 * ca2 * v5 * w4
            - 10.0 * ca3 * v5 * w4
            - 4.0 * v6 * w4
            - 12.0 * ca * v6 * w4
            + 4.0 * ca2 * v6 * w4
            + 22.0 * ca3 * v6 * w4
            + 12.0 * v7 * w4
            - 8.0 * ca * v7 * w4
            - 12.0 * ca2 * v7 * w4
            - 84.0 * ca3 * v7 * w4
            - 8.0 * ca * v8 * w4
            + 56.0 * ca3 * v8 * w4
            + 3.0 * ca * v4 * w5
            - 3.0 * ca3 * v4 * w5
            + 3.0 * ca * v5 * w5
            - 3.0 * ca3 * v5 * w5
            + 12.0 * v6 * w5
            - 11.0 * ca * v6 * w5
            - 12.0 * ca2 * v6 * w5
            - 13.0 * ca3 * v6 * w5
            - 12.0 * v7 * w5
            - 19.0 * ca * v7 * w5
            + 12.0 * ca2 * v7 * w5
            + 79.0 * ca3 * v7 * w5
            + 24.0 * ca * v8 * w5
            - 60.0 * ca3 * v8 * w5
            - ca * v4 * w6
            + ca3 * v4 * w6
            - 3.0 * ca * v5 * w6
            + ca3 * v5 * w6
            - 4.0 * v6 * w6
            + 2.0 * ca * v6 * w6
            + 4.0 * ca2 * v6 * w6
            + 4.0 * ca3 * v6 * w6
            + 4.0 * v7 * w6
            + 23.0 * ca * v7 * w6
            - 4.0 * ca2 * v7 * w6
            - 43.0 * ca3 * v7 * w6
            - 26.0 * ca * v8 * w6
            + 42.0 * ca3 * v8 * w6
            - 6.0 * ca * v7 * w7
            + 10.0 * ca3 * v7 * w7
            + 12.0 * ca * v8 * w7
            - 16.0 * ca3 * v8 * w7
            - 2.0 * ca * v8 * w8
            + 2.0 * ca3 * v8 * w8))
        / (ca * (1.0 - v) * v2 * w2 * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(2));

    part1 + part2 + part3 + part4 + part5 + part6 + part7 + part8 + part9
}

/// `STRUV11(W,V,X3,S)`.
#[must_use]
pub fn struv11(w: f64, v: f64, _x3: f64, _s: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let cf = ctx.cf;
    let ca = ctx.ca;
    let (v2, v3, v4, v5, v6, v7, v8, v9, v10) = (
        pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8, pre.v9, pre.v10,
    );
    let (w2, w3, w4, w5, w6, w7, w8, w9, w10) = (
        pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8, pre.w9, pre.w10,
    );
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let cacf = ca * cf;
    let ca2cf = ca2 * cf;
    let (l1v, lw, lvw, l1vw, lms, l1w, lv, lmss) = (
        pre.l1v, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.l1w, pre.lv, pre.lmss,
    );

    let part1 = (-4.0
        * cf
        * lvw
        * (ca - ca3 + ca * v2 - ca3 * v2 + 2.0 * w + 7.0 * ca * w
            - 7.0 * ca3 * w
            - 4.0 * v * w
            - 16.0 * ca * v * w
            + 13.0 * ca3 * v * w
            + 3.0 * v2 * w
            + 8.0 * ca * v2 * w
            - 3.0 * ca2 * v2 * w
            - 9.0 * ca3 * v2 * w
            - v3 * w
            - 3.0 * ca * v3 * w
            + 3.0 * ca2 * v3 * w
            + 3.0 * ca3 * v3 * w
            - ca3 * v * w2
            + 7.0 * ca * v2 * w2
            + ca2 * v2 * w2
            - 4.0 * ca3 * v2 * w2
            - 5.0 * ca * v3 * w2
            - ca2 * v3 * w2
            + 3.0 * ca3 * v3 * w2
            + 2.0 * ca * v4 * w2
            - 2.0 * ca3 * v4 * w2
            + 4.0 * ca * v2 * w3
            - 4.0 * ca2 * v2 * w3
            - 8.0 * ca3 * v2 * w3
            - 8.0 * ca * v3 * w3
            + 4.0 * ca2 * v3 * w3
            + 8.0 * ca3 * v3 * w3
            + 2.0 * ca * v4 * w4
            - 2.0 * ca3 * v4 * w4))
        / (ca * (1.0 - v).powi(2) * v * w);

    let part2 = -(4.0
        * cf
        * l1v
        * (v - ca2 * v - 2.0 * v2 + 2.0 * ca2 * v2 + v3 - ca2 * v3 + 4.0 * ca * w
            - 4.0 * ca3 * w
            - 6.0 * v * w
            - 30.0 * ca * v * w
            - 2.0 * ca2 * v * w
            + 10.0 * ca3 * v * w
            + 14.0 * v2 * w
            + 60.0 * ca * v2 * w
            + 4.0 * ca2 * v2 * w
            - 16.0 * ca3 * v2 * w
            - 9.0 * v3 * w
            - 48.0 * ca * v3 * w
            - 3.0 * ca2 * v3 * w
            + 14.0 * ca3 * v3 * w
            + v4 * w
            + 14.0 * ca * v4 * w
            + ca2 * v4 * w
            - 4.0 * ca3 * v4 * w
            - 2.0 * v * w2
            - 28.0 * ca * v * w2
            - 6.0 * ca2 * v * w2
            + 4.0 * v2 * w2
            + 68.0 * ca * v2 * w2
            + 12.0 * ca2 * v2 * w2
            - 10.0 * ca3 * v2 * w2
            - 6.0 * v3 * w2
            - 76.0 * ca * v3 * w2
            - 8.0 * ca2 * v3 * w2
            + 15.0 * ca3 * v3 * w2
            + 4.0 * v4 * w2
            + 48.0 * ca * v4 * w2
            + 2.0 * ca2 * v4 * w2
            - 13.0 * ca3 * v4 * w2
            - 14.0 * ca * v5 * w2
            + 4.0 * ca3 * v5 * w2
            + 2.0 * v2 * w3
            + 8.0 * ca * v2 * w3
            - 2.0 * ca3 * v2 * w3
            - 3.0 * v3 * w3
            - 18.0 * ca * v3 * w3
            - 2.0 * ca2 * v3 * w3
            + 3.0 * ca3 * v3 * w3
            + v4 * w3
            + 10.0 * ca * v4 * w3
            + 2.0 * ca2 * v4 * w3
            - ca3 * v4 * w3
            + 2.0 * v3 * w4
            + 2.0 * ca2 * v3 * w4
            - 2.0 * v4 * w4
            + 4.0 * ca * v4 * w4
            - 2.0 * ca2 * v4 * w4
            - 2.0 * ca * v5 * w4))
        / (ca * (1.0 - v).powi(2) * w * (1.0 - v * w) * (1.0 - v + v * w));

    let part3 = (4.0
        * cf
        * l1vw
        * (2.0 + 8.0 * ca - 8.0 * ca3 + 8.0 * ca2cf - 10.0 * v - 41.0 * ca * v + 42.0 * ca3 * v
            - 40.0 * ca2cf * v
            + 21.0 * v2
            + 89.0 * ca * v2
            - 3.0 * ca2 * v2
            - 96.0 * ca3 * v2
            + 84.0 * ca2cf * v2
            - 24.0 * v3
            - 106.0 * ca * v3
            + 12.0 * ca2 * v3
            + 124.0 * ca3 * v3
            - 96.0 * ca2cf * v3
            + 16.0 * v4
            + 74.0 * ca * v4
            - 18.0 * ca2 * v4
            - 96.0 * ca3 * v4
            + 64.0 * ca2cf * v4
            - 6.0 * v5
            - 29.0 * ca * v5
            + 12.0 * ca2 * v5
            + 42.0 * ca3 * v5
            - 24.0 * ca2cf * v5
            + v6
            + 5.0 * ca * v6
            - 3.0 * ca2 * v6
            - 8.0 * ca3 * v6
            + 4.0 * ca2cf * v6
            + 6.0 * v * w
            + 25.0 * ca * v * w
            - 22.0 * ca3 * v * w
            + 24.0 * ca2cf * v * w
            - 24.0 * v2 * w
            - 102.0 * ca * v2 * w
            + ca2 * v2 * w
            + 92.0 * ca3 * v2 * w
            - 104.0 * ca2cf * v2 * w
            + 39.0 * v3 * w
            + 170.0 * ca * v3 * w
            - 13.0 * ca2 * v3 * w
            - 160.0 * ca3 * v3 * w
            + 184.0 * ca2cf * v3 * w
            - 33.0 * v4 * w
            - 148.0 * ca * v4 * w
            + 33.0 * ca2 * v4 * w
            + 148.0 * ca3 * v4 * w
            - 168.0 * ca2cf * v4 * w
            + 15.0 * v5 * w
            + 69.0 * ca * v5 * w
            - 31.0 * ca2 * v5 * w
            - 74.0 * ca3 * v5 * w
            + 80.0 * ca2cf * v5 * w
            - 3.0 * v6 * w
            - 14.0 * ca * v6 * w
            + 10.0 * ca2 * v6 * w
            + 16.0 * ca3 * v6 * w
            - 16.0 * ca2cf * v6 * w
            + 6.0 * v2 * w2
            + 33.0 * ca * v2 * w2
            - 4.0 * ca2 * v2 * w2
            - 30.0 * ca3 * v2 * w2
            + 8.0 * cacf * v2 * w2
            + 52.0 * ca2cf * v2 * w2
            - 18.0 * v3 * w2
            - 101.0 * ca * v3 * w2
            + 19.0 * ca2 * v3 * w2
            + 94.0 * ca3 * v3 * w2
            - 24.0 * cacf * v3 * w2
            - 176.0 * ca2cf * v3 * w2
            + 21.0 * v4 * w2
            + 119.0 * ca * v4 * w2
            - 42.0 * ca2 * v4 * w2
            - 118.0 * ca3 * v4 * w2
            + 28.0 * cacf * v4 * w2
            + 232.0 * ca2cf * v4 * w2
            - 12.0 * v5 * w2
            - 67.0 * ca * v5 * w2
            + 43.0 * ca2 * v5 * w2
            + 74.0 * ca3 * v5 * w2
            - 16.0 * cacf * v5 * w2
            - 144.0 * ca2cf * v5 * w2
            + 3.0 * v6 * w2
            + 16.0 * ca * v6 * w2
            - 16.0 * ca2 * v6 * w2
            - 20.0 * ca3 * v6 * w2
            + 4.0 * cacf * v6 * w2
            + 36.0 * ca2cf * v6 * w2
            + 2.0 * v3 * w3
            + 29.0 * ca * v3 * w3
            - 12.0 * ca2 * v3 * w3
            - 32.0 * ca3 * v3 * w3
            + 16.0 * cacf * v3 * w3
            + 64.0 * ca2cf * v3 * w3
            - 4.0 * v4 * w3
            - 66.0 * ca * v4 * w3
            + 39.0 * ca2 * v4 * w3
            + 86.0 * ca3 * v4 * w3
            - 40.0 * cacf * v4 * w3
            - 168.0 * ca2cf * v4 * w3
            + 3.0 * v5 * w3
            + 53.0 * ca * v5 * w3
            - 45.0 * ca2 * v5 * w3
            - 84.0 * ca3 * v5 * w3
            + 36.0 * cacf * v5 * w3
            + 156.0 * ca2cf * v5 * w3
            - v6 * w3
            - 16.0 * ca * v6 * w3
            + 18.0 * ca2 * v6 * w3
            + 30.0 * ca3 * v6 * w3
            - 12.0 * cacf * v6 * w3
            - 52.0 * ca2cf * v6 * w3
            + 21.0 * ca * v4 * w4
            - 12.0 * ca2 * v4 * w4
            - 28.0 * ca3 * v4 * w4
            + 12.0 * cacf * v4 * w4
            + 48.0 * ca2cf * v4 * w4
            - 36.0 * ca * v5 * w4
            + 25.0 * ca2 * v5 * w4
            + 52.0 * ca3 * v5 * w4
            - 24.0 * cacf * v5 * w4
            - 88.0 * ca2cf * v5 * w4
            + 17.0 * ca * v6 * w4
            - 13.0 * ca2 * v6 * w4
            - 26.0 * ca3 * v6 * w4
            + 12.0 * cacf * v6 * w4
            + 44.0 * ca2cf * v6 * w4
            + 10.0 * ca * v5 * w5
            - 4.0 * ca2 * v5 * w5
            - 10.0 * ca3 * v5 * w5
            + 4.0 * cacf * v5 * w5
            + 20.0 * ca2cf * v5 * w5
            - 10.0 * ca * v6 * w5
            + 4.0 * ca2 * v6 * w5
            + 10.0 * ca3 * v6 * w5
            - 4.0 * cacf * v6 * w5
            - 20.0 * ca2cf * v6 * w5
            + 2.0 * ca * v6 * w6
            - 2.0 * ca3 * v6 * w6
            + 4.0 * ca2cf * v6 * w6))
        / (ca * (1.0 - v).powi(2) * v * (1.0 - v + v * w).powi(3));

    let part4 = -(2.0
        * cf
        * lmss
        * (2.0 * cacf + 2.0 * v - 2.0 * ca2 * v - 10.0 * cacf * v - 11.0 * v2
            + 11.0 * ca2 * v2
            + 22.0 * cacf * v2
            + 27.0 * v3
            - 27.0 * ca2 * v3
            - 26.0 * cacf * v3
            - 37.0 * v4
            + 37.0 * ca2 * v4
            + 16.0 * cacf * v4
            + 29.0 * v5
            - 29.0 * ca2 * v5
            - 4.0 * cacf * v5
            - 12.0 * v6
            + 12.0 * ca2 * v6
            + 2.0 * v7
            - 2.0 * ca2 * v7
            - 2.0 * v * w
            + 4.0 * cacf * v * w
            + 16.0 * v2 * w
            - 4.0 * ca2 * v2 * w
            - 8.0 * cf * v2 * w
            - 40.0 * cacf * v2 * w
            - 54.0 * v3 * w
            + 22.0 * ca2 * v3 * w
            + 32.0 * cf * v3 * w
            + 116.0 * cacf * v3 * w
            + 94.0 * v4 * w
            - 46.0 * ca2 * v4 * w
            - 48.0 * cf * v4 * w
            - 156.0 * cacf * v4 * w
            - 88.0 * v5 * w
            + 46.0 * ca2 * v5 * w
            + 32.0 * cf * v5 * w
            + 112.0 * cacf * v5 * w
            + 42.0 * v6 * w
            - 22.0 * ca2 * v6 * w
            - 8.0 * cf * v6 * w
            - 44.0 * cacf * v6 * w
            - 8.0 * v7 * w
            + 4.0 * ca2 * v7 * w
            + 8.0 * cacf * v7 * w
            - 5.0 * v2 * w2
            + ca2 * v2 * w2
            - 8.0 * cf * v2 * w2
            - 22.0 * cacf * v2 * w2
            + 35.0 * v3 * w2
            - 11.0 * ca2 * v3 * w2
            + 8.0 * cf * v3 * w2
            + 30.0 * cacf * v3 * w2
            - 88.0 * v4 * w2
            + 32.0 * ca2 * v4 * w2
            + 24.0 * cf * v4 * w2
            + 36.0 * cacf * v4 * w2
            + 106.0 * v5 * w2
            - 42.0 * ca2 * v5 * w2
            - 40.0 * cf * v5 * w2
            - 88.0 * cacf * v5 * w2
            - 62.0 * v6 * w2
            + 26.0 * ca2 * v6 * w2
            + 16.0 * cf * v6 * w2
            + 60.0 * cacf * v6 * w2
            + 14.0 * v7 * w2
            - 6.0 * ca2 * v7 * w2
            - 16.0 * cacf * v7 * w2
            - 8.0 * v3 * w3
            + 2.0 * ca2 * v3 * w3
            - 16.0 * cf * v3 * w3
            - 52.0 * cacf * v3 * w3
            + 38.0 * v4 * w3
            - 14.0 * ca2 * v4 * w3
            + 28.0 * cf * v4 * w3
            + 92.0 * cacf * v4 * w3
            - 70.0 * v5 * w3
            + 32.0 * ca2 * v5 * w3
            - 8.0 * cf * v5 * w3
            - 48.0 * cacf * v5 * w3
            + 56.0 * v6 * w3
            - 28.0 * ca2 * v6 * w3
            - 4.0 * cf * v6 * w3
            - 16.0 * v7 * w3
            + 8.0 * ca2 * v7 * w3
            + 8.0 * cacf * v7 * w3
            - 7.0 * v4 * w4
            + 3.0 * ca2 * v4 * w4
            - 12.0 * cf * v4 * w4
            - 44.0 * cacf * v4 * w4
            + 29.0 * v5 * w4
            - 13.0 * ca2 * v5 * w4
            + 20.0 * cf * v5 * w4
            + 64.0 * cacf * v5 * w4
            - 36.0 * v6 * w4
            + 16.0 * ca2 * v6 * w4
            - 8.0 * cf * v6 * w4
            - 28.0 * cacf * v6 * w4
            + 14.0 * v7 * w4
            - 6.0 * ca2 * v7 * w4
            - 6.0 * v5 * w5
            + 2.0 * ca2 * v5 * w5
            - 4.0 * cf * v5 * w5
            - 20.0 * cacf * v5 * w5
            + 14.0 * v6 * w5
            - 6.0 * ca2 * v6 * w5
            + 4.0 * cf * v6 * w5
            + 16.0 * cacf * v6 * w5
            - 8.0 * v7 * w5
            + 4.0 * ca2 * v7 * w5
            - 2.0 * v6 * w6
            + 2.0 * ca2 * v6 * w6
            - 4.0 * cacf * v6 * w6
            + 2.0 * v7 * w6
            - 2.0 * ca2 * v7 * w6))
        / ((1.0 - v).powi(2) * v * w * (1.0 - v + v * w).powi(3));

    let part5 = -(2.0
        * cf
        * lms
        * (6.0 * cacf - 8.0 * cacf * v + 4.0 * cf * v2 + 18.0 * cacf * v2
            - 4.0 * cf * v3
            - 12.0 * cacf * v3
            + 4.0 * cacf * v4
            - 4.0 * cacf * w
            + 2.0 * v * w
            - 8.0 * cacf * v * w
            - 12.0 * cf * v2 * w
            - 16.0 * cacf * v2 * w
            + 2.0 * v3 * w
            - 16.0 * cacf * v3 * w
            + 12.0 * cf * v4 * w
            + 24.0 * cacf * v4 * w
            - 12.0 * cacf * v5 * w
            + 12.0 * cacf * v * w2
            - 5.0 * v2 * w2
            + ca2 * v2 * w2
            - 8.0 * cf * v2 * w2
            - 34.0 * cacf * v2 * w2
            - 4.0 * v3 * w2
            + 44.0 * cf * v3 * w2
            + 116.0 * cacf * v3 * w2
            - v4 * w2
            + ca2 * v4 * w2
            - 24.0 * cf * v4 * w2
            - 58.0 * cacf * v4 * w2
            - 12.0 * cf * v5 * w2
            + 12.0 * cacf * v6 * w2
            - 12.0 * cacf * v2 * w3
            + 8.0 * v3 * w3
            - 2.0 * ca2 * v3 * w3
            + 16.0 * cf * v3 * w3
            + 72.0 * cacf * v3 * w3
            + 2.0 * v4 * w3
            - 2.0 * ca2 * v4 * w3
            - 48.0 * cf * v4 * w3
            - 164.0 * cacf * v4 * w3
            + 2.0 * v5 * w3
            + 28.0 * cf * v5 * w3
            + 100.0 * cacf * v5 * w3
            + 4.0 * cf * v6 * w3
            - 24.0 * cacf * v6 * w3
            - 4.0 * cacf * v7 * w3
            + 4.0 * cacf * v3 * w4
            - 7.0 * v4 * w4
            + 3.0 * ca2 * v4 * w4
            - 12.0 * cf * v4 * w4
            - 52.0 * cacf * v4 * w4
            - 4.0 * v5 * w4
            + 20.0 * cf * v5 * w4
            + 92.0 * cacf * v5 * w4
            - v6 * w4
            + ca2 * v6 * w4
            - 8.0 * cf * v6 * w4
            - 48.0 * cacf * v6 * w4
            + 12.0 * cacf * v7 * w4
            + 6.0 * v5 * w5
            - 2.0 * ca2 * v5 * w5
            + 4.0 * cf * v5 * w5
            + 20.0 * cacf * v5 * w5
            + 2.0 * v6 * w5
            - 2.0 * ca2 * v6 * w5
            - 4.0 * cf * v6 * w5
            - 24.0 * cacf * v6 * w5
            + 4.0 * cacf * v7 * w5
            - 2.0 * v6 * w6
            + 2.0 * ca2 * v6 * w6
            - 4.0 * cacf * v6 * w6
            + 4.0 * cacf * v7 * w6))
        / ((1.0 - v).powi(2) * v * w * (1.0 - v * w).powi(3));

    let part6 = -(cf
        * (4.0 + 10.0 * ca - 4.0 * ca2 - 10.0 * ca3 + 8.0 * cacf + 8.0 * ca2cf
            - 20.0 * v
            - 46.0 * ca * v
            + 20.0 * ca2 * v
            + 46.0 * ca3 * v
            - 40.0 * cacf * v
            - 40.0 * ca2cf * v
            + 48.0 * v2
            + 88.0 * ca * v2
            - 48.0 * ca2 * v2
            - 88.0 * ca3 * v2
            + 88.0 * cacf * v2
            + 80.0 * ca2cf * v2
            - 72.0 * v3
            - 88.0 * ca * v3
            + 72.0 * ca2 * v3
            + 88.0 * ca3 * v3
            - 112.0 * cacf * v3
            - 80.0 * ca2cf * v3
            + 68.0 * v4
            + 46.0 * ca * v4
            - 68.0 * ca2 * v4
            - 46.0 * ca3 * v4
            + 88.0 * cacf * v4
            + 40.0 * ca2cf * v4
            - 36.0 * v5
            - 10.0 * ca * v5
            + 36.0 * ca2 * v5
            + 10.0 * ca3 * v5
            - 40.0 * cacf * v5
            - 8.0 * ca2cf * v5
            + 8.0 * v6
            - 8.0 * ca2 * v6
            + 8.0 * cacf * v6
            - 4.0 * w
            - 24.0 * ca * w
            + 4.0 * ca2 * w
            + 8.0 * ca3 * w
            - 8.0 * cacf * w
            - 8.0 * ca2cf * w
            + 20.0 * v * w
            + 118.0 * ca * v * w
            - 20.0 * ca2 * v * w
            - 42.0 * ca3 * v * w
            + 40.0 * cacf * v * w
            + 40.0 * ca2cf * v * w
            - 44.0 * v2 * w
            - 199.0 * ca * v2 * w
            + 28.0 * ca2 * v2 * w
            + 43.0 * ca3 * v2 * w
            - 80.0 * cacf * v2 * w
            - 40.0 * ca2cf * v2 * w
            + 56.0 * v3 * w
            + 104.0 * ca * v3 * w
            + 8.0 * ca2 * v3 * w
            + 72.0 * ca3 * v3 * w
            + 80.0 * cacf * v3 * w
            - 80.0 * ca2cf * v3 * w
            - 20.0 * v4 * w
            + 80.0 * ca * v4 * w
            - 76.0 * ca2 * v4 * w
            - 184.0 * ca3 * v4 * w
            - 16.0 * cacf * v4 * w
            + 200.0 * ca2cf * v4 * w
            - 52.0 * v5 * w
            - 128.0 * ca * v5 * w
            + 116.0 * ca2 * v5 * w
            + 140.0 * ca3 * v5 * w
            - 64.0 * cacf * v5 * w
            - 152.0 * ca2cf * v5 * w
            + 68.0 * v6 * w
            + 55.0 * ca * v6 * w
            - 84.0 * ca2 * v6 * w
            - 35.0 * ca3 * v6 * w
            + 72.0 * cacf * v6 * w
            + 40.0 * ca2cf * v6 * w
            - 24.0 * v7 * w
            - 6.0 * ca * v7 * w
            + 24.0 * ca2 * v7 * w
            - 2.0 * ca3 * v7 * w
            - 24.0 * cacf * v7 * w
            - 4.0 * v2 * w2
            + 54.0 * ca * v2 * w2
            + 60.0 * ca2 * v2 * w2
            - 2.0 * ca3 * v2 * w2
            - 96.0 * cacf * v2 * w2
            - 64.0 * ca2cf * v2 * w2
            + 16.0 * v3 * w2
            - 234.0 * ca * v3 * w2
            - 240.0 * ca2 * v3 * w2
            + 18.0 * ca3 * v3 * w2
            + 384.0 * cacf * v3 * w2
            + 256.0 * ca2cf * v3 * w2
            - 72.0 * v4 * w2
            + 473.0 * ca * v4 * w2
            + 388.0 * ca2 * v4 * w2
            - 109.0 * ca3 * v4 * w2
            - 656.0 * cacf * v4 * w2
            - 320.0 * ca2cf * v4 * w2
            + 160.0 * v5 * w2
            - 567.0 * ca * v5 * w2
            - 324.0 * ca2 * v5 * w2
            + 227.0 * ca3 * v5 * w2
            + 624.0 * cacf * v5 * w2
            + 64.0 * ca2cf * v5 * w2
            - 124.0 * v6 * w2
            + 411.0 * ca * v6 * w2
            + 120.0 * ca2 * v6 * w2
            - 195.0 * ca3 * v6 * w2
            - 312.0 * cacf * v6 * w2
            + 128.0 * ca2cf * v6 * w2
            - 155.0 * ca * v7 * w2
            + 20.0 * ca2 * v7 * w2
            + 55.0 * ca3 * v7 * w2
            + 32.0 * cacf * v7 * w2
            - 64.0 * ca2cf * v7 * w2
            + 24.0 * v8 * w2
            + 18.0 * ca * v8 * w2
            - 24.0 * ca2 * v8 * w2
            + 6.0 * ca3 * v8 * w2
            + 24.0 * cacf * v8 * w2
            + 12.0 * v2 * w3
            + 72.0 * ca * v2 * w3
            - 12.0 * ca2 * v2 * w3
            - 24.0 * ca3 * v2 * w3
            + 24.0 * cacf * v2 * w3
            + 24.0 * ca2cf * v2 * w3
            - 48.0 * v3 * w3
            - 282.0 * ca * v3 * w3
            + 48.0 * ca2 * v3 * w3
            + 90.0 * ca3 * v3 * w3
            - 96.0 * cacf * v3 * w3
            - 96.0 * ca2cf * v3 * w3
            + 96.0 * v4 * w3
            + 397.0 * ca * v4 * w3
            - 153.0 * ca3 * v4 * w3
            + 56.0 * cacf * v4 * w3
            - 120.0 * v5 * w3
            - 233.0 * ca * v5 * w3
            - 168.0 * ca2 * v5 * w3
            + 169.0 * ca3 * v5 * w3
            + 168.0 * cacf * v5 * w3
            + 336.0 * ca2cf * v5 * w3
            + 14.0 * v6 * w3
            + 114.0 * ca * v6 * w3
            + 270.0 * ca2 * v6 * w3
            - 170.0 * ca3 * v6 * w3
            - 360.0 * cacf * v6 * w3
            - 368.0 * ca2cf * v6 * w3
            + 116.0 * v7 * w3
            - 171.0 * ca * v7 * w3
            - 204.0 * ca2 * v7 * w3
            + 147.0 * ca3 * v7 * w3
            + 328.0 * cacf * v7 * w3
            + 64.0 * ca2cf * v7 * w3
            - 62.0 * v8 * w3
            + 125.0 * ca * v8 * w3
            + 58.0 * ca2 * v8 * w3
            - 57.0 * ca3 * v8 * w3
            - 112.0 * cacf * v8 * w3
            + 40.0 * ca2cf * v8 * w3
            - 8.0 * v9 * w3
            - 18.0 * ca * v9 * w3
            + 8.0 * ca2 * v9 * w3
            - 6.0 * ca3 * v9 * w3
            - 8.0 * cacf * v9 * w3
            + 20.0 * v4 * w4
            + 118.0 * ca * v4 * w4
            - 84.0 * ca2 * v4 * w4
            + 18.0 * ca3 * v4 * w4
            + 152.0 * cacf * v4 * w4
            + 104.0 * ca2cf * v4 * w4
            - 60.0 * v5 * w4
            - 338.0 * ca * v5 * w4
            + 252.0 * ca2 * v5 * w4
            - 38.0 * ca3 * v5 * w4
            - 456.0 * cacf * v5 * w4
            - 312.0 * ca2cf * v5 * w4
            + 114.0 * v6 * w4
            + 190.0 * ca * v6 * w4
            - 278.0 * ca2 * v6 * w4
            + 86.0 * ca3 * v6 * w4
            + 536.0 * cacf * v6 * w4
            + 200.0 * ca2cf * v6 * w4
            - 128.0 * v7 * w4
            + 192.0 * ca * v7 * w4
            + 136.0 * ca2 * v7 * w4
            - 136.0 * ca3 * v7 * w4
            - 312.0 * cacf * v7 * w4
            + 120.0 * ca2cf * v7 * w4
            + 24.0 * v8 * w4
            - 167.0 * ca * v8 * w4
            + 4.0 * ca2 * v8 * w4
            + 55.0 * ca3 * v8 * w4
            + 32.0 * cacf * v8 * w4
            - 104.0 * ca2cf * v8 * w4
            + 30.0 * v9 * w4
            - 11.0 * ca * v9 * w4
            - 30.0 * ca2 * v9 * w4
            + 23.0 * ca3 * v9 * w4
            + 48.0 * cacf * v9 * w4
            - 8.0 * ca2cf * v9 * w4
            + 6.0 * ca * v10 * w4
            + 2.0 * ca3 * v10 * w4
            - 12.0 * v4 * w5
            - 72.0 * ca * v4 * w5
            + 12.0 * ca2 * v4 * w5
            + 24.0 * ca3 * v4 * w5
            - 24.0 * cacf * v4 * w5
            - 24.0 * ca2cf * v4 * w5
            + 36.0 * v5 * w5
            + 214.0 * ca * v5 * w5
            - 36.0 * ca2 * v5 * w5
            - 82.0 * ca3 * v5 * w5
            + 72.0 * cacf * v5 * w5
            + 72.0 * ca2cf * v5 * w5
            - 12.0 * v6 * w5
            - 61.0 * ca * v6 * w5
            - 20.0 * ca2 * v6 * w5
            + ca3 * v6 * w5
            + 16.0 * cacf * v6 * w5
            + 40.0 * ca2cf * v6 * w5
            - 36.0 * v7 * w5
            - 250.0 * ca * v7 * w5
            + 100.0 * ca2 * v7 * w5
            + 158.0 * ca3 * v7 * w5
            - 152.0 * cacf * v7 * w5
            - 200.0 * ca2cf * v7 * w5
            + 56.0 * v8 * w5
            + 130.0 * ca * v8 * w5
            - 88.0 * ca2 * v8 * w5
            - 66.0 * ca3 * v8 * w5
            + 160.0 * cacf * v8 * w5
            + 88.0 * ca2cf * v8 * w5
            - 32.0 * v9 * w5
            + 67.0 * ca * v9 * w5
            + 32.0 * ca2 * v9 * w5
            - 47.0 * ca3 * v9 * w5
            - 72.0 * cacf * v9 * w5
            + 24.0 * ca2cf * v9 * w5
            - 12.0 * ca * v10 * w5
            - 4.0 * ca3 * v10 * w5
            - 36.0 * v6 * w6
            - 118.0 * ca * v6 * w6
            + 44.0 * ca2 * v6 * w6
            + 82.0 * ca3 * v6 * w6
            - 80.0 * cacf * v6 * w6
            - 48.0 * ca2cf * v6 * w6
            + 72.0 * v7 * w6
            + 250.0 * ca * v7 * w6
            - 88.0 * ca2 * v7 * w6
            - 154.0 * ca3 * v7 * w6
            + 160.0 * cacf * v7 * w6
            + 96.0 * ca2cf * v7 * w6
            - 26.0 * v8 * w6
            - 103.0 * ca * v8 * w6
            + 34.0 * ca2 * v8 * w6
            + 31.0 * ca3 * v8 * w6
            - 96.0 * cacf * v8 * w6
            - 24.0 * ca2cf * v8 * w6
            - 10.0 * v9 * w6
            - 61.0 * ca * v9 * w6
            + 10.0 * ca2 * v9 * w6
            + 57.0 * ca3 * v9 * w6
            + 16.0 * cacf * v9 * w6
            - 24.0 * ca2cf * v9 * w6
            + 6.0 * ca * v10 * w6
            + 10.0 * ca3 * v10 * w6
            + 4.0 * v6 * w7
            + 24.0 * ca * v6 * w7
            - 4.0 * ca2 * v6 * w7
            - 8.0 * ca3 * v6 * w7
            + 8.0 * cacf * v6 * w7
            + 8.0 * ca2cf * v6 * w7
            - 8.0 * v7 * w7
            - 54.0 * ca * v7 * w7
            + 8.0 * ca2 * v7 * w7
            + 6.0 * ca3 * v7 * w7
            - 16.0 * cacf * v7 * w7
            - 16.0 * ca2cf * v7 * w7
            - 32.0 * v8 * w7
            + 23.0 * ca * v8 * w7
            + 32.0 * ca2 * v8 * w7
            + 13.0 * ca3 * v8 * w7
            - 24.0 * cacf * v8 * w7
            + 36.0 * v9 * w7
            + 35.0 * ca * v9 * w7
            - 36.0 * ca2 * v9 * w7
            - 23.0 * ca3 * v9 * w7
            + 32.0 * cacf * v9 * w7
            + 8.0 * ca2cf * v9 * w7
            - 16.0 * ca3 * v10 * w7
            + 16.0 * v8 * w8
            - 16.0 * ca2 * v8 * w8
            + 8.0 * ca3 * v8 * w8
            + 16.0 * cacf * v8 * w8
            - 16.0 * v9 * w8
            - 16.0 * ca * v9 * w8
            + 16.0 * ca2 * v9 * w8
            - 16.0 * cacf * v9 * w8
            + 8.0 * ca3 * v10 * w8
            + 4.0 * ca * v9 * w9
            - 4.0 * ca3 * v9 * w9))
        / (ca * (1.0 - v).powi(2) * v * w * (1.0 - v * w).powi(3) * (1.0 - v + v * w).powi(3));

    let part7 = -(2.0
        * cf
        * lv
        * (4.0 * ca - 4.0 * ca3 - 20.0 * ca * v + 20.0 * ca3 * v - 2.0 * v2
            + 52.0 * ca * v2
            + 2.0 * ca2 * v2
            - 52.0 * ca3 * v2
            + 8.0 * v3
            - 88.0 * ca * v3
            - 8.0 * ca2 * v3
            + 88.0 * ca3 * v3
            - 12.0 * v4
            + 96.0 * ca * v4
            + 12.0 * ca2 * v4
            - 96.0 * ca3 * v4
            + 8.0 * v5
            - 64.0 * ca * v5
            - 8.0 * ca2 * v5
            + 64.0 * ca3 * v5
            - 2.0 * v6
            + 24.0 * ca * v6
            + 2.0 * ca2 * v6
            - 24.0 * ca3 * v6
            - 4.0 * ca * v7
            + 4.0 * ca3 * v7
            - 2.0 * ca * w
            + 2.0 * ca3 * w
            + 10.0 * ca * v * w
            - 10.0 * ca3 * v * w
            + 14.0 * v2 * w
            + 39.0 * ca * v2 * w
            + 18.0 * ca2 * v2 * w
            + 35.0 * ca3 * v2 * w
            - 56.0 * v3 * w
            - 216.0 * ca * v3 * w
            - 72.0 * ca2 * v3 * w
            - 80.0 * ca3 * v3 * w
            + 82.0 * v4 * w
            + 400.0 * ca * v4 * w
            + 114.0 * ca2 * v4 * w
            + 72.0 * ca3 * v4 * w
            - 50.0 * v5 * w
            - 402.0 * ca * v5 * w
            - 90.0 * ca2 * v5 * w
            + 22.0 * ca3 * v5 * w
            + 8.0 * v6 * w
            + 239.0 * ca * v6 * w
            + 36.0 * ca2 * v6 * w
            - 81.0 * ca3 * v6 * w
            + 2.0 * v7 * w
            - 80.0 * ca * v7 * w
            - 6.0 * ca2 * v7 * w
            + 52.0 * ca3 * v7 * w
            + 12.0 * ca * v8 * w
            - 12.0 * ca3 * v8 * w
            + 70.0 * ca * v2 * w2
            + 32.0 * ca2 * v2 * w2
            + 38.0 * ca3 * v2 * w2
            - 280.0 * ca * v3 * w2
            - 128.0 * ca2 * v3 * w2
            - 152.0 * ca3 * v3 * w2
            + 36.0 * v4 * w2
            + 571.0 * ca * v4 * w2
            + 244.0 * ca2 * v4 * w2
            + 329.0 * ca3 * v4 * w2
            - 108.0 * v5 * w2
            - 733.0 * ca * v5 * w2
            - 284.0 * ca2 * v5 * w2
            - 455.0 * ca3 * v5 * w2
            + 110.0 * v6 * w2
            + 607.0 * ca * v6 * w2
            + 194.0 * ca2 * v6 * w2
            + 341.0 * ca3 * v6 * w2
            - 40.0 * v7 * w2
            - 319.0 * ca * v7 * w2
            - 64.0 * ca2 * v7 * w2
            - 101.0 * ca3 * v7 * w2
            + 2.0 * v8 * w2
            + 96.0 * ca * v8 * w2
            + 6.0 * ca2 * v8 * w2
            - 12.0 * ca3 * v8 * w2
            - 12.0 * ca * v9 * w2
            + 12.0 * ca3 * v9 * w2
            + 6.0 * ca * v2 * w3
            - 6.0 * ca3 * v2 * w3
            - 24.0 * ca * v3 * w3
            + 24.0 * ca3 * v3 * w3
            - 18.0 * v4 * w3
            + 135.0 * ca * v4 * w3
            + 26.0 * ca2 * v4 * w3
            - 23.0 * ca3 * v4 * w3
            + 54.0 * v5 * w3
            - 321.0 * ca * v5 * w3
            - 78.0 * ca2 * v5 * w3
            - 15.0 * ca3 * v5 * w3
            - 30.0 * v6 * w3
            + 416.0 * ca * v6 * w3
            + 126.0 * ca2 * v6 * w3
            + 154.0 * ca3 * v6 * w3
            - 30.0 * v7 * w3
            - 325.0 * ca * v7 * w3
            - 122.0 * ca2 * v7 * w3
            - 255.0 * ca3 * v7 * w3
            + 26.0 * v8 * w3
            + 161.0 * ca * v8 * w3
            + 50.0 * ca2 * v8 * w3
            + 157.0 * ca3 * v8 * w3
            - 2.0 * v9 * w3
            - 48.0 * ca * v9 * w3
            - 2.0 * ca2 * v9 * w3
            - 36.0 * ca3 * v9 * w3
            + 4.0 * ca * v10 * w3
            - 4.0 * ca3 * v10 * w3
            - 12.0 * v4 * w4
            - 172.0 * ca * v4 * w4
            - 68.0 * ca2 * v4 * w4
            - 60.0 * ca3 * v4 * w4
            + 36.0 * v5 * w4
            + 516.0 * ca * v5 * w4
            + 204.0 * ca2 * v5 * w4
            + 180.0 * ca3 * v5 * w4
            - 64.0 * v6 * w4
            - 602.0 * ca * v6 * w4
            - 240.0 * ca2 * v6 * w4
            - 306.0 * ca3 * v6 * w4
            + 68.0 * v7 * w4
            + 344.0 * ca * v7 * w4
            + 140.0 * ca2 * v7 * w4
            + 312.0 * ca3 * v7 * w4
            - 26.0 * v8 * w4
            - 91.0 * ca * v8 * w4
            - 22.0 * ca2 * v8 * w4
            - 125.0 * ca3 * v8 * w4
            - 2.0 * v9 * w4
            + 5.0 * ca * v9 * w4
            - 14.0 * ca2 * v9 * w4
            - ca3 * v9 * w4
            + 8.0 * ca * v10 * w4
            + 20.0 * ca3 * v10 * w4
            - 6.0 * ca * v4 * w5
            + 6.0 * ca3 * v4 * w5
            + 18.0 * ca * v5 * w5
            - 18.0 * ca3 * v5 * w5
            - 14.0 * v6 * w5
            - 193.0 * ca * v6 * w5
            - 66.0 * ca2 * v6 * w5
            + 15.0 * ca3 * v6 * w5
            + 28.0 * v7 * w5
            + 356.0 * ca * v7 * w5
            + 132.0 * ca2 * v7 * w5
            - 22.0 * v8 * w5
            - 278.0 * ca * v8 * w5
            - 94.0 * ca2 * v8 * w5
            - 72.0 * ca3 * v8 * w5
            + 8.0 * v9 * w5
            + 103.0 * ca * v9 * w5
            + 28.0 * ca2 * v9 * w5
            + 69.0 * ca3 * v9 * w5
            - 20.0 * ca * v10 * w5
            - 36.0 * ca3 * v10 * w5
            + 16.0 * v6 * w6
            + 74.0 * ca * v6 * w6
            + 48.0 * ca2 * v6 * w6
            + 26.0 * ca3 * v6 * w6
            - 32.0 * v7 * w6
            - 148.0 * ca * v7 * w6
            - 96.0 * ca2 * v7 * w6
            - 52.0 * ca3 * v7 * w6
            + 14.0 * v8 * w6
            + 105.0 * ca * v8 * w6
            + 42.0 * ca2 * v8 * w6
            + 63.0 * ca3 * v8 * w6
            + 2.0 * v9 * w6
            - 31.0 * ca * v9 * w6
            + 6.0 * ca2 * v9 * w6
            - 37.0 * ca3 * v9 * w6
            - 4.0 * ca * v10 * w6
            + 28.0 * ca3 * v10 * w6
            + 2.0 * ca * v6 * w7
            - 2.0 * ca3 * v6 * w7
            - 4.0 * ca * v7 * w7
            + 4.0 * ca3 * v7 * w7
            + 10.0 * v8 * w7
            + 11.0 * ca * v8 * w7
            + 30.0 * ca2 * v8 * w7
            + 21.0 * ca3 * v8 * w7
            - 10.0 * v9 * w7
            - 9.0 * ca * v9 * w7
            - 30.0 * ca2 * v9 * w7
            - 23.0 * ca3 * v9 * w7
            + 20.0 * ca * v10 * w7
            - 12.0 * ca3 * v10 * w7
            - 4.0 * v8 * w8
            + 8.0 * ca * v8 * w8
            - 12.0 * ca2 * v8 * w8
            - 16.0 * ca3 * v8 * w8
            + 4.0 * v9 * w8
            - 8.0 * ca * v9 * w8
            + 12.0 * ca2 * v9 * w8
            + 16.0 * ca3 * v9 * w8
            - 12.0 * ca * v10 * w8
            + 8.0 * ca3 * v10 * w8
            + 4.0 * ca * v10 * w9
            - 4.0 * ca3 * v10 * w9))
        / (ca * (1.0 - v).powi(2) * v * w * (1.0 - v * w).powi(3) * (1.0 - v + v * w).powi(3));

    let part8 = -(2.0
        * cf
        * l1w
        * (4.0 * ca - 4.0 * ca3 - 20.0 * ca * v + 20.0 * ca3 * v + 52.0 * ca * v2
            - 52.0 * ca3 * v2
            - 88.0 * ca * v3
            + 88.0 * ca3 * v3
            + 96.0 * ca * v4
            - 96.0 * ca3 * v4
            - 64.0 * ca * v5
            + 64.0 * ca3 * v5
            + 24.0 * ca * v6
            - 24.0 * ca3 * v6
            - 4.0 * ca * v7
            + 4.0 * ca3 * v7
            - 2.0 * ca * w
            + 2.0 * ca3 * w
            + 10.0 * ca * v * w
            - 10.0 * ca3 * v * w
            - 6.0 * v2 * w
            - 45.0 * ca * v2 * w
            + 14.0 * ca2 * v2 * w
            + 51.0 * ca3 * v2 * w
            + 24.0 * v3 * w
            + 120.0 * ca * v3 * w
            - 56.0 * ca2 * v3 * w
            - 144.0 * ca3 * v3 * w
            - 32.0 * v4 * w
            - 140.0 * ca * v4 * w
            + 84.0 * ca2 * v4 * w
            + 176.0 * ca3 * v4 * w
            + 12.0 * v5 * w
            + 42.0 * ca * v5 * w
            - 56.0 * ca2 * v5 * w
            - 66.0 * ca3 * v5 * w
            + 6.0 * v6 * w
            + 47.0 * ca * v6 * w
            + 14.0 * ca2 * v6 * w
            - 41.0 * ca3 * v6 * w
            - 4.0 * v7 * w
            - 44.0 * ca * v7 * w
            + 44.0 * ca3 * v7 * w
            + 12.0 * ca * v8 * w
            - 12.0 * ca3 * v8 * w
            - 12.0 * v2 * w2
            - 18.0 * ca * v2 * w2
            + 20.0 * ca2 * v2 * w2
            + 50.0 * ca3 * v2 * w2
            + 48.0 * v3 * w2
            + 72.0 * ca * v3 * w2
            - 80.0 * ca2 * v3 * w2
            - 200.0 * ca3 * v3 * w2
            - 100.0 * v4 * w2
            - 253.0 * ca * v4 * w2
            + 164.0 * ca2 * v4 * w2
            + 459.0 * ca3 * v4 * w2
            + 132.0 * v5 * w2
            + 507.0 * ca * v5 * w2
            - 212.0 * ca2 * v5 * w2
            - 677.0 * ca3 * v5 * w2
            - 88.0 * v6 * w2
            - 477.0 * ca * v6 * w2
            + 152.0 * ca2 * v6 * w2
            + 551.0 * ca3 * v6 * w2
            + 12.0 * v7 * w2
            + 193.0 * ca * v7 * w2
            - 44.0 * ca2 * v7 * w2
            - 207.0 * ca3 * v7 * w2
            + 8.0 * v8 * w2
            - 12.0 * ca * v8 * w2
            + 12.0 * ca3 * v8 * w2
            - 12.0 * ca * v9 * w2
            + 12.0 * ca3 * v9 * w2
            + 6.0 * ca * v2 * w3
            - 6.0 * ca3 * v2 * w3
            - 24.0 * ca * v3 * w3
            + 24.0 * ca3 * v3 * w3
            + 115.0 * ca * v4 * w3
            + 6.0 * ca2 * v4 * w3
            - 29.0 * ca3 * v4 * w3
            - 261.0 * ca * v5 * w3
            - 18.0 * ca2 * v5 * w3
            + 3.0 * ca3 * v5 * w3
            - 44.0 * v6 * w3
            + 88.0 * ca * v6 * w3
            + 62.0 * ca2 * v6 * w3
            + 188.0 * ca3 * v6 * w3
            + 88.0 * v7 * w3
            + 231.0 * ca * v7 * w3
            - 94.0 * ca2 * v7 * w3
            - 353.0 * ca3 * v7 * w3
            - 40.0 * v8 * w3
            - 215.0 * ca * v8 * w3
            + 44.0 * ca2 * v8 * w3
            + 233.0 * ca3 * v8 * w3
            - 4.0 * v9 * w3
            + 60.0 * ca * v9 * w3
            - 60.0 * ca3 * v9 * w3
            + 4.0 * ca * v10 * w3
            - 4.0 * ca3 * v10 * w3
            + 20.0 * v4 * w4
            + 12.0 * ca * v4 * w4
            - 40.0 * ca2 * v4 * w4
            - 84.0 * ca3 * v4 * w4
            - 60.0 * v5 * w4
            - 36.0 * ca * v5 * w4
            + 120.0 * ca2 * v5 * w4
            + 252.0 * ca3 * v5 * w4
            + 106.0 * v6 * w4
            + 266.0 * ca * v6 * w4
            - 162.0 * ca2 * v6 * w4
            - 446.0 * ca3 * v6 * w4
            - 112.0 * v7 * w4
            - 472.0 * ca * v7 * w4
            + 124.0 * ca2 * v7 * w4
            + 472.0 * ca3 * v7 * w4
            + 24.0 * v8 * w4
            + 241.0 * ca * v8 * w4
            - 28.0 * ca2 * v8 * w4
            - 199.0 * ca3 * v8 * w4
            + 22.0 * v9 * w4
            - 11.0 * ca * v9 * w4
            - 14.0 * ca2 * v9 * w4
            + 5.0 * ca3 * v9 * w4
            - 28.0 * ca * v10 * w4
            + 28.0 * ca3 * v10 * w4
            - 6.0 * ca * v4 * w5
            + 6.0 * ca3 * v4 * w5
            + 18.0 * ca * v5 * w5
            - 18.0 * ca3 * v5 * w5
            + 2.0 * v6 * w5
            - 61.0 * ca * v6 * w5
            - 30.0 * ca2 * v6 * w5
            + 3.0 * ca3 * v6 * w5
            - 4.0 * v7 * w5
            + 92.0 * ca * v7 * w5
            + 60.0 * ca2 * v7 * w5
            + 24.0 * ca3 * v7 * w5
            + 38.0 * v8 * w5
            + 70.0 * ca * v8 * w5
            - 56.0 * ca2 * v8 * w5
            - 126.0 * ca3 * v8 * w5
            - 36.0 * v9 * w5
            - 113.0 * ca * v9 * w5
            + 26.0 * ca2 * v9 * w5
            + 111.0 * ca3 * v9 * w5
            + 52.0 * ca * v10 * w5
            - 52.0 * ca3 * v10 * w5
            - 12.0 * v6 * w6
            - 30.0 * ca * v6 * w6
            + 28.0 * ca2 * v6 * w6
            + 38.0 * ca3 * v6 * w6
            + 24.0 * v7 * w6
            + 60.0 * ca * v7 * w6
            - 56.0 * ca2 * v7 * w6
            - 76.0 * ca3 * v7 * w6
            - 30.0 * v8 * w6
            - 103.0 * ca * v8 * w6
            + 30.0 * ca2 * v8 * w6
            + 97.0 * ca3 * v8 * w6
            + 18.0 * v9 * w6
            + 73.0 * ca * v9 * w6
            - 2.0 * ca2 * v9 * w6
            - 59.0 * ca3 * v9 * w6
            - 36.0 * ca * v10 * w6
            + 36.0 * ca3 * v10 * w6
            + 2.0 * ca * v6 * w7
            - 2.0 * ca3 * v6 * w7
            - 4.0 * ca * v7 * w7
            + 4.0 * ca3 * v7 * w7
            - 4.0 * v8 * w7
            - 17.0 * ca * v8 * w7
            + 18.0 * ca2 * v8 * w7
            + 23.0 * ca3 * v8 * w7
            + 4.0 * v9 * w7
            + 19.0 * ca * v9 * w7
            - 18.0 * ca2 * v9 * w7
            - 25.0 * ca3 * v9 * w7
            + 12.0 * ca * v10 * w7
            - 12.0 * ca3 * v10 * w7
            + 4.0 * v8 * w8
            + 16.0 * ca * v8 * w8
            - 8.0 * ca2 * v8 * w8
            - 16.0 * ca3 * v8 * w8
            - 4.0 * v9 * w8
            - 16.0 * ca * v9 * w8
            + 8.0 * ca2 * v9 * w8
            + 16.0 * ca3 * v9 * w8
            - 8.0 * ca * v10 * w8
            + 8.0 * ca3 * v10 * w8
            + 4.0 * ca * v10 * w9
            - 4.0 * ca3 * v10 * w9))
        / (ca * (1.0 - v).powi(2) * v * w * (1.0 - v * w).powi(3) * (1.0 - v + v * w).powi(3));

    let part9 = -(4.0
        * cf
        * lw
        * (3.0 * ca - 3.0 * ca3 + 2.0 * ca2cf - 15.0 * ca * v + 15.0 * ca3 * v - 10.0 * ca2cf * v
            + 2.0 * v2
            + 39.0 * ca * v2
            - 2.0 * ca2 * v2
            - 39.0 * ca3 * v2
            + 2.0 * cacf * v2
            + 26.0 * ca2cf * v2
            - 8.0 * v3
            - 66.0 * ca * v3
            + 8.0 * ca2 * v3
            + 66.0 * ca3 * v3
            - 8.0 * cacf * v3
            - 44.0 * ca2cf * v3
            + 12.0 * v4
            + 72.0 * ca * v4
            - 12.0 * ca2 * v4
            - 72.0 * ca3 * v4
            + 12.0 * cacf * v4
            + 48.0 * ca2cf * v4
            - 8.0 * v5
            - 48.0 * ca * v5
            + 8.0 * ca2 * v5
            + 48.0 * ca3 * v5
            - 8.0 * cacf * v5
            - 32.0 * ca2cf * v5
            + 2.0 * v6
            + 18.0 * ca * v6
            - 2.0 * ca2 * v6
            - 18.0 * ca3 * v6
            + 2.0 * cacf * v6
            + 12.0 * ca2cf * v6
            - 3.0 * ca * v7
            + 3.0 * ca3 * v7
            - 2.0 * ca2cf * v7
            - 10.0 * ca * w
            - 2.0 * ca3 * w
            + 8.0 * ca2cf * w
            + 50.0 * ca * v * w
            + 10.0 * ca3 * v * w
            - 40.0 * ca2cf * v * w
            - v2 * w
            - 105.0 * ca * v2 * w
            - 3.0 * ca2 * v2 * w
            - 28.0 * ca3 * v2 * w
            + 90.0 * ca2cf * v2 * w
            + 4.0 * v3 * w
            + 120.0 * ca * v3 * w
            + 12.0 * ca2 * v3 * w
            + 52.0 * ca3 * v3 * w
            - 120.0 * ca2cf * v3 * w
            - 55.0 * ca * v4 * w
            - 24.0 * ca2 * v4 * w
            - 85.0 * ca3 * v4 * w
            + 6.0 * cacf * v4 * w
            + 118.0 * ca2cf * v4 * w
            - 14.0 * v5 * w
            - 45.0 * ca * v5 * w
            + 30.0 * ca2 * v5 * w
            + 115.0 * ca3 * v5 * w
            - 18.0 * cacf * v5 * w
            - 102.0 * ca2cf * v5 * w
            + 17.0 * v6 * w
            + 79.0 * ca * v6 * w
            - 21.0 * ca2 * v6 * w
            - 98.0 * ca3 * v6 * w
            + 18.0 * cacf * v6 * w
            + 70.0 * ca2cf * v6 * w
            - 6.0 * v7 * w
            - 43.0 * ca * v7 * w
            + 6.0 * ca2 * v7 * w
            + 45.0 * ca3 * v7 * w
            - 6.0 * cacf * v7 * w
            - 30.0 * ca2cf * v7 * w
            + 9.0 * ca * v8 * w
            - 9.0 * ca3 * v8 * w
            + 6.0 * ca2cf * v8 * w
            + ca * w2
            - ca3 * w2
            + 2.0 * ca2cf * w2
            - 5.0 * ca * v * w2
            + 5.0 * ca3 * v * w2
            - 10.0 * ca2cf * v * w2
            - v2 * w2
            - 20.0 * ca * v2 * w2
            + ca2 * v2 * w2
            - 9.0 * ca3 * v2 * w2
            + 2.0 * cacf * v2 * w2
            + 36.0 * ca2cf * v2 * w2
            + 4.0 * v3 * w2
            + 110.0 * ca * v3 * w2
            - 4.0 * ca2 * v3 * w2
            + 6.0 * ca3 * v3 * w2
            - 8.0 * cacf * v3 * w2
            - 84.0 * ca2cf * v3 * w2
            - 16.0 * v4 * w2
            - 229.0 * ca * v4 * w2
            + 4.0 * ca2 * v4 * w2
            + 19.0 * ca3 * v4 * w2
            + 6.0 * cacf * v4 * w2
            + 106.0 * ca2cf * v4 * w2
            + 34.0 * v5 * w2
            + 281.0 * ca * v5 * w2
            + 2.0 * ca2 * v5 * w2
            - 57.0 * ca3 * v5 * w2
            + 10.0 * cacf * v5 * w2
            - 66.0 * ca2cf * v5 * w2
            - 25.0 * v6 * w2
            - 180.0 * ca * v6 * w2
            - 11.0 * ca2 * v6 * w2
            + 37.0 * ca3 * v6 * w2
            - 10.0 * cacf * v6 * w2
            + 28.0 * ca2cf * v6 * w2
            - 2.0 * v7 * w2
            + 30.0 * ca * v7 * w2
            + 14.0 * ca2 * v7 * w2
            + 18.0 * ca3 * v7 * w2
            - 6.0 * cacf * v7 * w2
            - 24.0 * ca2cf * v7 * w2
            + 6.0 * v8 * w2
            + 21.0 * ca * v8 * w2
            - 6.0 * ca2 * v8 * w2
            - 27.0 * ca3 * v8 * w2
            + 6.0 * cacf * v8 * w2
            + 18.0 * ca2cf * v8 * w2
            - 9.0 * ca * v9 * w2
            + 9.0 * ca3 * v9 * w2
            - 6.0 * ca2cf * v9 * w2
            - 2.0 * v2 * w3
            + 3.0 * ca * v2 * w3
            - 10.0 * ca2 * v2 * w3
            - 5.0 * ca3 * v2 * w3
            + 8.0 * cacf * v2 * w3
            + 10.0 * ca2cf * v2 * w3
            + 8.0 * v3 * w3
            - 12.0 * ca * v3 * w3
            + 40.0 * ca2 * v3 * w3
            + 20.0 * ca3 * v3 * w3
            - 32.0 * cacf * v3 * w3
            - 40.0 * ca2cf * v3 * w3
            - 4.0 * v4 * w3
            + 37.0 * ca * v4 * w3
            - 55.0 * ca2 * v4 * w3
            - 41.0 * ca3 * v4 * w3
            + 62.0 * cacf * v4 * w3
            + 86.0 * ca2cf * v4 * w3
            - 16.0 * v5 * w3
            - 69.0 * ca * v5 * w3
            + 25.0 * ca2 * v5 * w3
            + 53.0 * ca3 * v5 * w3
            - 74.0 * cacf * v5 * w3
            - 118.0 * ca2cf * v5 * w3
            + 6.0 * v6 * w3
            - 9.0 * ca * v6 * w3
            + 9.0 * ca2 * v6 * w3
            + 4.0 * ca3 * v6 * w3
            + 40.0 * cacf * v6 * w3
            + 74.0 * ca2cf * v6 * w3
            + 24.0 * v7 * w3
            + 119.0 * ca * v7 * w3
            - 13.0 * ca2 * v7 * w3
            - 73.0 * ca3 * v7 * w3
            + 6.0 * cacf * v7 * w3
            + 2.0 * ca2cf * v7 * w3
            - 14.0 * v8 * w3
            - 84.0 * ca * v8 * w3
            + 2.0 * ca2 * v8 * w3
            + 51.0 * ca3 * v8 * w3
            - 8.0 * cacf * v8 * w3
            - 20.0 * ca2cf * v8 * w3
            - 2.0 * v9 * w3
            + 15.0 * ca * v9 * w3
            + 2.0 * ca2 * v9 * w3
            - 9.0 * ca3 * v9 * w3
            - 2.0 * cacf * v9 * w3
            + 6.0 * ca2cf * v9 * w3
            + 3.0 * ca * v10 * w3
            - 3.0 * ca3 * v10 * w3
            + 2.0 * ca2cf * v10 * w3
            - 3.0 * ca * v2 * w4
            + 3.0 * ca3 * v2 * w4
            - 6.0 * ca2cf * v2 * w4
            + 12.0 * ca * v3 * w4
            - 12.0 * ca3 * v3 * w4
            + 24.0 * ca2cf * v3 * w4
            - 14.0 * v4 * w4
            - 62.0 * ca * v4 * w4
            - 17.0 * ca2 * v4 * w4
            + 32.0 * ca3 * v4 * w4
            - 2.0 * cacf * v4 * w4
            - 52.0 * ca2cf * v4 * w4
            + 42.0 * v5 * w4
            + 144.0 * ca * v5 * w4
            + 51.0 * ca2 * v5 * w4
            - 54.0 * ca3 * v5 * w4
            + 6.0 * cacf * v5 * w4
            + 72.0 * ca2cf * v5 * w4
            - 21.0 * v6 * w4
            - 55.0 * ca * v6 * w4
            - 46.0 * ca2 * v6 * w4
            + 12.0 * ca3 * v6 * w4
            + 10.0 * cacf * v6 * w4
            - 38.0 * ca2cf * v6 * w4
            - 28.0 * v7 * w4
            - 116.0 * ca * v7 * w4
            + 7.0 * ca2 * v7 * w4
            + 52.0 * ca3 * v7 * w4
            - 30.0 * cacf * v7 * w4
            - 16.0 * ca2cf * v7 * w4
            + 14.0 * v8 * w4
            + 75.0 * ca * v8 * w4
            + 8.0 * ca2 * v8 * w4
            - 26.0 * ca3 * v8 * w4
            + 12.0 * cacf * v8 * w4
            + 18.0 * ca2cf * v8 * w4
            + 7.0 * v9 * w4
            + 5.0 * ca * v9 * w4
            - 3.0 * ca2 * v9 * w4
            - 7.0 * ca3 * v9 * w4
            + 4.0 * cacf * v9 * w4
            - 2.0 * ca2cf * v9 * w4
            - 11.0 * ca * v10 * w4
            + 9.0 * ca3 * v10 * w4
            - 6.0 * ca2cf * v10 * w4
            + 10.0 * v4 * w5
            + 38.0 * ca * v4 * w5
            + 20.0 * ca2 * v4 * w5
            + 6.0 * ca3 * v4 * w5
            - 12.0 * cacf * v4 * w5
            - 24.0 * ca2cf * v4 * w5
            - 30.0 * v5 * w5
            - 114.0 * ca * v5 * w5
            - 60.0 * ca2 * v5 * w5
            - 18.0 * ca3 * v5 * w5
            + 36.0 * cacf * v5 * w5
            + 72.0 * ca2cf * v5 * w5
            + 2.0 * v6 * w5
            + 25.0 * ca * v6 * w5
            + 48.0 * ca2 * v6 * w5
            + 48.0 * ca3 * v6 * w5
            - 54.0 * cacf * v6 * w5
            - 86.0 * ca2cf * v6 * w5
            + 46.0 * v7 * w5
            + 140.0 * ca * v7 * w5
            + 4.0 * ca2 * v7 * w5
            - 66.0 * ca3 * v7 * w5
            + 48.0 * cacf * v7 * w5
            + 52.0 * ca2cf * v7 * w5
            - 17.0 * v8 * w5
            - 62.0 * ca * v8 * w5
            - 10.0 * ca2 * v8 * w5
            + 17.0 * ca3 * v8 * w5
            - 16.0 * cacf * v8 * w5
            - 16.0 * ca2cf * v8 * w5
            - 11.0 * v9 * w5
            - 27.0 * ca * v9 * w5
            - 2.0 * ca2 * v9 * w5
            + 13.0 * ca3 * v9 * w5
            - 2.0 * cacf * v9 * w5
            + 2.0 * ca2cf * v9 * w5
            + 19.0 * ca * v10 * w5
            - 13.0 * ca3 * v10 * w5
            + 10.0 * ca2cf * v10 * w5
            + 3.0 * ca * v4 * w6
            - 3.0 * ca3 * v4 * w6
            + 6.0 * ca2cf * v4 * w6
            - 9.0 * ca * v5 * w6
            + 9.0 * ca3 * v5 * w6
            - 18.0 * ca2cf * v5 * w6
            + 27.0 * v6 * w6
            + 90.0 * ca * v6 * w6
            + 23.0 * ca2 * v6 * w6
            - 11.0 * ca3 * v6 * w6
            - 2.0 * cacf * v6 * w6
            - 16.0 * ca2cf * v6 * w6
            - 54.0 * v7 * w6
            - 165.0 * ca * v7 * w6
            - 46.0 * ca2 * v7 * w6
            + 7.0 * ca3 * v7 * w6
            + 4.0 * cacf * v7 * w6
            + 62.0 * ca2cf * v7 * w6
            + 14.0 * v8 * w6
            + 42.0 * ca * v8 * w6
            + 15.0 * ca2 * v8 * w6
            + 15.0 * ca3 * v8 * w6
            - 36.0 * ca2cf * v8 * w6
            + 13.0 * v9 * w6
            + 39.0 * ca * v9 * w6
            + 8.0 * ca2 * v9 * w6
            - 17.0 * ca3 * v9 * w6
            - 2.0 * cacf * v9 * w6
            + 2.0 * ca2cf * v9 * w6
            - 23.0 * ca * v10 * w6
            + 15.0 * ca3 * v10 * w6
            - 14.0 * ca2cf * v10 * w6
            - 10.0 * v6 * w7
            - 23.0 * ca * v6 * w7
            - 14.0 * ca2 * v6 * w7
            - 11.0 * ca3 * v6 * w7
            + 8.0 * cacf * v6 * w7
            + 30.0 * ca2cf * v6 * w7
            + 20.0 * v7 * w7
            + 46.0 * ca * v7 * w7
            + 28.0 * ca2 * v7 * w7
            + 22.0 * ca3 * v7 * w7
            - 16.0 * cacf * v7 * w7
            - 60.0 * ca2cf * v7 * w7
            + 3.0 * v8 * w7
            + 15.0 * ca * v8 * w7
            - 2.0 * ca2 * v8 * w7
            - 21.0 * ca3 * v8 * w7
            + 18.0 * ca2cf * v8 * w7
            - 13.0 * v9 * w7
            - 38.0 * ca * v9 * w7
            - 12.0 * ca2 * v9 * w7
            + 10.0 * ca3 * v9 * w7
            + 8.0 * cacf * v9 * w7
            + 12.0 * ca2cf * v9 * w7
            + 21.0 * ca * v10 * w7
            - 13.0 * ca3 * v10 * w7
            + 14.0 * ca2cf * v10 * w7
            - ca * v6 * w8
            + ca3 * v6 * w8
            - 2.0 * ca2cf * v6 * w8
            + 2.0 * ca * v7 * w8
            - 2.0 * ca3 * v7 * w8
            + 4.0 * ca2cf * v7 * w8
            - 8.0 * v8 * w8
            - 16.0 * ca * v8 * w8
            - 11.0 * ca2 * v8 * w8
            - 4.0 * ca3 * v8 * w8
            + 10.0 * cacf * v8 * w8
            + 20.0 * ca2cf * v8 * w8
            + 8.0 * v9 * w8
            + 15.0 * ca * v9 * w8
            + 11.0 * ca2 * v9 * w8
            + 5.0 * ca3 * v9 * w8
            - 10.0 * cacf * v9 * w8
            - 22.0 * ca2cf * v9 * w8
            - 13.0 * ca * v10 * w8
            + 7.0 * ca3 * v10 * w8
            - 10.0 * ca2cf * v10 * w8
            + 2.0 * v8 * w9
            + 4.0 * ca2 * v8 * w9
            + 4.0 * ca3 * v8 * w9
            - 4.0 * cacf * v8 * w9
            - 8.0 * ca2cf * v8 * w9
            - 2.0 * v9 * w9
            - 4.0 * ca2 * v9 * w9
            - 4.0 * ca3 * v9 * w9
            + 4.0 * cacf * v9 * w9
            + 8.0 * ca2cf * v9 * w9
            + 5.0 * ca * v10 * w9
            - 3.0 * ca3 * v10 * w9
            + 6.0 * ca2cf * v10 * w9
            - ca * v10 * w10
            + ca3 * v10 * w10
            - 2.0 * ca2cf * v10 * w10))
        / (ca
            * (1.0 - v).powi(2)
            * v
            * (1.0 - w)
            * w
            * (1.0 - v * w).powi(3)
            * (1.0 - v + v * w).powi(3));

    part1 + part2 + part3 + part4 + part5 + part6 + part7 + part8 + part9
}

/// `STRUV12(W,V,X3,S)`.
#[must_use]
pub fn struv12(w: f64, v: f64, _x3: f64, _s: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let cf = ctx.cf;
    let ca = ctx.ca;
    let nf = ctx.nf;
    let (v2, v3, v4, v5, v6, v7, v8, v9, v10, v11) = (
        pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8, pre.v9, pre.v10, pre.v11,
    );
    let (w2, w3, w4, w5, w6, w7, w8, w9, w10) = (
        pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8, pre.w9, pre.w10,
    );
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let ca4 = pre.ca4;
    let cacf = ca * cf;
    let ca3cf = ca3 * cf;
    let (l1v, lw, lvw, l1vw, lms, l1w, lv, lmss) = (
        pre.l1v, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.l1w, pre.lv, pre.lmss,
    );

    let part1 = (8.0
        * ca
        * cf.powi(2)
        * l1w
        * nf
        * (1.0 - 2.0 * v + v2 + v2 * w2)
        * (1.0 + v2 - 2.0 * v2 * w + v2 * w2))
        / (1.0 - v + v * w).powi(4);

    let part2 = -(8.0
        * ca
        * cf.powi(2)
        * lmss
        * nf
        * (1.0 - 2.0 * v + v2 + v2 * w2)
        * (1.0 + v2 - 2.0 * v2 * w + v2 * w2))
        / (1.0 - v + v * w).powi(4);

    let part3 = (8.0
        * ca
        * cf.powi(2)
        * lv
        * nf
        * (1.0 - 2.0 * v + v2 + v2 * w2)
        * (1.0 + v2 - 2.0 * v2 * w + v2 * w2))
        / (1.0 - v + v * w).powi(4);

    let part4 = (16.0
        * ca
        * cf.powi(2)
        * l1vw
        * nf
        * (1.0 - w)
        * (1.0 - 2.0 * v + v2 + v2 * w2)
        * (1.0 + v2 - 2.0 * v2 * w + v2 * w2))
        / (1.0 - v + v * w).powi(4);

    let part5 = (4.0
        * cf
        * lvw
        * (4.0 * ca3 + 4.0 * ca3 * v2 + 4.0 * w - 2.0 * ca2 * w - 4.0 * ca3 * w + 2.0 * ca4 * w
            - 6.0 * v * w
            - 2.0 * ca * v * w
            + 5.0 * ca2 * v * w
            - 8.0 * ca3 * v * w
            - 4.0 * ca4 * v * w
            + 4.0 * v2 * w
            - 4.0 * ca2 * v2 * w
            - 10.0 * ca3 * v2 * w
            + 3.0 * ca4 * v2 * w
            - 2.0 * v3 * w
            + 2.0 * ca * v3 * w
            + ca2 * v3 * w
            - 2.0 * ca3 * v3 * w
            - ca4 * v3 * w
            - 4.0 * v * w2
            + 2.0 * ca * v * w2
            + 5.0 * ca2 * v * w2
            + 8.0 * ca3 * v * w2
            - 8.0 * ca4 * v * w2
            + 3.0 * v2 * w2
            - ca * v2 * w2
            - 9.0 * ca2 * v2 * w2
            + 17.0 * ca3 * v2 * w2
            + 15.0 * ca4 * v2 * w2
            - 4.0 * ca * v3 * w2
            + 2.0 * ca2 * v3 * w2
            + 6.0 * ca3 * v3 * w2
            - 11.0 * ca4 * v3 * w2
            + v4 * w2
            - ca * v4 * w2
            + 2.0 * ca2 * v4 * w2
            + ca3 * v4 * w2
            + 4.0 * ca4 * v4 * w2
            + 4.0 * v2 * w3
            + ca * v2 * w3
            - 12.0 * ca2 * v2 * w3
            - 11.0 * ca3 * v2 * w3
            + 13.0 * ca4 * v2 * w3
            - 2.0 * v3 * w3
            + 4.0 * ca * v3 * w3
            + 23.0 * ca2 * v3 * w3
            - 10.0 * ca3 * v3 * w3
            - 20.0 * ca4 * v3 * w3
            - 2.0 * v4 * w3
            + 3.0 * ca * v4 * w3
            - 15.0 * ca2 * v4 * w3
            - 3.0 * ca3 * v4 * w3
            + 11.0 * ca4 * v4 * w3
            + 4.0 * ca2 * v5 * w3
            - 4.0 * ca4 * v5 * w3
            - v3 * w4
            - 2.0 * ca * v3 * w4
            + 6.0 * ca2 * v3 * w4
            + 6.0 * ca3 * v3 * w4
            - 9.0 * ca4 * v3 * w4
            + v4 * w4
            - 4.0 * ca * v4 * w4
            - 6.0 * ca2 * v4 * w4
            + 4.0 * ca3 * v4 * w4
            + 9.0 * ca4 * v4 * w4
            + 2.0 * ca * v4 * w5
            - 4.0 * ca2 * v4 * w5
            - 2.0 * ca3 * v4 * w5
            + 4.0 * ca4 * v4 * w5
            + 4.0 * ca2 * v5 * w5
            - 4.0 * ca4 * v5 * w5))
        / (ca * (1.0 - v).powi(2) * v2 * w2);

    let part6 = -(4.0
        * cf
        * nf
        * (-2.0 * cacf - 4.0 * v + 4.0 * ca2 * v + 7.0 * v2 - 7.0 * ca2 * v2 - 8.0 * v3
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

    let part7 = -(4.0
        * cf
        * l1v
        * (2.0 * ca3 - 4.0 * ca3 * v + 4.0 * ca3 * v2 - 4.0 * ca3 * v3 + 2.0 * ca3 * v4 + w
            - ca2 * w
            + 2.0 * ca4 * w
            - 2.0 * v * w
            + 2.0 * ca * v * w
            + 5.0 * ca2 * v * w
            - 4.0 * ca3 * v * w
            - 7.0 * ca4 * v * w
            + 2.0 * v2 * w
            - 2.0 * ca * v2 * w
            - 8.0 * ca2 * v2 * w
            + 4.0 * ca3 * v2 * w
            + 11.0 * ca4 * v2 * w
            - 4.0 * v3 * w
            - 2.0 * ca * v3 * w
            + 6.0 * ca2 * v3 * w
            + 2.0 * ca3 * v3 * w
            - 10.0 * ca4 * v3 * w
            + 5.0 * v4 * w
            + 2.0 * ca * v4 * w
            - 3.0 * ca2 * v4 * w
            + 5.0 * ca4 * v4 * w
            - 2.0 * v5 * w
            + ca2 * v5 * w
            - 2.0 * ca3 * v5 * w
            - ca4 * v5 * w
            - v * w2
            - ca2 * v * w2
            + 4.0 * v2 * w2
            - 6.0 * ca * v2 * w2
            + 7.0 * ca2 * v2 * w2
            + 4.0 * ca3 * v2 * w2
            - 7.0 * ca4 * v2 * w2
            - 2.0 * v3 * w2
            + 6.0 * ca * v3 * w2
            - 22.0 * ca2 * v3 * w2
            - 6.0 * ca3 * v3 * w2
            + 19.0 * ca4 * v3 * w2
            + 2.0 * v4 * w2
            + 2.0 * ca * v4 * w2
            + 34.0 * ca2 * v4 * w2
            - 6.0 * ca3 * v4 * w2
            - 17.0 * ca4 * v4 * w2
            - 5.0 * v5 * w2
            - 2.0 * ca * v5 * w2
            - 23.0 * ca2 * v5 * w2
            + 4.0 * ca3 * v5 * w2
            + 4.0 * ca4 * v5 * w2
            + 2.0 * v6 * w2
            + 5.0 * ca2 * v6 * w2
            + ca4 * v6 * w2
            - 3.0 * v2 * w3
            - ca2 * v2 * w3
            - 2.0 * ca4 * v2 * w3
            + 2.0 * v3 * w3
            + 6.0 * ca * v3 * w3
            + 2.0 * ca2 * v3 * w3
            - 2.0 * ca3 * v3 * w3
            + 16.0 * ca4 * v3 * w3
            - 6.0 * v4 * w3
            - 6.0 * ca * v4 * w3
            + 2.0 * ca2 * v4 * w3
            + 10.0 * ca3 * v4 * w3
            - 33.0 * ca4 * v4 * w3
            + 8.0 * v5 * w3
            - 14.0 * ca2 * v5 * w3
            + 28.0 * ca4 * v5 * w3
            - v6 * w3
            + 15.0 * ca2 * v6 * w3
            - 9.0 * ca4 * v6 * w3
            - 4.0 * ca2 * v7 * w3
            - v3 * w4
            - ca2 * v3 * w4
            + 8.0 * v4 * w4
            - 2.0 * ca * v4 * w4
            + ca2 * v4 * w4
            - 2.0 * ca3 * v4 * w4
            - ca4 * v4 * w4
            - 6.0 * v5 * w4
            + 2.0 * ca * v5 * w4
            + 5.0 * ca2 * v5 * w4
            - 4.0 * ca3 * v5 * w4
            + 2.0 * ca4 * v5 * w4
            - v6 * w4
            - 5.0 * ca2 * v6 * w4
            - ca4 * v6 * w4
            - 2.0 * v5 * w5
            + 3.0 * ca2 * v5 * w5
            + 2.0 * ca3 * v5 * w5
            - ca4 * v5 * w5
            + 2.0 * v6 * w5
            - 7.0 * ca2 * v6 * w5
            + ca4 * v6 * w5
            + 4.0 * ca2 * v7 * w5))
        / (ca * (1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w) * (1.0 - v + v * w));

    let part8 = -(4.0
        * cf
        * lw
        * (2.0 * ca - 4.0 * ca3 - 6.0 * ca * v + 6.0 * ca3 * v + 8.0 * ca * v2
            - 8.0 * ca3 * v2
            - 8.0 * ca * v3
            + 8.0 * ca3 * v3
            + 6.0 * ca * v4
            - 4.0 * ca3 * v4
            - 2.0 * ca * v5
            + 2.0 * ca3 * v5
            - 2.0 * w
            - 2.0 * ca * w
            + 5.0 * ca2 * w
            + 4.0 * ca3 * w
            - 3.0 * ca4 * w
            + 7.0 * v * w
            + 6.0 * ca * v * w
            - 20.0 * ca2 * v * w
            + 13.0 * ca4 * v * w
            - 10.0 * v2 * w
            - 8.0 * ca * v2 * w
            + 37.0 * ca2 * v2 * w
            + 4.0 * ca3 * v2 * w
            - 27.0 * ca4 * v2 * w
            + 8.0 * v3 * w
            + 10.0 * ca * v3 * w
            - 43.0 * ca2 * v3 * w
            - 14.0 * ca3 * v3 * w
            + 35.0 * ca4 * v3 * w
            - 4.0 * v4 * w
            - 8.0 * ca * v4 * w
            + 34.0 * ca2 * v4 * w
            + 2.0 * ca3 * v4 * w
            - 30.0 * ca4 * v4 * w
            + v5 * w
            - 17.0 * ca2 * v5 * w
            - 2.0 * ca3 * v5 * w
            + 16.0 * ca4 * v5 * w
            + 2.0 * ca * v6 * w
            + 4.0 * ca2 * v6 * w
            - 2.0 * ca3 * v6 * w
            - 4.0 * ca4 * v6 * w
            + w2
            - ca2 * w2
            - 6.0 * v * w2
            + 2.0 * ca2 * v * w2
            - 6.0 * ca3 * v * w2
            + 5.0 * ca4 * v * w2
            + 8.0 * v2 * w2
            + 3.0 * ca * v2 * w2
            - 3.0 * ca2 * v2 * w2
            - ca3 * v2 * w2
            - 14.0 * ca4 * v2 * w2
            - 5.0 * v3 * w2
            - ca * v3 * w2
            + 8.0 * ca2 * v3 * w2
            + 17.0 * ca3 * v3 * w2
            + 19.0 * ca4 * v3 * w2
            - 5.0 * ca * v4 * w2
            - 7.0 * ca2 * v4 * w2
            + 15.0 * ca3 * v4 * w2
            - 20.0 * ca4 * v4 * w2
            + 3.0 * v5 * w2
            + 9.0 * ca * v5 * w2
            - 4.0 * ca2 * v5 * w2
            - 3.0 * ca3 * v5 * w2
            + 18.0 * ca4 * v5 * w2
            - v6 * w2
            - 6.0 * ca * v6 * w2
            + 9.0 * ca2 * v6 * w2
            + 6.0 * ca3 * v6 * w2
            - 12.0 * ca4 * v6 * w2
            - 4.0 * ca2 * v7 * w2
            + 4.0 * ca4 * v7 * w2
            + v * w3
            + ca2 * v * w3
            - v2 * w3
            - 3.0 * ca * v2 * w3
            - 5.0 * ca2 * v2 * w3
            + 5.0 * ca3 * v2 * w3
            - 7.0 * ca4 * v2 * w3
            + 4.0 * v3 * w3
            - 9.0 * ca * v3 * w3
            + 7.0 * ca2 * v3 * w3
            - 11.0 * ca3 * v3 * w3
            + 14.0 * ca4 * v3 * w3
            + v4 * w3
            + 14.0 * ca * v4 * w3
            - 15.0 * ca2 * v4 * w3
            - 30.0 * ca3 * v4 * w3
            - 7.0 * ca4 * v4 * w3
            - 5.0 * v5 * w3
            - 9.0 * ca * v5 * w3
            + 27.0 * ca2 * v5 * w3
            + 3.0 * ca3 * v5 * w3
            - 4.0 * ca4 * v5 * w3
            + 7.0 * ca * v6 * w3
            - 23.0 * ca2 * v6 * w3
            - 7.0 * ca3 * v6 * w3
            + 8.0 * ca4 * v6 * w3
            + 8.0 * ca2 * v7 * w3
            - 4.0 * ca4 * v7 * w3
            + v2 * w4
            + ca2 * v2 * w4
            - 6.0 * v3 * w4
            + 8.0 * ca * v3 * w4
            + ca2 * v3 * w4
            + 6.0 * ca4 * v3 * w4
            - v4 * w4
            - 4.0 * ca * v4 * w4
            + 6.0 * ca2 * v4 * w4
            + 22.0 * ca3 * v4 * w4
            - 12.0 * ca4 * v4 * w4
            + 3.0 * v5 * w4
            - 21.0 * ca2 * v5 * w4
            + 4.0 * ca3 * v5 * w4
            + 9.0 * ca4 * v5 * w4
            + 3.0 * v6 * w4
            - 4.0 * ca * v6 * w4
            + 21.0 * ca2 * v6 * w4
            + 4.0 * ca3 * v6 * w4
            - 7.0 * ca4 * v6 * w4
            - 8.0 * ca2 * v7 * w4
            + 4.0 * ca4 * v7 * w4
            + v3 * w5
            + ca2 * v3 * w5
            + 3.0 * v4 * w5
            - 3.0 * ca * v4 * w5
            - 10.0 * ca2 * v4 * w5
            - 5.0 * ca3 * v4 * w5
            + 2.0 * ca * v5 * w5
            + 21.0 * ca2 * v5 * w5
            - 6.0 * ca3 * v5 * w5
            - 2.0 * ca4 * v5 * w5
            - 4.0 * v6 * w5
            + ca * v6 * w5
            - 20.0 * ca2 * v6 * w5
            - ca3 * v6 * w5
            + 6.0 * ca4 * v6 * w5
            + 8.0 * ca2 * v7 * w5
            - 4.0 * ca4 * v7 * w5
            + 2.0 * ca2 * v4 * w6
            - 2.0 * v5 * w6
            - 7.0 * ca2 * v5 * w6
            + 2.0 * ca3 * v5 * w6
            - ca4 * v5 * w6
            + 2.0 * v6 * w6
            + 9.0 * ca2 * v6 * w6
            + ca4 * v6 * w6
            - 4.0 * ca2 * v7 * w6))
        / (ca * (1.0 - v).powi(2) * v2 * (1.0 - w) * w2 * (1.0 - v * w) * (1.0 - v + v * w));

    let part9 = (2.0
        * cf
        * lms
        * (2.0 - 6.0 * ca2 - 4.0 * v + 4.0 * v2 - 8.0 * ca2 * v2 - 4.0 * v3 + 2.0 * v4
            - 2.0 * ca2 * v4
            - 2.0 * w
            + 2.0 * ca2 * w
            + 6.0 * cf * w
            - 6.0 * ca2 * cf * w
            + 20.0 * ca2 * v * w
            - 14.0 * cf * v * w
            + 18.0 * ca2 * cf * v * w
            + 4.0 * v2 * w
            + 12.0 * ca2 * v2 * w
            + 14.0 * cf * v2 * w
            - 30.0 * ca2 * cf * v2 * w
            - 4.0 * v3 * w
            + 16.0 * ca2 * v3 * w
            - 6.0 * cf * v3 * w
            + 34.0 * ca2 * cf * v3 * w
            + 6.0 * v4 * w
            + 2.0 * ca2 * v4 * w
            - 24.0 * ca2 * cf * v4 * w
            - 4.0 * v5 * w
            + 4.0 * ca2 * v5 * w
            + 8.0 * ca2 * cf * v5 * w
            + w2
            - ca2 * w2
            - 2.0 * cf * w2
            + 2.0 * ca2 * cf * w2
            + 2.0 * v * w2
            - 4.0 * ca2 * v * w2
            - 6.0 * cf * v * w2
            + 2.0 * ca2 * cf * v * w2
            - 2.0 * v2 * w2
            - 36.0 * ca2 * v2 * w2
            + 14.0 * cf * v2 * w2
            + 10.0 * ca2 * cf * v2 * w2
            + 2.0 * v3 * w2
            - 24.0 * ca2 * v3 * w2
            - 18.0 * cf * v3 * w2
            - 22.0 * ca2 * cf * v3 * w2
            - v4 * w2
            - 13.0 * ca2 * v4 * w2
            + 12.0 * cf * v4 * w2
            - 4.0 * ca2 * v5 * w2
            + 24.0 * ca2 * cf * v5 * w2
            + 2.0 * v6 * w2
            - 2.0 * ca2 * v6 * w2
            - 16.0 * ca2 * cf * v6 * w2
            - 2.0 * v * w3
            + 2.0 * ca2 * v * w3
            + 4.0 * cf * v * w3
            - 4.0 * ca2 * cf * v * w3
            + 2.0 * v2 * w3
            + 2.0 * ca2 * v2 * w3
            - 6.0 * cf * v2 * w3
            + 14.0 * ca2 * cf * v2 * w3
            - 6.0 * v3 * w3
            + 40.0 * ca2 * v3 * w3
            + 10.0 * cf * v3 * w3
            - 90.0 * ca2 * cf * v3 * w3
            - 4.0 * v4 * w3
            + 20.0 * ca2 * v4 * w3
            + 160.0 * ca2 * cf * v4 * w3
            + 6.0 * ca2 * v5 * w3
            - 8.0 * cf * v5 * w3
            - 112.0 * ca2 * cf * v5 * w3
            - 2.0 * v6 * w3
            + 2.0 * ca2 * v6 * w3
            + 24.0 * ca2 * cf * v6 * w3
            + 8.0 * ca2 * cf * v7 * w3
            + v2 * w4
            - ca2 * v2 * w4
            - 2.0 * cf * v2 * w4
            + 2.0 * ca2 * cf * v2 * w4
            - 2.0 * v3 * w4
            + 6.0 * cf * v3 * w4
            - 10.0 * ca2 * cf * v3 * w4
            + 11.0 * v4 * w4
            - 27.0 * ca2 * v4 * w4
            - 12.0 * cf * v4 * w4
            + 88.0 * ca2 * cf * v4 * w4
            + 2.0 * v5 * w4
            - 8.0 * ca2 * v5 * w4
            + 8.0 * cf * v5 * w4
            - 144.0 * ca2 * cf * v5 * w4
            + 2.0 * v6 * w4
            - 2.0 * ca2 * v6 * w4
            + 88.0 * ca2 * cf * v6 * w4
            - 24.0 * ca2 * cf * v7 * w4
            - 6.0 * v5 * w5
            + 10.0 * ca2 * v5 * w5
            - 32.0 * ca2 * cf * v5 * w5
            - 2.0 * v6 * w5
            + 2.0 * ca2 * v6 * w5
            + 40.0 * ca2 * cf * v6 * w5
            - 8.0 * ca2 * cf * v7 * w5
            + 2.0 * v6 * w6
            - 2.0 * ca2 * v6 * w6
            + 8.0 * ca2 * cf * v6 * w6
            - 8.0 * ca2 * cf * v7 * w6))
        / ((1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w).powi(2));

    let part10 = -(4.0
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

    let part11 = (4.0
        * cf
        * lmss
        * (ca - ca3 - 6.0 * ca * v + 6.0 * ca3 * v + 16.0 * ca * v2
            - 16.0 * ca3 * v2
            - 26.0 * ca * v3
            + 26.0 * ca3 * v3
            + 30.0 * ca * v4
            - 30.0 * ca3 * v4
            - 26.0 * ca * v5
            + 26.0 * ca3 * v5
            + 16.0 * ca * v6
            - 16.0 * ca3 * v6
            - 6.0 * ca * v7
            + 6.0 * ca3 * v7
            + ca * v8
            - ca3 * v8
            - 4.0 * ca3cf * w
            + v * w
            + 3.0 * ca * v * w
            - 5.0 * ca2 * v * w
            - 3.0 * ca3 * v * w
            + 4.0 * ca4 * v * w
            + 20.0 * ca3cf * v * w
            - 5.0 * v2 * w
            - 17.0 * ca * v2 * w
            + 31.0 * ca2 * v2 * w
            + 17.0 * ca3 * v2 * w
            - 26.0 * ca4 * v2 * w
            - 40.0 * ca3cf * v2 * w
            + 11.0 * v3 * w
            + 45.0 * ca * v3 * w
            - 85.0 * ca2 * v3 * w
            - 45.0 * ca3 * v3 * w
            + 74.0 * ca4 * v3 * w
            + 40.0 * ca3cf * v3 * w
            - 15.0 * v4 * w
            - 75.0 * ca * v4 * w
            + 137.0 * ca2 * v4 * w
            + 75.0 * ca3 * v4 * w
            - 122.0 * ca4 * v4 * w
            - 20.0 * ca3cf * v4 * w
            + 15.0 * v5 * w
            + 85.0 * ca * v5 * w
            - 145.0 * ca2 * v5 * w
            - 85.0 * ca3 * v5 * w
            + 130.0 * ca4 * v5 * w
            + 4.0 * ca3cf * v5 * w
            - 11.0 * v6 * w
            - 63.0 * ca * v6 * w
            + 105.0 * ca2 * v6 * w
            + 63.0 * ca3 * v6 * w
            - 94.0 * ca4 * v6 * w
            + 5.0 * v7 * w
            + 27.0 * ca * v7 * w
            - 51.0 * ca2 * v7 * w
            - 27.0 * ca3 * v7 * w
            + 46.0 * ca4 * v7 * w
            - v8 * w
            - 5.0 * ca * v8 * w
            + 15.0 * ca2 * v8 * w
            + 5.0 * ca3 * v8 * w
            - 14.0 * ca4 * v8 * w
            - 2.0 * ca2 * v9 * w
            + 2.0 * ca4 * v9 * w
            - 16.0 * ca3cf * v * w2
            + v2 * w2
            + 5.0 * ca * v2 * w2
            - 15.0 * ca2 * v2 * w2
            - 5.0 * ca3 * v2 * w2
            + 22.0 * ca4 * v2 * w2
            + 80.0 * ca3cf * v2 * w2
            - 6.0 * v3 * w2
            - 28.0 * ca * v3 * w2
            + 88.0 * ca2 * v3 * w2
            + 28.0 * ca3 * v3 * w2
            - 130.0 * ca4 * v3 * w2
            - 168.0 * ca3cf * v3 * w2
            + 17.0 * v4 * w2
            + 73.0 * ca * v4 * w2
            - 227.0 * ca2 * v4 * w2
            - 73.0 * ca3 * v4 * w2
            + 338.0 * ca4 * v4 * w2
            + 192.0 * ca3cf * v4 * w2
            - 28.0 * v5 * w2
            - 112.0 * ca * v5 * w2
            + 338.0 * ca2 * v5 * w2
            + 112.0 * ca3 * v5 * w2
            - 506.0 * ca4 * v5 * w2
            - 128.0 * ca3cf * v5 * w2
            + 27.0 * v6 * w2
            + 103.0 * ca * v6 * w2
            - 317.0 * ca2 * v6 * w2
            - 103.0 * ca3 * v6 * w2
            + 474.0 * ca4 * v6 * w2
            + 48.0 * ca3cf * v6 * w2
            - 14.0 * v7 * w2
            - 52.0 * ca * v7 * w2
            + 188.0 * ca2 * v7 * w2
            + 52.0 * ca3 * v7 * w2
            - 278.0 * ca4 * v7 * w2
            - 8.0 * ca3cf * v7 * w2
            + 3.0 * v8 * w2
            + 11.0 * ca * v8 * w2
            - 65.0 * ca2 * v8 * w2
            - 11.0 * ca3 * v8 * w2
            + 94.0 * ca4 * v8 * w2
            + 10.0 * ca2 * v9 * w2
            - 14.0 * ca4 * v9 * w2
            - 24.0 * ca3cf * v2 * w3
            + v3 * w3
            + 7.0 * ca * v3 * w3
            - 25.0 * ca2 * v3 * w3
            - 7.0 * ca3 * v3 * w3
            + 32.0 * ca4 * v3 * w3
            + 112.0 * ca3cf * v3 * w3
            - 7.0 * v4 * w3
            - 35.0 * ca * v4 * w3
            + 133.0 * ca2 * v4 * w3
            + 35.0 * ca3 * v4 * w3
            - 182.0 * ca4 * v4 * w3
            - 216.0 * ca3cf * v4 * w3
            + 18.0 * v5 * w3
            + 78.0 * ca * v5 * w3
            - 304.0 * ca2 * v5 * w3
            - 78.0 * ca3 * v5 * w3
            + 442.0 * ca4 * v5 * w3
            + 216.0 * ca3cf * v5 * w3
            - 22.0 * v6 * w3
            - 94.0 * ca * v6 * w3
            + 386.0 * ca2 * v6 * w3
            + 94.0 * ca3 * v6 * w3
            - 588.0 * ca4 * v6 * w3
            - 112.0 * ca3cf * v6 * w3
            + 13.0 * v7 * w3
            + 59.0 * ca * v7 * w3
            - 289.0 * ca2 * v7 * w3
            - 59.0 * ca3 * v7 * w3
            + 452.0 * ca4 * v7 * w3
            + 24.0 * ca3cf * v7 * w3
            - 3.0 * v8 * w3
            - 15.0 * ca * v8 * w3
            + 121.0 * ca2 * v8 * w3
            + 15.0 * ca3 * v8 * w3
            - 190.0 * ca4 * v8 * w3
            - 22.0 * ca2 * v9 * w3
            + 34.0 * ca4 * v9 * w3
            - 16.0 * ca3cf * v3 * w4
            + 2.0 * v4 * w4
            + 8.0 * ca * v4 * w4
            - 32.0 * ca2 * v4 * w4
            - 8.0 * ca3 * v4 * w4
            + 30.0 * ca4 * v4 * w4
            + 64.0 * ca3cf * v4 * w4
            - 6.0 * v5 * w4
            - 32.0 * ca * v5 * w4
            + 140.0 * ca2 * v5 * w4
            + 32.0 * ca3 * v5 * w4
            - 138.0 * ca4 * v5 * w4
            - 104.0 * ca3cf * v5 * w4
            + 8.0 * v6 * w4
            + 56.0 * ca * v6 * w4
            - 260.0 * ca2 * v6 * w4
            - 56.0 * ca3 * v6 * w4
            + 276.0 * ca4 * v6 * w4
            + 80.0 * ca3cf * v6 * w4
            - 6.0 * v7 * w4
            - 48.0 * ca * v7 * w4
            + 258.0 * ca2 * v7 * w4
            + 48.0 * ca3 * v7 * w4
            - 296.0 * ca4 * v7 * w4
            - 24.0 * ca3cf * v7 * w4
            + 2.0 * v8 * w4
            + 16.0 * ca * v8 * w4
            - 136.0 * ca2 * v8 * w4
            - 16.0 * ca3 * v8 * w4
            + 166.0 * ca4 * v8 * w4
            + 30.0 * ca2 * v9 * w4
            - 38.0 * ca4 * v9 * w4
            - 4.0 * ca3cf * v4 * w5
            + v5 * w5
            + 7.0 * ca * v5 * w5
            - 29.0 * ca2 * v5 * w5
            - 7.0 * ca3 * v5 * w5
            + 16.0 * ca4 * v5 * w5
            + 12.0 * ca3cf * v5 * w5
            - 3.0 * v6 * w5
            - 23.0 * ca * v6 * w5
            + 105.0 * ca2 * v6 * w5
            + 23.0 * ca3 * v6 * w5
            - 62.0 * ca4 * v6 * w5
            - 16.0 * ca3cf * v6 * w5
            + 5.0 * v7 * w5
            + 31.0 * ca * v7 * w5
            - 155.0 * ca2 * v7 * w5
            - 31.0 * ca3 * v7 * w5
            + 98.0 * ca4 * v7 * w5
            + 8.0 * ca3cf * v7 * w5
            - 3.0 * v8 * w5
            - 15.0 * ca * v8 * w5
            + 109.0 * ca2 * v8 * w5
            + 15.0 * ca3 * v8 * w5
            - 74.0 * ca4 * v8 * w5
            - 30.0 * ca2 * v9 * w5
            + 22.0 * ca4 * v9 * w5
            + v6 * w6
            + 5.0 * ca * v6 * w6
            - 19.0 * ca2 * v6 * w6
            - 5.0 * ca3 * v6 * w6
            + 10.0 * ca4 * v6 * w6
            - 4.0 * v7 * w6
            - 14.0 * ca * v7 * w6
            + 58.0 * ca2 * v7 * w6
            + 14.0 * ca3 * v7 * w6
            - 26.0 * ca4 * v7 * w6
            + 3.0 * v8 * w6
            + 11.0 * ca * v8 * w6
            - 61.0 * ca2 * v8 * w6
            - 11.0 * ca3 * v8 * w6
            + 26.0 * ca4 * v8 * w6
            + 22.0 * ca2 * v9 * w6
            - 10.0 * ca4 * v9 * w6
            + v7 * w7
            + 3.0 * ca * v7 * w7
            - 9.0 * ca2 * v7 * w7
            - 3.0 * ca3 * v7 * w7
            + 4.0 * ca4 * v7 * w7
            - v8 * w7
            - 5.0 * ca * v8 * w7
            + 19.0 * ca2 * v8 * w7
            + 5.0 * ca3 * v8 * w7
            - 10.0 * ca4 * v8 * w7
            - 10.0 * ca2 * v9 * w7
            + 6.0 * ca4 * v9 * w7
            + ca * v8 * w8
            - 2.0 * ca2 * v8 * w8
            - ca3 * v8 * w8
            + 2.0 * ca4 * v8 * w8
            + 2.0 * ca2 * v9 * w8
            - 2.0 * ca4 * v9 * w8))
        / (ca * (1.0 - v).powi(2) * v2 * w2 * (1.0 - v + v * w).powi(4));

    let part12 = -(2.0
        * cf
        * (6.0 * ca - 18.0 * ca3 - 48.0 * ca * v + 120.0 * ca3 * v + 168.0 * ca * v2
            - 348.0 * ca3 * v2
            - 336.0 * ca * v3
            + 576.0 * ca3 * v3
            + 420.0 * ca * v4
            - 600.0 * ca3 * v4
            - 336.0 * ca * v5
            + 408.0 * ca3 * v5
            + 168.0 * ca * v6
            - 180.0 * ca3 * v6
            - 48.0 * ca * v7
            + 48.0 * ca3 * v7
            + 6.0 * ca * v8
            - 6.0 * ca3 * v8
            - 9.0 * w
            + 3.0 * ca * w
            + 7.0 * ca2 * w
            - 3.0 * ca3 * w
            + 2.0 * ca4 * w
            + 57.0 * v * w
            + 3.0 * ca * v * w
            - 59.0 * ca2 * v * w
            - 27.0 * ca3 * v * w
            + 2.0 * ca4 * v * w
            - 177.0 * v2 * w
            - 81.0 * ca * v2 * w
            + 239.0 * ca2 * v2 * w
            + 177.0 * ca3 * v2 * w
            - 62.0 * ca4 * v2 * w
            + 351.0 * v3 * w
            + 249.0 * ca * v3 * w
            - 585.0 * ca2 * v3 * w
            - 369.0 * ca3 * v3 * w
            + 234.0 * ca4 * v3 * w
            - 465.0 * v4 * w
            - 333.0 * ca * v4 * w
            + 927.0 * ca2 * v4 * w
            + 333.0 * ca3 * v4 * w
            - 462.0 * ca4 * v4 * w
            + 399.0 * v5 * w
            + 177.0 * ca * v5 * w
            - 977.0 * ca2 * v5 * w
            - 57.0 * ca3 * v5 * w
            + 578.0 * ca4 * v5 * w
            - 207.0 * v6 * w
            + 57.0 * ca * v6 * w
            + 689.0 * ca2 * v6 * w
            - 153.0 * ca3 * v6 * w
            - 482.0 * ca4 * v6 * w
            + 57.0 * v7 * w
            - 129.0 * ca * v7 * w
            - 319.0 * ca2 * v7 * w
            + 153.0 * ca3 * v7 * w
            + 262.0 * ca4 * v7 * w
            - 6.0 * v8 * w
            + 66.0 * ca * v8 * w
            + 90.0 * ca2 * v8 * w
            - 66.0 * ca3 * v8 * w
            - 84.0 * ca4 * v8 * w
            - 12.0 * ca * v9 * w
            - 12.0 * ca2 * v9 * w
            + 12.0 * ca3 * v9 * w
            + 12.0 * ca4 * v9 * w
            - 3.0 * ca * w2
            + 6.0 * ca2 * w2
            + 3.0 * ca3 * w2
            - 6.0 * ca4 * w2
            - 27.0 * v * w2
            + 15.0 * ca * v * w2
            - 37.0 * ca2 * v * w2
            - 27.0 * ca3 * v * w2
            + 52.0 * ca4 * v * w2
            + 135.0 * v2 * w2
            + 9.0 * ca * v2 * w2
            + 62.0 * ca2 * v2 * w2
            + 63.0 * ca3 * v2 * w2
            - 237.0 * ca4 * v2 * w2
            - 273.0 * v3 * w2
            - 81.0 * ca * v3 * w2
            + 80.0 * ca2 * v3 * w2
            - 135.0 * ca3 * v3 * w2
            + 639.0 * ca4 * v3 * w2
            + 225.0 * v4 * w2
            - 39.0 * ca * v4 * w2
            - 411.0 * ca2 * v4 * w2
            + 399.0 * ca3 * v4 * w2
            - 1136.0 * ca4 * v4 * w2
            + 87.0 * v5 * w2
            + 465.0 * ca * v5 * w2
            + 539.0 * ca2 * v5 * w2
            - 765.0 * ca3 * v5 * w2
            + 1466.0 * ca4 * v5 * w2
            - 351.0 * v6 * w2
            - 747.0 * ca * v6 * w2
            - 224.0 * ca2 * v6 * w2
            + 819.0 * ca3 * v6 * w2
            - 1421.0 * ca4 * v6 * w2
            + 297.0 * v7 * w2
            + 549.0 * ca * v7 * w2
            - 158.0 * ca2 * v7 * w2
            - 501.0 * ca3 * v7 * w2
            + 1003.0 * ca4 * v7 * w2
            - 105.0 * v8 * w2
            - 186.0 * ca * v8 * w2
            + 239.0 * ca2 * v8 * w2
            + 162.0 * ca3 * v8 * w2
            - 488.0 * ca4 * v8 * w2
            + 12.0 * v9 * w2
            + 12.0 * ca * v9 * w2
            - 120.0 * ca2 * v9 * w2
            - 12.0 * ca3 * v9 * w2
            + 152.0 * ca4 * v9 * w2
            + 6.0 * ca * v10 * w2
            + 24.0 * ca2 * v10 * w2
            - 6.0 * ca3 * v10 * w2
            - 24.0 * ca4 * v10 * w2
            - 6.0 * ca * v * w3
            + 12.0 * ca2 * v * w3
            + 6.0 * ca3 * v * w3
            - 12.0 * ca4 * v * w3
            - 9.0 * v2 * w3
            + 3.0 * ca * v2 * w3
            - 85.0 * ca2 * v2 * w3
            - 27.0 * ca3 * v2 * w3
            + 70.0 * ca4 * v2 * w3
            - 63.0 * v3 * w3
            + 21.0 * ca * v3 * w3
            + 238.0 * ca2 * v3 * w3
            + 51.0 * ca3 * v3 * w3
            - 61.0 * ca4 * v3 * w3
            + 450.0 * v4 * w3
            + 174.0 * ca * v4 * w3
            - 435.0 * ca2 * v4 * w3
            - 198.0 * ca3 * v4 * w3
            - 179.0 * ca4 * v4 * w3
            - 1044.0 * v5 * w3
            - 720.0 * ca * v5 * w3
            + 804.0 * ca2 * v5 * w3
            + 540.0 * ca3 * v5 * w3
            + 472.0 * ca4 * v5 * w3
            + 1185.0 * v6 * w3
            + 1005.0 * ca * v6 * w3
            - 1341.0 * ca2 * v6 * w3
            - 693.0 * ca3 * v6 * w3
            - 636.0 * ca4 * v6 * w3
            - 651.0 * v7 * w3
            - 615.0 * ca * v7 * w3
            + 1494.0 * ca2 * v7 * w3
            + 423.0 * ca3 * v7 * w3
            + 663.0 * ca4 * v7 * w3
            + 108.0 * v8 * w3
            + 96.0 * ca * v8 * w3
            - 989.0 * ca2 * v8 * w3
            - 72.0 * ca3 * v8 * w3
            - 467.0 * ca4 * v8 * w3
            + 30.0 * v9 * w3
            + 72.0 * ca * v9 * w3
            + 344.0 * ca2 * v9 * w3
            - 60.0 * ca3 * v9 * w3
            + 190.0 * ca4 * v9 * w3
            - 6.0 * v10 * w3
            - 30.0 * ca * v10 * w3
            - 30.0 * ca2 * v10 * w3
            + 30.0 * ca3 * v10 * w3
            - 52.0 * ca4 * v10 * w3
            - 12.0 * ca2 * v11 * w3
            + 12.0 * ca4 * v11 * w3
            + 3.0 * ca * v2 * w4
            - 6.0 * ca2 * v2 * w4
            - 3.0 * ca3 * v2 * w4
            + 6.0 * ca4 * v2 * w4
            + 45.0 * v3 * w4
            - 33.0 * ca * v3 * w4
            + 47.0 * ca2 * v3 * w4
            + 45.0 * ca3 * v3 * w4
            - 80.0 * ca4 * v3 * w4
            - 303.0 * v4 * w4
            - 42.0 * ca * v4 * w4
            - 83.0 * ca2 * v4 * w4
            - 102.0 * ca3 * v4 * w4
            + 240.0 * ca4 * v4 * w4
            + 702.0 * v5 * w4
            + 396.0 * ca * v5 * w4
            - 198.0 * ca2 * v5 * w4
            - 48.0 * ca3 * v5 * w4
            - 588.0 * ca4 * v5 * w4
            - 660.0 * v6 * w4
            - 597.0 * ca * v6 * w4
            + 922.0 * ca2 * v6 * w4
            + 285.0 * ca3 * v6 * w4
            + 1126.0 * ca4 * v6 * w4
            + 57.0 * v7 * w4
            + 243.0 * ca * v7 * w4
            - 1337.0 * ca2 * v7 * w4
            - 195.0 * ca3 * v7 * w4
            - 1222.0 * ca4 * v7 * w4
            + 321.0 * v8 * w4
            + 180.0 * ca * v8 * w4
            + 869.0 * ca2 * v8 * w4
            - 84.0 * ca3 * v8 * w4
            + 654.0 * ca4 * v8 * w4
            - 180.0 * v9 * w4
            - 222.0 * ca * v9 * w4
            - 148.0 * ca2 * v9 * w4
            + 174.0 * ca3 * v9 * w4
            - 150.0 * ca4 * v9 * w4
            + 18.0 * v10 * w4
            + 72.0 * ca * v10 * w4
            - 126.0 * ca2 * v10 * w4
            - 72.0 * ca3 * v10 * w4
            + 30.0 * ca4 * v10 * w4
            + 60.0 * ca2 * v11 * w4
            - 16.0 * ca4 * v11 * w4
            + 12.0 * ca * v3 * w5
            - 24.0 * ca2 * v3 * w5
            - 12.0 * ca3 * v3 * w5
            + 24.0 * ca4 * v3 * w5
            + 45.0 * v4 * w5
            - 15.0 * ca * v4 * w5
            + 149.0 * ca2 * v4 * w5
            + 63.0 * ca3 * v4 * w5
            - 146.0 * ca4 * v4 * w5
            - 111.0 * v5 * w5
            - 93.0 * ca * v5 * w5
            - 245.0 * ca2 * v5 * w5
            - 27.0 * ca3 * v5 * w5
            + 408.0 * ca4 * v5 * w5
            - 75.0 * v6 * w5
            + 165.0 * ca * v6 * w5
            - 45.0 * ca2 * v6 * w5
            - 165.0 * ca3 * v6 * w5
            - 248.0 * ca4 * v6 * w5
            + 519.0 * v7 * w5
            + 57.0 * ca * v7 * w5
            + 419.0 * ca2 * v7 * w5
            + 147.0 * ca3 * v7 * w5
            - 600.0 * ca4 * v7 * w5
            - 630.0 * v8 * w5
            - 342.0 * ca * v8 * w5
            - 208.0 * ca2 * v8 * w5
            + 126.0 * ca3 * v8 * w5
            + 1106.0 * ca4 * v8 * w5
            + 270.0 * v9 * w5
            + 330.0 * ca * v9 * w5
            - 280.0 * ca2 * v9 * w5
            - 246.0 * ca3 * v9 * w5
            - 748.0 * ca4 * v9 * w5
            - 18.0 * v10 * w5
            - 114.0 * ca * v10 * w5
            + 366.0 * ca2 * v10 * w5
            + 114.0 * ca3 * v10 * w5
            + 204.0 * ca4 * v10 * w5
            - 132.0 * ca2 * v11 * w5
            + 3.0 * ca * v4 * w6
            - 6.0 * ca2 * v4 * w6
            - 3.0 * ca3 * v4 * w6
            + 6.0 * ca4 * v4 * w6
            - 9.0 * v5 * w6
            + 21.0 * ca * v5 * w6
            + 17.0 * ca2 * v5 * w6
            - 9.0 * ca3 * v5 * w6
            + 4.0 * ca4 * v5 * w6
            + 129.0 * v6 * w6
            - 33.0 * ca * v6 * w6
            + 76.0 * ca2 * v6 * w6
            + 105.0 * ca3 * v6 * w6
            - 329.0 * ca4 * v6 * w6
            - 381.0 * v7 * w6
            - 81.0 * ca * v7 * w6
            - 216.0 * ca2 * v7 * w6
            - 111.0 * ca3 * v7 * w6
            + 835.0 * ca4 * v7 * w6
            + 465.0 * v8 * w6
            + 282.0 * ca * v8 * w6
            + 35.0 * ca2 * v8 * w6
            - 102.0 * ca3 * v8 * w6
            - 940.0 * ca4 * v8 * w6
            - 216.0 * v9 * w6
            - 324.0 * ca * v9 * w6
            + 370.0 * ca2 * v9 * w6
            + 240.0 * ca3 * v9 * w6
            + 580.0 * ca4 * v9 * w6
            + 12.0 * v10 * w6
            + 132.0 * ca * v10 * w6
            - 456.0 * ca2 * v10 * w6
            - 132.0 * ca3 * v10 * w6
            - 108.0 * ca4 * v10 * w6
            + 180.0 * ca2 * v11 * w6
            - 48.0 * ca4 * v11 * w6
            - 6.0 * ca * v5 * w7
            + 12.0 * ca2 * v5 * w7
            + 6.0 * ca3 * v5 * w7
            - 12.0 * ca4 * v5 * w7
            - 27.0 * v6 * w7
            + 9.0 * ca * v6 * w7
            - 71.0 * ca2 * v6 * w7
            - 33.0 * ca3 * v6 * w7
            + 74.0 * ca4 * v6 * w7
            + 111.0 * v7 * w7
            + 27.0 * ca * v7 * w7
            + 144.0 * ca2 * v7 * w7
            + 45.0 * ca3 * v7 * w7
            - 101.0 * ca4 * v7 * w7
            - 192.0 * v8 * w7
            - 132.0 * ca * v8 * w7
            - 41.0 * ca2 * v8 * w7
            + 60.0 * ca3 * v8 * w7
            + 109.0 * ca4 * v8 * w7
            + 126.0 * v9 * w7
            + 216.0 * ca * v9 * w7
            - 254.0 * ca2 * v9 * w7
            - 168.0 * ca3 * v9 * w7
            - 34.0 * ca4 * v9 * w7
            - 18.0 * v10 * w7
            - 114.0 * ca * v10 * w7
            + 390.0 * ca2 * v10 * w7
            + 114.0 * ca3 * v10 * w7
            - 172.0 * ca4 * v10 * w7
            - 180.0 * ca2 * v11 * w7
            + 136.0 * ca4 * v11 * w7
            - 3.0 * ca * v6 * w8
            + 6.0 * ca2 * v6 * w8
            + 3.0 * ca3 * v6 * w8
            - 6.0 * ca4 * v6 * w8
            - 9.0 * v7 * w8
            - 3.0 * ca * v7 * w8
            - 27.0 * ca2 * v7 * w8
            - 9.0 * ca3 * v7 * w8
            + 24.0 * ca4 * v7 * w8
            + 39.0 * v8 * w8
            + 30.0 * ca * v8 * w8
            + 5.0 * ca2 * v8 * w8
            - 18.0 * ca3 * v8 * w8
            - 54.0 * ca4 * v8 * w8
            - 48.0 * v9 * w8
            - 90.0 * ca * v9 * w8
            + 130.0 * ca2 * v9 * w8
            + 78.0 * ca3 * v9 * w8
            - 26.0 * ca4 * v9 * w8
            + 18.0 * v10 * w8
            + 72.0 * ca * v10 * w8
            - 246.0 * ca2 * v10 * w8
            - 72.0 * ca3 * v10 * w8
            + 194.0 * ca4 * v10 * w8
            + 132.0 * ca2 * v11 * w8
            - 132.0 * ca4 * v11 * w8
            + 6.0 * v9 * w9
            + 18.0 * ca * v9 * w9
            - 30.0 * ca2 * v9 * w9
            - 18.0 * ca3 * v9 * w9
            + 24.0 * ca4 * v9 * w9
            - 6.0 * v10 * w9
            - 30.0 * ca * v10 * w9
            + 90.0 * ca2 * v10 * w9
            + 30.0 * ca3 * v10 * w9
            - 84.0 * ca4 * v10 * w9
            - 60.0 * ca2 * v11 * w9
            + 60.0 * ca4 * v11 * w9
            + 6.0 * ca * v10 * w10
            - 12.0 * ca2 * v10 * w10
            - 6.0 * ca3 * v10 * w10
            + 12.0 * ca4 * v10 * w10
            + 12.0 * ca2 * v11 * w10
            - 12.0 * ca4 * v11 * w10))
        / (3.0
            * ca
            * (1.0 - v).powi(2)
            * v2
            * w2
            * (1.0 - v * w).powi(2)
            * (1.0 - v + v * w).powi(4));

    let part13 = -(2.0
        * cf
        * lv
        * (4.0 * ca - 16.0 * ca3 - 24.0 * ca * v + 76.0 * ca3 * v + 64.0 * ca * v2
            - 164.0 * ca3 * v2
            - 104.0 * ca * v3
            + 228.0 * ca3 * v3
            + 120.0 * ca * v4
            - 236.0 * ca3 * v4
            - 104.0 * ca * v5
            + 180.0 * ca3 * v5
            + 64.0 * ca * v6
            - 92.0 * ca3 * v6
            - 24.0 * ca * v7
            + 28.0 * ca3 * v7
            + 4.0 * ca * v8
            - 4.0 * ca3 * v8
            - 3.0 * w
            - 2.0 * ca * w
            + 18.0 * ca2 * w
            + 2.0 * ca3 * w
            - 15.0 * ca4 * w
            + 21.0 * v * w
            + 2.0 * ca * v * w
            - 126.0 * ca2 * v * w
            - 14.0 * ca3 * v * w
            + 101.0 * ca4 * v * w
            - 63.0 * v2 * w
            + 6.0 * ca * v2 * w
            + 396.0 * ca2 * v2 * w
            + 18.0 * ca3 * v2 * w
            - 313.0 * ca4 * v2 * w
            + 107.0 * v3 * w
            - 10.0 * ca * v3 * w
            - 746.0 * ca2 * v3 * w
            - 42.0 * ca3 * v3 * w
            + 595.0 * ca4 * v3 * w
            - 115.0 * v4 * w
            + 6.0 * ca * v4 * w
            + 948.0 * ca2 * v4 * w
            + 130.0 * ca3 * v4 * w
            - 773.0 * ca4 * v4 * w
            + 83.0 * v5 * w
            - 10.0 * ca * v5 * w
            - 858.0 * ca2 * v5 * w
            - 138.0 * ca3 * v5 * w
            + 715.0 * ca4 * v5 * w
            - 41.0 * v6 * w
            + 26.0 * ca * v6 * w
            + 556.0 * ca2 * v6 * w
            + 14.0 * ca3 * v6 * w
            - 471.0 * ca4 * v6 * w
            + 13.0 * v7 * w
            - 38.0 * ca * v7 * w
            - 246.0 * ca2 * v7 * w
            + 58.0 * ca3 * v7 * w
            + 213.0 * ca4 * v7 * w
            - 2.0 * v8 * w
            + 28.0 * ca * v8 * w
            + 66.0 * ca2 * v8 * w
            - 36.0 * ca3 * v8 * w
            - 60.0 * ca4 * v8 * w
            - 8.0 * ca * v9 * w
            - 8.0 * ca2 * v9 * w
            + 8.0 * ca3 * v9 * w
            + 8.0 * ca4 * v9 * w
            + w2
            + ca * w2
            - 2.0 * ca2 * w2
            - ca3 * w2
            + ca4 * w2
            - 13.0 * v * w2
            - 10.0 * ca * v * w2
            + 52.0 * ca2 * v * w2
            + 8.0 * ca3 * v * w2
            - 39.0 * ca4 * v * w2
            + 45.0 * v2 * w2
            + 32.0 * ca * v2 * w2
            - 268.0 * ca2 * v2 * w2
            + 6.0 * ca3 * v2 * w2
            + 259.0 * ca4 * v2 * w2
            - 73.0 * v3 * w2
            - 54.0 * ca * v3 * w2
            + 678.0 * ca2 * v3 * w2
            - 16.0 * ca3 * v3 * w2
            - 841.0 * ca4 * v3 * w2
            + 59.0 * v4 * w2
            - 2.0 * ca * v4 * w2
            - 1046.0 * ca2 * v4 * w2
            - 64.0 * ca3 * v4 * w2
            + 1627.0 * ca4 * v4 * w2
            - 3.0 * v5 * w2
            + 138.0 * ca * v5 * w2
            + 1040.0 * ca2 * v5 * w2
            - 20.0 * ca3 * v5 * w2
            - 1985.0 * ca4 * v5 * w2
            - 49.0 * v6 * w2
            - 188.0 * ca * v6 * w2
            - 608.0 * ca2 * v6 * w2
            + 302.0 * ca3 * v6 * w2
            + 1509.0 * ca4 * v6 * w2
            + 53.0 * v7 * w2
            + 134.0 * ca * v7 * w2
            + 102.0 * ca2 * v7 * w2
            - 312.0 * ca3 * v7 * w2
            - 639.0 * ca4 * v7 * w2
            - 24.0 * v8 * w2
            - 71.0 * ca * v8 * w2
            + 116.0 * ca2 * v8 * w2
            + 113.0 * ca3 * v8 * w2
            + 76.0 * ca4 * v8 * w2
            + 4.0 * v9 * w2
            + 16.0 * ca * v9 * w2
            - 80.0 * ca2 * v9 * w2
            - 12.0 * ca3 * v9 * w2
            + 48.0 * ca4 * v9 * w2
            + 4.0 * ca * v10 * w2
            + 16.0 * ca2 * v10 * w2
            - 4.0 * ca3 * v10 * w2
            - 16.0 * ca4 * v10 * w2
            + 2.0 * v * w3
            + 2.0 * ca * v * w3
            - 4.0 * ca2 * v * w3
            - 2.0 * ca3 * v * w3
            + 2.0 * ca4 * v * w3
            - 7.0 * v2 * w3
            - 6.0 * ca * v2 * w3
            + 6.0 * ca2 * v2 * w3
            + 2.0 * ca3 * v2 * w3
            + ca4 * v2 * w3
            + v3 * w3
            + 10.0 * ca * v3 * w3
            + 62.0 * ca2 * v3 * w3
            - 4.0 * ca3 * v3 * w3
            - 71.0 * ca4 * v3 * w3
            + 34.0 * v4 * w3
            + 72.0 * ca * v4 * w3
            - 304.0 * ca2 * v4 * w3
            + 56.0 * ca3 * v4 * w3
            + 386.0 * ca4 * v4 * w3
            - 110.0 * v5 * w3
            - 224.0 * ca * v5 * w3
            + 758.0 * ca2 * v5 * w3
            + 122.0 * ca3 * v5 * w3
            - 1148.0 * ca4 * v5 * w3
            + 175.0 * v6 * w3
            + 150.0 * ca * v6 * w3
            - 1276.0 * ca2 * v6 * w3
            - 514.0 * ca3 * v6 * w3
            + 2089.0 * ca4 * v6 * w3
            - 127.0 * v7 * w3
            + 18.0 * ca * v7 * w3
            + 1434.0 * ca2 * v7 * w3
            + 400.0 * ca3 * v7 * w3
            - 2359.0 * ca4 * v7 * w3
            + 24.0 * v8 * w3
            - 4.0 * ca * v8 * w3
            - 984.0 * ca2 * v8 * w3
            - 44.0 * ca3 * v8 * w3
            + 1616.0 * ca4 * v8 * w3
            + 10.0 * v9 * w3
            + 2.0 * ca * v9 * w3
            + 350.0 * ca2 * v9 * w3
            - 36.0 * ca3 * v9 * w3
            - 608.0 * ca4 * v9 * w3
            - 2.0 * v10 * w3
            - 20.0 * ca * v10 * w3
            - 34.0 * ca2 * v10 * w3
            + 20.0 * ca3 * v10 * w3
            + 84.0 * ca4 * v10 * w3
            - 8.0 * ca2 * v11 * w3
            + 8.0 * ca4 * v11 * w3
            - v2 * w4
            - ca * v2 * w4
            + 2.0 * ca2 * v2 * w4
            + ca3 * v2 * w4
            - ca4 * v2 * w4
            + 23.0 * v3 * w4
            + 18.0 * ca * v3 * w4
            - 96.0 * ca2 * v3 * w4
            - 16.0 * ca3 * v3 * w4
            + 73.0 * ca4 * v3 * w4
            - 78.0 * v4 * w4
            - 93.0 * ca * v4 * w4
            + 448.0 * ca2 * v4 * w4
            + 21.0 * ca3 * v4 * w4
            - 482.0 * ca4 * v4 * w4
            + 158.0 * v5 * w4
            + 166.0 * ca * v5 * w4
            - 1076.0 * ca2 * v5 * w4
            - 148.0 * ca3 * v5 * w4
            + 1550.0 * ca4 * v5 * w4
            - 195.0 * v6 * w4
            + 81.0 * ca * v6 * w4
            + 1720.0 * ca2 * v6 * w4
            + 391.0 * ca3 * v6 * w4
            - 2833.0 * ca4 * v6 * w4
            + 63.0 * v7 * w4
            - 318.0 * ca * v7 * w4
            - 1816.0 * ca2 * v7 * w4
            - 140.0 * ca3 * v7 * w4
            + 3013.0 * ca4 * v7 * w4
            + 88.0 * v8 * w4
            + 105.0 * ca * v8 * w4
            + 1084.0 * ca2 * v8 * w4
            - 193.0 * ca3 * v8 * w4
            - 1756.0 * ca4 * v8 * w4
            - 60.0 * v9 * w4
            - 2.0 * ca * v9 * w4
            - 204.0 * ca2 * v9 * w4
            + 112.0 * ca3 * v9 * w4
            + 372.0 * ca4 * v9 * w4
            + 2.0 * v10 * w4
            + 44.0 * ca * v10 * w4
            - 110.0 * ca2 * v10 * w4
            - 44.0 * ca3 * v10 * w4
            + 136.0 * ca4 * v10 * w4
            + 48.0 * ca2 * v11 * w4
            - 72.0 * ca4 * v11 * w4
            - 4.0 * v3 * w5
            - 4.0 * ca * v3 * w5
            + 8.0 * ca2 * v3 * w5
            + 4.0 * ca3 * v3 * w5
            - 4.0 * ca4 * v3 * w5
            + 23.0 * v4 * w5
            + 18.0 * ca * v4 * w5
            - 66.0 * ca2 * v4 * w5
            - 10.0 * ca3 * v4 * w5
            + 43.0 * ca4 * v4 * w5
            - 59.0 * v5 * w5
            - 18.0 * ca * v5 * w5
            + 188.0 * ca2 * v5 * w5
            + 38.0 * ca3 * v5 * w5
            - 189.0 * ca4 * v5 * w5
            + 59.0 * v6 * w5
            - 198.0 * ca * v6 * w5
            - 338.0 * ca2 * v6 * w5
            - 98.0 * ca3 * v6 * w5
            + 275.0 * ca4 * v6 * w5
            + 95.0 * v7 * w5
            + 350.0 * ca * v7 * w5
            + 306.0 * ca2 * v7 * w5
            - 134.0 * ca3 * v7 * w5
            + 135.0 * ca4 * v7 * w5
            - 214.0 * v8 * w5
            - 20.0 * ca * v8 * w5
            + 122.0 * ca2 * v8 * w5
            + 308.0 * ca3 * v8 * w5
            - 856.0 * ca4 * v8 * w5
            + 86.0 * v9 * w5
            - 68.0 * ca * v9 * w5
            - 522.0 * ca2 * v9 * w5
            - 112.0 * ca3 * v9 * w5
            + 1100.0 * ca4 * v9 * w5
            + 14.0 * v10 * w5
            - 60.0 * ca * v10 * w5
            + 414.0 * ca2 * v10 * w5
            + 60.0 * ca3 * v10 * w5
            - 688.0 * ca4 * v10 * w5
            - 112.0 * ca2 * v11 * w5
            + 184.0 * ca4 * v11 * w5
            - v4 * w6
            - ca * v4 * w6
            + 2.0 * ca2 * v4 * w6
            + ca3 * v4 * w6
            - ca4 * v4 * w6
            - 7.0 * v5 * w6
            - 6.0 * ca * v5 * w6
            + 36.0 * ca2 * v5 * w6
            + 8.0 * ca3 * v5 * w6
            - 29.0 * ca4 * v5 * w6
            + 37.0 * v6 * w6
            + 86.0 * ca * v6 * w6
            - 156.0 * ca2 * v6 * w6
            - 16.0 * ca3 * v6 * w6
            + 251.0 * ca4 * v6 * w6
            - 141.0 * v7 * w6
            - 130.0 * ca * v7 * w6
            + 350.0 * ca2 * v7 * w6
            + 128.0 * ca3 * v7 * w6
            - 797.0 * ca4 * v7 * w6
            + 192.0 * v8 * w6
            - 125.0 * ca * v8 * w6
            - 632.0 * ca2 * v8 * w6
            - 181.0 * ca3 * v8 * w6
            + 1316.0 * ca4 * v8 * w6
            - 40.0 * v9 * w6
            + 120.0 * ca * v9 * w6
            + 772.0 * ca2 * v9 * w6
            + 36.0 * ca3 * v9 * w6
            - 1268.0 * ca4 * v9 * w6
            - 40.0 * v10 * w6
            + 64.0 * ca * v10 * w6
            - 508.0 * ca2 * v10 * w6
            - 64.0 * ca3 * v10 * w6
            + 728.0 * ca4 * v10 * w6
            + 136.0 * ca2 * v11 * w6
            - 200.0 * ca4 * v11 * w6
            + 2.0 * v5 * w7
            + 2.0 * ca * v5 * w7
            - 4.0 * ca2 * v5 * w7
            - 2.0 * ca3 * v5 * w7
            + 2.0 * ca4 * v5 * w7
            - 13.0 * v6 * w7
            - 10.0 * ca * v6 * w7
            + 42.0 * ca2 * v6 * w7
            + 6.0 * ca3 * v6 * w7
            - 29.0 * ca4 * v6 * w7
            + 51.0 * v7 * w7
            + 10.0 * ca * v7 * w7
            - 122.0 * ca2 * v7 * w7
            - 28.0 * ca3 * v7 * w7
            + 135.0 * ca4 * v7 * w7
            - 68.0 * v8 * w7
            + 108.0 * ca * v8 * w7
            + 260.0 * ca2 * v8 * w7
            + 36.0 * ca3 * v8 * w7
            - 292.0 * ca4 * v8 * w7
            - 18.0 * v9 * w7
            - 74.0 * ca * v9 * w7
            - 394.0 * ca2 * v9 * w7
            + 12.0 * ca3 * v9 * w7
            + 348.0 * ca4 * v9 * w7
            + 46.0 * v10 * w7
            - 60.0 * ca * v10 * w7
            + 322.0 * ca2 * v10 * w7
            + 60.0 * ca3 * v10 * w7
            - 268.0 * ca4 * v10 * w7
            - 104.0 * ca2 * v11 * w7
            + 104.0 * ca4 * v11 * w7
            + v6 * w8
            + ca * v6 * w8
            - 2.0 * ca2 * v6 * w8
            - ca3 * v6 * w8
            + ca4 * v6 * w8
            - 3.0 * v7 * w8
            - 2.0 * ca * v7 * w8
            + 8.0 * ca2 * v7 * w8
            - 5.0 * ca4 * v7 * w8
            + 4.0 * v8 * w8
            - 25.0 * ca * v8 * w8
            - 32.0 * ca2 * v8 * w8
            + ca3 * v8 * w8
            + 4.0 * ca4 * v8 * w8
            + 24.0 * v9 * w8
            + 10.0 * ca * v9 * w8
            + 104.0 * ca2 * v9 * w8
            - 8.0 * ca3 * v9 * w8
            - 26.0 * v10 * w8
            + 44.0 * ca * v10 * w8
            - 142.0 * ca2 * v10 * w8
            - 44.0 * ca3 * v10 * w8
            + 40.0 * ca4 * v10 * w8
            + 64.0 * ca2 * v11 * w8
            - 40.0 * ca4 * v11 * w8
            - 6.0 * v9 * w9
            + 4.0 * ca * v9 * w9
            - 18.0 * ca2 * v9 * w9
            + 6.0 * v10 * w9
            - 20.0 * ca * v10 * w9
            + 50.0 * ca2 * v10 * w9
            + 20.0 * ca3 * v10 * w9
            - 24.0 * ca4 * v10 * w9
            - 32.0 * ca2 * v11 * w9
            + 24.0 * ca4 * v11 * w9
            + 4.0 * ca * v10 * w10
            - 8.0 * ca2 * v10 * w10
            - 4.0 * ca3 * v10 * w10
            + 8.0 * ca4 * v10 * w10
            + 8.0 * ca2 * v11 * w10
            - 8.0 * ca4 * v11 * w10))
        / (ca * (1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(4));

    let part14 = -(2.0
        * cf
        * l1w
        * (4.0 * ca - 12.0 * ca3 - 24.0 * ca * v + 56.0 * ca3 * v + 64.0 * ca * v2
            - 120.0 * ca3 * v2
            - 104.0 * ca * v3
            + 168.0 * ca3 * v3
            + 120.0 * ca * v4
            - 176.0 * ca3 * v4
            - 104.0 * ca * v5
            + 136.0 * ca3 * v5
            + 64.0 * ca * v6
            - 72.0 * ca3 * v6
            - 24.0 * ca * v7
            + 24.0 * ca3 * v7
            + 4.0 * ca * v8
            - 4.0 * ca3 * v8
            - 5.0 * w
            - 2.0 * ca * w
            + 16.0 * ca2 * w
            + 2.0 * ca3 * w
            - 11.0 * ca4 * w
            + 31.0 * v * w
            + 6.0 * ca * v * w
            - 114.0 * ca2 * v * w
            - 14.0 * ca3 * v * w
            + 75.0 * ca4 * v * w
            - 85.0 * v2 * w
            - 10.0 * ca * v2 * w
            + 364.0 * ca2 * v2 * w
            + 22.0 * ca3 * v2 * w
            - 237.0 * ca4 * v2 * w
            + 133.0 * v3 * w
            + 10.0 * ca * v3 * w
            - 698.0 * ca2 * v3 * w
            - 46.0 * ca3 * v3 * w
            + 463.0 * ca4 * v3 * w
            - 125.0 * v4 * w
            + 6.0 * ca * v4 * w
            + 908.0 * ca2 * v4 * w
            + 114.0 * ca3 * v4 * w
            - 623.0 * ca4 * v4 * w
            + 65.0 * v5 * w
            - 30.0 * ca * v5 * w
            - 846.0 * ca2 * v5 * w
            - 114.0 * ca3 * v5 * w
            + 601.0 * ca4 * v5 * w
            - 11.0 * v6 * w
            + 42.0 * ca * v6 * w
            + 564.0 * ca2 * v6 * w
            + 18.0 * ca3 * v6 * w
            - 415.0 * ca4 * v6 * w
            - 5.0 * v7 * w
            - 42.0 * ca * v7 * w
            - 254.0 * ca2 * v7 * w
            + 38.0 * ca3 * v7 * w
            + 197.0 * ca4 * v7 * w
            + 2.0 * v8 * w
            + 28.0 * ca * v8 * w
            + 68.0 * ca2 * v8 * w
            - 28.0 * ca3 * v8 * w
            - 58.0 * ca4 * v8 * w
            - 8.0 * ca * v9 * w
            - 8.0 * ca2 * v9 * w
            + 8.0 * ca3 * v9 * w
            + 8.0 * ca4 * v9 * w
            + w2
            + ca * w2
            - 2.0 * ca2 * w2
            - ca3 * w2
            + ca4 * w2
            - 15.0 * v * w2
            - 10.0 * ca * v * w2
            + 50.0 * ca2 * v * w2
            + 8.0 * ca3 * v * w2
            - 31.0 * ca4 * v * w2
            + 49.0 * v2 * w2
            + 28.0 * ca * v2 * w2
            - 270.0 * ca2 * v2 * w2
            - 2.0 * ca3 * v2 * w2
            + 205.0 * ca4 * v2 * w2
            - 67.0 * v3 * w2
            - 26.0 * ca * v3 * w2
            + 732.0 * ca2 * v3 * w2
            - 675.0 * ca4 * v3 * w2
            + 23.0 * v4 * w2
            - 66.0 * ca * v4 * w2
            - 1236.0 * ca2 * v4 * w2
            - 68.0 * ca3 * v4 * w2
            + 1335.0 * ca4 * v4 * w2
            + 63.0 * v5 * w2
            + 194.0 * ca * v5 * w2
            + 1374.0 * ca2 * v5 * w2
            + 8.0 * ca3 * v5 * w2
            - 1681.0 * ca4 * v5 * w2
            - 101.0 * v6 * w2
            - 192.0 * ca * v6 * w2
            - 942.0 * ca2 * v6 * w2
            + 206.0 * ca3 * v6 * w2
            + 1335.0 * ca4 * v6 * w2
            + 55.0 * v7 * w2
            + 114.0 * ca * v7 * w2
            + 288.0 * ca2 * v7 * w2
            - 224.0 * ca3 * v7 * w2
            - 601.0 * ca4 * v7 * w2
            - 4.0 * v8 * w2
            - 63.0 * ca * v8 * w2
            + 66.0 * ca2 * v8 * w2
            + 93.0 * ca3 * v8 * w2
            + 84.0 * ca4 * v8 * w2
            - 4.0 * v9 * w2
            + 16.0 * ca * v9 * w2
            - 76.0 * ca2 * v9 * w2
            - 16.0 * ca3 * v9 * w2
            + 44.0 * ca4 * v9 * w2
            + 4.0 * ca * v10 * w2
            + 16.0 * ca2 * v10 * w2
            - 4.0 * ca3 * v10 * w2
            - 16.0 * ca4 * v10 * w2
            + 2.0 * v * w3
            + 2.0 * ca * v * w3
            - 4.0 * ca2 * v * w3
            - 2.0 * ca3 * v * w3
            + 2.0 * ca4 * v * w3
            + 3.0 * v2 * w3
            - 6.0 * ca * v2 * w3
            + 12.0 * ca2 * v2 * w3
            + 2.0 * ca3 * v2 * w3
            - 3.0 * ca4 * v2 * w3
            - 57.0 * v3 * w3
            - 2.0 * ca * v3 * w3
            + 20.0 * ca2 * v3 * w3
            - 43.0 * ca4 * v3 * w3
            + 174.0 * v4 * w3
            + 96.0 * ca * v4 * w3
            - 188.0 * ca2 * v4 * w3
            + 44.0 * ca3 * v4 * w3
            + 276.0 * ca4 * v4 * w3
            - 286.0 * v5 * w3
            - 204.0 * ca * v5 * w3
            + 612.0 * ca2 * v5 * w3
            + 70.0 * ca3 * v5 * w3
            - 868.0 * ca4 * v5 * w3
            + 269.0 * v6 * w3
            + 70.0 * ca * v6 * w3
            - 1260.0 * ca2 * v6 * w3
            - 362.0 * ca3 * v6 * w3
            + 1651.0 * ca4 * v6 * w3
            - 97.0 * v7 * w3
            + 78.0 * ca * v7 * w3
            + 1620.0 * ca2 * v7 * w3
            + 300.0 * ca3 * v7 * w3
            - 1963.0 * ca4 * v7 * w3
            - 32.0 * v8 * w3
            - 12.0 * ca * v8 * w3
            - 1204.0 * ca2 * v8 * w3
            - 56.0 * ca3 * v8 * w3
            + 1430.0 * ca4 * v8 * w3
            + 22.0 * v9 * w3
            - 2.0 * ca * v9 * w3
            + 448.0 * ca2 * v9 * w3
            - 16.0 * ca3 * v9 * w3
            - 576.0 * ca4 * v9 * w3
            + 2.0 * v10 * w3
            - 20.0 * ca * v10 * w3
            - 48.0 * ca2 * v10 * w3
            + 20.0 * ca3 * v10 * w3
            + 86.0 * ca4 * v10 * w3
            - 8.0 * ca2 * v11 * w3
            + 8.0 * ca4 * v11 * w3
            - v2 * w4
            - ca * v2 * w4
            + 2.0 * ca2 * v2 * w4
            + ca3 * v2 * w4
            - ca4 * v2 * w4
            + 41.0 * v3 * w4
            + 18.0 * ca * v3 * w4
            - 86.0 * ca2 * v3 * w4
            - 16.0 * ca3 * v3 * w4
            + 57.0 * ca4 * v3 * w4
            - 146.0 * v4 * w4
            - 81.0 * ca * v4 * w4
            + 430.0 * ca2 * v4 * w4
            + 21.0 * ca3 * v4 * w4
            - 368.0 * ca4 * v4 * w4
            + 242.0 * v5 * w4
            + 106.0 * ca * v5 * w4
            - 1144.0 * ca2 * v5 * w4
            - 100.0 * ca3 * v5 * w4
            + 1204.0 * ca4 * v5 * w4
            - 193.0 * v6 * w4
            + 161.0 * ca * v6 * w4
            + 2028.0 * ca2 * v6 * w4
            + 291.0 * ca3 * v6 * w4
            - 2285.0 * ca4 * v6 * w4
            - 31.0 * v7 * w4
            - 334.0 * ca * v7 * w4
            - 2326.0 * ca2 * v7 * w4
            - 144.0 * ca3 * v7 * w4
            + 2565.0 * ca4 * v7 * w4
            + 152.0 * v8 * w4
            + 77.0 * ca * v8 * w4
            + 1482.0 * ca2 * v8 * w4
            - 101.0 * ca3 * v8 * w4
            - 1618.0 * ca4 * v8 * w4
            - 52.0 * v9 * w4
            + 10.0 * ca * v9 * w4
            - 316.0 * ca2 * v9 * w4
            + 76.0 * ca3 * v9 * w4
            + 406.0 * ca4 * v9 * w4
            - 12.0 * v10 * w4
            + 44.0 * ca * v10 * w4
            - 126.0 * ca2 * v10 * w4
            - 44.0 * ca3 * v10 * w4
            + 112.0 * ca4 * v10 * w4
            + 56.0 * ca2 * v11 * w4
            - 72.0 * ca4 * v11 * w4
            - 4.0 * v3 * w5
            - 4.0 * ca * v3 * w5
            + 8.0 * ca2 * v3 * w5
            + 4.0 * ca3 * v3 * w5
            - 4.0 * ca4 * v3 * w5
            + 25.0 * v4 * w5
            + 18.0 * ca * v4 * w5
            - 60.0 * ca2 * v4 * w5
            - 10.0 * ca3 * v4 * w5
            + 39.0 * ca4 * v4 * w5
            - 33.0 * v5 * w5
            - 6.0 * ca * v5 * w5
            + 186.0 * ca2 * v5 * w5
            + 26.0 * ca3 * v5 * w5
            - 165.0 * ca4 * v5 * w5
            - 47.0 * v6 * w5
            - 198.0 * ca * v6 * w5
            - 380.0 * ca2 * v6 * w5
            - 86.0 * ca3 * v6 * w5
            + 265.0 * ca4 * v6 * w5
            + 229.0 * v7 * w5
            + 298.0 * ca * v7 * w5
            + 374.0 * ca2 * v7 * w5
            - 42.0 * ca3 * v7 * w5
            + 15.0 * ca4 * v7 * w5
            - 262.0 * v8 * w5
            + 28.0 * ca * v8 * w5
            + 160.0 * ca2 * v8 * w5
            + 188.0 * ca3 * v8 * w5
            - 606.0 * ca4 * v8 * w5
            + 62.0 * v9 * w5
            - 76.0 * ca * v9 * w5
            - 688.0 * ca2 * v9 * w5
            - 92.0 * ca3 * v9 * w5
            + 902.0 * ca4 * v9 * w5
            + 30.0 * v10 * w5
            - 60.0 * ca * v10 * w5
            + 536.0 * ca2 * v10 * w5
            + 60.0 * ca3 * v10 * w5
            - 630.0 * ca4 * v10 * w5
            - 136.0 * ca2 * v11 * w5
            + 184.0 * ca4 * v11 * w5
            - v4 * w6
            - ca * v4 * w6
            + 2.0 * ca2 * v4 * w6
            + ca3 * v4 * w6
            - ca4 * v4 * w6
            - 21.0 * v5 * w6
            - 6.0 * ca * v5 * w6
            + 38.0 * ca2 * v5 * w6
            + 8.0 * ca3 * v5 * w6
            - 21.0 * ca4 * v5 * w6
            + 97.0 * v6 * w6
            + 74.0 * ca * v6 * w6
            - 178.0 * ca2 * v6 * w6
            - 8.0 * ca3 * v6 * w6
            + 185.0 * ca4 * v6 * w6
            - 215.0 * v7 * w6
            - 94.0 * ca * v7 * w6
            + 436.0 * ca2 * v7 * w6
            + 64.0 * ca3 * v7 * w6
            - 603.0 * ca4 * v7 * w6
            + 212.0 * v8 * w6
            - 141.0 * ca * v8 * w6
            - 814.0 * ca2 * v8 * w6
            - 121.0 * ca3 * v8 * w6
            + 1052.0 * ca4 * v8 * w6
            - 32.0 * v9 * w6
            + 112.0 * ca * v9 * w6
            + 984.0 * ca2 * v9 * w6
            + 56.0 * ca3 * v9 * w6
            - 1092.0 * ca4 * v9 * w6
            - 40.0 * v10 * w6
            + 64.0 * ca * v10 * w6
            - 620.0 * ca2 * v10 * w6
            - 64.0 * ca3 * v10 * w6
            + 680.0 * ca4 * v10 * w6
            + 152.0 * ca2 * v11 * w6
            - 200.0 * ca4 * v11 * w6
            + 2.0 * v5 * w7
            + 2.0 * ca * v5 * w7
            - 4.0 * ca2 * v5 * w7
            - 2.0 * ca3 * v5 * w7
            + 2.0 * ca4 * v5 * w7
            - 23.0 * v6 * w7
            - 10.0 * ca * v6 * w7
            + 36.0 * ca2 * v6 * w7
            + 6.0 * ca3 * v6 * w7
            - 25.0 * ca4 * v6 * w7
            + 69.0 * v7 * w7
            + 6.0 * ca * v7 * w7
            - 104.0 * ca2 * v7 * w7
            - 16.0 * ca3 * v7 * w7
            + 107.0 * ca4 * v7 * w7
            - 76.0 * v8 * w7
            + 100.0 * ca * v8 * w7
            + 236.0 * ca2 * v8 * w7
            + 32.0 * ca3 * v8 * w7
            - 246.0 * ca4 * v8 * w7
            - 2.0 * v9 * w7
            - 62.0 * ca * v9 * w7
            - 372.0 * ca2 * v9 * w7
            - 24.0 * ca3 * v9 * w7
            + 320.0 * ca4 * v9 * w7
            + 30.0 * v10 * w7
            - 60.0 * ca * v10 * w7
            + 296.0 * ca2 * v10 * w7
            + 60.0 * ca3 * v10 * w7
            - 262.0 * ca4 * v10 * w7
            - 88.0 * ca2 * v11 * w7
            + 104.0 * ca4 * v11 * w7
            + v6 * w8
            + ca * v6 * w8
            - 2.0 * ca2 * v6 * w8
            - ca3 * v6 * w8
            + ca4 * v6 * w8
            - 5.0 * v7 * w8
            - 2.0 * ca * v7 * w8
            - 2.0 * ca2 * v7 * w8
            - 5.0 * ca4 * v7 * w8
            + 8.0 * v8 * w8
            - 21.0 * ca * v8 * w8
            + 10.0 * ca2 * v8 * w8
            - 3.0 * ca3 * v8 * w8
            + 10.0 * ca4 * v8 * w8
            + 8.0 * v9 * w8
            + 6.0 * ca * v9 * w8
            + 32.0 * ca2 * v9 * w8
            + 12.0 * ca3 * v9 * w8
            - 14.0 * ca4 * v9 * w8
            - 12.0 * v10 * w8
            + 44.0 * ca * v10 * w8
            - 78.0 * ca2 * v10 * w8
            - 44.0 * ca3 * v10 * w8
            + 48.0 * ca4 * v10 * w8
            + 40.0 * ca2 * v11 * w8
            - 40.0 * ca4 * v11 * w8
            - 4.0 * ca2 * v8 * w9
            - 2.0 * v9 * w9
            + 4.0 * ca * v9 * w9
            - 4.0 * ca2 * v9 * w9
            - 4.0 * ca3 * v9 * w9
            + 2.0 * ca4 * v9 * w9
            + 2.0 * v10 * w9
            - 20.0 * ca * v10 * w9
            + 32.0 * ca2 * v10 * w9
            + 20.0 * ca3 * v10 * w9
            - 26.0 * ca4 * v10 * w9
            - 24.0 * ca2 * v11 * w9
            + 24.0 * ca4 * v11 * w9
            + 4.0 * ca * v10 * w10
            - 8.0 * ca2 * v10 * w10
            - 4.0 * ca3 * v10 * w10
            + 8.0 * ca4 * v10 * w10
            + 8.0 * ca2 * v11 * w10
            - 8.0 * ca4 * v11 * w10))
        / (ca * (1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w).powi(2) * (1.0 - v + v * w).powi(4));

    part1
        + part2
        + part3
        + part4
        + part5
        + part6
        + part7
        + part8
        + part9
        + part10
        + part11
        + part12
        + part13
        + part14
}

/// `STRUV13(W,V,X3,S)`.
#[must_use]
pub fn struv13(w: f64, v: f64, _x3: f64, _s: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let cf = ctx.cf;
    let ca = ctx.ca;
    let nf = ctx.nf;
    let (v2, v3, v4, v5, v6, v7, v8, v9, v10, v11) = (
        pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8, pre.v9, pre.v10, pre.v11,
    );
    let (w2, w3, w4, w5, w6, w7, w8, w9, w10) = (
        pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8, pre.w9, pre.w10,
    );
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let ca4 = pre.ca4;
    let cacf = ca * cf;
    let ca3cf = ca3 * cf;
    let cf2 = cf.powi(2);
    let cacf2 = ca * cf2;
    let (l1v, lw, lvw, l1vw, lms, l1w, lv) =
        (pre.l1v, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.l1w, pre.lv);

    let part1 = (8.0
        * ca
        * cf2
        * l1w
        * nf
        * (1.0 + v2 * w2)
        * (1.0 - 2.0 * v + 2.0 * v2 - 2.0 * v2 * w + v2 * w2))
        / ((1.0 - v) * (1.0 - v * w).powi(4));

    let part2 = -(8.0
        * ca
        * cf2
        * lms
        * nf
        * (1.0 + v2 * w2)
        * (1.0 - 2.0 * v + 2.0 * v2 - 2.0 * v2 * w + v2 * w2))
        / ((1.0 - v) * (1.0 - v * w).powi(4));

    let part3 = (8.0
        * ca
        * cf2
        * lv
        * nf
        * (1.0 + v2 * w2)
        * (1.0 - 2.0 * v + 2.0 * v2 - 2.0 * v2 * w + v2 * w2))
        / ((1.0 - v) * (1.0 - v * w).powi(4));

    let part4 = -(4.0
        * cf
        * lvw
        * (2.0 * ca - 2.0 * ca3 - 6.0 * ca * v + 6.0 * ca3 * v + 8.0 * ca * v2
            - 8.0 * ca3 * v2
            - 4.0 * ca * v3
            + 4.0 * ca3 * v3
            - 3.0 * w
            - 2.0 * ca * w
            + 2.0 * ca2 * w
            + 2.0 * ca3 * w
            - 3.0 * ca4 * w
            + 11.0 * v * w
            + 2.0 * ca * v * w
            + 2.0 * ca2 * v * w
            - 8.0 * ca3 * v * w
            + 4.0 * ca4 * v * w
            - 16.0 * v2 * w
            + 6.0 * ca * v2 * w
            - 6.0 * ca2 * v2 * w
            + 18.0 * ca3 * v2 * w
            - 6.0 * ca4 * v2 * w
            + 12.0 * v3 * w
            - 14.0 * ca * v3 * w
            + 10.0 * ca2 * v3 * w
            - 16.0 * ca3 * v3 * w
            + 3.0 * ca4 * v3 * w
            - 4.0 * v4 * w
            + 8.0 * ca * v4 * w
            - 4.0 * ca2 * v4 * w
            + 4.0 * ca3 * v4 * w
            - 2.0 * ca4 * v4 * w
            - v * w2
            + 4.0 * ca * v * w2
            - 5.0 * ca2 * v * w2
            + 2.0 * ca3 * v * w2
            - 6.0 * ca4 * v * w2
            + 3.0 * v2 * w2
            - 14.0 * ca * v2 * w2
            + ca2 * v2 * w2
            - 16.0 * ca3 * v2 * w2
            + 14.0 * ca4 * v2 * w2
            - 6.0 * v3 * w2
            + 16.0 * ca * v3 * w2
            - 10.0 * ca2 * v3 * w2
            + 26.0 * ca3 * v3 * w2
            - 13.0 * ca4 * v3 * w2
            + 4.0 * v4 * w2
            - 2.0 * ca * v4 * w2
            + 6.0 * ca2 * v4 * w2
            - 16.0 * ca3 * v4 * w2
            + 5.0 * ca4 * v4 * w2
            - 4.0 * ca * v5 * w2
            + 4.0 * ca3 * v5 * w2
            - v2 * w3
            + 7.0 * ca2 * v2 * w3
            + 6.0 * ca3 * v2 * w3
            - 14.0 * ca4 * v2 * w3
            + 5.0 * v3 * w3
            + 4.0 * ca * v3 * w3
            + ca2 * v3 * w3
            - 16.0 * ca3 * v3 * w3
            + 15.0 * ca4 * v3 * w3
            - 4.0 * v4 * w3
            - 12.0 * ca * v4 * w3
            + 18.0 * ca3 * v4 * w3
            - 9.0 * ca4 * v4 * w3
            + 8.0 * ca * v5 * w3
            - 8.0 * ca3 * v5 * w3
            - v3 * w4
            - 2.0 * ca * v3 * w4
            - 4.0 * ca2 * v3 * w4
            + 2.0 * ca3 * v3 * w4
            - 7.0 * ca4 * v3 * w4
            + v4 * w4
            + 8.0 * ca * v4 * w4
            - 4.0 * ca2 * v4 * w4
            - 8.0 * ca3 * v4 * w4
            + 7.0 * ca4 * v4 * w4
            - 6.0 * ca * v5 * w4
            + 6.0 * ca3 * v5 * w4
            - 2.0 * ca * v4 * w5
            + 4.0 * ca2 * v4 * w5
            + 2.0 * ca3 * v4 * w5
            - 4.0 * ca4 * v4 * w5
            + 2.0 * ca * v5 * w5
            - 2.0 * ca3 * v5 * w5))
        / (ca * (1.0 - v).powi(2) * pre.v2 * pre.w2);

    let part5 = -(4.0
        * cf
        * l1v
        * (2.0 * ca3 - 10.0 * ca3 * v + 22.0 * ca3 * v2 - 26.0 * ca3 * v3 + 16.0 * ca3 * v4
            - 4.0 * ca3 * v5
            - w
            - 3.0 * ca2 * w
            + 2.0 * ca4 * w
            + 5.0 * v * w
            - 6.0 * ca * v * w
            + 10.0 * ca2 * v * w
            + 4.0 * ca3 * v * w
            - 7.0 * ca4 * v * w
            - 11.0 * v2 * w
            + 30.0 * ca * v2 * w
            - 15.0 * ca2 * v2 * w
            - 20.0 * ca3 * v2 * w
            + 11.0 * ca4 * v2 * w
            + 11.0 * v3 * w
            - 70.0 * ca * v3 * w
            + 12.0 * ca2 * v3 * w
            + 38.0 * ca3 * v3 * w
            - 10.0 * ca4 * v3 * w
            - 4.0 * v4 * w
            + 90.0 * ca * v4 * w
            - 4.0 * ca2 * v4 * w
            - 32.0 * ca3 * v4 * w
            + 5.0 * ca4 * v4 * w
            - 60.0 * ca * v5 * w
            + 10.0 * ca3 * v5 * w
            - ca4 * v5 * w
            + 16.0 * ca * v6 * w
            - v * w2
            - ca2 * v * w2
            + 4.0 * v2 * w2
            - 10.0 * ca * v2 * w2
            - ca2 * v2 * w2
            + 6.0 * ca3 * v2 * w2
            - 9.0 * ca4 * v2 * w2
            - 5.0 * v3 * w2
            + 56.0 * ca * v3 * w2
            + 2.0 * ca2 * v3 * w2
            - 24.0 * ca3 * v3 * w2
            + 16.0 * ca4 * v3 * w2
            - 118.0 * ca * v4 * w2
            - 4.0 * ca2 * v4 * w2
            + 36.0 * ca3 * v4 * w2
            - 11.0 * ca4 * v4 * w2
            + 2.0 * v5 * w2
            + 108.0 * ca * v5 * w2
            + 4.0 * ca2 * v5 * w2
            - 28.0 * ca3 * v5 * w2
            + 3.0 * ca4 * v5 * w2
            - 36.0 * ca * v6 * w2
            + 14.0 * ca3 * v6 * w2
            + ca4 * v6 * w2
            - 4.0 * ca3 * v7 * w2
            + 3.0 * v2 * w3
            + 5.0 * ca2 * v2 * w3
            - 2.0 * ca4 * v2 * w3
            - 7.0 * v3 * w3
            - 14.0 * ca * v3 * w3
            - 19.0 * ca2 * v3 * w3
            + 4.0 * ca3 * v3 * w3
            - 8.0 * ca4 * v3 * w3
            + 6.0 * v4 * w3
            + 58.0 * ca * v4 * w3
            + 22.0 * ca2 * v4 * w3
            - 14.0 * ca3 * v4 * w3
            + 7.0 * ca4 * v4 * w3
            - 74.0 * ca * v5 * w3
            - 12.0 * ca2 * v5 * w3
            + 24.0 * ca3 * v5 * w3
            - 3.0 * ca4 * v5 * w3
            - 2.0 * v6 * w3
            + 30.0 * ca * v6 * w3
            - 22.0 * ca3 * v6 * w3
            - 2.0 * ca4 * v6 * w3
            + 8.0 * ca3 * v7 * w3
            - v3 * w4
            - ca2 * v3 * w4
            + 4.0 * v4 * w4
            - 10.0 * ca * v4 * w4
            + 3.0 * ca2 * v4 * w4
            + 2.0 * ca3 * v4 * w4
            - 3.0 * ca4 * v4 * w4
            - 8.0 * v5 * w4
            + 20.0 * ca * v5 * w4
            - 6.0 * ca2 * v5 * w4
            - 10.0 * ca3 * v5 * w4
            + 3.0 * ca4 * v5 * w4
            + 5.0 * v6 * w4
            - 10.0 * ca * v6 * w4
            + 4.0 * ca2 * v6 * w4
            + 14.0 * ca3 * v6 * w4
            - 6.0 * ca3 * v7 * w4
            + 2.0 * v5 * w5
            + 7.0 * ca2 * v5 * w5
            + 2.0 * ca3 * v5 * w5
            - ca4 * v5 * w5
            - 2.0 * v6 * w5
            - 3.0 * ca2 * v6 * w5
            - 4.0 * ca3 * v6 * w5
            + ca4 * v6 * w5
            + 2.0 * ca3 * v7 * w5))
        / (ca * (1.0 - v).powi(2) * pre.v2 * pre.w2 * (1.0 - v * w) * (1.0 - v + v * w));

    let part6 = -(4.0
        * cf
        * nf
        * (-2.0 * cacf - 4.0 * v + 4.0 * ca2 * v + 7.0 * v2 - 7.0 * ca2 * v2 - 6.0 * v3
            + 6.0 * ca2 * v3
            + 2.0 * v4
            - 2.0 * ca2 * v4
            - 4.0 * v * w
            + 4.0 * ca2 * v * w
            + 12.0 * v2 * w
            - ca2 * v2 * w
            - 11.0 * v3 * w
            - 11.0 * ca2 * v3 * w
            - v4 * w
            + 26.0 * ca2 * v4 * w
            + 4.0 * v5 * w
            - 14.0 * ca2 * v5 * w
            + 6.0 * v2 * w2
            - 6.0 * ca2 * v2 * w2
            - 26.0 * v3 * w2
            + 13.0 * ca2 * v3 * w2
            + 76.0 * v4 * w2
            - 62.0 * ca2 * v4 * w2
            - 92.0 * v5 * w2
            + 67.0 * ca2 * v5 * w2
            + 36.0 * v6 * w2
            - 24.0 * ca2 * v6 * w2
            - 4.0 * v3 * w3
            + 4.0 * ca2 * v3 * w3
            - 20.0 * v4 * w3
            + 27.0 * ca2 * v4 * w3
            + 50.0 * v5 * w3
            - 54.0 * ca2 * v5 * w3
            - 30.0 * v6 * w3
            + 49.0 * ca2 * v6 * w3
            + 4.0 * v7 * w3
            - 14.0 * ca2 * v7 * w3
            + v4 * w4
            - ca2 * v4 * w4
            - 6.0 * v5 * w4
            + 7.0 * ca2 * v5 * w4
            + 17.0 * v6 * w4
            - 27.0 * ca2 * v6 * w4
            - 14.0 * v7 * w4
            + 19.0 * ca2 * v7 * w4
            + 2.0 * v8 * w4
            - 2.0 * ca2 * v8 * w4
            - 4.0 * v6 * w5
            + 6.0 * ca2 * v6 * w5
            + 5.0 * v7 * w5
            - 7.0 * ca2 * v7 * w5
            - v8 * w5
            + ca2 * v8 * w5))
        / (3.0 * (1.0 - v).powi(2) * pre.v2 * w * (1.0 - v * w).powi(4));

    let part7 = -(4.0
        * cf
        * l1vw
        * (4.0 - 2.0 * ca2 + 2.0 * ca4 - 8.0 * ca2 * cf2 - 22.0 * v
            + 2.0 * ca * v
            + 9.0 * ca2 * v
            - 10.0 * ca4 * v
            - 8.0 * cacf * v
            + 8.0 * ca3cf * v
            + 24.0 * ca2 * cf2 * v
            + 52.0 * v2
            - 12.0 * ca * v2
            - 16.0 * ca2 * v2
            + 2.0 * ca3 * v2
            + 21.0 * ca4 * v2
            + 28.0 * cacf * v2
            - 28.0 * ca3cf * v2
            - 28.0 * ca2 * cf2 * v2
            - 68.0 * v3
            + 28.0 * ca * v3
            + 14.0 * ca2 * v3
            - 8.0 * ca3 * v3
            - 24.0 * ca4 * v3
            - 40.0 * cacf * v3
            + 40.0 * ca3cf * v3
            + 16.0 * ca2 * cf2 * v3
            + 52.0 * v4
            - 32.0 * ca * v4
            - 6.0 * ca2 * v4
            + 12.0 * ca3 * v4
            + 16.0 * ca4 * v4
            + 30.0 * cacf * v4
            - 30.0 * ca3cf * v4
            - 4.0 * ca2 * cf2 * v4
            - 22.0 * v5
            + 18.0 * ca * v5
            + ca2 * v5
            - 8.0 * ca3 * v5
            - 6.0 * ca4 * v5
            - 12.0 * cacf * v5
            + 12.0 * ca3cf * v5
            + 4.0 * v6
            - 4.0 * ca * v6
            + 2.0 * ca3 * v6
            + ca4 * v6
            + 2.0 * cacf * v6
            - 2.0 * ca3cf * v6
            + 12.0 * v * w
            - 2.0 * ca * v * w
            - 9.0 * ca2 * v * w
            + 12.0 * ca4 * v * w
            - 8.0 * ca3cf * v * w
            - 24.0 * ca2 * cf2 * v * w
            - 57.0 * v2 * w
            + 17.0 * ca * v2 * w
            + 35.0 * ca2 * v2 * w
            - 3.0 * ca3 * v2 * w
            - 49.0 * ca4 * v2 * w
            - 24.0 * cacf * v2 * w
            + 56.0 * ca3cf * v2 * w
            + 56.0 * ca2 * cf2 * v2 * w
            + 113.0 * v3 * w
            - 51.0 * ca * v3 * w
            - 50.0 * ca2 * v3 * w
            + 13.0 * ca3 * v3 * w
            + 82.0 * ca4 * v3 * w
            + 68.0 * cacf * v3 * w
            - 120.0 * ca3cf * v3 * w
            - 48.0 * ca2 * cf2 * v3 * w
            - 117.0 * v4 * w
            + 71.0 * ca * v4 * w
            + 30.0 * ca2 * v4 * w
            - 21.0 * ca3 * v4 * w
            - 72.0 * ca4 * v4 * w
            - 76.0 * cacf * v4 * w
            + 120.0 * ca3cf * v4 * w
            + 16.0 * ca2 * cf2 * v4 * w
            + 63.0 * v5 * w
            - 47.0 * ca * v5 * w
            - 5.0 * ca2 * v5 * w
            + 15.0 * ca3 * v5 * w
            + 34.0 * ca4 * v5 * w
            + 40.0 * cacf * v5 * w
            - 60.0 * ca3cf * v5 * w
            - 14.0 * v6 * w
            + 12.0 * ca * v6 * w
            - ca2 * v6 * w
            - 4.0 * ca3 * v6 * w
            - 7.0 * ca4 * v6 * w
            - 8.0 * cacf * v6 * w
            + 12.0 * ca3cf * v6 * w
            + 16.0 * v2 * w2
            - 5.0 * ca * v2 * w2
            - 24.0 * ca2 * v2 * w2
            + ca3 * v2 * w2
            + 31.0 * ca4 * v2 * w2
            - 32.0 * ca3cf * v2 * w2
            - 36.0 * ca2 * cf2 * v2 * w2
            - 66.0 * v3 * w2
            + 27.0 * ca * v3 * w2
            + 74.0 * ca2 * v3 * w2
            - 5.0 * ca3 * v3 * w2
            - 101.0 * ca4 * v3 * w2
            - 36.0 * cacf * v3 * w2
            + 140.0 * ca3cf * v3 * w2
            + 56.0 * ca2 * cf2 * v3 * w2
            + 106.0 * v4 * w2
            - 53.0 * ca * v4 * w2
            - 79.0 * ca2 * v4 * w2
            + 9.0 * ca3 * v4 * w2
            + 130.0 * ca4 * v4 * w2
            + 74.0 * cacf * v4 * w2
            - 206.0 * ca3cf * v4 * w2
            - 28.0 * ca2 * cf2 * v4 * w2
            - 78.0 * v5 * w2
            + 45.0 * ca * v5 * w2
            + 32.0 * ca2 * v5 * w2
            - 7.0 * ca3 * v5 * w2
            - 81.0 * ca4 * v5 * w2
            - 56.0 * cacf * v5 * w2
            + 136.0 * ca3cf * v5 * w2
            + 22.0 * v6 * w2
            - 14.0 * ca * v6 * w2
            - 3.0 * ca2 * v6 * w2
            + 2.0 * ca3 * v6 * w2
            + 21.0 * ca4 * v6 * w2
            + 14.0 * cacf * v6 * w2
            - 34.0 * ca3cf * v6 * w2
            + 13.0 * v3 * w3
            - 4.0 * ca * v3 * w3
            - 35.0 * ca2 * v3 * w3
            + 43.0 * ca4 * v3 * w3
            - 60.0 * ca3cf * v3 * w3
            - 24.0 * ca2 * cf2 * v3 * w3
            - 44.0 * v4 * w3
            + 16.0 * ca * v4 * w3
            + 79.0 * ca2 * v4 * w3
            - 108.0 * ca4 * v4 * w3
            - 24.0 * cacf * v4 * w3
            + 172.0 * ca3cf * v4 * w3
            + 24.0 * ca2 * cf2 * v4 * w3
            + 50.0 * v5 * w3
            - 20.0 * ca * v5 * w3
            - 59.0 * ca2 * v5 * w3
            + 99.0 * ca4 * v5 * w3
            + 36.0 * cacf * v5 * w3
            - 168.0 * ca3cf * v5 * w3
            - 19.0 * v6 * w3
            + 8.0 * ca * v6 * w3
            + 15.0 * ca2 * v6 * w3
            - 34.0 * ca4 * v6 * w3
            - 12.0 * cacf * v6 * w3
            + 56.0 * ca3cf * v6 * w3
            + 6.0 * v4 * w4
            - 2.0 * ca * v4 * w4
            - 28.0 * ca2 * v4 * w4
            + 35.0 * ca4 * v4 * w4
            - 60.0 * ca3cf * v4 * w4
            - 8.0 * ca2 * cf2 * v4 * w4
            - 14.0 * v5 * w4
            + 4.0 * ca * v5 * w4
            + 45.0 * ca2 * v5 * w4
            - 63.0 * ca4 * v5 * w4
            - 8.0 * cacf * v5 * w4
            + 112.0 * ca3cf * v5 * w4
            + 8.0 * v6 * w4
            - 2.0 * ca * v6 * w4
            - 21.0 * ca2 * v6 * w4
            + 32.0 * ca4 * v6 * w4
            + 4.0 * cacf * v6 * w4
            - 56.0 * ca3cf * v6 * w4
            + v5 * w5
            - 14.0 * ca2 * v5 * w5
            + 17.0 * ca4 * v5 * w5
            - 32.0 * ca3cf * v5 * w5
            - v6 * w5
            + 14.0 * ca2 * v6 * w5
            - 17.0 * ca4 * v6 * w5
            + 32.0 * ca3cf * v6 * w5
            - 4.0 * ca2 * v6 * w6
            + 4.0 * ca4 * v6 * w6
            - 8.0 * ca3cf * v6 * w6))
        / (ca * (1.0 - v).powi(2) * pre.v2 * w * (1.0 - v + v * w).powi(2));

    let part8 = -(2.0
        * cf
        * pre.lmss
        * (4.0 * ca2 - 28.0 * ca2 * v + 88.0 * ca2 * v2 - 160.0 * ca2 * v3 + 180.0 * ca2 * v4
            - 124.0 * ca2 * v5
            + 48.0 * ca2 * v6
            - 8.0 * ca2 * v7
            + 8.0 * cacf2 * w
            + 16.0 * ca2 * v * w
            + 8.0 * cf * v * w
            - 8.0 * ca2 * cf * v * w
            - 24.0 * cacf2 * v * w
            - 104.0 * ca2 * v2 * w
            - 28.0 * cf * v2 * w
            + 28.0 * ca2 * cf * v2 * w
            + 28.0 * cacf2 * v2 * w
            + 288.0 * ca2 * v3 * w
            + 40.0 * cf * v3 * w
            - 40.0 * ca2 * cf * v3 * w
            - 16.0 * cacf2 * v3 * w
            - 432.0 * ca2 * v4 * w
            - 30.0 * cf * v4 * w
            + 30.0 * ca2 * cf * v4 * w
            + 4.0 * cacf2 * v4 * w
            + 368.0 * ca2 * v5 * w
            + 12.0 * cf * v5 * w
            - 12.0 * ca2 * cf * v5 * w
            - 168.0 * ca2 * v6 * w
            - 2.0 * cf * v6 * w
            + 2.0 * ca2 * cf * v6 * w
            + 32.0 * ca2 * v7 * w
            + 16.0 * cacf2 * v * w2
            - 2.0 * v2 * w2
            + 32.0 * ca2 * v2 * w2
            + 16.0 * cf * v2 * w2
            - 32.0 * ca2 * cf * v2 * w2
            - 44.0 * cacf2 * v2 * w2
            + 10.0 * v3 * w2
            - 176.0 * ca2 * v3 * w2
            - 52.0 * cf * v3 * w2
            + 88.0 * ca2 * cf * v3 * w2
            + 48.0 * cacf2 * v3 * w2
            - 22.0 * v4 * w2
            + 392.0 * ca2 * v4 * w2
            + 70.0 * cf * v4 * w2
            - 98.0 * ca2 * cf * v4 * w2
            - 20.0 * cacf2 * v4 * w2
            + 26.0 * v5 * w2
            - 440.0 * ca2 * v5 * w2
            - 44.0 * cf * v5 * w2
            + 56.0 * ca2 * cf * v5 * w2
            - 16.0 * v6 * w2
            + 248.0 * ca2 * v6 * w2
            + 10.0 * cf * v6 * w2
            - 14.0 * ca2 * cf * v6 * w2
            + 4.0 * v7 * w2
            - 56.0 * ca2 * v7 * w2
            + 8.0 * cacf2 * v2 * w3
            - 6.0 * v3 * w3
            + 36.0 * ca2 * v3 * w3
            + 8.0 * cf * v3 * w3
            - 60.0 * ca2 * cf * v3 * w3
            - 24.0 * cacf2 * v3 * w3
            + 28.0 * v4 * w3
            - 160.0 * ca2 * v4 * w3
            - 28.0 * cf * v4 * w3
            + 116.0 * ca2 * cf * v4 * w3
            + 24.0 * cacf2 * v4 * w3
            - 50.0 * v5 * w3
            + 268.0 * ca2 * v5 * w3
            + 36.0 * cf * v5 * w3
            - 88.0 * ca2 * cf * v5 * w3
            + 40.0 * v6 * w3
            - 200.0 * ca2 * v6 * w3
            - 12.0 * cf * v6 * w3
            + 28.0 * ca2 * cf * v6 * w3
            - 12.0 * v7 * w3
            + 56.0 * ca2 * v7 * w3
            - 9.0 * v4 * w4
            + 25.0 * ca2 * v4 * w4
            - 60.0 * ca2 * cf * v4 * w4
            - 8.0 * cacf2 * v4 * w4
            + 31.0 * v5 * w4
            - 83.0 * ca2 * v5 * w4
            - 8.0 * cf * v5 * w4
            + 80.0 * ca2 * cf * v5 * w4
            - 36.0 * v6 * w4
            + 92.0 * ca2 * v6 * w4
            + 4.0 * cf * v6 * w4
            - 32.0 * ca2 * cf * v6 * w4
            + 14.0 * v7 * w4
            - 34.0 * ca2 * v7 * w4
            - 6.0 * v5 * w5
            + 10.0 * ca2 * v5 * w5
            - 32.0 * ca2 * cf * v5 * w5
            + 14.0 * v6 * w5
            - 22.0 * ca2 * v6 * w5
            + 24.0 * ca2 * cf * v6 * w5
            - 8.0 * v7 * w5
            + 12.0 * ca2 * v7 * w5
            - 2.0 * v6 * w6
            + 2.0 * ca2 * v6 * w6
            - 8.0 * ca2 * cf * v6 * w6
            + 2.0 * v7 * w6
            - 2.0 * ca2 * v7 * w6))
        / ((1.0 - v).powi(2) * pre.v2 * pre.w2 * (1.0 - v + v * w).powi(2));

    let part9 = -(4.0
        * cf
        * lw
        * (2.0 * ca - 4.0 * ca3 - 8.0 * ca * v + 22.0 * ca3 * v + 14.0 * ca * v2
            - 56.0 * ca3 * v2
            - 12.0 * ca * v3
            + 82.0 * ca3 * v3
            + 4.0 * ca * v4
            - 72.0 * ca3 * v4
            + 36.0 * ca3 * v5
            - 8.0 * ca3 * v6
            - 2.0 * w
            - 2.0 * ca * w
            + 5.0 * ca2 * w
            + 4.0 * ca3 * w
            - 3.0 * ca4 * w
            + 7.0 * v * w
            + 8.0 * ca * v * w
            - 15.0 * ca2 * v * w
            - 28.0 * ca3 * v * w
            + 8.0 * ca4 * v * w
            - 10.0 * v2 * w
            - 14.0 * ca * v2 * w
            + 22.0 * ca2 * v2 * w
            + 88.0 * ca3 * v2 * w
            - 12.0 * ca4 * v2 * w
            + 7.0 * v3 * w
            + 10.0 * ca * v3 * w
            - 17.0 * ca2 * v3 * w
            - 146.0 * ca3 * v3 * w
            + 10.0 * ca4 * v3 * w
            - 2.0 * v4 * w
            + 2.0 * ca * v4 * w
            + 7.0 * ca2 * v4 * w
            + 126.0 * ca3 * v4 * w
            - 5.0 * ca4 * v4 * w
            - 4.0 * ca * v5 * w
            - 2.0 * ca2 * v5 * w
            - 44.0 * ca3 * v5 * w
            + 2.0 * ca4 * v5 * w
            - 8.0 * ca3 * v6 * w
            + 8.0 * ca3 * v7 * w
            + w2
            - ca2 * w2
            - v * w2
            + 5.0 * ca2 * v * w2
            + 6.0 * ca3 * v * w2
            - 5.0 * ca4 * v * w2
            - 7.0 * v2 * w2
            + 3.0 * ca * v2 * w2
            - 12.0 * ca2 * v2 * w2
            - 37.0 * ca3 * v2 * w2
            + 16.0 * ca4 * v2 * w2
            + 20.0 * v3 * w2
            - 14.0 * ca * v3 * w2
            + 12.0 * ca2 * v3 * w2
            + 78.0 * ca3 * v3 * w2
            - 24.0 * ca4 * v3 * w2
            - 25.0 * v4 * w2
            + 21.0 * ca * v4 * w2
            - 47.0 * ca3 * v4 * w2
            + 16.0 * ca4 * v4 * w2
            + 16.0 * v5 * w2
            - 18.0 * ca * v5 * w2
            - 2.0 * ca2 * v5 * w2
            - 44.0 * ca3 * v5 * w2
            - 7.0 * ca4 * v5 * w2
            - 4.0 * v6 * w2
            + 8.0 * ca * v6 * w2
            + 2.0 * ca2 * v6 * w2
            + 72.0 * ca3 * v6 * w2
            - 28.0 * ca3 * v7 * w2
            - v * w3
            - ca2 * v * w3
            + 5.0 * v2 * w3
            - 3.0 * ca * v2 * w3
            + ca2 * v2 * w3
            + 5.0 * ca3 * v2 * w3
            - 7.0 * ca4 * v2 * w3
            - 14.0 * v3 * w3
            + 24.0 * ca * v3 * w3
            + 3.0 * ca2 * v3 * w3
            - 14.0 * ca3 * v3 * w3
            + 21.0 * ca4 * v3 * w3
            + 27.0 * v4 * w3
            - 52.0 * ca * v4 * w3
            - 17.0 * ca2 * v4 * w3
            - 24.0 * ca3 * v4 * w3
            - 21.0 * ca4 * v4 * w3
            - 27.0 * v5 * w3
            + 51.0 * ca * v5 * w3
            + 11.0 * ca2 * v5 * w3
            + 103.0 * ca3 * v5 * w3
            + 11.0 * ca4 * v5 * w3
            + 10.0 * v6 * w3
            - 20.0 * ca * v6 * w3
            - 5.0 * ca2 * v6 * w3
            - 110.0 * ca3 * v6 * w3
            + 40.0 * ca3 * v7 * w3
            + v2 * w4
            + ca2 * v2 * w4
            + v3 * w4
            - 8.0 * ca * v3 * w4
            - 6.0 * ca2 * v3 * w4
            - 6.0 * ca4 * v3 * w4
            - 15.0 * v4 * w4
            + 28.0 * ca * v4 * w4
            + 20.0 * ca2 * v4 * w4
            + 22.0 * ca3 * v4 * w4
            + 12.0 * ca4 * v4 * w4
            + 26.0 * v5 * w4
            - 36.0 * ca * v5 * w4
            - 13.0 * ca2 * v5 * w4
            - 70.0 * ca3 * v5 * w4
            - 9.0 * ca4 * v5 * w4
            - 13.0 * v6 * w4
            + 16.0 * ca * v6 * w4
            + 6.0 * ca2 * v6 * w4
            + 78.0 * ca3 * v6 * w4
            - ca4 * v6 * w4
            - 30.0 * ca3 * v7 * w4
            - v3 * w5
            - ca2 * v3 * w5
            + 7.0 * v4 * w5
            - 3.0 * ca * v4 * w5
            - 6.0 * ca2 * v4 * w5
            - 5.0 * ca3 * v4 * w5
            - 15.0 * v5 * w5
            + 7.0 * ca * v5 * w5
            + 3.0 * ca2 * v5 * w5
            + 21.0 * ca3 * v5 * w5
            + 2.0 * ca4 * v5 * w5
            + 9.0 * v6 * w5
            - 4.0 * ca * v6 * w5
            - 4.0 * ca2 * v6 * w5
            - 28.0 * ca3 * v6 * w5
            + 2.0 * ca4 * v6 * w5
            + 12.0 * ca3 * v7 * w5
            + 2.0 * ca2 * v4 * w6
            + 2.0 * v5 * w6
            + ca2 * v5 * w6
            - 2.0 * ca3 * v5 * w6
            + ca4 * v5 * w6
            - 2.0 * v6 * w6
            + ca2 * v6 * w6
            + 4.0 * ca3 * v6 * w6
            - ca4 * v6 * w6
            - 2.0 * ca3 * v7 * w6))
        / (ca
            * (1.0 - v).powi(2)
            * pre.v2
            * (1.0 - w)
            * pre.w2
            * (1.0 - v * w)
            * (1.0 - v + v * w));

    let part10 = (2.0
        * cf
        * lms
        * (4.0 * ca - 4.0 * ca3 - 12.0 * ca * v + 16.0 * ca3 * v + 16.0 * ca * v2
            - 32.0 * ca3 * v2
            - 8.0 * ca * v3
            + 36.0 * ca3 * v3
            - 24.0 * ca3 * v4
            + 8.0 * ca3 * v5
            - w
            - 2.0 * ca * w
            + 6.0 * ca2 * w
            + 2.0 * ca3 * w
            - 5.0 * ca4 * w
            - 8.0 * ca * v * w
            - 8.0 * ca2 * v * w
            + 4.0 * ca3 * v * w
            + 8.0 * ca4 * v * w
            + 4.0 * v2 * w
            + 30.0 * ca * v2 * w
            + 10.0 * ca2 * v2 * w
            - 30.0 * ca3 * v2 * w
            - 14.0 * ca4 * v2 * w
            - 6.0 * v3 * w
            - 48.0 * ca * v3 * w
            - 2.0 * ca2 * v3 * w
            + 84.0 * ca3 * v3 * w
            + 8.0 * ca4 * v3 * w
            + 3.0 * v4 * w
            + 28.0 * ca * v4 * w
            + 2.0 * ca2 * v4 * w
            - 116.0 * ca3 * v4 * w
            - 5.0 * ca4 * v4 * w
            + 88.0 * ca3 * v5 * w
            - 32.0 * ca3 * v6 * w
            + w2
            + ca * w2
            - 2.0 * ca2 * w2
            - ca3 * w2
            + ca4 * w2
            + 2.0 * v * w2
            + 5.0 * ca * v * w2
            - 22.0 * ca2 * v * w2
            - 3.0 * ca3 * v * w2
            + 20.0 * ca4 * v * w2
            - 4.0 * v2 * w2
            + 2.0 * ca * v2 * w2
            + 22.0 * ca2 * v2 * w2
            + 6.0 * ca3 * v2 * w2
            - 2.0 * ca4 * v2 * w2
            + 4.0 * v3 * w2
            - 20.0 * ca * v3 * w2
            - 36.0 * ca2 * v3 * w2
            - 6.0 * ca3 * v3 * w2
            + 16.0 * ca4 * v3 * w2
            - 3.0 * v4 * w2
            + 52.0 * ca * v4 * w2
            + 10.0 * ca2 * v4 * w2
            - 48.0 * ca3 * v4 * w2
            + 9.0 * ca4 * v4 * w2
            - 40.0 * ca * v5 * w2
            - 12.0 * ca2 * v5 * w2
            + 116.0 * ca3 * v5 * w2
            + 4.0 * ca4 * v5 * w2
            - 112.0 * ca3 * v6 * w2
            + 48.0 * ca3 * v7 * w2
            - 4.0 * v * w3
            - 4.0 * ca * v * w3
            + 8.0 * ca2 * v * w3
            + 4.0 * ca3 * v * w3
            - 4.0 * ca4 * v * w3
            + 2.0 * v2 * w3
            + 28.0 * ca2 * v2 * w3
            - 8.0 * ca3 * v2 * w3
            - 30.0 * ca4 * v2 * w3
            + 2.0 * v3 * w3
            - 2.0 * ca * v3 * w3
            - 2.0 * ca2 * v3 * w3
            + 10.0 * ca3 * v3 * w3
            - 16.0 * ca4 * v3 * w3
            - 6.0 * v4 * w3
            - 2.0 * ca * v4 * w3
            + 30.0 * ca2 * v4 * w3
            + 26.0 * ca3 * v4 * w3
            - 40.0 * ca4 * v4 * w3
            + 12.0 * v5 * w3
            - 24.0 * ca * v5 * w3
            - 32.0 * ca3 * v5 * w3
            - 4.0 * ca4 * v5 * w3
            - 6.0 * v6 * w3
            + 32.0 * ca * v6 * w3
            + 20.0 * ca2 * v6 * w3
            - 16.0 * ca3 * v6 * w3
            - 14.0 * ca4 * v6 * w3
            + 48.0 * ca3 * v7 * w3
            - 32.0 * ca3 * v8 * w3
            + 6.0 * v2 * w4
            + 6.0 * ca * v2 * w4
            - 12.0 * ca2 * v2 * w4
            - 6.0 * ca3 * v2 * w4
            + 6.0 * ca4 * v2 * w4
            - 8.0 * v3 * w4
            - 10.0 * ca * v3 * w4
            - 12.0 * ca2 * v3 * w4
            + 22.0 * ca3 * v3 * w4
            + 20.0 * ca4 * v3 * w4
            + 8.0 * v4 * w4
            + 18.0 * ca * v4 * w4
            - 48.0 * ca2 * v4 * w4
            - 50.0 * ca3 * v4 * w4
            + 40.0 * ca4 * v4 * w4
            - 12.0 * v5 * w4
            - 2.0 * ca * v5 * w4
            + 12.0 * ca2 * v5 * w4
            + 26.0 * ca3 * v5 * w4
            + 8.0 * ca4 * v5 * w4
            + 6.0 * v6 * w4
            + 8.0 * ca * v6 * w4
            - 28.0 * ca2 * v6 * w4
            + 16.0 * ca3 * v6 * w4
            + 38.0 * ca4 * v6 * w4
            - 20.0 * ca * v7 * w4
            - 12.0 * ca2 * v7 * w4
            - 24.0 * ca3 * v7 * w4
            + 4.0 * ca4 * v7 * w4
            + 8.0 * ca3 * v8 * w4
            + 8.0 * ca3 * v9 * w4
            - 4.0 * v3 * w5
            - 4.0 * ca * v3 * w5
            + 8.0 * ca2 * v3 * w5
            + 4.0 * ca3 * v3 * w5
            - 4.0 * ca4 * v3 * w5
            + 7.0 * v4 * w5
            + 10.0 * ca * v4 * w5
            - 2.0 * ca2 * v4 * w5
            - 18.0 * ca3 * v4 * w5
            - 5.0 * ca4 * v4 * w5
            - 8.0 * v5 * w5
            - 24.0 * ca * v5 * w5
            + 64.0 * ca2 * v5 * w5
            + 52.0 * ca3 * v5 * w5
            - 32.0 * ca4 * v5 * w5
            + 8.0 * v6 * w5
            + 10.0 * ca * v6 * w5
            - 18.0 * ca2 * v6 * w5
            - 50.0 * ca3 * v6 * w5
            - 6.0 * ca4 * v6 * w5
            - 6.0 * v7 * w5
            - 4.0 * ca * v7 * w5
            + 26.0 * ca2 * v7 * w5
            + 24.0 * ca3 * v7 * w5
            - 12.0 * ca4 * v7 * w5
            + 3.0 * v8 * w5
            + 12.0 * ca * v8 * w5
            + 2.0 * ca2 * v8 * w5
            - 4.0 * ca3 * v8 * w5
            - 5.0 * ca4 * v8 * w5
            - 8.0 * ca3 * v9 * w5
            + v4 * w6
            + ca * v4 * w6
            - 2.0 * ca2 * v4 * w6
            - ca3 * v4 * w6
            + ca4 * v4 * w6
            - 2.0 * v5 * w6
            - 3.0 * ca * v5 * w6
            + 2.0 * ca2 * v5 * w6
            + 5.0 * ca3 * v5 * w6
            + 4.0 * v6 * w6
            + 14.0 * ca * v6 * w6
            - 42.0 * ca2 * v6 * w6
            - 22.0 * ca3 * v6 * w6
            + 22.0 * ca4 * v6 * w6
            - 4.0 * ca * v7 * w6
            + 18.0 * ca3 * v7 * w6
            - 8.0 * ca4 * v7 * w6
            - 3.0 * v8 * w6
            - 4.0 * ca * v8 * w6
            - 6.0 * ca2 * v8 * w6
            - 8.0 * ca3 * v8 * w6
            + 9.0 * ca4 * v8 * w6
            - 4.0 * ca * v9 * w6
            + 8.0 * ca3 * v9 * w6
            - 2.0 * v7 * w7
            - 6.0 * ca * v7 * w7
            + 18.0 * ca2 * v7 * w7
            + 6.0 * ca3 * v7 * w7
            - 8.0 * ca4 * v7 * w7
            + 2.0 * v8 * w7
            + 2.0 * ca * v8 * w7
            + 2.0 * ca2 * v8 * w7
            - 2.0 * ca3 * v8 * w7
            - 4.0 * ca4 * v8 * w7
            + 4.0 * ca * v9 * w7
            - 4.0 * ca3 * v9 * w7
            + 2.0 * ca * v8 * w8
            - 4.0 * ca2 * v8 * w8
            - 2.0 * ca3 * v8 * w8
            + 4.0 * ca4 * v8 * w8
            - 2.0 * ca * v9 * w8
            + 2.0 * ca3 * v9 * w8))
        / (ca * (1.0 - v).powi(2) * pre.v2 * pre.w2 * (1.0 - v * w).powi(4));

    let part11 = -(2.0
        * cf
        * (12.0 * ca * v - 36.0 * ca3 * v - 48.0 * ca * v2 + 192.0 * ca3 * v2 + 72.0 * ca * v3
            - 432.0 * ca3 * v3
            - 48.0 * ca * v4
            + 528.0 * ca3 * v4
            + 12.0 * ca * v5
            - 372.0 * ca3 * v5
            + 144.0 * ca3 * v6
            - 24.0 * ca3 * v7
            - 9.0 * w
            + 3.0 * ca * w
            + 7.0 * ca2 * w
            - 3.0 * ca3 * w
            + 2.0 * ca4 * w
            + 48.0 * v * w
            - 30.0 * ca * v * w
            - 24.0 * ca2 * v * w
            + 54.0 * ca3 * v * w
            - 24.0 * ca4 * v * w
            - 132.0 * v2 * w
            + 84.0 * ca * v2 * w
            + 64.0 * ca2 * v2 * w
            - 204.0 * ca3 * v2 * w
            + 68.0 * ca4 * v2 * w
            + 228.0 * v3 * w
            - 72.0 * ca * v3 * w
            - 132.0 * ca2 * v3 * w
            + 216.0 * ca3 * v3 * w
            - 96.0 * ca4 * v3 * w
            - 237.0 * v4 * w
            - 33.0 * ca * v4 * w
            + 159.0 * ca2 * v4 * w
            + 273.0 * ca3 * v4 * w
            + 78.0 * ca4 * v4 * w
            + 132.0 * v5 * w
            + 78.0 * ca * v5 * w
            - 100.0 * ca2 * v5 * w
            - 918.0 * ca3 * v5 * w
            - 32.0 * ca4 * v5 * w
            - 30.0 * v6 * w
            - 30.0 * ca * v6 * w
            + 26.0 * ca2 * v6 * w
            + 966.0 * ca3 * v6 * w
            + 4.0 * ca4 * v6 * w
            - 480.0 * ca3 * v7 * w
            + 96.0 * ca3 * v8 * w
            - 3.0 * ca * w2
            + 6.0 * ca2 * w2
            + 3.0 * ca3 * w2
            - 6.0 * ca4 * w2
            + 27.0 * v * w2
            + 18.0 * ca * v * w2
            - 29.0 * ca2 * v * w2
            - 6.0 * ca3 * v * w2
            + 14.0 * ca4 * v * w2
            - 129.0 * v2 * w2
            - 12.0 * ca * v2 * w2
            + 16.0 * ca2 * v2 * w2
            - 66.0 * ca3 * v2 * w2
            - 47.0 * ca4 * v2 * w2
            + 243.0 * v3 * w2
            - 150.0 * ca * v3 * w2
            + 67.0 * ca2 * v3 * w2
            + 504.0 * ca3 * v3 * w2
            + 144.0 * ca4 * v3 * w2
            - 285.0 * v4 * w2
            + 387.0 * ca * v4 * w2
            - 53.0 * ca2 * v4 * w2
            - 1365.0 * ca3 * v4 * w2
            - 296.0 * ca4 * v4 * w2
            + 282.0 * v5 * w2
            - 336.0 * ca * v5 * w2
            - 82.0 * ca2 * v5 * w2
            + 1698.0 * ca3 * v5 * w2
            + 354.0 * ca4 * v5 * w2
            - 198.0 * v6 * w2
            + 84.0 * ca * v6 * w2
            + 131.0 * ca2 * v6 * w2
            - 780.0 * ca3 * v6 * w2
            - 227.0 * ca4 * v6 * w2
            + 60.0 * v7 * w2
            + 12.0 * ca * v7 * w2
            - 56.0 * ca2 * v7 * w2
            - 324.0 * ca3 * v7 * w2
            + 64.0 * ca4 * v7 * w2
            + 480.0 * ca3 * v8 * w2
            - 144.0 * ca3 * v9 * w2
            + 6.0 * ca * v * w3
            - 12.0 * ca2 * v * w3
            - 6.0 * ca3 * v * w3
            + 12.0 * ca4 * v * w3
            - 9.0 * v2 * w3
            - 57.0 * ca * v2 * w3
            + 35.0 * ca2 * v2 * w3
            + 33.0 * ca3 * v2 * w3
            - 50.0 * ca4 * v2 * w3
            + 144.0 * v3 * w3
            + 228.0 * ca * v3 * w3
            - 13.0 * ca2 * v3 * w3
            - 114.0 * ca3 * v3 * w3
            - 29.0 * ca4 * v3 * w3
            - 378.0 * v4 * w3
            - 330.0 * ca * v4 * w3
            - 151.0 * ca2 * v4 * w3
            + 78.0 * ca3 * v4 * w3
            + 413.0 * ca4 * v4 * w3
            + 384.0 * v5 * w3
            + 66.0 * ca * v5 * w3
            + 227.0 * ca2 * v5 * w3
            + 624.0 * ca3 * v5 * w3
            - 871.0 * ca4 * v5 * w3
            - 225.0 * v6 * w3
            + 183.0 * ca * v6 * w3
            - 76.0 * ca2 * v6 * w3
            - 1815.0 * ca3 * v6 * w3
            + 817.0 * ca4 * v6 * w3
            + 144.0 * v7 * w3
            - 84.0 * ca * v7 * w3
            - 58.0 * ca2 * v7 * w3
            + 2052.0 * ca3 * v7 * w3
            - 328.0 * ca4 * v7 * w3
            - 60.0 * v8 * w3
            - 12.0 * ca * v8 * w3
            + 60.0 * ca2 * v8 * w3
            - 948.0 * ca3 * v8 * w3
            + 24.0 * ca4 * v8 * w3
            + 96.0 * ca3 * v10 * w3
            + 3.0 * ca * v2 * w4
            - 6.0 * ca2 * v2 * w4
            - 3.0 * ca3 * v2 * w4
            + 6.0 * ca4 * v2 * w4
            - 45.0 * v3 * w4
            + 6.0 * ca * v3 * w4
            + 7.0 * ca2 * v3 * w4
            - 18.0 * ca3 * v3 * w4
            + 26.0 * ca4 * v3 * w4
            + 51.0 * v4 * w4
            - 183.0 * ca * v4 * w4
            + 83.0 * ca2 * v4 * w4
            + 183.0 * ca3 * v4 * w4
            - 184.0 * ca4 * v4 * w4
            + 213.0 * v5 * w4
            + 525.0 * ca * v5 * w4
            - 87.0 * ca2 * v5 * w4
            - 597.0 * ca3 * v5 * w4
            + 644.0 * ca4 * v5 * w4
            - 423.0 * v6 * w4
            - 549.0 * ca * v6 * w4
            - ca2 * v6 * w4
            + 993.0 * ca3 * v6 * w4
            - 1086.0 * ca4 * v6 * w4
            + 288.0 * v7 * w4
            + 282.0 * ca * v7 * w4
            - 52.0 * ca2 * v7 * w4
            - 678.0 * ca3 * v7 * w4
            + 856.0 * ca4 * v7 * w4
            - 144.0 * v8 * w4
            - 156.0 * ca * v8 * w4
            + 52.0 * ca2 * v8 * w4
            - 312.0 * ca3 * v8 * w4
            - 310.0 * ca4 * v8 * w4
            + 60.0 * v9 * w4
            + 72.0 * ca * v9 * w4
            - 56.0 * ca2 * v9 * w4
            + 696.0 * ca3 * v9 * w4
            + 64.0 * ca4 * v9 * w4
            - 240.0 * ca3 * v10 * w4
            - 24.0 * ca3 * v11 * w4
            - 12.0 * ca * v3 * w5
            + 24.0 * ca2 * v3 * w5
            + 12.0 * ca3 * v3 * w5
            - 24.0 * ca4 * v3 * w5
            + 45.0 * v4 * w5
            + 81.0 * ca * v4 * w5
            - 43.0 * ca2 * v4 * w5
            - 33.0 * ca3 * v4 * w5
            + 46.0 * ca4 * v4 * w5
            - 216.0 * v5 * w5
            - 144.0 * ca * v5 * w5
            - 114.0 * ca2 * v5 * w5
            - 24.0 * ca3 * v5 * w5
            - 58.0 * ca4 * v5 * w5
            + 264.0 * v6 * w5
            + 54.0 * ca * v6 * w5
            + 210.0 * ca2 * v6 * w5
            + 138.0 * ca3 * v6 * w5
            + 478.0 * ca4 * v6 * w5
            - 132.0 * v7 * w5
            - 48.0 * ca * v7 * w5
            + 24.0 * ca2 * v7 * w5
            - 348.0 * ca3 * v7 * w5
            - 850.0 * ca4 * v7 * w5
            + 81.0 * v8 * w5
            + 93.0 * ca * v8 * w5
            - 41.0 * ca2 * v8 * w5
            + 819.0 * ca3 * v8 * w5
            + 620.0 * ca4 * v8 * w5
            - 12.0 * v9 * w5
            + 54.0 * ca * v9 * w5
            + 46.0 * ca2 * v9 * w5
            - 810.0 * ca3 * v9 * w5
            - 216.0 * ca4 * v9 * w5
            - 30.0 * v10 * w5
            - 78.0 * ca * v10 * w5
            + 26.0 * ca2 * v10 * w5
            + 150.0 * ca3 * v10 * w5
            + 4.0 * ca4 * v10 * w5
            + 96.0 * ca3 * v11 * w5
            + 3.0 * ca * v4 * w6
            - 6.0 * ca2 * v4 * w6
            - 3.0 * ca3 * v4 * w6
            + 6.0 * ca4 * v4 * w6
            + 9.0 * v5 * w6
            - 42.0 * ca * v5 * w6
            + 25.0 * ca2 * v5 * w6
            + 30.0 * ca3 * v5 * w6
            - 46.0 * ca4 * v5 * w6
            + 69.0 * v6 * w6
            + 132.0 * ca * v6 * w6
            + 58.0 * ca2 * v6 * w6
            - 18.0 * ca3 * v6 * w6
            - 179.0 * ca4 * v6 * w6
            - 123.0 * v7 * w6
            - 114.0 * ca * v7 * w6
            - 215.0 * ca2 * v7 * w6
            + 12.0 * ca3 * v7 * w6
            + 540.0 * ca4 * v7 * w6
            + 81.0 * v8 * w6
            + 165.0 * ca * v8 * w6
            + 31.0 * ca2 * v8 * w6
            - 339.0 * ca3 * v8 * w6
            - 600.0 * ca4 * v8 * w6
            - 114.0 * v9 * w6
            - 312.0 * ca * v9 * w6
            - 2.0 * ca2 * v9 * w6
            + 426.0 * ca3 * v9 * w6
            + 334.0 * ca4 * v9 * w6
            + 78.0 * v10 * w6
            + 144.0 * ca * v10 * w6
            - 71.0 * ca2 * v10 * w6
            + 60.0 * ca3 * v10 * w6
            - 7.0 * ca4 * v10 * w6
            + 24.0 * ca * v11 * w6
            - 168.0 * ca3 * v11 * w6
            + 6.0 * ca * v5 * w7
            - 12.0 * ca2 * v5 * w7
            - 6.0 * ca3 * v5 * w7
            + 12.0 * ca4 * v5 * w7
            - 27.0 * v6 * w7
            - 27.0 * ca * v6 * w7
            + ca2 * v6 * w7
            + 3.0 * ca3 * v6 * w7
            + 2.0 * ca4 * v6 * w7
            + 24.0 * v7 * w7
            + 24.0 * ca * v7 * w7
            + 31.0 * ca2 * v7 * w7
            - 6.0 * ca3 * v7 * w7
            - 89.0 * ca4 * v7 * w7
            - 42.0 * v8 * w7
            - 114.0 * ca * v8 * w7
            + 89.0 * ca2 * v8 * w7
            + 174.0 * ca3 * v8 * w7
            + 205.0 * ca4 * v8 * w7
            + 120.0 * v9 * w7
            + 210.0 * ca * v9 * w7
            - 23.0 * ca2 * v9 * w7
            - 180.0 * ca3 * v9 * w7
            - 247.0 * ca4 * v9 * w7
            - 75.0 * v10 * w7
            - 27.0 * ca * v10 * w7
            + 94.0 * ca2 * v10 * w7
            - 165.0 * ca3 * v10 * w7
            - 19.0 * ca4 * v10 * w7
            - 72.0 * ca * v11 * w7
            + 180.0 * ca3 * v11 * w7
            - 3.0 * ca * v6 * w8
            + 6.0 * ca2 * v6 * w8
            + 3.0 * ca3 * v6 * w8
            - 6.0 * ca4 * v6 * w8
            + 9.0 * v7 * w8
            + 18.0 * ca * v7 * w8
            - 3.0 * ca2 * v7 * w8
            - 6.0 * ca3 * v7 * w8
            + 6.0 * ca4 * v7 * w8
            + 9.0 * v8 * w8
            - 3.0 * ca * v8 * w8
            - 49.0 * ca2 * v8 * w8
            - 33.0 * ca3 * v8 * w8
            - 18.0 * ca4 * v8 * w8
            - 45.0 * v9 * w8
            - 15.0 * ca * v9 * w8
            - 13.0 * ca2 * v9 * w8
            + 3.0 * ca3 * v9 * w8
            + 104.0 * ca4 * v9 * w8
            + 27.0 * v10 * w8
            - 75.0 * ca * v10 * w8
            - 73.0 * ca2 * v10 * w8
            + 159.0 * ca3 * v10 * w8
            + 46.0 * ca4 * v10 * w8
            + 78.0 * ca * v11 * w8
            - 126.0 * ca3 * v11 * w8
            - 6.0 * ca * v9 * w9
            + 24.0 * ca2 * v9 * w9
            + 18.0 * ca3 * v9 * w9
            - 24.0 * ca4 * v9 * w9
            + 42.0 * ca * v10 * w9
            + 36.0 * ca2 * v10 * w9
            - 66.0 * ca3 * v10 * w9
            - 36.0 * ca4 * v10 * w9
            - 36.0 * ca * v11 * w9
            + 48.0 * ca3 * v11 * w9
            - 6.0 * ca * v10 * w10
            - 12.0 * ca2 * v10 * w10
            + 6.0 * ca3 * v10 * w10
            + 12.0 * ca4 * v10 * w10
            + 6.0 * ca * v11 * w10
            - 6.0 * ca3 * v11 * w10))
        / (3.0
            * ca
            * (1.0 - v).powi(2)
            * pre.v2
            * pre.w2
            * (1.0 - v * w).powi(4)
            * (1.0 - v + v * w).powi(2));

    let part12 = -(2.0
        * cf
        * lv
        * (4.0 * ca - 16.0 * ca3 - 20.0 * ca * v + 100.0 * ca3 * v + 44.0 * ca * v2
            - 284.0 * ca3 * v2
            - 52.0 * ca * v3
            + 468.0 * ca3 * v3
            + 32.0 * ca * v4
            - 476.0 * ca3 * v4
            - 8.0 * ca * v5
            + 296.0 * ca3 * v5
            - 104.0 * ca3 * v6
            + 16.0 * ca3 * v7
            - 3.0 * w
            - 2.0 * ca * w
            + 18.0 * ca2 * w
            + 2.0 * ca3 * w
            - 15.0 * ca4 * w
            + 12.0 * v * w
            + 20.0 * ca * v * w
            - 72.0 * ca2 * v * w
            - 8.0 * ca3 * v * w
            + 64.0 * ca4 * v * w
            - 18.0 * v2 * w
            - 84.0 * ca * v2 * w
            + 126.0 * ca2 * v2 * w
            - 12.0 * ca3 * v2 * w
            - 128.0 * ca4 * v2 * w
            + 10.0 * v3 * w
            + 196.0 * ca * v3 * w
            - 118.0 * ca2 * v3 * w
            + 180.0 * ca3 * v3 * w
            + 152.0 * ca4 * v3 * w
            + 3.0 * v4 * w
            - 278.0 * ca * v4 * w
            + 56.0 * ca2 * v4 * w
            - 578.0 * ca3 * v4 * w
            - 111.0 * ca4 * v4 * w
            - 6.0 * v5 * w
            + 248.0 * ca * v5 * w
            - 10.0 * ca2 * v5 * w
            + 908.0 * ca3 * v5 * w
            + 48.0 * ca4 * v5 * w
            + 2.0 * v6 * w
            - 132.0 * ca * v6 * w
            - 772.0 * ca3 * v6 * w
            - 10.0 * ca4 * v6 * w
            + 32.0 * ca * v7 * w
            + 344.0 * ca3 * v7 * w
            - 64.0 * ca3 * v8 * w
            + w2
            + ca * w2
            - 2.0 * ca2 * w2
            - ca3 * w2
            + ca4 * w2
            + 2.0 * v * w2
            - ca * v * w2
            - 30.0 * ca2 * v * w2
            + 3.0 * ca3 * v * w2
            + 28.0 * ca4 * v * w2
            - 30.0 * v2 * w2
            - 13.0 * ca * v2 * w2
            + 142.0 * ca2 * v2 * w2
            + 31.0 * ca3 * v2 * w2
            - 76.0 * ca4 * v2 * w2
            + 88.0 * v3 * w2
            + 51.0 * ca * v3 * w2
            - 276.0 * ca2 * v3 * w2
            - 233.0 * ca3 * v3 * w2
            + 100.0 * ca4 * v3 * w2
            - 135.0 * v4 * w2
            - 152.0 * ca * v4 * w2
            + 310.0 * ca2 * v4 * w2
            + 654.0 * ca3 * v4 * w2
            - 127.0 * ca4 * v4 * w2
            + 122.0 * v5 * w2
            + 338.0 * ca * v5 * w2
            - 186.0 * ca2 * v5 * w2
            - 806.0 * ca3 * v5 * w2
            + 116.0 * ca4 * v5 * w2
            - 60.0 * v6 * w2
            - 452.0 * ca * v6 * w2
            + 46.0 * ca2 * v6 * w2
            + 252.0 * ca3 * v6 * w2
            - 62.0 * ca4 * v6 * w2
            + 12.0 * v7 * w2
            + 324.0 * ca * v7 * w2
            - 4.0 * ca2 * v7 * w2
            + 356.0 * ca3 * v7 * w2
            + 20.0 * ca4 * v7 * w2
            - 96.0 * ca * v8 * w2
            - 352.0 * ca3 * v8 * w2
            + 96.0 * ca3 * v9 * w2
            - 2.0 * v * w3
            - 2.0 * ca * v * w3
            + 4.0 * ca2 * v * w3
            + 2.0 * ca3 * v * w3
            - 2.0 * ca4 * v * w3
            + 13.0 * v2 * w3
            + 14.0 * ca * v2 * w3
            - 34.0 * ca2 * v2 * w3
            - 18.0 * ca3 * v2 * w3
            + 21.0 * ca4 * v2 * w3
            - 28.0 * v3 * w3
            - 46.0 * ca * v3 * w3
            + 64.0 * ca2 * v3 * w3
            + 76.0 * ca3 * v3 * w3
            - 28.0 * ca4 * v3 * w3
            + 30.0 * v4 * w3
            + 176.0 * ca * v4 * w3
            - 72.0 * ca2 * v4 * w3
            - 144.0 * ca3 * v4 * w3
            + 94.0 * ca4 * v4 * w3
            + 12.0 * v5 * w3
            - 476.0 * ca * v5 * w3
            - 30.0 * ca2 * v5 * w3
            - 150.0 * ca3 * v5 * w3
            - 70.0 * ca4 * v5 * w3
            - 93.0 * v6 * w3
            + 626.0 * ca * v6 * w3
            + 108.0 * ca2 * v6 * w3
            + 918.0 * ca3 * v6 * w3
            - 39.0 * ca4 * v6 * w3
            + 104.0 * v7 * w3
            - 292.0 * ca * v7 * w3
            - 40.0 * ca2 * v7 * w3
            - 1172.0 * ca3 * v7 * w3
            + 48.0 * ca4 * v7 * w3
            - 36.0 * v8 * w3
            - 96.0 * ca * v8 * w3
            + 8.0 * ca2 * v8 * w3
            + 520.0 * ca3 * v8 * w3
            - 32.0 * ca4 * v8 * w3
            + 96.0 * ca * v9 * w3
            + 32.0 * ca3 * v9 * w3
            - 64.0 * ca3 * v10 * w3
            - v2 * w4
            - ca * v2 * w4
            + 2.0 * ca2 * v2 * w4
            + ca3 * v2 * w4
            - ca4 * v2 * w4
            - 14.0 * v3 * w4
            - 9.0 * ca * v3 * w4
            + 78.0 * ca2 * v3 * w4
            + 7.0 * ca3 * v3 * w4
            - 64.0 * ca4 * v3 * w4
            + 70.0 * v4 * w4
            + 15.0 * ca * v4 * w4
            - 248.0 * ca2 * v4 * w4
            - 71.0 * ca3 * v4 * w4
            + 66.0 * ca4 * v4 * w4
            - 172.0 * v5 * w4
            + 65.0 * ca * v5 * w4
            + 460.0 * ca2 * v5 * w4
            + 365.0 * ca3 * v5 * w4
            - 136.0 * ca4 * v5 * w4
            + 277.0 * v6 * w4
            + 6.0 * ca * v6 * w4
            - 452.0 * ca2 * v6 * w4
            - 826.0 * ca3 * v6 * w4
            + 307.0 * ca4 * v6 * w4
            - 212.0 * v7 * w4
            - 456.0 * ca * v7 * w4
            + 144.0 * ca2 * v7 * w4
            + 664.0 * ca3 * v7 * w4
            - 212.0 * ca4 * v7 * w4
            + 24.0 * v8 * w4
            + 632.0 * ca * v8 * w4
            - 28.0 * ca2 * v8 * w4
            + 120.0 * ca3 * v8 * w4
            + 100.0 * ca4 * v8 * w4
            + 28.0 * v9 * w4
            - 220.0 * ca * v9 * w4
            - 4.0 * ca2 * v9 * w4
            - 412.0 * ca3 * v9 * w4
            + 12.0 * ca4 * v9 * w4
            - 32.0 * ca * v10 * w4
            + 136.0 * ca3 * v10 * w4
            + 16.0 * ca3 * v11 * w4
            + 4.0 * v3 * w5
            + 4.0 * ca * v3 * w5
            - 8.0 * ca2 * v3 * w5
            - 4.0 * ca3 * v3 * w5
            + 4.0 * ca4 * v3 * w5
            - 9.0 * v4 * w5
            - 14.0 * ca * v4 * w5
            - 2.0 * ca2 * v4 * w5
            + 22.0 * ca3 * v4 * w5
            + 11.0 * ca4 * v4 * w5
            + 10.0 * v5 * w5
            + 4.0 * ca * v5 * w5
            + 50.0 * ca2 * v5 * w5
            - 80.0 * ca3 * v5 * w5
            - 36.0 * v6 * w5
            - 152.0 * ca * v6 * w5
            - 148.0 * ca2 * v6 * w5
            + 144.0 * ca3 * v6 * w5
            - 180.0 * ca4 * v6 * w5
            - 30.0 * v7 * w5
            + 560.0 * ca * v7 * w5
            + 314.0 * ca2 * v7 * w5
            + 124.0 * ca3 * v7 * w5
            + 100.0 * ca4 * v7 * w5
            + 157.0 * v8 * w5
            - 554.0 * ca * v8 * w5
            - 136.0 * ca2 * v8 * w5
            - 574.0 * ca3 * v8 * w5
            - 65.0 * ca4 * v8 * w5
            - 90.0 * v9 * w5
            + 12.0 * ca * v9 * w5
            + 42.0 * ca2 * v9 * w5
            + 500.0 * ca3 * v9 * w5
            - 48.0 * ca4 * v9 * w5
            - 6.0 * v10 * w5
            + 140.0 * ca * v10 * w5
            - 76.0 * ca3 * v10 * w5
            - 6.0 * ca4 * v10 * w5
            - 56.0 * ca3 * v11 * w5
            - v4 * w6
            - ca * v4 * w6
            + 2.0 * ca2 * v4 * w6
            + ca3 * v4 * w6
            - ca4 * v4 * w6
            + 14.0 * v5 * w6
            + 13.0 * ca * v5 * w6
            - 50.0 * ca2 * v5 * w6
            - 15.0 * ca3 * v5 * w6
            + 36.0 * ca4 * v5 * w6
            - 26.0 * v6 * w6
            + 29.0 * ca * v6 * w6
            + 102.0 * ca2 * v6 * w6
            + 53.0 * ca3 * v6 * w6
            + 56.0 * ca4 * v6 * w6
            + 96.0 * v7 * w6
            - 175.0 * ca * v7 * w6
            - 180.0 * ca2 * v7 * w6
            - 203.0 * ca3 * v7 * w6
            + 12.0 * ca4 * v7 * w6
            - 177.0 * v8 * w6
            + 60.0 * ca * v8 * w6
            - 2.0 * ca2 * v8 * w6
            + 366.0 * ca3 * v8 * w6
            + 23.0 * ca4 * v8 * w6
            + 66.0 * v9 * w6
            + 286.0 * ca * v9 * w6
            + 2.0 * ca2 * v9 * w6
            - 242.0 * ca3 * v9 * w6
            + 48.0 * ca4 * v9 * w6
            + 28.0 * v10 * w6
            - 204.0 * ca * v10 * w6
            - 10.0 * ca2 * v10 * w6
            - 48.0 * ca3 * v10 * w6
            + 26.0 * ca4 * v10 * w6
            - 8.0 * ca * v11 * w6
            + 88.0 * ca3 * v11 * w6
            - 2.0 * v5 * w7
            - 2.0 * ca * v5 * w7
            + 4.0 * ca2 * v5 * w7
            + 2.0 * ca3 * v5 * w7
            - 2.0 * ca4 * v5 * w7
            - v6 * w7
            + 2.0 * ca * v6 * w7
            + 18.0 * ca2 * v6 * w7
            - 6.0 * ca3 * v6 * w7
            - 17.0 * ca4 * v6 * w7
            - 16.0 * v7 * w7
            + 10.0 * ca * v7 * w7
            - 28.0 * ca2 * v7 * w7
            + 28.0 * ca3 * v7 * w7
            - 20.0 * ca4 * v7 * w7
            + 46.0 * v8 * w7
            + 88.0 * ca * v8 * w7
            + 112.0 * ca2 * v8 * w7
            - 56.0 * ca3 * v8 * w7
            - 2.0 * ca4 * v8 * w7
            + 16.0 * v9 * w7
            - 240.0 * ca * v9 * w7
            - 14.0 * ca2 * v9 * w7
            + 18.0 * ca3 * v9 * w7
            - 22.0 * ca4 * v9 * w7
            - 43.0 * v10 * w7
            + 118.0 * ca * v10 * w7
            + 12.0 * ca2 * v10 * w7
            + 98.0 * ca3 * v10 * w7
            - 41.0 * ca4 * v10 * w7
            + 24.0 * ca * v11 * w7
            - 84.0 * ca3 * v11 * w7
            + v6 * w8
            + ca * v6 * w8
            - 2.0 * ca2 * v6 * w8
            - ca3 * v6 * w8
            + ca4 * v6 * w8
            - 2.0 * v7 * w8
            - 3.0 * ca * v7 * w8
            + 2.0 * ca2 * v7 * w8
            + 5.0 * ca3 * v7 * w8
            + 2.0 * v8 * w8
            - 23.0 * ca * v8 * w8
            - 20.0 * ca2 * v8 * w8
            - 9.0 * ca3 * v8 * w8
            - 6.0 * ca4 * v8 * w8
            - 28.0 * v9 * w8
            + 67.0 * ca * v9 * w8
            - 36.0 * ca2 * v9 * w8
            + 15.0 * ca3 * v9 * w8
            + 8.0 * ca4 * v9 * w8
            + 27.0 * v10 * w8
            - 14.0 * ca * v10 * w8
            - 8.0 * ca2 * v10 * w8
            - 62.0 * ca3 * v10 * w8
            + 37.0 * ca4 * v10 * w8
            - 28.0 * ca * v11 * w8
            + 52.0 * ca3 * v11 * w8
            + 6.0 * v9 * w9
            - 4.0 * ca * v9 * w9
            + 18.0 * ca2 * v9 * w9
            - 6.0 * v10 * w9
            - 12.0 * ca * v10 * w9
            + 14.0 * ca2 * v10 * w9
            + 20.0 * ca3 * v10 * w9
            - 24.0 * ca4 * v10 * w9
            + 16.0 * ca * v11 * w9
            - 20.0 * ca3 * v11 * w9
            + 4.0 * ca * v10 * w10
            - 8.0 * ca2 * v10 * w10
            - 4.0 * ca3 * v10 * w10
            + 8.0 * ca4 * v10 * w10
            - 4.0 * ca * v11 * w10
            + 4.0 * ca3 * v11 * w10))
        / (ca
            * (1.0 - v).powi(2)
            * pre.v2
            * pre.w2
            * (1.0 - v * w).powi(4)
            * (1.0 - v + v * w).powi(2));

    let part13 = -(2.0
        * cf
        * l1w
        * (4.0 * ca - 12.0 * ca3 - 20.0 * ca * v + 76.0 * ca3 * v + 44.0 * ca * v2
            - 220.0 * ca3 * v2
            - 52.0 * ca * v3
            + 372.0 * ca3 * v3
            + 32.0 * ca * v4
            - 392.0 * ca3 * v4
            - 8.0 * ca * v5
            + 256.0 * ca3 * v5
            - 96.0 * ca3 * v6
            + 16.0 * ca3 * v7
            - 5.0 * w
            - 2.0 * ca * w
            + 16.0 * ca2 * w
            + 2.0 * ca3 * w
            - 11.0 * ca4 * w
            + 24.0 * v * w
            + 16.0 * ca * v * w
            - 62.0 * ca2 * v * w
            - 8.0 * ca3 * v * w
            + 46.0 * ca4 * v * w
            - 50.0 * v2 * w
            - 60.0 * ca * v2 * w
            + 104.0 * ca2 * v2 * w
            - 8.0 * ca3 * v2 * w
            - 92.0 * ca4 * v2 * w
            + 62.0 * v3 * w
            + 140.0 * ca * v3 * w
            - 88.0 * ca2 * v3 * w
            + 148.0 * ca3 * v3 * w
            + 110.0 * ca4 * v3 * w
            - 51.0 * v4 * w
            - 214.0 * ca * v4 * w
            + 28.0 * ca2 * v4 * w
            - 482.0 * ca3 * v4 * w
            - 81.0 * ca4 * v4 * w
            + 26.0 * v5 * w
            + 212.0 * ca * v5 * w
            + 6.0 * ca2 * v5 * w
            + 772.0 * ca3 * v5 * w
            + 36.0 * ca4 * v5 * w
            - 6.0 * v6 * w
            - 124.0 * ca * v6 * w
            - 4.0 * ca2 * v6 * w
            - 680.0 * ca3 * v6 * w
            - 8.0 * ca4 * v6 * w
            + 32.0 * ca * v7 * w
            + 320.0 * ca3 * v7 * w
            - 64.0 * ca3 * v8 * w
            + w2
            + ca * w2
            - 2.0 * ca2 * w2
            - ca3 * w2
            + ca4 * w2
            + 4.0 * v * w2
            - ca * v * w2
            - 28.0 * ca2 * v * w2
            + 3.0 * ca3 * v * w2
            + 20.0 * ca4 * v * w2
            - 46.0 * v2 * w2
            - 17.0 * ca * v2 * w2
            + 120.0 * ca2 * v2 * w2
            + 23.0 * ca3 * v2 * w2
            - 50.0 * ca4 * v2 * w2
            + 136.0 * v3 * w2
            + 59.0 * ca * v3 * w2
            - 222.0 * ca2 * v3 * w2
            - 177.0 * ca3 * v3 * w2
            + 60.0 * ca4 * v3 * w2
            - 219.0 * v4 * w2
            - 136.0 * ca * v4 * w2
            + 240.0 * ca2 * v4 * w2
            + 490.0 * ca3 * v4 * w2
            - 75.0 * ca4 * v4 * w2
            + 224.0 * v5 * w2
            + 282.0 * ca * v5 * w2
            - 114.0 * ca2 * v5 * w2
            - 582.0 * ca3 * v5 * w2
            + 64.0 * ca4 * v5 * w2
            - 136.0 * v6 * w2
            - 400.0 * ca * v6 * w2
            - 6.0 * ca2 * v6 * w2
            + 128.0 * ca3 * v6 * w2
            - 36.0 * ca4 * v6 * w2
            + 36.0 * v7 * w2
            + 308.0 * ca * v7 * w2
            + 12.0 * ca2 * v7 * w2
            + 356.0 * ca3 * v7 * w2
            + 16.0 * ca4 * v7 * w2
            - 96.0 * ca * v8 * w2
            - 336.0 * ca3 * v8 * w2
            + 96.0 * ca3 * v9 * w2
            - 2.0 * v * w3
            - 2.0 * ca * v * w3
            + 4.0 * ca2 * v * w3
            + 2.0 * ca3 * v * w3
            - 2.0 * ca4 * v * w3
            + 23.0 * v2 * w3
            + 14.0 * ca * v2 * w3
            - 28.0 * ca2 * v2 * w3
            - 18.0 * ca3 * v2 * w3
            + 17.0 * ca4 * v2 * w3
            - 60.0 * v3 * w3
            - 34.0 * ca * v3 * w3
            + 52.0 * ca2 * v3 * w3
            + 72.0 * ca3 * v3 * w3
            - 20.0 * ca4 * v3 * w3
            + 66.0 * v4 * w3
            + 104.0 * ca * v4 * w3
            - 76.0 * ca2 * v4 * w3
            - 124.0 * ca3 * v4 * w3
            + 64.0 * ca4 * v4 * w3
            - 8.0 * v5 * w3
            - 328.0 * ca * v5 * w3
            - 24.0 * ca2 * v5 * w3
            - 126.0 * ca3 * v5 * w3
            - 28.0 * ca4 * v5 * w3
            - 103.0 * v6 * w3
            + 498.0 * ca * v6 * w3
            + 88.0 * ca2 * v6 * w3
            + 730.0 * ca3 * v6 * w3
            - 43.0 * ca4 * v6 * w3
            + 144.0 * v7 * w3
            - 252.0 * ca * v7 * w3
            + 8.0 * ca2 * v7 * w3
            - 912.0 * ca3 * v7 * w3
            + 36.0 * ca4 * v7 * w3
            - 60.0 * v8 * w3
            - 96.0 * ca * v8 * w3
            - 16.0 * ca2 * v8 * w3
            + 392.0 * ca3 * v8 * w3
            - 32.0 * ca4 * v8 * w3
            + 96.0 * ca * v9 * w3
            + 48.0 * ca3 * v9 * w3
            - 64.0 * ca3 * v10 * w3
            - v2 * w4
            - ca * v2 * w4
            + 2.0 * ca2 * v2 * w4
            + ca3 * v2 * w4
            - ca4 * v2 * w4
            - 32.0 * v3 * w4
            - 9.0 * ca * v3 * w4
            + 68.0 * ca2 * v3 * w4
            + 7.0 * ca3 * v3 * w4
            - 48.0 * ca4 * v3 * w4
            + 146.0 * v4 * w4
            + 27.0 * ca * v4 * w4
            - 186.0 * ca2 * v4 * w4
            - 71.0 * ca3 * v4 * w4
            + 52.0 * ca4 * v4 * w4
            - 284.0 * v5 * w4
            + 41.0 * ca * v5 * w4
            + 374.0 * ca2 * v5 * w4
            + 317.0 * ca3 * v5 * w4
            - 140.0 * ca4 * v5 * w4
            + 363.0 * v6 * w4
            - 22.0 * ca * v6 * w4
            - 370.0 * ca2 * v6 * w4
            - 638.0 * ca3 * v6 * w4
            + 277.0 * ca4 * v6 * w4
            - 268.0 * v7 * w4
            - 360.0 * ca * v7 * w4
            + 64.0 * ca2 * v7 * w4
            + 448.0 * ca3 * v7 * w4
            - 184.0 * ca4 * v7 * w4
            + 40.0 * v8 * w4
            + 560.0 * ca * v8 * w4
            - 20.0 * ca2 * v8 * w4
            + 156.0 * ca3 * v8 * w4
            + 100.0 * ca4 * v8 * w4
            + 36.0 * v9 * w4
            - 204.0 * ca * v9 * w4
            + 12.0 * ca2 * v9 * w4
            - 348.0 * ca3 * v9 * w4
            + 16.0 * ca4 * v9 * w4
            - 32.0 * ca * v10 * w4
            + 112.0 * ca3 * v10 * w4
            + 16.0 * ca3 * v11 * w4
            + 4.0 * v3 * w5
            + 4.0 * ca * v3 * w5
            - 8.0 * ca2 * v3 * w5
            - 4.0 * ca3 * v3 * w5
            + 4.0 * ca4 * v3 * w5
            - 7.0 * v4 * w5
            - 14.0 * ca * v4 * w5
            + 4.0 * ca2 * v4 * w5
            + 22.0 * ca3 * v4 * w5
            + 7.0 * ca4 * v4 * w5
            - 30.0 * v5 * w5
            - 8.0 * ca * v5 * w5
            + 10.0 * ca2 * v5 * w5
            - 68.0 * ca3 * v5 * w5
            + 4.0 * ca4 * v5 * w5
            + 56.0 * v6 * w5
            - 80.0 * ca * v6 * w5
            - 76.0 * ca2 * v6 * w5
            + 84.0 * ca3 * v6 * w5
            - 130.0 * ca4 * v6 * w5
            - 94.0 * v7 * w5
            + 432.0 * ca * v7 * w5
            + 276.0 * ca2 * v7 * w5
            + 152.0 * ca3 * v7 * w5
            + 50.0 * ca4 * v7 * w5
            + 175.0 * v8 * w5
            - 474.0 * ca * v8 * w5
            - 76.0 * ca2 * v8 * w5
            - 446.0 * ca3 * v8 * w5
            - 55.0 * ca4 * v8 * w5
            - 98.0 * v9 * w5
            + 8.0 * ca * v9 * w5
            + 10.0 * ca2 * v9 * w5
            + 348.0 * ca3 * v9 * w5
            - 56.0 * ca4 * v9 * w5
            - 6.0 * v10 * w5
            + 132.0 * ca * v10 * w5
            - 4.0 * ca2 * v10 * w5
            - 40.0 * ca3 * v10 * w5
            - 8.0 * ca4 * v10 * w5
            - 48.0 * ca3 * v11 * w5
            - v4 * w6
            - ca * v4 * w6
            + 2.0 * ca2 * v4 * w6
            + ca3 * v4 * w6
            - ca4 * v4 * w6
            + 28.0 * v5 * w6
            + 13.0 * ca * v5 * w6
            - 52.0 * ca2 * v5 * w6
            - 15.0 * ca3 * v5 * w6
            + 28.0 * ca4 * v5 * w6
            - 50.0 * v6 * w6
            + 17.0 * ca * v6 * w6
            + 92.0 * ca2 * v6 * w6
            + 61.0 * ca3 * v6 * w6
            + 38.0 * ca4 * v6 * w6
            + 80.0 * v7 * w6
            - 151.0 * ca * v7 * w6
            - 186.0 * ca2 * v7 * w6
            - 179.0 * ca3 * v7 * w6
            + 28.0 * ca4 * v7 * w6
            - 133.0 * v8 * w6
            + 68.0 * ca * v8 * w6
            - 20.0 * ca2 * v8 * w6
            + 250.0 * ca3 * v8 * w6
            + 35.0 * ca4 * v8 * w6
            + 52.0 * v9 * w6
            + 246.0 * ca * v9 * w6
            + 10.0 * ca2 * v9 * w6
            - 138.0 * ca3 * v9 * w6
            + 40.0 * ca4 * v9 * w6
            + 24.0 * v10 * w6
            - 184.0 * ca * v10 * w6
            + 2.0 * ca2 * v10 * w6
            - 44.0 * ca3 * v10 * w6
            + 32.0 * ca4 * v10 * w6
            - 8.0 * ca * v11 * w6
            + 64.0 * ca3 * v11 * w6
            - 2.0 * v5 * w7
            - 2.0 * ca * v5 * w7
            + 4.0 * ca2 * v5 * w7
            + 2.0 * ca3 * v5 * w7
            - 2.0 * ca4 * v5 * w7
            - 11.0 * v6 * w7
            + 2.0 * ca * v6 * w7
            + 12.0 * ca2 * v6 * w7
            - 6.0 * ca3 * v6 * w7
            - 13.0 * ca4 * v6 * w7
            + 16.0 * v7 * w7
            + 14.0 * ca * v7 * w7
            - 16.0 * ca2 * v7 * w7
            + 16.0 * ca3 * v7 * w7
            - 12.0 * ca4 * v7 * w7
            + 10.0 * v8 * w7
            + 64.0 * ca * v8 * w7
            + 100.0 * ca2 * v8 * w7
            - 12.0 * ca3 * v8 * w7
            - 28.0 * ca4 * v8 * w7
            + 16.0 * v9 * w7
            - 204.0 * ca * v9 * w7
            - 12.0 * ca2 * v9 * w7
            - 6.0 * ca3 * v9 * w7
            - 4.0 * ca4 * v9 * w7
            - 29.0 * v10 * w7
            + 102.0 * ca * v10 * w7
            + 62.0 * ca3 * v10 * w7
            - 45.0 * ca4 * v10 * w7
            + 24.0 * ca * v11 * w7
            - 56.0 * ca3 * v11 * w7
            + v6 * w8
            + ca * v6 * w8
            - 2.0 * ca2 * v6 * w8
            - ca3 * v6 * w8
            + ca4 * v6 * w8
            - 3.0 * ca * v7 * w8
            + 12.0 * ca2 * v7 * w8
            + 5.0 * ca3 * v7 * w8
            - 2.0 * v8 * w8
            - 19.0 * ca * v8 * w8
            - 18.0 * ca2 * v8 * w8
            - 13.0 * ca3 * v8 * w8
            - 12.0 * v9 * w8
            + 59.0 * ca * v9 * w8
            - 30.0 * ca2 * v9 * w8
            + 7.0 * ca3 * v9 * w8
            + 4.0 * ca4 * v9 * w8
            + 13.0 * v10 * w8
            - 10.0 * ca * v10 * w8
            - 2.0 * ca2 * v10 * w8
            - 34.0 * ca3 * v10 * w8
            + 35.0 * ca4 * v10 * w8
            - 28.0 * ca * v11 * w8
            + 36.0 * ca3 * v11 * w8
            - 4.0 * ca2 * v8 * w9
            + 2.0 * v9 * w9
            - 4.0 * ca * v9 * w9
            + 16.0 * ca2 * v9 * w9
            + 4.0 * ca3 * v9 * w9
            - 2.0 * ca4 * v9 * w9
            - 2.0 * v10 * w9
            - 12.0 * ca * v10 * w9
            + 12.0 * ca2 * v10 * w9
            + 12.0 * ca3 * v10 * w9
            - 22.0 * ca4 * v10 * w9
            + 16.0 * ca * v11 * w9
            - 16.0 * ca3 * v11 * w9
            + 4.0 * ca * v10 * w10
            - 8.0 * ca2 * v10 * w10
            - 4.0 * ca3 * v10 * w10
            + 8.0 * ca4 * v10 * w10
            - 4.0 * ca * v11 * w10
            + 4.0 * ca3 * v11 * w10))
        / (ca
            * (1.0 - v).powi(2)
            * pre.v2
            * pre.w2
            * (1.0 - v * w).powi(4)
            * (1.0 - v + v * w).powi(2));

    part1
        + part2
        + part3
        + part4
        + part5
        + part6
        + part7
        + part8
        + part9
        + part10
        + part11
        + part12
        + part13
}

/// `STRUV14(W,V,X3,S)`.
#[must_use]
pub fn struv14(w: f64, v: f64, _x3: f64, _s: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let cf = ctx.cf;
    let ca = ctx.ca;
    let (v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12) = (
        pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8, pre.v9, pre.v10, pre.v11, pre.v12,
    );
    let (w2, w3, w4, w5, w6, w7, w8, w9, w10) = (
        pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8, pre.w9, pre.w10,
    );
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let ca4 = pre.ca4;
    let cacf = ca * cf;
    let ca3cf = ca3 * cf;
    let (l1v, lw, lvw, l1vw, lms, l1w, lv, lmss) = (
        pre.l1v, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.l1w, pre.lv, pre.lmss,
    );

    let part1 = (4.0
        * cf
        * lvw
        * (16.0 * ca4 - 32.0 * ca4 * v + 48.0 * ca4 * v2 - 32.0 * ca4 * v3 + 16.0 * ca4 * v4
            - ca2 * v * w
            - 20.0 * ca4 * v * w
            + v2 * w
            + 2.0 * ca2 * v2 * w
            - 15.0 * ca4 * v2 * w
            - 3.0 * v3 * w
            + 21.0 * ca4 * v3 * w
            + 4.0 * v4 * w
            - 3.0 * ca2 * v4 * w
            - 40.0 * ca4 * v4 * w
            - 2.0 * v5 * w
            + 2.0 * ca2 * v5 * w
            + 6.0 * ca4 * v5 * w
            - 6.0 * ca2 * v2 * w2
            + 40.0 * ca4 * v2 * w2
            + 9.0 * ca2 * v3 * w2
            - 29.0 * ca4 * v3 * w2
            - 3.0 * v4 * w2
            - 18.0 * ca2 * v4 * w2
            + 72.0 * ca4 * v4 * w2
            + 3.0 * v5 * w2
            + 15.0 * ca2 * v5 * w2
            - 27.0 * ca4 * v5 * w2
            - 8.0 * ca2 * v6 * w2
            + 8.0 * ca4 * v6 * w2
            - 15.0 * ca4 * v3 * w3
            + 5.0 * v4 * w3
            + 25.0 * ca2 * v4 * w3
            - 42.0 * ca4 * v4 * w3
            - 5.0 * v5 * w3
            - 25.0 * ca2 * v5 * w3
            + 25.0 * ca4 * v5 * w3
            + 16.0 * ca2 * v6 * w3
            - 16.0 * ca4 * v6 * w3
            - 2.0 * v4 * w4
            - 10.0 * ca2 * v4 * w4
            + 16.0 * ca4 * v4 * w4
            + 2.0 * v5 * w4
            + 10.0 * ca2 * v5 * w4
            - 8.0 * ca4 * v5 * w4
            - 12.0 * ca2 * v6 * w4
            + 12.0 * ca4 * v6 * w4
            + 4.0 * ca2 * v6 * w5
            - 4.0 * ca4 * v6 * w5))
        / (ca * (1.0 - v).powi(2) * v3 * w2);

    let part2 = -(4.0
        * cf
        * l1v
        * (8.0 * ca4 - 32.0 * ca4 * v + 64.0 * ca4 * v2 - 80.0 * ca4 * v3 + 64.0 * ca4 * v4
            - 32.0 * ca4 * v5
            + 8.0 * ca4 * v6
            + 8.0 * ca4 * w
            - 34.0 * ca4 * v * w
            + 51.0 * ca4 * v2 * w
            - 39.0 * ca4 * v3 * w
            + v4 * w
            - ca2 * v4 * w
            + 18.0 * ca4 * v4 * w
            - 2.0 * v5 * w
            + 2.0 * ca2 * v5 * w
            - 9.0 * ca4 * v5 * w
            + v6 * w
            - ca2 * v6 * w
            + 13.0 * ca4 * v6 * w
            - 8.0 * ca4 * v7 * w
            + 2.0 * ca4 * v3 * w2
            - 2.0 * v4 * w2
            + 2.0 * ca2 * v4 * w2
            + 3.0 * ca4 * v4 * w2
            + 4.0 * v5 * w2
            - 2.0 * ca2 * v5 * w2
            - 22.0 * ca4 * v5 * w2
            - v6 * w2
            - ca2 * v6 * w2
            + 14.0 * ca4 * v6 * w2
            - v7 * w2
            + ca2 * v7 * w2
            - 13.0 * ca4 * v7 * w2
            + 8.0 * ca4 * v8 * w2
            - 8.0 * ca4 * v2 * w3
            + 26.0 * ca4 * v3 * w3
            + 2.0 * v4 * w3
            - 2.0 * ca2 * v4 * w3
            - 41.0 * ca4 * v4 * w3
            - 4.0 * v5 * w3
            + 56.0 * ca4 * v5 * w3
            - v6 * w3
            + 3.0 * ca2 * v6 * w3
            - 38.0 * ca4 * v6 * w3
            + 3.0 * v7 * w3
            - ca2 * v7 * w3
            + 37.0 * ca4 * v7 * w3
            - 16.0 * ca4 * v8 * w3
            + 4.0 * ca2 * v5 * w4
            - 8.0 * ca4 * v5 * w4
            + 4.0 * v6 * w4
            - 4.0 * ca2 * v6 * w4
            + 10.0 * ca4 * v6 * w4
            - 4.0 * v7 * w4
            - 26.0 * ca4 * v7 * w4
            + 12.0 * ca4 * v8 * w4
            - 2.0 * v6 * w5
            - 2.0 * ca2 * v6 * w5
            + 2.0 * v7 * w5
            + 2.0 * ca2 * v7 * w5
            + 8.0 * ca4 * v7 * w5
            - 4.0 * ca4 * v8 * w5))
        / (ca * (1.0 - v).powi(2) * v3 * w2 * (1.0 - v * w) * (1.0 - v + v * w));

    let part3 = -(4.0
        * cf
        * l1vw
        * (ca2 + 4.0 * ca4 - 3.0 * ca2 * v - 12.0 * ca4 * v + 3.0 * ca2 * v2 + 12.0 * ca4 * v2
            - 3.0 * ca2 * v4
            - 12.0 * ca4 * v4
            + 3.0 * ca2 * v5
            + 12.0 * ca4 * v5
            - ca2 * v6
            - 4.0 * ca4 * v6
            + v * w
            + 3.0 * ca2 * v * w
            + 5.0 * ca4 * v * w
            - 2.0 * v2 * w
            - 5.0 * ca2 * v2 * w
            - 15.0 * ca4 * v2 * w
            + v3 * w
            - 8.0 * ca2 * v3 * w
            + 3.0 * ca4 * v3 * w
            + v4 * w
            + 30.0 * ca2 * v4 * w
            + 29.0 * ca4 * v4 * w
            - 2.0 * v5 * w
            - 31.0 * ca2 * v5 * w
            - 32.0 * ca4 * v5 * w
            + v6 * w
            + 11.0 * ca2 * v6 * w
            + 10.0 * ca4 * v6 * w
            + 2.0 * v2 * w2
            + 3.0 * ca2 * v2 * w2
            + 7.0 * ca4 * v2 * w2
            - v3 * w2
            + 3.0 * ca2 * v3 * w2
            - 2.0 * ca4 * v3 * w2
            - 8.0 * v4 * w2
            - 47.0 * ca2 * v4 * w2
            - 33.0 * ca4 * v4 * w2
            + 13.0 * v5 * w2
            + 71.0 * ca2 * v5 * w2
            + 34.0 * ca4 * v5 * w2
            - 6.0 * v6 * w2
            - 30.0 * ca2 * v6 * w2
            - 6.0 * ca4 * v6 * w2
            + v3 * w3
            + 5.0 * ca2 * v3 * w3
            - 2.0 * ca4 * v3 * w3
            + 11.0 * v4 * w3
            + 22.0 * ca2 * v4 * w3
            + 23.0 * ca4 * v4 * w3
            - 24.0 * v5 * w3
            - 69.0 * ca2 * v5 * w3
            - 21.0 * ca4 * v5 * w3
            + 14.0 * v6 * w3
            + 36.0 * ca2 * v6 * w3
            - 2.0 * ca4 * v6 * w3
            - 4.0 * v4 * w4
            - 2.0 * ca2 * v4 * w4
            - 7.0 * ca4 * v4 * w4
            + 17.0 * v5 * w4
            + 34.0 * ca2 * v5 * w4
            + 8.0 * ca4 * v5 * w4
            - 16.0 * v6 * w4
            - 23.0 * ca2 * v6 * w4
            + 2.0 * ca4 * v6 * w4
            - 4.0 * v5 * w5
            - 8.0 * ca2 * v5 * w5
            - ca4 * v5 * w5
            + 9.0 * v6 * w5
            + 9.0 * ca2 * v6 * w5
            - 2.0 * v6 * w6
            - 2.0 * ca2 * v6 * w6))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v + v * w).powi(3));

    let part4 = (4.0
        * cf
        * lw
        * (24.0 * ca4 - 96.0 * ca4 * v + 208.0 * ca4 * v2 - 288.0 * ca4 * v3 + 272.0 * ca4 * v4
            - 176.0 * ca4 * v5
            + 72.0 * ca4 * v6
            - 16.0 * ca4 * v7
            - 8.0 * ca4 * w
            + 32.0 * ca4 * v * w
            + 2.0 * v2 * w
            - ca2 * v2 * w
            - 129.0 * ca4 * v2 * w
            - 6.0 * v3 * w
            + 3.0 * ca2 * v3 * w
            + 275.0 * ca4 * v3 * w
            + 8.0 * v4 * w
            - 5.0 * ca2 * v4 * w
            - 323.0 * ca4 * v4 * w
            - 6.0 * v5 * w
            + 5.0 * ca2 * v5 * w
            + 225.0 * ca4 * v5 * w
            + 2.0 * v6 * w
            - 2.0 * ca2 * v6 * w
            - 56.0 * ca4 * v6 * w
            - 16.0 * ca4 * v7 * w
            + 16.0 * ca4 * v8 * w
            + 4.0 * ca2 * v2 * w2
            + 29.0 * ca4 * v2 * w2
            - 12.0 * ca2 * v3 * w2
            - 87.0 * ca4 * v3 * w2
            - v4 * w2
            + 19.0 * ca2 * v4 * w2
            + 62.0 * ca4 * v4 * w2
            + 2.0 * v5 * w2
            - 18.0 * ca2 * v5 * w2
            + 21.0 * ca4 * v5 * w2
            + v6 * w2
            + 7.0 * ca2 * v6 * w2
            - 169.0 * ca4 * v6 * w2
            - 2.0 * v7 * w2
            + 144.0 * ca4 * v7 * w2
            - 56.0 * ca4 * v8 * w2
            + 2.0 * ca4 * v2 * w3
            - 6.0 * ca4 * v3 * w3
            + 4.0 * v4 * w3
            - 7.0 * ca2 * v4 * w3
            + 85.0 * ca4 * v4 * w3
            - 8.0 * v5 * w3
            + 14.0 * ca2 * v5 * w3
            - 160.0 * ca4 * v5 * w3
            - v6 * w3
            - 7.0 * ca2 * v6 * w3
            + 299.0 * ca4 * v6 * w3
            + 5.0 * v7 * w3
            - 220.0 * ca4 * v7 * w3
            + 80.0 * ca4 * v8 * w3
            - 2.0 * v4 * w4
            + 4.0 * ca2 * v4 * w4
            - 41.0 * ca4 * v4 * w4
            + 4.0 * v5 * w4
            - 8.0 * ca2 * v5 * w4
            + 82.0 * ca4 * v5 * w4
            + 7.0 * v6 * w4
            + 4.0 * ca2 * v6 * w4
            - 194.0 * ca4 * v6 * w4
            - 9.0 * v7 * w4
            + 153.0 * ca4 * v7 * w4
            - 60.0 * ca4 * v8 * w4
            + 6.0 * ca4 * v4 * w5
            - 12.0 * ca4 * v5 * w5
            - 8.0 * v6 * w5
            + 57.0 * ca4 * v6 * w5
            + 8.0 * v7 * w5
            - 51.0 * ca4 * v7 * w5
            + 24.0 * ca4 * v8 * w5
            + 2.0 * v6 * w6
            - 6.0 * ca4 * v6 * w6
            - 2.0 * v7 * w6
            + 6.0 * ca4 * v7 * w6
            - 4.0 * ca4 * v8 * w6))
        / (ca * (1.0 - v).powi(2) * v3 * (1.0 - w) * w2 * (1.0 - v * w) * (1.0 - v + v * w));

    let part5 = -(2.0
        * cf
        * lms
        * (32.0 * ca4 - 80.0 * ca4 * v + 144.0 * ca4 * v2 - 144.0 * ca4 * v3 + 112.0 * ca4 * v4
            - 48.0 * ca4 * v5
            + 16.0 * ca4 * v6
            - 4.0 * ca2 * w
            + 4.0 * ca4 * w
            + 12.0 * ca2 * v * w
            - 108.0 * ca4 * v * w
            + 3.0 * v2 * w
            - 20.0 * ca2 * v2 * w
            + 177.0 * ca4 * v2 * w
            - 7.0 * v3 * w
            + 22.0 * ca2 * v3 * w
            - 319.0 * ca4 * v3 * w
            + 7.0 * v4 * w
            - 16.0 * ca2 * v4 * w
            + 297.0 * ca4 * v4 * w
            - 3.0 * v5 * w
            + 6.0 * ca2 * v5 * w
            - 275.0 * ca4 * v5 * w
            + 128.0 * ca4 * v6 * w
            - 48.0 * ca4 * v7 * w
            + 4.0 * ca2 * w2
            + 4.0 * ca4 * w2
            - 24.0 * ca4 * v * w2
            - 2.0 * v2 * w2
            - 18.0 * ca2 * v2 * w2
            + 164.0 * ca4 * v2 * w2
            - 4.0 * v3 * w2
            + 44.0 * ca2 * v3 * w2
            - 96.0 * ca4 * v3 * w2
            + 13.0 * v4 * w2
            - 66.0 * ca2 * v4 * w2
            + 213.0 * ca4 * v4 * w2
            - 14.0 * v5 * w2
            + 54.0 * ca2 * v5 * w2
            - 128.0 * ca4 * v5 * w2
            + 7.0 * v6 * w2
            - 26.0 * ca2 * v6 * w2
            + 195.0 * ca4 * v6 * w2
            - 96.0 * ca4 * v7 * w2
            + 48.0 * ca4 * v8 * w2
            - 12.0 * ca2 * v * w3
            - 12.0 * ca4 * v * w3
            + 24.0 * ca2 * v2 * w3
            + 48.0 * ca4 * v2 * w3
            + 6.0 * v3 * w3
            - 18.0 * ca2 * v3 * w3
            - 164.0 * ca4 * v3 * w3
            - 3.0 * v4 * w3
            + 2.0 * ca2 * v4 * w3
            - 23.0 * ca4 * v4 * w3
            - 6.0 * v5 * w3
            + 44.0 * ca2 * v5 * w3
            - 54.0 * ca4 * v5 * w3
            + 10.0 * v6 * w3
            - 42.0 * ca2 * v6 * w3
            - 80.0 * ca4 * v6 * w3
            - 7.0 * v7 * w3
            + 34.0 * ca2 * v7 * w3
            - 11.0 * ca4 * v7 * w3
            - 16.0 * ca4 * v9 * w3
            + 12.0 * ca2 * v2 * w4
            + 12.0 * ca4 * v2 * w4
            - 32.0 * ca2 * v3 * w4
            - 40.0 * ca4 * v3 * w4
            - 6.0 * v4 * w4
            + 42.0 * ca2 * v4 * w4
            + 108.0 * ca4 * v4 * w4
            + 8.0 * v5 * w4
            - 56.0 * ca2 * v5 * w4
            + 24.0 * ca4 * v5 * w4
            - 7.0 * v6 * w4
            + 2.0 * ca2 * v6 * w4
            + 77.0 * ca4 * v6 * w4
            + 4.0 * v7 * w4
            + 2.0 * ca2 * v7 * w4
            + 26.0 * ca4 * v7 * w4
            + v8 * w4
            - 22.0 * ca2 * v8 * w4
            - 3.0 * ca4 * v8 * w4
            + 16.0 * ca4 * v9 * w4
            - 4.0 * ca2 * v3 * w5
            - 4.0 * ca4 * v3 * w5
            + 12.0 * ca2 * v4 * w5
            + 12.0 * ca4 * v4 * w5
            + 2.0 * v5 * w5
            - 18.0 * ca2 * v5 * w5
            - 32.0 * ca4 * v5 * w5
            - 2.0 * v6 * w5
            + 38.0 * ca2 * v6 * w5
            - 20.0 * ca4 * v6 * w5
            + v7 * w5
            - 2.0 * ca2 * v7 * w5
            - 55.0 * ca4 * v7 * w5
            - v8 * w5
            + 10.0 * ca2 * v8 * w5
            + 15.0 * ca4 * v8 * w5
            + 8.0 * ca2 * v9 * w5
            - 16.0 * ca4 * v9 * w5
            - 8.0 * ca2 * v7 * w6
            + 16.0 * ca4 * v7 * w6
            - 4.0 * ca2 * v8 * w6
            + 4.0 * ca4 * v8 * w6
            - 8.0 * ca2 * v9 * w6
            + 8.0 * ca4 * v9 * w6
            + 4.0 * ca2 * v9 * w7
            - 4.0 * ca4 * v9 * w7))
        / (ca * (1.0 - v).powi(2) * v3 * w2 * (1.0 - v * w).powi(3));

    let part6 = -(2.0
        * cf
        * lmss
        * (16.0 * ca4 - 112.0 * ca4 * v + 368.0 * ca4 * v2 - 752.0 * ca4 * v3 + 1056.0 * ca4 * v4
            - 1056.0 * ca4 * v5
            + 752.0 * ca4 * v6
            - 368.0 * ca4 * v7
            + 112.0 * ca4 * v8
            - 16.0 * ca4 * v9
            + 16.0 * ca4 * w
            - 48.0 * ca4 * v * w
            + v2 * w
            - 2.0 * ca2 * v2 * w
            - 79.0 * ca4 * v2 * w
            - 4.0 * v3 * w
            + 8.0 * ca2 * v3 * w
            + 684.0 * ca4 * v3 * w
            + 7.0 * v4 * w
            - 14.0 * ca2 * v4 * w
            - 1721.0 * ca4 * v4 * w
            - 8.0 * v5 * w
            + 16.0 * ca2 * v5 * w
            + 2488.0 * ca4 * v5 * w
            + 7.0 * v6 * w
            - 14.0 * ca2 * v6 * w
            - 2313.0 * ca4 * v6 * w
            - 4.0 * v7 * w
            + 8.0 * ca2 * v7 * w
            + 1388.0 * ca4 * v7 * w
            + v8 * w
            - 2.0 * ca2 * v8 * w
            - 495.0 * ca4 * v8 * w
            + 80.0 * ca4 * v9 * w
            + 48.0 * ca4 * v * w2
            - 192.0 * ca4 * v2 * w2
            + 2.0 * v3 * w2
            - 6.0 * ca2 * v3 * w2
            + 148.0 * ca4 * v3 * w2
            - 8.0 * v4 * w2
            + 10.0 * ca2 * v4 * w2
            + 638.0 * ca4 * v4 * w2
            + 16.0 * v5 * w2
            + 8.0 * ca2 * v5 * w2
            - 2008.0 * ca4 * v5 * w2
            - 20.0 * v6 * w2
            - 32.0 * ca2 * v6 * w2
            + 2764.0 * ca4 * v6 * w2
            + 14.0 * v7 * w2
            + 38.0 * ca2 * v7 * w2
            - 2156.0 * ca4 * v7 * w2
            - 4.0 * v8 * w2
            - 26.0 * ca2 * v8 * w2
            + 934.0 * ca4 * v8 * w2
            + 8.0 * ca2 * v9 * w2
            - 176.0 * ca4 * v9 * w2
            + 48.0 * ca4 * v2 * w3
            - 176.0 * ca4 * v3 * w3
            + 3.0 * v4 * w3
            + 8.0 * ca2 * v4 * w3
            + 85.0 * ca4 * v4 * w3
            - 12.0 * v5 * w3
            - 60.0 * ca2 * v5 * w3
            + 664.0 * ca4 * v5 * w3
            + 22.0 * v6 * w3
            + 136.0 * ca2 * v6 * w3
            - 1654.0 * ca4 * v6 * w3
            - 20.0 * v7 * w3
            - 156.0 * ca2 * v7 * w3
            + 1792.0 * ca4 * v7 * w3
            + 7.0 * v8 * w3
            + 104.0 * ca2 * v8 * w3
            - 983.0 * ca4 * v8 * w3
            - 32.0 * ca2 * v9 * w3
            + 224.0 * ca4 * v9 * w3
            + 16.0 * ca4 * v3 * w4
            - 48.0 * ca4 * v4 * w4
            + 2.0 * v5 * w4
            + 26.0 * ca2 * v5 * w4
            - 76.0 * ca4 * v5 * w4
            - 10.0 * v6 * w4
            - 110.0 * ca2 * v6 * w4
            + 512.0 * ca4 * v6 * w4
            + 14.0 * v7 * w4
            + 174.0 * ca2 * v7 * w4
            - 844.0 * ca4 * v7 * w4
            - 6.0 * v8 * w4
            - 142.0 * ca2 * v8 * w4
            + 620.0 * ca4 * v8 * w4
            + 52.0 * ca2 * v9 * w4
            - 180.0 * ca4 * v9 * w4
            + 2.0 * v6 * w5
            + 22.0 * ca2 * v6 * w5
            - 64.0 * ca4 * v6 * w5
            - 4.0 * v7 * w5
            - 72.0 * ca2 * v7 * w5
            + 204.0 * ca4 * v7 * w5
            + 2.0 * v8 * w5
            + 86.0 * ca2 * v8 * w5
            - 224.0 * ca4 * v8 * w5
            - 44.0 * ca2 * v9 * w5
            + 92.0 * ca4 * v9 * w5
            + 8.0 * ca2 * v7 * w6
            - 16.0 * ca4 * v7 * w6
            - 20.0 * ca2 * v8 * w6
            + 36.0 * ca4 * v8 * w6
            + 20.0 * ca2 * v9 * w6
            - 28.0 * ca4 * v9 * w6
            - 4.0 * ca2 * v9 * w7
            + 4.0 * ca4 * v9 * w7))
        / (ca * (1.0 - v).powi(2) * v3 * w2 * (1.0 - v + v * w).powi(3));

    let part7 = -(cf
        * (12.0 * ca2 - 28.0 * ca4 - 72.0 * ca2 * v + 168.0 * ca4 * v - 6.0 * v2
            + 208.0 * ca2 * v2
            - 474.0 * ca4 * v2
            + 8.0 * ca3cf * v2
            + 28.0 * v3
            - 376.0 * ca2 * v3
            + 828.0 * ca4 * v3
            - 40.0 * ca3cf * v3
            - 54.0 * v4
            + 452.0 * ca2 * v4
            - 974.0 * ca4 * v4
            - 4.0 * cacf * v4
            + 84.0 * ca3cf * v4
            + 56.0 * v5
            - 360.0 * ca2 * v5
            + 784.0 * ca4 * v5
            + 16.0 * cacf * v5
            - 96.0 * ca3cf * v5
            - 34.0 * v6
            + 184.0 * ca2 * v6
            - 422.0 * ca4 * v6
            - 24.0 * cacf * v6
            + 64.0 * ca3cf * v6
            + 12.0 * v7
            - 56.0 * ca2 * v7
            + 140.0 * ca4 * v7
            + 16.0 * cacf * v7
            - 24.0 * ca3cf * v7
            - 2.0 * v8
            + 8.0 * ca2 * v8
            - 22.0 * ca4 * v8
            - 4.0 * cacf * v8
            + 4.0 * ca3cf * v8
            + 16.0 * ca2 * w
            + 16.0 * ca4 * w
            - 96.0 * ca2 * v * w
            - 96.0 * ca4 * v * w
            - 4.0 * v2 * w
            + 254.0 * ca2 * v2 * w
            + 146.0 * ca4 * v2 * w
            - 8.0 * ca3cf * v2 * w
            + 22.0 * v3 * w
            - 390.0 * ca2 * v3 * w
            + 148.0 * ca4 * v3 * w
            + 40.0 * ca3cf * v3 * w
            - 75.0 * v4 * w
            + 420.0 * ca2 * v4 * w
            - 781.0 * ca4 * v4 * w
            + 4.0 * cacf * v4 * w
            - 92.0 * ca3cf * v4 * w
            + 158.0 * v5 * w
            - 384.0 * ca2 * v5 * w
            + 1178.0 * ca4 * v5 * w
            - 16.0 * cacf * v5 * w
            + 128.0 * ca3cf * v5 * w
            - 196.0 * v6 * w
            + 302.0 * ca2 * v6 * w
            - 990.0 * ca4 * v6 * w
            + 12.0 * cacf * v6 * w
            - 100.0 * ca3cf * v6 * w
            + 144.0 * v7 * w
            - 174.0 * ca2 * v7 * w
            + 538.0 * ca4 * v7 * w
            + 20.0 * cacf * v7 * w
            + 20.0 * ca3cf * v7 * w
            - 61.0 * v8 * w
            + 64.0 * ca2 * v8 * w
            - 199.0 * ca4 * v8 * w
            - 32.0 * cacf * v8 * w
            + 24.0 * ca3cf * v8 * w
            + 12.0 * v9 * w
            - 12.0 * ca2 * v9 * w
            + 40.0 * ca4 * v9 * w
            + 12.0 * cacf * v9 * w
            - 12.0 * ca3cf * v9 * w
            + 12.0 * ca2 * v2 * w2
            + 132.0 * ca4 * v2 * w2
            - 60.0 * ca2 * v3 * w2
            - 660.0 * ca4 * v3 * w2
            + 4.0 * v4 * w2
            + 70.0 * ca2 * v4 * w2
            + 1250.0 * ca4 * v4 * w2
            - 80.0 * ca3cf * v4 * w2
            - 16.0 * v5 * w2
            + 84.0 * ca2 * v5 * w2
            - 1044.0 * ca4 * v5 * w2
            + 320.0 * ca3cf * v5 * w2
            + 49.0 * v6 * w2
            - 238.0 * ca2 * v6 * w2
            + 101.0 * ca4 * v6 * w2
            + 24.0 * cacf * v6 * w2
            - 576.0 * ca3cf * v6 * w2
            - 93.0 * v7 * w2
            + 186.0 * ca2 * v7 * w2
            + 563.0 * ca4 * v7 * w2
            - 72.0 * cacf * v7 * w2
            + 608.0 * ca3cf * v7 * w2
            + 81.0 * v8 * w2
            - 64.0 * ca2 * v8 * w2
            - 525.0 * ca4 * v8 * w2
            + 60.0 * cacf * v8 * w2
            - 356.0 * ca3cf * v8 * w2
            - 23.0 * v9 * w2
            + 6.0 * ca2 * v9 * w2
            + 241.0 * ca4 * v9 * w2
            + 72.0 * ca3cf * v9 * w2
            - 2.0 * v10 * w2
            + 4.0 * ca2 * v10 * w2
            - 58.0 * ca4 * v10 * w2
            - 12.0 * cacf * v10 * w2
            + 12.0 * ca3cf * v10 * w2
            - 48.0 * ca2 * v2 * w3
            - 48.0 * ca4 * v2 * w3
            + 240.0 * ca2 * v3 * w3
            + 240.0 * ca4 * v3 * w3
            + 12.0 * v4 * w3
            - 438.0 * ca2 * v4 * w3
            - 234.0 * ca4 * v4 * w3
            + 24.0 * ca3cf * v4 * w3
            - 44.0 * v5 * w3
            + 300.0 * ca2 * v5 * w3
            - 496.0 * ca4 * v5 * w3
            - 96.0 * ca3cf * v5 * w3
            - 47.0 * v6 * w3
            - 28.0 * ca2 * v6 * w3
            + 1423.0 * ca4 * v6 * w3
            - 12.0 * cacf * v6 * w3
            + 100.0 * ca3cf * v6 * w3
            + 295.0 * v7 * w3
            + 32.0 * ca2 * v7 * w3
            - 1515.0 * ca4 * v7 * w3
            + 36.0 * cacf * v7 * w3
            + 36.0 * ca3cf * v7 * w3
            - 368.0 * v8 * w3
            - 114.0 * ca2 * v8 * w3
            + 790.0 * ca4 * v8 * w3
            - 232.0 * ca3cf * v8 * w3
            + 189.0 * v9 * w3
            + 104.0 * ca2 * v9 * w3
            - 169.0 * ca4 * v9 * w3
            - 60.0 * cacf * v9 * w3
            + 292.0 * ca3cf * v9 * w3
            - 45.0 * v10 * w3
            - 56.0 * ca2 * v10 * w3
            - 23.0 * ca4 * v10 * w3
            + 32.0 * cacf * v10 * w3
            - 120.0 * ca3cf * v10 * w3
            + 8.0 * v11 * w3
            + 8.0 * ca2 * v11 * w3
            + 32.0 * ca4 * v11 * w3
            + 4.0 * cacf * v11 * w3
            - 4.0 * ca3cf * v11 * w3
            - 60.0 * ca2 * v4 * w4
            - 180.0 * ca4 * v4 * w4
            + 240.0 * ca2 * v5 * w4
            + 720.0 * ca4 * v5 * w4
            + 102.0 * v6 * w4
            - 244.0 * ca2 * v6 * w4
            - 1018.0 * ca4 * v6 * w4
            + 136.0 * ca3cf * v6 * w4
            - 320.0 * v7 * w4
            - 88.0 * ca2 * v7 * w4
            + 528.0 * ca4 * v7 * w4
            - 408.0 * ca3cf * v7 * w4
            + 408.0 * v8 * w4
            + 250.0 * ca2 * v8 * w4
            + 186.0 * ca4 * v8 * w4
            - 36.0 * cacf * v8 * w4
            + 540.0 * ca3cf * v8 * w4
            - 272.0 * v9 * w4
            - 160.0 * ca2 * v9 * w4
            - 336.0 * ca4 * v9 * w4
            + 72.0 * cacf * v9 * w4
            - 400.0 * ca3cf * v9 * w4
            + 109.0 * v10 * w4
            + 28.0 * ca2 * v10 * w4
            + 155.0 * ca4 * v10 * w4
            - 20.0 * cacf * v10 * w4
            + 84.0 * ca3cf * v10 * w4
            - 27.0 * v11 * w4
            + 50.0 * ca2 * v11 * w4
            - 71.0 * ca4 * v11 * w4
            - 16.0 * cacf * v11 * w4
            + 48.0 * ca3cf * v11 * w4
            - 16.0 * ca2 * v12 * w4
            + 48.0 * ca2 * v4 * w5
            + 48.0 * ca4 * v4 * w5
            - 192.0 * ca2 * v5 * w5
            - 192.0 * ca4 * v5 * w5
            - 12.0 * v6 * w5
            + 210.0 * ca2 * v6 * w5
            + 126.0 * ca4 * v6 * w5
            - 24.0 * ca3cf * v6 * w5
            + 42.0 * v7 * w5
            + 34.0 * ca2 * v7 * w5
            + 296.0 * ca4 * v7 * w5
            + 72.0 * ca3cf * v7 * w5
            - 101.0 * v8 * w5
            - 92.0 * ca2 * v8 * w5
            - 515.0 * ca4 * v8 * w5
            + 12.0 * cacf * v8 * w5
            - 36.0 * ca3cf * v8 * w5
            + 138.0 * v9 * w5
            - 54.0 * ca2 * v9 * w5
            + 264.0 * ca4 * v9 * w5
            - 24.0 * cacf * v9 * w5
            - 48.0 * ca3cf * v9 * w5
            - 106.0 * v10 * w5
            + 228.0 * ca2 * v10 * w5
            - 74.0 * ca4 * v10 * w5
            - 12.0 * cacf * v10 * w5
            + 140.0 * ca3cf * v10 * w5
            + 39.0 * v11 * w5
            - 246.0 * ca2 * v11 * w5
            + 111.0 * ca4 * v11 * w5
            + 24.0 * cacf * v11 * w5
            - 104.0 * ca3cf * v11 * w5
            + 64.0 * ca2 * v12 * w5
            - 16.0 * ca4 * v12 * w5
            + 36.0 * ca2 * v6 * w6
            + 76.0 * ca4 * v6 * w6
            - 108.0 * ca2 * v7 * w6
            - 228.0 * ca4 * v7 * w6
            + 12.0 * v8 * w6
            + 22.0 * ca2 * v8 * w6
            + 202.0 * ca4 * v8 * w6
            - 64.0 * ca3cf * v8 * w6
            - 36.0 * v9 * w6
            + 140.0 * ca2 * v9 * w6
            - 16.0 * ca4 * v9 * w6
            + 128.0 * ca3cf * v9 * w6
            + 53.0 * v10 * w6
            - 344.0 * ca2 * v10 * w6
            + 47.0 * ca4 * v10 * w6
            + 16.0 * cacf * v10 * w6
            - 144.0 * ca3cf * v10 * w6
            - 29.0 * v11 * w6
            + 358.0 * ca2 * v11 * w6
            - 185.0 * ca4 * v11 * w6
            - 16.0 * cacf * v11 * w6
            + 80.0 * ca3cf * v11 * w6
            - 96.0 * ca2 * v12 * w6
            + 48.0 * ca4 * v12 * w6
            - 16.0 * ca2 * v6 * w7
            - 16.0 * ca4 * v6 * w7
            + 48.0 * ca2 * v7 * w7
            + 48.0 * ca4 * v7 * w7
            + 4.0 * v8 * w7
            - 26.0 * ca2 * v8 * w7
            - 38.0 * ca4 * v8 * w7
            + 8.0 * ca3cf * v8 * w7
            - 4.0 * v9 * w7
            - 32.0 * ca2 * v9 * w7
            - 4.0 * ca4 * v9 * w7
            - 16.0 * ca3cf * v9 * w7
            - 9.0 * v10 * w7
            + 164.0 * ca2 * v10 * w7
            - 71.0 * ca4 * v10 * w7
            - 4.0 * cacf * v10 * w7
            + 28.0 * ca3cf * v10 * w7
            + 9.0 * v11 * w7
            - 226.0 * ca2 * v11 * w7
            + 169.0 * ca4 * v11 * w7
            + 4.0 * cacf * v11 * w7
            - 20.0 * ca3cf * v11 * w7
            + 64.0 * ca2 * v12 * w7
            - 48.0 * ca4 * v12 * w7
            - 24.0 * ca2 * v10 * w8
            + 24.0 * ca4 * v10 * w8
            + 64.0 * ca2 * v11 * w8
            - 64.0 * ca4 * v11 * w8
            - 16.0 * ca2 * v12 * w8
            + 16.0 * ca4 * v12 * w8
            - 8.0 * ca2 * v11 * w9
            + 8.0 * ca4 * v11 * w9))
        / (ca * (1.0 - v).powi(2) * v3 * w * (1.0 - v * w).powi(3) * (1.0 - v + v * w).powi(3));

    let part8 = (2.0
        * cf
        * l1w
        * (64.0 * ca4 - 384.0 * ca4 * v + 1120.0 * ca4 * v2 - 2080.0 * ca4 * v3
            + 2688.0 * ca4 * v4
            - 2496.0 * ca4 * v5
            + 1664.0 * ca4 * v6
            - 768.0 * ca4 * v7
            + 224.0 * ca4 * v8
            - 32.0 * ca4 * v9
            - 4.0 * ca2 * w
            + 52.0 * ca4 * w
            + 24.0 * ca2 * v * w
            - 312.0 * ca4 * v * w
            + 4.0 * v2 * w
            - 70.0 * ca2 * v2 * w
            + 864.0 * ca4 * v2 * w
            - 20.0 * v3 * w
            + 130.0 * ca2 * v3 * w
            - 1460.0 * ca4 * v3 * w
            + 46.0 * v4 * w
            - 170.0 * ca2 * v4 * w
            + 1906.0 * ca4 * v4 * w
            - 64.0 * v5 * w
            + 164.0 * ca2 * v5 * w
            - 2296.0 * ca4 * v5 * w
            + 56.0 * v6 * w
            - 114.0 * ca2 * v6 * w
            + 2524.0 * ca4 * v6 * w
            - 28.0 * v7 * w
            + 50.0 * ca2 * v7 * w
            - 2236.0 * ca4 * v7 * w
            + 6.0 * v8 * w
            - 10.0 * ca2 * v8 * w
            + 1390.0 * ca4 * v8 * w
            - 528.0 * ca4 * v9 * w
            + 96.0 * ca4 * v10 * w
            + 4.0 * ca2 * w2
            + 4.0 * ca4 * w2
            - 24.0 * ca2 * v * w2
            - 24.0 * ca4 * v * w2
            - 2.0 * v2 * w2
            + 54.0 * ca2 * v2 * w2
            + 32.0 * ca4 * v2 * w2
            + 10.0 * v3 * w2
            - 50.0 * ca2 * v3 * w2
            + 60.0 * ca4 * v3 * w2
            - 20.0 * v4 * w2
            - 44.0 * ca2 * v4 * w2
            - 598.0 * ca4 * v4 * w2
            + 20.0 * v5 * w2
            + 212.0 * ca2 * v5 * w2
            + 1768.0 * ca4 * v5 * w2
            + 6.0 * v6 * w2
            - 336.0 * ca2 * v6 * w2
            - 2752.0 * ca4 * v6 * w2
            - 46.0 * v7 * w2
            + 320.0 * ca2 * v7 * w2
            + 2584.0 * ca4 * v7 * w2
            + 48.0 * v8 * w2
            - 182.0 * ca2 * v8 * w2
            - 1262.0 * ca4 * v8 * w2
            - 16.0 * v9 * w2
            + 46.0 * ca2 * v9 * w2
            + 28.0 * ca4 * v9 * w2
            + 256.0 * ca4 * v10 * w2
            - 96.0 * ca4 * v11 * w2
            + 24.0 * ca2 * v2 * w3
            - 144.0 * ca4 * v2 * w3
            - 120.0 * ca2 * v3 * w3
            + 720.0 * ca4 * v3 * w3
            - 8.0 * v4 * w3
            + 312.0 * ca2 * v4 * w3
            - 1462.0 * ca4 * v4 * w3
            + 32.0 * v5 * w3
            - 528.0 * ca2 * v5 * w3
            + 1528.0 * ca4 * v5 * w3
            - 76.0 * v6 * w3
            + 550.0 * ca2 * v6 * w3
            - 1118.0 * ca4 * v6 * w3
            + 116.0 * v7 * w3
            - 306.0 * ca2 * v7 * w3
            + 1030.0 * ca4 * v7 * w3
            - 76.0 * v8 * w3
            - 4.0 * ca2 * v8 * w3
            - 1592.0 * ca4 * v8 * w3
            - 4.0 * v9 * w3
            + 142.0 * ca2 * v9 * w3
            + 1810.0 * ca4 * v9 * w3
            + 16.0 * v10 * w3
            - 70.0 * ca2 * v10 * w3
            - 948.0 * ca4 * v10 * w3
            + 176.0 * ca4 * v11 * w3
            + 32.0 * ca4 * v12 * w3
            - 12.0 * ca2 * v2 * w4
            - 12.0 * ca4 * v2 * w4
            + 60.0 * ca2 * v3 * w4
            + 60.0 * ca4 * v3 * w4
            + 6.0 * v4 * w4
            - 102.0 * ca2 * v4 * w4
            - 252.0 * ca4 * v4 * w4
            - 24.0 * v5 * w4
            + 48.0 * ca2 * v5 * w4
            + 648.0 * ca4 * v5 * w4
            + 43.0 * v6 * w4
            + 202.0 * ca2 * v6 * w4
            - 545.0 * ca4 * v6 * w4
            - 45.0 * v7 * w4
            - 522.0 * ca2 * v7 * w4
            - 381.0 * ca4 * v7 * w4
            - 29.0 * v8 * w4
            + 584.0 * ca2 * v8 * w4
            + 1583.0 * ca4 * v8 * w4
            + 105.0 * v9 * w4
            - 362.0 * ca2 * v9 * w4
            - 1895.0 * ca4 * v9 * w4
            - 52.0 * v10 * w4
            + 54.0 * ca2 * v10 * w4
            + 722.0 * ca4 * v10 * w4
            - 4.0 * v11 * w4
            + 50.0 * ca2 * v11 * w4
            + 72.0 * ca4 * v11 * w4
            - 128.0 * ca4 * v12 * w4
            - 36.0 * ca2 * v4 * w5
            + 132.0 * ca4 * v4 * w5
            + 144.0 * ca2 * v5 * w5
            - 528.0 * ca4 * v5 * w5
            + 12.0 * v6 * w5
            - 330.0 * ca2 * v6 * w5
            + 688.0 * ca4 * v6 * w5
            - 36.0 * v7 * w5
            + 486.0 * ca2 * v7 * w5
            - 216.0 * ca4 * v7 * w5
            + 116.0 * v8 * w5
            - 320.0 * ca2 * v8 * w5
            - 518.0 * ca4 * v8 * w5
            - 172.0 * v9 * w5
            - 2.0 * ca2 * v9 * w5
            + 780.0 * ca4 * v9 * w5
            + 62.0 * v10 * w5
            + 186.0 * ca2 * v10 * w5
            + 102.0 * ca4 * v10 * w5
            + 18.0 * v11 * w5
            - 128.0 * ca2 * v11 * w5
            - 440.0 * ca4 * v11 * w5
            - 16.0 * ca2 * v12 * w5
            + 224.0 * ca4 * v12 * w5
            + 12.0 * ca2 * v4 * w6
            + 12.0 * ca4 * v4 * w6
            - 48.0 * ca2 * v5 * w6
            - 48.0 * ca4 * v5 * w6
            - 6.0 * v6 * w6
            + 66.0 * ca2 * v6 * w6
            + 176.0 * ca4 * v6 * w6
            + 18.0 * v7 * w6
            - 30.0 * ca2 * v7 * w6
            - 360.0 * ca4 * v7 * w6
            - 72.0 * v8 * w6
            - 172.0 * ca2 * v8 * w6
            + 382.0 * ca4 * v8 * w6
            + 114.0 * v9 * w6
            + 338.0 * ca2 * v9 * w6
            - 220.0 * ca4 * v9 * w6
            - 19.0 * v10 * w6
            - 244.0 * ca2 * v10 * w6
            - 465.0 * ca4 * v10 * w6
            - 35.0 * v11 * w6
            + 78.0 * ca2 * v11 * w6
            + 523.0 * ca4 * v11 * w6
            + 64.0 * ca2 * v12 * w6
            - 240.0 * ca4 * v12 * w6
            + 16.0 * ca2 * v6 * w7
            - 40.0 * ca4 * v6 * w7
            - 48.0 * ca2 * v7 * w7
            + 120.0 * ca4 * v7 * w7
            + 12.0 * v8 * w7
            + 128.0 * ca2 * v8 * w7
            - 110.0 * ca4 * v8 * w7
            - 24.0 * v9 * w7
            - 176.0 * ca2 * v9 * w7
            + 20.0 * ca4 * v9 * w7
            - 24.0 * v10 * w7
            + 28.0 * ca2 * v10 * w7
            + 348.0 * ca4 * v10 * w7
            + 36.0 * v11 * w7
            + 52.0 * ca2 * v11 * w7
            - 338.0 * ca4 * v11 * w7
            - 104.0 * ca2 * v12 * w7
            + 184.0 * ca4 * v12 * w7
            - 4.0 * ca2 * v6 * w8
            - 4.0 * ca4 * v6 * w8
            + 12.0 * ca2 * v7 * w8
            + 12.0 * ca4 * v7 * w8
            + 2.0 * v8 * w8
            - 18.0 * ca2 * v8 * w8
            - 20.0 * ca4 * v8 * w8
            - 4.0 * v9 * w8
            + 16.0 * ca2 * v9 * w8
            + 20.0 * ca4 * v9 * w8
            + 21.0 * v10 * w8
            + 70.0 * ca2 * v10 * w8
            - 131.0 * ca4 * v10 * w8
            - 19.0 * v11 * w8
            - 76.0 * ca2 * v11 * w8
            + 123.0 * ca4 * v11 * w8
            + 88.0 * ca2 * v12 * w8
            - 104.0 * ca4 * v12 * w8
            - 4.0 * v10 * w9
            - 24.0 * ca2 * v10 * w9
            + 20.0 * ca4 * v10 * w9
            + 4.0 * v11 * w9
            + 24.0 * ca2 * v11 * w9
            - 20.0 * ca4 * v11 * w9
            - 40.0 * ca2 * v12 * w9
            + 40.0 * ca4 * v12 * w9
            + 8.0 * ca2 * v12 * w10
            - 8.0 * ca4 * v12 * w10))
        / (ca * (1.0 - v).powi(2) * v3 * w2 * (1.0 - v * w).powi(3) * (1.0 - v + v * w).powi(3));

    let part9 = (2.0
        * cf
        * lv
        * (80.0 * ca4 - 480.0 * ca4 * v + 1392.0 * ca4 * v2 - 2560.0 * ca4 * v3
            + 3264.0 * ca4 * v4
            - 2976.0 * ca4 * v5
            + 1936.0 * ca4 * v6
            - 864.0 * ca4 * v7
            + 240.0 * ca4 * v8
            - 32.0 * ca4 * v9
            - 4.0 * ca2 * w
            + 52.0 * ca4 * w
            + 24.0 * ca2 * v * w
            - 312.0 * ca4 * v * w
            + 4.0 * v2 * w
            - 70.0 * ca2 * v2 * w
            + 866.0 * ca4 * v2 * w
            - 20.0 * v3 * w
            + 130.0 * ca2 * v3 * w
            - 1470.0 * ca4 * v3 * w
            + 48.0 * v4 * w
            - 172.0 * ca2 * v4 * w
            + 1948.0 * ca4 * v4 * w
            - 72.0 * v5 * w
            + 172.0 * ca2 * v5 * w
            - 2404.0 * ca4 * v5 * w
            + 68.0 * v6 * w
            - 126.0 * ca2 * v6 * w
            + 2698.0 * ca4 * v6 * w
            - 36.0 * v7 * w
            + 58.0 * ca2 * v7 * w
            - 2422.0 * ca4 * v7 * w
            + 8.0 * v8 * w
            - 12.0 * ca2 * v8 * w
            + 1508.0 * ca4 * v8 * w
            - 560.0 * ca4 * v9 * w
            + 96.0 * ca4 * v10 * w
            + 4.0 * ca2 * w2
            + 4.0 * ca4 * w2
            - 24.0 * ca2 * v * w2
            - 24.0 * ca4 * v * w2
            - 2.0 * v2 * w2
            + 54.0 * ca2 * v2 * w2
            - 4.0 * ca4 * v2 * w2
            + 10.0 * v3 * w2
            - 50.0 * ca2 * v3 * w2
            + 240.0 * ca4 * v3 * w2
            - 24.0 * v4 * w2
            - 36.0 * ca2 * v4 * w2
            - 1060.0 * ca4 * v4 * w2
            + 36.0 * v5 * w2
            + 180.0 * ca2 * v5 * w2
            + 2536.0 * ca4 * v5 * w2
            - 12.0 * v6 * w2
            - 290.0 * ca2 * v6 * w2
            - 3686.0 * ca4 * v6 * w2
            - 48.0 * v7 * w2
            + 294.0 * ca2 * v7 * w2
            + 3454.0 * ca4 * v7 * w2
            + 62.0 * v8 * w2
            - 180.0 * ca2 * v8 * w2
            - 1790.0 * ca4 * v8 * w2
            - 22.0 * v9 * w2
            + 48.0 * ca2 * v9 * w2
            + 170.0 * ca4 * v9 * w2
            + 256.0 * ca4 * v10 * w2
            - 96.0 * ca4 * v11 * w2
            + 24.0 * ca2 * v2 * w3
            - 144.0 * ca4 * v2 * w3
            - 120.0 * ca2 * v3 * w3
            + 720.0 * ca4 * v3 * w3
            - 4.0 * v4 * w3
            + 296.0 * ca2 * v4 * w3
            - 1436.0 * ca4 * v4 * w3
            + 16.0 * v5 * w3
            - 464.0 * ca2 * v5 * w3
            + 1424.0 * ca4 * v5 * w3
            - 70.0 * v6 * w3
            + 468.0 * ca2 * v6 * w3
            - 874.0 * ca4 * v6 * w3
            + 154.0 * v7 * w3
            - 284.0 * ca2 * v7 * w3
            + 662.0 * ca4 * v7 * w3
            - 120.0 * v8 * w3
            + 24.0 * ca2 * v8 * w3
            - 1460.0 * ca4 * v8 * w3
            + 2.0 * v9 * w3
            + 124.0 * ca2 * v9 * w3
            + 2038.0 * ca4 * v9 * w3
            + 22.0 * v10 * w3
            - 68.0 * ca2 * v10 * w3
            - 1138.0 * ca4 * v10 * w3
            + 208.0 * ca4 * v11 * w3
            + 32.0 * ca4 * v12 * w3
            - 12.0 * ca2 * v2 * w4
            - 12.0 * ca4 * v2 * w4
            + 60.0 * ca2 * v3 * w4
            + 60.0 * ca4 * v3 * w4
            + 6.0 * v4 * w4
            - 102.0 * ca2 * v4 * w4
            - 240.0 * ca4 * v4 * w4
            - 24.0 * v5 * w4
            + 48.0 * ca2 * v5 * w4
            + 600.0 * ca4 * v5 * w4
            + 67.0 * v6 * w4
            + 154.0 * ca2 * v6 * w4
            - 437.0 * ca4 * v6 * w4
            - 117.0 * v7 * w4
            - 378.0 * ca2 * v7 * w4
            - 537.0 * ca4 * v7 * w4
            + 19.0 * v8 * w4
            + 436.0 * ca2 * v8 * w4
            + 2073.0 * ca4 * v8 * w4
            + 129.0 * v9 * w4
            - 306.0 * ca2 * v9 * w4
            - 2671.0 * ca4 * v9 * w4
            - 74.0 * v10 * w4
            + 52.0 * ca2 * v10 * w4
            + 1130.0 * ca4 * v10 * w4
            - 6.0 * v11 * w4
            + 48.0 * ca2 * v11 * w4
            + 34.0 * ca4 * v11 * w4
            - 144.0 * ca4 * v12 * w4
            - 36.0 * ca2 * v4 * w5
            + 132.0 * ca4 * v4 * w5
            + 144.0 * ca2 * v5 * w5
            - 528.0 * ca4 * v5 * w5
            - 298.0 * ca2 * v6 * w5
            + 642.0 * ca4 * v6 * w5
            + 390.0 * ca2 * v7 * w5
            - 78.0 * ca4 * v7 * w5
            + 122.0 * v8 * w5
            - 246.0 * ca2 * v8 * w5
            - 972.0 * ca4 * v8 * w5
            - 244.0 * v9 * w5
            + 10.0 * ca2 * v9 * w5
            + 1458.0 * ca4 * v9 * w5
            + 94.0 * v10 * w5
            + 154.0 * ca2 * v10 * w5
            - 124.0 * ca4 * v10 * w5
            + 28.0 * v11 * w5
            - 118.0 * ca2 * v11 * w5
            - 530.0 * ca4 * v11 * w5
            - 16.0 * ca2 * v12 * w5
            + 288.0 * ca4 * v12 * w5
            + 12.0 * ca2 * v4 * w6
            + 12.0 * ca4 * v4 * w6
            - 48.0 * ca2 * v5 * w6
            - 48.0 * ca4 * v5 * w6
            - 6.0 * v6 * w6
            + 66.0 * ca2 * v6 * w6
            + 196.0 * ca4 * v6 * w6
            + 18.0 * v7 * w6
            - 30.0 * ca2 * v7 * w6
            - 420.0 * ca4 * v7 * w6
            - 108.0 * v8 * w6
            - 132.0 * ca2 * v8 * w6
            + 552.0 * ca4 * v8 * w6
            + 186.0 * v9 * w6
            + 258.0 * ca2 * v9 * w6
            - 460.0 * ca4 * v9 * w6
            - 33.0 * v10 * w6
            - 190.0 * ca2 * v10 * w6
            - 573.0 * ca4 * v10 * w6
            - 57.0 * v11 * w6
            + 64.0 * ca2 * v11 * w6
            + 741.0 * ca4 * v11 * w6
            + 64.0 * ca2 * v12 * w6
            - 344.0 * ca4 * v12 * w6
            + 16.0 * ca2 * v6 * w7
            - 40.0 * ca4 * v6 * w7
            - 48.0 * ca2 * v7 * w7
            + 120.0 * ca4 * v7 * w7
            + 24.0 * v8 * w7
            + 112.0 * ca2 * v8 * w7
            - 104.0 * ca4 * v8 * w7
            - 48.0 * v9 * w7
            - 144.0 * ca2 * v9 * w7
            + 8.0 * ca4 * v9 * w7
            - 38.0 * v10 * w7
            + 6.0 * ca2 * v10 * w7
            + 540.0 * ca4 * v10 * w7
            + 62.0 * v11 * w7
            + 58.0 * ca2 * v11 * w7
            - 524.0 * ca4 * v11 * w7
            - 104.0 * ca2 * v12 * w7
            + 272.0 * ca4 * v12 * w7
            - 4.0 * ca2 * v6 * w8
            - 4.0 * ca4 * v6 * w8
            + 12.0 * ca2 * v7 * w8
            + 12.0 * ca4 * v7 * w8
            + 2.0 * v8 * w8
            - 18.0 * ca2 * v8 * w8
            - 32.0 * ca4 * v8 * w8
            - 4.0 * v9 * w8
            + 16.0 * ca2 * v9 * w8
            + 44.0 * ca4 * v9 * w8
            + 37.0 * v10 * w8
            + 70.0 * ca2 * v10 * w8
            - 219.0 * ca4 * v10 * w8
            - 35.0 * v11 * w8
            - 76.0 * ca2 * v11 * w8
            + 199.0 * ca4 * v11 * w8
            + 88.0 * ca2 * v12 * w8
            - 144.0 * ca4 * v12 * w8
            - 8.0 * v10 * w9
            - 24.0 * ca2 * v10 * w9
            + 32.0 * ca4 * v10 * w9
            + 8.0 * v11 * w9
            + 24.0 * ca2 * v11 * w9
            - 32.0 * ca4 * v11 * w9
            - 40.0 * ca2 * v12 * w9
            + 48.0 * ca4 * v12 * w9
            + 8.0 * ca2 * v12 * w10
            - 8.0 * ca4 * v12 * w10))
        / (ca * (1.0 - v).powi(2) * v3 * w2 * (1.0 - v * w).powi(3) * (1.0 - v + v * w).powi(3));

    part1 + part2 + part3 + part4 + part5 + part6 + part7 + part8 + part9
}

/// `STRUV15(W,V,X3,S)`. Uses `v**13`, `v**14`, `w**12`, `w**13` beyond the
/// `PREV`/`PREW` common range (`V2..V12`, `W2..W12`); Fortran computes
/// these via inline exponentiation rather than a precomputed common
/// variable, so we do the same here with local `.powi()` calls.
#[must_use]
pub fn struv15(w: f64, v: f64, _x3: f64, _s: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let ca3 = ctx.ca.powi(3);
    let nf = ctx.nf;
    let (v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12) = (
        pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8, pre.v9, pre.v10, pre.v11, pre.v12,
    );
    let v13 = v.powi(13);
    let v14 = v.powi(14);
    let (w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12) = (
        pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8, pre.w9, pre.w10, pre.w11, pre.w12,
    );
    let w13 = w.powi(13);
    let (l1v, lw, lvw, l1vw, lms, l1w, lv, lmss) = (
        pre.l1v, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.l1w, pre.lv, pre.lmss,
    );

    let part1 = (-256.0
        * ca3
        * lmss
        * nf
        * (1.0 - 2.0 * v + v2 + v2 * w2)
        * (1.0 + v2 - 2.0 * v2 * w + v2 * w2)
        * (4.0 - 8.0 * v + 4.0 * v2 - v * w + v2 * w + 4.0 * v2 * w2))
        / (243.0 * (1.0 - v) * v * w * (1.0 - v + v * w).powi(4));

    let part2 = (32.0
        * ca3
        * lvw
        * nf
        * (81.0 - 179.0 * v + 115.0 * v2 - 34.0 * v3 - 145.0 * v * w + 130.0 * v2 * w
            - 13.0 * v3 * w
            + 79.0 * v2 * w2
            - 51.0 * v3 * w2
            + 17.0 * v3 * w3))
        / (243.0 * (1.0 - v) * v2 * w)
        - (32.0
            * ca3
            * l1v
            * nf
            * (18.0 - 45.0 * v + 54.0 * v2 - 44.0 * v3 + 17.0 * v4 + 16.0 * v3 * w
                - 17.0 * v5 * w
                + 20.0 * v2 * w2
                - 46.0 * v3 * w2
                + 14.0 * v4 * w2
                + 24.0 * v5 * w2
                + 12.0 * v4 * w3
                - 24.0 * v5 * w3
                + 2.0 * v4 * w4
                - v5 * w4))
            / (243.0 * (1.0 - v) * v2 * w * (1.0 - v * w) * (1.0 - v + v * w));

    let part3 = -(32.0
        * ca3
        * lw
        * nf
        * (88.0 - 176.0 * v + 104.0 * v2 - 16.0 * v3 + 13.0 * w - 26.0 * v * w + 82.0 * v2 * w
            - 69.0 * v3 * w
            + 18.0 * v4 * w
            - 20.0 * w2
            + 40.0 * v * w2
            - 164.0 * v2 * w2
            + 144.0 * v3 * w2
            - 45.0 * v4 * w2
            - 2.0 * v2 * w3
            + 2.0 * v3 * w3
            - 10.0 * v4 * w3
            - 20.0 * v2 * w4
            + 20.0 * v3 * w4
            + 20.0 * v4 * w4
            + 17.0 * v4 * w5))
        / (243.0 * (1.0 - v) * v * (1.0 - w) * w * (1.0 - v * w) * (1.0 - v + v * w));

    let part4 = (128.0
        * ca3
        * lvw
        * (4.0 - 8.0 * v + 12.0 * v2 - 8.0 * v3 + 4.0 * v4 - 5.0 * v * w - 4.0 * v2 * w
            + 6.0 * v3 * w
            - 11.0 * v4 * w
            + 2.0 * v5 * w
            + 18.0 * v2 * w2
            - 20.0 * v3 * w2
            + 35.0 * v4 * w2
            - 13.0 * v5 * w2
            + 4.0 * v6 * w2
            - 8.0 * v3 * w3
            - 21.0 * v4 * w3
            + 9.0 * v5 * w3
            - 8.0 * v6 * w3
            + 19.0 * v4 * w4
            - 7.0 * v5 * w4
            + 12.0 * v6 * w4
            - 4.0 * v5 * w5
            - 8.0 * v6 * w5
            + 4.0 * v6 * w6))
        / ((1.0 - v).powi(2) * v3 * w2);

    let part5 = -(128.0
        * ca3
        * l1v
        * (2.0 - 8.0 * v + 16.0 * v2 - 20.0 * v3 + 16.0 * v4 - 8.0 * v5 + 2.0 * v6 + 2.0 * w
            - 8.0 * v * w
            + 11.0 * v2 * w
            - 7.0 * v3 * w
            + 2.0 * v4 * w
            - v5 * w
            + 3.0 * v6 * w
            - 2.0 * v7 * w
            + 2.0 * v3 * w2
            - 5.0 * v4 * w2
            + 4.0 * v5 * w2
            - 4.0 * v6 * w2
            - v7 * w2
            + 2.0 * v8 * w2
            - 2.0 * v2 * w3
            + 8.0 * v3 * w3
            - 19.0 * v4 * w3
            + 25.0 * v5 * w3
            - 16.0 * v6 * w3
            + 14.0 * v7 * w3
            - 6.0 * v8 * w3
            + v5 * w4
            - 3.0 * v6 * w4
            - 10.0 * v7 * w4
            + 6.0 * v8 * w4
            - 2.0 * v5 * w5
            + 3.0 * v6 * w5
            + 7.0 * v7 * w5
            - 4.0 * v8 * w5
            - 4.0 * v7 * w6
            + 2.0 * v8 * w6))
        / ((1.0 - v).powi(2) * v3 * w2 * (1.0 - v * w) * (1.0 - v + v * w));

    let part6 = -(256.0
        * ca3
        * lms
        * nf
        * (18.0 - 36.0 * v + 39.0 * v2 - 25.0 * v3 + 12.0 * v4 - 36.0 * w + 74.0 * v2 * w
            - 105.0 * v3 * w
            + 50.0 * v4 * w
            - 14.0 * v5 * w
            + 36.0 * w2
            + 72.0 * v * w2
            - 110.0 * v2 * w2
            + 30.0 * v3 * w2
            + 94.0 * v4 * w2
            - 88.0 * v5 * w2
            + 40.0 * v6 * w2
            - 144.0 * v * w3
            + 72.0 * v2 * w3
            + 80.0 * v3 * w3
            - 140.0 * v4 * w3
            + 34.0 * v5 * w3
            + 2.0 * v6 * w3
            - 14.0 * v7 * w3
            + 216.0 * v2 * w4
            - 288.0 * v3 * w4
            + 150.0 * v4 * w4
            + 40.0 * v5 * w4
            - 41.0 * v6 * w4
            + 5.0 * v7 * w4
            + 12.0 * v8 * w4
            - 144.0 * v3 * w5
            + 252.0 * v4 * w5
            - 208.0 * v5 * w5
            + 66.0 * v6 * w5
            + 3.0 * v7 * w5
            - 16.0 * v8 * w5
            + 36.0 * v4 * w6
            - 72.0 * v5 * w6
            + 70.0 * v6 * w6
            - 34.0 * v7 * w6
            + 12.0 * v8 * w6))
        / (243.0 * (1.0 - v) * v3 * w * (1.0 - v * w).powi(4));

    let part7 = (32.0
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

    let part8 = -(128.0
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

    let part9 = (128.0
        * ca3
        * lw
        * (6.0 - 24.0 * v + 52.0 * v2 - 72.0 * v3 + 68.0 * v4 - 44.0 * v5 + 18.0 * v6
            - 4.0 * v7
            - 2.0 * w
            + 8.0 * v * w
            - 32.0 * v2 * w
            + 68.0 * v3 * w
            - 80.0 * v4 * w
            + 56.0 * v5 * w
            - 14.0 * v6 * w
            - 4.0 * v7 * w
            + 4.0 * v8 * w
            + 13.0 * v2 * w2
            - 39.0 * v3 * w2
            + 48.0 * v4 * w2
            - 31.0 * v5 * w2
            - 21.0 * v6 * w2
            + 30.0 * v7 * w2
            - 14.0 * v8 * w2
            - v2 * w3
            + 3.0 * v3 * w3
            + v4 * w3
            - 7.0 * v5 * w3
            + 65.0 * v6 * w3
            - 61.0 * v7 * w3
            + 26.0 * v8 * w3
            - 3.0 * v4 * w4
            + 6.0 * v5 * w4
            - 69.0 * v6 * w4
            + 66.0 * v7 * w4
            - 30.0 * v8 * w4
            + v4 * w5
            - 2.0 * v5 * w5
            + 43.0 * v6 * w5
            - 42.0 * v7 * w5
            + 22.0 * v8 * w5
            - 15.0 * v6 * w6
            + 15.0 * v7 * w6
            - 10.0 * v8 * w6
            + 2.0 * v6 * w7
            - 2.0 * v7 * w7
            + 2.0 * v8 * w7))
        / ((1.0 - v).powi(2) * v3 * (1.0 - w) * w2 * (1.0 - v * w) * (1.0 - v + v * w));

    let part10 = -(256.0
        * ca3
        * lms
        * (2.0 - 5.0 * v + 9.0 * v2 - 9.0 * v3 + 7.0 * v4 - 3.0 * v5 + v6 - 8.0 * v * w
            + 15.0 * v2 * w
            - 28.0 * v3 * w
            + 27.0 * v4 * w
            - 24.0 * v5 * w
            + 11.0 * v6 * w
            - 4.0 * v7 * w
            + w2
            - 3.0 * v * w2
            + 18.0 * v2 * w2
            - 17.0 * v3 * w2
            + 34.0 * v4 * w2
            - 27.0 * v5 * w2
            + 30.0 * v6 * w2
            - 14.0 * v7 * w2
            + 6.0 * v8 * w2
            - w3
            - v * w3
            + 6.0 * v2 * w3
            - 25.0 * v3 * w3
            + 2.0 * v4 * w3
            - 15.0 * v5 * w3
            - 2.0 * v6 * w3
            - 13.0 * v7 * w3
            + 6.0 * v8 * w3
            - 4.0 * v9 * w3
            + 4.0 * v * w4
            - 6.0 * v2 * w4
            + 6.0 * v3 * w4
            + 10.0 * v4 * w4
            + 27.0 * v5 * w4
            - 3.0 * v6 * w4
            + 20.0 * v7 * w4
            + v9 * w4
            + v10 * w4
            - 6.0 * v2 * w5
            + 14.0 * v3 * w5
            - 24.0 * v4 * w5
            + 18.0 * v5 * w5
            - 53.0 * v6 * w5
            + 12.0 * v7 * w5
            - 18.0 * v8 * w5
            - v10 * w5
            + 4.0 * v3 * w6
            - 11.0 * v4 * w6
            + 21.0 * v5 * w6
            - 22.0 * v6 * w6
            + 47.0 * v7 * w6
            - 9.0 * v8 * w6
            + 9.0 * v9 * w6
            + v10 * w6
            - v4 * w7
            + 3.0 * v5 * w7
            - 6.0 * v6 * w7
            + 7.0 * v7 * w7
            - 21.0 * v8 * w7
            + 2.0 * v9 * w7
            - 3.0 * v10 * w7
            + 5.0 * v9 * w8
            + v10 * w8
            - v10 * w9))
        / ((1.0 - v).powi(2) * v3 * w2 * (1.0 - v * w).powi(4));

    let part11 = -(256.0
        * ca3
        * lmss
        * (1.0 - 8.0 * v + 30.0 * v2 - 70.0 * v3 + 113.0 * v4 - 132.0 * v5 + 113.0 * v6
            - 70.0 * v7
            + 30.0 * v8
            - 8.0 * v9
            + v10
            + w
            - 3.0 * v * w
            - 9.0 * v2 * w
            + 71.0 * v3 * w
            - 198.0 * v4 * w
            + 330.0 * v5 * w
            - 367.0 * v6 * w
            + 279.0 * v7 * w
            - 141.0 * v8 * w
            + 43.0 * v9 * w
            - 6.0 * v10 * w
            + 4.0 * v * w2
            - 18.0 * v2 * w2
            + 16.0 * v3 * w2
            + 76.0 * v4 * w2
            - 282.0 * v5 * w2
            + 471.0 * v6 * w2
            - 472.0 * v7 * w2
            + 294.0 * v8 * w2
            - 106.0 * v9 * w2
            + 17.0 * v10 * w2
            + 6.0 * v2 * w3
            - 26.0 * v3 * w3
            + 18.0 * v4 * w3
            + 114.0 * v5 * w3
            - 355.0 * v6 * w3
            + 497.0 * v7 * w3
            - 393.0 * v8 * w3
            + 171.0 * v9 * w3
            - 32.0 * v10 * w3
            + 4.0 * v3 * w4
            - 15.0 * v4 * w4
            - 18.0 * v5 * w4
            + 170.0 * v6 * w4
            - 357.0 * v7 * w4
            + 372.0 * v8 * w4
            - 201.0 * v9 * w4
            + 45.0 * v10 * w4
            + v4 * w5
            - 3.0 * v5 * w5
            - 39.0 * v6 * w5
            + 155.0 * v7 * w5
            - 234.0 * v8 * w5
            + 167.0 * v9 * w5
            - 47.0 * v10 * w5
            - 30.0 * v7 * w6
            + 87.0 * v8 * w6
            - 92.0 * v9 * w6
            + 35.0 * v10 * w6
            - 15.0 * v8 * w7
            + 31.0 * v9 * w7
            - 18.0 * v10 * w7
            - 5.0 * v9 * w8
            + 6.0 * v10 * w8
            - v10 * w9))
        / ((1.0 - v).powi(2) * v3 * w2 * (1.0 - v + v * w).powi(4));

    let part12 = (64.0
        * ca3
        * lv
        * nf
        * (72.0 - 432.0 * v + 1180.0 * v2 - 1940.0 * v3 + 2136.0 * v4 - 1656.0 * v5 + 908.0 * v6
            - 324.0 * v7
            + 56.0 * v8
            - 144.0 * w
            + 864.0 * v * w
            - 2008.0 * v2 * w
            + 2120.0 * v3 * w
            - 524.0 * v4 * w
            - 1120.0 * v5 * w
            + 1399.0 * v6 * w
            - 877.0 * v7 * w
            + 377.0 * v8 * w
            - 87.0 * v9 * w
            + 144.0 * w2
            - 864.0 * v * w2
            + 1432.0 * v2 * w2
            + 760.0 * v3 * w2
            - 5168.0 * v4 * w2
            + 6608.0 * v5 * w2
            - 3166.0 * v6 * w2
            - 934.0 * v7 * w2
            + 2011.0 * v8 * w2
            - 1028.0 * v9 * w2
            + 205.0 * v10 * w2
            + 1152.0 * v2 * w3
            - 5760.0 * v3 * w3
            + 10880.0 * v4 * w3
            - 8960.0 * v5 * w3
            - 162.0 * v6 * w3
            + 7654.0 * v7 * w3
            - 7149.0 * v8 * w3
            + 2608.0 * v9 * w3
            - 178.0 * v10 * w3
            - 85.0 * v11 * w3
            - 576.0 * v2 * w4
            + 2880.0 * v3 * w4
            - 3280.0 * v4 * w4
            - 4160.0 * v5 * w4
            + 14136.0 * v6 * w4
            - 15752.0 * v7 * w4
            + 8900.0 * v8 * w4
            - 2160.0 * v9 * w4
            + 60.0 * v10 * w4
            - 48.0 * v11 * w4
            + 55.0 * v12 * w4
            - 2592.0 * v4 * w5
            + 10368.0 * v5 * w5
            - 14832.0 * v6 * w5
            + 8208.0 * v7 * w5
            + 286.0 * v8 * w5
            - 2156.0 * v9 * w5
            + 57.0 * v10 * w5
            + 661.0 * v11 * w5
            - 239.0 * v12 * w5
            + 864.0 * v4 * w6
            - 3456.0 * v5 * w6
            + 2256.0 * v6 * w6
            + 5328.0 * v7 * w6
            - 9360.0 * v8 * w6
            + 5808.0 * v9 * w6
            + 82.0 * v10 * w6
            - 1522.0 * v11 * w6
            + 533.0 * v12 * w6
            + 2304.0 * v6 * w7
            - 6912.0 * v7 * w7
            + 7360.0 * v8 * w7
            - 3200.0 * v9 * w7
            - 1638.0 * v10 * w7
            + 2086.0 * v11 * w7
            - 707.0 * v12 * w7
            - 576.0 * v6 * w8
            + 1728.0 * v7 * w8
            - 760.0 * v8 * w8
            - 1360.0 * v9 * w8
            + 2604.0 * v10 * w8
            - 1636.0 * v11 * w8
            + 560.0 * v12 * w8
            - 720.0 * v8 * w9
            + 1440.0 * v9 * w9
            - 1400.0 * v10 * w9
            + 680.0 * v11 * w9
            - 266.0 * v12 * w9
            + 144.0 * v8 * w10
            - 288.0 * v9 * w10
            + 280.0 * v10 * w10
            - 136.0 * v11 * w10
            + 64.0 * v12 * w10))
        / (243.0 * (1.0 - v) * v3 * w * (1.0 - v * w).powi(4) * (1.0 - v + v * w).powi(4));

    let part13 = (32.0
        * ca3
        * l1w
        * nf
        * (144.0 - 864.0 * v + 2369.0 * v2 - 3925.0 * v3 + 4361.0 * v4 - 3398.0 * v5
            + 1855.0 * v6
            - 653.0 * v7
            + 111.0 * v8
            - 288.0 * w
            + 1728.0 * v * w
            - 3996.0 * v2 * w
            + 4140.0 * v3 * w
            - 781.0 * v4 * w
            - 2708.0 * v5 * w
            + 3298.0 * v6 * w
            - 2036.0 * v7 * w
            + 815.0 * v8 * w
            - 172.0 * v9 * w
            + 288.0 * w2
            - 1728.0 * v * w2
            + 2864.0 * v2 * w2
            + 1520.0 * v3 * w2
            - 10317.0 * v4 * w2
            + 13140.0 * v5 * w2
            - 6044.0 * v6 * w2
            - 2466.0 * v7 * w2
            + 4563.0 * v8 * w2
            - 2230.0 * v9 * w2
            + 410.0 * v10 * w2
            + 2304.0 * v2 * w3
            - 11520.0 * v3 * w3
            + 21720.0 * v4 * w3
            - 17760.0 * v5 * w3
            - 744.0 * v6 * w3
            + 16008.0 * v7 * w3
            - 14674.0 * v8 * w3
            + 4988.0 * v9 * w3
            - 150.0 * v10 * w3
            - 172.0 * v11 * w3
            - 1152.0 * v2 * w4
            + 5760.0 * v3 * w4
            - 6560.0 * v4 * w4
            - 8320.0 * v5 * w4
            + 28298.0 * v6 * w4
            - 31582.0 * v7 * w4
            + 17442.0 * v8 * w4
            - 3474.0 * v9 * w4
            - 207.0 * v10 * w4
            - 205.0 * v11 * w4
            + 111.0 * v12 * w4
            - 5184.0 * v4 * w5
            + 20736.0 * v5 * w5
            - 29664.0 * v6 * w5
            + 16416.0 * v7 * w5
            + 926.0 * v8 * w5
            - 5020.0 * v9 * w5
            + 118.0 * v10 * w5
            + 1672.0 * v11 * w5
            - 457.0 * v12 * w5
            + 1728.0 * v4 * w6
            - 6912.0 * v5 * w6
            + 4512.0 * v6 * w6
            + 10656.0 * v7 * w6
            - 18882.0 * v8 * w6
            + 11940.0 * v9 * w6
            + 452.0 * v10 * w6
            - 3494.0 * v11 * w6
            + 973.0 * v12 * w6
            + 4608.0 * v6 * w7
            - 13824.0 * v7 * w7
            + 14760.0 * v8 * w7
            - 6480.0 * v9 * w7
            - 3552.0 * v10 * w7
            + 4488.0 * v11 * w7
            - 1254.0 * v12 * w7
            - 1152.0 * v6 * w8
            + 3456.0 * v7 * w8
            - 1520.0 * v8 * w8
            - 2720.0 * v9 * w8
            + 5333.0 * v10 * w8
            - 3397.0 * v11 * w8
            + 973.0 * v12 * w8
            - 1440.0 * v8 * w9
            + 2880.0 * v9 * w9
            - 2820.0 * v10 * w9
            + 1380.0 * v11 * w9
            - 457.0 * v12 * w9
            + 288.0 * v8 * w10
            - 576.0 * v9 * w10
            + 560.0 * v10 * w10
            - 272.0 * v11 * w10
            + 111.0 * v12 * w10))
        / (243.0 * (1.0 - v) * v3 * w * (1.0 - v * w).powi(4) * (1.0 - v + v * w).powi(4));

    let part14 = -(64.0
        * ca3
        * nf
        * (126.0 - 882.0 * v + 2882.0 * v2 - 5842.0 * v3 + 8172.0 * v4 - 8236.0 * v5
            + 6002.0 * v6
            - 3042.0 * v7
            + 962.0 * v8
            - 142.0 * v9
            - 630.0 * w
            + 4410.0 * v * w
            - 13105.0 * v2 * w
            + 21368.0 * v3 * w
            - 20644.0 * v4 * w
            + 12127.0 * v5 * w
            - 4721.0 * v6 * w
            + 1654.0 * v7 * w
            - 444.0 * v8 * w
            - 87.0 * v9 * w
            + 72.0 * v10 * w
            + 684.0 * w2
            - 4788.0 * v * w2
            + 11890.0 * v2 * w2
            - 9096.0 * v3 * w2
            - 12922.0 * v4 * w2
            + 33660.0 * v5 * w2
            - 26657.0 * v6 * w2
            + 1132.0 * v7 * w2
            + 14112.0 * v8 * w2
            - 11608.0 * v9 * w2
            + 4245.0 * v10 * w2
            - 652.0 * v11 * w2
            + 5256.0 * v2 * w3
            - 31536.0 * v3 * w3
            + 77264.0 * v4 * w3
            - 96952.0 * v5 * w3
            + 55397.0 * v6 * w3
            + 12308.0 * v7 * w3
            - 44489.0 * v8 * w3
            + 31249.0 * v9 * w3
            - 9058.0 * v10 * w3
            + 219.0 * v11 * w3
            + 342.0 * v12 * w3
            - 2736.0 * v2 * w4
            + 16416.0 * v3 * w4
            - 30820.0 * v4 * w4
            + 3620.0 * v5 * w4
            + 68334.0 * v6 * w4
            - 115112.0 * v7 * w4
            + 95658.0 * v8 * w4
            - 48090.0 * v9 * w4
            + 16335.0 * v10 * w4
            - 4656.0 * v11 * w4
            + 1409.0 * v12 * w4
            - 358.0 * v13 * w4
            - 11988.0 * v4 * w5
            + 59940.0 * v5 * w5
            - 116946.0 * v6 * w5
            + 108584.0 * v7 * w5
            - 45969.0 * v8 * w5
            + 9015.0 * v9 * w5
            - 9443.0 * v10 * w5
            + 10929.0 * v11 * w5
            - 5060.0 * v12 * w5
            + 938.0 * v13 * w5
            + 54.0 * v14 * w5
            + 4104.0 * v4 * w6
            - 20520.0 * v5 * w6
            + 28068.0 * v6 * w6
            + 10848.0 * v7 * w6
            - 59922.0 * v8 * w6
            + 54966.0 * v9 * w6
            - 14329.0 * v10 * w6
            - 8688.0 * v11 * w6
            + 6651.0 * v12 * w6
            - 1178.0 * v13 * w6
            - 162.0 * v14 * w6
            + 10728.0 * v6 * w7
            - 42912.0 * v7 * w7
            + 65464.0 * v8 * w7
            - 45912.0 * v9 * w7
            + 7619.0 * v10 * w7
            + 11162.0 * v11 * w7
            - 7507.0 * v12 * w7
            + 1358.0 * v13 * w7
            + 162.0 * v14 * w7
            - 2736.0 * v6 * w8
            + 10944.0 * v7 * w8
            - 11866.0 * v8 * w8
            - 2706.0 * v9 * w8
            + 16200.0 * v10 * w8
            - 15338.0 * v11 * w8
            + 6788.0 * v12 * w8
            - 1286.0 * v13 * w8
            - 54.0 * v14 * w8
            - 3366.0 * v8 * w9
            + 10098.0 * v9 * w9
            - 12677.0 * v10 * w9
            + 8592.0 * v11 * w9
            - 3315.0 * v12 * w9
            + 668.0 * v13 * w9
            + 684.0 * v8 * w10
            - 2052.0 * v9 * w10
            + 2602.0 * v10 * w10
            - 1784.0 * v11 * w10
            + 692.0 * v12 * w10
            - 142.0 * v13 * w10))
        / (243.0 * (1.0 - v).powi(2) * v3 * w * (1.0 - v * w).powi(4) * (1.0 - v + v * w).powi(4));

    let part15 = (64.0
        * ca3
        * (17.0 - 119.0 * v + 391.0 * v2 - 799.0 * v3 + 1122.0 * v4 - 1122.0 * v5 + 799.0 * v6
            - 391.0 * v7
            + 119.0 * v8
            - 17.0 * v9
            - 38.0 * w
            + 266.0 * v * w
            - 758.0 * v2 * w
            + 1090.0 * v3 * w
            - 683.0 * v4 * w
            - 237.0 * v5 * w
            + 833.0 * v6 * w
            - 788.0 * v7 * w
            + 449.0 * v8 * w
            - 163.0 * v9 * w
            + 29.0 * v10 * w
            + 40.0 * w2
            - 280.0 * v * w2
            + 670.0 * v2 * w2
            - 380.0 * v3 * w2
            - 1053.0 * v4 * w2
            + 2075.0 * v5 * w2
            - 1015.0 * v6 * w2
            - 1130.0 * v7 * w2
            + 2111.0 * v8 * w2
            - 1495.0 * v9 * w2
            + 543.0 * v10 * w2
            - 86.0 * v11 * w2
            + 312.0 * v2 * w3
            - 1872.0 * v3 * w3
            + 4560.0 * v4 * w3
            - 5640.0 * v5 * w3
            + 3131.0 * v6 * w3
            + 724.0 * v7 * w3
            - 2216.0 * v8 * w3
            + 1018.0 * v9 * w3
            + 233.0 * v10 * w3
            - 334.0 * v11 * w3
            + 84.0 * v12 * w3
            - 160.0 * v2 * w4
            + 960.0 * v3 * w4
            - 1802.0 * v4 * w4
            + 210.0 * v5 * w4
            + 3970.0 * v6 * w4
            - 6580.0 * v7 * w4
            + 5059.0 * v8 * w4
            - 1825.0 * v9 * w4
            + 181.0 * v10 * w4
            - 137.0 * v11 * w4
            + 185.0 * v12 * w4
            - 61.0 * v13 * w4
            - 708.0 * v4 * w5
            + 3540.0 * v5 * w5
            - 7028.0 * v6 * w5
            + 6872.0 * v7 * w5
            - 3309.0 * v8 * w5
            + 743.0 * v9 * w5
            - 719.0 * v10 * w5
            + 1137.0 * v11 * w5
            - 651.0 * v12 * w5
            + 123.0 * v13 * w5
            + 11.0 * v14 * w5
            + 240.0 * v4 * w6
            - 1200.0 * v5 * w6
            + 1696.0 * v6 * w6
            + 416.0 * v7 * w6
            - 3350.0 * v8 * w6
            + 3554.0 * v9 * w6
            - 1121.0 * v10 * w6
            - 796.0 * v11 * w6
            + 642.0 * v12 * w6
            - 81.0 * v13 * w6
            - 33.0 * v14 * w6
            + 632.0 * v6 * w7
            - 2528.0 * v7 * w7
            + 4048.0 * v8 * w7
            - 3296.0 * v9 * w7
            + 869.0 * v10 * w7
            + 806.0 * v11 * w7
            - 602.0 * v12 * w7
            + 71.0 * v13 * w7
            + 33.0 * v14 * w7
            - 160.0 * v6 * w8
            + 640.0 * v7 * w8
            - 751.0 * v8 * w8
            + 13.0 * v9 * w8
            + 935.0 * v10 * w8
            - 1145.0 * v11 * w8
            + 571.0 * v12 * w8
            - 103.0 * v13 * w8
            - 11.0 * v14 * w8
            - 198.0 * v8 * w9
            + 594.0 * v9 * w9
            - 822.0 * v10 * w9
            + 654.0 * v11 * w9
            - 296.0 * v12 * w9
            + 68.0 * v13 * w9
            + 40.0 * v8 * w10
            - 120.0 * v9 * w10
            + 170.0 * v10 * w10
            - 140.0 * v11 * w10
            + 67.0 * v12 * w10
            - 17.0 * v13 * w10))
        / (3.0 * (1.0 - v).powi(2) * v3 * w * (1.0 - v * w).powi(4) * (1.0 - v + v * w).powi(4));

    let part16 = (128.0
        * ca3
        * l1w
        * (8.0 - 56.0 * v + 188.0 * v2 - 400.0 * v3 + 596.0 * v4 - 648.0 * v5 + 520.0 * v6
            - 304.0 * v7
            + 124.0 * v8
            - 32.0 * v9
            + 4.0 * v10
            + 6.0 * w
            - 42.0 * v * w
            + 144.0 * v2 * w
            - 318.0 * v3 * w
            + 535.0 * v4 * w
            - 761.0 * v5 * w
            + 920.0 * v6 * w
            - 896.0 * v7 * w
            + 657.0 * v8 * w
            - 335.0 * v9 * w
            + 106.0 * v10 * w
            - 16.0 * v11 * w
            + 2.0 * w2
            - 14.0 * v * w2
            + 37.0 * v2 * w2
            - 40.0 * v3 * w2
            - 62.0 * v4 * w2
            + 343.0 * v5 * w2
            - 657.0 * v6 * w2
            + 702.0 * v7 * w2
            - 388.0 * v8 * w2
            - 13.0 * v9 * w2
            + 172.0 * v10 * w2
            - 106.0 * v11 * w2
            + 24.0 * v12 * w2
            - 2.0 * w3
            + 14.0 * v * w3
            - 64.0 * v2 * w3
            + 202.0 * v3 * w3
            - 451.0 * v4 * w3
            + 737.0 * v5 * w3
            - 1005.0 * v6 * w3
            + 1248.0 * v7 * w3
            - 1375.0 * v8 * w3
            + 1231.0 * v9 * w3
            - 741.0 * v10 * w3
            + 220.0 * v11 * w3
            + 2.0 * v12 * w3
            - 16.0 * v13 * w3
            - 16.0 * v2 * w4
            + 96.0 * v3 * w4
            - 290.0 * v4 * w4
            + 570.0 * v5 * w4
            - 739.0 * v6 * w4
            + 592.0 * v7 * w4
            - 221.0 * v8 * w4
            - 71.0 * v9 * w4
            - 22.0 * v10 * w4
            + 241.0 * v11 * w4
            - 190.0 * v12 * w4
            + 50.0 * v13 * w4
            + 4.0 * v14 * w4
            + 8.0 * v2 * w5
            - 48.0 * v3 * w5
            + 144.0 * v4 * w5
            - 280.0 * v5 * w5
            + 370.0 * v6 * w5
            - 328.0 * v7 * w5
            + 166.0 * v8 * w5
            + 2.0 * v9 * w5
            + 127.0 * v10 * w5
            - 344.0 * v11 * w5
            + 214.0 * v12 * w5
            - 31.0 * v13 * w5
            - 20.0 * v14 * w5
            + 36.0 * v4 * w6
            - 180.0 * v5 * w6
            + 466.0 * v6 * w6
            - 784.0 * v7 * w6
            + 975.0 * v8 * w6
            - 937.0 * v9 * w6
            + 481.0 * v10 * w6
            + 45.0 * v11 * w6
            - 88.0 * v12 * w6
            - 14.0 * v13 * w6
            + 42.0 * v14 * w6
            - 12.0 * v4 * w7
            + 60.0 * v5 * w7
            - 132.0 * v6 * w7
            + 168.0 * v7 * w7
            - 122.0 * v8 * w7
            + 30.0 * v9 * w7
            + 183.0 * v10 * w7
            - 340.0 * v11 * w7
            + 166.0 * v12 * w7
            - v13 * w7
            - 62.0 * v14 * w7
            - 32.0 * v6 * w8
            + 128.0 * v7 * w8
            - 284.0 * v8 * w8
            + 404.0 * v9 * w8
            - 437.0 * v10 * w8
            + 350.0 * v11 * w8
            - 115.0 * v12 * w8
            - 14.0 * v13 * w8
            + 90.0 * v14 * w8
            + 8.0 * v6 * w9
            - 32.0 * v7 * w9
            + 58.0 * v8 * w9
            - 62.0 * v9 * w9
            + 22.0 * v10 * w9
            + 22.0 * v11 * w9
            - 75.0 * v12 * w9
            + 59.0 * v13 * w9
            - 110.0 * v14 * w9
            + 10.0 * v8 * w10
            - 30.0 * v9 * w10
            + 65.0 * v10 * w10
            - 80.0 * v11 * w10
            + 71.0 * v12 * w10
            - 36.0 * v13 * w10
            + 92.0 * v14 * w10
            - 2.0 * v8 * w11
            + 6.0 * v9 * w11
            - 12.0 * v10 * w11
            + 14.0 * v11 * w11
            - 7.0 * v12 * w11
            + v13 * w11
            - 52.0 * v14 * w11
            - 2.0 * v12 * w12
            + 2.0 * v13 * w12
            + 20.0 * v14 * w12
            - 4.0 * v14 * w13))
        / ((1.0 - v).powi(2) * v3 * w2 * (1.0 - v * w).powi(4) * (1.0 - v + v * w).powi(4));

    let part17 = (128.0
        * ca3
        * lv
        * (10.0 - 70.0 * v + 234.0 * v2 - 494.0 * v3 + 728.0 * v4 - 780.0 * v5 + 614.0 * v6
            - 350.0 * v7
            + 138.0 * v8
            - 34.0 * v9
            + 4.0 * v10
            + 6.0 * w
            - 42.0 * v * w
            + 146.0 * v2 * w
            - 330.0 * v3 * w
            + 572.0 * v4 * w
            - 836.0 * v5 * w
            + 1026.0 * v6 * w
            - 1002.0 * v7 * w
            + 730.0 * v8 * w
            - 366.0 * v9 * w
            + 112.0 * v10 * w
            - 16.0 * v11 * w
            + 2.0 * w2
            - 14.0 * v * w2
            + 32.0 * v2 * w2
            - 10.0 * v3 * w2
            - 152.0 * v4 * w2
            + 518.0 * v5 * w2
            - 896.0 * v6 * w2
            + 938.0 * v7 * w2
            - 548.0 * v8 * w2
            + 46.0 * v9 * w2
            + 170.0 * v10 * w2
            - 110.0 * v11 * w2
            + 24.0 * v12 * w2
            - 2.0 * w3
            + 14.0 * v * w3
            - 64.0 * v2 * w3
            + 202.0 * v3 * w3
            - 448.0 * v4 * w3
            + 722.0 * v5 * w3
            - 973.0 * v6 * w3
            + 1210.0 * v7 * w3
            - 1380.0 * v8 * w3
            + 1316.0 * v9 * w3
            - 839.0 * v10 * w3
            + 260.0 * v11 * w3
            - 2.0 * v12 * w3
            - 16.0 * v13 * w3
            - 16.0 * v2 * w4
            + 96.0 * v3 * w4
            - 288.0 * v4 * w4
            + 560.0 * v5 * w4
            - 703.0 * v6 * w4
            + 508.0 * v7 * w4
            - 41.0 * v8 * w4
            - 359.0 * v9 * w4
            + 214.0 * v10 * w4
            + 171.0 * v11 * w4
            - 198.0 * v12 * w4
            + 56.0 * v13 * w4
            + 4.0 * v14 * w4
            + 8.0 * v2 * w5
            - 48.0 * v3 * w5
            + 144.0 * v4 * w5
            - 280.0 * v5 * w5
            + 358.0 * v6 * w5
            - 280.0 * v7 * w5
            + 30.0 * v8 * w5
            + 242.0 * v9 * w5
            - 23.0 * v10 * w5
            - 388.0 * v11 * w5
            + 293.0 * v12 * w5
            - 56.0 * v13 * w5
            - 22.0 * v14 * w5
            + 36.0 * v4 * w6
            - 180.0 * v5 * w6
            + 468.0 * v6 * w6
            - 792.0 * v7 * w6
            + 1005.0 * v8 * w6
            - 999.0 * v9 * w6
            + 430.0 * v10 * w6
            + 241.0 * v11 * w6
            - 228.0 * v12 * w6
            + 19.0 * v13 * w6
            + 54.0 * v14 * w6
            - 12.0 * v4 * w7
            + 60.0 * v5 * w7
            - 132.0 * v6 * w7
            + 168.0 * v7 * w7
            - 124.0 * v8 * w7
            + 36.0 * v9 * w7
            + 265.0 * v10 * w7
            - 514.0 * v11 * w7
            + 261.0 * v12 * w7
            - 8.0 * v13 * w7
            - 92.0 * v14 * w7
            - 32.0 * v6 * w8
            + 128.0 * v7 * w8
            - 282.0 * v8 * w8
            + 398.0 * v9 * w8
            - 473.0 * v10 * w8
            + 432.0 * v11 * w8
            - 137.0 * v12 * w8
            - 34.0 * v13 * w8
            + 132.0 * v14 * w8
            + 8.0 * v6 * w9
            - 32.0 * v7 * w9
            + 58.0 * v8 * w9
            - 62.0 * v9 * w9
            + 40.0 * v10 * w9
            - 14.0 * v11 * w9
            - 80.0 * v12 * w9
            + 82.0 * v13 * w9
            - 148.0 * v14 * w9
            + 10.0 * v8 * w10
            - 30.0 * v9 * w10
            + 60.0 * v10 * w10
            - 70.0 * v11 * w10
            + 83.0 * v12 * w10
            - 53.0 * v13 * w10
            + 116.0 * v14 * w10
            - 2.0 * v8 * w11
            + 6.0 * v9 * w11
            - 12.0 * v10 * w11
            + 14.0 * v11 * w11
            - 16.0 * v12 * w11
            + 10.0 * v13 * w11
            - 62.0 * v14 * w11
            + 22.0 * v14 * w12
            - 4.0 * v14 * w13))
        / ((1.0 - v).powi(2) * v3 * w2 * (1.0 - v * w).powi(4) * (1.0 - v + v * w).powi(4));

    part1
        + part2
        + part3
        + part4
        + part5
        + part6
        + part7
        + part8
        + part9
        + part10
        + part11
        + part12
        + part13
        + part14
        + part15
        + part16
        + part17
}

/// `STRUV16(W,V,X3,S)`.
#[must_use]
pub fn struv16(w: f64, v: f64, _x3: f64, _s: f64, ctx: &MeContext, pre: &Precalc) -> f64 {
    let cf = ctx.cf;
    let ca = ctx.ca;
    let (v2, v3, v4, v5, v6, v7, v8, v9, v10, v11) = (
        pre.v2, pre.v3, pre.v4, pre.v5, pre.v6, pre.v7, pre.v8, pre.v9, pre.v10, pre.v11,
    );
    let (w2, w3, w4, w5, w6, w7, w8, w9, w10, w11) = (
        pre.w2, pre.w3, pre.w4, pre.w5, pre.w6, pre.w7, pre.w8, pre.w9, pre.w10, pre.w11,
    );
    let ca2 = pre.ca2;
    let ca3 = ca.powi(3);
    let ca4 = pre.ca4;
    let cacf = ca * cf;
    let ca3cf = ca3 * cf;
    let cf2 = cf.powi(2);
    let cacf2 = ca * cf2;
    let (l1v, lw, lvw, l1vw, lms, l1w, lv, lmss) = (
        pre.l1v, pre.lw, pre.lvw, pre.l1vw, pre.lms, pre.l1w, pre.lv, pre.lmss,
    );

    let part1 = (-4.0
        * cf
        * lvw
        * (4.0 * ca2 - 4.0 * ca4 - 12.0 * ca2 * v + 12.0 * ca4 * v + 16.0 * ca2 * v2
            - 16.0 * ca4 * v2
            - 8.0 * ca2 * v3
            + 8.0 * ca4 * v3
            - 2.0 * w
            - 2.0 * ca2 * w
            + 7.0 * v * w
            + 11.0 * ca2 * v * w
            - 9.0 * ca4 * v * w
            - 8.0 * v2 * w
            - 16.0 * ca2 * v2 * w
            + 28.0 * ca4 * v2 * w
            + 5.0 * v3 * w
            + 9.0 * ca2 * v3 * w
            - 29.0 * ca4 * v3 * w
            - 2.0 * v4 * w
            + 2.0 * ca2 * v4 * w
            + 6.0 * ca4 * v4 * w
            - 3.0 * v * w2
            - 5.0 * ca2 * v * w2
            + 19.0 * ca4 * v * w2
            + 5.0 * v2 * w2
            + 4.0 * ca2 * v2 * w2
            - 54.0 * ca4 * v2 * w2
            - 6.0 * v3 * w2
            - 6.0 * ca2 * v3 * w2
            + 87.0 * ca4 * v3 * w2
            + 4.0 * v4 * w2
            - 5.0 * ca2 * v4 * w2
            - 48.0 * ca4 * v4 * w2
            + 16.0 * ca4 * v5 * w2
            + 5.0 * ca2 * v2 * w3
            - 13.0 * ca4 * v2 * w3
            + 3.0 * v3 * w3
            + 8.0 * ca2 * v3 * w3
            - 36.0 * ca4 * v3 * w3
            - 3.0 * v4 * w3
            + 3.0 * ca2 * v4 * w3
            + 33.0 * ca4 * v4 * w3
            - 32.0 * ca4 * v5 * w3
            - v3 * w4
            - 7.0 * ca2 * v3 * w4
            + 43.0 * ca4 * v3 * w4
            + v4 * w4
            - ca2 * v4 * w4
            - 27.0 * ca4 * v4 * w4
            + 48.0 * ca4 * v5 * w4
            - 16.0 * ca4 * v4 * w5
            - 32.0 * ca4 * v5 * w5
            + 16.0 * ca4 * v5 * w6))
        / (ca * (1.0 - v).powi(2) * v2 * w2);

    let part2 = -(4.0
        * cf
        * l1vw
        * (3.0 - 2.0 * ca2 + 3.0 * ca4 - 8.0 * ca2 * cf2 - 14.0 * v + 11.0 * ca2 * v
            - 15.0 * ca4 * v
            - 8.0 * cacf * v
            + 8.0 * ca3cf * v
            + 24.0 * ca2 * cf2 * v
            + 27.0 * v2
            - 25.0 * ca2 * v2
            + 35.0 * ca4 * v2
            + 28.0 * cacf * v2
            - 28.0 * ca3cf * v2
            - 28.0 * ca2 * cf2 * v2
            - 28.0 * v3
            + 30.0 * ca2 * v3
            - 50.0 * ca4 * v3
            - 40.0 * cacf * v3
            + 40.0 * ca3cf * v3
            + 16.0 * ca2 * cf2 * v3
            + 17.0 * v4
            - 20.0 * ca2 * v4
            + 45.0 * ca4 * v4
            + 30.0 * cacf * v4
            - 30.0 * ca3cf * v4
            - 4.0 * ca2 * cf2 * v4
            - 6.0 * v5
            + 7.0 * ca2 * v5
            - 23.0 * ca4 * v5
            - 12.0 * cacf * v5
            + 12.0 * ca3cf * v5
            + v6
            - ca2 * v6
            + 5.0 * ca4 * v6
            + 2.0 * cacf * v6
            - 2.0 * ca3cf * v6
            + 12.0 * v * w
            - 6.0 * ca2 * v * w
            + ca4 * v * w
            + 8.0 * cacf * v * w
            - 8.0 * ca2 * cf2 * v * w
            - 44.0 * v2 * w
            + 29.0 * ca2 * v2 * w
            - 8.0 * ca4 * v2 * w
            - 40.0 * cacf * v2 * w
            + 8.0 * ca3cf * v2 * w
            + 24.0 * ca2 * cf2 * v2 * w
            + 64.0 * v3 * w
            - 56.0 * ca2 * v3 * w
            + 18.0 * ca4 * v3 * w
            + 80.0 * cacf * v3 * w
            - 28.0 * ca3cf * v3 * w
            - 24.0 * ca2 * cf2 * v3 * w
            - 48.0 * v4 * w
            + 54.0 * ca2 * v4 * w
            - 16.0 * ca4 * v4 * w
            - 80.0 * cacf * v4 * w
            + 36.0 * ca3cf * v4 * w
            + 8.0 * ca2 * cf2 * v4 * w
            + 20.0 * v5 * w
            - 26.0 * ca2 * v5 * w
            + 5.0 * ca4 * v5 * w
            + 40.0 * cacf * v5 * w
            - 20.0 * ca3cf * v5 * w
            - 4.0 * v6 * w
            + 5.0 * ca2 * v6 * w
            - 8.0 * cacf * v6 * w
            + 4.0 * ca3cf * v6 * w
            + 19.0 * v2 * w2
            - 9.0 * ca2 * v2 * w2
            + 3.0 * ca4 * v2 * w2
            + 12.0 * cacf * v2 * w2
            - 4.0 * ca3cf * v2 * w2
            - 12.0 * ca2 * cf2 * v2 * w2
            - 52.0 * v3 * w2
            + 38.0 * ca2 * v3 * w2
            - 6.0 * ca4 * v3 * w2
            - 56.0 * cacf * v3 * w2
            + 24.0 * ca3cf * v3 * w2
            + 16.0 * ca2 * cf2 * v3 * w2
            + 54.0 * v4 * w2
            - 60.0 * ca2 * v4 * w2
            - 10.0 * ca4 * v4 * w2
            + 84.0 * cacf * v4 * w2
            - 36.0 * ca3cf * v4 * w2
            - 8.0 * ca2 * cf2 * v4 * w2
            - 28.0 * v5 * w2
            + 42.0 * ca2 * v5 * w2
            + 26.0 * ca4 * v5 * w2
            - 56.0 * cacf * v5 * w2
            + 24.0 * ca3cf * v5 * w2
            + 7.0 * v6 * w2
            - 11.0 * ca2 * v6 * w2
            - 13.0 * ca4 * v6 * w2
            + 14.0 * cacf * v6 * w2
            - 6.0 * ca3cf * v6 * w2
            + 16.0 * v3 * w3
            - 12.0 * ca2 * v3 * w3
            + 6.0 * ca4 * v3 * w3
            + 16.0 * cacf * v3 * w3
            - 4.0 * ca3cf * v3 * w3
            - 8.0 * ca2 * cf2 * v3 * w3
            - 32.0 * v4 * w3
            + 38.0 * ca2 * v4 * w3
            - 48.0 * cacf * v4 * w3
            + 20.0 * ca3cf * v4 * w3
            + 8.0 * ca2 * cf2 * v4 * w3
            + 24.0 * v5 * w3
            - 40.0 * ca2 * v5 * w3
            - 22.0 * ca4 * v5 * w3
            + 48.0 * cacf * v5 * w3
            - 24.0 * ca3cf * v5 * w3
            - 8.0 * v6 * w3
            + 14.0 * ca2 * v6 * w3
            + 16.0 * ca4 * v6 * w3
            - 16.0 * cacf * v6 * w3
            + 8.0 * ca3cf * v6 * w3
            + 9.0 * v4 * w4
            - 12.0 * ca2 * v4 * w4
            - 3.0 * ca4 * v4 * w4
            + 14.0 * cacf * v4 * w4
            - 6.0 * ca3cf * v4 * w4
            - 4.0 * ca2 * cf2 * v4 * w4
            - 14.0 * v5 * w4
            + 23.0 * ca2 * v5 * w4
            + 21.0 * ca4 * v5 * w4
            - 28.0 * cacf * v5 * w4
            + 12.0 * ca3cf * v5 * w4
            + 7.0 * v6 * w4
            - 11.0 * ca2 * v6 * w4
            - 13.0 * ca4 * v6 * w4
            + 14.0 * cacf * v6 * w4
            - 6.0 * ca3cf * v6 * w4
            + 4.0 * v5 * w5
            - 6.0 * ca2 * v5 * w5
            - 7.0 * ca4 * v5 * w5
            + 8.0 * cacf * v5 * w5
            - 4.0 * ca3cf * v5 * w5
            - 4.0 * v6 * w5
            + 5.0 * ca2 * v6 * w5
            - 8.0 * cacf * v6 * w5
            + 4.0 * ca3cf * v6 * w5
            + v6 * w6
            - ca2 * v6 * w6
            + 5.0 * ca4 * v6 * w6
            + 2.0 * cacf * v6 * w6
            - 2.0 * ca3cf * v6 * w6))
        / (ca * (1.0 - v) * v2 * w * (1.0 - v + v * w).powi(3));

    let part3 = -(4.0
        * cf
        * l1v
        * (4.0 * ca4 - 20.0 * ca4 * v + 44.0 * ca4 * v2 - 52.0 * ca4 * v3 + 32.0 * ca4 * v4
            - 8.0 * ca4 * v5
            - 2.0 * ca2 * w
            + 2.0 * ca4 * w
            - v * w
            + 5.0 * ca2 * v * w
            - 3.0 * ca4 * v * w
            + 2.0 * v2 * w
            - 6.0 * ca2 * v2 * w
            - 10.0 * ca4 * v2 * w
            - 2.0 * v3 * w
            + 6.0 * ca2 * v3 * w
            + 24.0 * ca4 * v3 * w
            + 2.0 * v4 * w
            - 4.0 * ca2 * v4 * w
            - 8.0 * ca4 * v4 * w
            - v5 * w
            + ca2 * v5 * w
            - 13.0 * ca4 * v5 * w
            + 8.0 * ca4 * v6 * w
            + 2.0 * ca2 * v2 * w2
            - ca4 * v2 * w2
            + v3 * w2
            - 7.0 * ca2 * v3 * w2
            + 6.0 * ca4 * v3 * w2
            - 3.0 * v4 * w2
            + 5.0 * ca2 * v4 * w2
            - 24.0 * ca4 * v4 * w2
            + v5 * w2
            + ca2 * v5 * w2
            + 22.0 * ca4 * v5 * w2
            + v6 * w2
            - ca2 * v6 * w2
            + 5.0 * ca4 * v6 * w2
            - 8.0 * ca4 * v7 * w2
            - 10.0 * ca4 * v2 * w3
            + 2.0 * ca2 * v3 * w3
            + 42.0 * ca4 * v3 * w3
            + v4 * w3
            + 3.0 * ca2 * v4 * w3
            - 56.0 * ca4 * v4 * w3
            + 2.0 * v5 * w3
            - 8.0 * ca2 * v5 * w3
            + 52.0 * ca4 * v5 * w3
            - 3.0 * v6 * w3
            + 3.0 * ca2 * v6 * w3
            - 56.0 * ca4 * v6 * w3
            + 24.0 * ca4 * v7 * w3
            - 6.0 * ca2 * v4 * w4
            - 17.0 * ca4 * v4 * w4
            - 3.0 * v5 * w4
            + 9.0 * ca2 * v5 * w4
            + 13.0 * ca4 * v5 * w4
            + 3.0 * v6 * w4
            - 3.0 * ca2 * v6 * w4
            + 40.0 * ca4 * v6 * w4
            - 24.0 * ca4 * v7 * w4
            + 2.0 * ca2 * v4 * w5
            + 8.0 * ca4 * v4 * w5
            + v5 * w5
            - 3.0 * ca2 * v5 * w5
            - 13.0 * ca4 * v5 * w5
            - v6 * w5
            + ca2 * v6 * w5
            - 27.0 * ca4 * v6 * w5
            + 16.0 * ca4 * v7 * w5
            + 16.0 * ca4 * v6 * w6
            - 8.0 * ca4 * v7 * w6))
        / (ca * (1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w) * (1.0 - v + v * w));

    let part4 = -(4.0
        * cf
        * lw
        * (4.0 * ca2 - 8.0 * ca4 - 16.0 * ca2 * v + 44.0 * ca4 * v + 28.0 * ca2 * v2
            - 112.0 * ca4 * v2
            - 24.0 * ca2 * v3
            + 164.0 * ca4 * v3
            + 8.0 * ca2 * v4
            - 144.0 * ca4 * v4
            + 72.0 * ca4 * v5
            - 16.0 * ca4 * v6
            - w
            + 5.0 * ca4 * w
            + 3.0 * v * w
            + 5.0 * ca2 * v * w
            - 48.0 * ca4 * v * w
            - 3.0 * v2 * w
            - 13.0 * ca2 * v2 * w
            + 164.0 * ca4 * v2 * w
            + v3 * w
            + 9.0 * ca2 * v3 * w
            - 282.0 * ca4 * v3 * w
            + 9.0 * ca2 * v4 * w
            + 247.0 * ca4 * v4 * w
            - 10.0 * ca2 * v5 * w
            - 86.0 * ca4 * v5 * w
            - 16.0 * ca4 * v6 * w
            + 16.0 * ca4 * v7 * w
            - v * w2
            - 6.0 * ca2 * v * w2
            + 28.0 * ca4 * v * w2
            + v2 * w2
            + 9.0 * ca2 * v2 * w2
            - 129.0 * ca4 * v2 * w2
            + v3 * w2
            + 3.0 * ca2 * v3 * w2
            + 266.0 * ca4 * v3 * w2
            - v4 * w2
            - 29.0 * ca2 * v4 * w2
            - 227.0 * ca4 * v4 * w2
            + 27.0 * ca2 * v5 * w2
            - 4.0 * ca4 * v5 * w2
            + 118.0 * ca4 * v6 * w2
            - 56.0 * ca4 * v7 * w2
            - 6.0 * ca4 * v * w3
            + 6.0 * ca2 * v2 * w3
            + 29.0 * ca4 * v2 * w3
            - 17.0 * ca2 * v3 * w3
            - 97.0 * ca4 * v3 * w3
            - v4 * w3
            + 35.0 * ca2 * v4 * w3
            + 67.0 * ca4 * v4 * w3
            + v5 * w3
            - 40.0 * ca2 * v5 * w3
            + 162.0 * ca4 * v5 * w3
            - 239.0 * ca4 * v6 * w3
            + 104.0 * ca4 * v7 * w3
            + 3.0 * ca2 * v3 * w4
            + 25.0 * ca4 * v3 * w4
            - 11.0 * ca2 * v4 * w4
            - 4.0 * ca4 * v4 * w4
            + v5 * w4
            + 34.0 * ca2 * v5 * w4
            - 202.0 * ca4 * v5 * w4
            - v6 * w4
            + 2.0 * ca2 * v6 * w4
            + 257.0 * ca4 * v6 * w4
            - 120.0 * ca4 * v7 * w4
            - 2.0 * ca4 * v3 * w5
            + v4 * w5
            - 2.0 * ca2 * v4 * w5
            - 8.0 * ca4 * v4 * w5
            - 3.0 * v5 * w5
            - 19.0 * ca2 * v5 * w5
            + 137.0 * ca4 * v5 * w5
            + 2.0 * v6 * w5
            - 3.0 * ca2 * v6 * w5
            - 163.0 * ca4 * v6 * w5
            + 88.0 * ca4 * v7 * w5
            + v5 * w6
            + 7.0 * ca2 * v5 * w6
            - 51.0 * ca4 * v5 * w6
            - v6 * w6
            + ca2 * v6 * w6
            + 59.0 * ca4 * v6 * w6
            - 40.0 * ca4 * v7 * w6
            + 8.0 * ca4 * v5 * w7
            - 8.0 * ca4 * v6 * w7
            + 8.0 * ca4 * v7 * w7))
        / (ca * (1.0 - v).powi(2) * v2 * (1.0 - w) * w2 * (1.0 - v * w) * (1.0 - v + v * w));

    let part5 = -(4.0
        * cf
        * lmss
        * (4.0 * ca3 - 32.0 * ca3 * v + 116.0 * ca3 * v2 - 248.0 * ca3 * v3 + 340.0 * ca3 * v4
            - 304.0 * ca3 * v5
            + 172.0 * ca3 * v6
            - 56.0 * ca3 * v7
            + 8.0 * ca3 * v8
            + 4.0 * cacf2 * w
            + 20.0 * ca3 * v * w
            + 4.0 * cf * v * w
            - 4.0 * ca2 * cf * v * w
            - 16.0 * cacf2 * v * w
            - 148.0 * ca3 * v2 * w
            - 18.0 * cf * v2 * w
            + 18.0 * ca2 * cf * v2 * w
            + 26.0 * cacf2 * v2 * w
            + 480.0 * ca3 * v3 * w
            + 34.0 * cf * v3 * w
            - 34.0 * ca2 * cf * v3 * w
            - 22.0 * cacf2 * v3 * w
            - 880.0 * ca3 * v4 * w
            - 35.0 * cf * v4 * w
            + 35.0 * ca2 * cf * v4 * w
            + 10.0 * cacf2 * v4 * w
            + 980.0 * ca3 * v5 * w
            + 21.0 * cf * v5 * w
            - 21.0 * ca2 * cf * v5 * w
            - 2.0 * cacf2 * v5 * w
            - 660.0 * ca3 * v6 * w
            - 7.0 * cf * v6 * w
            + 7.0 * ca2 * cf * v6 * w
            + 248.0 * ca3 * v7 * w
            + cf * v7 * w
            - ca2 * cf * v7 * w
            - 40.0 * ca3 * v8 * w
            + 12.0 * cacf2 * v * w2
            + 56.0 * ca3 * v2 * w2
            + 6.0 * cf * v2 * w2
            - 14.0 * ca2 * cf * v2 * w2
            - 54.0 * cacf2 * v2 * w2
            - 360.0 * ca3 * v3 * w2
            - 26.0 * cf * v3 * w2
            + 72.0 * ca2 * cf * v3 * w2
            + 98.0 * cacf2 * v3 * w2
            + 984.0 * ca3 * v4 * w2
            + 45.0 * cf * v4 * w2
            - 157.0 * ca2 * cf * v4 * w2
            - 90.0 * cacf2 * v4 * w2
            - 1456.0 * ca3 * v5 * w2
            - 39.0 * cf * v5 * w2
            + 187.0 * ca2 * cf * v5 * w2
            + 42.0 * cacf2 * v5 * w2
            + 1224.0 * ca3 * v6 * w2
            + 17.0 * cf * v6 * w2
            - 129.0 * ca2 * cf * v6 * w2
            - 8.0 * cacf2 * v6 * w2
            - 552.0 * ca3 * v7 * w2
            - 3.0 * cf * v7 * w2
            + 49.0 * ca2 * cf * v7 * w2
            + 104.0 * ca3 * v8 * w2
            - 8.0 * ca2 * cf * v8 * w2
            + 12.0 * cacf2 * v2 * w3
            + 100.0 * ca3 * v3 * w3
            + 8.0 * cf * v3 * w3
            - 14.0 * ca2 * cf * v3 * w3
            - 48.0 * cacf2 * v3 * w3
            - 548.0 * ca3 * v4 * w3
            - 25.0 * cf * v4 * w3
            + 65.0 * ca2 * cf * v4 * w3
            + 78.0 * cacf2 * v4 * w3
            + 1220.0 * ca3 * v5 * w3
            + 31.0 * cf * v5 * w3
            - 131.0 * ca2 * cf * v5 * w3
            - 58.0 * cacf2 * v5 * w3
            - 1372.0 * ca3 * v6 * w3
            - 18.0 * cf * v6 * w3
            + 138.0 * ca2 * cf * v6 * w3
            + 16.0 * cacf2 * v6 * w3
            + 776.0 * ca3 * v7 * w3
            + 4.0 * cf * v7 * w3
            - 74.0 * ca2 * cf * v7 * w3
            - 176.0 * ca3 * v8 * w3
            + 16.0 * ca2 * cf * v8 * w3
            + 4.0 * cacf2 * v3 * w4
            + 128.0 * ca3 * v4 * w4
            + 7.0 * cf * v4 * w4
            - 7.0 * ca2 * cf * v4 * w4
            - 14.0 * cacf2 * v4 * w4
            - 568.0 * ca3 * v5 * w4
            - 17.0 * cf * v5 * w4
            + 23.0 * ca2 * cf * v5 * w4
            + 18.0 * cacf2 * v5 * w4
            + 960.0 * ca3 * v6 * w4
            + 14.0 * cf * v6 * w4
            - 34.0 * ca2 * cf * v6 * w4
            - 8.0 * cacf2 * v6 * w4
            - 728.0 * ca3 * v7 * w4
            - 4.0 * cf * v7 * w4
            + 26.0 * ca2 * cf * v7 * w4
            + 208.0 * ca3 * v8 * w4
            - 8.0 * ca2 * cf * v8 * w4
            + 116.0 * ca3 * v5 * w5
            + 4.0 * cf * v5 * w5
            - 2.0 * ca2 * cf * v5 * w5
            - 396.0 * ca3 * v6 * w5
            - 7.0 * cf * v6 * w5
            + 3.0 * ca2 * cf * v6 * w5
            + 456.0 * ca3 * v7 * w5
            + 3.0 * cf * v7 * w5
            - ca2 * cf * v7 * w5
            - 176.0 * ca3 * v8 * w5
            + 76.0 * ca3 * v6 * w6
            + cf * v6 * w6
            - ca2 * cf * v6 * w6
            - 176.0 * ca3 * v7 * w6
            - cf * v7 * w6
            + ca2 * cf * v7 * w6
            + 104.0 * ca3 * v8 * w6
            + 32.0 * ca3 * v7 * w7
            - 40.0 * ca3 * v8 * w7
            + 8.0 * ca3 * v8 * w8))
        / ((1.0 - v).powi(2) * v2 * w2 * (1.0 - v + v * w).powi(3));

    let part6 = (2.0
        * cf
        * lms
        * (8.0 * ca2 - 8.0 * ca4 - 24.0 * ca2 * v + 32.0 * ca4 * v + 32.0 * ca2 * v2
            - 64.0 * ca4 * v2
            - 16.0 * ca2 * v3
            + 72.0 * ca4 * v3
            - 48.0 * ca4 * v4
            + 16.0 * ca4 * v5
            - w
            - 2.0 * ca2 * w
            + 3.0 * ca4 * w
            - 8.0 * ca3cf * w
            + 2.0 * v * w
            - 14.0 * ca2 * v * w
            + 4.0 * ca4 * v * w
            + 8.0 * ca3cf * v * w
            - 2.0 * v2 * w
            + 52.0 * ca2 * v2 * w
            - 42.0 * ca4 * v2 * w
            + 2.0 * v3 * w
            - 74.0 * ca2 * v3 * w
            + 112.0 * ca4 * v3 * w
            - v4 * w
            + 46.0 * ca2 * v4 * w
            - 165.0 * ca4 * v4 * w
            + 128.0 * ca4 * v5 * w
            - 48.0 * ca4 * v6 * w
            + 2.0 * w2
            - 2.0 * ca4 * w2
            - v * w2
            - 2.0 * ca2 * v * w2
            + 11.0 * ca4 * v * w2
            + 24.0 * ca3cf * v * w2
            - 4.0 * v2 * w2
            + 8.0 * ca2 * v2 * w2
            + 4.0 * ca4 * v2 * w2
            - 16.0 * ca3cf * v2 * w2
            + 8.0 * v3 * w2
            - 46.0 * ca2 * v3 * w2
            - 10.0 * ca4 * v3 * w2
            - 12.0 * v4 * w2
            + 62.0 * ca2 * v4 * w2
            + 6.0 * ca4 * v4 * w2
            + 7.0 * v5 * w2
            - 58.0 * ca2 * v5 * w2
            + 83.0 * ca4 * v5 * w2
            - 96.0 * ca4 * v6 * w2
            + 48.0 * ca4 * v7 * w2
            - 2.0 * w3
            + 2.0 * ca4 * w3
            - 2.0 * v * w3
            + 8.0 * ca2 * v * w3
            - 14.0 * ca4 * v * w3
            + 5.0 * v2 * w3
            + 10.0 * ca2 * v2 * w3
            - 7.0 * ca4 * v2 * w3
            - 24.0 * ca3cf * v2 * w3
            + 4.0 * ca2 * v3 * w3
            - 68.0 * ca4 * v3 * w3
            + 8.0 * ca3cf * v3 * w3
            + 2.0 * v4 * w3
            + 22.0 * ca2 * v4 * w3
            + 72.0 * ca4 * v4 * w3
            + 4.0 * v5 * w3
            + 2.0 * ca2 * v5 * w3
            - 134.0 * ca4 * v5 * w3
            - 7.0 * v6 * w3
            + 34.0 * ca2 * v6 * w3
            + 37.0 * ca4 * v6 * w3
            - 16.0 * ca4 * v8 * w3
            + 6.0 * v * w4
            - 6.0 * ca4 * v * w4
            - 6.0 * v2 * w4
            - 24.0 * ca2 * v2 * w4
            + 54.0 * ca4 * v2 * w4
            + v3 * w4
            + 2.0 * ca2 * v3 * w4
            - 75.0 * ca4 * v3 * w4
            + 8.0 * ca3cf * v3 * w4
            - 5.0 * v4 * w4
            - 18.0 * ca2 * v4 * w4
            + 263.0 * ca4 * v4 * w4
            - 3.0 * v5 * w4
            - 24.0 * ca2 * v5 * w4
            - 149.0 * ca4 * v5 * w4
            + 4.0 * v6 * w4
            - 30.0 * ca2 * v6 * w4
            + 138.0 * ca4 * v6 * w4
            + 3.0 * v7 * w4
            - 6.0 * ca2 * v7 * w4
            - 13.0 * ca4 * v7 * w4
            + 16.0 * ca4 * v8 * w4
            - 6.0 * v2 * w5
            + 6.0 * ca4 * v2 * w5
            + 10.0 * v3 * w5
            + 24.0 * ca2 * v3 * w5
            - 58.0 * ca4 * v3 * w5
            - 8.0 * v4 * w5
            - 16.0 * ca2 * v4 * w5
            + 112.0 * ca4 * v4 * w5
            + 10.0 * v5 * w5
            + 22.0 * ca2 * v5 * w5
            - 352.0 * ca4 * v5 * w5
            - 2.0 * v6 * w5
            + 26.0 * ca2 * v6 * w5
            + 152.0 * ca4 * v6 * w5
            - 4.0 * v7 * w5
            + 8.0 * ca2 * v7 * w5
            - 100.0 * ca4 * v7 * w5
            - 16.0 * ca4 * v8 * w5
            + 2.0 * v3 * w6
            - 2.0 * ca4 * v3 * w6
            - 4.0 * v4 * w6
            - 8.0 * ca2 * v4 * w6
            + 20.0 * ca4 * v4 * w6
            + 4.0 * v5 * w6
            + 8.0 * ca2 * v5 * w6
            - 44.0 * ca4 * v5 * w6
            - 5.0 * v6 * w6
            - 10.0 * ca2 * v6 * w6
            + 207.0 * ca4 * v6 * w6
            + 3.0 * v7 * w6
            - 6.0 * ca2 * v7 * w6
            - 45.0 * ca4 * v7 * w6
            + 48.0 * ca4 * v8 * w6
            - 64.0 * ca4 * v7 * w7
            - 16.0 * ca4 * v8 * w7
            + 16.0 * ca4 * v8 * w8))
        / (ca * (1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w).powi(3));

    let part7 = -(cf
        * (16.0 * ca2 * v - 48.0 * ca4 * v - 80.0 * ca2 * v2
            + 304.0 * ca4 * v2
            + 160.0 * ca2 * v3
            - 832.0 * ca4 * v3
            - 160.0 * ca2 * v4
            + 1280.0 * ca4 * v4
            + 80.0 * ca2 * v5
            - 1200.0 * ca4 * v5
            - 16.0 * ca2 * v6
            + 688.0 * ca4 * v6
            - 224.0 * ca4 * v7
            + 32.0 * ca4 * v8
            - w
            + 6.0 * ca2 * w
            - 5.0 * ca4 * w
            + 7.0 * v * w
            - 58.0 * ca2 * v * w
            + 83.0 * ca4 * v * w
            - 24.0 * v2 * w
            + 232.0 * ca2 * v2 * w
            - 464.0 * ca4 * v2 * w
            + 50.0 * v3 * w
            - 452.0 * ca2 * v3 * w
            + 1170.0 * ca4 * v3 * w
            - 65.0 * v4 * w
            + 438.0 * ca2 * v4 * w
            - 1397.0 * ca4 * v4 * w
            + 51.0 * v5 * w
            - 178.0 * ca2 * v5 * w
            + 447.0 * ca4 * v5 * w
            - 22.0 * v6 * w
            - 4.0 * ca2 * v6 * w
            + 794.0 * ca4 * v6 * w
            + 4.0 * v7 * w
            + 16.0 * ca2 * v7 * w
            - 1044.0 * ca4 * v7 * w
            + 512.0 * ca4 * v8 * w
            - 96.0 * ca4 * v9 * w
            - 14.0 * w2
            + 8.0 * ca2 * w2
            + 6.0 * ca4 * w2
            + 72.0 * v * w2
            - 22.0 * ca2 * v * w2
            - 54.0 * ca4 * v * w2
            - 161.0 * v2 * w2
            + 12.0 * ca2 * v2 * w2
            + 201.0 * ca4 * v2 * w2
            + 186.0 * v3 * w2
            - 38.0 * ca2 * v3 * w2
            - 264.0 * ca4 * v3 * w2
            - 74.0 * v4 * w2
            + 248.0 * ca2 * v4 * w2
            - 510.0 * ca4 * v4 * w2
            - 80.0 * v5 * w2
            - 466.0 * ca2 * v5 * w2
            + 2342.0 * ca4 * v5 * w2
            + 119.0 * v6 * w2
            + 348.0 * ca2 * v6 * w2
            - 3471.0 * ca4 * v6 * w2
            - 58.0 * v7 * w2
            - 82.0 * ca2 * v7 * w2
            + 2400.0 * ca4 * v7 * w2
            + 10.0 * v8 * w2
            - 8.0 * ca2 * v8 * w2
            - 554.0 * ca4 * v8 * w2
            - 192.0 * ca4 * v9 * w2
            + 96.0 * ca4 * v10 * w2
            + 16.0 * w3
            - 8.0 * ca2 * w3
            - 8.0 * ca4 * w3
            - 80.0 * v * w3
            + 12.0 * ca2 * v * w3
            + 76.0 * ca4 * v * w3
            + 127.0 * v2 * w3
            + 34.0 * ca2 * v2 * w3
            - 217.0 * ca4 * v2 * w3
            - 2.0 * v3 * w3
            - 74.0 * ca2 * v3 * w3
            + 176.0 * ca4 * v3 * w3
            - 315.0 * v4 * w3
            + 118.0 * ca2 * v4 * w3
            + 429.0 * ca4 * v4 * w3
            + 609.0 * v5 * w3
            - 414.0 * ca2 * v5 * w3
            - 1571.0 * ca4 * v5 * w3
            - 590.0 * v6 * w3
            + 824.0 * ca2 * v6 * w3
            + 2054.0 * ca4 * v6 * w3
            + 293.0 * v7 * w3
            - 756.0 * ca2 * v7 * w3
            - 661.0 * ca4 * v7 * w3
            - 58.0 * v8 * w3
            + 288.0 * ca2 * v8 * w3
            - 1094.0 * ca4 * v8 * w3
            - 24.0 * ca2 * v9 * w3
            + 1104.0 * ca4 * v9 * w3
            - 256.0 * ca4 * v10 * w3
            - 32.0 * ca4 * v11 * w3
            + 90.0 * v2 * w4
            - 48.0 * ca2 * v2 * w4
            - 42.0 * ca4 * v2 * w4
            - 366.0 * v3 * w4
            + 54.0 * ca2 * v3 * w4
            + 348.0 * ca4 * v3 * w4
            + 624.0 * v4 * w4
            + 124.0 * ca2 * v4 * w4
            - 908.0 * ca4 * v4 * w4
            - 658.0 * v5 * w4
            - 126.0 * ca2 * v5 * w4
            + 1404.0 * ca4 * v5 * w4
            + 501.0 * v6 * w4
            - 96.0 * ca2 * v6 * w4
            - 1441.0 * ca4 * v6 * w4
            - 222.0 * v7 * w4
            - 42.0 * ca2 * v7 * w4
            + 4.0 * ca4 * v7 * w4
            + 21.0 * v8 * w4
            + 352.0 * ca2 * v8 * w4
            + 1919.0 * ca4 * v8 * w4
            + 8.0 * v9 * w4
            - 254.0 * ca2 * v9 * w4
            - 1646.0 * ca4 * v9 * w4
            + 2.0 * v10 * w4
            + 36.0 * ca2 * v10 * w4
            + 202.0 * ca4 * v10 * w4
            + 160.0 * ca4 * v11 * w4
            - 48.0 * v2 * w5
            + 24.0 * ca2 * v2 * w5
            + 24.0 * ca4 * v2 * w5
            + 192.0 * v3 * w5
            - 12.0 * ca2 * v3 * w5
            - 204.0 * ca4 * v3 * w5
            - 177.0 * v4 * w5
            - 150.0 * ca2 * v4 * w5
            + 471.0 * ca4 * v4 * w5
            - 113.0 * v5 * w5
            + 30.0 * ca2 * v5 * w5
            - 397.0 * ca4 * v5 * w5
            + 262.0 * v6 * w5
            + 244.0 * ca2 * v6 * w5
            + 126.0 * ca4 * v6 * w5
            - 171.0 * v7 * w5
            + 76.0 * ca2 * v7 * w5
            + 771.0 * ca4 * v7 * w5
            + 57.0 * v8 * w5
            - 466.0 * ca2 * v8 * w5
            - 2183.0 * ca4 * v8 * w5
            + 12.0 * v9 * w5
            + 262.0 * ca2 * v9 * w5
            + 1642.0 * ca4 * v9 * w5
            - 14.0 * v10 * w5
            + 20.0 * ca2 * v10 * w5
            + 138.0 * ca4 * v10 * w5
            - 12.0 * ca2 * v11 * w5
            - 404.0 * ca4 * v11 * w5
            - 138.0 * v4 * w6
            + 72.0 * ca2 * v4 * w6
            + 66.0 * ca4 * v4 * w6
            + 420.0 * v5 * w6
            + 6.0 * ca2 * v5 * w6
            - 486.0 * ca4 * v5 * w6
            - 429.0 * v6 * w6
            - 172.0 * ca2 * v6 * w6
            + 941.0 * ca4 * v6 * w6
            + 156.0 * v7 * w6
            - 66.0 * ca2 * v7 * w6
            - 1102.0 * ca4 * v7 * w6
            + 35.0 * v8 * w6
            + 220.0 * ca2 * v8 * w6
            + 1581.0 * ca4 * v8 * w6
            - 76.0 * v9 * w6
            - 28.0 * ca2 * v9 * w6
            - 1016.0 * ca4 * v9 * w6
            + 32.0 * v10 * w6
            - 120.0 * ca2 * v10 * w6
            - 600.0 * ca4 * v10 * w6
            + 24.0 * ca2 * v11 * w6
            + 680.0 * ca4 * v11 * w6
            + 48.0 * v4 * w7
            - 24.0 * ca2 * v4 * w7
            - 24.0 * ca4 * v4 * w7
            - 144.0 * v5 * w7
            - 12.0 * ca2 * v5 * w7
            + 180.0 * ca4 * v5 * w7
            + 73.0 * v6 * w7
            + 126.0 * ca2 * v6 * w7
            - 319.0 * ca4 * v6 * w7
            + 100.0 * v7 * w7
            + 26.0 * ca2 * v7 * w7
            + 102.0 * ca4 * v7 * w7
            - 137.0 * v8 * w7
            - 82.0 * ca2 * v8 * w7
            - 277.0 * ca4 * v8 * w7
            + 92.0 * v9 * w7
            - 10.0 * ca2 * v9 * w7
            + 202.0 * ca4 * v9 * w7
            - 32.0 * v10 * w7
            + 84.0 * ca2 * v10 * w7
            + 828.0 * ca4 * v10 * w7
            - 12.0 * ca2 * v11 * w7
            - 820.0 * ca4 * v11 * w7
            + 62.0 * v6 * w8
            - 32.0 * ca2 * v6 * w8
            - 30.0 * ca4 * v6 * w8
            - 126.0 * v7 * w8
            - 38.0 * ca2 * v7 * w8
            + 192.0 * ca4 * v7 * w8
            + 94.0 * v8 * w8
            + 36.0 * ca2 * v8 * w8
            - 138.0 * ca4 * v8 * w8
            - 44.0 * v9 * w8
            - 10.0 * ca2 * v9 * w8
            + 82.0 * ca4 * v9 * w8
            + 14.0 * v10 * w8
            - 20.0 * ca2 * v10 * w8
            - 618.0 * ca4 * v10 * w8
            + 704.0 * ca4 * v11 * w8
            - 16.0 * v6 * w9
            + 8.0 * ca2 * v6 * w9
            + 8.0 * ca4 * v6 * w9
            + 32.0 * v7 * w9
            + 12.0 * ca2 * v7 * w9
            - 52.0 * ca4 * v7 * w9
            - 22.0 * v8 * w9
            - 16.0 * ca2 * v8 * w9
            + 70.0 * ca4 * v8 * w9
            + 8.0 * v9 * w9
            + 12.0 * ca2 * v9 * w9
            - 60.0 * ca4 * v9 * w9
            - 2.0 * v10 * w9
            + 242.0 * ca4 * v10 * w9
            - 416.0 * ca4 * v11 * w9
            - 32.0 * ca4 * v10 * w10
            + 160.0 * ca4 * v11 * w10
            - 32.0 * ca4 * v11 * w11))
        / (ca * (1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w).powi(3) * (1.0 - v + v * w).powi(3));

    let part8 = -(2.0
        * cf
        * lv
        * (8.0 * ca2 - 32.0 * ca4 - 48.0 * ca2 * v + 232.0 * ca4 * v + 128.0 * ca2 * v2
            - 768.0 * ca4 * v2
            - 192.0 * ca2 * v3
            + 1504.0 * ca4 * v3
            + 168.0 * ca2 * v4
            - 1888.0 * ca4 * v4
            - 80.0 * ca2 * v5
            + 1544.0 * ca4 * v5
            + 16.0 * ca2 * v6
            - 800.0 * ca4 * v6
            + 240.0 * ca4 * v7
            - 32.0 * ca4 * v8
            - 3.0 * w
            + 14.0 * ca2 * w
            - 11.0 * ca4 * w
            + 17.0 * v * w
            - 68.0 * ca2 * v * w
            + 3.0 * ca4 * v * w
            - 42.0 * v2 * w
            + 162.0 * ca2 * v2 * w
            + 176.0 * ca4 * v2 * w
            + 60.0 * v3 * w
            - 268.0 * ca2 * v3 * w
            - 496.0 * ca4 * v3 * w
            - 55.0 * v4 * w
            + 354.0 * ca2 * v4 * w
            + 341.0 * ca4 * v4 * w
            + 33.0 * v5 * w
            - 348.0 * ca2 * v5 * w
            + 651.0 * ca4 * v5 * w
            - 12.0 * v6 * w
            + 206.0 * ca2 * v6 * w
            - 1530.0 * ca4 * v6 * w
            + 2.0 * v7 * w
            - 52.0 * ca2 * v7 * w
            + 1330.0 * ca4 * v7 * w
            - 560.0 * ca4 * v8 * w
            + 96.0 * ca4 * v9 * w
            + 2.0 * w2
            - 2.0 * ca4 * w2
            - 10.0 * v * w2
            - 8.0 * ca2 * v * w2
            + 26.0 * ca4 * v * w2
            + 14.0 * v2 * w2
            + 22.0 * ca2 * v2 * w2
            - 76.0 * ca4 * v2 * w2
            + 12.0 * v3 * w2
            + 12.0 * ca2 * v3 * w2
            - 112.0 * ca4 * v3 * w2
            - 68.0 * v4 * w2
            - 96.0 * ca2 * v4 * w2
            + 1192.0 * ca4 * v4 * w2
            + 110.0 * v5 * w2
            + 100.0 * ca2 * v5 * w2
            - 3066.0 * ca4 * v5 * w2
            - 98.0 * v6 * w2
            + 42.0 * ca2 * v6 * w2
            + 3800.0 * ca4 * v6 * w2
            + 48.0 * v7 * w2
            - 136.0 * ca2 * v7 * w2
            - 2256.0 * ca4 * v7 * w2
            - 10.0 * v8 * w2
            + 64.0 * ca2 * v8 * w2
            + 334.0 * ca4 * v8 * w2
            + 256.0 * ca4 * v9 * w2
            - 96.0 * ca4 * v10 * w2
            - 2.0 * w3
            + 2.0 * ca4 * w3
            + 10.0 * v * w3
            + 8.0 * ca2 * v * w3
            - 26.0 * ca4 * v * w3
            - 7.0 * v2 * w3
            - 74.0 * ca2 * v2 * w3
            + 137.0 * ca4 * v2 * w3
            - 41.0 * v3 * w3
            + 218.0 * ca2 * v3 * w3
            - 337.0 * ca4 * v3 * w3
            + 120.0 * v4 * w3
            - 444.0 * ca2 * v4 * w3
            + 300.0 * ca4 * v4 * w3
            - 166.0 * v5 * w3
            + 716.0 * ca2 * v5 * w3
            + 322.0 * ca4 * v5 * w3
            + 129.0 * v6 * w3
            - 818.0 * ca2 * v6 * w3
            - 471.0 * ca4 * v6 * w3
            - 37.0 * v7 * w3
            + 574.0 * ca2 * v7 * w3
            - 889.0 * ca4 * v7 * w3
            - 16.0 * v8 * w3
            - 160.0 * ca2 * v8 * w3
            + 1896.0 * ca4 * v8 * w3
            + 10.0 * v9 * w3
            - 20.0 * ca2 * v9 * w3
            - 1174.0 * ca4 * v9 * w3
            + 208.0 * ca4 * v10 * w3
            + 32.0 * ca4 * v11 * w3
            - 12.0 * v2 * w4
            + 12.0 * ca4 * v2 * w4
            + 48.0 * v3 * w4
            + 48.0 * ca2 * v3 * w4
            - 144.0 * ca4 * v3 * w4
            - 70.0 * v4 * w4
            - 144.0 * ca2 * v4 * w4
            + 550.0 * ca4 * v4 * w4
            + 48.0 * v5 * w4
            + 142.0 * ca2 * v5 * w4
            - 1286.0 * ca4 * v5 * w4
            + 2.0 * v6 * w4
            - 132.0 * ca2 * v6 * w4
            + 1434.0 * ca4 * v6 * w4
            - 62.0 * v7 * w4
            + 234.0 * ca2 * v7 * w4
            + 196.0 * ca4 * v7 * w4
            + 68.0 * v8 * w4
            - 340.0 * ca2 * v8 * w4
            - 1652.0 * ca4 * v8 * w4
            - 18.0 * v9 * w4
            + 216.0 * ca2 * v9 * w4
            + 1010.0 * ca4 * v9 * w4
            - 4.0 * v10 * w4
            - 16.0 * ca2 * v10 * w4
            + 16.0 * ca4 * v10 * w4
            - 144.0 * ca4 * v11 * w4
            + 6.0 * v2 * w5
            - 6.0 * ca4 * v2 * w5
            - 24.0 * v3 * w5
            - 24.0 * ca2 * v3 * w5
            + 72.0 * ca4 * v3 * w5
            + 15.0 * v4 * w5
            + 114.0 * ca2 * v4 * w5
            - 273.0 * ca4 * v4 * w5
            + 37.0 * v5 * w5
            - 148.0 * ca2 * v5 * w5
            + 639.0 * ca4 * v5 * w5
            - 74.0 * v6 * w5
            + 206.0 * ca2 * v6 * w5
            - 772.0 * ca4 * v6 * w5
            + 88.0 * v7 * w5
            - 346.0 * ca2 * v7 * w5
            - 374.0 * ca4 * v7 * w5
            - 60.0 * v8 * w5
            + 386.0 * ca2 * v8 * w5
            + 1546.0 * ca4 * v8 * w5
            - 4.0 * v9 * w5
            - 206.0 * ca2 * v9 * w5
            - 862.0 * ca4 * v9 * w5
            + 16.0 * v10 * w5
            - 30.0 * ca2 * v10 * w5
            - 218.0 * ca4 * v10 * w5
            + 8.0 * ca2 * v11 * w5
            + 296.0 * ca4 * v11 * w5
            + 18.0 * v4 * w6
            - 18.0 * ca4 * v4 * w6
            - 54.0 * v5 * w6
            - 72.0 * ca2 * v5 * w6
            + 198.0 * ca4 * v5 * w6
            + 58.0 * v6 * w6
            + 122.0 * ca2 * v6 * w6
            - 628.0 * ca4 * v6 * w6
            - 42.0 * v7 * w6
            - 36.0 * ca2 * v7 * w6
            + 1486.0 * ca4 * v7 * w6
            + 12.0 * v8 * w6
            - 26.0 * ca2 * v8 * w6
            - 1934.0 * ca4 * v8 * w6
            + 36.0 * v9 * w6
            + 14.0 * ca2 * v9 * w6
            + 926.0 * ca4 * v9 * w6
            - 28.0 * v10 * w6
            + 102.0 * ca2 * v10 * w6
            + 290.0 * ca4 * v10 * w6
            - 16.0 * ca2 * v11 * w6
            - 464.0 * ca4 * v11 * w6
            - 6.0 * v4 * w7
            + 6.0 * ca4 * v4 * w7
            + 18.0 * v5 * w7
            + 24.0 * ca2 * v5 * w7
            - 66.0 * ca4 * v5 * w7
            - 9.0 * v6 * w7
            - 62.0 * ca2 * v6 * w7
            + 191.0 * ca4 * v6 * w7
            - 7.0 * v7 * w7
            + 22.0 * ca2 * v7 * w7
            - 391.0 * ca4 * v7 * w7
            + 20.0 * v8 * w7
            - 12.0 * ca2 * v8 * w7
            + 472.0 * ca4 * v8 * w7
            - 44.0 * v9 * w7
            - 6.0 * ca2 * v9 * w7
            - 30.0 * ca4 * v9 * w7
            + 28.0 * v10 * w7
            - 78.0 * ca2 * v10 * w7
            - 526.0 * ca4 * v10 * w7
            + 8.0 * ca2 * v11 * w7
            + 616.0 * ca4 * v11 * w7
            - 8.0 * v6 * w8
            + 8.0 * ca4 * v6 * w8
            + 16.0 * v7 * w8
            + 32.0 * ca2 * v7 * w8
            - 80.0 * ca4 * v7 * w8
            - 18.0 * v8 * w8
            - 24.0 * ca2 * v8 * w8
            + 186.0 * ca4 * v8 * w8
            + 26.0 * v9 * w8
            + 26.0 * ca2 * v9 * w8
            - 316.0 * ca4 * v9 * w8
            - 16.0 * v10 * w8
            + 30.0 * ca2 * v10 * w8
            + 434.0 * ca4 * v10 * w8
            - 576.0 * ca4 * v11 * w8
            + 2.0 * v6 * w9
            - 2.0 * ca4 * v6 * w9
            - 4.0 * v7 * w9
            - 8.0 * ca2 * v7 * w9
            + 20.0 * ca4 * v7 * w9
            + 4.0 * v8 * w9
            + 8.0 * ca2 * v8 * w9
            - 44.0 * ca4 * v8 * w9
            - 6.0 * v9 * w9
            - 8.0 * ca2 * v9 * w9
            + 70.0 * ca4 * v9 * w9
            + 4.0 * v10 * w9
            - 8.0 * ca2 * v10 * w9
            - 108.0 * ca4 * v10 * w9
            + 352.0 * ca4 * v11 * w9
            - 144.0 * ca4 * v11 * w10
            + 32.0 * ca4 * v11 * w11))
        / (ca * (1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w).powi(3) * (1.0 - v + v * w).powi(3));

    let part9 = -(2.0
        * cf
        * l1w
        * (8.0 * ca2 - 24.0 * ca4 - 48.0 * ca2 * v + 176.0 * ca4 * v + 128.0 * ca2 * v2
            - 592.0 * ca4 * v2
            - 192.0 * ca2 * v3
            + 1184.0 * ca4 * v3
            + 168.0 * ca2 * v4
            - 1528.0 * ca4 * v4
            - 80.0 * ca2 * v5
            + 1296.0 * ca4 * v5
            + 16.0 * ca2 * v6
            - 704.0 * ca4 * v6
            + 224.0 * ca4 * v7
            - 32.0 * ca4 * v8
            - 3.0 * w
            + 14.0 * ca2 * w
            - 11.0 * ca4 * w
            + 19.0 * v * w
            - 66.0 * ca2 * v * w
            + 21.0 * ca4 * v * w
            - 50.0 * v2 * w
            + 154.0 * ca2 * v2 * w
            + 72.0 * ca4 * v2 * w
            + 74.0 * v3 * w
            - 254.0 * ca2 * v3 * w
            - 254.0 * ca4 * v3 * w
            - 71.0 * v4 * w
            + 338.0 * ca2 * v4 * w
            + 85.0 * ca4 * v4 * w
            + 47.0 * v5 * w
            - 334.0 * ca2 * v5 * w
            + 705.0 * ca4 * v5 * w
            - 20.0 * v6 * w
            + 198.0 * ca2 * v6 * w
            - 1394.0 * ca4 * v6 * w
            + 4.0 * v7 * w
            - 50.0 * ca2 * v7 * w
            + 1208.0 * ca4 * v7 * w
            - 528.0 * ca4 * v8 * w
            + 96.0 * ca4 * v9 * w
            + 2.0 * w2
            - 2.0 * ca4 * w2
            - 10.0 * v * w2
            - 8.0 * ca2 * v * w2
            + 14.0 * ca4 * v * w2
            + 14.0 * v2 * w2
            + 26.0 * ca2 * v2 * w2
            - 18.0 * ca4 * v2 * w2
            + 14.0 * v3 * w2
            - 2.0 * ca2 * v3 * w2
            - 188.0 * ca4 * v3 * w2
            - 70.0 * v4 * w2
            - 70.0 * ca2 * v4 * w2
            + 1058.0 * ca4 * v4 * w2
            + 110.0 * v5 * w2
            + 68.0 * ca2 * v5 * w2
            - 2494.0 * ca4 * v5 * w2
            - 106.0 * v6 * w2
            + 62.0 * ca2 * v6 * w2
            + 3030.0 * ca4 * v6 * w2
            + 62.0 * v7 * w2
            - 138.0 * ca2 * v7 * w2
            - 1780.0 * ca4 * v7 * w2
            - 16.0 * v8 * w2
            + 62.0 * ca2 * v8 * w2
            + 220.0 * ca4 * v8 * w2
            + 256.0 * ca4 * v9 * w2
            - 96.0 * ca4 * v10 * w2
            - 2.0 * w3
            + 2.0 * ca4 * w3
            + 10.0 * v * w3
            + 8.0 * ca2 * v * w3
            - 26.0 * ca4 * v * w3
            - 7.0 * v2 * w3
            - 74.0 * ca2 * v2 * w3
            + 137.0 * ca4 * v2 * w3
            - 45.0 * v3 * w3
            + 218.0 * ca2 * v3 * w3
            - 385.0 * ca4 * v3 * w3
            + 130.0 * v4 * w3
            - 446.0 * ca2 * v4 * w3
            + 592.0 * ca4 * v4 * w3
            - 184.0 * v5 * w3
            + 722.0 * ca2 * v5 * w3
            - 412.0 * ca4 * v5 * w3
            + 169.0 * v6 * w3
            - 810.0 * ca2 * v6 * w3
            + 409.0 * ca4 * v6 * w3
            - 77.0 * v7 * w3
            + 546.0 * ca2 * v7 * w3
            - 1309.0 * ca4 * v7 * w3
            - 10.0 * v8 * w3
            - 142.0 * ca2 * v8 * w3
            + 1828.0 * ca4 * v8 * w3
            + 16.0 * v9 * w3
            - 22.0 * ca2 * v9 * w3
            - 1044.0 * ca4 * v9 * w3
            + 176.0 * ca4 * v10 * w3
            + 32.0 * ca4 * v11 * w3
            - 12.0 * v2 * w4
            + 12.0 * ca4 * v2 * w4
            + 48.0 * v3 * w4
            + 48.0 * ca2 * v3 * w4
            - 124.0 * ca4 * v3 * w4
            - 70.0 * v4 * w4
            - 148.0 * ca2 * v4 * w4
            + 436.0 * ca4 * v4 * w4
            + 54.0 * v5 * w4
            + 156.0 * ca2 * v5 * w4
            - 1000.0 * ca4 * v5 * w4
            - 28.0 * v6 * w4
            - 174.0 * ca2 * v6 * w4
            + 1210.0 * ca4 * v6 * w4
            - 42.0 * v7 * w4
            + 302.0 * ca2 * v7 * w4
            - 90.0 * ca4 * v7 * w4
            + 96.0 * v8 * w4
            - 380.0 * ca2 * v8 * w4
            - 1004.0 * ca4 * v8 * w4
            - 40.0 * v9 * w4
            + 218.0 * ca2 * v9 * w4
            + 574.0 * ca4 * v9 * w4
            - 6.0 * v10 * w4
            - 14.0 * ca2 * v10 * w4
            + 106.0 * ca4 * v10 * w4
            - 128.0 * ca4 * v11 * w4
            + 6.0 * v2 * w5
            - 6.0 * ca4 * v2 * w5
            - 24.0 * v3 * w5
            - 24.0 * ca2 * v3 * w5
            + 72.0 * ca4 * v3 * w5
            + 15.0 * v4 * w5
            + 114.0 * ca2 * v4 * w5
            - 273.0 * ca4 * v4 * w5
            + 37.0 * v5 * w5
            - 152.0 * ca2 * v5 * w5
            + 623.0 * ca4 * v5 * w5
            - 68.0 * v6 * w5
            + 232.0 * ca2 * v6 * w5
            - 866.0 * ca4 * v6 * w5
            + 106.0 * v7 * w5
            - 400.0 * ca2 * v7 * w5
            + 136.0 * ca4 * v7 * w5
            - 120.0 * v8 * w5
            + 414.0 * ca2 * v8 * w5
            + 800.0 * ca4 * v8 * w5
            + 22.0 * v9 * w5
            - 192.0 * ca2 * v9 * w5
            - 424.0 * ca4 * v9 * w5
            + 26.0 * v10 * w5
            - 40.0 * ca2 * v10 * w5
            - 238.0 * ca4 * v10 * w5
            + 8.0 * ca2 * v11 * w5
            + 216.0 * ca4 * v11 * w5
            + 18.0 * v4 * w6
            - 18.0 * ca4 * v4 * w6
            - 54.0 * v5 * w6
            - 72.0 * ca2 * v5 * w6
            + 210.0 * ca4 * v5 * w6
            + 58.0 * v6 * w6
            + 118.0 * ca2 * v6 * w6
            - 622.0 * ca4 * v6 * w6
            - 60.0 * v7 * w6
            - 22.0 * ca2 * v7 * w6
            + 1262.0 * ca4 * v7 * w6
            + 54.0 * v8 * w6
            - 20.0 * ca2 * v8 * w6
            - 1588.0 * ca4 * v8 * w6
            + 32.0 * v9 * w6
            - 22.0 * ca2 * v9 * w6
            + 800.0 * ca4 * v9 * w6
            - 48.0 * v10 * w6
            + 122.0 * ca2 * v10 * w6
            + 156.0 * ca4 * v10 * w6
            - 16.0 * ca2 * v11 * w6
            - 304.0 * ca4 * v11 * w6
            - 6.0 * v4 * w7
            + 6.0 * ca4 * v4 * w7
            + 18.0 * v5 * w7
            + 24.0 * ca2 * v5 * w7
            - 66.0 * ca4 * v5 * w7
            - 9.0 * v6 * w7
            - 62.0 * ca2 * v6 * w7
            + 191.0 * ca4 * v6 * w7
            - 3.0 * v7 * w7
            + 22.0 * ca2 * v7 * w7
            - 287.0 * ca4 * v7 * w7
            + 10.0 * v8 * w7
            - 26.0 * ca2 * v8 * w7
            + 320.0 * ca4 * v8 * w7
            - 58.0 * v9 * w7
            + 28.0 * ca2 * v9 * w7
            - 48.0 * ca4 * v9 * w7
            + 48.0 * v10 * w7
            - 98.0 * ca2 * v10 * w7
            - 372.0 * ca4 * v10 * w7
            + 8.0 * ca2 * v11 * w7
            + 440.0 * ca4 * v11 * w7
            - 8.0 * v6 * w8
            + 8.0 * ca4 * v6 * w8
            + 16.0 * v7 * w8
            + 32.0 * ca2 * v7 * w8
            - 116.0 * ca4 * v7 * w8
            - 18.0 * v8 * w8
            - 20.0 * ca2 * v8 * w8
            + 244.0 * ca4 * v8 * w8
            + 36.0 * v9 * w8
            + 12.0 * ca2 * v9 * w8
            - 262.0 * ca4 * v9 * w8
            - 26.0 * v10 * w8
            + 40.0 * ca2 * v10 * w8
            + 334.0 * ca4 * v10 * w8
            - 448.0 * ca4 * v11 * w8
            + 2.0 * v6 * w9
            - 2.0 * ca4 * v6 * w9
            - 4.0 * v7 * w9
            - 8.0 * ca2 * v7 * w9
            + 20.0 * ca4 * v7 * w9
            + 4.0 * v8 * w9
            + 8.0 * ca2 * v8 * w9
            - 44.0 * ca4 * v8 * w9
            - 8.0 * v9 * w9
            - 6.0 * ca2 * v9 * w9
            + 12.0 * ca4 * v9 * w9
            + 6.0 * v10 * w9
            - 10.0 * ca2 * v10 * w9
            - 50.0 * ca4 * v10 * w9
            + 288.0 * ca4 * v11 * w9
            + 16.0 * ca4 * v9 * w10
            - 16.0 * ca4 * v10 * w10
            - 128.0 * ca4 * v11 * w10
            + 32.0 * ca4 * v11 * w11))
        / (ca * (1.0 - v).powi(2) * v2 * w2 * (1.0 - v * w).powi(3) * (1.0 - v + v * w).powi(3));

    part1 + part2 + part3 + part4 + part5 + part6 + part7 + part8 + part9
}

/// `STRU(XUHA,...,XGPROA,XUHB,...,XGPROB,XDUP,...,XDGP,GPPV,GPPC)`: combines
/// parton densities of hadrons A/B with fragmentation functions into the
/// flavor-summed weight for each of the 16 channels. `GPPV` uses A as the
/// "unintegrated" (v-side) hadron and B as the "collinear" (w-side) one;
/// `GPPC` swaps A and B (Fortran's convention for the two `DPLUS` terms).
///
/// Bottom-quark densities (`PartonDensities::bottom`) are never referenced
/// here, matching the Fortran subroutine's argument list (`XCHA`/`XCHB` are
/// its last quark flavor; no `XBHA`/`XBHB` exist).
#[must_use]
pub fn stru(
    a: &crate::pdf_ff::PartonDensities,
    b: &crate::pdf_ff::PartonDensities,
    ff: &crate::pdf_ff::FragmentationFunctions,
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
