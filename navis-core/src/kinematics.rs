//! Phase-space mapping shared by the unpolarized and polarized integrands.

/// Cross-section type requested by the runcard's `isigm` key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossSectionType {
    /// `isigm = 1`: `dsigma/dy/dpt2`.
    DSigDyDpt2,
    /// `isigm = 2`: `E dsigma/d3p`.
    EDSigD3p,
    /// `isigm = 3`: `dsigma/dy/dpt`.
    DSigDyDpt,
    /// `isigm = 4`: `pt^3 dsigma/dy/dpt`.
    Pt3DSigDyDpt,
}

impl CrossSectionType {
    pub fn from_isigm(isigm: i32) -> anyhow::Result<Self> {
        match isigm {
            1 => Ok(Self::DSigDyDpt2),
            2 => Ok(Self::EDSigD3p),
            3 => Ok(Self::DSigDyDpt),
            4 => Ok(Self::Pt3DSigDyDpt),
            other => anyhow::bail!("isigm = {other} not valid, allowed values are 1..=4"),
        }
    }

    /// `FACIN` in `DPLUS`.
    #[must_use]
    pub fn facin(self, pi: f64, pt: f64) -> f64 {
        match self {
            Self::DSigDyDpt2 => pi,
            Self::EDSigD3p => 1.0,
            Self::DSigDyDpt => pi * 2.0 * pt,
            Self::Pt3DSigDyDpt => pi * 2.0 * pt.powi(4),
        }
    }
}

/// Per-bin kinematic inputs, computed once per pT bin (mirrors `PT0`,
/// `ETA0`, `PTDO`, `PTUP`, `YDO`, `YUP`, `IPT`, `IY` common-block state set
/// up in the `hadrive-ms.f` main-program bin loop).
#[derive(Debug, Clone, Copy)]
pub struct BinInputs {
    /// `PT0`: the bin-center pT, used when `integrate_pt` is `false`.
    pub pt0: f64,
    /// `ETA0` (`YDUM` in the runcard): the fixed rapidity, used when
    /// `integrate_y` is `false`.
    pub eta0: f64,
    pub ptdo: f64,
    pub ptup: f64,
    pub ydo: f64,
    pub yup: f64,
    /// `IY != 0`.
    pub integrate_y: bool,
    /// `IPT != 0`.
    pub integrate_pt: bool,
}

/// Run-wide kinematic inputs (mirrors `COMMON /KINVAR/`, `/SCALESF/`, etc).
#[derive(Debug, Clone, Copy)]
pub struct RunParams {
    pub sqs: f64,
    /// `S = SQS**2`.
    pub s: f64,
    pub scfac: f64,
    pub scmu: f64,
    pub scfrag: f64,
    pub isigm: CrossSectionType,
    pub hc2: f64,
    pub pi: f64,
}

/// The result of mapping one VEGAS point through the 5D phase space.
#[derive(Debug, Clone, Copy)]
pub struct PhaseSpace {
    pub pt: f64,
    pub eta: f64,
    /// `GV = 1 - PT/SQS*exp(-ETA)`, needed (together with [`Self::gw`]) by
    /// the `FDEL*`/`FVWPL*`/`FVLO*`/`FRESC*` helper functions, which
    /// recompute `BX1`/`X1` from `GV*GW` the same way `DPLUS` does via its
    /// `COMMON /CONS/` globals.
    pub gv: f64,
    /// `GW = PT^2/S/GV/(1-GV)`.
    pub gw: f64,
    /// NLO momentum fraction of hadron 1.
    pub x1: f64,
    /// NLO momentum fraction of hadron 2.
    pub x2: f64,
    /// Fragmentation momentum fraction.
    pub x3: f64,
    /// Born momentum fraction of hadron 1.
    pub bx1: f64,
    /// Born momentum fraction of hadron 2 (equals `x2`).
    pub bx2: f64,
    pub v: f64,
    pub w: f64,
    pub wmin: f64,
    pub wmax: f64,
    /// `SHD` at Born level: `BX1 * BX2 * S`.
    pub shd_born: f64,
    /// `BXJAC`: Born-level VEGAS Jacobian.
    pub bxjac: f64,
    /// `XJAC`: NLO-level VEGAS Jacobian.
    pub xjac: f64,
    pub q2fac: f64,
    pub q2mu: f64,
    pub q2frag: f64,
    /// `PHASE_SPACE = FACIN * HC2/PI/S * XJETA * XJPT`.
    pub phase_space: f64,
}

impl PhaseSpace {
    /// `SHD` at NLO: `X1 * X2 * S`. Only needed when `iord == 1`.
    #[must_use]
    pub fn shd_nlo(&self, run: &RunParams) -> f64 {
        self.x1 * self.x2 * run.s
    }
}

/// Map a 5D VEGAS point `xx` (`XX(1..5)` in Fortran, 0-indexed here) to a
/// [`PhaseSpace`] point.
#[must_use]
pub fn map_phase_space(
    xx: &[f64; 5],
    bin: &BinInputs,
    run: &RunParams,
    apply_pt_kinematic_cap: bool,
) -> PhaseSpace {
    let (pt, eta, xjeta, xjpt);

    if !bin.integrate_y {
        eta = bin.eta0;
        xjeta = 1.0;
        if bin.integrate_pt {
            pt = bin.ptdo + xx[3] * (bin.ptup - bin.ptdo);
            xjpt = bin.ptup - bin.ptdo;
        } else {
            pt = bin.pt0;
            xjpt = 1.0;
        }
    } else {
        eta = bin.ydo + xx[3] * (bin.yup - bin.ydo);
        xjeta = bin.yup - bin.ydo;
        if bin.integrate_pt {
            let ptup = if apply_pt_kinematic_cap {
                let ptup2 = run.sqs / 2.0 / eta.cosh();
                bin.ptup.min(ptup2)
            } else {
                bin.ptup
            };
            pt = bin.ptdo + xx[4] * (ptup - bin.ptdo);
            xjpt = ptup - bin.ptdo;
        } else {
            pt = bin.pt0;
            xjpt = 1.0;
        }
    }

    let q2fac = run.scfac * pt * pt;
    let q2mu = run.scmu * pt * pt;
    let q2frag = run.scfrag * pt * pt;

    let gv = 1.0 - pt / run.sqs * (-eta).exp();
    let gw = pt * pt / run.s / gv / (1.0 - gv);

    let x3min = 1.0 - gv + gv * gw;
    let x3max = 1.0;
    let x3 = x3min + (x3max - x3min) * xx[2];

    let vmin = gv * gw / x3;
    let vmax = 1.0 - (1.0 - gv) / x3;
    let v = vmin + (vmax - vmin) * xx[1];

    let wmin = gv * gw / x3 / v;
    let wmax = 1.0;
    let w = wmin + (wmax - wmin) * xx[0];

    let x1 = gv * gw / v / w / x3;
    let x2 = (1.0 - gv) / (1.0 - v) / x3;

    let bx1 = gv * gw / v / x3;
    let bx2 = x2;

    let shd_born = bx1 * bx2 * run.s;

    let bxjac = (x3max - x3min) * (vmax - vmin) / bx1 / bx2 / x3.powi(2);
    let xjac = (x3max - x3min) * (vmax - vmin) / x1 / x2 / x3.powi(2);

    let facin = run.isigm.facin(run.pi, pt);
    let phase_space = facin * run.hc2 / run.pi / run.s * xjeta * xjpt;

    PhaseSpace {
        pt,
        eta,
        gv,
        gw,
        x1,
        x2,
        x3,
        bx1,
        bx2,
        v,
        w,
        wmin,
        wmax,
        shd_born,
        bxjac,
        xjac,
        q2fac,
        q2mu,
        q2frag,
        phase_space,
    }
}
