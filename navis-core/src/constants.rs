//! Physics constants shared by the unpolarized and polarized codes.

/// Color factor `C_F = 4/3`.
pub const CF: f64 = 4.0 / 3.0;
/// Number of colors `C_A = N_C = 3`.
pub const CA: f64 = 3.0;
/// Conversion factor `(hbar*c)^2` in GeV^2 pbarn, i.e. `HC2` in the Fortran code.
pub const HC2: f64 = 0.389_429_57e9;

/// Physics parameters derived from the runcard, computed once per run.
#[derive(Debug, Clone, Copy)]
pub struct Params {
    pub cf: f64,
    pub ca: f64,
    pub nf: f64,
    pub pi: f64,
    pub hc2: f64,
    /// `VC = C_A^2 - 1`.
    pub vc: f64,
    /// `GTR = NF / 2`.
    pub gtr: f64,
    /// The 16-entry channel normalization prefactor `CC(J0)`, see [`prefactors`].
    pub cc: [f64; 16],
}

impl Params {
    #[must_use]
    pub fn new(nf: f64) -> Self {
        Self {
            cf: CF,
            ca: CA,
            nf,
            pi: std::f64::consts::PI,
            hc2: HC2,
            vc: CA * CA - 1.0,
            gtr: nf / 2.0,
            cc: prefactors(CA),
        }
    }
}

/// Per-channel color prefactor `CC(J0)`, `J0 = 1..16`, computed in the `DO 2
/// J0=1,16`.
///
/// Channels 15/16 get `8*VC^2`; channels 8,9,10,13,14 get `8*VC*NC`; all
/// others get `8*NC^2`.
#[must_use]
pub fn prefactors(nc: f64) -> [f64; 16] {
    let vc = nc * nc - 1.0;
    let mut cc = [0.0; 16];
    for (j0, entry) in cc.iter_mut().enumerate() {
        let channel = j0 + 1;
        *entry = if channel == 16 || channel == 15 {
            8.0 * vc * vc
        } else if matches!(channel, 14 | 13 | 10 | 9 | 8) {
            8.0 * vc * nc
        } else {
            8.0 * nc * nc
        };
    }
    cc
}
