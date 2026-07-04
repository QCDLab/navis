//! PDF and fragmentation-function interface.

use neopdf::pdf::PDF;

/// Polarized parton densities at a given `(x, Q^2)`, x times the
/// distribution (`STRUCI`'s output convention).
///
/// `charm`/`bottom` are always zero: the Fortran subroutine hardcodes
/// `CH=0.D0`, `BO=0.D0` rather than reading them off the PDF set (marked
/// `TODO: Check why are these set to ZERO` in the source, but preserved
/// here for fidelity).
#[derive(Debug, Clone, Copy)]
pub struct PartonDensities {
    pub up: f64,
    pub upb: f64,
    pub down: f64,
    pub downb: f64,
    pub strange: f64,
    pub charm: f64,
    pub bottom: f64,
    pub gluon: f64,
}

/// Unpolarized fragmentation functions at a given `(z, Q^2)` (`STRUCF`'s
/// output convention: *not* z times the distribution).
#[derive(Debug, Clone, Copy)]
pub struct FragmentationFunctions {
    pub u: f64,
    pub ub: f64,
    pub d: f64,
    pub db: f64,
    pub s: f64,
    pub sb: f64,
    /// Always zero: `DPC`/`DPCB` are initialized and never overwritten in
    /// `STRUCF`, i.e. charm fragmentation is switched off.
    pub c: f64,
    pub cb: f64,
    pub g: f64,
}

/// Bundles the polarized PDF and unpolarized fragmentation-function sets
/// used by one run (loaded once, reused for every phase-space point).
pub struct PdfFf {
    pdf: PDF,
    ff: PDF,
}

impl PdfFf {
    #[must_use]
    pub fn load(pdf_name: &str, pdf_member: usize, ff_name: &str, ff_member: usize) -> Self {
        Self {
            pdf: PDF::load(pdf_name, pdf_member),
            ff: PDF::load(ff_name, ff_member),
        }
    }

    #[must_use]
    pub fn alphas_q2(&self, q2: f64) -> f64 {
        self.pdf.alphas_q2(q2)
    }

    /// `STRUCI(X,Q2,ITAR,...)`. `itar` is accepted for API symmetry with
    /// the unpolarized crate but is unused, matching the Fortran: the
    /// polarized `STRUCI` never reads its `ITAR` argument (no
    /// target-switching branch exists in `polarized/part-pol-ms.f`).
    #[must_use]
    pub fn struci(&self, x: f64, q2: f64, _itar: i32) -> PartonDensities {
        let up = self.pdf.xfxq2(2, &[x, q2]);
        let down = self.pdf.xfxq2(1, &[x, q2]);
        let upb = self.pdf.xfxq2(-2, &[x, q2]);
        let downb = self.pdf.xfxq2(-1, &[x, q2]);
        let strange = self.pdf.xfxq2(3, &[x, q2]);
        let gluon = self.pdf.xfxq2(0, &[x, q2]);

        PartonDensities {
            up,
            upb,
            down,
            downb,
            strange,
            charm: 0.0,
            bottom: 0.0,
            gluon,
        }
    }

    /// `STRUCF(Z,Q2,...)`. `Q2` is floored at `0.45` GeV^2, matching the
    /// Fortran `IF(Q2.LT.0.45) Q2=0.45D0` fragmentation-scale cutoff.
    ///
    /// Unlike the unpolarized `STRUCF`, there is no `ZCUT` zeroing branch
    /// here at all (confirmed absent from `polarized/part-pol-ms.f`).
    #[must_use]
    pub fn strucf(&self, z: f64, q2: f64) -> FragmentationFunctions {
        let q2 = q2.max(0.45);

        let u = self.ff.xfxq2(2, &[z, q2]);
        let d = self.ff.xfxq2(1, &[z, q2]);
        let s = self.ff.xfxq2(3, &[z, q2]);
        let ub = self.ff.xfxq2(-2, &[z, q2]);
        let db = self.ff.xfxq2(-1, &[z, q2]);
        let sb = self.ff.xfxq2(-3, &[z, q2]);
        let gl = self.ff.xfxq2(0, &[z, q2]);

        FragmentationFunctions {
            u: u / z,
            ub: ub / z,
            d: d / z,
            db: db / z,
            s: s / z,
            sb: sb / z,
            c: 0.0,
            cb: 0.0,
            g: gl / z,
        }
    }
}
