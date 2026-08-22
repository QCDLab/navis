//! PDF and fragmentation-function interface.

use neopdf::pdf::PDF;

/// Unpolarized parton densities at a given `(x, Q^2)`, x times the
/// distribution.
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
    pub photon: f64,
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
    pub c: f64,
    pub cb: f64,
    pub g: f64,
    pub photon: f64,
}

/// Bundles the unpolarized PDF and fragmentation-function sets used by one
/// run (loaded once, reused for every phase-space point).
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

    /// `STRUCI(X,Q2,ITAR,...)`: `itar` selects the target hadron — `0`
    /// proton (unchanged), `1` antiproton (swap quark/antiquark), `2`
    /// isoscalar nuclear target (average up/down with their antiquarks).
    #[must_use]
    pub fn struci(&self, x: f64, q2: f64, itar: i32) -> PartonDensities {
        let mut up = self.pdf.xfxq2(2, &[x, q2]);
        let mut down = self.pdf.xfxq2(1, &[x, q2]);
        let mut upb = self.pdf.xfxq2(-2, &[x, q2]);
        let mut downb = self.pdf.xfxq2(-1, &[x, q2]);
        let strange = self.pdf.xfxq2(3, &[x, q2]);
        let charm = self.pdf.xfxq2(4, &[x, q2]);
        let bottom = self.pdf.xfxq2(5, &[x, q2]);
        let gluon = self.pdf.xfxq2(0, &[x, q2]);
        let photon = self.pdf.xfxq2(22, &[x, q2]);

        if itar == 1 {
            std::mem::swap(&mut up, &mut upb);
            std::mem::swap(&mut down, &mut downb);
        } else if itar == 2 {
            let udn = 0.5 * (up + down);
            let udbn = 0.5 * (upb + downb);
            up = udn;
            down = udn;
            upb = udbn;
            downb = udbn;
        }

        PartonDensities {
            up,
            upb,
            down,
            downb,
            strange,
            charm,
            bottom,
            gluon,
            photon,
        }
    }

    /// `STRUCF(Z,Q2,...)`. `Q2` is floored at `0.45` GeV^2.
    #[must_use]
    pub fn strucf(&self, z: f64, q2: f64) -> FragmentationFunctions {
        const ZCUT: f64 = 0.0;
        let q2 = q2.max(0.45);

        if z < ZCUT {
            return FragmentationFunctions {
                u: 0.0,
                ub: 0.0,
                d: 0.0,
                db: 0.0,
                s: 0.0,
                sb: 0.0,
                c: 0.0,
                cb: 0.0,
                g: 0.0,
                photon: 0.0,
            };
        }

        let u = self.ff.xfxq2(2, &[z, q2]);
        let d = self.ff.xfxq2(1, &[z, q2]);
        let s = self.ff.xfxq2(3, &[z, q2]);
        let ub = self.ff.xfxq2(-2, &[z, q2]);
        let db = self.ff.xfxq2(-1, &[z, q2]);
        let sb = self.ff.xfxq2(-3, &[z, q2]);
        let gl = self.ff.xfxq2(0, &[z, q2]);
        let pho = self.ff.xfxq2(22, &[z, q2]);

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
            photon: pho / z,
        }
    }
}
