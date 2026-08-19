//! YAML runcard parsing.

use serde::Deserialize;
use std::path::Path;

fn default_zero_i32() -> i32 {
    0
}

fn default_zero_f64() -> f64 {
    0.0
}

fn default_empty_string() -> String {
    String::new()
}

fn default_true() -> bool {
    true
}

/// Mirrors the keys read out of `runcards/template.yml` / `runcards/example.yml`.
///
/// Both the unpolarized and polarized binaries share this same runcard shape;
/// each one only reads the PDF key relevant to it (`unpolarized_pdf` or
/// `polarized_pdf`).
#[derive(Debug, Clone, Deserialize)]
pub struct RunCard {
    pub experiment: String,
    #[serde(default = "default_empty_string")]
    pub expname_suffix: String,

    pub ih1: i32,
    pub ih2: i32,
    pub ih3_charge: i32,

    #[serde(default = "default_empty_string")]
    pub unpolarized_pdf: String,
    #[serde(default = "default_empty_string")]
    pub polarized_pdf: String,
    pub fragmentation_pdf: String,

    #[serde(default = "default_zero_i32")]
    pub irep_unpol: i32,
    #[serde(default = "default_zero_i32")]
    pub irep_pol: i32,
    pub irep_frag: i32,

    /// Cross-section type: 1 dsig/dy/dpt2, 2 E dsig/d3p, 3 dsig/dy/dpt, 4 pt^3 dsig/dy/dpt.
    pub isigm: i32,
    /// Perturbative order: 0 LO, 1 NLO.
    pub iord: i32,
    /// Integrate over pT: 0 no, 1 yes.
    pub ipt: i32,
    /// Integrate over rapidity: 0 no, 1 yes.
    pub iy: i32,

    pub nf: f64,

    pub iter_max: i32,
    pub ncalls_vegas: i32,

    pub sqs: f64,

    pub scfac: f64,
    pub scmu: f64,
    pub scfrag: f64,

    /// Virtual-photon virtuality/regulator scale for the QED-correction.
    #[serde(default = "default_zero_f64")]
    pub q2pho: f64,

    pub n_bins: usize,
    pub center_bins: Vec<f64>,

    pub ydo: f64,
    pub yup: f64,
    #[serde(default = "default_zero_f64")]
    pub ydum: f64,

    /// Whether to build and write the PineAPPL grid. When false, only the
    /// Monte Carlo predictions (integral, MC error) are computed and
    /// written to the `.res` file -- skipping all interpolation-grid
    /// filling/merging, which otherwise dominates the runtime. Default:
    /// true.
    #[serde(default = "default_true")]
    pub generate_grids: bool,
}

impl RunCard {
    /// Read and parse a runcard from a YAML file path.
    pub fn read(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_yaml::from_str(&text)?)
    }

    /// Reconstruct bin edges from the bin centers. The first edge is
    /// placed half a step below the first center, and each subsequent
    /// edge mirrors the previous center across the previous edge.
    ///
    /// ```text
    /// bins(1) = centers(1) - 0.5 * (centers(2) - centers(1))
    /// bins(i) = 2 * centers(i-1) - bins(i-1)   for i = 2..=nbins+1
    /// ```
    pub fn bin_edges(&self) -> Vec<f64> {
        let centers = &self.center_bins;
        let n = centers.len();
        let mut edges = vec![0.0; n + 1];
        edges[0] = centers[0] - 0.5 * (centers[1] - centers[0]);
        for i in 1..=n {
            edges[i] = 2.0 * centers[i - 1] - edges[i - 1];
        }
        edges
    }
}
