//! Builds the PineAPPL [`Grid`] configuration.

use crate::pdg_channels::sihp_pdg_channels;
use pineappl::boc::{BinsWithFillLimits, Channel, Kinematics, Order, ScaleFuncForm, Scales};
use pineappl::convolutions::Conv;
use pineappl::grid::Grid;
use pineappl::interpolation::{Interp, InterpMeth, Map, ReweightMeth};
use pineappl::pids::PidBasis;

/// Perturbative order selector, mirrors `IORD` in the runcard (0 = LO, 1 = NLO).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PertOrder {
    Lo,
    Nlo,
}

impl PertOrder {
    /// Parse the runcard's `iord` integer.
    pub fn from_iord(iord: i32) -> anyhow::Result<Self> {
        match iord {
            0 => Ok(Self::Lo),
            1 => Ok(Self::Nlo),
            other => anyhow::bail!("PTO: {other} not valid. Allowed [0,1]"),
        }
    }

    #[must_use]
    pub fn suffix(self) -> &'static str {
        match self {
            Self::Lo => "-LO",
            Self::Nlo => "-NLO",
        }
    }

    /// The order index used for the born (LO) grid fills, always 0.
    #[must_use]
    pub const fn born_index() -> usize {
        0
    }

    /// The order index used for the NLO remainder grid fills, only valid
    /// when `self == Nlo`.
    #[must_use]
    pub const fn nlo_index() -> usize {
        1
    }

    #[must_use]
    pub fn is_nlo(self) -> bool {
        self == Self::Nlo
    }

    fn orders(self) -> Vec<Order> {
        match self {
            Self::Lo => vec![Order::new(2, 0, 0, 0, 0)],
            Self::Nlo => vec![Order::new(2, 0, 0, 0, 0), Order::new(3, 0, 0, 0, 0)],
        }
    }
}

/// Everything needed to build a per-bin [`Grid`], computed once per run and
/// reused for every pT bin.
#[derive(Clone)]
pub struct GridConfig {
    channels: Vec<Channel>,
    orders: Vec<Order>,
    convolutions: Vec<Conv>,
    interps: Vec<Interp>,
    kinematics: Vec<Kinematics>,
    scales: Scales,
}

impl GridConfig {
    /// `global_edges` must be the full list of bin edges (length `n_bins +
    /// 1`) across *all* pT bins in the run: the Q2 interpolation range is
    /// derived from the overall pT range.
    #[must_use]
    pub fn new(
        order: PertOrder,
        convolutions: Vec<Conv>,
        global_edges: &[f64],
        include_qed: bool,
    ) -> Self {
        let bin_min = global_edges[0];
        let bin_max = global_edges[global_edges.len() - 1];

        let interps = vec![
            Interp::new(
                0.9 * bin_min * bin_min,
                1.10 * bin_max * bin_max,
                40,
                3,
                ReweightMeth::NoReweight,
                Map::ApplGridH0,
                InterpMeth::Lagrange,
            ),
            Interp::new(
                2e-7,
                1.0,
                50,
                3,
                ReweightMeth::ApplGridX,
                Map::ApplGridF2,
                InterpMeth::Lagrange,
            ),
            Interp::new(
                2e-7,
                1.0,
                50,
                3,
                ReweightMeth::ApplGridX,
                Map::ApplGridF2,
                InterpMeth::Lagrange,
            ),
            Interp::new(
                2e-7,
                1.0,
                50,
                3,
                ReweightMeth::ApplGridX,
                Map::ApplGridF2,
                InterpMeth::Lagrange,
            ),
        ];

        Self {
            channels: sihp_pdg_channels(include_qed),
            orders: order.orders(),
            convolutions,
            interps,
            kinematics: vec![
                Kinematics::Scale(0),
                Kinematics::X(0),
                Kinematics::X(1),
                Kinematics::X(2),
            ],
            scales: Scales {
                ren: ScaleFuncForm::Scale(0),
                fac: ScaleFuncForm::Scale(0),
                frg: ScaleFuncForm::Scale(0),
            },
        }
    }

    /// Build a fresh [`Grid`] containing exactly one bin, `[lo, hi]`. Used
    /// so pT bins can be integrated concurrently, each into its own `Grid`,
    /// then merged with [`pineappl::grid::Grid::merge`].
    pub fn new_grid_for_bin(&self, lo: f64, hi: f64) -> anyhow::Result<Grid> {
        let bwfl = BinsWithFillLimits::from_fill_limits(vec![lo, hi])?;
        Ok(Grid::new(
            bwfl,
            self.orders.clone(),
            self.channels.clone(),
            PidBasis::Pdg,
            self.convolutions.clone(),
            self.interps.clone(),
            self.kinematics.clone(),
            self.scales.clone(),
        ))
    }
}
