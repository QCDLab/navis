//! Shared infrastructure for `navis-unpol` and `navis-pol`: runcard
//! parsing, physics constants, the 16-channel PDG table, PineAPPL grid
//! setup, phase-space kinematics, and the VEGAS integrator.

pub mod constants;
pub mod grid_setup;
pub mod kinematics;
pub mod pdg_channels;
pub mod runcard;
pub mod vegas;
