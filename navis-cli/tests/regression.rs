//! Regression tests: run the `navis-unpol`/`navis-pol` binaries.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::process::Command;

use pineappl::grid::Grid;

const RUNCARD: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/runcards/template.yml"
);

/// Relative tolerance for comparing MC results across runs. VEGAS is seeded
/// deterministically per bin, so runs should be reproducible up to floating
/// point summation-order differences from `rayon`'s parallel reduction --
/// this tolerance is far looser than that noise floor, but far tighter than
/// any physics-affecting change would produce.
const RTOL: f64 = 1.0e-6;

/// Parses the `sihp-pp::MC-RESULT` metadata value: one `pT value error`
/// triple per (non-empty) line, see `navis-cli/src/bin/navis-{unpol,pol}.rs`.
fn parse_mc_result(s: &str) -> Vec<(f64, f64, f64)> {
    s.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let mut fields = line.split_whitespace();
            let pt: f64 = fields.next().unwrap().parse().unwrap();
            let value: f64 = fields.next().unwrap().parse().unwrap();
            let error: f64 = fields.next().unwrap().parse().unwrap();
            (pt, value, error)
        })
        .collect()
}

fn mc_result(grid: &Grid) -> Vec<(f64, f64, f64)> {
    parse_mc_result(
        grid.metadata()
            .get("sihp-pp::MC-RESULT")
            .expect("grid is missing the sihp-pp::MC-RESULT metadata key"),
    )
}

fn assert_close(label: &str, got: f64, want: f64) {
    let diff = (got - want).abs();
    let scale = want.abs().max(1.0e-300);
    assert!(
        diff / scale < RTOL,
        "{label}: {got:e} vs reference {want:e} (relative diff {:e}, tolerance {RTOL:e})",
        diff / scale
    );
}

/// Runs `bin_exe` against the fixture runcard inside a fresh temp
/// directory, reads back `grid_filename` (the deterministic output name for
/// this runcard/binary combination), and returns it alongside the reference
/// grid loaded from `fixtures/`.
fn run_and_load(bin_exe: &str, grid_filename: &str) -> (Grid, Grid) {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");

    let status = Command::new(bin_exe)
        .arg(RUNCARD)
        .current_dir(tmp.path())
        .status()
        .unwrap_or_else(|e| panic!("failed to run {bin_exe}: {e}"));
    assert!(status.success(), "{bin_exe} exited with {status}");

    let produced_path = tmp.path().join(grid_filename);
    let produced = Grid::read(BufReader::new(File::open(&produced_path).unwrap_or_else(
        |e| panic!("missing output grid {}: {e}", produced_path.display()),
    )))
    .expect("failed to read freshly produced grid");

    let reference_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(grid_filename);
    let reference = Grid::read(BufReader::new(File::open(&reference_path).unwrap_or_else(
        |e| panic!("missing reference grid {}: {e}", reference_path.display()),
    )))
    .expect("failed to read reference grid");

    (produced, reference)
}

#[test]
fn unpolarized_lo_matches_reference() {
    let (produced, reference) = run_and_load(
        env!("CARGO_BIN_EXE_navis-unpol"),
        "SIHP-PP-UNPOLARIZED-STAR-2006-LO.pineappl.lz4",
    );

    let got = mc_result(&produced);
    let want = mc_result(&reference);
    assert_eq!(got.len(), want.len(), "bin count changed");

    for ((got_pt, got_val, got_err), (want_pt, want_val, want_err)) in got.iter().zip(&want) {
        assert_close("pT", *got_pt, *want_pt);
        assert_close("cross section", *got_val, *want_val);
        assert_close("MC error", *got_err, *want_err);
    }
}

#[test]
fn polarized_lo_matches_reference() {
    let (produced, reference) = run_and_load(
        env!("CARGO_BIN_EXE_navis-pol"),
        "SIHP-PP-POLARIZED-STAR-2006-LO.pineappl.lz4",
    );

    let got = mc_result(&produced);
    let want = mc_result(&reference);
    assert_eq!(got.len(), want.len(), "bin count changed");

    for ((got_pt, got_val, got_err), (want_pt, want_val, want_err)) in got.iter().zip(&want) {
        assert_close("pT", *got_pt, *want_pt);
        assert_close("cross section", *got_val, *want_val);
        assert_close("MC error", *got_err, *want_err);
    }
}
