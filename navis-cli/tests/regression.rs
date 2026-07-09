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

const RTOL: f64 = 1.0e-6;

/// Parses the `sihp-pp::MC-RESULT`.
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

/// Runs `bin_exe` against the fixture runcard.
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

/// Runs `bin_exe` against the fixture runcard with `generate_grids: false`
fn run_without_grid(bin_exe: &str, results_path: &str) -> Vec<(f64, f64, f64)> {
    let tmp = tempfile::tempdir().expect("failed to create temp dir");

    let base_runcard = std::fs::read_to_string(RUNCARD).expect("failed to read fixture runcard");
    let runcard_path = tmp.path().join("runcard.yml");
    std::fs::write(
        &runcard_path,
        format!("{base_runcard}\ngenerate_grids: false\n"),
    )
    .expect("failed to write temp runcard");

    let status = Command::new(bin_exe)
        .arg(&runcard_path)
        .current_dir(tmp.path())
        .status()
        .unwrap_or_else(|e| panic!("failed to run {bin_exe}: {e}"));
    assert!(status.success(), "{bin_exe} exited with {status}");

    let no_grids_written = walkdir_no_pineappl_files(tmp.path());
    assert!(
        no_grids_written,
        "generate_grids: false still produced a .pineappl.lz4 file under {}",
        tmp.path().display()
    );

    let results_file = tmp.path().join(results_path);
    let contents = std::fs::read_to_string(&results_file)
        .unwrap_or_else(|e| panic!("missing predictions file {}: {e}", results_file.display()));
    parse_mc_result(&contents)
}

/// Returns `true` if no `.pineappl.lz4` file exists anywhere under `dir`.
fn walkdir_no_pineappl_files(dir: &Path) -> bool {
    fn visit(dir: &Path, found: &mut bool) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                visit(&path, found);
            } else if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".pineappl.lz4"))
            {
                *found = true;
            }
        }
    }

    let mut found = false;
    visit(dir, &mut found);
    !found
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

/// `generate_grids: false` must skip writing any PineAPPL grid, while the
/// Monte Carlo predictions themselves (same seed, same physics) must be
/// unaffected -- i.e. identical to the `generate_grids: true` reference.
#[test]
fn unpolarized_no_grid_matches_reference() {
    let reference_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join("SIHP-PP-UNPOLARIZED-STAR-2006-LO.pineappl.lz4");
    let reference = Grid::read(BufReader::new(File::open(&reference_path).unwrap_or_else(
        |e| panic!("missing reference grid {}: {e}", reference_path.display()),
    )))
    .expect("failed to read reference grid");

    let got = run_without_grid(
        env!("CARGO_BIN_EXE_navis-unpol"),
        "results/STAR_SIHP-PP-UNPOLARIZED-STAR-LO.res",
    );
    let want = mc_result(&reference);
    assert_eq!(got.len(), want.len(), "bin count changed");

    for ((got_pt, got_val, got_err), (want_pt, want_val, want_err)) in got.iter().zip(&want) {
        assert_close("pT", *got_pt, *want_pt);
        assert_close("cross section", *got_val, *want_val);
        assert_close("MC error", *got_err, *want_err);
    }
}

/// Polarized counterpart of [`unpolarized_no_grid_matches_reference`].
#[test]
fn polarized_no_grid_matches_reference() {
    let reference_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join("SIHP-PP-POLARIZED-STAR-2006-LO.pineappl.lz4");
    let reference = Grid::read(BufReader::new(File::open(&reference_path).unwrap_or_else(
        |e| panic!("missing reference grid {}: {e}", reference_path.display()),
    )))
    .expect("failed to read reference grid");

    let got = run_without_grid(
        env!("CARGO_BIN_EXE_navis-pol"),
        "results/STAR_SIHP-PP-POLARIZED-STAR-LO.res",
    );
    let want = mc_result(&reference);
    assert_eq!(got.len(), want.len(), "bin count changed");

    for ((got_pt, got_val, got_err), (want_pt, want_val, want_err)) in got.iter().zip(&want) {
        assert_close("pT", *got_pt, *want_pt);
        assert_close("cross section", *got_val, *want_val);
        assert_close("MC error", *got_err, *want_err);
    }
}
