//! CLI driver for unpolarized single-inclusive hadron production,

use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use pineappl::convolutions::{Conv, ConvType};
use pineappl::grid::Grid;
use rayon::prelude::*;

use navis_core::constants::Params;
use navis_core::grid_setup::{GridConfig, PertOrder};
use navis_core::kinematics::{BinInputs, CrossSectionType, RunParams};
use navis_core::runcard::RunCard;
use navis_core::vegas::{vegas, vegas_value_only, VegasOutcome};
use navis_unpol::integrand::{dplus, Targets};
use navis_unpol::pdfs::PdfFf;

#[derive(Parser)]
#[command(about = "Unpolarized single-inclusive hadron production cross sections")]
struct Args {
    /// Path to the YAML runcard.
    runcard: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let card = RunCard::read(&args.runcard)
        .with_context(|| format!("reading runcard {}", args.runcard.display()))?;

    let order = PertOrder::from_iord(card.iord)?;
    let pto_suffix = order.suffix();

    let params = Params::new(card.nf);
    let run = RunParams {
        sqs: card.sqs,
        s: card.sqs * card.sqs,
        scfac: card.scfac,
        scmu: card.scmu,
        scfrag: card.scfrag,
        isigm: CrossSectionType::from_isigm(card.isigm)?,
        hc2: params.hc2,
        pi: params.pi,
    };
    let targets = Targets {
        ih1: card.ih1,
        ih2: card.ih2,
    };

    let pdf_ff = PdfFf::load(
        &card.unpolarized_pdf,
        card.irep_unpol as usize,
        &card.fragmentation_pdf,
        card.irep_frag as usize,
    );

    let convolutions = vec![
        Conv::new(ConvType::UnpolPDF, 2212),
        Conv::new(ConvType::UnpolPDF, 2212),
        Conv::new(ConvType::UnpolFF, 211),
    ];
    let edges = card.bin_edges();
    let grid_config = card
        .generate_grids
        .then(|| GridConfig::new(order, convolutions, &edges));

    let n_iter = card.iter_max as usize;
    let n_eval = card.ncalls_vegas as usize;

    println!();
    println!("Predictions:");
    println!("{:>14}{:>14}{:>14}", "x", "dsig/dx", "MC Error");

    let results: Vec<(Option<Grid>, VegasOutcome)> = (0..card.n_bins)
        .into_par_iter()
        .map(|i| {
            let ptdo = edges[i];
            let ptup = edges[i + 1];
            let bin = BinInputs {
                pt0: card.center_bins[i],
                eta0: card.ydum,
                ptdo,
                ptup,
                ydo: card.ydo,
                yup: card.yup,
                integrate_y: card.iy != 0,
                integrate_pt: card.ipt != 0,
            };
            let bin_width = ptup - ptdo;
            let seed = 0x5eed_1234_u64 ^ (i as u64);

            match &grid_config {
                Some(grid_config) => {
                    let mut grid = grid_config
                        .new_grid_for_bin(ptdo, ptup)
                        .expect("failed to build per-bin grid");

                    let outcome = vegas(
                        |xx, fill_weight| {
                            dplus(
                                xx,
                                fill_weight,
                                &bin,
                                &run,
                                &params,
                                order,
                                targets,
                                &pdf_ff,
                                bin_width,
                            )
                        },
                        n_iter,
                        n_eval,
                        seed,
                        &mut grid,
                    );

                    (Some(grid), outcome)
                }
                None => {
                    let outcome = vegas_value_only(
                        |xx| {
                            dplus(
                                xx, 1.0, &bin, &run, &params, order, targets, &pdf_ff, bin_width,
                            )
                            .0
                        },
                        n_iter,
                        n_eval,
                        seed,
                    );

                    (None, outcome)
                }
            }
        })
        .collect();

    let mut result_string = String::new();
    let mut merged: Option<Grid> = None;
    for (i, (grid, outcome)) in results.into_iter().enumerate() {
        let pt = card.center_bins[i];
        println!(
            "{:14.6e}{:14.6e}{:14.6e}",
            pt, outcome.integral, outcome.std_dev
        );
        result_string.push('\n');
        result_string.push_str(&format!(
            " {:12.6e} {:12.6e} {:12.6e}",
            pt, outcome.integral, outcome.std_dev
        ));

        if let Some(grid) = grid {
            merged = Some(match merged {
                None => grid,
                Some(mut acc) => {
                    acc.merge(grid)?;
                    acc
                }
            });
        }
    }

    let outfile = format!("SIHP-PP-UNPOLARIZED-{}{}", card.experiment, pto_suffix);
    std::fs::create_dir_all("results")?;
    std::fs::write(
        format!("results/{}_{}.res", card.experiment, outfile),
        result_string.clone(),
    )?;

    let Some(mut grid) = merged else {
        println!("generate_grids is false: skipped PineAPPL grid generation.");
        return Ok(());
    };
    grid.optimize();

    let meta = grid.metadata_mut();
    meta.insert("x1_label".into(), "pT".into());
    meta.insert("x1_label_tex".into(), "$p_{T}$".into());
    meta.insert("x1_unit".into(), "GeV".into());
    meta.insert("y_label".into(), "dsig/dpT (unpol)".into());
    meta.insert(
        "y_label_tex".into(),
        r"$\frac{\mathrm{d}\sigma}{\mathrm{d}p_{T}}$".into(),
    );
    meta.insert("y_unit".into(), "pb/GeV".into());
    meta.insert("type_hadron_1".into(), card.ih1.to_string());
    meta.insert("type_hadron_2".into(), card.ih2.to_string());
    meta.insert("charge_hadron_3".into(), card.ih3_charge.to_string());
    meta.insert("convolution_type_hadron_1".into(), "UnpolPDF".into());
    meta.insert("convolution_type_hadron_2".into(), "UnpolPDF".into());
    meta.insert("convolution_type_hadron_3".into(), "UnpolFF".into());
    meta.insert("pid_hadron_1".into(), "2212".into());
    meta.insert("pid_hadron_2".into(), "2212".into());
    meta.insert("pid_hadron_3".into(), "211".into());
    meta.insert(
        "sihp-pp::unpolarized_pdf".into(),
        card.unpolarized_pdf.clone(),
    );
    meta.insert(
        "sihp-pp::fragmentation_pdf".into(),
        card.fragmentation_pdf.clone(),
    );
    meta.insert("sihp-pp::irep_unpol".into(), card.irep_unpol.to_string());
    meta.insert("sihp-pp::irep_ff".into(), card.irep_frag.to_string());
    meta.insert("sihp-pp::iter_max".into(), card.iter_max.to_string());
    meta.insert(
        "sihp-pp::ncalls_vegas".into(),
        card.ncalls_vegas.to_string(),
    );
    meta.insert("sihp-pp::MC-RESULT".into(), result_string);

    let gridname = format!(
        "SIHP-PP-UNPOLARIZED-{}{}{}.pineappl.lz4",
        card.experiment, card.expname_suffix, pto_suffix
    );
    grid.write_lz4(BufWriter::new(File::create(&gridname)?))?;
    println!("Wrote grid to {gridname}");

    Ok(())
}
