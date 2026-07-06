<h1 align="center">ηaνis</h1>

<p align="justify">
  <b>ηaνis</b> is a code that computes both polarized and unpolarized single-inclusive
  hadron production in proton-proton collision up to next-to-leading order (NLO) in
  QCD and dump the results in the form of
  <a href="https://github.com/NNPDF/pineappl">PineAPPL</a>
  fast interpolation grids. It is based on the following papers:
  <a href="https://arxiv.org/abs/hep-ph/0210442">[arXiv:0210442]</a> and
  <a href="https://arxiv.org/abs/hep-ph/0211007">[arXiv:0211007]</a>.
</p>

## Installation

To compile the code, you only need [cargo](https://doc.rust-lang.org/cargo/) and then
run the following command:

```sh
cargo install --path navis-cli
```

This will install two binaries `navis-pol` and `navis-unpol` that can be used to run
a YAML card containing the input settings. For example:

```sh
navis-unpol example.yml
```

## Citation

If you use **ηaνis**, please cite the following:

<table>
  <tr>
  <td valign="middle"><b>MCHEP</b> for the MC integration</td>
  <td valign="middle">
    <a href="https://doi.org/10.5281/zenodo.21206363"><img
          alt="Crates.io"
          src="https://zenodo.org/badge/DOI/10.5281/zenodo.21206363.svg?style=for-the-badge&logo=zenodo"
          height="20"
    /></a>
  </td>
  </tr>
  <tr>
  <td valign="middle"><b>NeoPDF</b> for the PDF/FF interpolation</td>
  <td valign="middle">
    <a href="https://doi.org/10.5281/zenodo.17286769"><img
          alt="Crates.io"
          src="https://zenodo.org/badge/DOI/10.5281/zenodo.17286769.svg?style=for-the-badge&logo=zenodo"
          height="20"
    /></a>
    <a href="https://arxiv.org/abs/2606.17134"><img
          alt="Crates.io"
          src="http://img.shields.io/badge/hep.ph-2606.17134-B31B1B.svg?logo=arxiv&logoColor=red"
          height="25"
    /></a>
  </td>
  </tr>
  <tr>
  <td valign="middle"><b>PineAPPL</b> for the fast interpolation grid</td>
  <td valign="middle">
    <a href="https://doi.org/10.5281/zenodo.15635174"><img
          alt="Crates.io"
          src="https://zenodo.org/badge/DOI/10.5281/zenodo.15635174.svg?style=for-the-badge&logo=zenodo"
          height="20"
    /></a>
    <a href="https://arxiv.org/abs/2510.05079"><img
          alt="Crates.io"
          src="http://img.shields.io/badge/hep.ph-2510.05079-B31B1B.svg?logo=arxiv&logoColor=red"
          height="25"
    /></a>
  </td>
  </tr>
</table>
