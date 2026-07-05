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

To compile the code, you only need [cargo](https://doc.rust-lang.org/cargo/) and then run the following command:

```sh
cargo install --path navis-cli
```

This will install two binaries `navis-pol` and `navis-unpol` that can be used to run
a YAML card containing the input settings. For example:

```sh
navis-unpol example.yml
```
