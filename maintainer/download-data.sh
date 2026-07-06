#!/bin/bash

set -eou pipefail

NEOPDF_SETS=(
  NNPDF40_nnlo_as_01180
  NNPDFpol11_100
  NNFF10_PIsum_lo
)

PINEAPPL_GRIDS=(
  SIHP-PP-POLARIZED-STAR-2006-LO
  SIHP-PP-UNPOLARIZED-STAR-2006-LO
)

# Store the data in the root of the repository
cd ..
test -d navis-cli/tests/fixtures/ || mkdir -p navis-cli/tests/fixtures/

# Download PineAPPL grids
for grid in "${PINEAPPL_GRIDS[@]}"; do
  wget --no-verbose --no-clobber -P navis-cli/tests/fixtures/ "https://data.nnpdf.science/neopdf/data/${grid}.pineappl.lz4"
done

# Download NeoPDF sets
for lha in "${NEOPDF_SETS[@]}"; do
  # TODO: edit info file fot the FF set
  curl "https://lhapdfsets.web.cern.ch/current/${lha}.tar.gz" | tar xzf - --no-same-owner -C navis-cli/tests/fixtures/
done

# Fix FF set metadata: fragfn -> timelike
sed -i 's/SetType: fragfn/SetType: timelike/g' navis-cli/tests/fixtures/NNFF10_PIsum_lo/NNFF10_PIsum_lo.info
