#!/bin/bash

set -euo pipefail

crates=(
    navis-core
    navis-unpol
    navis-pol
    navis-cli
)

main=main
root_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "${root_dir}"

this_branch=$(git rev-parse --abbrev-ref HEAD)

if [[ $# != 1 ]]; then
    echo "No version number given."
    exit 1
fi

version=$1

prerelease=$(echo ${version} | perl -pe 's/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$/\4/')

if [[ $(echo ${version} | grep -oP '^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$') != ${version} ]]; then
    echo "Version string incorrect."
    exit 1
fi

if [[ $(git tag -l v${version}) ]]; then
    echo "Version already exists."
    exit 1
fi

# in branches that are not main we only allow prereleases
if [[ ${this_branch} != ${main} ]] && [[ ${prerelease} == "" ]]; then
    echo "Ordinary releases are only allowed in the '${main}' branch."
    echo "If you really want to make a release from '${this_branch}', consider making a prerelease."
    exit 1
fi

echo ">>> Updating version strings ..."

# modify the version in the main workspace
sed -i \
    -e "/^\[workspace\.package\]/,/^\[/ s:^version = \".*\":version = \"${version}\":" \
    Cargo.toml

# the internal crates are only depended upon via path in
# `[workspace.dependencies]`; `cargo publish` refuses path dependencies that
# don't also carry a version, so pin one here for whichever depends on them
for crate in navis-core navis-pol navis-unpol; do
    sed -i \
        -e "s:${crate} = { path = \"\./${crate}\" }:${crate} = { path = \"./${crate}\", version = \"${version}\" }:" \
        -e "s:${crate} = { path = \"\./${crate}\", version = \"[^\"]*\" }:${crate} = { path = \"./${crate}\", version = \"${version}\" }:" \
        Cargo.toml
done
git add Cargo.toml

echo ">>> Updating Cargo.lock ..."

for crate in "${crates[@]}"; do
    cargo pkgid path+file://$(cd "${crate}" && pwd)
done | xargs printf " -p %s" | xargs cargo update
git add Cargo.lock

echo ">>> Committing and pushing changes ..."

git commit -m "Release v${version}"
git tag -a v${version} -m v${version}
git push --follow-tags
