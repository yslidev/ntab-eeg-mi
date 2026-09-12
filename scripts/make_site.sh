#!/bin/bash
# Assemble site/ for static hosting. slides/deck.html stays canonical; this
# copies it to site/index.html with the two figures it references, so the
# directory can be served with no build step.
set -e
cd "$(dirname "$0")/.."
mkdir -p site
cp slides/deck.html site/index.html
cp figures/fig2_regime_ladder.png figures/fig4_time_resolved.png site/
echo "site/ rebuilt from slides/deck.html"
