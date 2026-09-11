#!/bin/bash
BASE="$HOME/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0"
OUT="$BASE/$1"
[ -s "$OUT" ] && exit 0
mkdir -p "$(dirname "$OUT")"
curl -sfL --retry 4 --retry-delay 2 --max-time 300 \
  "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0/$1" -o "$OUT.part" \
  && mv "$OUT.part" "$OUT" || { rm -f "$OUT.part"; echo "FAIL $1"; }
