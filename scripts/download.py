"""Fetch EEGMMIDB runs from PhysioNet via MNE. Idempotent; safe to re-run."""
import sys, time, warnings
import mne
from mne.datasets import eegbci

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

RUNS = [1, 3, 4, 7, 8, 11, 12]  # baseline-eyes-open, executed L/R, imagined L/R
SUBJECTS = range(1, 110)

ok, fail = 0, 0
for s in SUBJECTS:
    for attempt in range(4):
        try:
            eegbci.load_data(s, RUNS, update_path=True, verbose="ERROR")
            ok += 1
            print(f"S{s:03d} ok", flush=True)
            break
        except Exception as e:
            if attempt == 3:
                fail += 1
                print(f"S{s:03d} FAIL {type(e).__name__}: {e}", flush=True)
            else:
                time.sleep(3 * (attempt + 1))
print(f"done ok={ok} fail={fail}", flush=True)
