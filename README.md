# Left fist vs right fist from scalp EEG

A motor-imagery decoder for the PhysioNet EEG Motor Movement/Imagery Database
(EEGMMIDB), built for the NT@B software-division recruitment project.

The interesting part of this repository is not the classifier. It is the gap
between the first accuracy figure you can get out of this dataset and the one
that survives contact with a held-out person. Most of the code exists to
measure that gap and to rule out the boring explanations for it.

**Short version of the result.** See `RESULTS.md` for every number, and the
summary table in [Results](#results) below.

---

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Data is fetched from PhysioNet. The S3 mirror is roughly eighty times faster
than the main site, so `scripts/fetch_one.sh` pulls from there into the same
cache directory MNE would use (`~/mne_data/MNE-eegbci-data/...`).

```bash
bash -c 'xargs -n1 -P 12 ./scripts/fetch_one.sh < scripts/filelist.txt'     # task runs
bash -c 'xargs -n1 -P 12 ./scripts/fetch_one.sh < scripts/filelist_r2.txt'  # eyes-closed baseline
```

Roughly 2 GB, a couple of minutes.

## Predicting

```bash
python predict.py ~/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0/S042/S042R04.edf
```

It takes any number of raw EDF paths, prints per-file accuracy when the file
carries `T1`/`T2` annotations, and writes per-trial predictions with `--out`.
Label 0 is the left fist, label 1 is the right fist.

To rebuild the model from scratch:

```bash
python scripts/02_build_cache.py      # epoch cache, ~2 min
python scripts/08_train_final.py      # writes model/mi_lr_model.joblib
```

## Reproducing the analysis

Scripts are numbered in dependency order.

| script | what it does |
|---|---|
| `01_audit.py` | structural audit of all 763 recordings |
| `02_build_cache.py` | wide broadband epoch cache |
| `03_lateralisation.py` | establishes which annotation is which hand |
| `04_model_selection.py` | feature/classifier/band/window sweep, DEV subjects only |
| `05_main_regimes.py` | the headline table, EVAL subjects |
| `06_controls.py` | placebo windows, electrode lesions, subject identity, transfer |
| `07_deep.py` | EEGNet, including its overfitting behaviour |
| `08_train_final.py` | trains the shipped model |
| `09_figures.py` | figures from the result CSVs |
| `10_who_is_decodable.py` | resting-state predictors of per-subject accuracy |
| `11_holdout_check.py` | runs `predict.py` on the 12 untouched subjects |
| `12_interpretation.py` | time-resolved decoding and scalp weight maps |
| `13_make_results_md.py` | regenerates `RESULTS.md` |

---

## What the pipeline does

**Getting the data honest first.** All 763 recordings are read and tabulated
before any modelling (`scripts/01_audit.py`). Three subjects are recorded at
128 Hz with annotation durations that cannot be reconciled with their run
lengths, and one has a malformed run; those four are excluded. Everything else
is 64 channels at 160 Hz, 123-125 s per run, 15 trials per run in strict
alternation of 4.2 s rest and 4.1 s task. Class balance within a subject is
never worse than 24/21, so a majority-class classifier gets 51-53% and chance
really is near 50%.

**Establishing the labels from the signal.** `T1` and `T2` mean different things
in different runs, and getting the mapping backwards would silently invert
every interpretation downstream. `scripts/03_lateralisation.py` settles it with
a baseline-free laterality index rather than the documentation, and finds that
T1 desynchronises the right hemisphere more, in both paradigms and all three
bands, in 74-83% of subjects individually. T1 is the left fist.

**Preprocessing.** Band-pass 8-30 Hz, fourth-order Butterworth, zero-phase,
applied to the *continuous* run before epoching so the filter's edge transient
lands at run boundaries rather than at every trial. Epochs are cut 0.5 to 3.5 s
after the cue. No ICA, no artefact rejection, no re-referencing, no per-channel
normalisation. That is a deliberate choice: every one of those is a judgement
call that could be tuned to the test set, and the audit showed no evidence of
clipping, flat channels, or amplitude outliers severe enough to need them. The
one thing that does get removed is the between-subject covariance shift, and
that is removed by an explicit, measurable step rather than by cleaning.

**Alignment.** Each recording is whitened by its own mean spatial covariance,
so every subject's mean covariance becomes the identity. No labels are used.
This is the single most important component: it moves cross-subject accuracy
by about four points, and it takes 93-way subject identification in these same
features from 98% down to chance.

**Classifier.** Common spatial patterns computed in the covariance domain
(`models.CovCSP`), log relative power of the components, shrinkage LDA.

## How the evaluation is built

Five regimes, in decreasing order of what the training and test sets share.

| | regime | train and test share |
|---|---|---|
| A | within-subject, random 5-fold | the same person, the same session, the same recording run, and trials minutes apart |
| B | within-subject, leave-one-run-out | the same person and session, but different runs |
| C | cross-subject, leave-one-subject-out | nothing except the task |
| D | C, plus 16 labelled trials from the new person | a short calibration block |
| F | C, with the calibration block swept from 0 to 32 trials | |

Every regime is run twice: once with real labels and once with labels permuted
inside each run, which preserves the class counts. The shuffled twin is not a
formality. It is the only thing that distinguishes "the model found signal"
from "the split leaked".

Three further things establish that regime A is optimistic rather than merely
different:

- A classifier can identify **which of a subject's three runs** a trial came
  from with about 99% accuracy against a 33% chance level. Random folds inside
  a subject therefore share a large, easily learned nuisance variable with the
  test trials.
- A classifier can identify **which of 93 people** a trial came from with about
  98% accuracy against a 1.1% chance level, using the same covariance features
  the decoder uses. It can do it just as well from the one-minute eyes-open
  baseline run, where nobody is doing anything.
- After the alignment step, that same subject identification falls to chance.
  The alignment is not cosmetic; it removes the dominant source of variance in
  these features, and that is measurable rather than assumed.

## The design decision I would defend hardest

CSP is normally implemented by estimating each class's spatial covariance from
the concatenated raw trials. MNE does it that way, and it is the right default
for a single subject's single session. For this project it is the wrong shape:
in leave-one-subject-out there are 58 folds, and each one recomputes a
64 x 1.9-million-sample covariance twice. That is what made the first sweep use
4 GB per worker, take 150 seconds per configuration, and drive a 24 GB machine
into 40 GB of swap.

The covariance of an *epoch* does not depend on which fold the epoch is in. So
`models.CovCSP` takes precomputed per-epoch covariances, averages them within
class, trace-normalises, and solves the same generalised eigenvalue problem.
It agrees with MNE's implementation to within 0.7 points on the same folds
(67.6% against 68.3%, against a confidence interval of plus or minus 2.4), and
it runs in 7 seconds instead of 150.

The alternative I considered was to keep MNE's CSP and simply run fewer
controls. I think that is the wrong trade. The controls are the part of this
submission that is worth reading; the 22-fold speedup is what made it possible
to run the placebo windows, the electrode lesions, the carry-over test, the
alignment-scope comparison and the cross-paradigm transfer rather than picking
two of them.

The same principle applies one level up, and it is the general lesson: separate
the features that depend on the training fold from the ones that do not. Log
variance, filter-bank power and covariance matrices do not. CSP filters, LDA
weights and the tangent-space projection point do. Recomputing the first group
inside every fold is the single most common way to make an EEG sweep 20 times
slower than it needs to be.

## What I trust least

**The alignment sees the test subject's data.** Whitening a recording by its
own mean covariance uses no labels, so it is legal on unlabelled data, but it
is transductive: the mean is estimated over all of that person's trials,
including the ones being predicted. A strict online system would have to
estimate it from a calibration block. `scripts/06_controls.py` measures exactly
that cost by estimating the alignment from a subject's first run only and
testing on their other two. If my conclusion is wrong anywhere, this is the
most likely place.

**Model selection on 35 subjects is itself noisy.** A DEV leave-one-subject-out
estimate carries a confidence interval of roughly plus or minus 2.4 points, and
I compared about fifty configurations against it. The winner is therefore
somewhat optimistic *on DEV*. The reported numbers come from EVAL folds, which
protects the headline figure, but it does not guarantee that the configuration
I picked is the best one. This is why the component count was chosen by a
one-standard-error rule rather than by taking the maximum.

**Forty-three trials per subject.** A single subject's accuracy has a standard
error of about 7.6 points. Any claim about an *individual* subject in this
dataset is weak, and the analysis of which people are decodable is attenuated
by that noise. Group-level claims are fine; per-person ones are not.

**One session per subject.** Every cross-subject result here is also a
cross-session result, and they cannot be separated. The question a real BCI
faces is whether a model still works on the *same* person next week, and this
dataset cannot answer it. I would not quote any number here as an estimate of
session-to-session stability.

**Four excluded subjects.** If the three 128 Hz subjects and the one malformed
subject are unusual in ways that correlate with decodability, excluding them
inflates the result. I cannot rule this out; I can only note that it is four
subjects out of 109 and that the exclusion rule is about file structure, not
about accuracy.

## What I would do next

**With more time, no more compute.** Two things I did not get to. First, a
per-subject decision-threshold correction: the cross-subject classifier's
decision function sits off-centre for individual people, and re-centring it on
each subject's own median is unsupervised and appears to help
(`scripts/14_subject_bias.py`). Second, a proper filter-bank CSP with
mutual-information feature selection, which is the standard strong baseline on
this task and which I only approximated.

**With more compute.** Train a single network across all subjects with a small
per-subject adapter, so the shared part learns the task and the adapter absorbs
the person. That is the natural model of the structure this analysis found:
one large nuisance factor that is identity, and one small factor that is the
task. I would also want the scaling curve continued past 93 subjects, since
the EEGNet numbers here do not obviously saturate.

**What I would want that this dataset cannot give me.** Multiple sessions per
subject. Every cross-subject number here is confounded with cross-session, and
the deployment question a BCI actually faces is the one this data cannot
address.

## AI use

I used Claude (Claude Code) throughout: for scaffolding the loaders, the
plotting code, and the argument parsing; for the EEGNet implementation; and for
the performance work, including writing the covariance-domain CSP once I had
identified where the time was going.

The judgement calls are mine, and these are the ones I would defend in
conversation:

- Quarantining twelve subjects before doing anything, so the prediction script
  could be checked on recordings that are unseen in every sense, and splitting
  the rest into selection and reporting folds.
- Using the four seconds of rest before each cue as a placebo window, which the
  strict alternating trial structure hands you for free.
- Not trusting the documented `T1`/`T2` mapping, and, when my first attempt to
  verify it gave contradictory answers across paradigms, recognising that the
  pre-cue baseline is contaminated by the previous trial's beta rebound rather
  than concluding the labels were ambiguous.
- Catching that "alignment helps CSP and hurts Riemannian models" was an
  artefact of comparing a tuned model against an untuned one, and re-running
  the comparison with each family's regularisation tuned.

### One choice, explained in my own words

Why whiten each recording by its own mean spatial covariance, and why that is
the step that matters most.

Two people wearing the same cap do not produce the same numbers. Skull
thickness, hair, electrode impedance, how much the reference electrode drifts,
how tense they are: all of it scales and mixes the 64 channels differently.
When you stack many people's trials into one training set, the spread you see
across rows is mostly the spread between people, not the spread between
imagining a left hand and imagining a right hand. CSP solves an eigenvalue
problem that finds the directions where two classes differ most in variance,
and if the dominant variance in the matrix is who the person is, that is what
it will find.

Whitening fixes the frame. For each recording you compute the average spatial
covariance over its own trials and apply its inverse square root, which forces
that average to become the identity. Afterwards, every person's *typical* trial
looks the same, and what survives is how each individual trial deviates from
that person's own typical trial. That deviation is the part that carries the
task.

The reason I believe this is doing what I think it is doing, rather than just
helping by accident, is the subject-identification probe. Before whitening, a
classifier on these features names which of 93 people a trial came from about
98% of the time. After whitening, it is at chance. The step removes identity
almost completely and leaves the class information largely intact, and
cross-subject accuracy rises by about four points as a result.

The cost is that it needs a batch of the subject's data before it can predict
anything, which is why I measure separately what happens when the whitening is
estimated from a short calibration block instead of the whole recording.
