# Left fist vs right fist from scalp EEG

A motor-imagery decoder for the PhysioNet EEG Motor Movement/Imagery Database
(EEGMMIDB), built for the NT@B software-division recruitment project.

The interesting part of this repository is not the classifier. It is
everything built to find out what the classifier's accuracy is actually
measuring. Three things turned up that I did not expect and would not have
found by optimising a number:

- **Cross-subject beats within-subject here.** Training on 92 other people
  labels a new person's trials *better* than training on 28 of their own. With
  roughly 43 trials per person, the within-subject regimes are starved, not
  inflated. The usual story about optimistic within-subject cross-validation
  does not describe this dataset, and saying so requires having measured both.
- **Most of the accuracy is in the first second of the trial.** Sliding a window
  across the epoch, accuracy rises from 55% to 74.8% in 600 ms, peaks 450 ms
  after the cue, and decays steadily to 57% by the end. I initially read that
  shape as a lateralised visual response, because this protocol puts the cue on
  the *left or right of the screen* and screen position is perfectly confounded
  with the label. My own control says otherwise, and I was wrong: the early
  advantage lives in sensorimotor electrodes, while parieto-occipital
  electrodes contribute the same ~60% at every window position. The confound is
  real and undismissable by design, but it is not what drives the peak.
- **Roughly a third of people cannot be decoded at all**, and the spread across
  people is larger than the difference between any two models I tried.

See [`RESULTS.md`](RESULTS.md) for every number and
[`ANALYSIS_LOG.md`](ANALYSIS_LOG.md) for how each one was arrived at,
including the things that did not work.

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
python predict.py ~/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0/S034/S034R04.edf
```

```
S034R04.edf: 15 trials  accuracy 1.000  (15/15)

pooled accuracy over 1 file(s), 15 trials: 1.0000
```

It takes any number of raw EDF paths, prints per-file accuracy when the file
carries `T1`/`T2` annotations, and writes per-trial predictions with `--out`
(file, trial index, onset in seconds, predicted label, probability of right
fist, and the true label when one is available). Label 0 is the left fist,
label 1 is the right fist.

S034 is one of the twelve quarantined subjects and one of the easy ones; across
all twelve the model scores 76.3%, and three of them are at chance. The
per-subject spread is the point, not the headline average.

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
| `04_model_selection.py` | first selection pass: feature and classifier families, DEV only |
| `15_selection2.py`, `15b_bands_windows.py` | second pass: each family tuned, then band and window, DEV only |
| `05_main_regimes.py` | the headline table, EVAL subjects |
| `06_controls.py` | placebo windows, electrode lesions, subject identity, alignment scope, cross-paradigm transfer |
| `07_deep.py` | EEGNet across subjects, and its scaling behaviour |
| `08_train_final.py` | trains the shipped model |
| `09_figures.py` | figures from the result CSVs |
| `10_who_is_decodable.py` | resting-state predictors of per-subject accuracy |
| `11_holdout_check.py` | runs `predict.py` on the 12 untouched subjects |
| `12_interpretation.py` | time-resolved decoding and scalp weight maps |
| `14_subject_bias.py` | per-subject decision bias and within-session drift |
| `17_naive_splits.py` | what the easy, leaky splits would have reported |
| `19_sequence_confound.py` | separates the alternating cue sequence from real carry-over |
| `20_missing_controls.py` | artefact rejection, and where the early-window advantage lives |
| `13_make_results_md.py` | regenerates `RESULTS.md` |

---

## Results

Full tables in [`RESULTS.md`](RESULTS.md); figures in `figures/`; the reasoning
behind each number, including the attempts that failed, in
[`ANALYSIS_LOG.md`](ANALYSIS_LOG.md).

**The number I defend: 69.3%,** and a floor of 59.8% under the strictest
reading.

- **69.3%** on imagined left-fist versus right-fist for a person the model has
  never seen, over the conventional 0.5 to 3.5 s window. 58 evaluation subjects,
  2,483 trials, 95% confidence interval 67.4 to 71.1. The same pipeline gets
  73.3% on executed movement.
- **59.8%** over a 2.5 to 4.5 s window, which starts after the cue-locked burst
  has passed. This is a floor rather than the headline: it is what remains under
  the strictest reading, in which everything time-locked to the cue is discarded
  on the grounds that a real BCI has no cue. The evidence says that reading is
  too strict, but the number is worth knowing.

| regime (imagined) | accuracy | 95% CI | per-subject mean | shuffled twin |
|---|---|---|---|---|
| A within-subject, random 5-fold | 60.4% | 58.4-62.3 | 60.4 ± 15.6 | 49.7% |
| B within-subject, leave-one-run-out | 61.7% | 59.7-63.6 | 61.7 ± 13.6 | 49.1% |
| C cross-subject, leave-one-subject-out | **69.3%** | 67.4-71.1 | 69.2 ± 14.6 | 48.7% |
| D C plus 16 labelled calibration trials | 69.9% | 67.5-72.1 | 69.8 ± 15.7 | 47.5% |

| regime (executed) | accuracy | 95% CI | shuffled twin |
|---|---|---|---|
| A within-subject, random 5-fold | 66.1% | 64.2-67.9 | 51.0% |
| B within-subject, leave-one-run-out | 67.2% | 65.3-69.1 | 50.3% |
| C cross-subject, leave-one-subject-out | 73.3% | 71.5-75.0 | 48.6% |
| D C plus 16 labelled calibration trials | 76.8% | 74.6-78.8 | 49.0% |

Every shuffled twin lands within a point and a half of 50%, in every regime and
both paradigms. That is the check that matters: it says the splits do not leak.

**The ladder runs backwards, and that is a finding rather than a bug.** Each
person contributes about 43 usable trials, so a within-subject fold trains on
roughly 34 and a leave-one-run-out fold on about 28. A cross-subject fold
trains on about 3,900. Twenty-eight trials are not enough to estimate two
64-channel class covariances; 3,900 trials from other people are, and enough of
what those people share survives the transfer to more than pay for the loss of
personalisation. The usual warning about optimistic within-subject
cross-validation is a statement about leakage, and on this dataset leakage
loses to sample size.

**Labelled calibration data from the new person buys almost nothing on
imagined trials.** Giving the model 0, 4, 8, 16, 24 or 32 labelled trials from
the held-out subject moves accuracy from 69.3% to 70.7%, and the confidence
interval widens as the test set shrinks, so the trend is not even clearly
positive. The unsupervised alignment has already taken the subject-specific
adaptation that was available. On executed movement the same 16 trials are
worth 3.5 points (73.3% to 76.8%), which is a real difference between the two
paradigms rather than noise.

**The end-to-end check.** `predict.py`, running on raw EDF files from the 12
quarantined subjects, scores **76.3%** on imagined trials and 75.6% on executed,
with 9 of 12 subjects individually beating chance. That is higher than the 69.3%
EVAL estimate, and it should not be read as the model being better than
measured: twelve subjects carry a standard error of about 4.8 points, so the two
figures are a single standard error and a half apart. It is a consistency check,
and what it establishes is that the shipped artefact reproduces the reported
behaviour on recordings it has never seen, not that it is better than reported.

**Two things I expected to find and did not.** The cross-subject classifier is
*not* biased per person: the spread of how often it says "right" across
subjects (sd 0.080) is exactly what trial sampling alone predicts (sd 0.076),
and re-centring each subject's decision threshold on their own median is worth
0.08 points. The alignment appears to have already removed the offset. Accuracy
also does not drift within a session: 68.2%, 69.7% and 70.0% on the three
imagined runs, which are separated by several minutes of other tasks
(chi-square p = 0.70), and trial position within a run explains nothing
(rho = +0.03, p = 0.09). Threshold-free, mean per-subject area under the ROC
curve is 0.748.

**69.3% is a group number and it hides most of what is going on.** Per-subject
accuracy has a standard deviation of about 15 points. Only about 69% of
evaluation subjects individually beat their own binomial chance threshold, so
close to a third of people are not decodable at all by this model. The spread
across people is larger than the difference between any two classifiers I
compared.

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
so every subject's mean covariance becomes a multiple of the identity. No labels
are used.
This is the single most important component: it moves cross-subject accuracy
by about four points, and it takes 93-way subject identification in these same
features from 98% down to chance.

**Classifier.** Common spatial patterns computed in the covariance domain
(`models.CovCSP`), log relative power of the components, shrinkage LDA.

## On noise, and on not cleaning

There is no ICA in this pipeline, no epoch rejection, no channel
interpolation and no re-referencing. That is a decision, so it comes with a
measurement rather than an assertion.

The audit found no flat channels, no ADC clipping (the largest amplitudes are
isolated samples on isolated channels, not a rail), and no runs with
pathological durations beyond the four excluded subjects. Within-band
peak-to-peak amplitude per epoch has a median of 167 µV, a 99th percentile of
774 µV and a maximum of 1,712 µV, so there are genuinely noisy epochs. Dropping
them changes almost nothing:

| epochs kept | cross-subject accuracy |
|---|---|
| all 3,975 | 69.28% |
| drop the noisiest 1% | 69.19% |
| drop the noisiest 5% | 69.14% |
| drop the noisiest 20% | 68.72% |

Throwing away the worst fifth of the data costs 0.6 points. There is no cleaning
step here worth making.

The reasoning behind the choice: every cleaning step is a judgement call with
a knob on it, and knobs that are tuned while looking at accuracy are how an
evaluation quietly becomes circular. Band-passing to 8-30 Hz already removes
the two artefact classes that matter most here, since eye blinks live below
8 Hz and drift below 1 Hz. Muscle artefact above 30 Hz is also filtered out,
and the electrode-lesion control is what tells us whether what remains is
sensorimotor or something facial.

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

## Controls: what else could explain 69.3%

Every number here is on the same EVAL subjects and the same pipeline.

**Label shuffles.** Every regime, both paradigms, labels permuted inside each
run so class counts are preserved: 47.5% to 51.0%. Nothing in the splitting
machinery produces accuracy on its own.

**The easy split, which I expected to be inflated and was not.** Pooling every
subject's trials into one pile and taking a random 5-fold — the canonical EEG
mistake, because the test set then contains trials from people whose other
trials are in training — gives **69.0%**, against 69.3% for honest
leave-one-subject-out. No inflation at all. The reason is the alignment:
pooling only helps a model that can recognise the person, and after whitening
it cannot. Without the alignment the pooled split gives 62.1% and honest
leave-one-subject-out gives 63.4%, so again no gap. The shortcut is only worth
something when the shortcut is available. The inflated number in this dataset
comes from the choice of analysis window instead.

**Electrode lesions.** Restricting the model to electrode subsets, cross-subject:

| electrodes | accuracy |
|---|---|
| all 64 | 69.3% |
| sensorimotor strip (15) | 66.1% |
| C3, Cz, C4 only (3) | 63.0% |
| parieto-occipital (17) | 59.8% |
| frontal (17) | 56.1% |

Three electrodes over the hand area get within six points of all 64. Frontal
electrodes, where eye and facial-muscle artefact would dominate, are the weakest
set. Parieto-occipital electrodes reach 59.8%, which looked at first like
evidence of a visual contribution; the `early-window` block in `RESULTS.md`
shows it is the same 59.8% no matter which part of the trial you look at, which
is what volume conduction from sensorimotor sources gives, and the set includes
P3 and P4.

**Subject identity in the same features.** A 93-way subject classifier on these
covariances, trained on two runs and tested on the third, reaches **97.9%**
against a 1.1% chance level. It reaches **98.8%** on the one-minute eyes-open
baseline run, where nobody is doing anything, so this is a property of the
person and the cap, not of the task. After the alignment step the same
classifier falls to **1.6%**, which is chance. The alignment removes essentially
all of it, and that is measured rather than assumed.

**Run identity.** Within a subject, a classifier tells which of their three runs
a trial came from with **99.2%** accuracy against 33% chance. Random folds
inside a subject genuinely do share a large nuisance variable with the test
trials.

**Placebo windows, and the surprise in them.** Two controls came back above
chance. The 2 s of rest *before* the cue predicts the upcoming label at 55.6%,
and the task window predicts the *previous* trial's label at 61.0%. Read
naively, the pipeline leaks and trials are not independent.

Neither is true. Consecutive cues in a run carry different labels 76.8% of the
time, lag-1 correlation −0.54: the sequence strongly alternates, and I had
assumed it was random. Given 69.3% on the current label, 60.3% on the previous
one follows arithmetically; observed 61.0%, so there is nothing left for genuine
carry-over to explain. And the rest period sits 2 to 4 s after the previous
movement, which is when lateralised beta rebound peaks, so it carries a trace of
the previous trial and predicts the next label without containing any
information about it.

The test that separates the two readings splits trials by whether the label
repeated:

| test | accuracy |
|---|---|
| pre-cue rest → the **previous** trial's label | 63.7% |
| pre-cue rest → the **current** label, on alternations | 60.7% |
| pre-cue rest → the **current** label, on repeats | **40.3%** |

Mirrored about chance, which is the signature of a model that has learned
"predict the opposite of the previous trial" and nothing else. Genuine advance
information would be above chance on both.

## How much of this is the person rather than the task

**Most of the spread across people is real.** Per-subject accuracy under
cross-subject evaluation has a standard deviation of 14.6 points. Trial
sampling alone, at 43 trials per person, would produce 6.7. Subtracting the
sampling variance leaves an implied between-subject standard deviation of
**13.0 points, about 79% of the observed variance**. The same decomposition
gives 72-84% across all four regimes and both paradigms. When someone scores
40% and someone else scores 90%, that is mostly not luck.

**Decodability is a stable property of a person.** A subject's accuracy on
imagined trials correlates with their accuracy on *executed* trials, a separate
set of recordings and a different task, at Spearman rho = +0.29 to +0.44
depending on regime (p = 0.001 to 0.03, n = 58). It also correlates 0.54 between
the within-subject and cross-subject regimes.

**But you cannot screen for it from resting EEG.** I tested whether a subject's
decodability is predictable from their one-minute eyes-open and eyes-closed
baseline runs, using nine spectral features per condition: sensorimotor mu and
beta power, posterior alpha power and peak frequency, the aperiodic 1/f slope,
how far the mu bump rises above that subject's own 1/f background, broadband
amplitude, line-noise ratio, and alpha reactivity between eyes open and closed.
Fifty-four correlations, none surviving Bonferroni correction at p < 0.00093;
the strongest is rho = 0.25 at an uncorrected p of 0.054. A sixty-second
recording of someone doing nothing does not tell you whether they will be
decodable, at least not through the standard spectral summaries.

This is the clearest "person, not task" result in the project, and it cuts
both ways. Identity is overwhelmingly present in the signal: 98% decodable from
the same features, including from rest. Decodability is a durable trait that
replicates across paradigms. And yet the obvious markers of a good
sensorimotor rhythm do not predict who has it.

## The thing nobody asked about: the cue is on the wrong side of the screen

PhysioNet's description of this protocol says, of the left-versus-right fist
runs: "A visual target appeared on either the left or right side of the
screen", and the subject acts "until the target disappeared".

Screen position is therefore perfectly confounded with the class label, and the
target is visible for the whole trial. A lateralised visual stimulus produces
lateralised occipital activity and lateralised spatial attention, both decodable
from scalp EEG and neither of them motor imagery. Any classifier trained on
these runs is free to read the screen instead of the motor cortex, and nothing
in a standard cross-validation would tell you which it did. This is a property
of the dataset that no amount of careful modelling can remove.

I did not go looking for this; the window sweep pointed at it. Tuning the
analysis window on DEV subjects picks 0 to 2 s after the cue over 0.5 to 3.5 s,
74.4% against 68.3%, and two windows of the same length — 0.0 to 2.0 and 0.5 to
2.5 — differ by 5.5 points. All of the advantage is in the first 500 ms. The
time-resolved curve says the same thing more sharply: accuracy peaks 450 ms
after the cue and decays monotonically from there.

**Then the control refuted my own hypothesis.** Splitting both windows by
electrode region:

| window | all 64 | sensorimotor (15) | parieto-occipital (17) |
|---|---|---|---|
| 0.0 - 2.0 s | 71.0% | 67.6% | 59.6% |
| 0.5 - 3.5 s | 69.3% | 66.1% | 59.8% |

If the early advantage were visual, it would live in the parieto-occipital
electrodes and appear only in the early window. It does neither.
Parieto-occipital sites contribute the same ~60% regardless of when you look,
which is what volume conduction from sensorimotor sources looks like — and the
set includes P3 and P4, which sit close enough to the hand areas to pick up mu
rhythm directly. The part that *does* change with the window is the
sensorimotor strip.

Two other things point the same way. The classifier's spatial patterns, fitted
across all 93 pool subjects, are four clean dipoles over C3 and C4 with opposite
LDA weights — contralateral hand areas, not occipital cortex
(`figures/fig5_spatial.png`). And 13 to 30 Hz alone reaches 64.2%, while a
visual evoked response does not live in the beta band.

**So what I now believe:** the early peak is the initial, strongest phase of
event-related desynchronisation, which is fastest in the first second after a
movement cue and then partially recovers. That is a well-documented time
course, and it fits both the shape of the curve and where on the scalp it
lives.

I am leaving this section in, rather than deleting a hypothesis I disproved,
because the confound is real and a skeptical reader should know it exists. What
changed is my estimate of how much it matters. I cannot rule out a visual or
attentional contribution — the design makes that impossible — but my own
controls say it is not the driver, and reporting 59.8% as the headline on the
strength of a hypothesis my data contradicts would have been its own kind of
dishonesty.

## The convolutional network, and two bugs in my own harness

EEGNet, trained across subjects, reaches **50.5%** on held-out people while
sitting at 58.5% on its own training set. It is a null result, and the first
version of it was my fault twice over.

**Early stopping was restoring an untrained network.** I checkpointed on
validation accuracy every five epochs from the start of training. On a
validation set of about 340 trials, early validation accuracy is noise, so the
best-scoring checkpoint was often from epoch five, and that is what got restored
and reported. The giveaway was a verbose trace showing 0.560 training accuracy
at epoch 40 while the returned model scored 0.507 on the same data. Fixed by
only considering checkpoints once the one-cycle schedule has annealed.

**Per-channel trial normalisation was deleting the signal.** I was z-scoring
every channel of every trial independently. Left-versus-right imagery is a
*relative* power difference between channels over the two hemispheres, and
forcing every channel to unit variance removes exactly that. Fixed by scaling
each trial by a single scalar.

**Neither fix changed the answer**, which is the useful part. Before the fixes:
50.7% test, 53.5% train. After, with 90 epochs instead of 70: 50.5% test, 58.5%
train. The bugs were real and worth finding; they were not why the network
failed.

**What remains is a capacity and compute limit.** The same network trained on
300 trials from 8 subjects reaches 87.7% on its own training set; trained on
3,900 trials from 93 subjects it reaches 58.5%, and none of that transfers —
the eight-point gap between train and test is subject-specific memorisation,
not learning. A 2,000-parameter network at 80 Hz cannot fit 3,900 heterogeneous
trials, and at roughly six minutes per fold on this laptop I could not give it
the epochs to try.

I report this as a null result about my compute budget, not about EEGNet.
Published cross-subject figures for EEGNet on this dataset sit in the
low-to-mid sixties and I have no reason to doubt them. What I can say is that
a classical covariance pipeline reached 69.3% in seven seconds per fold on the
same machine, and that on a dataset with 43 trials per person that is the
correct engineering choice. `results/deep_prebugfix.csv` keeps the numbers from
before the fixes so the comparison is checkable.

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

**The cue confound, which no analysis can remove.** The target's screen position
is perfectly correlated with the label, for the whole trial, by design. My
electrode-region control says the early advantage is sensorimotor rather than
occipital, and I believe it, but "not driven by the confound" is a weaker claim
than "free of the confound", and only the latter would be safe. Settling it
needs a dataset whose cue is not lateralised.

**The alignment sees the test subject's data.** Whitening a recording by its
own mean covariance uses no labels, so it is legal on unlabelled data, but it
is transductive: the mean is estimated over all of that person's trials,
including the ones being predicted. Estimating it from a subject's first run
only and testing on their other two costs **0.96 points**, 68.7% against 69.6%.
That is smaller than I expected and it makes this a much weaker objection than
I thought it would be when I wrote the control.

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

**With more time, no more compute.** A proper filter-bank CSP with
mutual-information feature selection, which is the standard strong baseline on
this task and which I only approximated with a filter-bank log-variance model.
I would also want the visual-cue question answered properly rather than
circumstantially, which means decoding the same contrast from a dataset whose
cue is not lateralised, and checking whether the models agree.

**With more compute.** Train a single network across all subjects with a small
per-subject adapter, so the shared part learns the task and the adapter absorbs
the person. That is the natural model of the structure this analysis found:
one large nuisance factor that is identity, and one small factor that is the
task. I would also want the network given enough epochs to actually fit 3,900
heterogeneous trials, which it never got here.

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
  gapless trial structure hands you for free — and then, when that placebo came
  back above chance, going to the stimulus sequence rather than concluding the
  pipeline leaked.
- Running the electrode-region control that killed my own most interesting
  finding, after the write-up had already been built around it.
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
that average to become a multiple of the identity. Afterwards every person's
*typical* trial looks the same, and what survives is how each individual trial
deviates from that person's own typical trial. That deviation is the part that
carries the task.

The reason I believe this is doing what I think it is doing, rather than just
helping by accident, is the subject-identification probe. Before whitening, a
classifier on these features names which of 93 people a trial came from about
98% of the time. After whitening, it is at chance. The step removes identity
almost completely and leaves the class information largely intact, and
cross-subject accuracy rises by 5.2 points as a result: 63.4% without it,
69.6% with it, on the same trials.

The cost is that it needs a batch of the subject's data before it can predict
anything. Estimating the whitening from a subject's first run only, instead of
all three, costs 0.96 points — so the cost is real but small, and a short
calibration block is enough.
