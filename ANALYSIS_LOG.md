# Analysis log

Things I tried, in the order I tried them, including the ones that did not
work. The brief asks for this; it is also the only honest way to show which
results were predictions and which were discoveries.

---

### 1. Getting the data at all

`mne.datasets.eegbci.load_data` fetches from `physionet.org`, which gave me
**29 kB/s**. 763 files at 2.5 MB each would have taken about five hours.
PhysioNet mirrors open datasets to S3, and `physionet-open.s3.amazonaws.com`
gave **2.3 MB/s**, eighty times faster. Twelve parallel `curl` workers pulled
the whole set in under two minutes. `src/data.py` reads from MNE's own cache
directory, so the fast path and the MNE fetcher are interchangeable.

Not a research decision, but it changed what was feasible: I could use all 109
subjects instead of a subset, which matters for every cross-subject claim in
this project.

### 2. Auditing before modelling

I read all 763 recordings and tabulated sampling rate, duration, channel
count, annotation counts and durations, and peak amplitude before writing a
single classifier. What came out:

- **S088, S092, S100** are recorded at 128 Hz, not 160 Hz, and their
  annotation durations are 5.125 s rather than 4.1 s. Resampling alone does not
  reconcile them: the annotation durations and the run length are mutually
  inconsistent, so I cannot tell what the true trial timing is. Excluded.
- **S089 run 3** is 181 s with 22 trials rather than 123 s with 15. Excluded.
- **S104 run 8** is truncated at 106 s with 13 trials. Kept; it is simply short.
- Everything else is 64 channels, 160 Hz, 123–125 s, 15 task trials, strict
  alternation of 4.2 s rest and 4.1 s task.
- Peak amplitudes reach ~800 µV in some subjects, but only on isolated
  channels and isolated samples, not a flat rail. This is not ADC clipping.
- Class balance within a subject is never worse than 24/21. A majority-class
  classifier gets 51–53%, so chance really is close to 50%.

The strict alternation matters later: **every task trial is preceded by exactly
4.2 s of rest**, which gives a free placebo window, and also means the "rest"
period is contaminated by the previous trial.

### 3. Which annotation is the left hand?

The documentation says T1 is the left fist in runs 3/4/7/8/11/12. I wanted to
confirm it from the signal rather than take it on faith, since getting it
backwards would silently invert every interpretation.

**First attempt, which was wrong.** I computed mu-band event-related
desynchronisation as dB change from the pre-cue rest window, then compared C3
against C4. The executed runs came out *non-significant and pointing the wrong
way* (p = 0.12), while the imagined runs came out significant in the opposite
direction. Two paradigms disagreeing, with the stronger-signal paradigm losing,
is a sign the measure is broken, not that the labels are ambiguous.

**Why it was broken.** The only available baseline is the 4.2 s of rest before
the cue, and in a design with no gaps that rest period sits 2–4 s after the
*previous* trial's movement. That is exactly when post-movement beta rebound
peaks. So the "baseline" carries a lateralised signature of the previous trial,
which partially cancels the current one. Executed movement has the strongest
rebound, which is why it was hit hardest.

**Second attempt.** Drop the baseline entirely and use a laterality index:
`log10(power over a left-hemisphere cluster / power over the right cluster)`,
compared between classes. Nothing is referenced to rest. This gives a
consistent answer in both paradigms and all three bands, with 74–83% of the 105
subjects showing the correct sign individually. T1 is the left fist.

### 4. A machine that could not hold the data

The first model sweep started swapping: 40 GB of swap in use on a 24 GB
machine, and a single CSP configuration ran for 20 minutes without finishing.
Three causes, all mine:

- The epoch cache was being upcast to float64 in one shot (2.6 GB), then
  filtered into another copy.
- `joblib` workers were each slicing a fresh ~1.1 GB copy of the trial array.
- `mne.decoding.CSP(reg="ledoit_wolf")` computes the Ledoit-Wolf estimate over
  a 64 × 2.1M matrix per class per fold.

Fixes: filter in blocks and keep float32; precompute every fold-independent
feature once (log-variance, filter-bank power, and covariance matrices do not
depend on the training split, so computing them inside each of 35 folds was
pure waste); scalar shrinkage for CSP. The same sweep then ran in seconds per
configuration instead of tens of minutes.

The general lesson I would carry to a bigger dataset: separate the features
that depend on the fold from the ones that do not, and only recompute the
former. CSP is supervised, so it genuinely has to be refitted. A covariance
matrix is not.

### 5. Splitting the subjects before looking at anything

Twelve subjects were drawn at random and quarantined (`HOLDOUT`). They are not
in model selection, not in any reported cross-validation, and not in the
training set of the shipped model. Of the remaining 93, 35 are `DEV` and 58 are
`EVAL`. Everything I chose — feature family, classifier, metric, band, window,
regularisation — was chosen against DEV folds. Every reported number uses EVAL
folds. `predict.py` is then checked once on the HOLDOUT recordings.

This costs accuracy: I am reporting a number from subjects I never tuned on.
It buys the only thing that makes the number worth reporting.

### 6. A finding that turned out to be an artefact of my own tuning

Euclidean alignment whitens each recording by its own mean spatial covariance,
so every subject's mean covariance becomes the identity. It is the standard
first thing to try for cross-subject EEG. In the first sweep, with every model
sitting at one arbitrary regularisation strength, it did this on DEV folds:

| model | no alignment | with alignment |
|---|---|---|
| CSP + LDA, 6 components | 61.6% | **68.3%** |
| log-variance + logistic regression | 57.6% | 64.9% |
| tangent space, log-Euclidean, C=0.1 | 62.8% | 59.5% |
| tangent space, affine-invariant, C=0.1 | 56.7% | 59.7% |

The tangent-space row is the interesting one, and my first reaction was that
the alignment must be broken. It is not: after alignment each subject's mean
covariance equals the identity to within 3e-6 in Frobenius norm, and the
between-subject dispersion of trace-normalised mean covariances falls from 14.6
to 3e-6. The transform does exactly what it claims.

My second reaction was to write it up as a real asymmetry between model
families, with a plausible story about CSP needing a common frame and the
tangent space already absorbing scale through the matrix logarithm. That story
was wrong too.

What actually happened is that whitening changes the scale of the tangent-space
features, and the tangent-space model has 2,080 of them for about 3,900
training trials, so it is extremely sensitive to its L2 penalty. At C=0.1 it is
badly under-regularised; the first sweep was comparing a tuned CSP against an
untuned tangent-space model. Sweeping the penalty:

| C | 0.1 | 0.01 | 0.003 | 0.001 |
|---|---|---|---|---|
| tangent space + alignment | 59.5% | 62.6% | 64.0% | ~65.7% |

and once the penalty is set sensibly, alignment helps the tangent-space model
as well: 61.4% without it, 65.7% with it. The families end up close together,
with CSP slightly ahead.

I am keeping this in the log rather than quietly fixing it, because it is the
most instructive mistake I made. A single-hyper-parameter comparison between
model families is not a comparison between model families. It is a comparison
between one tuned model and one untuned one, and it will invent effects that
are not there. `scripts/15_selection2.py` re-runs the comparison with each
family's own knob tuned on DEV folds.

### 7. A convolutional network that was too slow before it was too weak

EEGNet at the native 160 Hz over a 3 s window ran at 19 s per training epoch on
this laptop, which would have been about 14 hours for the folds I wanted. The
cost is dominated by the first temporal convolution, which runs at full
64-channel spatial resolution: the activation tensor is
`batch x F1 x 64 channels x 481 samples`, and the whole thing is
memory-bandwidth bound, not compute bound. Halving the sample rate to 80 Hz
(ample for an 8-30 Hz phenomenon) and halving the kernel length cut it by
roughly a factor of four.

It was also not learning: training accuracy sat at 0.52 after 30 epochs with a
loss stuck at 0.695. Dropout 0.5 plus label smoothing plus a cosine schedule
from 1e-3 was too conservative for a 2,000-parameter model on 3,500 samples.
Switched to one-cycle at 3e-3, dropout 0.4, no label smoothing.

### 8. The window sweep found something I did not want to find

Tuning the analysis window on DEV folds produced this:

| window after cue | cross-subject accuracy |
|---|---|
| 0.0 - 2.0 s | **74.4%** |
| 0.0 - 4.0 s | 71.6% |
| 0.5 - 2.5 s | 69.0% |
| 0.5 - 3.5 s | 68.3% |

Two windows of identical length, 0.0-2.0 and 0.5-2.5, differ by five and a
half points. All of the advantage is in the first half second after the cue.

Sensorimotor desynchronisation does not behave like that. It builds over
roughly half a second and is sustained for the length of the trial, so if the
signal were purely sensorimotor, shifting a 2 s window forward by 500 ms should
cost very little. Something sharp and early is contributing.

The likely culprit is the protocol. In BCI2000's version of this task the cue
is a target that appears on the **left or right side of the screen**, and it
stays on screen while the subject performs the trial. A lateralised visual
stimulus produces a lateralised occipital response. A classifier that picks
that up is decoding where the target was, not what the subject imagined, and it
would be worthless in a real BCI, where by definition nothing tells the screen
which hand the user is thinking about.

I am deliberately **not** taking the 74.4% window. Three things decide it, and
all three are in the results:

1. `scripts/12_interpretation.py` slides a 0.75 s window across the whole epoch.
   A sharp early peak means evoked; a slow rise that plateaus means sensorimotor.
2. `scripts/06_controls.py` block 9 splits both windows by electrode region. If
   parieto-occipital electrodes carry the 0-2 s advantage while the
   sensorimotor strip does not, that settles it.
3. The band sweep is already a weak argument against a purely visual account:
   13-30 Hz alone reaches 64.2%, and a visual evoked response does not live in
   the beta band.

The headline number stays on the 0.5-3.5 s window, chosen before I saw any of
this, because that is the window whose result would survive in a setting where
no lateralised cue is on the screen. Reporting the higher number as the
headline would be optimising the metric rather than the claim.

### 9. The result I expected to get, and did not

I built the five-regime ladder expecting the usual shape: random folds inside
one subject would look great, holding out a whole run would cost a few points,
and holding out a person would cost a lot. That is the standard story about
optimistic cross-validation, and I had already written most of the controls to
explain it.

On EVAL subjects, imagined trials:

| regime | accuracy | shuffled twin |
|---|---|---|
| A within-subject, random 5-fold | 60.4% | 49.7% |
| B within-subject, leave-one-run-out | 61.7% | 49.1% |
| C cross-subject, leave-one-subject-out | **69.3%** | 48.7% |
| D C plus 16 labelled calibration trials | 69.9% | 47.5% |

The ladder runs the wrong way. Holding out an entire person is *easier* than
holding out nine of that person's own trials.

The explanation is arithmetic rather than neuroscience. Each subject has about
43 usable imagined trials, so a within-subject fold trains on roughly 34 of
them, and a leave-one-run-out fold on about 28. A cross-subject fold trains on
about 3,900. Twenty-eight trials is not enough to estimate two class
covariances over 64 channels, even with shrinkage; 3,900 trials from other
people is, and what those people share survives the transfer.

Two things follow that I would not have said before measuring it:

- The folklore that within-subject cross-validation is inflated is a statement
  about *leakage*, and leakage has to compete with *sample size*. On a dataset
  with three runs and 45 trials per person, sample size wins. On a dataset with
  an hour of data per person it would not.
- Regime A is still optimistic in the way the folklore says. It is just
  optimistic about a smaller number. A classifier can tell which of a
  subject's three runs a trial came from about 99% of the time, so random folds
  inside a subject genuinely do share a large nuisance variable; that shows up
  as A and B being close, rather than as A beating C.

I expected the inflated figure the brief anticipates to come from pooling
everyone's trials into one pile and taking a random split, which is the
canonical EEG mistake. It does not, in this pipeline: the pooled random split
gives 69.0% against 69.3% for honest leave-one-subject-out. The reason is the
alignment. Pooling only helps a model that can recognise the person, and after
whitening each recording by its own mean covariance it cannot: 93-way subject
identification falls from 97.9% to 1.6%. Without alignment the pooled split
gives 62.1% and honest leave-one-subject-out gives 63.4%, so again no
inflation. The shortcut is only worth something when the shortcut is available.

The inflated figure in this dataset comes from somewhere else entirely: the
choice of analysis window, and what the subject is looking at while the window
is open. That is sections 8 and 11.

### 10. Two controls failed, and the reason was in the stimulus sequence

Two of the controls came back above chance, which is the outcome you least
want to see:

| control | accuracy | 95% CI |
|---|---|---|
| the 2 s of rest *before* the cue, predicting the upcoming label | 55.6% | 53.6 - 57.6 |
| the task window, predicting the *previous* trial's label | 61.0% | 59.0 - 63.0 |

Read naively, the first says the pipeline leaks and the second says trials are
not independent. Either would invalidate the headline number.

Before believing either, I looked at the label sequence itself, which I had
already checked once for a different reason (whether the same sequence is
reused across subjects; it is not). Consecutive trials within a run carry
**different** labels 76.8% of the time, with a 95% interval of 75.4 to 78.2 and
a lag-1 correlation of -0.54. BCI2000's target sequence strongly alternates. It
is not a coin flip, and I had assumed it was.

That single fact accounts for both results.

**The carry-over control.** A decoder that is right about the current label a
fraction `a` of the time is automatically right about the previous label a
fraction `a*q + (1-a)*(1-q)` of the time, where `q = 0.768` is the alternation
rate. With `a = 0.693` that comes to **60.4%**. The observed figure is 61.0%.
There is nothing left for genuine trial-to-trial carry-over to explain.

**The pre-cue control.** The 4.2 s of rest before a cue sits 2 to 4 s after the
previous trial's movement, which is exactly when post-movement beta rebound
peaks, and beta rebound is lateralised. So the rest period carries a trace of
the *previous* trial. Because the next label is usually the opposite, that
trace predicts the upcoming label without containing any information about it.
A window that knew the previous label perfectly would score 76.8% on the
current one; the pre-cue window manages 55.6%, which is a weak trace, not a
leak of future information.

`scripts/19_sequence_confound.py` runs the decisive test: split trials by
whether the label repeated. If the pre-cue window works only through the
previous trial, its accuracy on the current label must sit above chance on
repeats and *below* chance on alternations, in mirror image. Genuine advance
information would be above chance on both.

What I take from this. The pre-cue placebo is the right control to run, and it
would have been easy to report "55.6%, above chance, therefore my pipeline
leaks" and equally easy to report "61% on the previous trial, therefore EEG
carries over between trials". Both readings are wrong, and distinguishing them
needed a property of the *stimulus protocol* rather than of the signal. It is
the clearest case in this project of a number meaning something other than what
it appears to mean.

### 11. Where the information actually is in the trial

Sliding a 0.75 s window across the epoch and running full leave-one-subject-out
at each position, on all 58 EVAL subjects:

| window centre (s from cue) | accuracy |
|---|---|
| -1.75 to -0.15 | 50 - 55% |
| +0.05 | 68.3% |
| +0.25 | 72.8% |
| **+0.45** | **74.8%** |
| +0.65 | 72.0% |
| +1.05 | 67.0% |
| +1.85 | 62.9% |
| +2.65 | 60.7% |
| +3.45 | 56.8% |

This settles the window question. Accuracy rises from 54.6% to 74.8% in 600 ms,
peaks at 450 ms after the cue, and then decays monotonically for the rest of the
trial. Sensorimotor desynchronisation does not have that shape. It ramps over
roughly half a second and then *sustains* for as long as the imagery continues,
which would show as a plateau. A sharp peak at 450 ms that decays steadily is
the signature of a stimulus-evoked response, and the stimulus here is a target
appearing on the left or right of the screen.

So the 69.3% I measure over 0.5 to 3.5 s is a mixture. Most of it is the first
second. The part that survives into the sustained portion of the trial, where
no evoked transient remains, is about 57 to 61%, which matches the 59.8% I get
from a fixed 2.5 to 4.5 s window.

That is the number I would quote to someone building a brain-computer
interface, because it is the part that does not depend on a lateralised cue
being on screen. It is also, as the brief predicted, considerably lower than
the first figure I saw.

### 12. The convolutional network, and two bugs in my own harness

EEGNet cross-subject came back at **50.7%**, dead chance, with 53.5% accuracy on
its own training set. Two things were wrong, and only one of them was fixable
in the time I had.

**Bug one: early stopping restored an untrained network.** I checkpointed on
validation accuracy every five epochs from the beginning of training. On a
validation set of about 340 trials, validation accuracy is noise early on, so
the best-scoring checkpoint was frequently from epoch five. The function then
restored that checkpoint and reported it. The giveaway was that the verbose
trace showed training accuracy at 0.560 by epoch 40 while the returned model
scored 0.507 on the same training data. Fixed by only considering checkpoints
after the one-cycle schedule has annealed.

**Bug two, which I could not fix: per-channel trial normalisation.** I was
z-scoring every channel of every trial independently. Left-versus-right motor
imagery is a *relative* power difference between channels over the two
hemispheres, and forcing every channel to unit variance deletes exactly that.
Fixed by scaling each trial by a single scalar. It did not rescue the result.

**What was left is a capacity and compute limit, not a bug.** The diagnostic
that matters: the same network trained on 300 trials from 8 subjects reaches
87.7% on its own training set; trained on 3,200 trials from 93 subjects it
reaches 56.7%. It can fit a small, homogeneous set and cannot fit a large,
heterogeneous one. A 2,000-parameter network at 80 Hz for 70 epochs is simply
not enough for this, and on an M-series laptop each fold cost about six minutes,
which is why it was 70 epochs.

I am reporting this as a null result about my compute budget rather than about
EEGNet. The published cross-subject figures for EEGNet on this dataset are in
the low-to-mid sixties, and I have no reason to doubt them. What I can say is
that a classical covariance pipeline reached 69.3% in seven seconds per fold on
the same laptop, and that on a dataset with 43 trials per person that is the
correct engineering choice.
