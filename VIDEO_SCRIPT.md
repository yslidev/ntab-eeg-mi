# Demo video script

I cannot record audio or video, so this is a shot-by-shot script with the real
numbers already in it. About 430 words at a normal 150 wpm pace lands near 2:50.

Open before you start: the repo, `RESULTS.md`, `figures/`, and
`slides/deck.html` (arrow keys advance it).

---

## 0:00 – 0:20 · The task and the number

> Left fist versus right fist, imagined, 64-channel scalp EEG, 109 subjects.
> The number I defend is **69.3%** on a person the model has never seen. Within
> that same person, training on their own trials, I get **60.4%**. The
> cross-subject number is the higher one, and explaining why is most of this
> project.

*Deck slide 1.*

## 0:20 – 0:50 · The evaluation, and why the ladder runs backwards

> Four definitions of held out, each with a label-shuffled twin. Every twin
> lands within a point and a half of fifty percent, which is what says the
> splits do not leak.
>
> Holding out a whole person beats holding out nine of their own trials, because
> each person has only 43 trials. A within-subject fold trains on 28; a
> cross-subject fold trains on 3,900. The usual warning about optimistic
> within-subject cross-validation is a statement about leakage, and here leakage
> loses to sample size.
>
> Subjects were split three ways before anything was fitted: 35 for choosing,
> 58 for reporting, 12 quarantined and opened once at the end by the prediction
> script, which scored 76.3% on them.

*Deck slides 2 and 3.*

## 0:50 – 1:25 · Ruling out the boring explanations

> Ninety-three-way subject identification from these same features: 98%,
> against 1.1% chance. It works just as well on a minute of eyes-open rest where
> nobody is doing anything. After the alignment step it falls to 1.6%, which is
> chance. That is the single most important component, worth five points, and
> it is measured rather than asserted.
>
> Three electrodes over the hand area get 63% against 69.3% for all sixty-four.
> Frontal electrodes, where eye and muscle artefact would dominate, are the
> weakest set. Dropping the noisiest fifth of all epochs costs 0.6 points, which
> is why there is no ICA in this pipeline.

*Deck slide 7.*

## 1:25 – 1:50 · Two controls that failed, and why

> The rest period *before* the cue predicts the upcoming label at 55.6%. The
> task window predicts the *previous* trial's label at 61%. Read naively, the
> pipeline leaks and trials are not independent.
>
> Neither is true. Consecutive cues in a run differ 77% of the time — the
> sequence strongly alternates, and I had assumed it was random. Given 69.3% on
> the current label, 60.4% on the previous one follows arithmetically. Observed:
> 61%. And a lateralised trace of the previous trial in the rest period predicts
> the next label without containing any information about it.

*Deck slide 6.*

## 1:50 – 2:15 · The mistake I made, and unmade

> The window sweep prefers zero to two seconds over the standard window, and all
> of the advantage is in the first half second. This protocol puts the cue on the
> left or right of the screen, so I concluded I was decoding the screen, and I
> rewrote the whole report around it.
>
> Then I split both windows by electrode region. Parieto-occipital electrodes
> give 59.6% and 59.8% — identical in both. Nothing about them is early. The
> component that changes is the sensorimotor strip. The early peak is motor, not
> visual. The confound is real and I cannot exclude it by design, but it is not
> what drives the number.

*Deck slides 4 and 5, then `ANALYSIS_LOG.md` section 13.*

## 2:15 – 2:35 · One design decision

> CSP normally estimates class covariances from concatenated raw trials: a
> 64-by-1.9-million matrix, twice, in each of 58 folds. That took 150 seconds a
> configuration and drove a 24-gigabyte laptop into 40 gigabytes of swap. An
> epoch's covariance does not depend on its fold, so I compute it once. Same
> answer to within 0.7 points, twenty-two times faster, and that is what made
> the whole control suite affordable instead of a choice of two.

*Deck slide 8.*

## 2:35 – 3:00 · Weakest point, next, AI

> What I trust least: the alignment sees the test subject's unlabelled data.
> Estimating it from their first run only costs 0.96 points, which is smaller
> than I feared. And every cross-subject result here is also cross-session, and
> this dataset cannot separate them.
>
> Next, with compute: one network with per-subject adapters, because that is the
> structure this analysis found — a large nuisance factor that is identity and a
> small one that is the task.
>
> Claude wrote the scaffolding, the plots and the covariance-domain CSP. The
> judgement calls were quarantining twelve subjects before touching anything,
> using the pre-cue rest as a placebo, and running the control that killed my
> own favourite finding.

*Deck slide 9.*
