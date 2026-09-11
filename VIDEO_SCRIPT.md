# Demo video script (3 minutes)

I cannot record audio or video, so this is a shot-by-shot script with the
numbers already in place. Timings assume a normal speaking pace of about
150 words per minute. Total is about 440 words, which lands near 2:55.

Open the repo, `RESULTS.md`, and the `figures/` folder before you start.

---

## 0:00 - 0:20  What the task is and what the answer is

> Left fist versus right fist, imagined, from 64-channel scalp EEG, 109
> subjects. The number I defend is **{{LOSO}}** on a person the model has never
> seen. The first number I got was **{{WITHIN_RANDOM}}**. Almost everything I
> built exists to explain the gap between those two.

*Show: `figures/fig2_regime_ladder.png`.*

## 0:20 - 0:50  The evaluation, and why those splits

> The same model, five definitions of held out. Random folds inside one person
> share the recording block with the test trials. Holding out a whole run
> removes that. Holding out a person removes everything. Each bar has a
> shuffled-label twin next to it; they all sit on chance, which is what tells
> you the pipeline is not leaking labels through some other path.
>
> The subjects are split three ways before anything else happened. Thirty-five
> for choosing the model, fifty-eight for reporting, twelve quarantined and
> only touched at the end by the prediction script.

*Show: the split definition in `src/config.py`, then `figures/fig3_per_subject.png`.*

## 0:50 - 1:20  Ruling out the boring explanations

> Four things could make that number real-looking and wrong, so I measured all
> four. Decode the four seconds of rest *before* the cue: **{{PRECUE}}**, chance.
> Restrict to occipital electrodes: **{{OCCIP}}**, against **{{MOTOR}}** on the
> sensorimotor strip. Label each trial with the previous trial's class:
> **{{CARRY}}**. And time-resolved decoding shows nothing before the cue and a
> rise afterwards.

*Show: `figures/fig9_controls.png`, then `figures/fig4_time_resolved.png`.*

## 1:20 - 1:50  The design decision

> The one decision I would defend hardest: CSP in the covariance domain rather
> than on raw trials. MNE's CSP estimates class covariances from concatenated
> raw data, which is a 64-by-1.9-million matrix per class per fold. Per-epoch
> covariances do not depend on the fold, so I compute them once. Same accuracy
> to within noise, twenty times faster, and that is what made the whole control
> suite affordable rather than a thing I would have skipped.

*Show: the `CovCSP` docstring in `src/models.py`.*

## 1:50 - 2:20  The mistake, and what I trust least

> I originally concluded that alignment helps CSP and hurts Riemannian models.
> That was wrong. It was an artefact of comparing a tuned model against an
> untuned one. With the penalty set properly, alignment helps both.
>
> What I trust least is the alignment itself. It is estimated from the test
> subject's own trials. It uses no labels, but it does see data I would not
> have in a strict online setting. Estimating it from the first run only costs
> **{{ALIGN_COST}}** points.

*Show: `ANALYSIS_LOG.md` section 6.*

## 2:20 - 2:45  The thing nobody asked about

> {{EXTRA_FINDING}}

*Show: `figures/fig6_who_is_decodable.png` and `figures/fig10_subject_bias.png`.*

## 2:45 - 3:00  Next, and AI use

> With more compute: multiple sessions per subject, which this dataset cannot
> give me, and a network pretrained across subjects with per-subject adapters.
>
> I used Claude for the scaffolding, the plotting, and for speed work like the
> covariance-domain CSP. The judgements are mine: the three-way subject split,
> the pre-cue placebo, and catching that the alignment result was a tuning
> artefact rather than a finding.
