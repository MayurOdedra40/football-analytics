# Pass Receiver Prediction from DFL Tracking + Event Data

Predicting who a pass was aimed at, from player tracking + event data for 7 released
Bundesliga matches (2022/23, DFL, CC-BY 4.0, Bassek et al. 2025).

Data path: `data/` (raw XML, tracked via git-lfs — see `.gitattributes`).
Match ids: `J03WMX`, `J03WN1`, `J03WPY`, `J03WOH`, `J03WQQ`, `J03WOY`, `J03WR9`.

## Status

- [x] Stage 1 — XML → tabular (`src/parse.py`)
- [x] Stage 2 — EDA (`notebooks/01_eda.ipynb`)
- [x] Stage 3 — modelling dataset (`src/build_dataset.py`)
- [x] Stage 4 — baselines (`src/baselines.py`)
- [x] Stage 5 — model (`src/model.py`, `src/train.py`)
- [x] Stage 6 — visualisation (`src/viz.py`, `notebooks/02_pass_visualisation.ipynb`)

## Repo layout

```
├── config.yaml                  # paths, TARGET_HZ, TEAM_NAME, SYNC_METHOD, TRAIN_SCOPE, seed
├── README.md
├── requirements.txt
├── data/
│   ├── (raw XML — tracked via git-lfs, see .gitattributes)
│   └── processed/                # gitignored, reproducible -- see below
│       ├── matches.csv, players.csv, events.csv, positions_sample.csv
│       ├── positions/<match_id>.parquet
│       ├── pass_candidates.parquet
│       ├── baseline_fold_results.csv, nn_fold_results_{all_matches,single_team}.csv
│       └── pass_animation_demo.gif
├── src/
│   ├── parse.py           # Stage 1: XML -> tabular
│   ├── pitch.py            # Stage 2: draw_pitch(ax), shared by both notebooks
│   ├── build_dataset.py    # Stage 3: pass_candidates.parquet (sync, features, rotation)
│   ├── baselines.py        # Stage 4: 3 baselines + shared eval/CV infra (reused by Stage 5)
│   ├── model.py             # Stage 5: CandidateScorer (shared-weight MLP)
│   ├── train.py             # Stage 5: training loop, leave-one-match-out CV
│   └── viz.py                # Stage 6: snapshot/trajectory/plotting helpers
├── notebooks/
│   ├── 00_full_walkthrough.ipynb       # everything, condensed -- the one to show an interviewer
│   ├── 01_eda.ipynb                    # Stage 2, executed
│   └── 02_pass_visualisation.ipynb     # Stage 6, executed
└── tests/
    └── test_sanity.py    # coordinate transform, distance-covered range, leakage tripwire
```

Reproduce everything from raw data: `python -m src.parse && python -m src.build_dataset
&& python -m src.baselines && python -m src.train`, then re-execute the two notebooks
with `jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb`.

## Decisions

| Decision | Alternatives considered | Why |
|---|---|---|
| Primary event coordinate = `X-Position`/`Y-Position`, fall back to `X-Source-Position`/`Y-Source-Position` only when missing | Use `X-Source-Position` only, as a literal reading of the DFL schema docs suggests | Empirically, `X-Position` is populated on 91.7% of events vs 76.3% for `X-Source-Position`, and where both exist they disagree by >5m on 12% of events (up to 50m). `X-Position` reads as DFL's maintained/QC'd coordinate; `X-Source-Position` as the original tag before correction. Preferring the better-covered field maximises usable rows; both raw values are kept as columns so nothing is lost. |
| 10 Hz downsampling via keep-every-3rd-frame (actual rate ≈8.33 Hz) | Keep full 25 Hz; downsample to an exactly-divisible rate (e.g. 12.5 Hz via every-2nd-frame) | Consecutive 40ms frames are near-identical for the position/velocity features Stage 3 needs, so this discards redundancy, not signal, and cuts the working set ~60% (420MB → ~25MB per match). 25 Hz isn't evenly divisible by 10, so "10 Hz" is a nominal label, not an exact rate — logged explicitly at runtime rather than silently mislabelling the output. |
| Drop `Delete` events, log the count | Keep them with a flag column | They're explicitly annotation artifacts removed in DFL's own post-processing — carry no football information, only noise for anything downstream. 537 dropped across 7 matches (~24% of a raw file's rows on average — a large fraction, worth knowing about). |
| Coordinate transform: `x_centred = x_event − 52.5`, `y_centred = y_event − 34.0` | Read positions in corner-origin instead | Matches how the tracking (position) files are already stored (centre-origin), so events and positions become directly comparable without transforming the much larger position files. Verified: the kickoff event lands within 1m of the ball's tracked position for the 2 matches with clean event timestamps (see Sanity checks). |
| Distance-covered sanity check restricted to full-match, non-red-carded outfield starters | Check all starters | A starter subbed off at minute 60 or sent off legitimately covers less ground — that's not a data bug, and checking them anyway produced false alarms (verified against 3 specific players' substitution/red-card events and matched exactly). |
| `data/processed/` gitignored; `data/` (raw) is not | Gitignore all of `data/`, per the spec's default repo layout | The raw XMLs were already committed via git-lfs before this project started (see `.gitattributes`, commit history). Processed outputs (~180MB, dominated by position parquets) are fully reproducible from `python -m src.parse` and don't need to live in git. |
| Attack-direction normalisation (`x_norm`/`y_norm`) applied in Stage 2's heatmaps and pass network | Skip it for EDA, only handle it in Stage 3's snapshot rotation | Verified empirically (via `ShotAtGoal` locations) that `TeamLeft` attacks +x and ends swap at half-time. Any spatial aggregation across halves/matches without this blends a team's defensive third into its attacking third — visibly wrong heatmaps otherwise, not just a Stage 3 concern. |
| Pass "distance" in EDA = DFL's categorical `Distance` bucket (short/medium/long) + continuous `PlayAngle`; destination shown as a rough bucket-midpoint projection, explicitly labelled approximate | Treat pass length as continuous from the start | DFL doesn't tag a pass's end coordinate on the event — only origin + angle + a distance bucket. A true continuous length needs the receiving player's synchronised tracking position, which is Stage 3's job. Using an unlabelled approximation here would risk it being mistaken for real data downstream. |
| Pass network built from **completed** passes only | Use all passes (successful + failed), matching Stage 3's "intent" framing | A pass network is meant to show actual ball circulation structure — a different question from Stage 3's "who was this aimed at" (which deliberately keeps failed passes). Conflating the two would misrepresent both. |
| Distance-covered in Stage 2 uses `compute_distance_covered_km` (full-resolution 25Hz re-stream), not the downsampled parquet | Approximate from the stored 8.33Hz parquet with a correction factor | Stage 1 established downsampling undercounts distance ~3x (drops most incremental movement between kept frames). A fudge factor is less defensible than just re-reading the true value; the ~30-40s cost across 5 matches is acceptable for a notebook cell. |
| **Modelling dataset = completed passes only** (~4,725 of the spec's estimated ~6,600), not "both successful and unsuccessful" as originally planned | Keep unsuccessful passes, using `recipient_id` as-is | **Data contradicts the spec's stated plan, confirmed with the user before building anything**: checked all 5,903 passes with a tagged recipient — every unsuccessful one has an *opponent* (the interceptor) as `recipient_id`, every successful one has a teammate (a perfect split, not noise). DFL tags "who touched the ball next," not "who it was aimed at" — there is no intent label for failed passes in this data. The spec's own "same team" filter step already enforces this mechanically; the "include failed passes" framing just doesn't survive contact with how `recipient_id` is actually populated. |
| Synchronisation default = `refined` (mean 3.79m ball-to-event distance, vs `nearest`'s 8.83m) | Default to `nearest` (the literal baseline) | Cost function: within ±3s of the tagged time, minimise `dist(ball, passer) + 2.0 * |time_offset_s|`. The weight (2.0 m/s) was checked for coarse sensitivity (2–5 moved the result <1m on two test matches) rather than exhaustively tuned — we don't have ground truth to tune against. We reach 3.79m vs the paper's reported 2.61m: a real, useful improvement over the 8.83m baseline (paper: 9.37m→2.61m), but not as good as their fuller method — stated plainly, not overclaimed. |
| Velocity smoothing uses a **causal** (trailing) Savitzky-Golay window, not the centred window the spec suggests | Centre the smoothing window on the synced frame | A centred window needs frames *after* the pass moment — exactly the leakage the spec's own rules forbid ("never a later frame"). Trailing window (last ≤5 frames up to and including the synced moment) gets most of the noise-reduction benefit while staying strictly causal. |
| `min_perp_dist_in_lane` left as `NaN` when no opponent's projection falls in the passing lane, not a large sentinel constant | Fill with a large constant (e.g. 50.0) meaning "lane is clear" | An explicit missing value is more honest than a magic number, and defers the imputation choice to Stage 4/5 where it can be made per-model (LightGBM handles `NaN` natively; sklearn models need it decided explicitly) rather than baked in silently upstream. |
| Leave-one-match-out CV used for the Stage 4 baselines too, not just Stage 5's neural net | Evaluate baselines on a single train/test split | The README's final results table compares every model on the same footing; using a different validation scheme for baselines vs. the neural net would make the comparison meaningless. `leave_one_match_out_cv`/`evaluate_predictions` live in `src/baselines.py` and Stage 5 imports them directly rather than reimplementing. |
| `nearest_teammate` scored via `softmax(-dist_to_passer)`, not a one-hot on the argmin | One-hot encode only the argmin candidate, 0 elsewhere | Caught during testing: one-hot makes top-3/MRR meaningless, since all non-winning candidates tie at probability 0 and get sorted arbitrarily rather than by actual distance. Scoring every candidate by (negative) raw distance and softmaxing gives an identical top-1 pick (softmax preserves ranking) while making top-3/MRR reflect a real distance-based ranking, and cross-entropy finite and comparable to the other models. |
| Gradient-boosting baseline includes `dist_rank` as a feature | Exclude it as redundant with `dist_to_passer` | It's derived from information already known before the pass happens (all candidates' distances), not future information — not a leak, just correlated with `dist_to_passer`. Harmless, even useful, for a tree model; excluding a spec-listed feature for no real reason would be an arbitrary restriction. |
| **Shared-weight MLP scorer + softmax over candidates**, not a fixed-N (e.g. 10-way) classifier | A fixed classifier with one output per "candidate slot" | Rosters change between matches and within them (substitutions, red cards) — "slot 3" is a different person in every match, so a fixed-N classifier bakes in an assumption that doesn't hold. Scoring every candidate through the *same* small MLP and combining only at the softmax step is permutation-invariant and handles a variable candidate count via padding+mask. It's the DeepSets idea (shared per-element function + symmetric combination) applied to ranking. |
| Stage 4's `leave_one_match_out_cv`/`evaluate_predictions` reused as-is for the neural net (via a `predict_fn(train_df, test_df) -> df` adapter) | Write a separate CV/eval loop for the neural net | Same reasoning as Stage 4: one shared implementation means the comparison table is genuinely apples-to-apples, and a bug fixed in one model's evaluation is fixed for all of them. |
| Neural net: internal train/val split (85/15, **by pass**, random) carved out of the 6 training-fold matches only, for early stopping | No internal validation; train for a fixed epoch count | This is a normal model-selection split, not the leave-one-match-out test fold — it never touches the held-out match, so a random pass-level split here doesn't violate the "never split randomly at pass level" rule (that rule is specifically about the outer train/test boundary, which stays match-based throughout). |
| `min_perp_dist_in_lane`'s `NaN` (Stage 3 left it unresolved on purpose) imputed for the neural net as a fixed sentinel (50.0, well past any real in-lane distance) **plus** an explicit `lane_missing_flag` feature | Impute with the training fold's mean/median | A learned "no opponent in the lane" pattern is more informative than blending it into the middle of the real distribution — the flag lets the network tell "genuinely far but present" apart from "not applicable" instead of guessing from a suspicious large number alone. LightGBM needs no such choice (handles `NaN` natively), which is itself a small practical advantage tree models have here. |
| Feature standardisation (z-score) computed from each fold's **training** data only, applied to val/test with those same statistics | Standardise using the full dataset's statistics upfront | Computing scale/mean from data that includes the held-out match would leak (mild) information about that match into every fold's preprocessing, undermining the leave-one-match-out setup's whole point. |
| Stage 6's `src/viz.py` re-derives raw per-player snapshots (position, velocity) by re-running Stage 3's exact `MatchPositions`/sync/rotation code, rather than reading them from `pass_candidates.parquet` | Store raw per-player snapshots in Stage 3's output for later reuse | `pass_candidates.parquet` only ever needed opponents' *aggregate* features (nearest-opponent distance, etc.), never their raw coordinates, so it doesn't have them — and drawing "all 22 players" needs them back. Re-running the same synchronisation code (not a separate approximation of it) guarantees the picture matches the exact moment the model was trained/evaluated on. |
| Stage 6 retrains one demo leave-one-match-out fold (J03WOH held out) instead of loading a saved model | Persist trained model weights from Stage 5's `train.py` for reuse | Stage 5 only needed metrics from each of its 7 folds, not the weights, so nothing was saved to load. Retraining one fold (~7s) inside the notebook is simple, reproducible, and guarantees the demo predictions are genuinely out-of-sample (the model never saw J03WOH), matching the actual evaluation protocol rather than a separately-trained "for show" model. |
| `notebooks/00_full_walkthrough.ipynb` sets `os.environ["OMP_NUM_THREADS"] = "1"` as its very first line, before any other import | Leave threading at its default | **Found by debugging a real hang, not a preemptive guess**: LightGBM (Stage 4) and PyTorch (Stage 5) each bundle their own OpenMP runtime; training both, multi-threaded, in the same process deadlocks on macOS — confirmed directly by reproducing it outside any notebook (LightGBM finishes in <1s, the very next PyTorch training call then hangs indefinitely, using near-0% CPU, i.e. genuinely blocked, not slow). Forcing single-threaded OpenMP fixes it at no real speed cost given how tiny this model/data is. Relevant to anyone extending this codebase to run LightGBM and PyTorch training back-to-back in one process elsewhere. |

## Sanity checks (Stage 1)

Full output: see `src/parse.py`'s `main()` — re-run anytime with `python -m src.parse`.

- **Event chain structure**: no branching child chains across any of the 7 matches — the
  "concatenate the single nested child chain" approach (`KickOff_Play_Pass`, etc.) holds
  empirically, not just in the spec's example.
- **Raw event count**: J03WMX 1,715 vs paper's 1,840 (6.8% diff, within the 10% tolerance).
- **Coordinate transform**: verified directly on 2/7 matches (J03WMX 0.25m, J03WR9 0.82m —
  both ≤1m). The other 5 matches show 2.8–25.7m gaps between the kickoff event's tagged
  position and the ball's tracked position at that timestamp. This is **not** a transform
  bug — the events file is not strictly chronologically ordered (confirmed: 104/1280
  out-of-order timestamp steps in one file) and event timestamps carry real tagging jitter,
  exactly the phenomenon the paper documents (mean offset −0.37±1.82s, up to 27s worst
  case) and that Stage 3's synchronisation step exists to correct. Two clean matches is
  enough to confirm the *formula* is right; the rest is a preview of the Stage 2/3 problem.
- **Recipient IDs**: 0/5,903 pass recipients failed to resolve to a player in that match's
  roster.
- **Distance covered**: 71/76 full-match, non-red-carded outfield starters fall in 9–13km.
  The 5 remaining are all in one match (J03WN1, 8.5–8.9km) — that match has the most
  cautions of the 7 (11, vs a handful in the others), consistent with a stoppage-heavy,
  lower-tempo game rather than a parsing bug; verified by cross-match comparison that
  J03WN1's *whole* distance distribution sits ~12% below the others, not just these 5
  individuals.
- **Ball position bounds**: all matches show a few metres of overshoot past the nominal
  ±52.5/±34 pitch edges (up to ~55.5m / ~38m) — plausible tracking margin for a ball still
  in frame just past a boundary line (throw-ins, goal kicks, behind the goal), not a unit
  bug. Hard-fail threshold set at ±65/±45 (would indicate a real coordinate-system error);
  none triggered.

## Stage 2 — EDA highlights (Fortuna Düsseldorf, 5 matches)

Full notebook: `notebooks/01_eda.ipynb` (executed, outputs saved). Headline findings:

- Pass success rate drops sharply with distance: short 88% → medium 84% → long 43% —
  clean and monotonic, a good sanity check on the `Distance`/`Evaluation` fields and a
  strong prior for `dist_to_passer` as a baseline feature.
- Success rate varies by position from 66% (RM) to 92% (IVZ) — deep/central roles play
  safer, shorter passes; wide/forward roles attempt riskier ones.
- PageRank in the pass network surfaces different players than raw pass volume: the
  right centre-back, right-back, and keeper top PageRank (structurally central to
  build-up), while the keeper alone leads on raw pass count.
- **Synchronisation audit reproduces the paper's own finding**: mean ball-to-event
  distance of 8.6m (median 6.6m) across Fortuna's 2,378 passes, in line with the paper's
  reported 9.37m unsynced mean — a strong end-to-end validation of the parsing pipeline,
  and the number Stage 3's synchronisation methods will try to beat.
- Minutes-vs-distance is linear as expected, with the goalkeeper (~5.3 km/90) clearly
  separated from every outfield starter (9-11.5 km/90).

## Stage 3 — modelling dataset

`src/build_dataset.py` → `data/processed/pass_candidates.parquet`. Full sanity-check
output: re-run anytime with `python -m src.build_dataset`.

**Pass selection funnel** (all 7 matches):

| Step | Passes remaining | Dropped |
|---|---|---|
| `_Pass` / `_Cross` events | 6,062 | — |
| has a tagged `recipient_id` | 5,903 | 159 (no recipient tagged) |
| passer and recipient same team (== completed passes; see decisions table) | 4,731 | 1,172 (cross-team = unsuccessful passes) |
| passer & recipient tracked on pitch at event time | 4,725 | 6 (sub-timing / red-card edge cases) |

**Final dataset**: 4,725 passes → 46,884 candidate rows (9.9 candidates/pass on average;
min 9, max 10 — the occasional dip below 10 is a red card or a substitution-boundary
gap, as anticipated).

**Sanity checks** (all pass):

- Exactly one `is_recipient=1` row per pass: 4,725/4,725.
- Candidates per pass never exceeds 10 (11 - passer).
- **Leakage tripwire**: max `|corr|` with `is_recipient` across all 18 features is 0.257
  (`dist_rank`) — well under the 0.9 flag threshold. Signs are all football-sensible:
  closer to the passer (−0.23), faster-moving candidate (+0.06), more open lane
  (`min_perp_dist_in_lane` +0.20), more opponents in the lane (−0.14) all point the
  direction intuition would predict — a good sign the features are informative without
  any of them secretly encoding the answer.
- `NaN` appears only in `min_perp_dist_in_lane` (844/46,884 rows, ~1.8%) — exactly the
  passes where no opponent's projection falls between passer and candidate. Left as
  `NaN` rather than a sentinel value (see decisions table).

Runtime: full 7-match build in ~34s on CPU.

## Stage 4-5 — baselines & model results

`src/baselines.py` → `data/processed/baseline_fold_results.csv`. Re-run anytime with
`python -m src.baselines` (~10s on CPU for all 3 models × 7 folds).

All three, and Stage 5's neural net, are evaluated identically: **leave-one-match-out
CV** (7 folds — never a random pass-level split, which would leak team identity,
tactical setup, and adjacent passes from the same possession across train/test), scored
with the same four metrics via a shared `evaluate_predictions` helper.

### Results (mean ± std across 7 folds)

| Model | top-1 acc | top-3 acc | MRR | cross-entropy |
|---|---|---|---|---|
| Nearest teammate (argmin distance) | 0.272 ± 0.039 | 0.635 ± 0.038 | 0.496 ± 0.030 | 7.473 ± 0.828 |
| Logistic regression (distance only) | 0.272 ± 0.039 | 0.635 ± 0.038 | 0.496 ± 0.030 | 1.945 ± 0.061 |
| LightGBM (full feature set) | 0.469 ± 0.033 | 0.806 ± 0.030 | 0.655 ± 0.026 | 1.485 ± 0.098 |
| **Neural net, all_matches** (7 folds, 4,725 passes) | **0.477 ± 0.025** | **0.818 ± 0.026** | **0.662 ± 0.021** | **1.469 ± 0.076** |
| Neural net, single_team (5 folds, 1,919 passes) | 0.478 ± 0.025 | 0.817 ± 0.012 | 0.659 ± 0.014 | 1.504 ± 0.015 |

**Reading these numbers:** nearest-teammate and logistic regression are **identical**
on every ranking metric (top-1/top-3/MRR) — expected, since both rank candidates purely
by distance, a monotonic relationship softmax preserves regardless of scale. They differ
only on cross-entropy (7.47 vs 1.95): logistic regression *fits* a scale and intercept
by MLE, while nearest-teammate uses a fixed "1 logit unit per metre" with no calibration
— the gap is purely a calibration story, not a ranking one, and is itself a useful thing
to be able to say plainly. LightGBM (all 17 engineered features) is a clear step up over
distance alone — roughly +20pp top-1 accuracy — confirming the Stage 3 features carry
real signal beyond raw proximity.

**The neural net edges out LightGBM on every metric (+0.8pp top-1, +1.2pp top-3, +0.7pp
MRR, marginally lower cross-entropy) — but stated plainly, per the spec: that gap is
well within one fold's standard deviation (±0.025–0.033 on top-1 alone), so it is
*not* a result I'd claim as "the neural net wins." With 7 folds, a difference this size
is statistically indistinguishable from noise. A back-of-envelope estimate (the gap is
roughly a third of the per-fold std, and detecting an effect that small at conventional
significance needs on the order of `(std/effect)² ≈ 9x` the folds/data) says this would
need on the order of 15–25 matches — 2–3x what's available here — before I'd trust a
claim that the neural net actually outperforms LightGBM on this task. What *is* a solid,
supportable conclusion: the neural net **matches** the gradient-boosted baseline, using
the same features, with none of GBM's non-differentiability (useful if this were ever
extended to a learned, end-to-end pipeline) — not that it beats it.**

**TRAIN_SCOPE gap — a genuine surprise, reported as found**: training on all 7 matches
(4,725 passes) vs. Fortuna's 5 matches alone (1,919 passes, 2.5x less data) produces
*nearly identical* top-1/top-3/MRR (0.477 vs 0.478, 0.818 vs 0.817, 0.662 vs 0.659) —
essentially no data-volume benefit at this scale, for this feature set and model
capacity. Only cross-entropy calibration is measurably worse with less data (1.504 vs
1.469). I expected more data to help more than this; it doesn't, which itself says the
17-feature representation may already be close to saturating what a model this size can
extract from it — more data alone likely isn't the lever that would move these numbers
further (see "What I'd do next").

## Stage 6 — visualisation

`src/viz.py` (snapshot/trajectory/plotting helpers) + `notebooks/02_pass_visualisation.ipynb`
(executed, outputs saved). Demo match: **J03WOH** (Fortuna Düsseldorf 4-0 SSV Jahn
Regensburg), held out of training for a genuine leave-one-match-out fold — every
prediction shown is on a match the model never saw during training.

All 5 planned pieces, plus the optional GIF:

1. Single pass snapshot (all on-pitch players, velocity arrows, pass arrow).
2. Model prediction overlay: 3 confidently-correct + 3 confidently-wrong cases from the
   J03WOH fold (top-1 accuracy on this match: 42.0%, 581 passes — in line with the
   per-fold CV numbers in the results table above).
3. 3-second trajectory trails ending at the pass moment.
4. Pass network for J03WOH alone (not aggregated across matches, unlike Stage 2's).
5. `animate_sequence()`: a ~5s GIF (2s before → 3s after the pass) — attempted despite
   being marked "nice to have, skip if short on time," since the infrastructure from
   items 1-3 made it cheap (~1.5s to render).

**The failures (item 2) are genuinely the most interesting slide, as the spec
predicted**: all 3 confidently-correct cases are isolated out-balls/switches of play —
an easy call once a candidate is far from everyone else. All 3 confident mistakes have
the *same* failure shape: the model still gravitates to an isolated, uncontested
candidate, but picks the wrong one when two such options exist — a concrete symptom of
the feature set having no signal about the passer's body orientation or gaze, which the
event/tracking data doesn't capture. This is the clearest, most specific "what I'd do
next" finding in the whole project.

## Limitations

Written honestly, per the brief:

- **Seven matches is a small sample.** All leave-one-match-out std devs in the results
  table are non-trivial relative to the between-model gaps — the headline "neural net
  edges out LightGBM" result is explicitly *not* one I'm claiming as real (see Stage 5),
  and the same caution applies to any other close comparison in this project. Nothing
  here should be read as "the neural net architecture is better than gradient boosting
  for this task," only "on this data, they perform about the same."
- **Synchronisation has no video ground truth.** The `refined` method (mean 3.79m) is
  optimising a proxy — ball-close-to-passer-near-the-tagged-time — not validated against
  an actual video frame showing the true kick moment. The paper's own 2.61m used a more
  sophisticated cost function than the two-term one here; the gap between 3.79m and
  2.61m may partly be that, or may be irreducible noise in this data. I can't tell which
  without ground truth, and didn't claim to.
- **Recipient labels are the tagger's judgement of an outcome, not observed intent.**
  Even restricted to completed passes (see Stage 3's central finding), `recipient_id` is
  "who touched the ball next," which usually but not always coincides with "who the pass
  was aimed at" (a slightly mis-hit pass that still reaches a teammate is tagged as
  successful to that teammate, not to whoever was actually the target).
- **The gradient-boosted baseline matches the neural net, stated plainly**: LightGBM
  (0.469 top-1) and the neural net (0.477 top-1) are statistically indistinguishable
  given 7 folds — see Stage 5 for the back-of-envelope sample size that would be needed
  (~15-25 matches) to tell them apart with any confidence. On this project's evidence, a
  simpler, faster-to-train, no-GPU-needed gradient-boosted model is an entirely
  reasonable choice over the neural net; the latter's actual advantage is architectural
  (differentiable, extensible to end-to-end pipelines like Stage 6's optional GAT
  variant), not measured predictive performance.
- **No test of tactical generalisation.** All 7 matches are single-season Bundesliga
  1st/2nd division games; nothing here says whether these features/model would transfer
  to a different league, a back-3 system, or a different possession style.
- **The visualisation failure analysis (Stage 6) is anecdotal.** Three examples is a
  vivid illustration of a hypothesis (isolation-seeking without orientation/gaze
  signal), not a statistically validated explanation of the model's error mode.

## What I'd do next

With more data or more time, roughly in priority order:

1. **More matches, same pipeline.** The single strongest lever for the two "we can't
   tell" results above (neural net vs. LightGBM; `TRAIN_SCOPE` gap) is simply more
   leave-one-match-out folds — everything built here (parsing, sync, features, CV
   harness) already scales to more matches with no code changes.
2. **Passer orientation/gaze proxy.** Stage 6's failure analysis pointed at a specific,
   fixable gap: nothing in the feature set represents where the passer is *facing* or
   *looking*. A cheap proxy — passer's own recent velocity direction as a feature, or
   the angle between passer's velocity and each candidate — could directly test the
   "model can't tell which isolated player the passer is actually playing to" hypothesis.
3. **Video-validated synchronisation**, even for a handful of passes, to know whether
   the `refined` method's 3.79m residual is mostly tagging noise or mostly a limitation
   of the two-term cost function.
4. **The optional GAT variant** (players as graph nodes, attention over teammates *and*
   opponents jointly) — not attempted here per the spec's own "don't start until
   everything else runs" framing, but the natural next architecture once more data
   justifies a more expressive model.
5. **An RNN/trajectory-encoding variant** was discussed but not built this session
   (deferred back to Stage 6 planning) — replacing the single-snapshot velocity feature
   with a short learned encoding of each candidate's last ~3s of movement is a natural
   follow-on to try, especially alongside item 2 above.

