# One Clock, Three Learners

A conserved developmental schedule in mammals, children, and language models.

Comparative neurobiology holds one quantitative law of developmental timing. The order in
which neurodevelopmental events happen stays fixed across mammals, while a single rate
parameter for each species absorbs almost everything else. This repository fits that law,
without modifying the estimator, to three systems that share no substrate, and it holds in
all three.

The interesting part is not that the law fits. It is how the variance splits once it does.
In mammals the species rate alone carries an R-squared of 0.788, so biological development
runs on a clock. In children the language rate carries 0.078 and across model scales the
rate carries 0.039, so both learning systems run on an order instead. A neural network has
neither a genome nor a maturational clock, which makes it the condition biology cannot
supply, and the regularity still shows up there.

## Results the code reproduces

| system   | cells | units | events |   rho | rate-only R2 |     z vs null | rate span |
|----------|------:|------:|-------:|------:|-------------:|--------------:|----------:|
| mammals  |   372 |    10 |     95 | 0.989 |        0.788 |          19.2 |      4.64 |
| children |  5120 |    15 |    657 | 0.807 |        0.078 |          46.0 |      1.13 |
| machines |   234 |     6 |     48 | 0.882 |        0.039 |          10.2 |      5.73 |

The null is a within-unit permutation that shuffles which of a unit's observed times belongs
to which of its events. It preserves the marginal distribution of times inside each unit and
the number of observations attached to each event, so it destroys the shared ordering and
leaves everything else alone.

A rank concordance check drops the additive model entirely and returns 0.873, 0.738, and
0.845 against nulls within 0.001 of 0.500, so the finding does not depend on the regression
form. Holding each unit out and anchoring its rate on three of its own events places an
unseen mammal at a median relative error of 8.1 percent against the observed day.

## The estimator

    log t_ie = alpha_i + beta_e + eps_ie

Fitted by ordinary least squares with unit and event dummies at k = 0. The event term is the
schedule that every unit shares and the unit term is the single number that stretches or
compresses it. For the mammalian panel only, the primate-by-cortical and primate-by-limbic
interactions are added back, because the published mammalian model carries them. That
interaction returns cortical events in macaque and human at 1.271 times late with a bootstrap
interval of 1.164 to 1.387.

## Running it

Requires Python 3.9 or newer with numpy, pandas, scipy, matplotlib, and pyarrow. Scripts read
their inputs as bare filenames from the working directory, so run them from the repository
root.

    pip install -r requirements.txt
    python final_results.py

That prints the main table, the fitted rate for every unit in all three systems, and the
permutation null statistics. It takes about a minute.

Other entry points worth knowing. `controls.py` runs the geometry, rival-model, and ceiling
controls. `adversarial.py` reports the adjusted R-squared and the parameter-to-observation
ratio for each panel, including the complete-case machine table. `loo.py` and `loo_anchors.py`
run the leave-one-out transfer and vary how many anchors a held-out unit gets. `domains.py`
tests whether a domain label the fit never saw organises the recovered schedule. `machines.py`
builds the machine panel from the raw BLiMP sweep. `figs.py` regenerates both figures.

## Data

`panel_mammals.csv`, `panel_children.csv`, and `panel_machines.csv` are the three fitted
panels, each carrying observed time, predicted time, unit, and event.

`event_data.csv` is the mammalian source table, drawn from the `ttime` package version 1.2.5
in the CRAN archive. `aoas.feather` and `items.feather` are the Wordbank age-of-acquisition
and item tables. `blimp_by_step_all.csv` holds the raw sweep, which is 6,834 paradigm by
checkpoint accuracies from scoring 67 BLiMP paradigms on six Pythia scales at seventeen
checkpoints each, and `blimp_by_step_dedup.csv` repeats it on the deduplicated Pythia ladder.
The `machine_acq_*.csv` files give acquisition times at the 0.65, 0.70, and 0.75 criteria, and
the paper reports 0.70 throughout.

The machine sweep is the only part that is expensive rather than instant, since it took 18.85
hours of Kaggle CPU across two sessions. Everything downstream of `blimp_by_step_all.csv` runs
in seconds on a laptop.

## Matched CDI words

`matched/` times children and Pythia on the same 696 CDI words. Child times are the American
and Australian English WS production ages in `child_aoa.tsv`. Machine times are surprisal
acquisition crossings for Pythia 14M to 410M over 22 checkpoints and ten seeds, computed from
the gzipped surprisal tables in the same folder, so no model is run. The analysis plan is in the
docstring of `matched_cdi.py`, including the two deviations made after crashes and before any
primary value was seen.

    cd matched && pip install wordfreq && OUT_DIR=out python matched_cdi.py

`out/` holds the results the paper reports: `matched_results.json`, the per-word table
`matched_words.csv`, and `log.txt`. The recorded SHA-256 hashes are of the decompressed
`.jsonl` files. American English child and model orders correlate at -0.211 over 648 words,
and the correlation comes from lexical class, since without function words it is -0.037.

## Robustness and uncertainty

`gaps/gaps.py` runs the checks the paper reports for the stretch, the decomposition and the
machine crossings from files already in the repository root, so no model is run: 10-fold
cross-validation of the additive and stretch fits, held-unit baselines, a 200-draw binomial
resampling of every BLiMP accuracy, event-resampling intervals for the variance decomposition
and the multipliers, three alternative crossing rules, a censored (Tobit) likelihood, the 500-draw
lexical-norm bootstrap, a power simulation for event pairing, the matched-window deduplication
comparison, a refit without extrapolated crossings, and the spiny-mouse and lexical restart checks.

    python gaps/gaps.py

It takes about 3.6 hours on four CPU threads, most of it in the binomial resampling, and writes
`gaps_results.json` to the working directory. `gaps/out/` holds the run the paper reports and its log.

`gaps/seeds.py` repeats the stretch fit across ten PolyPythias seeds per size (14M to 410M) and the
160M weight-only and data-only seeds, using the BLiMP accuracies in `matched/`. It runs in a few
minutes and writes `seeds_results.json` (reported run in `gaps/out/`, log in `seeds_log.txt`).

    python gaps/seeds.py

`gaps/seeds_iso.py` reruns the seed check with both the logistic rule and the shape-free isotonic
crossing rule, from step 128 and step 16, and compares crossings on cells dated in both windows.
It writes `seeds_iso_results.json` (reported run in `gaps/out/`, log in `seeds_iso_log.txt`).

    python gaps/seeds_iso.py

`gaps/window.py` splits the drop in the seed-panel stretch when checkpoints before step 128 are
added into cell selection and crossing placement, by refitting the pooled seed table on matched
cell sets (`window_results.json`).

    python gaps/window.py

`gaps/early/` tests the main panel's window. `early.py` scores all 67 BLiMP paradigms at steps 16 to
512 (its own 200-pair sample per paradigm, since the main panel's sample isn't in this release; set
TAG to `small` for 14M to 410M or `1b`). It downloads Pythia checkpoints and needs about eight CPU
hours for the five small sizes and about eleven for 1B. `extend.py` checks agreement with the released accuracies and refits
the crossings and the stretch with the early checkpoints added; `out/` holds the reported run.

The rest of `gaps/early/` asks why the stretch depends on the window. `extend2.py` repeats the
window comparison under six crossing rules with 100 binomial resamples. `simwin.py`, `simwin2.py`
and `simmech.py` (set VARIANT to `slope`, `resid`, `resid_shuf`, `stretch` or `stretch_resid`)
simulate panels with shift-only or stretched true crossings and measure them from both windows.
`mechdiag.py` locates the window effect in the real curves. `isorule.py` dates crossings with a
shape-free isotonic rule and tests two-phase curves, and `isoadded.py` examines the cells that only
the early checkpoints date. Under the isotonic rule, crossings observed in both windows are almost
all identical and keep the same stretch (gain 0.103 against 0.105); the cells crossing before step
128 don't follow it. Each script writes one JSON file to `out/` and runs on CPU in under two hours.

## Caveats the paper states

The machine table is the smallest at 234 cells over six units, and its permutation null is ten
times wider than the mammalian one. Wordbank gives a rate for each language and never for each
child, so this does not answer the original question about children as individuals. BLiMP
accuracy is a thin proxy for grammatical knowledge. Every machine number comes from one
architecture family on one corpus and its deduplicated variant, and we did not measure how
often each construction appears in that corpus, so rare phenomena may arrive late because they
are rare.

## Citing

A citation will go here once the paper is public.

## License

MIT for the code. The mammalian data come from the `ttime` package and the child data from
Wordbank, and both carry their own terms.
