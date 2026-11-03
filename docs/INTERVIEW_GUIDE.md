# A short technical walkthrough

## Explain the project

“RankForge is a recommendation service. It learns a shared space for users and items, retrieves unseen candidates, reranks them with profile signals, and serves them through an API and browser dashboard. I compare it with simple baselines and keep feedback linked to the exact exposure that produced it.”

Describe this as a portfolio project. Do not imply employment, customers, production scale, or business impact that has not been measured.

## Questions to prepare

1. **Why two stages?** Retrieval narrows the candidate set; ranking and diversity cost more per candidate. Exact retrieval is adequate for 320 items. Candidate Recall@80 shows what ranking cannot recover.
2. **Why pairwise loss?** Positive-only implicit data does not supply calibrated negative labels. Pairwise loss trains relative preference, though unobserved does not mean disliked.
3. **How do you prevent leakage?** Global time windows; training-only profiles and negatives; validation-only checkpoint/blend selection; test labels cannot alter the selected artifact. See the regression test that changes test labels.
4. **Why include a category baseline?** It catches complexity that fails to help. Here it slightly beats neural NDCG, so shipping the complex model is not automatically justified by this dataset.
5. **What does coverage mean?** Fraction of catalog items appearing in users' top ten. Higher coverage alone does not imply higher relevance or fairness.
6. **What happens on a save?** Validate exposure ownership and item membership; deduplicate; persist; use it in the request-time category profile and seen-item filter. Retraining is separate.
7. **Does the A/B ledger prove uplift?** No. It provides stable assignment and attributed descriptive counts. A real experiment needs identity, exposure verification, sample planning, controlled settings and statistical analysis.
8. **What would you change at scale?** Measure first, then consider approximate vector search, a shared transactional database, a feature store, scheduled retraining, stronger authorization and deployment monitoring.
9. **What did the tests find?** API dependency annotations initially caused valid requests to be treated as missing query parameters. Endpoint tests caught this before publishing. Test current behavior and explain the fix rather than hiding the debugging process.

## Files worth reading first

`data.py` → `model.py` → `train.py` → `engine.py` → `store.py` → `api.py` → `tests/test_pipeline.py`.

Understand each stage before adding the project to a CV. Good follow-up work is a licensed real dataset, repeated-seed uncertainty analysis, and an evidence-based decision between the category and neural models.
