# Model and data card

**Purpose:** demonstrate end-to-end implicit-feedback recommendation engineering for a fictional catalog. Not intended for decisions about credit, hiring, health, insurance, or other high-impact eligibility.

**Training fixture:** 400 synthetic users, 320 synthetic items in eight generic categories, 40 unique positive item interactions per user. A seeded latent preference generator produces observations. It does not emulate real shopping conversion, item exposure propensities, or changing preferences.

**Time protocol:** first 28 observation days for training, next six for validation, final six for testing. Exact global boundaries and dataset SHA-256 are in the report. No random interaction split. Repeated positives are rejected by the import contract to make unseen-item evaluation explicit.

**Model:** two MLP towers, 24-dimensional output, normalized similarity, pairwise logistic objective with AdamW and clipped gradients. Training negatives come from training-available items unseen by the user at the training cutoff. We do not remove future positives from training negatives: doing that would leak future labels. Checkpoint and ranking blend are selected on validation NDCG@10.

**Evaluation:** user-macro Recall@10 and NDCG@10 over the entire eligible unseen catalog, catalog coverage@10, and candidate Recall@80. The frozen model uses training profiles; validation positives are excluded from test candidates without updating profiles. Reported timing is measured locally in-process and not a service-level objective.

**Findings:** neural ranking improves substantially over global popularity on this synthetic fixture. Category-only ranking is a strong baseline and scores slightly better on test NDCG. Neural catalog coverage is higher. No confidence interval, causal uplift, statistical superiority, or real-user satisfaction result is claimed.

**Failure modes:** popularity and observation bias, sparse history, unseen items without trained ID embeddings, weak generalization to out-of-distribution behavior, stale catalog availability, and feedback manipulation. A category description is not a content embedding; the system has no text/image understanding. There is no price, stock, safety or legal eligibility policy layer.

**Privacy:** shipped IDs and interactions are synthetic. Imported user data is written into training histories in the model artifact and into the local feedback ledger. Use pseudonymous IDs and retain artifacts privately when working with real data; review and sanitize before publishing. Code MIT license does not grant rights to third-party imported datasets.

**Operational status:** validated locally with tests and browser interaction. Container and CI configuration are included; actual CI/container outcomes must be checked in their execution environments. No cloud production deployment is included.
