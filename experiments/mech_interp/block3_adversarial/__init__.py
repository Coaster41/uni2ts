"""Block 3 — adversarial robustness / boundary-vulnerability probe.

Tests the claim of arXiv 2505.19397 ("points near the forecast boundary are most
vulnerable") on our own models, and asks whether that vulnerability is an
artifact of causal next-token readout (see `.claude/HANDOFF_ADVERSARIAL_BOUNDARY.md`).
"""
