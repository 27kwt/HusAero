# Current output categories

Use `drone`, `bird`, `airplane`, `helicopter` as the unified categories. `output_map.json` records the explicit mapping from the existing five-output checkpoints; the original model names and native manifests are preserved.

The models have not been retrained for four outputs. `tools/predict_unified.py` adapts PC predictions and conservatively abstains on low/conflicting hypotheses. Its default drone focus does not assert the identity of other aircraft; raw native and mapped classes stay in the JSONL log. Unknown is a display state, not an extra trained class. A high model score can still be wrong. The existing Pi HEFs and measured historical scores remain native RGB5.
