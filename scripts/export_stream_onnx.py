"""Stream ONNX export — NOT IMPLEMENTED.

Qwen3Guard-Stream exposes a stateful custom API
`stream_moderate_from_ids(ids, role, stream_state)` with two classification
heads on top of Qwen3's last hidden state. ONNX export of this path requires
either:

  (a) splitting into (backbone ONNX) + (Python head) + manual KV-cache
      plumbing, or
  (b) tracing a `forward(input_ids, past_kv) -> (risk_logits, cat_logits,
      new_past_kv)` wrapper and exporting that.

Both are non-trivial and are deferred. The corresponding line in
`run_all.sh` is commented out with a pointer to this file.

Running this script prints the explanation and exits non-zero so it fails
loud if accidentally wired into a pipeline.
"""
import sys

MSG = __doc__


def main() -> int:
    print(MSG)
    return 2


if __name__ == "__main__":
    sys.exit(main())
