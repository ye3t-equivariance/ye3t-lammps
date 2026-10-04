"""A real-only YACE loader rejects a scalar with unmatched magnetic rows."""

from pathlib import Path
import subprocess
import sys
import tempfile


evaluator, model_path, environment_path = sys.argv[1:]
source = Path(model_path).read_text(encoding="utf-8")
radial_row = "      ns: [2, 2]\n"
if source.count(radial_row) != 1:
    raise SystemExit("The radial YACE fixture has no two-slot scalar row to alter")
source = source.replace(radial_row, "      ns: [1, 2]\n", 1)
original = "        - 0.5773502691896257\n"
offset = source.rfind(original)
if offset < 0:
    raise SystemExit("The radial YACE fixture has no conjugate scalar row to alter")
altered = source[:offset] + source[offset:].replace(
    original, "        - 0.3773502691896257\n", 1,
)
with tempfile.TemporaryDirectory(prefix="ye3t_nonreal_yace_") as temporary:
    path = Path(temporary) / "nonreal.yace"
    path.write_text(altered, encoding="utf-8")
    completed = subprocess.run(
        [evaluator, str(path), environment_path],
        capture_output=True, text=True, timeout=30, check=False,
    )
if completed.returncode == 0 or "complex-conjugation reality" not in completed.stderr:
    raise SystemExit(
        f"Non-real YACE was not rejected by the reality check: "
        f"returncode={completed.returncode}, stderr={completed.stderr!r}"
    )
