"""Exercise projected V4 loading without requiring a Python compiler package."""
import copy
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import tempfile


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def main():
    library = ctypes.CDLL(sys.argv[1])
    library.ye3t_tagged_open.argtypes = (ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t)
    library.ye3t_tagged_open.restype = ctypes.c_void_p
    library.ye3t_tagged_close.argtypes = (ctypes.c_void_p,)
    original = json.loads(Path(sys.argv[2]).read_text())
    assert original["schema"] == "ye3t_tagged_cauchy_slice_v4"
    error = ctypes.create_string_buffer(4096)

    def open_model(path, expected=None):
        handle = library.ye3t_tagged_open(str(path).encode(), 0, error, len(error))
        if handle:
            library.ye3t_tagged_close(handle)
        if expected is None:
            assert handle, error.value.decode()
        else:
            assert not handle, "Accepted tampered model: " + str(path)
            assert expected in error.value.decode(), error.value.decode()

    with tempfile.TemporaryDirectory(prefix="ye3t_v4_loading_") as directory:
        path = Path(directory) / "model.json"
        # Pretty and compact encodings must yield the same accepted artifact.
        for indent in (None, 2):
            path.write_text(json.dumps(original, sort_keys=True, indent=indent))
            open_model(path)
        for case in ("offline_proof", "forward", "adjoint", "coherent_schedule"):
            modified = copy.deepcopy(original)
            if case == "offline_proof":
                modified["compiler_artifact"]["payload"]["unused_offline_proof"] = {"tampered": True}
                expected = "compiler artifact self-hash mismatch"
            else:
                key = "adjoint_terms" if case == "adjoint" else "terms"
                modified["schedule_binding"]["payload"][key][0]["coefficient"] += 0.125
                expected = "payload hash mismatch"
                if case == "coherent_schedule":
                    program = modified["schedule_binding"]["payload"]
                    program["program_hash"] = digest({k: v for k, v in program.items() if k != "program_hash"})
                    modified["schedule_binding"]["hash"] = digest(program)
                    expected = "schedule differs from its compiler commitment"
            modified["self_hash"] = digest({k: v for k, v in modified.items() if k != "self_hash"})
            path.write_text(json.dumps(modified, sort_keys=True))
            open_model(path, expected)
        # A later load of the same pathname must read its current contents.
        path.write_text(json.dumps(original, sort_keys=True))
        open_model(path)
        path.write_text('{"schema":')
        open_model(path, "unexpected end of input")
    print("V4 projected loading preserves integrity and observes file replacement.")


if __name__ == "__main__":
    main()
