"""Keep command-line drivers out of pytest collection.

CTest invokes four of these scripts with executable, model, and fixture
arguments; the saved-model parity script is run manually with similar
arguments. Pytest collection cannot provide those command-line inputs.
"""

collect_ignore = [
    "test_density_mean_property_tamper.py",
    "test_mean_property_lammps.py",
    "test_mean_property_saved_parity.py",
    "test_mean_property_tamper.py",
    "test_nonreal_yace_rejection.py",
]
