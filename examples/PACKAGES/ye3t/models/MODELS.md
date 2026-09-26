# Shipped Ta model records

All `.yace` files use the standard ML-PACE linear format and `metal` units.
The two rank-through-8 models were fit outside LAMMPS to energies and forces
from the Ta SNAP dataset. Neither model includes a ZBL term.

## Physically qualified rank-through-8 models

### `ta_l8_compact`

- Role: matched compact control and direct/block/AUTO example.
- Catalogue: 58 functions, maximum rank 8, maximum angular degree 4.
- `.yace` SHA-256:
  `a8da0b71a05bd8a4fb356216ba792ffb7af05db07916995dcaf5f114d392fd56`.
- Manifest SHA-256:
  `d4058da0c0af5a6d40481008dd455e03e7c2f8cdfd0c752750040698b008cc15`.
- Validation RMSE: 0.07631 eV/atom energy and 0.16605 eV/A force.
- BCC equilibrium lattice constant: 3.31611 A.
- BCC bulk modulus: 230.87 GPa.

### `ta_l8_full`

- Role: larger main rank-through-8 model and direct/coupled-product example.
- Catalogue: 61 functions, maximum rank 8, maximum angular degree 7.
- `.yace` SHA-256:
  `3f194cf921c5ebf7a9333c2fb9425478484f405e5a9fee9bf25e44106f7962c8`.
- Manifest SHA-256:
  `d711a8ae617b67e1da0d94297796eda2caa72c25ef6c25028b4a48bf4f06cee5`.
- Validation RMSE: 0.06593 eV/atom energy and 0.16422 eV/A force.
- BCC equilibrium lattice constant: 3.31777 A.
- BCC bulk modulus: 230.07 GPa.

Both models passed the workflow's BCC EOS, cubic elastic/Born stability, and
bounded 300 K NVE gates at 0.25, 0.5, and 1.0 fs. Those checks do not establish
close-collision behavior or universal transferability.

The `provenance.json` files are optional fit-time records and are not read
by LAMMPS; their `status` and benchmark flags describe the state at export,
and the later physical qualification and timing evidence is recorded in
`REFERENCE_RESULTS.md`. The plan bundles are also optional. The compact
payload hashes are:

- `execution_plan.json`:
  `4c8b7d3ec2499f12feb97a3ac645878f4c84b7175f880fe24914209b195ce2c5`;
- `yace_function_map.json`:
  `700bc9540bdee48e3e6ccc85449edc7a16f5d32f79ada6da77737ab9900787bf`.

The full-model plan bundle retains the explicit direct route for every
function and adds one compiler-certified coupled-product alternative. Forced
`coupled_product` therefore evaluates that eligible row with its coupled DAG
and the remaining rows through the exact direct residual. Its payload hashes
are:

- `execution_plan.json`:
  `127c154e61708fdcd1717f7c9bdf75dbc5e86f24d010c4b6350ea19a689042ed`;
- `yace_function_map.json`:
  `f64f16172965ab30df9d32adc58e14d404209991f4c4924a20c8a8a2c78bcc48`.

## Rank-16 execution fixture

`ta_l8_h16` starts from the fitted compact control and appends tiny
homogeneous rank-16 sentinel coefficients. It tests exact forward/adjoint
execution beyond rank 8; the added coefficients were not fitted to the Ta
dataset.

| bundle | added rank | model SHA-256 | manifest SHA-256 |
|---|---|---|---|
| `ta_l8_h16` | 16 | `f9391945097d24bb5992c9dbc4fb7ea52c019727245e5349c1d665929af3657f` | `af92cab5e1c5831672a8d3979821e214fbfa26eb4067f3cbc1d5433a423fa8a5` |

Its plan uses compiler-certified homogeneous symmetric-power relations, so
direct and block execution are expected to agree within floating-point
tolerance. Do not use this bundle for material MD or accuracy claims.
