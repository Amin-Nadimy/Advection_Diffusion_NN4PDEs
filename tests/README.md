# What the tests check

Run them from the repository root with `pytest -v`. They take about two seconds on a CPU.
GitHub Actions runs them on every push (see the **tests** workflow under the Actions tab, then the **pytest** job, then the **Run the tests** step).

| Test | What it checks | Why it matters |
|---|---|---|
| `test_stencils_sum_to_zero` | Each of the three filters gives zero on a field that is the same everywhere. | A filter that did not would create or destroy mass out of nothing. |
| `test_laplacian_of_quadratic_is_two` | The diffusion filter applied to f = x² returns 2, the true second derivative. | Proves the diffusion stencil is a correct Laplacian. |
| `test_first_derivative_stencils_are_exact_for_linear_fields` | The two advection filters applied to f = x and f = y give a constant of size 1. | Proves the derivative stencils are correct. |
| `test_mass_is_conserved_while_the_blob_is_inside_the_domain` | Run the blob for 100 steps; the total sum of the field stays the same to 0.01%. | The scheme is conservative while the solution is inside the domain. |
| `test_solution_stays_finite_and_bounded` | Run 300 steps at the script's CFL number; no NaN, maximum below 1.05, minimum above -0.05. | The explicit scheme is stable at that time step. |
| `test_pure_diffusion_spreads_the_blob_and_lowers_the_peak` | Switch advection off; after 50 steps the peak is lower and the blob is wider. | Diffusion behaves like diffusion. |
| `test_shapes_and_devices` | A 40 by 24 grid runs and keeps its (1, 1, ny, nx) shape. | The code works for any grid size, not only 128 by 128. |
| `test_gpu_matches_cpu` | The same 20 steps on GPU and on CPU agree to single precision. | Skipped where there is no GPU, including on GitHub's machines. |

The tests import `nn4pdes_advdiff.py` from the repository root. `conftest.py` adds that folder to the Python path so `pytest` works from any directory.
