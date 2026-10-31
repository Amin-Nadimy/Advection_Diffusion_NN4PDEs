"""Tests for the NN4PDEs advection-diffusion solver.

Each test checks one physical or numerical property that the discretisation must have.
They run on the CPU in a few seconds.
"""
import math

import pytest
import torch

from nn4pdes_advdiff import AdvDiff, default_timestep, run, square_initial_condition

torch.manual_seed(0)


def _coordinate_fields(n=32):
    """Fields x and y holding the cell coordinates, shape (1, 1, n, n)."""
    y, x = torch.meshgrid(torch.arange(n, dtype=torch.float32),
                          torch.arange(n, dtype=torch.float32), indexing="ij")
    return x.view(1, 1, n, n), y.view(1, 1, n, n)


def _interior(t):
    """Drop the one-cell border, where zero padding affects the stencils."""
    return t[..., 1:-1, 1:-1]


# ---------- stencil consistency ----------

def test_stencils_sum_to_zero():
    """Each stencil must annihilate a constant field, otherwise it would create or destroy mass."""
    model = AdvDiff()
    ones = torch.ones(1, 1, 16, 16)
    for conv in (model.diff, model.xadv, model.yadv):
        assert torch.allclose(_interior(conv(ones)), torch.zeros(1, 1, 14, 14), atol=1e-6)


def test_laplacian_of_quadratic_is_two():
    """The diffusion stencil applied to f = x^2 must give d2f/dx2 = 2 in the interior."""
    model = AdvDiff()
    x, _ = _coordinate_fields()
    lap = _interior(model.diff(x ** 2))
    # x**2 reaches ~1000 on this grid, so single precision leaves rounding of order 1e-4
    assert torch.allclose(lap, torch.full_like(lap, 2.0), atol=1e-3)


def test_first_derivative_stencils_are_exact_for_linear_fields():
    """On a linear field the derivative stencils must return a constant of magnitude 1."""
    model = AdvDiff()
    x, y = _coordinate_fields()
    dx_field = _interior(model.xadv(x))
    dy_field = _interior(model.yadv(y))
    # magnitude 1 everywhere, and the same sign everywhere (the sign is the stencil's convention)
    assert torch.allclose(dx_field.abs(), torch.ones_like(dx_field), atol=1e-5)
    assert torch.allclose(dy_field.abs(), torch.ones_like(dy_field), atol=1e-5)
    assert dx_field.min() * dx_field.max() > 0
    assert dy_field.min() * dy_field.max() > 0


# ---------- time stepping ----------

def test_mass_is_conserved_while_the_blob_is_inside_the_domain():
    """The stencils are conservative, so the total 'mass' must stay constant until the
    solution reaches the zero-padded boundary."""
    values = square_initial_condition(nx=128, ny=128, size=40, offset=43)
    model = AdvDiff(k=0.5, dx=1.0, dt=default_timestep(), ml=0.25)
    mass0 = values.sum().item()
    out = run(model, values, ntime=100)
    assert math.isclose(out.sum().item(), mass0, rel_tol=1e-4)


def test_solution_stays_finite_and_bounded():
    """With the script's CFL number the explicit scheme must stay stable: no NaN, no blow-up."""
    values = square_initial_condition()
    model = AdvDiff(k=0.5, dx=1.0, dt=default_timestep(), ml=0.25)
    out = run(model, values, ntime=300)
    assert torch.isfinite(out).all()
    assert out.max().item() < 1.05  # a small overshoot is allowed for a central scheme
    assert out.min().item() > -0.05


def test_pure_diffusion_spreads_the_blob_and_lowers_the_peak():
    """Diffusion alone must lower the maximum and widen the support of the blob."""
    values = square_initial_condition(nx=64, ny=64, size=10, offset=27)
    model = AdvDiff(k=0.5, dx=1.0, dt=0.05, ml=0.25)
    # switch advection off by zeroing its stencils
    with torch.no_grad():
        model.xadv.weight.zero_()
        model.yadv.weight.zero_()
    out = run(model, values, ntime=50)
    assert out.max().item() < values.max().item()
    assert (out > 1e-3).sum().item() > (values > 1e-3).sum().item()


def test_shapes_and_devices():
    """The solver must accept any grid size and keep the (1, 1, ny, nx) layout."""
    values = square_initial_condition(nx=40, ny=24, size=6, offset=8)
    model = AdvDiff(dt=default_timestep())
    out = run(model, values, ntime=5)
    assert out.shape == (1, 1, 24, 40)
    assert out.device == values.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU on this machine")
def test_gpu_matches_cpu():
    """The same step on GPU and CPU must agree to single precision."""
    values = square_initial_condition(nx=64, ny=64, size=16, offset=24)
    model = AdvDiff(dt=default_timestep())
    cpu = run(model, values, ntime=20)
    gpu = run(model.cuda(), values.cuda(), ntime=20).cpu()
    assert torch.allclose(cpu, gpu, atol=1e-5)
