"""Parity tests between bed.grid and bed_jax.grid for Phase 3."""

import numpy as np
import pytest

pytest.importorskip("jax")

from bed.grid import (
    CosineBump as NPCosineBump,
)
from bed.grid import (
    Gaussian as NPGaussian,
)
from bed.grid import (
    Grid as NPGrid,
)
from bed.grid import (
    GridStack as NPGridStack,
)
from bed.grid import (
    PermutationInvariant as NPPermutationInvariant,
)
from bed.grid import (
    TopHat as NPTopHat,
)
from bed_jax.grid import (
    CosineBump as JCosineBump,
)
from bed_jax.grid import (
    Gaussian as JGaussian,
)
from bed_jax.grid import (
    Grid as JGrid,
)
from bed_jax.grid import (
    GridStack as JGridStack,
)
from bed_jax.grid import (
    PermutationInvariant as JPermutationInvariant,
)
from bed_jax.grid import (
    TopHat as JTopHat,
)

RTOL = 2e-5


def _assert_max_dict_equal(expected, actual):
    assert expected.keys() == actual.keys()
    for key in expected:
        assert expected[key] == actual[key]


def test_constructor_and_axes_parity():
    ng = NPGrid(x=np.arange(3), y=np.arange(4))
    jg = JGrid(x=np.arange(3), y=np.arange(4))
    assert ng.names == jg.names
    assert ng.shape == jg.shape
    np.testing.assert_array_equal(ng.x, jg.x)
    np.testing.assert_array_equal(ng.y, jg.y)


def test_constraint_constructor_parity():
    ng = NPGrid(
        x=np.arange(3),
        y=np.arange(4),
        constraint=lambda x, y: x + y < 3,
    )
    jg = JGrid(
        x=np.arange(3),
        y=np.arange(4),
        constraint=lambda x, y: x + y < 3,
    )
    np.testing.assert_array_equal(ng.x, jg.x)
    np.testing.assert_array_equal(ng.y, jg.y)
    np.testing.assert_array_equal(ng.axes_in["y"], jg.axes_in["y"])


def test_constraint_constructor_invalid_parity():
    with pytest.raises(ValueError):
        NPGrid(
            x=[1, 2, 3],
            y=[4, 5],
            z=[6, 7, 8],
            constraint=lambda a, b: a < b,
        )
    with pytest.raises(ValueError):
        JGrid(
            x=[1, 2, 3],
            y=[4, 5],
            z=[6, 7, 8],
            constraint=lambda a, b: a < b,
        )


def test_expand_parity():
    ng1 = NPGrid(x=[1, 2, 3], u=[4, 5], y=1, v=[6, 7, 8, 9])
    jg1 = JGrid(x=[1, 2, 3], u=[4, 5], y=1, v=[6, 7, 8, 9])
    ng2 = NPGrid(
        x=[1, 2, 3],
        u=[4, 5],
        y=1,
        v=[6, 7, 8, 9],
        constraint=lambda u, v: (u + v) % 2,
    )
    jg2 = JGrid(
        x=[1, 2, 3],
        u=[4, 5],
        y=1,
        v=[6, 7, 8, 9],
        constraint=lambda u, v: (u + v) % 2,
    )
    nv1 = ng1.x * ((ng1.u + ng1.v) % 2) ** ng1.y
    jv1 = jg1.x * ((jg1.u + jg1.v) % 2) ** jg1.y
    nv2 = ng2.x * ((ng2.u + ng2.v) % 2) ** ng2.y
    jv2 = jg2.x * ((jg2.u + jg2.v) % 2) ** jg2.y
    np.testing.assert_array_equal(ng1.expand(nv1), jg1.expand(jv1))
    np.testing.assert_array_equal(ng2.expand(nv2), jg2.expand(jv2))


def test_expand_3d_parity():
    ng1 = NPGrid(x=[1, 2, 3], u=2, y=[1, 2, 3], v=[1, 2], z=[1, 2, 3])
    jg1 = JGrid(x=[1, 2, 3], u=2, y=[1, 2, 3], v=[1, 2], z=[1, 2, 3])
    nv1 = (ng1.x + ng1.y + ng1.z) * ng1.u + ng1.v
    jv1 = (jg1.x + jg1.y + jg1.z) * jg1.u + jg1.v
    ng2 = NPGrid(
        x=[1, 2, 3],
        u=2,
        y=[1, 2, 3],
        v=[1, 2],
        z=[1, 2, 3],
        constraint=lambda x, y, z: NPPermutationInvariant(x, y, z),
    )
    jg2 = JGrid(
        x=[1, 2, 3],
        u=2,
        y=[1, 2, 3],
        v=[1, 2],
        z=[1, 2, 3],
        constraint=lambda x, y, z: JPermutationInvariant(x, y, z),
    )
    nv2 = (ng2.x + ng2.y + ng2.z) * ng2.u + ng2.v
    jv2 = (jg2.x + jg2.y + jg2.z) * jg2.u + jg2.v
    np.testing.assert_array_equal(ng1.expand(nv1), np.asarray(jg1.expand(jv1)))
    np.testing.assert_array_equal(ng2.expand(nv2), np.asarray(jg2.expand(jv2)))


def test_sum_and_normalize_parity():
    ng = NPGrid(x=np.arange(3), y=np.arange(4))
    jg = JGrid(x=np.arange(3), y=np.arange(4))
    nvals = np.ones(ng.shape)
    jvals = np.ones(jg.shape)

    np.testing.assert_allclose(np.asarray(jg.sum(jvals)), ng.sum(nvals), rtol=RTOL)
    np.testing.assert_array_equal(
        ng.sum(nvals, axis_names=("x",)),
        np.asarray(jg.sum(jvals, axis_names=("x",))),
    )

    nvals2 = np.exp(-np.linspace(0, 1, 12)).reshape(ng.shape)
    jvals2 = nvals2.copy()
    ng.normalize(nvals2)
    jvals2_out = jg.normalize(jvals2)
    np.testing.assert_allclose(nvals2, np.asarray(jvals2_out), rtol=RTOL)


def test_sum_constraint_parity():
    ng1 = NPGrid(x=[1, 2, 3], u=[4, 5], y=1, v=[6, 7, 8, 9])
    jg1 = JGrid(x=[1, 2, 3], u=[4, 5], y=1, v=[6, 7, 8, 9])
    ng2 = NPGrid(
        x=[1, 2, 3],
        u=[4, 5],
        y=1,
        v=[6, 7, 8, 9],
        constraint=lambda u, v: (u + v) % 2 == 1,
    )
    jg2 = JGrid(
        x=[1, 2, 3],
        u=[4, 5],
        y=1,
        v=[6, 7, 8, 9],
        constraint=lambda u, v: (u + v) % 2 == 1,
    )
    nz1 = ng1.x * ((ng1.u + ng1.v) % 2) ** ng1.y
    jz1 = jg1.x * ((jg1.u + jg1.v) % 2) ** jg1.y
    nz2 = ng2.x * ((ng2.u + ng2.v) % 2) ** ng2.y
    jz2 = jg2.x * ((jg2.u + jg2.v) % 2) ** jg2.y
    np.testing.assert_allclose(np.asarray(ng1.sum(nz1)), np.asarray(jg1.sum(jz1)), rtol=RTOL)
    np.testing.assert_allclose(np.asarray(ng2.sum(nz2)), np.asarray(jg2.sum(jz2)), rtol=RTOL)


def test_sum_permutation_invariant_2d_parity():
    ng1 = NPGrid(u=2, x=[1, 2, 3], y=[1, 2], z=[1, 2, 3])
    jg1 = JGrid(u=2, x=[1, 2, 3], y=[1, 2], z=[1, 2, 3])
    ng2 = NPGrid(
        u=2,
        x=[1, 2, 3],
        y=[1, 2],
        z=[1, 2, 3],
        constraint=lambda x, z: NPPermutationInvariant(x, z),
    )
    jg2 = JGrid(
        u=2,
        x=[1, 2, 3],
        y=[1, 2],
        z=[1, 2, 3],
        constraint=lambda x, z: JPermutationInvariant(x, z),
    )
    nz1 = (ng1.x + ng1.z) * ng1.y + ng1.u
    jz1 = (jg1.x + jg1.z) * jg1.y + jg1.u
    nz2 = (ng2.x + ng2.z) * ng2.y + ng2.u
    jz2 = (jg2.x + jg2.z) * jg2.y + jg2.u
    np.testing.assert_allclose(np.asarray(ng1.sum(nz1)), np.asarray(jg1.sum(jz1)), rtol=RTOL)
    np.testing.assert_allclose(np.asarray(ng2.sum(nz2)), np.asarray(jg2.sum(jz2)), rtol=RTOL)


def test_sum_permutation_invariant_3d_parity():
    ng1 = NPGrid(u=[1, 2], x=[1, 2, 3], y=[1, 2, 3], z=[1, 2, 3])
    jg1 = JGrid(u=[1, 2], x=[1, 2, 3], y=[1, 2, 3], z=[1, 2, 3])
    ng2 = NPGrid(
        u=[1, 2],
        x=[1, 2, 3],
        y=[1, 2, 3],
        z=[1, 2, 3],
        constraint=lambda x, y, z: NPPermutationInvariant(x, y, z),
    )
    jg2 = JGrid(
        u=[1, 2],
        x=[1, 2, 3],
        y=[1, 2, 3],
        z=[1, 2, 3],
        constraint=lambda x, y, z: JPermutationInvariant(x, y, z),
    )
    nz1 = (ng1.x + ng1.y + ng1.z) * ng1.u
    jz1 = (jg1.x + jg1.y + jg1.z) * jg1.u
    nz2 = (ng2.x + ng2.y + ng2.z) * ng2.u
    jz2 = (jg2.x + jg2.y + jg2.z) * jg2.u
    np.testing.assert_allclose(np.asarray(ng1.sum(nz1)), np.asarray(jg1.sum(jz1)), rtol=RTOL)
    np.testing.assert_allclose(np.asarray(ng2.sum(nz2)), np.asarray(jg2.sum(jz2)), rtol=RTOL)


def test_sum_marginalized_axis_names_parity():
    ng = NPGrid(x=np.arange(3), y=np.arange(4), z=np.arange(5))
    jg = JGrid(x=np.arange(3), y=np.arange(4), z=np.arange(5))
    nvalues = np.ones(ng.shape)
    jvalues = np.ones(jg.shape)
    nmarginal = ng.sum(nvalues, axis_names=("x", "z"), keepdims=True)
    jmarginal = jg.sum(jvalues, axis_names=("x", "z"), keepdims=True)
    np.testing.assert_array_equal(nmarginal, jmarginal)
    np.testing.assert_allclose(
        np.asarray(jg.sum(jmarginal, axis_names=("y",))),
        ng.sum(nmarginal, axis_names=("y",)),
        rtol=RTOL,
    )


def test_sum_marginalized_in_stack_parity():
    ng1 = NPGrid(a=np.arange(2))
    jg1 = JGrid(a=np.arange(2))
    ng2 = NPGrid(x=np.arange(3), y=np.arange(4))
    jg2 = JGrid(x=np.arange(3), y=np.arange(4))
    with NPGridStack(ng1, ng2), JGridStack(jg1, jg2):
        nvalues = np.ones(ng1.shape + ng2.shape)
        jvalues = np.ones(jg1.shape + jg2.shape)
        nmarginal = ng2.sum(nvalues, axis_names=("x",), keepdims=True)
        jmarginal = jg2.sum(jvalues, axis_names=("x",), keepdims=True)
        assert nmarginal.shape == jmarginal.shape
        np.testing.assert_allclose(nmarginal, np.asarray(jmarginal), rtol=RTOL)
        ntotal = ng2.sum(nmarginal, axis_names=("y",))
        jtotal = jg2.sum(jmarginal, axis_names=("y",))
        np.testing.assert_allclose(ntotal, np.asarray(jtotal), rtol=RTOL)


def test_sum_marginalized_rejects_without_axis_names_parity():
    ng = NPGrid(x=np.arange(3), y=np.arange(4), z=np.arange(5))
    jg = JGrid(x=np.arange(3), y=np.arange(4), z=np.arange(5))
    nmarginal = ng.sum(np.ones(ng.shape), axis_names=("x", "z"), keepdims=True)
    jmarginal = jg.sum(np.ones(jg.shape), axis_names=("x", "z"), keepdims=True)
    with pytest.raises(ValueError):
        ng.sum(nmarginal)
    with pytest.raises(ValueError):
        jg.sum(jmarginal)


def test_sum_marginalized_rejects_bad_named_axis_parity():
    ng = NPGrid(x=np.arange(3), y=np.arange(4), z=np.arange(5))
    jg = JGrid(x=np.arange(3), y=np.arange(4), z=np.arange(5))
    nmarginal = ng.sum(np.ones(ng.shape), axis_names=("x",), keepdims=True)
    jmarginal = jg.sum(np.ones(jg.shape), axis_names=("x",), keepdims=True)
    with pytest.raises(ValueError):
        ng.sum(nmarginal, axis_names=("x",))
    with pytest.raises(ValueError):
        jg.sum(jmarginal, axis_names=("x",))


def test_sum_marginalized_rejects_wrong_ndim_parity():
    ng = NPGrid(x=np.arange(3), y=np.arange(4))
    jg = JGrid(x=np.arange(3), y=np.arange(4))
    with pytest.raises(ValueError):
        ng.sum(np.ones((3,)), axis_names=("x",))
    with pytest.raises(ValueError):
        jg.sum(np.ones((3,)), axis_names=("x",))


def test_sum_invalid_axis_names_parity():
    ng = NPGrid(x=np.arange(3), y=np.arange(4))
    jg = JGrid(x=np.arange(3), y=np.arange(4))
    with pytest.raises(ValueError):
        ng.sum(np.ones(ng.shape), axis_names=("x", "z"))
    with pytest.raises(ValueError):
        jg.sum(np.ones(jg.shape), axis_names=("x", "z"))


def test_extent_index_getmax_parity():
    ng = NPGrid(x=np.linspace(0, 2, 3), y=np.arange(4))
    jg = JGrid(x=np.linspace(0, 2, 3), y=np.arange(4))
    nvals = (ng.x + ng.y).astype(float)
    jvals = (jg.x + jg.y).astype(float)

    assert ng.extent("x") == jg.extent("x")
    assert ng.index("x", 1.1) == jg.index("x", 1.1)
    _assert_max_dict_equal(ng.getmax(nvals), jg.getmax(jvals))


def test_subgrid_parity():
    ng = NPGrid(x=np.arange(3), y=np.arange(3))
    jg = JGrid(x=np.arange(3), y=np.arange(3))
    nsub = list(ng.subgrid(2))
    jsub = list(jg.subgrid(2))
    assert len(nsub) == len(jsub)
    for (ns, nmask), (js, jmask) in zip(nsub, jsub):
        np.testing.assert_array_equal(ns.x, js.x)
        np.testing.assert_array_equal(ns.y, js.y)
        np.testing.assert_array_equal(nmask, jmask)


def test_subgrid_constrained_parity():
    ng = NPGrid(x=np.arange(3), y=np.arange(3), constraint=lambda x, y: x + y < 3)
    jg = JGrid(x=np.arange(3), y=np.arange(3), constraint=lambda x, y: x + y < 3)
    nsub = list(ng.subgrid(3))
    jsub = list(jg.subgrid(3))
    assert len(nsub) == len(jsub)
    (ns, nmask), (js, jmask) = nsub[0], jsub[0]
    np.testing.assert_array_equal(ns.x, js.x)
    np.testing.assert_array_equal(ns.y, js.y)
    np.testing.assert_array_equal(nmask, jmask)


def test_subgrid_invalid_parity():
    ng = NPGrid(x=np.arange(3), y=np.arange(3))
    jg = JGrid(x=np.arange(3), y=np.arange(3))
    with pytest.raises(ValueError):
        next(ng.subgrid(2.1))
    with pytest.raises(ValueError):
        next(jg.subgrid(2.1))
    with pytest.raises(ValueError):
        next(ng.subgrid(-1))
    with pytest.raises(ValueError):
        next(jg.subgrid(-1))


def test_gridstack_parity():
    ng1 = NPGrid(x=np.arange(3), y=np.arange(4))
    ng2 = NPGrid(u=np.arange(2), v=np.arange(5))
    jg1 = JGrid(x=np.arange(3), y=np.arange(4))
    jg2 = JGrid(u=np.arange(2), v=np.arange(5))
    with NPGridStack(ng1, ng2) as nstack, JGridStack(jg1, jg2) as jstack:
        assert ng1.x.shape == jg1.x.shape
        assert ng1.y.shape == jg1.y.shape
        assert ng2.u.shape == jg2.u.shape
        assert ng2.v.shape == jg2.v.shape
        assert nstack.at(x=1, u=0) == jstack.at(x=1, u=0)


def test_gridstack_keep_parity():
    ng1 = NPGrid(x=np.arange(3), y=np.arange(4), constraint=lambda x, y: x + y < 3)
    ng2 = NPGrid(x=np.arange(2), y=np.arange(5))
    jg1 = JGrid(x=np.arange(3), y=np.arange(4), constraint=lambda x, y: x + y < 3)
    jg2 = JGrid(x=np.arange(2), y=np.arange(5))
    with NPGridStack(ng1, ng2), JGridStack(jg1, jg2):
        assert ng1.x.shape == jg1.x.shape
        assert ng1.y.shape == jg1.y.shape
        assert ng2.x.shape == jg2.x.shape
        assert ng2.y.shape == jg2.y.shape


def test_helpers_parity():
    x1 = np.linspace(0.2, 2.0, 181)
    x2 = np.linspace(0.8, 1.2, 50)
    x3 = np.linspace(-3, 3, 101)
    pi1 = NPPermutationInvariant(np.arange(3), np.arange(3).reshape(-1, 1))
    pi2 = JPermutationInvariant(np.arange(3), np.arange(3).reshape(-1, 1))
    np.testing.assert_allclose(NPTopHat(x1), JTopHat(x1), rtol=RTOL)
    np.testing.assert_allclose(NPCosineBump(x2), JCosineBump(x2), rtol=RTOL)
    np.testing.assert_allclose(NPGaussian(x3, 0, 1), JGaussian(x3, 0, 1), rtol=RTOL)
    np.testing.assert_allclose(pi1, pi2, rtol=RTOL)


def test_permutation_invariant_3d_parity():
    x = np.arange(3)
    pi1 = NPPermutationInvariant(x.reshape((-1, 1, 1)), x.reshape((-1, 1)), x)
    pi2 = JPermutationInvariant(x.reshape((-1, 1, 1)), x.reshape((-1, 1)), x)
    np.testing.assert_allclose(pi1, pi2, rtol=RTOL)


def test_permutation_invariant_invalid_axes_parity():
    with pytest.raises(ValueError):
        NPPermutationInvariant(np.array([1, 2, 3]), np.array([1, 2]))
    with pytest.raises(ValueError):
        JPermutationInvariant(np.array([1, 2, 3]), np.array([1, 2]))
