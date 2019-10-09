#!/usr/bin/env python3
"""Integral equation test infrastructure"""

from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2014 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.clmath  # noqa
import gzip
import os
import pickle
import pytest
from pytools import Record

from functools import partial
from meshmode.mesh.generation import (  # noqa
        ellipse, cloverleaf, starfish, drop, n_gon, qbx_peanut, WobblyCircle,
        make_curve_mesh)
from meshmode.discretization.visualization import make_visualizer
from pytential import bind, sym
from pytential.qbx import QBXTargetAssociationFailedException
from pyopencl.tools import pytest_generate_tests_for_pyopencl as pytest_generate_tests  # noqa


import logging
logger = logging.getLogger(__name__)

circle = partial(ellipse, 1)

try:
    import matplotlib.pyplot as pt
except ImportError:
    pass


def make_circular_point_group(
        ambient_dim, npoints, radius,
        center=np.array([0., 0.]), func=lambda x: x):
    t = func(np.linspace(0, 1, npoints, endpoint=False)) * (2 * np.pi)
    center = np.asarray(center)
    result = np.zeros((ambient_dim, npoints))
    result[:2, :] = (
            center[:, np.newaxis]
            + radius*np.vstack((np.cos(t), np.sin(t))))
    return result


# {{{ test cases

class IntEqTestCase:
    def __init__(self, helmholtz_k, bc_type, prob_side):
        """
        :arg prob_side: may be -1, +1, or ``'scat'`` for a scattering problem
        """

        if helmholtz_k is None:
            helmholtz_k = self.default_helmholtz_k

        self.helmholtz_k = helmholtz_k
        self.bc_type = bc_type
        self.prob_side = prob_side

    @property
    def k(self):
        return self.helmholtz_k

    def __str__(self):
        return (
                "name: %s, bc_type: %s, prob_side: %s, "
                "helmholtz_k: %s, qbx_order: %d, target_order: %d"
                % (
                    self.name, self.bc_type, self.prob_side, self.helmholtz_k,
                    self.qbx_order, self.target_order))

    fmm_backend = "sumpy"
    gmres_tol = 1e-14


class SphereIntEqTestCase(IntEqTestCase):
    resolutions = [1, 2]
    name = "sphere"

    def get_mesh(self, resolution, target_order):
        from meshmode.mesh.generation import generate_icosphere
        from meshmode.mesh.refinement import refine_uniformly
        mesh = refine_uniformly(
                generate_icosphere(1, target_order),
                resolution)

        return mesh

    fmm_backend = "fmmlib"
    use_refinement = True

    fmm_tol = 1e-4

    inner_radius = 0.4
    outer_radius = 5

    qbx_order = 3
    target_order = 8
    check_gradient = False
    check_tangential_deriv = False

    gmres_tol = 1e-3


class BetterplaneIntEqTestCase(IntEqTestCase):
    name = "betterplane"

    default_helmholtz_k = 20
    resolutions = [0.25]

    fmm_backend = "fmmlib"
    use_refinement = True

    qbx_order = 4
    fmm_tol = 1e-5
    target_order = 7
    check_gradient = False
    check_tangential_deriv = False

    visualize_geometry = False

    scaled_max_curvature_threshold = 5
    expansion_disturbance_tolerance = 0.1
    refinement_maxiter = 20

    gmres_tol = 1e-6

    vis_grid_spacing = (0.025, 0.2, 0.025)
    vis_extend_factor = 0.2

    def __init__(self, helmholtz_k, bc_type, prob_side):
        IntEqTestCase.__init__(self, helmholtz_k, bc_type, prob_side)

    def get_mesh(self, resolution, target_order):
        # mesh order is fixed at 2
        del target_order

        assert resolution == 0.25

        try:
            with gzip.open(f"plane-mesh.pkl.gz", "rb") as inf:
                return pickle.load(inf)
        except FileNotFoundError:
            pass

        from pytools import download_from_web_if_not_present

        download_from_web_if_not_present(
                "https://raw.githubusercontent.com/inducer/geometries/a869fc3/"
                "surface-3d/betterplane.brep")

        from meshmode.mesh.io import generate_gmsh, ScriptWithFilesSource
        mesh = generate_gmsh(
                ScriptWithFilesSource("""
                    Merge "betterplane.brep";

                    Mesh.CharacteristicLengthMax = %(lcmax)f;
                    Mesh.ElementOrder = 2;
                    Mesh.CharacteristicLengthExtendFromBoundary = 0;

                    // 2D mesh optimization
                    // Mesh.Lloyd = 1;

                    l_superfine() = Unique(Abs(Boundary{ Surface{
                        27, 25, 17, 13, 18  }; }));
                    l_fine() = Unique(Abs(Boundary{ Surface{ 2, 6, 7}; }));
                    l_coarse() = Unique(Abs(Boundary{ Surface{ 14, 16  }; }));

                    // p() = Unique(Abs(Boundary{ Line{l_fine()}; }));
                    // Characteristic Length{p()} = 0.05;

                    Field[1] = Attractor;
                    Field[1].NNodesByEdge = 100;
                    Field[1].EdgesList = {l_superfine()};

                    Field[2] = Threshold;
                    Field[2].IField = 1;
                    Field[2].LcMin = 0.075;
                    Field[2].LcMax = %(lcmax)f;
                    Field[2].DistMin = 0.1;
                    Field[2].DistMax = 0.4;

                    Field[3] = Attractor;
                    Field[3].NNodesByEdge = 100;
                    Field[3].EdgesList = {l_fine()};

                    Field[4] = Threshold;
                    Field[4].IField = 3;
                    Field[4].LcMin = 0.1;
                    Field[4].LcMax = %(lcmax)f;
                    Field[4].DistMin = 0.15;
                    Field[4].DistMax = 0.4;

                    Field[5] = Attractor;
                    Field[5].NNodesByEdge = 100;
                    Field[5].EdgesList = {l_coarse()};

                    Field[6] = Threshold;
                    Field[6].IField = 5;
                    Field[6].LcMin = 0.2;
                    Field[6].LcMax = %(lcmax)f;
                    Field[6].DistMin = 0.2;
                    Field[6].DistMax = 0.4;

                    Field[7] = Min;
                    Field[7].FieldsList = {2, 4, 6};

                    Background Field = 7;
                    """ % {
                        "lcmax": resolution,
                        }, ["betterplane.brep"]), 2)

        # Flip elements--gmsh generates inside-out geometry.
        from meshmode.mesh.processing import perform_flips
        mesh = perform_flips(mesh, np.ones(mesh.nelements))

        with gzip.open("plane-mesh.pkl.gz", "wb") as outf:
            pickle.dump(mesh, outf)

        return mesh

    inner_radius = 0.2
    outer_radius = 15

# }}}


# {{{ lpot source from case

def lpot_source_from_case(cl_ctx, queue, case, resolution, base_lpot_kwargs):
    mesh = case.get_mesh(resolution, case.target_order)
    logger.info("%d elements" % mesh.nelements)

    from pytential.qbx import QBXLayerPotentialSource
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
            InterpolatoryQuadratureSimplexGroupFactory)

    pre_density_discr = Discretization(
            cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(case.target_order))

    source_order = 4*case.target_order

    refiner_extra_kwargs = {}

    qbx_lpot_kwargs = base_lpot_kwargs.copy()

    if case.fmm_backend is None:
        qbx_lpot_kwargs["fmm_order"] = False
    else:
        if hasattr(case, "fmm_tol"):
            from sumpy.expansion.level_to_order import\
                    SimpleExpansionOrderFinder
            qbx_lpot_kwargs["fmm_level_to_order"] = SimpleExpansionOrderFinder(
                    case.fmm_tol)

        elif hasattr(case, "fmm_order"):
            qbx_lpot_kwargs["fmm_order"] = case.fmm_order
        else:
            qbx_lpot_kwargs["fmm_order"] = case.qbx_order + 5

    qbx = QBXLayerPotentialSource(
            pre_density_discr,
            fine_order=source_order,
            qbx_order=case.qbx_order,
            fmm_backend="fmmlib",
            **qbx_lpot_kwargs)

    if case.use_refinement:
        if case.k != 0 and getattr(case, "refine_on_helmholtz_k", True):
            refiner_extra_kwargs["kernel_length_scale"] = 5/case.k

        if hasattr(case, "scaled_max_curvature_threshold"):
            refiner_extra_kwargs["_scaled_max_curvature_threshold"] = \
                    case.scaled_max_curvature_threshold

        if hasattr(case, "expansion_disturbance_tolerance"):
            refiner_extra_kwargs["_expansion_disturbance_tolerance"] = \
                    case.expansion_disturbance_tolerance

        if hasattr(case, "refinement_maxiter"):
            refiner_extra_kwargs["maxiter"] = case.refinement_maxiter

        logger.info(
                "%d elements before refinement",
                pre_density_discr.mesh.nelements)
        qbx, _ = qbx.with_refinement(**refiner_extra_kwargs)

        logger.info(
                "%d stage-1 elements after refinement",
                qbx.density_discr.mesh.nelements)
        logger.info(
                "%d stage-2 elements after refinement",
                qbx.stage2_density_discr.mesh.nelements)
        logger.info(
                "quad stage-2 elements have %d nodes",
                qbx.quad_stage2_density_discr.groups[0].nunit_nodes)

    return qbx

# }}}


# {{{ test backend

def run_int_eq_test(
        cl_ctx, queue, case, resolution, visualize, lpot_kwargs=None,
        output_dir=None):
    if lpot_kwargs is None:
        lpot_kwargs = {}
    qbx = lpot_source_from_case(cl_ctx, queue, case, resolution, lpot_kwargs)

    def output_file_path(fname):
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            return os.path.join(output_dir, fname)
        return fname

    density_discr = qbx.density_discr
    mesh = density_discr.mesh

    if (
            visualize
            and getattr(case, "visualize_geometry", False)):
        bdry_normals = bind(
                density_discr, sym.normal(mesh.ambient_dim)
                )(queue).as_vector(dtype=object)

        bdry_vis = make_visualizer(queue, density_discr, case.target_order)
        bdry_vis.write_vtk_file(output_file_path("geometry.vtu"), [
                ("normals", bdry_normals)
                ])

    # {{{ plot geometry

    if 0:
        if mesh.ambient_dim == 2:
            # show geometry, centers, normals
            nodes_h = density_discr.nodes().get(queue=queue)
            pt.plot(nodes_h[0], nodes_h[1], "x-")
            normal = (
                    bind(density_discr, sym.normal(2))(queue)
                    ).as_vector(np.object)
            pt.quiver(
                    nodes_h[0], nodes_h[1],
                    normal[0].get(queue), normal[1].get(queue))
            pt.gca().set_aspect("equal")
            pt.show()

        elif mesh.ambient_dim == 3:
            bdry_vis = make_visualizer(
                    queue, density_discr, case.target_order+3)

            bdry_normals = (
                    bind(density_discr, sym.normal(3))(queue)
                    ).as_vector(dtype=object)

            bdry_vis.write_vtk_file(
                    output_file_path("pre-solve-source-%s.vtu" % resolution),
                    [("bdry_normals", bdry_normals),])

        else:
            raise ValueError("invalid mesh dim")

    # }}}

    # {{{ set up operator

    from pytential.symbolic.pde.scalar import (
            DirichletOperator,
            NeumannOperator)

    from sumpy.kernel import LaplaceKernel, HelmholtzKernel
    if case.k:
        knl = HelmholtzKernel(mesh.ambient_dim)
        knl_kwargs = {"k": sym.var("k")}
        concrete_knl_kwargs = {"k": case.k}
    else:
        knl = LaplaceKernel(mesh.ambient_dim)
        knl_kwargs = {}
        concrete_knl_kwargs = {}

    if knl.is_complex_valued:
        dtype = np.complex128
    else:
        dtype = np.float64

    loc_sign = +1 if case.prob_side in [+1, "scat"] else -1

    if case.bc_type == "dirichlet":
        op = DirichletOperator(
                knl, loc_sign, use_l2_weighting=True,
                kernel_arguments=knl_kwargs)
    elif case.bc_type == "neumann":
        op = NeumannOperator(
                knl, loc_sign, use_l2_weighting=True,
                use_improved_operator=False, kernel_arguments=knl_kwargs)
    else:
        assert False

    op_u = op.operator(sym.var("u"))

    # }}}

    # {{{ set up test data

    if case.prob_side == -1:
        test_src_geo_radius = case.outer_radius
        test_tgt_geo_radius = case.inner_radius
    elif case.prob_side == +1:
        test_src_geo_radius = case.inner_radius
        test_tgt_geo_radius = case.outer_radius
    elif case.prob_side == "scat":
        test_src_geo_radius = case.outer_radius
        test_tgt_geo_radius = case.outer_radius
    else:
        raise ValueError("unknown problem_side")

    point_sources = make_circular_point_group(
            mesh.ambient_dim, 10, test_src_geo_radius,
            func=lambda x: x**1.5)
    test_targets = make_circular_point_group(
            mesh.ambient_dim, 20, test_tgt_geo_radius)

    np.random.seed(22)
    source_charges = np.random.randn(point_sources.shape[1])
    source_charges[-1] = -np.sum(source_charges[:-1])
    source_charges = source_charges.astype(dtype)
    assert np.sum(source_charges) < 1e-15

    source_charges_dev = cl.array.to_device(queue, source_charges)

    # }}}

    # {{{ establish BCs

    from pytential.source import PointPotentialSource
    from pytential.target import PointsTarget

    point_source = PointPotentialSource(cl_ctx, point_sources)

    pot_src = sym.IntG(
            # FIXME: qbx_forced_limit--really?
            knl, sym.var("charges"), qbx_forced_limit=None, **knl_kwargs)

    test_direct = bind((point_source, PointsTarget(test_targets)), pot_src)(
            queue, charges=source_charges_dev, **concrete_knl_kwargs)

    if case.bc_type == "dirichlet":
        bc = bind((point_source, density_discr), pot_src)(
                queue, charges=source_charges_dev, **concrete_knl_kwargs)

    elif case.bc_type == "neumann":
        bc = bind(
                (point_source, density_discr),
                sym.normal_derivative(
                    qbx.ambient_dim, pot_src, where=sym.DEFAULT_TARGET)
                )(queue, charges=source_charges_dev, **concrete_knl_kwargs)

    # }}}

    # {{{ solve

    bound_op = bind(qbx, op_u)

    rhs = bind(density_discr, op.prepare_rhs(sym.var("bc")))(queue, bc=bc)

    try:
        from pytential.solve import gmres
        gmres_result = gmres(
                bound_op.scipy_op(queue, "u", dtype, **concrete_knl_kwargs),
                rhs,
                tol=case.gmres_tol,
                progress=True,
                hard_failure=True,
                stall_iterations=50, no_progress_factor=1.05)
    except QBXTargetAssociationFailedException as e:
        bdry_vis = make_visualizer(queue, density_discr, case.target_order+3)
        bdry_vis.write_vtk_file(
                output_file_path("failed-targets-%s.vtu" % resolution),
                [("failed_targets", e.failed_target_flags),])
        raise

    logger.info("gmres state: %s", gmres_result.state)
    weighted_u = gmres_result.solution

    # }}}

    if case.prob_side != "scat":
        # {{{ error check

        points_target = PointsTarget(test_targets)
        bound_tgt_op = bind(
                (qbx, points_target),
                op.representation(sym.var("u")))

        test_via_bdry = bound_tgt_op(queue, u=weighted_u, k=case.k)

        err = test_via_bdry - test_direct

        err = err.get()
        test_direct = test_direct.get()
        test_via_bdry = test_via_bdry.get()

        # {{{ remove effect of net source charge

        if case.k == 0 and case.bc_type == "neumann" and loc_sign == -1:

            # remove constant offset in interior Laplace Neumann error
            tgt_ones = np.ones_like(test_direct)
            tgt_ones = tgt_ones/la.norm(tgt_ones)
            err = err - np.vdot(tgt_ones, err)*tgt_ones

        # }}}

        rel_err_2 = la.norm(err)/la.norm(test_direct)
        rel_err_inf = la.norm(err, np.inf)/la.norm(test_direct, np.inf)

        # }}}

        logger.info("rel_err_2: %g rel_err_inf: %g" % (rel_err_2, rel_err_inf))

    else:
        rel_err_2 = None
        rel_err_inf = None

    # {{{ test gradient

    if case.check_gradient and case.prob_side != "scat":
        bound_grad_op = bind(
                (qbx, points_target),
                op.representation(
                    sym.var("u"),
                    map_potentials=lambda pot: sym.grad(mesh.ambient_dim, pot),
                    qbx_forced_limit=None))

        # logger.info(bound_t_deriv_op.code)

        grad_from_src = bound_grad_op(
                queue, u=weighted_u, **concrete_knl_kwargs)

        grad_ref = (bind(
                (point_source, points_target),
                sym.grad(mesh.ambient_dim, pot_src)
                )(queue, charges=source_charges_dev, **concrete_knl_kwargs)
                )

        grad_err = (grad_from_src - grad_ref)

        rel_grad_err_inf = (
                la.norm(grad_err[0].get(), np.inf)
                /
                la.norm(grad_ref[0].get(), np.inf))

        logger.info("rel_grad_err_inf: %g" % rel_grad_err_inf)

    # }}}

    # {{{ test tangential derivative

    if case.check_tangential_deriv and case.prob_side != "scat":
        bound_t_deriv_op = bind(
                qbx,
                op.representation(
                    sym.var("u"),
                    map_potentials=partial(sym.tangential_derivative, 2),
                    qbx_forced_limit=loc_sign))

        # logger.info(bound_t_deriv_op.code)

        tang_deriv_from_src = bound_t_deriv_op(
                queue, u=weighted_u, **concrete_knl_kwargs).as_scalar().get()

        tang_deriv_ref = (bind(
                (point_source, density_discr),
                sym.tangential_derivative(2, pot_src)
                )(queue, charges=source_charges_dev, **concrete_knl_kwargs)
                .as_scalar().get())

        if 0:
            pt.plot(tang_deriv_ref.real)
            pt.plot(tang_deriv_from_src.real)
            pt.show()

        td_err = (tang_deriv_from_src - tang_deriv_ref)

        rel_td_err_inf = (
                la.norm(td_err, np.inf)
                / la.norm(tang_deriv_ref, np.inf))

        logger.info("rel_td_err_inf: %g" % rel_td_err_inf)

    else:
        rel_td_err_inf = None

    # }}}

    # {{{ any-D file plotting

    if visualize:
        bdry_vis = make_visualizer(queue, density_discr, case.target_order+3)

        bdry_normals = (
                bind(
                    density_discr, sym.normal(qbx.ambient_dim))(queue)
                ).as_vector(dtype=object)

        sym_sqrt_j = sym.sqrt_jac_q_weight(density_discr.ambient_dim)
        u = bind(density_discr, sym.var("u")/sym_sqrt_j)(queue, u=weighted_u)

        bdry_vis.write_vtk_file(
                output_file_path("source-%s.vtu" % resolution),
                [
                    ("u", u),
                    ("bc", bc),
                    # ("bdry_normals", bdry_normals),
                    ])

        from sumpy.visualization import make_field_plotter_from_bbox  # noqa
        from meshmode.mesh.processing import find_bounding_box

        vis_grid_spacing = (0.1, 0.1, 0.1)[:qbx.ambient_dim]
        if hasattr(case, "vis_grid_spacing"):
            vis_grid_spacing = case.vis_grid_spacing
        vis_extend_factor = 0.2
        if hasattr(case, "vis_extend_factor"):
            vis_grid_spacing = case.vis_grid_spacing

        fplot = make_field_plotter_from_bbox(
                find_bounding_box(mesh),
                h=vis_grid_spacing,
                extend_factor=vis_extend_factor)

        qbx_tgt_tol = qbx.copy(target_association_tolerance=0.15)
        from pytential.target import PointsTarget

        try:
            solved_pot = bind(
                    (qbx_tgt_tol, PointsTarget(fplot.points)),
                    op.representation(sym.var("u"))
                    )(queue, u=weighted_u, k=case.k)
        except QBXTargetAssociationFailedException as e:
            fplot.write_vtk_file(
                    output_file_path("failed-targets.vts"),
                    [
                        ("failed_targets", e.failed_target_flags.get(queue))
                        ])
            raise

        from sumpy.kernel import LaplaceKernel
        ones_density = density_discr.zeros(queue)
        ones_density.fill(1)
        indicator_func = bind(
                (
                    qbx_tgt_tol, PointsTarget(fplot.points)),
                -sym.D(
                    LaplaceKernel(density_discr.ambient_dim),
                    sym.var("sigma"),
                    qbx_forced_limit=None))
        indicator = indicator_func(queue, sigma=ones_density).get()

        solved_pot = solved_pot.get()

        true_pot = bind((point_source, PointsTarget(fplot.points)), pot_src)(
                queue, charges=source_charges_dev, **concrete_knl_kwargs).get()

        # fplot.show_scalar_in_mayavi(solved_pot.real, max_val=5)
        if case.prob_side == "scat":
            fplot.write_vtk_file(
                    output_file_path("potential-%s.vts" % resolution),
                    [
                        ("pot_scattered", solved_pot),
                        ("pot_incoming", -true_pot),
                        ("indicator", indicator),
                        ]
                    )
        else:
            fplot.write_vtk_file(
                    output_file_path("potential-%s.vts" % resolution),
                    [
                        ("solved_pot", solved_pot),
                        ("true_pot", true_pot),
                        ("indicator", indicator),
                        ]
                    )

    # }}}

    class Result(Record):
        pass

    return Result(
            h_max=qbx.h_max,
            rel_err_2=rel_err_2,
            rel_err_inf=rel_err_inf,
            rel_td_err_inf=rel_td_err_inf,
            gmres_result=gmres_result)

# }}}


# {{{ test frontend

SphereHelmholtzDirichletTestCase = SphereIntEqTestCase(5, "dirichlet", +1)


@pytest.mark.parametrize(
        "case",
        (SphereHelmholtzDirichletTestCase,))
def test_integral_equation(ctx_getter, case, visualize=False):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    from pytools.convergence import EOCRecorder
    print("qbx_order: %d, %s" % (case.qbx_order, case))

    lpot_kwargs = {"_use_target_specific_qbx": True}

    eoc_rec_target = EOCRecorder()
    eoc_rec_td = EOCRecorder()

    have_error_data = False
    for resolution in case.resolutions:
        result = run_int_eq_test(
                cl_ctx, queue, case, resolution,
                visualize=visualize, lpot_kwargs=lpot_kwargs)

        if result.rel_err_2 is not None:
            have_error_data = True
            eoc_rec_target.add_data_point(result.h_max, result.rel_err_2)

        if result.rel_td_err_inf is not None:
            eoc_rec_td.add_data_point(result.h_max, result.rel_td_err_inf)

    if case.bc_type == "dirichlet":
        tgt_order = case.qbx_order
    elif case.bc_type == "neumann":
        tgt_order = case.qbx_order-1
    else:
        assert False

    if have_error_data:
        print("TARGET ERROR:")
        print(eoc_rec_target)
        assert eoc_rec_target.order_estimate() > tgt_order - 1.3

        if case.check_tangential_deriv:
            print("TANGENTIAL DERIVATIVE ERROR:")
            print(eoc_rec_td)
            assert eoc_rec_td.order_estimate() > tgt_order - 2.3

# }}}


def plane_lpot_source(queue, lpot_kwargs):
    cl_ctx = queue.context
    lpot_kwargs = lpot_kwargs.copy()
    case = BetterplaneIntEqTestCase(20, "dirichlet", +1)
    return lpot_source_from_case(cl_ctx, queue, case, 0.25, lpot_kwargs)


def sphere_lpot_source(queue, lpot_kwargs):
    cl_ctx = queue.context
    lpot_kwargs = lpot_kwargs.copy()
    case = SphereIntEqTestCase(20, "dirichlet", +1)
    return lpot_source_from_case(cl_ctx, queue, case, 1, lpot_kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__])
