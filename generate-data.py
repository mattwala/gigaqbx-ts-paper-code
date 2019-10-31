#!/usr/bin/env python3
"""Gather experimental data"""
import numpy as np  # noqa
import numpy.linalg as la  # noqa
import pyopencl as cl  # noqa
import pyopencl.clmath  # noqa
import json
import utils
import os
import gzip
import pickle

from functools import partial  # noqa: F401
from meshmode.mesh.generation import (  # noqa
        ellipse, cloverleaf, NArmedStarfish, drop, n_gon, qbx_peanut,
        WobblyCircle, make_curve_mesh, starfish)
from pytential import bind, sym, norm  # noqa
from pytential.qbx.performance import PerformanceModel
from pytools import one


import logging
import multiprocessing

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


TARGET_ORDER = 8
OVSMP_FACTOR = 5
TCF = 0.9
QBX_ORDER = 5
FMM_ORDER = 15
MESH_TOL = 1e-10
FORCE_STAGE2_UNIFORM_REFINEMENT_ROUNDS = 1
SCALED_MAX_CURVATURE_THRESHOLD = 0.8
MAX_LEAF_REFINE_WEIGHT = 128
RUNS = 1
POOL_WORKERS = min(5, 1 + multiprocessing.cpu_count())

URCHIN_PARAMS = list(range(2, 12, 2))
TUNING_URCHIN = 6
# URCHIN_PARAMS = list(range(2, 7, 2))
DONUT_PARAMS = list(range(1, 6))


DEFAULT_LPOT_KWARGS = dict(
        fmm_backend="fmmlib",
        target_association_tolerance=1e-3,
        fmm_order=FMM_ORDER, qbx_order=QBX_ORDER,
        _box_extent_norm="l2",
        _from_sep_smaller_crit="static_l2",
        _well_sep_is_n_away=2,
        _expansions_in_tree_have_extent=True,
        _expansion_stick_out_factor=TCF,
        _max_leaf_refine_weight=MAX_LEAF_REFINE_WEIGHT,
        _from_sep_smaller_min_nsources_cumul=0,
        _use_target_specific_qbx=False,
        performance_model=PerformanceModel(),
        )


# {{{ general utils

class GeometryGetter(object):

    def __init__(self, getter, label):
        self.getter = getter
        self.label = label

    def __call__(self, queue, lpot_kwargs):
        return self.getter(queue, lpot_kwargs)


def lpot_source_from_mesh(queue, mesh, lpot_kwargs=None):
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
            InterpolatoryQuadratureSimplexGroupFactory)

    target_order = TARGET_ORDER

    pre_density_discr = Discretization(
            queue.context, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    refiner_extra_kwargs = {
            "_force_stage2_uniform_refinement_rounds": (
                 FORCE_STAGE2_UNIFORM_REFINEMENT_ROUNDS),
            "_scaled_max_curvature_threshold": (
                 SCALED_MAX_CURVATURE_THRESHOLD),
            }

    if lpot_kwargs is None:
        lpot_kwargs = DEFAULT_LPOT_KWARGS

    from pytential.qbx import QBXLayerPotentialSource
    lpot_source = QBXLayerPotentialSource(
            pre_density_discr, OVSMP_FACTOR*target_order,
            **lpot_kwargs,)

    lpot_source, _ = lpot_source.with_refinement(**refiner_extra_kwargs)

    return lpot_source


def _urchin_lpot_source(k, queue, lpot_kwargs):
    sph_m = k // 2
    sph_n = k

    from meshmode.mesh.generation import generate_urchin
    mesh = generate_urchin(
            order=TARGET_ORDER, m=sph_m, n=sph_n,
            est_rel_interp_tolerance=MESH_TOL)

    return lpot_source_from_mesh(queue, mesh, lpot_kwargs)


def urchin_geometry_getter(k, label=None):
    if label is None:
        label = k

    return GeometryGetter(partial(_urchin_lpot_source, k), label)


def replicate_along_axes(mesh, shape, sep_ratio):
    from meshmode.mesh.processing import (
            find_bounding_box, affine_map, merge_disjoint_meshes)

    bbox = find_bounding_box(mesh)
    sizes = bbox[1] - bbox[0]

    meshes = [mesh]

    for i in range(mesh.ambient_dim):
        for j in range(1, shape[i]):
            vec = np.zeros(mesh.ambient_dim)
            vec[i] = j * sizes[i] * (1 + sep_ratio)
            meshes.append(affine_map(mesh, A=None, b=vec))

        # FIXME: https://gitlab.tiker.net/inducer/pytential/issues/6
        mesh = merge_disjoint_meshes(meshes, single_group=True)
        meshes = [mesh]

    return mesh


def _torus_lpot_source(r_outer, r_inner, n_outer, n_inner, replicant_shape,
                       sep_ratio, queue, lpot_kwargs):
    from meshmode.mesh.generation import generate_torus
    mesh = generate_torus(
            r_outer, r_inner,
            n_outer, n_inner,
            order=TARGET_ORDER)
    mesh = replicate_along_axes(mesh, replicant_shape, sep_ratio)
    return lpot_source_from_mesh(queue, mesh, lpot_kwargs)


def torus_geometry_getter(r_outer, r_inner, n_outer, n_inner, label):
    if label is None:
        label = (r_outer, r_inner, n_outer, n_inner)

    return GeometryGetter(
            partial(
                _torus_lpot_source,
                r_outer, r_inner,
                n_outer, n_inner,
                (1, 1, 1), 0),
            label)


def donut_geometry_getter(nrows, label=None):
    if label is None:
        label = nrows

    getter = partial(_torus_lpot_source, 2, 1, 40, 20, (2, nrows, 1), 0.1)
    return GeometryGetter(getter, label)


def plane_geometry_getter(label="plane"):
    from inteq_tests import plane_lpot_source
    return GeometryGetter(plane_lpot_source, label)


PARAMS_DIR = "params"
OUTPUT_DIR = "raw-data"
BVP_OUTPUT_DIR = "raw-data-bvp"


def make_output_file(filename, **flags):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return open(os.path.join(OUTPUT_DIR, filename), "w", **flags)


def make_params_file(filename, **flags):
    os.makedirs(PARAMS_DIR, exist_ok=True)
    return open(os.path.join(PARAMS_DIR, filename), "w", **flags)


def load_params(filename, **flags):
    with open(os.path.join(PARAMS_DIR, filename), "r", **flags) as outf:
        return json.load(outf, cls=utils.CostResultDecoder)


def output_data(obj, outfile):
    json.dump(obj, outfile, cls=utils.CostResultEncoder, indent=1)
    if hasattr(outfile, "name"):
        logger.info("Wrote '%s'", outfile.name)

# }}}


# {{{ cost getter

def get_lpot_cost(which, helmholtz_k, geometry_getter, lpot_kwargs, kind):
    """
    Parameters:

        which: "D" or "S"
        kind: "actual" or "model"
    """
    context = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(context)

    lpot_source = geometry_getter(queue, lpot_kwargs)

    from sumpy.kernel import LaplaceKernel, HelmholtzKernel
    sigma_sym = sym.var("sigma")
    if helmholtz_k == 0:
        k_sym = LaplaceKernel(lpot_source.ambient_dim)
        kernel_kwargs = {}
    else:
        k_sym = HelmholtzKernel(lpot_source.ambient_dim, "k")
        kernel_kwargs = {"k": helmholtz_k}

    if which == "S":
        op = sym.S(k_sym, sigma_sym, qbx_forced_limit=+1, **kernel_kwargs)
    elif which == "D":
        op = sym.D(k_sym, sigma_sym, qbx_forced_limit="avg", **kernel_kwargs)
    else:
        raise ValueError("unknown lpot symbol: '%s'" % which)

    bound_op = bind(lpot_source, op)

    density_discr = lpot_source.density_discr
    nodes = density_discr.nodes().with_queue(queue)
    sigma = cl.clmath.sin(10 * nodes[0])

    if kind == "actual":
        timing_data = {}
        result = bound_op.eval(
                queue, {"sigma": sigma}, timing_data=timing_data)
        assert not np.isnan(result.get(queue)).any()
        result = one(timing_data.values())

    elif kind == "model":
        perf_results = bound_op.get_modeled_performance(queue, sigma=sigma)
        result = one(perf_results.values())

    return result

# }}}


# {{{ green error getter

def get_green_error(geometry_getter, lpot_kwargs, center, k,
                    vis_error_filename=None, vis_order=TARGET_ORDER):
    """Return the Green identity error for a geometry.

    The density function for the Green error is the on-surface restriction of
    the potential due to a source charge in the exterior of the geometry, whose
    location is specified. The error is reported relative to the norm of the
    density.

    Params:

        geometry_getter: Geometry getter
        lpot_kwargs: Constructor args to QBXLayerPotentialSource
        center: Center of source charge used to obtain the constructed density
        k: Helmholtz parameter

    Returns:

        A dictionary containing Green identity errors in l^2 and l^infty norm
    """
    context = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(context)
    lpot_source = geometry_getter(queue, lpot_kwargs)

    d = lpot_source.ambient_dim

    u_sym = sym.var("u")
    dn_u_sym = sym.var("dn_u")

    from sumpy.kernel import LaplaceKernel, HelmholtzKernel
    lap_k_sym = LaplaceKernel(d)
    if k == 0:
        k_sym = lap_k_sym
        knl_kwargs = {}
    else:
        k_sym = HelmholtzKernel(d)
        knl_kwargs = {"k": sym.var("k")}

    S_part = (
            sym.S(k_sym, dn_u_sym, qbx_forced_limit=-1, **knl_kwargs))

    D_part = (
            sym.D(k_sym, u_sym, qbx_forced_limit="avg", **knl_kwargs))

    sym_op = S_part - D_part - 0.5 * u_sym

    density_discr = lpot_source.density_discr

    # {{{ compute values of a solution to the PDE

    nodes_host = density_discr.nodes().get(queue)
    normal = bind(density_discr, sym.normal(d))(queue).as_vector(np.object)
    normal_host = [normal[j].get() for j in range(d)]

    if k != 0:
        if d == 2:
            angle = 0.3
            wave_vec = np.array([np.cos(angle), np.sin(angle)])
            u = np.exp(1j*k*np.tensordot(wave_vec, nodes_host, axes=1))
            grad_u = 1j*k*wave_vec[:, np.newaxis]*u
        else:
            diff = nodes_host - center[:, np.newaxis]
            r = la.norm(diff, axis=0)
            u = np.exp(1j*k*r) / r
            grad_u = diff * (1j*k*u/r - u/r**2)
    else:
        diff = nodes_host - center[:, np.newaxis]
        dist_squared = np.sum(diff**2, axis=0)
        dist = np.sqrt(dist_squared)
        if d == 2:
            u = np.log(dist)
            grad_u = diff/dist_squared
        elif d == 3:
            u = 1/dist
            grad_u = -diff/dist**3
        else:
            assert False

    dn_u = 0
    for i in range(d):
        dn_u = dn_u + normal_host[i]*grad_u[i]

    # }}}

    u_dev = cl.array.to_device(queue, u)
    dn_u_dev = cl.array.to_device(queue, dn_u)
    grad_u_dev = cl.array.to_device(queue, grad_u)

    bound_op = bind(lpot_source, sym_op)
    error = bound_op(queue, u=u_dev, dn_u=dn_u_dev, grad_u=grad_u_dev, k=k)

    scaling_l2 = 1 / norm(density_discr, queue, u_dev, p=2)
    scaling_linf = 1 / norm(density_discr, queue, u_dev, p="inf")

    if vis_error_filename is not None:
        from meshmode.discretization.visualization import make_visualizer
        bdry_vis = make_visualizer(queue, lpot_source.density_discr, vis_order)
        bdry_vis.write_vtk_file(vis_error_filename, [
            ("green_zero", error),
            ("u_dev", u_dev),
            ])

    err_l2 = scaling_l2 * norm(density_discr, queue, error, p=2)
    err_linf = scaling_linf * norm(density_discr, queue, error, p="inf")

    return dict(err_l2=err_l2, err_linf=err_linf)

# }}}


# {{{ parameter study - vary parameter with constant geometry

def run_parameter_study(
        param_name, param_values, geometry_getter,
        lpot_kwargs, which_op, helmholtz_k):
    """Run the cost model over a geometry, varying a single parameter value.

    Params:

        param_name: Parameter name (a constructor arg
            to QBXLayerPotentialSource)
        param_values: Range of values to check
        geometry_getter: Geometry getter
        lpot_kwargs: Baseline constructor args to QBXLayerPotentialSource
        which_op: "S" or "D"
        helmholtz_k: Helmholtz parameter

    Returns:
        A list of dictionaries, each of which holds the parameter value and
        cost model result
    """
    param_values = list(param_values)
    task_params = []
    for value in param_values:
        task_param = lpot_kwargs.copy()
        task_param[param_name] = value
        task_params.append(task_param)

    with multiprocessing.Pool(POOL_WORKERS) as pool:
        results = pool.map(
               partial(
                   get_lpot_cost,
                   which_op, helmholtz_k, geometry_getter, kind="model"),
               task_params)

    results = [
            {
                "param_name": param_name,
                "param_value": value,
                "cost": cost}
            for value, cost in zip(param_values, results)]

    return results


def get_optimal_parameter_value(results):
    """Given a list of results as returned by *run_parameter_study*,
    return the parameter value minimizing total cost.
    """
    best_result = min(
            results,
            key=lambda res: sum(res["cost"].get_predicted_times().values()))

    return best_result["param_value"]

# }}}


# {{{ geometry study - vary geometry with constant parameters

def run_geometry_study(geometry_getters, lpot_kwargs, which_op, helmholtz_k):
    """Run the cost model over a set of geometries.

    Params:

        geometry_getters: Geometry getters
        lpot_kwargs: Constructor kwargs to QBXLayerPotentialSource
        which_op: "S" or "D"
        helmholtz_k: Helmholtz parameter

    Returns:

        A list of dictionaries, each of which contain a geometry label and a
        cost model output
    """
    with multiprocessing.Pool(POOL_WORKERS) as pool:
        results = pool.map(
                partial(get_lpot_cost, which_op, helmholtz_k,
                        lpot_kwargs=lpot_kwargs, kind="model"),
                geometry_getters)

    results = [
            {
                "geometry": geo.label,
                "cost": cost}
            for geo, cost in zip(geometry_getters, results)]

    return results

# }}}


# {{{ green error study - obtain green error for a geometry family

def run_green_error_study(
        geometry_getters, lpot_kwargs, center, helmholtz_k):
    """Compute the Green error for a family of geometries, in parallel.

    Params:

        geometry_getters: Geometry getters
        geometry_label: Geometry labels in output
        lpot_kwargs: Constructor kwargs to QBXLayerPotentialSource
        center: Center used for constructed density
        helmholtz_k: Helmholtz parameter

    Returns:

        A list of dictionaries, each of which contain a geometry label and an
        error result
    """
    with multiprocessing.Pool(POOL_WORKERS) as pool:
        err_results = pool.map(
                partial(
                    get_green_error,
                    lpot_kwargs=lpot_kwargs, center=center, k=helmholtz_k),
                geometry_getters)

    results = [
            {
                "geometry": geo.label,
                "error": err}
            for geo, err in zip(geometry_getters, err_results)]

    return results

# }}}


# {{{ tune parameters for a single geometry

def run_tuning_study(
        tuning_geometry, lpot_kwargs, baseline_nmax_range,
        baseline_nmpole_range, tsqbx_nmax_range,
        tsqbx_nmpole_range, which_op, helmholtz_k):
    """Find the parameters which give the best observed performance, with and
    without target-specific QBX.

    Params:

        tuning_geometry: Geometry getter
        label: Label for saving results
        lpot_kwargs: Base kwargs for the QBXLayerPotentialSource
        which_op: "S" or "D"
        helmholtz_k: Helmholtz parameter

        The arguments *baseline_nmax_range*, *baseline_nmpole_range*,
        *tsqbx_nmax_range*, *tsqbx_nmpole_range* are the ranges of values which
        are to be checked for the respective parameter value.
    """
    lpot_kwargs = lpot_kwargs.copy()
    label = tuning_geometry.label

    # {{{ figure out baseline nmax

    logger.info("finding baseline value of nmax")

    baseline_nmax_results = run_parameter_study(
            "_max_leaf_refine_weight",
            baseline_nmax_range,
            tuning_geometry,
            lpot_kwargs,
            which_op,
            helmholtz_k)

    output_fname = f"tuning-study-{label}-baseline-nmax.json"
    with make_output_file(output_fname) as outfile:
        output_data(baseline_nmax_results, outfile)

    baseline_nmax = get_optimal_parameter_value(baseline_nmax_results)
    lpot_kwargs["_max_leaf_refine_weight"] = baseline_nmax
    logger.info("baseline value of nmax: %d", baseline_nmax)

    # }}}

    # {{{ figure out baseline nmpole

    logger.info("finding baseline value of nmpole")

    baseline_nmpole_results = run_parameter_study(
            "_from_sep_smaller_min_nsources_cumul",
            baseline_nmpole_range,
            tuning_geometry,
            lpot_kwargs,
            which_op,
            helmholtz_k)

    output_fname = f"tuning-study-{label}-baseline-nmpole.json"
    with make_output_file(output_fname) as outfile:
        output_data(baseline_nmpole_results, outfile)

    baseline_nmpole = get_optimal_parameter_value(baseline_nmpole_results)
    lpot_kwargs["_from_sep_smaller_min_nsources_cumul"] = baseline_nmpole
    logger.info("baseline value of nmpole: %d", baseline_nmpole)

    # }}}

    # {{{ figure out nmax for tsqbx

    logger.info("finding optimal nmax value when using tsqbx")
    lpot_kwargs["_use_target_specific_qbx"] = True

    tsqbx_nmax_results = run_parameter_study(
            "_max_leaf_refine_weight",
            tsqbx_nmax_range,
            tuning_geometry,
            lpot_kwargs,
            which_op,
            helmholtz_k)

    output_fname = f"tuning-study-{label}-tsqbx-nmax.json"
    with make_output_file(output_fname) as outfile:
        output_data(tsqbx_nmax_results, outfile)

    tsqbx_nmax = get_optimal_parameter_value(tsqbx_nmax_results)
    lpot_kwargs["_max_leaf_refine_weight"] = tsqbx_nmax
    logger.info("optimal nmax value for tsqbx: %d", tsqbx_nmax)

    # }}}

    # {{{ figure out nmpole for tsqbx

    logger.info("finding optimal nmpole value for tsqbx")

    tsqbx_nmpole_results = run_parameter_study(
            "_from_sep_smaller_min_nsources_cumul",
            tsqbx_nmpole_range,
            tuning_geometry,
            lpot_kwargs,
            which_op,
            helmholtz_k)

    output_fname = f"tuning-study-{label}-tsqbx-nmpole.json"
    with make_output_file(output_fname) as outfile:
        output_data(tsqbx_nmpole_results, outfile)

    tsqbx_nmpole = get_optimal_parameter_value(tsqbx_nmpole_results)
    lpot_kwargs["_from_sep_smaller_min_nsources_cumul"] = tsqbx_nmpole
    logger.info("optimal nmpole value for tsqbx: %d", tsqbx_nmpole)

    # }}}

    result = dict(
            baseline_nmax=baseline_nmax,
            baseline_nmpole=baseline_nmpole,
            tsqbx_nmax=tsqbx_nmax,
            tsqbx_nmpole=tsqbx_nmpole)

    params_fname = f"tuning-params-{label}.json"
    with make_params_file(params_fname) as outfile:
        output_data(result, outfile)

# }}}


# {{{ collect results of applying optimizations on a set of geometries

def run_optimization_study(
        geometry_getters, label, lpot_kwargs, params, which_op, helmholtz_k):
    """Apply a sequence of optimizations to a set of geometries and record
    performance results.

    Params:

        geometry_getters: List of geometry getters
        label: Label for saving results
        lpot_kwargs: Baseline kwargs for the QBXLayerPotentialSource
        params: Params obtained as result of *run_tuning_study*
        which_op: "S" or "D"
        helmholtz_k: Helmholtz parameter
    """

    # {{{ opt level 0

    logger.info("Obtaining baseline performance")

    lpot_kwargs = lpot_kwargs.copy()

    lpot_kwargs["_use_target_specific_qbx"] = False
    lpot_kwargs["_max_leaf_refine_weight"] = params["baseline_nmax"]
    lpot_kwargs["_from_sep_smaller_min_nsources_cumul"] = (
            params["baseline_nmpole"])

    opt0_results = run_geometry_study(
            geometry_getters,
            lpot_kwargs,
            which_op,
            helmholtz_k)

    output_fname = f"optimization-study-{label}-opt0.json"
    with make_output_file(output_fname) as outfile:
        output_data(opt0_results, outfile)

    # }}}

    # {{{ opt level 1

    logger.info("Obtaining performance with TSQBX")

    lpot_kwargs["_use_target_specific_qbx"] = True

    opt1_results = run_geometry_study(
            geometry_getters,
            lpot_kwargs,
            which_op,
            helmholtz_k)

    output_fname = f"optimization-study-{label}-opt1.json"
    with make_output_file(output_fname) as outfile:
        output_data(opt1_results, outfile)

    # }}}

    # {{{ opt level 2

    logger.info("Obtaining performance with TSQBX + optimal nmax")

    lpot_kwargs["_max_leaf_refine_weight"] = params["tsqbx_nmax"]

    opt2_results = run_geometry_study(
            geometry_getters,
            lpot_kwargs,
            which_op,
            helmholtz_k)

    output_fname = f"optimization-study-{label}-opt2.json"
    with make_output_file(output_fname) as outfile:
        output_data(opt2_results, outfile)

    # }}}

    # {{{ opt level 3

    logger.info(
            "Obtaining performance with TSQBX + "
            "optimal nmax + optimal nmpole")

    lpot_kwargs["_from_sep_smaller_min_nsources_cumul"] = (
            params["tsqbx_nmpole"])

    opt3_results = run_geometry_study(
            geometry_getters,
            lpot_kwargs,
            which_op,
            helmholtz_k)

    output_fname = f"optimization-study-{label}-opt3.json"
    with make_output_file(output_fname) as outfile:
        output_data(opt3_results, outfile)

    # }}}

# }}}


# {{{ lpot kwargs

def urchin_lpot_kwargs():
    lpot_kwargs = DEFAULT_LPOT_KWARGS.copy()

    lpot_kwargs["performance_model"] = PerformanceModel(
            calibration_params=load_params(
                "calibration-params-urchin.json"))

    assert lpot_kwargs["fmm_order"] == 15
    assert lpot_kwargs["qbx_order"] == 5

    return lpot_kwargs


def donut_lpot_kwargs():
    lpot_kwargs = DEFAULT_LPOT_KWARGS.copy()

    lpot_kwargs["performance_model"] = PerformanceModel(
            calibration_params=load_params(
                "calibration-params-donut.json"))

    lpot_kwargs["fmm_order"] = 20
    lpot_kwargs["qbx_order"] = 9

    return lpot_kwargs


def plane_lpot_kwargs():
    lpot_kwargs = DEFAULT_LPOT_KWARGS.copy()

    lpot_kwargs["performance_model"] = PerformanceModel(
            calibration_params=load_params(
                "calibration-params-plane.json"))

    # These are supplied by the geometry getter.
    del lpot_kwargs["qbx_order"]
    del lpot_kwargs["fmm_order"]
    del lpot_kwargs["fmm_backend"]

    return lpot_kwargs

# }}}


# {{{ fit calibration params

def fit_calibration_params(geometry_getters, lpot_kwargs_list, which_op, helmholtz_k):
    """Find calibration parameters for running a layer potential operator
    on a particular list of geometries.

    Params:
        geometry_getters: A list of geometry getters
        lpot_kwargs_list: A list of corresponding lpot kwargs
        which_op: "S" or "D"
        helmholtz_k: Helmholtz parameter

    Returns:
        The calibration parameters as a dictionary.
    """
    context = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(context)

    from pytential.qbx.performance import (
            PerformanceModel, estimate_calibration_params)

    model_results = []
    timing_results = []

    for geo_getter, lpot_kwargs in zip(geometry_getters, lpot_kwargs_list):
        model_result = get_lpot_cost(which_op, helmholtz_k,
            geo_getter, lpot_kwargs, "model")
        model_results.append(model_result)
        timing_result = get_lpot_cost(which_op, helmholtz_k,
            geo_getter, lpot_kwargs, "actual")
        timing_results.append(timing_result)

    return estimate_calibration_params(model_results, timing_results)

# }}}


def run_urchin_calibration_params_experiment():
    urchins = (
            urchin_geometry_getter(3),
            urchin_geometry_getter(3),
            urchin_geometry_getter(5),
            urchin_geometry_getter(5))

    lpot_kwargs_nots = urchin_lpot_kwargs()
    lpot_kwargs_ts = lpot_kwargs_nots.copy()
    lpot_kwargs_ts["_use_target_specific_qbx"] = True

    lpot_kwargs_list = (
            lpot_kwargs_nots,
            lpot_kwargs_ts,
            lpot_kwargs_nots,
            lpot_kwargs_ts)

    result = fit_calibration_params(urchins, lpot_kwargs_list, "S", 0)

    with make_params_file("calibration-params-urchin.json") as outfile:
        output_data(result, outfile)


def run_urchin_time_prediction_experiment():
    urchins = [urchin_geometry_getter(k) for k in URCHIN_PARAMS]

    results = run_geometry_study(urchins, urchin_lpot_kwargs(), "S", 0)

    with make_output_file("time-prediction-urchin-modeled-costs.json")\
            as outfile:
        output_data(results, outfile)


def run_urchin_tuning_study_experiment():
    tuning_urchin = urchin_geometry_getter(TUNING_URCHIN, "urchin")

    baseline_nmax_range = range(32, 512, 32)
    baseline_nmpole_range = range(0, 300, 20)
    tsqbx_nmax_range = range(32, 2000, 64)
    tsqbx_nmpole_range = range(0, 500, 20)

    run_tuning_study(
            tuning_urchin, urchin_lpot_kwargs(),
            baseline_nmax_range, baseline_nmpole_range,
            tsqbx_nmax_range, tsqbx_nmpole_range,
            which_op="S", helmholtz_k=0)


def run_urchin_optimization_study_experiment():
    tuning_params = load_params("tuning-params-urchin.json")

    urchins = [urchin_geometry_getter(k) for k in URCHIN_PARAMS]

    run_optimization_study(
            urchins, "urchin", urchin_lpot_kwargs(),
            tuning_params, "S", helmholtz_k=0)


def run_urchin_green_error_experiment():
    urchins = [urchin_geometry_getter(k) for k in URCHIN_PARAMS]
    center = np.array([3., 1., 2.])

    results = run_green_error_study(
            urchins, urchin_lpot_kwargs(), center, helmholtz_k=0)

    with make_output_file("green-error-urchin.json") as outfile:
        output_data(results, outfile)


def run_donut_calibration_params_experiment():
    # nrows=5 is the same as tau_{10}
    donuts = (
            donut_geometry_getter(5, "donut"),
            donut_geometry_getter(5, "donut"))

    lpot_kwargs_nots = donut_lpot_kwargs()
    lpot_kwargs_ts = lpot_kwargs_nots.copy()
    lpot_kwargs_ts["_use_target_specific_qbx"] = True

    lpot_kwargs_list = (
            lpot_kwargs_nots,
            lpot_kwargs_ts)

    result = fit_calibration_params(donuts, lpot_kwargs_list, "S", 0)

    with make_params_file("calibration-params-donut.json") as outfile:
        output_data(result, outfile)


def run_donut_tuning_study_experiment():
    # nrows=5 is the same as tau_{10}
    tuning_donut = donut_geometry_getter(5, "donut")

    baseline_nmax_range = range(32, 512, 32)
    baseline_nmpole_range = range(0, 300, 20)
    tsqbx_nmax_range = range(32, 2000, 64)
    tsqbx_nmpole_range = range(0, 500, 20)

    run_tuning_study(
            tuning_donut, donut_lpot_kwargs(),
            baseline_nmax_range, baseline_nmpole_range,
            tsqbx_nmax_range, tsqbx_nmpole_range,
            which_op="S", helmholtz_k=0)


def run_donut_optimization_study_experiment():
    tuning_params = load_params("tuning-params-donut.json")
    donut = [donut_geometry_getter(5)]

    run_optimization_study(
            donut, "donut", donut_lpot_kwargs(),
            tuning_params, "S", helmholtz_k=0)


def run_donut_green_error_experiment():
    donut = [donut_geometry_getter(5)]
    center = np.array([0.] * 3)

    results = run_green_error_study(
            donut, donut_lpot_kwargs(), center, helmholtz_k=0)

    with make_output_file("green-error-donut.json") as outfile:
        output_data(results, outfile)


def run_plane_calibration_params_experiment():
    planes = (
            plane_geometry_getter(),
            plane_geometry_getter())

    lpot_kwargs_nots = plane_lpot_kwargs()
    lpot_kwargs_ts = lpot_kwargs_nots.copy()
    lpot_kwargs_ts["_use_target_specific_qbx"] = True

    lpot_kwargs_list = (
            lpot_kwargs_nots,
            lpot_kwargs_ts)

    result = fit_calibration_params(planes, lpot_kwargs_list, "D", 20)

    with make_params_file("calibration-params-plane.json") as outfile:
        output_data(result, outfile)


def run_plane_tuning_study_experiment():
    tuning_plane = plane_geometry_getter()

    baseline_nmax_range=range(50, 200, 50)
    baseline_nmpole_range=range(10, 100, 10)
    tsqbx_nmax_range=range(100, 500, 50)
    tsqbx_nmpole_range=range(50, 300, 50)

    run_tuning_study(
            tuning_plane, plane_lpot_kwargs(),
            baseline_nmax_range, baseline_nmpole_range,
            tsqbx_nmax_range, tsqbx_nmpole_range,
            which_op="D", helmholtz_k=20)


def run_plane_optimization_study_experiment():
    tuning_params = load_params("tuning-params-plane.json")
    plane = [plane_geometry_getter()]

    run_optimization_study(
            plane, "plane", plane_lpot_kwargs(),
            tuning_params, "D", helmholtz_k=20)


def run_plane_bvp_experiment():
    if any(
            os.path.exists(os.path.join(BVP_OUTPUT_DIR, fname))
            for fname in (
                "potential-0.25.vts", "result.pkl.gz", "source-0.25.vtu")):
        raise RuntimeError(
                "not running plane-bvp experiment - delete or move "
                "the output files in "
                "the directory '%s' to run" % BVP_OUTPUT_DIR)

    lpot_kwargs = plane_lpot_kwargs()

    tuning_params = load_params("tuning-params-plane.json")
    lpot_kwargs["_use_target_specific_qbx"] = True
    lpot_kwargs["_max_leaf_refine_weight"] = (
            tuning_params["tsqbx_nmax"])
    lpot_kwargs["_from_sep_smaller_min_nsources_cumul"] = (
            tuning_params["tsqbx_nmpole"])

    cl_ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(cl_ctx)

    from inteq_tests import (
            run_int_eq_test, BetterplaneIntEqTestCase)

    result = run_int_eq_test(
            cl_ctx,
            queue,
            BetterplaneIntEqTestCase(20, "dirichlet", +1),
            resolution=0.25,
            visualize=True,
            lpot_kwargs=lpot_kwargs,
            output_dir=BVP_OUTPUT_DIR)

    gmres_result = result.gmres_result

    result_dict = dict(
            h_max=result.h_max,
            rel_err_2=result.rel_err_2,
            rel_err_inf=result.rel_err_inf,
            rel_td_err_inf=result.rel_td_err_inf,
            gmres_result=dict(
                solution=gmres_result.solution.get(queue),
                residual_norms=gmres_result.residual_norms,
                iteration_count=gmres_result.iteration_count,
                success=gmres_result.success,
                stat=gmres_result.state))

    with gzip.open(os.path.join(BVP_OUTPUT_DIR, "result.pkl.gz"), "wb")\
            as outfile:
        pickle.dump(result_dict, outfile)


def run_experiments(experiments):
    # Urchin calibration params
    if "urchin-calibration-params" in experiments:
        run_urchin_calibration_params_experiment()

    # Time prediction
    if "urchin-time-prediction" in experiments:
        run_urchin_time_prediction_experiment()

    # Tuning study for urchins
    if "urchin-tuning-study" in experiments:
        run_urchin_tuning_study_experiment()

    # Optimization study for urchin family
    if "urchin-optimization-study" in experiments:
        run_urchin_optimization_study_experiment()

    # Green error for urchin family
    if "urchin-green-error" in experiments:
        run_urchin_green_error_experiment()

    # Torus grid calibration params
    if "donut-calibration-params" in experiments:
        run_donut_calibration_params_experiment()

    # Optimization study for torus grid
    if "donut-optimization-study" in experiments:
        run_donut_optimization_study_experiment()

    # Tuning study for torus grid
    if "donut-tuning-study" in experiments:
        run_donut_tuning_study_experiment()

    # Green error for torus grid
    if "donut-green-error" in experiments:
        run_donut_green_error_experiment()

    # Plane calibration params
    if "plane-calibration-params" in experiments:
        run_plane_calibration_params_experiment()

    # Plane tuning study
    if "plane-tuning-study" in experiments:
        run_plane_tuning_study_experiment()

    # Plane tuning study
    if "plane-optimization-study" in experiments:
        run_plane_optimization_study_experiment()

    # Plane BVP
    if "plane-bvp" in experiments:
        run_plane_bvp_experiment()


EXPERIMENTS = (
        "urchin-calibration-params",
        "urchin-time-prediction",
        "urchin-tuning-study",
        "urchin-optimization-study",
        "urchin-green-error",

        "donut-calibration-params",
        "donut-tuning-study",
        "donut-optimization-study",
        "donut-green-error",

        "plane-calibration-params",
        "plane-tuning-study",
        "plane-optimization-study",
        "plane-bvp",
)


def main():
    description = "This script collects data from one or more experiments."
    experiments = utils.parse_args(description, EXPERIMENTS)
    run_experiments(experiments)


if __name__ == "__main__":
    # Avoid issues with fork()-based multiprocessing and pyopencl - see
    # https://github.com/inducer/pyopencl/issues/156
    multiprocessing.set_start_method("spawn")
    main()


# vim: foldmethod=marker
