import json
import numpy as np
import pyopencl as cl
import pyopencl.clmath  # noqa
import pytest
from pyopencl.tools import pytest_generate_tests_for_pyopencl \
        as pytest_generate_tests


from pytential.qbx.performance import ParametrizedCosts
from boxtree.fmm import TimingResult


class CostResultEncoder(json.JSONEncoder):
    """JSON encoder supporting serialization of cost results."""

    def default(self, obj):
        if isinstance(obj, ParametrizedCosts):
            raw_costs = {}
            for key, val in obj.raw_costs.items():
                raw_costs[key] = str(val)

            return {
                    "_cost_result_type": "ParametrizedCosts",
                    "_data": {
                        "params": obj.params,
                        "raw_costs": raw_costs,
                    },
            }

        elif isinstance(obj, TimingResult):
            return {
                    "_cost_result_type": "TimingResult",
                    "_data": dict(obj.items()),
            }

        return obj


class CostResultDecoder(json.JSONDecoder):
    """JSON decoder supporting serialization of cost results."""

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        type = obj.get("_cost_result_type")
        data = obj.get("_data")

        if type == "ParametrizedCosts":
            from pymbolic import parse
            raw_costs = {}
            for key, val in data["raw_costs"].items():
                raw_costs[key] = parse(val)

            return ParametrizedCosts(
                    params=data["params"],
                    raw_costs=raw_costs)

        elif type == "TimingResult":
            return TimingResult(data)

        return obj


def get_sphere_lpot_source(queue):
    from meshmode.discretization import Discretization
    from meshmode.discretization.poly_element import (
            InterpolatoryQuadratureSimplexGroupFactory)

    target_order = 8
    from meshmode.mesh.generation import generate_icosphere
    mesh = generate_icosphere(1, target_order)

    pre_density_discr = Discretization(
            queue.context, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

    from pytential.qbx import QBXLayerPotentialSource
    lpot_source = QBXLayerPotentialSource(
            pre_density_discr, qbx_order=5, fmm_order=10, fine_order=4*target_order,
            fmm_backend="fmmlib")
    lpot_source, _ = lpot_source.with_refinement()

    return lpot_source


def get_lpot_cost(queue, geometry_getter, kind):
    lpot_source = geometry_getter(queue)

    from pytential import sym, bind
    sigma_sym = sym.var("sigma")
    from sumpy.kernel import LaplaceKernel
    k_sym = LaplaceKernel(lpot_source.ambient_dim)
    op = sym.S(k_sym, sigma_sym, qbx_forced_limit=+1)

    bound_op = bind(lpot_source, op)

    density_discr = lpot_source.density_discr
    nodes = density_discr.nodes().with_queue(queue)
    sigma = cl.clmath.sin(10 * nodes[0])

    from pytools import one
    if kind == "actual":
        timing_data = {}
        result = bound_op.eval(queue, {"sigma": sigma}, timing_data=timing_data)
        assert not np.isnan(result.get(queue)).any()
        result = one(timing_data.values())

    elif kind == "model":
        perf_results = bound_op.get_modeled_performance(queue, sigma=sigma)
        result = one(perf_results.values())

    return result


@pytest.mark.parametrize("kind", ("actual", "model"))
def test_cost_result_serialization(ctx_factory, kind):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    cost = get_lpot_cost(queue, get_sphere_lpot_source, kind)
    cost_str = json.dumps(cost, cls=CostResultEncoder)
    cost_decoded = json.loads(cost_str, cls=CostResultDecoder)

    if kind == "actual":
        assert dict(cost_decoded.items()) == dict(cost.items())

    elif kind == "model":
        assert cost.params == cost_decoded.params
        assert cost.raw_costs == cost_decoded.raw_costs

    else:
        raise ValueError("unknown kind: '%s'" % kind)


if __name__ == "__main__":
    pytest.main([__file__])
