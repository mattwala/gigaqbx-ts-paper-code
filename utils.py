import collections
import enum
import json
import numpy as np
import pyopencl as cl
import pyopencl.clmath  # noqa
import pytest
import re
from pyopencl.tools import pytest_generate_tests_for_pyopencl as pytest_generate_tests  # noqa
from pymbolic.mapper import Mapper

from pytential.qbx.performance import ParametrizedCosts
from boxtree.fmm import TimingResult


class SExprStringifier(Mapper):

    def build_sexpr(self, *args):
        args = " ".join(args)
        return "".join(["(", args, ")"])

    def map_constant(self, expr):
        assert type(expr) in (int, float)
        return repr(expr)

    def map_variable(self, expr):
        assert _STRING_RE.fullmatch(repr(expr.name))
        return self.build_sexpr("Var", repr(expr.name))

    def map_sum(self, expr):
        return self.build_sexpr(
                "Sum", *[self.rec(child) for child in expr.children])

    def map_product(self, expr):
        return self.build_sexpr(
                "Product", *[self.rec(child) for child in expr.children])

    def map_power(self, expr):
        return self.build_sexpr(
                "Power", self.rec(expr.base), self.rec(expr.exponent))


def pymbolic_to_sexpr(expr):
    return SExprStringifier()(expr)


_WHITESPACE_RE = re.compile("[ \t\n]+")
_LPAREN_RE = re.compile(r"\(")
_RPAREN_RE = re.compile(r"\)")
_FLOAT_RE = re.compile(r"[-+]?([0-9]+\.[0-9]*)([eE][-+]?[0-9]+)?")
_INT_RE = re.compile("[-+]?[0-9][0-9]*")
_IDENT_RE = re.compile("[a-zA-Z_][a-zA-Z_0-9]*")
_STRING_RE = re.compile("'[a-zA-Z_0-9]*'")


TokenType = enum.Enum(
        "TokenType", "LPAREN, RPAREN, FLOAT, INT, IDENT, STRING, END")


Token = collections.namedtuple("Token", "token_type, data, pos")


class ParserError(RuntimeError):
    pass


def lex(s):
    pos = 0

    match = None

    def matches(reg):
        nonlocal match
        match = reg.match(s, pos)
        return match

    while pos < len(s):
        if matches(_WHITESPACE_RE):
            pass

        elif matches(_LPAREN_RE):
            yield Token(TokenType.LPAREN, None, pos)

        elif matches(_RPAREN_RE):
            yield Token(TokenType.RPAREN, None, pos)

        elif matches(_FLOAT_RE):
            yield Token(TokenType.FLOAT, float(s[pos:match.end()]), pos)

        elif matches(_INT_RE):
            yield Token(TokenType.INT, int(s[pos:match.end()]), pos)

        elif matches(_IDENT_RE):
            yield Token(TokenType.IDENT, s[pos:match.end()], pos)

        elif matches(_STRING_RE):
            yield Token(TokenType.STRING, s[pos:match.end()], pos)

        else:
            raise ParserError(
                    "unexpected token starting at position %d" % pos)

        pos = match.end()

    yield Token(TokenType.END, None, pos)


def sexpr_to_pymbolic(expr):
    tokens = lex(expr)
    token = next(tokens)
    from pymbolic.primitives import Variable, Sum, Product, Power

    def error(token):
        raise ParserError(
                "unexpected token starting at position %d (got '%r')"
                % (token.pos, token))

    def next_token():
        nonlocal token
        try:
            token = next(tokens)
        except StopIteration:
            raise ParserError("unexpected end of input")

    def expect(*token_types):
        next_token()
        if token.token_type not in token_types:
            error(token)
        return token

    def parse_expr():
        if token.token_type == TokenType.LPAREN:
            expect(TokenType.IDENT)

            if token.data == "Var":
                expect(TokenType.STRING)
                varname = token.data[1:-1]
                expect(TokenType.RPAREN)
                next_token()
                return Variable(varname)

            elif token.data == "Sum":
                next_token()
                children = parse_children()
                return Sum(children)

            elif token.data == "Product":
                next_token()
                children = parse_children()
                return Product(children)

            elif token.data == "Power":
                next_token()
                base = parse_expr()
                exp = parse_expr()
                if token.token_type != TokenType.RPAREN:
                    error(token)
                next_token()
                return Power(base, exp)

            else:
                error(token)

        elif token.token_type in (TokenType.INT, TokenType.FLOAT):
            val = token.data
            next_token()
            return val

        else:
            error(token)

    def parse_children():
        children = [parse_expr()]

        while True:
            if token.token_type == TokenType.RPAREN:
                next_token()
                return tuple(children)
            children.append(parse_expr())

    result = parse_expr()
    if token.token_type != TokenType.END:
        error(token)

    return result


class CostResultEncoder(json.JSONEncoder):
    """JSON encoder supporting serialization of cost results."""

    def default(self, obj):
        if isinstance(obj, ParametrizedCosts):
            raw_costs = {}
            for key, val in obj.raw_costs.items():
                raw_costs[key] = pymbolic_to_sexpr(val)

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
            raw_costs = {}
            for key, val in data["raw_costs"].items():
                raw_costs[key] = sexpr_to_pymbolic(val)

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
            pre_density_discr, qbx_order=5, fmm_order=10,
            fine_order=4*target_order,
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
        result = bound_op.eval(
                queue, {"sigma": sigma}, timing_data=timing_data)
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


def test_pymbolic_sexprs():
    def check_round_trip(expr):
        assert sexpr_to_pymbolic(pymbolic_to_sexpr(expr)) == expr

    from pymbolic.primitives import Variable, Sum, Product, Power
    check_round_trip(Variable("x"))
    check_round_trip(1)
    check_round_trip(-11)
    check_round_trip(1.1)
    check_round_trip(1.1e-2)
    check_round_trip(Sum((7,)))
    check_round_trip(Sum((1, 2, 3)))
    check_round_trip(
            Sum((1, Product((2, 3, Power(1, Sum((Product((-1, 2)), 2))))), 3)))
    check_round_trip(Product((1, 2, 3)))
    check_round_trip(Power(1, Variable("x")))
    check_round_trip(Power(Power(1, 2), 3))
    check_round_trip(Power(1, Power(2, 3)))
    check_round_trip(Power(Power(Sum((1, 2)), 3), 3.5))

    from pymbolic import parse
    check_round_trip(
            parse(
                "c_m2l * (40 * ((p_fmm + 1)**2)"
                "** 1.5 * (p_qbx + 1) ** 0.5 + 1)"))

    def check_error(expr):
        with pytest.raises(ParserError):
            sexpr_to_pymbolic(expr)

    check_error("")
    check_error(")")
    check_error("()")
    check_error("1 2 3")
    check_error("(Power 1)")
    check_error("(Var ''')")
    check_error("(Power (Var 'x'")
    check_error("(Power 1 2 3)")
    check_error("(Product 1 2) (Sum 1 2)")
    check_error("(Sum)")
    check_error("(Var 4)")
    check_error("(Product Sum)")
    check_error("(Sum (Sum 1 2) 3")


if __name__ == "__main__":
    pytest.main([__file__])
