"""JAX implementation of the dense qpth QP layer API."""

__all__ = ["QPFunction", "solve_qp"]


def __getattr__(name):
    if name in __all__:
        from .qp import QPFunction, solve_qp

        return {"QPFunction": QPFunction, "solve_qp": solve_qp}[name]
    raise AttributeError(name)
