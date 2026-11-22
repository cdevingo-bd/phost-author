"""Emitters: how a checked equation becomes something that runs on an array backend.

A stage declares the form its law takes (``phost.equation``); ``laws`` and ``program`` check the text into trees; a backend
turns the trees into a callable for the arrays a host's ports carry. The backend is chosen by the host's ``backend`` name
(``"numpy"``, ``"warp"``). Adding one (JAX, Torch, CUDA C, ...) is one class with three methods registered here, and touches
no stage and no grammar:

    class MyBackend:
        name = "jax"
        def pointwise(self, plan): ...    # -> run(system, constants)
        def channels(self, plan): ...     # -> run(system, constants)
        def iteration(self, plan): ...    # -> run(system, constants)
    register_backend(MyBackend())

``run`` is called once per execution of the stage with the stage's system (its ports as attributes) and the constants ``a``..``h``
as a mapping of floats. It writes the outputs into the system's ports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from phost import CapabilityError, Channels, Iteration, Pointwise, StageContract

from ..expr import Expression
from ..program import Program

Run = Callable[[Any, Mapping[str, float]], None]


@dataclass(frozen=True)
class PointwisePlan:
    contract: StageContract
    form: Pointwise
    laws: Mapping[str, Expression]               # output name -> its law; an output that is absent is 0
    slopes: Mapping[str, Expression]             # output name -> d law / d (form.slope.of); empty when the slope is off


@dataclass(frozen=True)
class ChannelsPlan:
    contract: StageContract
    form: Channels
    channels: tuple[tuple[str, Expression], ...]


@dataclass(frozen=True)
class IterationPlan:
    contract: StageContract
    form: Iteration
    program: Program
    iterations: int


class Backend(Protocol):
    name: str

    def pointwise(self, plan: PointwisePlan) -> Run: ...
    def channels(self, plan: ChannelsPlan) -> Run: ...
    def iteration(self, plan: IterationPlan) -> Run: ...


BACKENDS: dict[str, Backend] = {}


def register_backend(backend: Backend, *, replace: bool = False) -> Backend:
    if backend.name in BACKENDS and not replace and BACKENDS[backend.name] is not backend:
        raise ValueError(f"a backend named {backend.name!r} is already registered")
    BACKENDS[backend.name] = backend
    return backend


def backend_for(name: str) -> Backend:
    if name not in BACKENDS:
        _load_shipped(name)
    try:
        return BACKENDS[name]
    except KeyError:
        raise CapabilityError(f"no typed-equation backend for host arrays of kind {name!r}; known: {sorted(BACKENDS)}. "
                              "Register one with phost_author.backends.register_backend") from None


def _load_shipped(name: str) -> None:
    if name == "numpy":
        from . import numpy_backend            # noqa: F401  (registers itself)
    elif name == "warp":
        from . import warp_backend             # noqa: F401
