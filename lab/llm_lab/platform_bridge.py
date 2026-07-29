"""Expose the book's `llm_platform` modules in the Streamlit lab.

The LangChain lab covers eight systems end to end. The companion project has 41
modules, and 30 of them were reachable from no UI at all — no HTTP route, no lab
page — so a reader could not run the code a chapter was describing.

This module closes that gap generically. Instead of hand-writing a page per
module, it introspects `llm_platform`, turns each public callable into a
:class:`Demo` with a machine-readable parameter spec, and lets the UI render an
input widget per parameter. Adding a function to `llm_platform` makes it appear
in the lab automatically.

    from llm_lab.platform_bridge import catalog, run_demo
    demos = catalog()["Chapter 9-11 · Inference & Serving"]
    run_demo(demos["paged_kv_blocks"], seq_len=1000, block_size=16)

Everything here is offline and dependency-free: it calls the same functions the
project's own pytest suite covers.
"""
from __future__ import annotations

import dataclasses
import importlib
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

LAB_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = LAB_DIR.parent / "project"

if PROJECT_DIR.is_dir() and str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


class PlatformUnavailable(RuntimeError):
    """Raised when the `project/` package cannot be imported."""


# --------------------------------------------------------------------------- #
# Which module belongs to which part of the book.
# --------------------------------------------------------------------------- #
GROUPS: List[Tuple[str, Tuple[str, ...]]] = [
    ("Ch 1 · Capacity & Cost Estimation", ("estimation",)),
    ("Ch 4-8 · Training Systems", ("training",)),
    ("Ch 9-11 · Inference & Serving", ("serving",)),
    ("Ch 12 · API Gateway", ("gateway",)),
    ("Ch 13-16 · RAG", ("chunking", "retrieval_hybrid", "rag_eval")),
    ("Ch 17-20 · Agents", ("agents", "multi_agent", "tools", "memory")),
    ("Ch 21-23 · Safety", ("safety",)),
    ("Ch 24-26 · Data Platform", ("context_assembler", "feedback", "sessions")),
    ("Ch 27-29 · Evaluation", ("metrics", "eval_offline", "experiments", "observability")),
    ("Ch 30-32 · Scale & Reliability", ("scaling", "reliability", "caching")),
    ("Ch 33-40 · Reference Blueprints", ("blueprints",)),
    ("Ch 41-42 · Cost Engineering", ("cost",)),
    ("Ch 43-45 · MLOps", ("registry",)),
    ("Ch 46-50 · Emerging Patterns", ("compound", "reasoning", "llm_gateway")),
]

#: Callables that need live objects rather than scalars; the UI links to the
#: dedicated page instead of trying to synthesise arguments.
_NEEDS_OBJECTS = {"LLMPipeline", "RAGQueryEngine", "RAGIndex"}


# --------------------------------------------------------------------------- #
# Parameter model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Param:
    """One argument of a demo, described well enough for a UI to render it."""

    name: str
    kind: str  # "int" | "float" | "bool" | "str" | "seq" | "any"
    default: Any = inspect.Parameter.empty
    required: bool = True
    annotation: str = ""

    @property
    def has_default(self) -> bool:
        return self.default is not inspect.Parameter.empty


@dataclass(frozen=True)
class Demo:
    """A single callable from `llm_platform`, ready to be driven from a UI."""

    module: str
    name: str
    doc: str
    params: Tuple[Param, ...]
    func: Callable[..., Any] = field(repr=False, compare=False, default=None)

    @property
    def qualname(self) -> str:
        return f"llm_platform.{self.module}.{self.name}"

    @property
    def runnable(self) -> bool:
        """True when every required parameter can be supplied from a form."""
        return all(p.kind != "any" or not p.required for p in self.params)

    def source(self) -> str:
        try:
            return inspect.getsource(self.func)
        except (OSError, TypeError):  # pragma: no cover - defensive
            return f"# source unavailable for {self.qualname}"


_KIND_BY_TYPE = {int: "int", float: "float", bool: "bool", str: "str"}


def _classify(annotation: Any, default: Any) -> str:
    """Map an annotation (or a default value) onto a widget kind."""
    if annotation in _KIND_BY_TYPE:
        return _KIND_BY_TYPE[annotation]
    text = str(annotation)
    if annotation is not inspect.Parameter.empty:
        # Order matters: bool before int, because "bool" contains no "int".
        for needle, kind in (("bool", "bool"), ("float", "float"), ("int", "int"),
                             ("str", "str")):
            if needle in text.lower() and "sequence" not in text.lower() \
                    and "list" not in text.lower():
                return kind
        if any(n in text for n in ("Sequence", "List", "Iterable", "Tuple", "Dict")):
            return "seq"
    if default is not inspect.Parameter.empty and default is not None:
        return _KIND_BY_TYPE.get(type(default), "any")
    return "any"


def _describe(func: Callable[..., Any]) -> Optional[Tuple[Param, ...]]:
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):  # pragma: no cover - builtins
        return None
    params: List[Param] = []
    for p in sig.parameters.values():
        if p.name in ("self", "cls"):
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        params.append(
            Param(
                name=p.name,
                kind=_classify(p.annotation, p.default),
                default=p.default,
                required=p.default is inspect.Parameter.empty,
                annotation="" if p.annotation is inspect.Parameter.empty
                else getattr(p.annotation, "__name__", str(p.annotation)),
            )
        )
    return tuple(params)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _public_callables(module) -> List[Tuple[str, Callable[..., Any]]]:
    out = []
    for name, obj in vars(module).items():
        if name.startswith("_") or name in _NEEDS_OBJECTS:
            continue
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        if getattr(obj, "__module__", "") != module.__name__:
            continue  # re-export, documented where it is defined
        if dataclasses.is_dataclass(obj) and not inspect.isfunction(obj):
            continue  # value objects are results, not demos
        out.append((name, obj))
    return sorted(out)


def discover(module_name: str) -> List[Demo]:
    """Build the demo list for one `llm_platform` module."""
    try:
        module = importlib.import_module(f"llm_platform.{module_name}")
    except ImportError as exc:  # pragma: no cover - surfaced in the UI
        raise PlatformUnavailable(
            f"cannot import llm_platform.{module_name} — is {PROJECT_DIR} present?"
        ) from exc

    demos: List[Demo] = []
    for name, obj in _public_callables(module):
        params = _describe(obj)
        if params is None:
            continue
        demos.append(
            Demo(
                module=module_name,
                name=name,
                doc=inspect.cleandoc(obj.__doc__ or "").split("\n\n")[0],
                params=params,
                func=obj,
            )
        )
    return demos


def catalog(groups: Sequence[Tuple[str, Tuple[str, ...]]] = tuple(GROUPS)
            ) -> Dict[str, Dict[str, Demo]]:
    """All demos, grouped by book part and keyed by callable name."""
    out: Dict[str, Dict[str, Demo]] = {}
    for label, modules in groups:
        bucket: Dict[str, Demo] = {}
        for mod in modules:
            for demo in discover(mod):
                bucket[demo.name] = demo
        if bucket:
            out[label] = bucket
    return out


def covered_modules() -> List[str]:
    return sorted({m for _, mods in GROUPS for m in mods})


# --------------------------------------------------------------------------- #
# Invocation
# --------------------------------------------------------------------------- #
def coerce(param: Param, raw: Any) -> Any:
    """Turn a UI string into the type the parameter expects."""
    if raw is None or raw == "":
        if param.has_default:
            return param.default
        raise ValueError(f"{param.name} is required")
    if param.kind == "int":
        return int(float(raw)) if isinstance(raw, str) else int(raw)
    if param.kind == "float":
        return float(raw)
    if param.kind == "bool":
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)
    if param.kind == "seq":
        if isinstance(raw, str):
            parts = [x.strip() for x in raw.replace("\n", ",").split(",") if x.strip()]
            try:
                return [float(x) if "." in x else int(x) for x in parts]
            except ValueError:
                return parts
        return list(raw)
    return raw


def run_demo(demo: Demo, **kwargs: Any) -> Any:
    """Call a demo with UI-supplied values, coercing each to its declared type."""
    call: Dict[str, Any] = {}
    for p in demo.params:
        if p.name in kwargs:
            call[p.name] = coerce(p, kwargs[p.name])
        elif p.has_default:
            call[p.name] = p.default
        else:
            raise ValueError(f"missing required argument: {p.name}")
    return demo.func(**call)


def as_display(value: Any) -> Any:
    """Render dataclasses/tuples as dicts so Streamlit can show them as JSON."""
    if dataclasses.is_dataclass(value) and not inspect.isclass(value):
        return {k: as_display(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {k: as_display(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [as_display(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return repr(value)
