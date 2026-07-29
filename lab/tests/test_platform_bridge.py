"""Tests for the lab -> llm_platform bridge.

These assert that every chapter's code is reachable and callable from the UI,
which is the property the bridge exists to guarantee.
"""
from __future__ import annotations

import inspect

import pytest

from llm_lab import platform_bridge as bridge


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def test_catalog_covers_every_declared_module():
    cat = bridge.catalog()
    assert cat, "catalog is empty — is project/ importable?"
    found = {demo.module for group in cat.values() for demo in group.values()}
    missing = set(bridge.covered_modules()) - found
    assert not missing, f"modules declared in GROUPS but contributing no demos: {missing}"


def test_catalog_exposes_a_substantial_surface():
    total = sum(len(group) for group in bridge.catalog().values())
    assert total >= 100, f"only {total} callables exposed; expected the full platform"


def test_every_group_is_non_empty():
    for label, group in bridge.catalog().items():
        assert group, f"group {label!r} has no demos"


def test_demos_carry_a_resolvable_source():
    for group in bridge.catalog().values():
        for demo in group.values():
            src = demo.source()
            assert src.strip(), f"{demo.qualname} has no source"
            assert "source unavailable" not in src, demo.qualname


def test_params_are_introspected_not_guessed():
    demo = bridge.catalog()["Ch 1 · Capacity & Cost Estimation"]["kv_cache_memory"]
    names = [p.name for p in demo.params]
    sig = list(inspect.signature(demo.func).parameters)
    assert names == sig


# --------------------------------------------------------------------------- #
# Coercion
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kind, raw, expected",
    [
        ("int", "32", 32),
        ("int", "32.0", 32),
        ("float", "0.75", 0.75),
        ("bool", "true", True),
        ("bool", "no", False),
        ("seq", "1, 2, 3", [1, 2, 3]),
        ("seq", "a, b", ["a", "b"]),
        ("str", "hello", "hello"),
    ],
)
def test_coerce_maps_ui_strings_to_types(kind, raw, expected):
    p = bridge.Param(name="x", kind=kind, required=True)
    assert bridge.coerce(p, raw) == expected


def test_coerce_falls_back_to_default_when_blank():
    p = bridge.Param(name="x", kind="int", default=7, required=False)
    assert bridge.coerce(p, "") == 7


def test_coerce_rejects_blank_required_value():
    p = bridge.Param(name="x", kind="int", required=True)
    with pytest.raises(ValueError, match="required"):
        bridge.coerce(p, "")


# --------------------------------------------------------------------------- #
# Invocation — the numbers here are the ones the book quotes.
# --------------------------------------------------------------------------- #
def test_kv_cache_matches_chapter_1_figures():
    demo = bridge.catalog()["Ch 1 · Capacity & Cost Estimation"]["kv_cache_memory"]
    out = bridge.as_display(
        bridge.run_demo(demo, num_layers="32", hidden_size="4096", seq_len="4096",
                        num_kv_heads="8", head_dim="128")
    )
    assert out["bytes_per_token"] == 131072
    assert out["per_request_gb"] == 0.5
    assert out["max_concurrent_requests"] == 128


def test_paged_kv_blocks_reports_fragmentation():
    demo = bridge.catalog()["Ch 9-11 · Inference & Serving"]["paged_kv_blocks"]
    out = bridge.as_display(bridge.run_demo(demo, seq_len="1000", block_size="16"))
    assert out["blocks_needed"] == 63
    assert out["tokens_wasted"] == 8


def test_system_availability_uses_k_of_n():
    demo = bridge.catalog()["Ch 30-32 · Scale & Reliability"]["system_availability"]
    out = bridge.run_demo(demo, replica_availability="0.99", n="3", required="2")
    assert 0.999 < out < 1.0


def test_run_demo_requires_missing_arguments():
    demo = bridge.catalog()["Ch 1 · Capacity & Cost Estimation"]["kv_cache_memory"]
    with pytest.raises(ValueError, match="missing required argument"):
        bridge.run_demo(demo)


def test_every_form_runnable_demo_has_typed_params():
    """A demo the UI offers must not need hand-built objects."""
    for group in bridge.catalog().values():
        for demo in group.values():
            if not demo.runnable:
                continue
            for p in demo.params:
                assert p.kind != "any" or not p.required, \
                    f"{demo.qualname} claims to be runnable but {p.name} is untyped"


# --------------------------------------------------------------------------- #
# Display
# --------------------------------------------------------------------------- #
def test_as_display_flattens_dataclasses():
    demo = bridge.catalog()["Ch 1 · Capacity & Cost Estimation"]["kv_cache_memory"]
    raw = bridge.run_demo(demo, num_layers="32", hidden_size="4096", seq_len="4096",
                          num_kv_heads="8", head_dim="128")
    assert not isinstance(raw, dict)
    assert isinstance(bridge.as_display(raw), dict)


def test_as_display_is_json_safe():
    import json

    demo = bridge.catalog()["Ch 9-11 · Inference & Serving"]["paged_kv_blocks"]
    out = bridge.as_display(bridge.run_demo(demo, seq_len="1000"))
    json.dumps(out)  # must not raise
