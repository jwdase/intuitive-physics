"""Tests for src.preprocess.ray_trace (Möller–Trumbore style ray/triangle hit test).

Each test fires a ray at a triangle and asserts whether it should hit.
The expected values describe the *correct* geometric behaviour of a ray
(half-line) against a triangle, so this file doubles as a spec.

Run directly (no pytest needed):
    .venv/bin/python tests/test_ray_trace.py
Or, if pytest is installed:
    pytest tests/test_ray_trace.py
"""

import os
import sys

import numpy as np

# Make `src` importable no matter what the current working directory is.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import ray_trace


# A standard unit triangle lying in the z = 0 plane.
#   V0 = origin, E_1 -> +x, E_2 -> +y
# For a point (x, y, 0) the barycentrics are simply u = x, v = y, so the
# triangle interior is { x >= 0, y >= 0, x + y <= 1 }.
T0 = np.array([0.0, 0.0, 0.0])
T1 = np.array([1.0, 0.0, 0.0])
T2 = np.array([0.0, 1.0, 0.0])

DOWN = np.array([0.0, 0.0, 1.0])  # shoots from z=-1 up toward the z=0 plane


def test_hit_through_centroid():
    # Straight shot through a clearly interior point (u=v=0.25, u+v=0.5).
    O = np.array([0.25, 0.25, -1.0])
    assert ray_trace(O, DOWN, T0, T1, T2) is True


def test_hit_on_edge_is_inclusive():
    # Lands on the V0->V1 edge (u=0.5, v=0): boundary counts as a hit.
    O = np.array([0.5, 0.0, -1.0])
    assert ray_trace(O, DOWN, T0, T1, T2) is True


def test_miss_outside_negative_barycentric():
    # u = -0.5 < 0, so the point is off the triangle entirely.
    O = np.array([-0.5, 0.25, -1.0])
    assert ray_trace(O, DOWN, T0, T1, T2) is False


def test_miss_beyond_hypotenuse():
    # u=0.8, v=0.8 -> inside the E_1/E_2 parallelogram but u+v=1.6 > 1,
    # so it is OUTSIDE the triangle. Guards against the parallelogram bug.
    O = np.array([0.8, 0.8, -1.0])
    assert ray_trace(O, DOWN, T0, T1, T2) is False


def test_miss_triangle_behind_origin():
    # Same geometry, but the ray points away from the triangle (t < 0).
    # A ray is a half-line, so this must be a miss.
    O = np.array([0.25, 0.25, 1.0])
    assert ray_trace(O, DOWN, T0, T1, T2) is False


def test_parallel_ray_does_not_crash():
    # Ray travels within the z=-1 plane, parallel to the triangle. The
    # linear system is singular; the function must report a miss, not raise.
    O = np.array([0.25, 0.25, -1.0])
    D = np.array([1.0, 0.0, 0.0])
    assert ray_trace(O, D, T0, T1, T2) is False


def test_original_example_hits():
    # The example baked into preprocess.py's __main__: ray hits (1,0,0),
    # which lies on the bottom edge of the triangle.
    O = np.array([0.0, 0.0, 0.0])
    D = np.array([1.0, 0.0, 0.0])
    V0 = np.array([1.0, 1.0, 0.0])
    V1 = np.array([1.0, -1.0, 0.0])
    V2 = np.array([1.0, 0.0, 1.0])
    assert ray_trace(O, D, V0, V1, V2) is True


if __name__ == "__main__":
    tests = [obj for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    passed = failed = 0
    for t in tests:
        try:
            t()
        except AssertionError:
            failed += 1
            print(f"FAIL  {t.__name__}  (wrong result)")
        except Exception as e:  # e.g. LinAlgError on the parallel-ray case
            failed += 1
            print(f"ERROR {t.__name__}  ({type(e).__name__}: {e})")
        else:
            passed += 1
            print(f"PASS  {t.__name__}")
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    sys.exit(1 if failed else 0)
