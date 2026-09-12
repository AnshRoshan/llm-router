import pathlib
import sys

# The delegation regression tests drive the router through the demo's fake
# transport, so `examples/` must be importable as `demo`.
_EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))
