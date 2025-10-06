"""Run examples to make sure they don't go stale"""

import os
import pathlib
import runpy

import pytest

os.environ["MPLBACKEND"] = "Agg"

EXAMPLES_DIR = pathlib.Path(__file__).parent / "../examples"
EXAMPLES = [
    EXAMPLES_DIR / x
    for x in os.listdir(EXAMPLES_DIR)
    if (os.path.isfile(EXAMPLES_DIR / x) and x.endswith(".py"))
]


@pytest.mark.parametrize("fp", EXAMPLES)
def test_example(fp: pathlib.Path):
    try:
        runpy.run_path(str(fp), run_name="__main__")
    except:
        print(f"Failed to run example {fp}")
        raise
