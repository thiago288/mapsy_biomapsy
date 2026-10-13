"""Run the fixed implementation regression suite with the selected Python."""
from pathlib import Path
import runpy
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "tests/test_fixed_regressions.py"), run_name="__main__")
