"""SimVascular's SuperEstimator package."""
import os
import subprocess

NAME = "svSuperEstimator"
VERSION = (
    subprocess.check_output(
        ["git", "describe", "--tags", "--always"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    .strip()
    .decode()
)
COPYRIGHT = (
    "Stanford University, The Regents of the University of California, "
    "and others."
)
