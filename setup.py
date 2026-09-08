# Metadata lives in pyproject.toml. requires-python is set here so that the Poetry lock file
# (which hashes [project].requires-python) stays valid.
from setuptools import setup

setup(python_requires=">=3.10")
