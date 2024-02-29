import os
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "requirements.txt")) as f:
    install_requires = f.readlines()

with open(os.path.join(here, "mluqprop", "version.py"), encoding="utf-8") as f:
    version = f.read()
version = version.split("=")[-1].strip().strip('"').strip("'")

setup(
    name='mluqprop',
    version=version,
    description="UQ for ML closure models",
    url="https://github.com/NREL/mluq-prop",
    license="BSD 3-Clause",
    package_dir={"mluqprop": "mluqprop", "applications": "applications"},
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires='>=3.10',
    install_requires=install_requires,
)

