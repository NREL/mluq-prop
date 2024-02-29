import os
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "requirements.txt")) as f:
    install_requires = f.readlines()

setup(
    name='mluqprop',
    version='0.0.1',
    description="UQ for ML closure models",
    url="https://github.com/NREL/S-mluq-prop",
    license="BSD 3-Clause",
    package_dir={"mluqprop": "mluqprop", "mluqprop_applications": "mluqprop_applications"},
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: BSD 3 License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires='>=3.10',
    install_requires=install_requires,
)

