from setuptools import setup, find_packages

setup(
    name="forest_fire_risk",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
