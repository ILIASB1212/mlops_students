from setuptools import find_packages, setup


with open("requirements.txt") as f:
    load_requirements = f.read().splitlines()


setup(
    name="mlops_reservation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=load_requirements)