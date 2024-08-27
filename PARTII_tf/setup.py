from setuptools import setup, find_packages


setup(
    name="FlowUQ",
    version="v0.1.0-beta", 
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tensorflow",
        "tensorflow_probability"],
    author="Theo Hanon",
    author_email="theo.hanon@student.uclouvain.be",
    description="Uncertainty quantification for Deep Ensembles",
)
    