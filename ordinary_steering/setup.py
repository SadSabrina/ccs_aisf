from setuptools import find_packages, setup

setup(
    name="ordinary_steering",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "tabulate",
        "tqdm",
    ],
)
