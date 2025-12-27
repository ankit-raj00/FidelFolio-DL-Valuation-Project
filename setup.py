from setuptools import setup, find_packages

setup(
    name="fidelfolio_ml",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "tensorflow",
        "pyyaml",
    ],
    python_requires=">=3.8",
)
