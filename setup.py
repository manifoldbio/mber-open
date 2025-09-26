from setuptools import setup, find_packages

# Read requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="mber",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=requirements,
    author="Erik Swanson",
    author_email="erik@manifold.bio",
    description="Manifold Binder Engineering and Refinement. A package for format-specific protein binder design.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/manifoldbio/mber",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)