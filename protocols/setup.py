from setuptools import setup, find_packages


requirements = [
    "mber>=1.0.0",
    "pyyaml",
]

setup(
    name="mber-protocols",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=requirements,
    author="Erik Swanson",
    author_email="erik@manifold.bio",
    description="Optional protocols for mber.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/manifoldbio/mber",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "mber-vhh=mber_protocols.stable.VHH_binder_design.cli:main",
        ],
    },
)