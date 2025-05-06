from setuptools import setup
from setuptools import find_packages

version_py = "HiCPlot/_version.py"
exec(open(version_py).read())

setup(
    name="hicplot",
    version=__version__,
    author="Benxia Hu",
    author_email="hubenxia@gmail.com",
    description="Plot Hi-C heatmaps and genomic tracks.",
    long_description="Plot heatmaps from Hi-C contact matrices and tracks from bigWig files.",
    url="https://github.com/BenxiaHu/HiCPlot",
    packages=find_packages(),             # auto-discover HiCPlot package
    python_requires=">=3.12",              # relax unless 3.12 really required
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "pyBigWig",
        "pyranges",
        "cooler",
    ],
    entry_points={
        "console_scripts": [
            "HiCPlot = HiCPlot.Cli:main",  # dotted path must match the real file
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
