from setuptools import setup
from setuptools import find_packages

version_py = "HiCPlot/_version.py"
exec(open(version_py).read())

setup(
    name="hicplot", # Replace with your own username
    version=__version__,
    author="Benxia Hu",
    author_email="hubenxia@gmail.com",
    description="plot heatmaps from Hi-C matrix and tracks from bigwig files",
    long_description="plot heatmaps from Hi-C matrix and tracks from bigwig files",
    url="https://pypi.org/project/HiCPlot/",
    entry_points = {
        "console_scripts": ['TriHeatmap = HiCPlot.TriHeatmap:main',
                            'SquHeatmap = HiCPlot.SquHeatmap:main',
                            'NGStrack= HiCPlot.NGStrack:main',
                            'DiffSquHeatmap= HiCPlot.DiffSquHeatmap:main',
                            'upper_lower_triangle_heatmap= HiCPlot.upper_lower_triangle_heatmap:main']
        },
    python_requires = '>=3.12',
    packages = ['HiCPlot'],
    install_requires = [
        'numpy',
        'pandas',
        'argparse',
        'matplotlib',
        'pyBigWig',
        'pyranges',
        'cooler',
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    zip_safe = False,
  )
