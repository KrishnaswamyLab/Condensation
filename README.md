Diffusion Condenastion
================

[![Latest PyPi version](https://img.shields.io/pypi/v/multiscale_phate.svg)](https://pypi.org/project/multiscale_phate/)
[![Travis CI Build](https://api.travis-ci.com/KrishnaswamyLab/Diffusion_Condensation.svg?branch=master)](https://travis-ci.com/KrishnaswamyLab/Diffusion_Condensation/)
[![Coverage Status](https://coveralls.io/repos/github/KrishnaswamyLab/Diffusion_Condensation/badge.svg?branch=master)](https://coveralls.io/github/KrishnaswamyLab/Diffusion_Condensation?branch=master)
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)
[![GitHub stars](https://img.shields.io/github/stars/KrishnaswamyLab/Diffusion_Condensation.svg?style=social&label=Stars)](https://github.com/KrishnaswamyLab/Diffusion_Condensation/)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Diffusion condensation is a python package for topological analysis of high dimensional single cell data.

Cells naturally exist in a hierarchy of cell types, subtypes and states.  Diffusion condensation learns this hierarchy by calculating a diffusion homology from the single cell data and then identifying cell types through persistence-based analysis. While existing clustering approaches present only a few levels of the cellular hierarchy often missing information, such as rare cell types, diffusion condensation sweeps through different granularities of data to identify natural groupings of cells at each level. In this package, users will be able to calculate the diffusion homology of a single cell dataset, visualize the homology and sweep through levels of granularity to find both abundant and rare cellular types and states.

Installation
------------

Diffusion Condensation is available on `pip`. Install by running the following in a terminal:

```
pip install --user git+https://github.com/KrishnaswamyLab/Diffusion_Condensation
```

Quick Start
-----------

```
import diffusion_condensation
dc_op = diffusion_condensation.Diffusion_Condensation()
data_hierarchy = dc_op.fit_transform(data)
```

Guided Tutorial
-----------

For more details on using Diffusion Condensation, see our [guided tutorial](tutorial/10X_pbmc.ipynb) using 10X's public PBMC4k dataset.
