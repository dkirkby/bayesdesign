# CHANGELOG for bayesdesign

## Introduction

This is the log of changes to the [bayesdesign package](https://github.com/dkirkby/bayesdesign).

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2024-12-04

### Added

- Support for running notebooks via google colab (issue #6)
- Support for subgrids on the designs for memory efficiency
- Notebook demonstrating subgrid capabilities
- Memory constraint argument for ExperimentDesigner
- Constraint can take in kwargs for variable axis names

### Fixed

- Handle cases where the posterior is un-normalizable for some (parameter,design) combinations (issue #5)

## Changed

- ExperimentDesigner now takes in unnormalized likelihood as a function
- Example notebooks are now compatible with API changes

## [0.4.0] - 2024-09-24

### Added

- TopHat and CosineBump helper functions for building priors
- plot.cornerPlot to visualize multi-dimensional arrays defined on a Grid
- MultiParameter example notebook
- Location finding example notebook

### Changed

- Add optional arg to set missing value in Grid.expand()
- Add verbose option to Grid.sum()
- Allow constraint in Grid.sum()

### Fixed

- Handling of p=0 in p*log2(p) for entropy calculations

## [0.3.0] - 2024-09-05

### Added

- Instructions on starting work on next version added to CONTRIBUTING.md
- Notebook to demonstrate grid constraints (see [issue#1](https://github.com/dkirkby/bayesdesign/issues/1))
- Section on optimal second measurement in sine wave example notebook
- grid.PermutationInvariant helper function to calculate constraint weights
- expand() and getmax() methods to grid.Grid

### Changed

- design.ExperimentDesigner.calculateEIG now returns a dict of best design values

### Fixed

- Ignore examples when checking MANIFEST.in (was breaking Test action)
- Typo in README example

## [0.2.0] - 2024-08-29

### Added

- Examples folder with a simple sine-wave example
- CONTRIBUTING.md with guidelines for contributors and developer tasks
- CHANGELOG.md (this file)
- requirements.txt listing packages required to use this one (only numpy so far)

### Changed

- Add simple example to README.md

### Fixed

- Debug github Test action

## [0.1.0] - 2024-08-28

Initial commit
