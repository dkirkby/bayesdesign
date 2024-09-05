# CHANGELOG for bayesdesign

## Introduction

This is the log of changes to the [bayesdesign package](https://github.com/dkirkby/bayesdesign).

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
