# How to Contribute to this Package

[![PyPI package](https://img.shields.io/badge/pip%20install-bayesdesign-brightgreen)](https://pypi.org/project/bayesdesign/) [![version number](https://img.shields.io/pypi/v/example-pypi-package?color=green&label=version)](https://github.com/dkirkby/bayesdesign/releases) [![Actions Status](https://github.com/dkirkby/bayesdesign/workflows/Test/badge.svg)](https://github.com/dkirkby/bayesdesign/actions) [![License](https://img.shields.io/github/license/dkirkby/bayesdesign)](https://github.com/dkirkby/bayesdesign/blob/main/LICENSE)

## Report a problem or suggest a feature

If there is not already a relevant [open issue](https://github.com/dkirkby/bayesdesign/issues), please [open a new one](https://github.com/dkirkby/bayesdesign/issues/new). Be sure to include a title and clear description and as much relevant information as possible. When reporting a problem, a code sample or executable test case demonstrating the problem is very helpful.

## Developer Tasks

### Configure VScode

### Run unit tests locally

### Change the matrix of test configurations

### Release a new version

Update the version string in `src/bed/__init__.py` and the top of the `CHANGELOG.md` file, then commit:
```
git add src/bed/__init__.py CHANGELOG.md
git commit -m 'Prepare for release'
git push
```
Tag the version in github, adding "v" in front of the numerical version, e.g. for version 0.1.0:
```
git tag v0.1.0
git push --tags
```
Create a new release on github [here](https://github.com/dkirkby/bayesdesign/releases/new). Select the newly created tag from the "Choose a tag" drop-down menu and leave the "Release title" blank.
