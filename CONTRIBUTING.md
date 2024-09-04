# How to Contribute to this Package

[![PyPI package](https://img.shields.io/badge/pip%20install-bayesdesign-brightgreen)](https://pypi.org/project/bayesdesign/) [![GitHub Release](https://img.shields.io/github/v/release/dkirkby/bayesdesign?color=green)](https://github.com/dkirkby/bayesdesign/releases) [![Actions Status](https://github.com/dkirkby/bayesdesign/workflows/Test/badge.svg)](https://github.com/dkirkby/bayesdesign/actions) [![License](https://img.shields.io/github/license/dkirkby/bayesdesign)](https://github.com/dkirkby/bayesdesign/blob/main/LICENSE)

## Report a problem or suggest a feature

If there is not already a relevant [open issue](https://github.com/dkirkby/bayesdesign/issues), please [open a new one](https://github.com/dkirkby/bayesdesign/issues/new). Be sure to include a title and clear description and as much relevant information as possible. When reporting a problem, a code sample or executable test case demonstrating the problem is very helpful.

## Developer Tasks

### Development Environment

Required development packages are `tox`, `pytest` and `black`.

### Configure VScode

Run `black` code formatter on save.

### Run unit tests locally

Use the flask sidebar icon in VScode or run `tox -e py` from the command line in the top-level directory.

### Change the matrix of test configurations

Edit `python-version` in `.github/workflows/test.yml`

If the range of supported python versions has changed, edit `classifiers` and `python_requires` in `setup.py` accordingly.

(What is the role of `envlist` in `tox.ini`??)

### Add a dependency

Update `requirements.txt`

(What about `install_requires` in `setup.py`??)

### Release a new version

Update the version string in `src/bed/__init__.py` (by removing "dev") and the top of the `CHANGELOG.md` file, then commit:
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
Create a new release on github [here](https://github.com/dkirkby/bayesdesign/releases/new):
 - Select the newly created tag from the "Choose a tag" drop-down menu.
 - Leave the "Release title" blank.
 - Click "Generate release notes".
 - Click "Publish release".
 - Check that the Release github action has uploaded this version to pypi [here](https://pypi.org/project/bayesdesign).

## Start work on the next version

Bump the version string in `src/bed/__init__.py` and append "dev".

Add a new `## [Unreleased]` section to the top of the `CHANGELOG.md` file.

Push to github:
```
git add src/bed/__init__.py CHANGELOG.md
git commit -m 'Start work on next version'
git push
```
