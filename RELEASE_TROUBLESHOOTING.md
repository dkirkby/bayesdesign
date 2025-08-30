# Release Troubleshooting Guide

This guide helps diagnose and resolve common issues with the automated release workflow.

## Common Issues and Solutions

### 1. Version Consistency Errors

**Error**: `Version mismatch: package=X.Y.Zdev, tag=X.Y.Z`

**Solution**:
```bash
# Fix the package version
vim src/bed/__init__.py
# Remove "dev" from __version__ = "X.Y.Zdev"

# Commit and recreate tag
git add src/bed/__init__.py
git commit -m "Fix version for release"
git tag -d vX.Y.Z  # Delete old tag
git push origin :refs/tags/vX.Y.Z  # Delete remote tag
git tag vX.Y.Z
git push --tags
```

### 2. Missing Changelog Entry

**Error**: `No changelog entry found for version X.Y.Z`

**Solution**:
```bash
# Add changelog entry
vim CHANGELOG.md
# Add: ## [X.Y.Z] - YYYY-MM-DD

git add CHANGELOG.md
git commit -m "Add changelog entry for vX.Y.Z"
git tag -f vX.Y.Z  # Force update tag
git push --tags --force
```

### 3. Notebook Execution Failures

**Error**: `Notebook validation failed: examples/SomeNotebook.ipynb`

**Solutions**:
- **Timeout issues**: Increase timeout in workflow (currently 300s)
- **Missing dependencies**: Check if notebook requires packages not in requirements.txt
- **Data dependencies**: Ensure notebook doesn't require external data files
- **Skip problematic notebook**: Add to skip list in workflow

### 4. Test Suite Failures

**Error**: `tox -e py` fails during pre-release validation

**Solution**:
```bash
# Run tests locally to debug
tox -e py
pytest tests/ -v

# Fix issues and commit
git add .
git commit -m "Fix test failures"
git tag -f vX.Y.Z
git push --tags --force
```

### 5. PyPI Publishing Errors

**Error**: `OIDC token request failed`

**Solutions**:
1. Verify PyPI trusted publishing configuration
2. Check GitHub environment settings
3. Ensure workflow permissions include `id-token: write`

**Error**: `File already exists on PyPI`

**Solutions**:
1. Increment version number
2. Delete and recreate release with new version

### 6. Build Artifacts Issues

**Error**: Build produces unexpected artifacts

**Solution**:
```bash
# Test build locally
python -m build
ls dist/
python -m zipfile -l dist/*.whl
```

## Emergency Procedures

### Quick Rollback
If a release fails after PyPI publication:
1. Mark the release as pre-release on GitHub
2. Consider yanking the version from PyPI if critical issues exist
3. Prepare hotfix version immediately

### Manual Release Override
If automation completely fails:
```bash
# Manual release process (emergency only)
git checkout vX.Y.Z
python -m build
python -m twine upload dist/* --repository pypi
```

## Workflow Monitoring

### Key Logs to Check
1. **Pre-release validation**: Version checks, test results, notebook execution
2. **Build job**: Artifact creation and verification
3. **Publish job**: OIDC authentication, PyPI upload

### Status Indicators
- ✅ All jobs green: Release successful
- ⚠️ Validation fails: Pre-release issues, safe to fix and retry
- ❌ Publish fails: May need manual intervention

## Prevention Best Practices

1. **Always test releases**: Use pre-release tags for testing
2. **Validate locally**: Run `tox -e py` and notebook tests before tagging
3. **Check versions**: Verify version consistency before creating release
4. **Review changelog**: Ensure documentation is complete
5. **Monitor workflow**: Watch the release process to catch issues early

## Getting Help

If issues persist:
1. Check GitHub Actions logs for detailed error messages
2. Review PyPI project settings for trusted publishing
3. Verify environment protection rules
4. Consult Python packaging documentation