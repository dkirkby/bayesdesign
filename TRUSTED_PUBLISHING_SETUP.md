# PyPI Trusted Publishing Setup for bayesdesign

This document outlines the steps to configure PyPI trusted publishing for automated releases.

## Overview

Trusted publishing eliminates the need for API tokens by using OpenID Connect (OIDC) to authenticate GitHub Actions with PyPI. This is more secure and follows current best practices.

## Setup Steps

### 1. PyPI Configuration

1. Go to https://pypi.org/manage/account/publishing/
2. Add a new trusted publisher with these settings:
   - **PyPI Project Name**: `bayesdesign`
   - **Owner**: `dkirkby`
   - **Repository name**: `bayesdesign`
   - **Workflow filename**: `release.yml`
   - **Environment name**: `pypi` (optional but recommended)

### 2. GitHub Environment Setup

Create a protected environment in GitHub repository settings:

1. Go to repository Settings → Environments
2. Create new environment named `pypi`
3. Configure protection rules:
   - **Required reviewers**: 
     - For solo maintainers: Leave empty (repository owners can always approve)
     - For teams: Add trusted co-maintainers
   - **Wait timer**: 5-10 minutes (optional - gives time to cancel accidental releases)  
   - **Deployment branches**: See branch protection options in `GITHUB_ENVIRONMENT_SETUP.md`

### 3. Workflow Configuration

The new workflow will use:
```yaml
environment:
  name: pypi
  url: https://pypi.org/p/bayesdesign
permissions:
  id-token: write
```

## Benefits

- **Enhanced Security**: No API tokens to manage or rotate
- **Audit Trail**: Clear deployment history with approvals
- **Reduced Risk**: Tokens cannot be leaked or misused
- **Best Practice**: Follows Python Packaging Authority recommendations

## Migration Notes

After successful setup and testing:
1. Remove old `PYPI_API_TOKEN` secret
2. Update release documentation
3. Test with pre-release before production use

## References

- [PyPI Trusted Publishing Guide](https://docs.pypi.org/trusted-publishers/)
- [GitHub OIDC Documentation](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect)
- [Python Packaging Best Practices](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)