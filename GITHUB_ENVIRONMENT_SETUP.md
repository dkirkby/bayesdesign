# GitHub Environment Setup Guide

This guide explains how to set up GitHub environments for secure releases.

## Creating the PyPI Environment

### Step 1: Navigate to Environment Settings
1. Go to repository: https://github.com/dkirkby/bayesdesign
2. Click **Settings** tab
3. Select **Environments** from left sidebar
4. Click **New environment**

### Step 2: Configure Environment
1. **Environment name**: `pypi`
2. **Environment protection rules**:
   - ☑️ **Required reviewers**: Add repository maintainers
   - ☑️ **Wait timer**: 0 minutes (or desired delay)
   - ☑️ **Deployment branches**: Select "Protected branches only"

### Step 3: Environment Variables (Optional)
If needed, add environment-specific variables:
- `PYPI_URL`: https://pypi.org/p/bayesdesign

## Protection Benefits

- **Manual Approval**: Releases require explicit approval
- **Branch Protection**: Only protected branches can deploy
- **Audit Trail**: Complete history of deployments
- **Emergency Stop**: Ability to pause deployments

## Testing the Setup

1. Create a test release workflow run
2. Verify approval process works
3. Check deployment logs for OIDC authentication
4. Confirm artifacts are published correctly

## Troubleshooting

### Common Issues
- **OIDC Token Error**: Verify PyPI trusted publisher configuration matches exactly
- **Permission Denied**: Ensure `id-token: write` permission is set
- **Environment Not Found**: Check environment name matches workflow exactly

### Verification Steps
1. Check GitHub Actions logs for OIDC token exchange
2. Verify PyPI shows trusted publisher in project settings
3. Confirm environment shows in deployment history

## Security Notes

- Environment protection rules apply to all deployments
- OIDC tokens are short-lived and scoped to specific runs
- No long-term secrets stored in repository
- Deployment approvals create audit trail