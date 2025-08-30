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
   - **Required reviewers**: 
     - For solo maintainers: Leave empty (you can still approve as repository owner)
     - For teams: Add trusted co-maintainers who should approve releases
   - **Wait timer**: 5-10 minutes (gives you time to cancel accidental releases)
   - **Deployment branches**: See "Branch Protection Setup" section below

### Step 3: Environment Variables (Optional)
If needed, add environment-specific variables:
- `PYPI_URL`: https://pypi.org/p/bayesdesign

## Protection Benefits

- **Release Control**: Optional approval step (repository owners can always approve)
- **Accidental Release Prevention**: Wait timer provides cancellation window
- **Branch Protection**: Only protected branches can deploy
- **Audit Trail**: Complete history of deployments
- **Emergency Stop**: Ability to pause deployments

## Solo Maintainer vs Team Setup

### Solo Maintainer (Recommended for bayesdesign)
- **Required reviewers**: Leave empty
- **Wait timer**: 5-10 minutes
- **Benefit**: You can approve your own releases, but have time to cancel accidents

### Team Maintenance
- **Required reviewers**: Add trusted co-maintainers
- **Wait timer**: 0 minutes (approval provides the gate)
- **Benefit**: Ensures multiple people review critical releases

## Branch Protection Setup

You have two options for deployment branches:

### Option 1: Protected Branches Only (Recommended)

Set up basic branch protection for your main branch:

1. Go to repository Settings → Branches
2. Click "Add rule" 
3. Branch name pattern: `main`
4. Configure protection rules:

**For Solo Maintainers (bayesdesign)**:
```
☐ Require a pull request before merging (optional for solo)
☑️ Require status checks to pass before merging
  └── Select: Test / build (your existing test workflow)
☐ Require branches to be up to date before merging (optional)
☑️ Restrict pushes that create files larger than 100MB
☐ Include administrators (optional - applies rules to you too)
```

5. In Environment settings, select "Protected branches only"

**Benefits**:
- Ensures tests pass before any release
- Prevents accidental branch deletion
- Creates audit trail of changes
- Still allows direct pushes for quick fixes (if admin override enabled)

### Option 2: All Branches (Simpler)

In Environment settings, select "All branches"

**Benefits**:
- Simpler setup, no branch protection needed
- Can release from any branch (including feature branches)
- More flexible for experimentation

**Drawbacks**:
- No guarantee that tests pass before release
- Less protection against accidents

## Recommendation

For scientific packages like bayesdesign, **Option 1 with basic protection** is recommended:
- Ensures release quality through automated testing
- Prevents common accidents (force push, branch deletion)  
- Maintains flexibility for solo development

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