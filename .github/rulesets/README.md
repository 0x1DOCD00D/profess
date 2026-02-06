# Branch Protection Rulesets

This directory contains GitHub repository rulesets that define branch protection rules.

## Master Branch Protection

**File:** `master-branch-protection.json`

This ruleset protects the `master` branch with the following rules:

### Protection Rules

1. **Pull Request Reviews Required**
   - At least 1 approving review required before merging
   - Stale reviews are dismissed when new commits are pushed
   - Review threads must be resolved before merging

2. **Required Status Checks**
   - `build` job must pass
   - `coverage` job must pass
   - Branch must be up to date before merging

3. **Branch Deletion Prevention**
   - The master branch cannot be deleted

4. **Force Push Prevention**
   - Force pushes are not allowed

5. **Linear History Required**
   - Merge commits and squash merges are required to maintain linear history

### Applying These Rules

GitHub will automatically apply these rulesets when you push this configuration to the repository. The rulesets are stored in the `.github/rulesets/` directory and are enforced by GitHub.

To manually apply or view these rules in the GitHub UI:
1. Go to repository Settings
2. Navigate to Rules → Rulesets
3. The "Master Branch Protection" ruleset should be visible and active
