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
   - Only fast-forward or rebase merges are allowed to maintain linear history
   - This prevents merge commits from being created

### Applying These Rules

This JSON file serves as a configuration template for the branch protection rules. To apply these rules to the repository, you need to create a ruleset using one of the following methods:

**Option 1: Using the GitHub UI**
1. Go to repository Settings
2. Navigate to Rules → Rulesets
3. Click "New ruleset" → "New branch ruleset"
4. Use the settings from `master-branch-protection.json` to configure the ruleset
5. Set enforcement to "Active"

**Option 2: Using the GitHub API**
Use the GitHub REST API to create a ruleset by sending a POST request to `/repos/{owner}/{repo}/rulesets` with the contents of this JSON file.

**Note:** The JSON files in this directory do not automatically activate protection rules. They serve as documentation and configuration templates that need to be manually applied through the GitHub UI or API.
