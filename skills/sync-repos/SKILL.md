# Sync GitHub → GitLab

Sync the koi repository from GitHub to GitLab, overwriting everything in the GitLab copy except for `.koi/` and `src/koi/bundled_skills/`.

## When to Use

- When asked to sync koi repos between GitHub and GitLab
- After pushing changes to GitHub and needing them on GitLab

## Process

1. Pull latest from the GitHub repo:
   ```bash
   cd ~/git/github/koi
   git pull origin main
   ```

2. Sync files to the GitLab repo using rsync, excluding protected dirs:
   ```bash
   rsync -av --delete \
     --exclude='.git/' \
     --exclude='.koi/' \
     --exclude='src/koi/bundled_skills/' \
     ~/git/github/koi/ ~/git/koi/
   ```

3. Commit and push the GitLab repo:
   ```bash
   cd ~/git/koi
   git add -A
   git status
   git commit -m "Sync from GitHub"
   git push origin main
   ```

## Notes

- `.git/` is always excluded (each repo has its own remote)
- `.koi/` contains project-specific config and memory — GitLab-only
- `src/koi/bundled_skills/` may differ between environments
- Always review `git status` before committing
