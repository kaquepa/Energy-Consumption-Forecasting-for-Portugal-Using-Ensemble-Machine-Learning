#!/bin/bash

set -e  # stop on first error

# Colors
C_RED="\033[1;31m"; C_GREEN="\033[1;32m"; C_YELLOW="\033[1;33m"; C_CYAN="\033[1;36m"; C_RESET="\033[0m"
say()  { echo -e "${C_CYAN}$1${C_RESET}"; }
ok()   { echo -e "${C_GREEN}$1${C_RESET}"; }
warn() { echo -e "${C_YELLOW}$1${C_RESET}"; }
err()  { echo -e "${C_RED}$1${C_RESET}" >&2; }

# Validate Git repository
git rev-parse --git-dir >/dev/null 2>&1 || { err "This directory is not a Git repository."; exit 1; }

say "Data Science Pipeline — Git Automation"
git status --short || exit 1

warn "Adding files..."
git add -A

# Check if there is anything to commit
git diff --cached --quiet && { ok "No changes to commit."; exit 0; }

# Commit message
read -rp "Commit message: " MSG
[[ -z "$MSG" ]] && { err "Commit message cannot be empty."; exit 1; }

BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo ""
say "Commit: $MSG"
say "Branch: $BRANCH"
read -rp "Confirm commit and push? (y/N): " CONFIRM
[[ ! "$CONFIRM" =~ ^[yY]$ ]] && { warn "Operation cancelled."; exit 0; }

# Commit & Push
say "Committing..."
git commit -m "$MSG"

say "Pushing changes..."
git push origin "$BRANCH" && ok "Push completed!"

# Merge if branch is development
if [[ "$BRANCH" == "development" ]]; then
    echo ""
    read -rp "Would you like to merge 'development' into 'main'? (y/N): " DO_MERGE
    if [[ "$DO_MERGE" =~ ^[yY]$ ]]; then
        
        say "Checking differences with 'main'..."
        git fetch origin main
        
        if git diff --quiet "$BRANCH" origin/main; then
            warn "No differences detected between 'development' and 'main'."
        else
            say "Merging '$BRANCH' → 'main'..."
            git checkout main
            git pull origin main

            if git merge --no-edit "$BRANCH"; then
                ok "Merge completed!"
                git push origin main && ok "Main pushed successfully!"
            else
                err "Merge conflicts detected! Please resolve manually."
                exit 1
            fi
            
            git checkout "$BRANCH"
        fi
    else
        warn "Merge skipped."
    fi
fi

# Write log
echo "$(date '+%Y-%m-%d %H:%M:%S') | [$BRANCH] $MSG" >> git_auto_log.txt
ok "Operation completed and logged!"
