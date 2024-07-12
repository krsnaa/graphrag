TO sync this repo with upstream(microsoft/graphrag) and push changes to origin (krsnaa/graphrag):

bring in the latest changes from upstream/main:

- on the bottom-left of the status-bar, click on the branch name and switch to main
- click the refresh button next to the branch name to pull and push changes to upstream/main

this will end up updating the local/main. now, to push changes up to origin/main:
❯ git push origin main

now, to rebase the annotations branch off the latest main:
switch to the annotations branch
then, in the Git Graph view, right-click on the local/main and choose the 'Rebase current branch on Branch...' option.

now, to sync the local branch of annotations with origin/annotations:
❯ git push -f origin annotations

---

TO run GraphRAG on a project:
❯ poetry shell

SETUP:
❯ mkdir projects/qa-gsd
❯ mkdir projects/qa-gsd/input
❯ mkdir projects/qa-gsd/output

INIT: https://microsoft.github.io/graphrag/posts/config/init/
python -m graphrag.index [--init] [--root PATH]

❯ poetry run poe index --init --root ./projects/qa-gsd
ends up calling:
❯ python -m graphrag.index --init --root ./projects/qa-gsd
