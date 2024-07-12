TO sync this repo with upstream(microsoft/graphrag) and push changes to origin (krsnaa/graphrag):
❯ git remote add upstream
❯ git fetch upstream

TO run GraphRAG ona project:
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
