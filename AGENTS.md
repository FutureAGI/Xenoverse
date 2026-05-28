# Repository Guidelines

## Project Structure & Module Organization
`xenoverse/` is the installable Python package. Each environment family lives in its own subpackage, including `anymdp/`, `anyhvac/`, `linds/`, `mazeworld/`, `metacontrol/`, `metalang/`, and `sci_research/`. Shared helpers belong in `xenoverse/utils/`. Module-level documentation is kept close to code in files like `xenoverse/anymdp/README.md`.

Tests are currently local to modules instead of a single top-level suite. Existing examples live in `xenoverse/mazeworld/tests/` and `xenoverse/metacontrol/tests/`. Keep new assets next to the package that uses them, for example `xenoverse/mazeworld/envs/img/`.

## Build, Test, and Development Commands
Install dependencies and the package from the repository root:

```bash
pip install -r requirements.txt
pip install .
```

Run lightweight module tests directly:

```bash
python -m xenoverse.mazeworld.tests.test
python -m xenoverse.mazeworld.tests.test_agent
python -m xenoverse.metacontrol.tests.test
```

Use `pip install -e .` during development if you need editable imports while changing package code.

## Coding Style & Naming Conventions
Follow existing Python conventions: 4-space indentation, snake_case for modules/functions, PascalCase for environment classes, and short explicit import paths such as `xenoverse.anymdp.task_sampler`. Keep Gymnasium registration blocks in package `__init__.py` files and use descriptive environment IDs like `anymdp-v0` or `random-cartpole-v0`.

There is no repository-wide formatter or linter configuration checked in. Match surrounding style, keep lines readable, and avoid unrelated formatting churn in touched files.

## Testing Guidelines
`setup.py` declares `pytest`, but current tests are executable scripts that exercise environment creation and stepping. Add new tests near the affected module under a `tests/` directory and prefer names like `test.py`, `test_agent.py`, or `test_<feature>.py`. For new environments, cover import/registration, `env.set_task(...)`, `reset()`, and at least one sampled rollout.

## Commit & Pull Request Guidelines
Recent history favors short imperative subjects such as `update readme`, `fix problem`, and `Update module links in README to relative paths`. Keep commit titles concise, action-oriented, and scoped to one change.

Pull requests should describe the affected module, list verification commands you ran, and note any API or environment-ID changes. Include screenshots only for rendering or visualization changes, and update the nearest module `README.md` when behavior or usage changes.
