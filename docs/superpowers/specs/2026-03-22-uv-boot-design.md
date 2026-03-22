# UV Boot Design

## Goal

Make the repository runnable from a single stable demo entrypoint, then switch dependency management to `uv` without refactoring every legacy script.

## Scope

- Use `generate_defect.py` as the first-class demo entrypoint.
- Add a dependency-managed `pyproject.toml`.
- Support a `--dry-run` mode that validates startup configuration without loading diffusion models.
- Keep older scripts available, but do not normalize all of their hardcoded paths in this pass.

## Design

### Entry point

`generate_defect.py` becomes the boot path because it already uses repo-local demo assets under `data/sg`.

### Startup configuration

Move CLI and environment resolution into a standard-library-only helper so startup can be validated before installing model dependencies.

### Dependency strategy

Use `uv` in project mode with `package = false`, because this repository is still organized as scripts rather than a distributable package.

### Verification

The first passing condition is `python generate_defect.py --dry-run`.
The second passing condition is `uv run python generate_defect.py --dry-run`.
