# Releasing

One-time setup (your PyPI account, nothing stored in the repository):

1. Create an account on https://pypi.org and, for rehearsals, on https://test.pypi.org.
2. On each site, go to *Your account → Publishing* and add a **pending trusted publisher**:
   PyPI project name `kac-zwanzig`, owner `william-pfalzgraff`, repository `kac-zwanzig`,
   workflow `publish.yml`, environment `pypi` (use `testpypi` for the test site).
3. In the GitHub repository, *Settings → Environments*, create environments named `pypi`
   and `testpypi` (optionally require your approval before a deployment runs).

Each release:

1. Bump `__version__` in `src/kac_zwanzig/__init__.py`, update `CHANGELOG.md` and the
   `version` / `date-released` fields in `CITATION.cff`. Commit.
2. Rehearse: *Actions → publish → Run workflow* with target `testpypi`, then
   `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple kac-zwanzig`
   in a fresh environment.
3. Tag and push: `git tag v0.4.0 && git push origin v0.4.0`. The `publish` workflow builds
   the wheel and source distribution, checks them, and uploads to PyPI.
4. Create the GitHub release from the tag (the CHANGELOG entry is the release note).

Local checks before tagging:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check src tests examples
python -m build && python -m twine check dist/*
```
