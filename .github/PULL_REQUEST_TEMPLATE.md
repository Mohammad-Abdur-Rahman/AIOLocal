# Pull request

## Summary

<!-- What changed and why. Link the issue this closes. -->

Closes #

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Performance
- [ ] Security hardening
- [ ] Documentation / packaging
- [ ] Refactor (no behaviour change)

## How it was tested

<!-- Commands, manual steps, hardware and models used. -->

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Checklist

- [ ] The change is focused and does not include unrelated refactors.
- [ ] A regression test was added for the fixed bug or new behaviour.
- [ ] The full test suite passes locally.
- [ ] `CHANGELOG.md` is updated for user-visible changes.
- [ ] No model weights, `dist/`, `build/`, logs or secrets are committed.
- [ ] Error responses contain curated messages, not raw exception text.
- [ ] UI changes include a screenshot (light or dark theme).
- [ ] No new CodeQL alerts or Dependabot alerts are introduced.
- [ ] Documentation (README / SECURITY / CONTRIBUTING) is updated if needed.
