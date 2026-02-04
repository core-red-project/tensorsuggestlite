# Contributing to TensorSuggestLite

TensorSuggestLite is maintained by CoreRed Project and accepts contributions from the community under the terms specified in this document.

## Scope

This project is experimental. Contributions are accepted for:

- Bug fixes
- Documentation improvements
- Code quality enhancements
- Test coverage additions
- Performance optimizations

Feature proposals require prior discussion via GitHub Issues.

## Prerequisites

- Python 3.8 or later
- Familiarity with TensorFlow/Keras, PyQt6, and ML classification workflows
- Git and GitHub workflow knowledge

## Development Setup

1. Fork and clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/TensorSuggestLite.git
cd TensorSuggestLite
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application to verify setup:

```bash
python tensorsuggestlite.py
```

## Code Standards

### Python Style

- Follow PEP 8
- Use type hints (PEP 484) for function signatures
- Maximum line length: 100 characters
- Use descriptive variable names

### Documentation

- All public functions must have docstrings (Google or NumPy style)
- Update README.md for user-facing changes
- Update CHANGELOG.md for all changes

### Error Handling

- Use specific exception types, not bare `except:`
- Log errors with descriptive messages
- Fail gracefully in the UI layer

### Testing

- Add tests for new functionality (when test framework is established)
- Verify existing functionality is not broken

## Commit Guidelines

Use conventional commit format:

```
type(scope): brief description

Longer explanation if needed.

Fixes #issue_number
```

Types: `fix`, `feat`, `docs`, `refactor`, `test`, `chore`, `perf`

Example:
```
fix(trainer): handle empty input files gracefully

Added validation to prevent crashes when JSON/YAML/TOML
files contain no training data.

Fixes #42
```

## Pull Request Process

1. Create a feature branch from `main`:
   ```bash
   git checkout -b fix/issue-description
   ```

2. Make changes following code standards

3. Test your changes thoroughly

4. Commit with clear messages

5. Push to your fork:
   ```bash
   git push origin fix/issue-description
   ```

6. Open a Pull Request against `main` with:
   - Clear title describing the change
   - Description of what changed and why
   - Reference to related issues
   - Confirmation that testing was performed

7. Respond to review feedback

8. Maintainers will merge when approved

## Review Criteria

Pull requests are evaluated on:

- Code quality and adherence to standards
- Test coverage (when applicable)
- Documentation completeness
- Backwards compatibility
- Security implications
- Performance impact

## Licensing

By contributing, you agree that your contributions will be licensed under the same CR-NC-1.0 license as the project. You retain copyright to your contributions but grant CoreRed Project a perpetual, irrevocable license to use, modify, and distribute them under the project license.

Contributions must be your original work or properly attributed open source code compatible with CR-NC-1.0.

## Communication

- **Issues**: Use GitHub Issues for bug reports and feature discussions
- **Pull Requests**: Use PR comments for code review discussions
- **Maintainer Contact**: houjou.sxnnyside@sxnnysideproject.com
- **Security Issues**: security.sxnnyside@sxnnysideproject.com (see SECURITY.md)

## Code of Conduct

All contributors must follow the project Code of Conduct (see CODE_OF_CONDUCT.md).

## Questions

For support questions not suitable for GitHub Issues, contact:
support.sxnnyside@sxnnysideproject.com

---

Thank you for contributing to TensorSuggestLite.

