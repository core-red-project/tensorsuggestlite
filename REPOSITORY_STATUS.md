# Repository Status

**Project**: TensorSuggestLite  
**Maintainer**: CoreRed Project  
**Status**: Experimental  
**Last Updated**: 2026-02-03

## Current State

### Stability: Experimental

This project is functional but **not production-ready**. It serves as:

- A learning tool for TensorFlow/Keras text classification
- A prototype for configuration-driven model training
- An experimental GUI for ML workflows

### Maintenance Level: Community-Driven

- **Active Development**: Irregular
- **Issue Response**: Best effort, no SLA
- **Pull Requests**: Reviewed as maintainer time permits
- **Security Updates**: Best effort within 30-90 days

### Known Limitations

1. **No Test Suite**: Automated testing is not yet implemented
2. **Unpinned Dependencies**: Some dependencies lack specific version constraints
3. **Limited Validation**: Input files are not extensively validated
4. **No Logging Framework**: Uses print/callback-based logging
5. **Single-Threaded Training**: GUI may freeze during intensive operations despite threading
6. **No Model Versioning**: Generated artifacts lack version metadata

## Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| macOS (x86_64/arm64) | Supported | Build script provided |
| Windows (x64) | Supported | Build script provided |
| Linux | Untested | Should work, not officially supported |

## Python Version Support

- **Minimum**: Python 3.8
- **Tested**: Python 3.8, 3.9, 3.10
- **Recommended**: Python 3.10+

## Dependency Health

| Dependency | Status | Risk |
|------------|--------|------|
| TensorFlow | Active | High complexity, large CVE surface |
| PyQt6 | Active | Stable |
| NumPy | Active | Stable |
| scikit-learn | Active | Stable |
| PyYAML | Active | Known CVE history, monitor updates |
| toml | Active | Simple, low risk |

**Recommendation**: Users should pin dependencies and monitor security advisories independently.

## Feature Completeness

| Feature | Status | Notes |
|---------|--------|-------|
| JSON training | Complete | Stable |
| YAML training | Complete | Stable |
| TOML training | Complete | Stable |
| TFLite conversion | Complete | Functional, not optimized |
| GUI | Complete | Usable, lacks advanced features |
| CLI mode | Missing | Planned, not prioritized |
| Batch processing | Missing | Not planned |
| Model evaluation | Missing | Not planned |
| Hyperparameter tuning | Missing | Not planned |

## Roadmap

No formal roadmap exists. Future direction depends on:

- Community contributions
- Maintainer availability
- Security requirements

Potential areas for improvement:

- Add test framework (pytest)
- Implement CLI mode for scripting
- Add model evaluation metrics
- Improve input validation
- Pin all dependency versions

## Risk Assessment

**Overall Risk**: Medium

- **Security**: Medium (processes untrusted files, dependencies have CVE history)
- **Stability**: Medium (functional but limited testing)
- **Maintenance**: Medium (best-effort support)

**Recommended Use Cases**:
- Local experimentation ✓
- Educational purposes ✓
- Production deployment ✗
- Security-sensitive data ✗

## Getting Help

- **Issues**: GitHub Issues (no SLA)
- **Support**: support.sxnnyside@sxnnysideproject.com (best effort)
- **Security**: security.sxnnyside@sxnnysideproject.com (see SECURITY.md)

## Contributing

Contributions are welcome. See CONTRIBUTING.md for guidelines.

Priority areas:
1. Test coverage
2. Documentation improvements
3. Code quality (type hints, error handling)
4. Security hardening

## Governance

**Decision Making**: CoreRed Project maintainers have final authority on:
- Architecture changes
- Feature acceptance
- Release timing
- Security policies

**Transparency**: Major decisions will be documented in GitHub Issues when feasible.

## Legal

- **License**: CR-NC-1.0 (see LICENSE)
- **Copyright**: CoreRed Project
- **Trademark**: "TensorSuggestLite" and "CoreRed" are property of CoreRed Project

---

**Maintainer Contact**: houjou.sxnnyside@sxnnysideproject.com

