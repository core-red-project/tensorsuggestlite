# Changelog

All notable changes to TensorSuggestLite are documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- OSS documentation suite (CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md)
- FILE_MANIFEST.md for complete repository structure documentation
- REPOSITORY_STATUS.md for project status tracking
- Type hints throughout codebase (PEP 484 compliance)
- CI/CD workflows for linting and validation
- Dependency pinning for reproducibility

### Changed
- Improved error handling in trainer modules
- Enhanced docstrings for public APIs
- Standardized naming conventions across modules
- Hardened file I/O operations with proper resource management

### Fixed
- Bare exception handlers replaced with specific exception types
- Resource leaks in file operations
- Thread safety issues in UI callbacks

## [2.0.0] - 2025

### Added
- Support for JSON, TOML, and YAML configuration formats
- PyQt6-based graphical user interface
- TensorFlow/Keras text classification trainer
- TFLite conversion for embedded deployment
- Dark/light theme support
- Progress callbacks and logging during training
- File system watcher for artifact detection
- Build scripts for macOS (.app) and Windows (.exe) packaging

### Changed
- Modular architecture with separate interpreter packages per format
- Improved UI responsiveness with threaded training
- Enhanced error reporting throughout application

### Fixed
- Various stability improvements

## [1.0.0] - Prior

- Initial internal release
- Basic functionality established

---

**Versioning**: This project follows [Semantic Versioning](https://semver.org/).

**Maintenance**: As an experimental project, releases are irregular and driven by community contributions and maintainer availability.

