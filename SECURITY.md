# Security Policy

## Scope

TensorSuggestLite is an experimental machine learning utility. This policy defines how security vulnerabilities are handled.

## Supported Versions

| Version | Status | Security Updates |
|---------|--------|------------------|
| 2.x     | Experimental | Best effort |
| < 2.0   | Unsupported | No updates |

As an experimental project, security updates are provided on a best-effort basis without guarantees of timeliness or completeness.

## Known Limitations

### Not for Production Deployment

TensorSuggestLite is **not intended** for production systems or security-sensitive environments. It is designed for:

- Local experimentation
- Educational purposes
- Prototype development

### Security Boundaries

This application:

- Executes arbitrary Python code (TensorFlow model training)
- Reads and writes local files
- Loads untrusted configuration files (JSON/YAML/TOML)
- Has no input sanitization for ML training data
- Has no authentication or access controls

**Do not:**
- Deploy as a web service
- Process untrusted or sensitive data
- Run with elevated privileges
- Use in multi-tenant environments

## Reporting Vulnerabilities

### Contact

Report security issues to:

**security.sxnnyside@sxnnysideproject.com**

Do **not** report security vulnerabilities via public GitHub Issues.

### What to Include

Provide:

1. Detailed description of the vulnerability
2. Steps to reproduce
3. Affected versions (if known)
4. Proof of concept (if applicable)
5. Potential impact assessment
6. Suggested mitigation (if any)

### Response Process

1. **Acknowledgment**: Within 72 hours of report receipt
2. **Assessment**: Maintainers evaluate severity and impact
3. **Resolution**: 
   - Critical: Best effort within 30 days
   - High: Best effort within 60 days
   - Medium/Low: Addressed in regular development cycle
4. **Disclosure**: After fix is released or 90 days, whichever comes first

### Coordinated Disclosure

We follow coordinated disclosure:

- Reporters are kept informed of progress
- Fixes are developed privately
- Public disclosure occurs after patch availability or 90 days
- Reporter receives credit unless anonymity is requested

## Security Considerations

### Dependencies

TensorFlow and other dependencies may have known vulnerabilities. Users are responsible for:

- Reviewing dependency security advisories
- Updating dependencies in their deployments
- Testing compatibility after updates

### File Processing

The application processes configuration files without extensive validation. Maliciously crafted files could:

- Cause crashes (denial of service)
- Consume excessive resources
- Trigger unexpected behavior

**Mitigation**: Only process files from trusted sources.

### Model Artifacts

Generated models, tokenizers, and label encoders:

- Are not signed or verified
- May contain embedded information from training data
- Should not be distributed if trained on sensitive data

### GUI Security

The PyQt6 GUI:

- Has no authentication
- Trusts the local filesystem
- Provides full file system access via file dialogs

**Mitigation**: Run only on trusted local machines.

## Responsible Use

Users must:

- Comply with applicable laws and regulations
- Respect data privacy and intellectual property
- Not use for malicious purposes
- Understand and accept the experimental nature and limitations

## Legal

CoreRed Project provides TensorSuggestLite "as is" without warranty. Security issues do not constitute a breach of contract or liability. See LICENSE for full terms.

For legal inquiries related to security: legal.sxnnyside@sxnnysideproject.com

---

**Version**: 1.0  
**Last Updated**: 2026-02-03  
**Contact**: security.sxnnyside@sxnnysideproject.com

