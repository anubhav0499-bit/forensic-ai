# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability (e.g., hardcoded credentials, API key exposure, or a code path that could leak secrets), **do not open a public GitHub issue**.

Instead, report it privately via GitHub's Security tab:
1. Go to the repository's **Security** tab → **Advisories** → **New draft advisory**.
2. Describe the vulnerability, affected file(s), and steps to reproduce.

Alternatively, email the maintainer directly at anubhav0499@gmail.com.

We aim to acknowledge reports within 48 hours and patch confirmed vulnerabilities within 7 days.

## API Keys & Secrets

This repository **never stores real API keys**. Configuration uses environment variables only:

- Copy `.env.example` → `.env` and fill in your keys
- `.env` is listed in `.gitignore` and must never be committed
- GitHub secret scanning is enabled on this repository — any accidentally pushed secrets will be automatically detected and alerted

## Supported Versions

| Version | Supported |
|---------|-----------|
| main / master | Yes |
| Older branches | No — please upgrade |
