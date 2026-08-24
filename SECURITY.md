# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately rather than opening a public
issue. Use GitHub's private vulnerability reporting on this repository, or
contact the maintainer directly. Include reproduction steps and the affected
version or commit. Expect an acknowledgement within a few days.

Please do not include real credentials, private datasets, or patient-derived
data in a report.

## Scope of this repository

BioFlow orchestrates external workflow engines and model checkpoints. Two
consequences are worth stating explicitly:

- **No proprietary or employer-internal content.** This repository must contain
  no internal workflows, code, schemas, thresholds, datasets, unpublished
  models, or sequence-design logic. It targets public datasets, public model
  checkpoints, public APIs, and public workflow repositories only.
- **No secrets in version control.** Configuration comes from the environment;
  see `.env.example`. `.env` is git-ignored and must stay that way.

BioFlow executes external processes (Nextflow, Snakemake, container runtimes).
Command construction is deterministic and argument-array based — never shell
string interpolation — and every execution is gated by a preflight plan and a
policy decision.
