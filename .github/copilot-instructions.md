# Copilot Cloud Agent Onboarding

This repository is a Summer 2026 sandbox for graduate CFD research, Formula SAE team work, and Python learning on macOS.

## Repository Intent
- Primary language: Python.
- Target environment: macOS and Python terminal workflows.
- This is a general-purpose sandbox; do not assume a specific framework or build system until explicit project metadata is added.
- The user wants detailed notes explaining why libraries, functions, or files are being added.

## Files and Structure
- `.github/copilot-instructions.md` — this agent guidance file.
- `.gitignore` — local and sensitive environment files are excluded.
- `venv/` — local Python virtual environment; ignore this directory.
- `Python-Tools-Examples/` — Python examples, experiments, and utility tools.
- `Formula-SAE/` — Formula SAE code, simulations, and related team work.
- `Grad-Research/` — graduate research code, CFD and analysis scripts.

## Agent Behavior
- Trust this file for repository-level assumptions.
- Only search for additional project metadata when new files appear or the instruction set is incomplete.
- Do not generate code relying on a build, test, or CI pipeline unless those systems are explicitly added.
- Provide detailed notes with each proposed change, including the rationale for chosen libraries and code structure.

## Validation Guidance
- No validated build/test commands exist yet in this repository.
- Do not run build or test steps until the repository contains explicit Python project files, such as `requirements.txt`, `pyproject.toml`, or CI workflow definitions.
- If the repository remains a skeleton, focus on creating safe scaffolding and documentation rather than project-specific implementation.
