# Legacy Key Tests

This directory preserves the historical current-vs-reference comparison
harness.  It is no longer part of the default package or workspace boundary.

Why it is legacy:

- It contains copied reference implementation files.
- Several configs, reports, and notebook helpers contain absolute workstation
  paths from the original comparison run.
- It writes comparison artifacts under its own output tree rather than the new
  `workspace/outputs/` run layout.
- It is useful provenance, but it should not influence default test collection
  or new production CLI behavior.

Run it only as a deliberate historical comparison exercise after checking the
paths in `configs/`, `workspace_support.py`, and `reports/run_manifest.json`.
