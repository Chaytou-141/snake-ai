# Version 1.0 verification

- 5H checkpoint: 4504adcace104397e826c9092056d4a395c6087e, verified and pushed before portfolio development.
- Focused demo/dashboard/legacy playback checks: 44 passed, one dependency deprecation warning.
- Full suite: **336 passed, 0 failed**, one existing Pygame dependency deprecation warning; run once at completion.
- Historical preservation: 502 saved result/checkpoint files match the pre-Version 1.0 SHA-256 manifest.
- All five fixed catalog checkpoints are present and compatible locally.
- Genuine GIF/PNG captures were created by inference only; original checkpoint hash and in-memory weights/counters were verified unchanged.
- Dashboard means/SDs/paired comparisons are checked against saved per-seed CSVs; source JSON/CSV hashes are included in dashboard/data.json.
- Desktop and compact layouts, actual Pygame screenshots, and the architecture diagram were visually reviewed.
- Editable package installed as 1.0.0 with --no-deps; new console entry points were checked.

No historical experiment was regenerated, no model was retrained, and no checkpoint was uploaded.
Staging excludes model weights and sensitive files. The repository remains private;
the final commit/push and branch synchronisation are reported in the completion record.
