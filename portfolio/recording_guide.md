# Demonstration recording guide

## Ready-to-use genuine assets

- `assets/playback.gif`: approximately 12 seconds of actual original11 inference,
  seed 800000, first 120 moves, every second move rendered at 200ms per frame.
- `assets/playback.png`: actual state after 60 moves, with live Q estimates.
- `assets/menu.png`: the real Pygame selection screen and available local models.
- `assets/capture.json`: checkpoint hash, seed, frame timing and capture provenance.

The clip is an animated GIF, not an encoded video. No ffmpeg/video-editing
package was installed. The score shown belongs to this demonstration only;
it must not be presented as a new held-out experiment result.

## Record an MP4 from the running application

1. Activate the existing project environment.
2. Run `python -m snake_ai.demo --model original11 --seed 800000 --speed 10`.
3. On Windows, focus the Pygame window and use Xbox Game Bar's
   `Win+Alt+R` to start/stop recording, if enabled. Alternatively use an
   already-installed screen recorder. Keep the complete board and Q panel visible.
4. Record 20?30 seconds. Show Space to pause, R to replay the same seed,
   and Tab to switch models. Switching pauses at the same current seed;
   press Space to continue. The displayed model and algorithm identify each clip.
5. Keep the evidence separate: show the dashboard with
   `python -m snake_ai.dashboard --open`, and state each comparison's budget.
6. Review the recording for desktop notifications or unrelated personal content
   before manually sharing it. Do not include checkpoint files in a post.

## Suggested LinkedIn screenshots

- Post 1: the architecture diagram and actual human-play screen.
- Post 2: `assets/playback.png`; retain the Q-value interpretation text.
- Post 3: `results/milestone5c_trap_analysis/safe_actions.png` and the
  diagnostic definition. Explain that static space is not future safety.
- Post 4: dashboard sections for the individual experiments; include their
  context labels. Use the 5G curve to illustrate non-monotonic performance.

Reuse the original charts without adjusting values or mixing studies into a
ranking. Additional screenshots should come from the actual running application.

## Recreate a demonstration capture explicitly

```powershell
python portfolio/capture_playback.py --output portfolio/my_capture --model original11 --seed 800000
```

This performs inference only, writes a new capture directory, and refuses to
replace existing capture files or write under historical `results/`. Pillow is
already available through the existing Matplotlib plotting dependencies.
The architecture can be rebuilt with `python portfolio/build_architecture.py`.
