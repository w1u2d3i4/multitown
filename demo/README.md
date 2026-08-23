# MultiTown Arena demo

The Arena is a zero-dependency deterministic visualization of the frozen A4/A8
comparison. It does not call an LLM, require credentials, or expose raw
experiment records. The displayed benchmark aggregates are measured results;
the animated work order is an explanatory scenario, not a raw experimental
episode.

Its replay-first presentation pattern was informed by Google Cloud's
[Race Condition](https://github.com/GoogleCloudPlatform/race-condition). The
MultiTown implementation is a lightweight static frontend focused on
organization control, validation, escalation, and token efficiency.

Run it locally:

```bash
python3 -m http.server 8000 --directory demo
```

Then open <http://127.0.0.1:8000>.

Regenerate the README GIF from the same replay states:

```bash
python3 demo/generate_gif.py
```

GIF generation requires Firefox and Pillow. The browser renders each replay
state at 960×540; Pillow assembles the screenshots without any manual capture.
