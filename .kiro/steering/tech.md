# Code standards

- Modules are named `<verb>_<corpus>.py`, one per stage per corpus.
- Records are built by one factory function; never construct a record inline.
- Shared helpers are imported, never copied.
- Scripts take `--config` and any input folder as arguments; no hard-coded relative paths.
- Every identifier regex family has a fixture test with asserted counts.
- Output to chat is counts and paths, never corpus text.
- Pipeline scripts run no git commands.
