# Contributing

Read [README.md](README.md) and [RELEASE_PROVENANCE.md](RELEASE_PROVENANCE.md) for
setup, the scope of this public snapshot, and the scientific limitations. Use a
new output directory for each run. Changes should preserve input hashes, observation
metadata, uncertainty limits, and inconclusive outcomes.

For the full regression suite, install the optional components and run:

```sh
uv sync --frozen --extra catalog --extra specialist --extra photometry
uv run --no-sync pytest -q
```

The identity fixtures and simulated artifact controls test bounded interfaces and
methods; they do not establish real-sky completeness, object identity, or a discovery.
The existing source includes the original collaborator interface attribution.
