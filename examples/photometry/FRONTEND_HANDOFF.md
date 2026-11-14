# QR3 marker handoff

Generate fresh measurements and validate/export them (outputs must not exist):

```sh
.venv/bin/python examples/photometry/run_demo.py --output /tmp/qr3-scan.json
.venv/bin/python examples/photometry/export_frontend.py /tmp/qr3-scan.json /tmp/qr3-markers.json
```

`markers[].ra_deg` and `dec_deg` are ICRS-compatible sky coordinates in degrees.
The exporter validates the input scan and its output with Pydantic; its schema can be
obtained with `Evidence.model_json_schema()` from `examples.photometry.export_frontend`.
The status is always measurement-only. The aperture value is a sum of surface-brightness
pixels, not calibrated flux density. `aperture_usable` only describes the scan's mask
policy, not astrophysical validity. Flagged and artifact-like peaks may remain. The MJD
is the scan's recorded observation time (the pinned example uses MJD-OBS, exposure
start), not a separately inferred midpoint. No identities, paths, uncertainty in position,
external catalog association or independent verification are supplied.

The local sample's parent URL and source hashes are in `source.json`. The export includes
the input hash, code commit, and limitations; it does not automatically validate a URL
or expose the original parent link. A frontend integration is not yet demonstrated.
