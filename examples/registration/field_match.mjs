// Use the same unmodified SearchX engine for all field detections in one process.
import { readFileSync } from 'node:fs';
import { identify } from '../source_investigation/identify/engine.mjs';
const { detections, catalog } = JSON.parse(readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(identify(detections, catalog)));
