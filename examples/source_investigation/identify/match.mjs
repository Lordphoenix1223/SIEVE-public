import { readFileSync } from "node:fs";
import { identify } from "./engine.mjs";

// Each job has its own epoch-propagated catalog. Keep matcher semantics unchanged.
const { jobs } = JSON.parse(readFileSync(0, "utf8"));
if (!Array.isArray(jobs)) throw new TypeError("jobs must be an array");
process.stdout.write(JSON.stringify(jobs.map(({ detection, catalog }) =>
  identify([detection], catalog).results[0])));
