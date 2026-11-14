import { readFileSync } from "node:fs";
import { identify } from "./engine.mjs";

const { detection, catalog } = JSON.parse(readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(identify([detection], catalog).results[0]));
