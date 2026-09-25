/**
 * Syncs the compiled decision engine (npm/llmrouter/dist) into the website so
 * the live demo runs the REAL engine in the browser.
 *
 * Run from npm/llmrouter:  node scripts/sync-engine.mjs
 * (after `npm run build`). cli.js and index.js are excluded: cli.js needs
 * node:fs, and index.js re-exports it; the demo imports the engine modules
 * directly.
 */
import { mkdirSync, copyFileSync, readdirSync, statSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dist = join(here, "..", "dist");
const dest = join(here, "..", "..", "..", "website", "assets", "engine");

mkdirSync(dest, { recursive: true });

const included = new Set([
  "types.js", "pricing.js", "economics.js", "affinity.js",
  "backends.js", "signals.js", "policy.js", "learning.js", "hash.js",
]);

let copied = 0;
for (const f of readdirSync(dist)) {
  if (included.has(f) && statSync(join(dist, f)).isFile()) {
    copyFileSync(join(dist, f), join(dest, f));
    copied += 1;
  }
}
if (copied !== included.size) {
  console.error(`expected ${included.size} engine modules, copied ${copied} — run "npm run build" first`);
  process.exit(1);
}
console.log(`synced ${copied} engine modules -> website/assets/engine/`);
