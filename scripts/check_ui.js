// Dev-only: sanity-check that the UI's selectors match its markup.
const fs = require('fs');
const path = require('path');

const dir = process.argv[2] || 'frontend';
const html = fs.readFileSync(path.join(dir, 'index.html'), 'utf8');
const js = fs.readFileSync(path.join(dir, 'app.js'), 'utf8');
const css = fs.readFileSync(path.join(dir, 'app.css'), 'utf8');

const ids = new Set([...html.matchAll(/id="([^"]+)"/g)].map((m) => m[1]));
const selectors = [...new Set([...js.matchAll(/\$\('#([a-zA-Z0-9_-]+)'\)/g)].map((m) => m[1]))];
const missingIds = selectors.filter((s) => !ids.has(s));

const classes = new Set(
  [...html.matchAll(/class="([^"]+)"/g)].flatMap((m) => m[1].split(/\s+/)).filter(Boolean)
);
const missingCss = [...classes].filter((c) => !css.includes('.' + c));

const jsClasses = new Set(
  [...js.matchAll(/classList\.(?:add|toggle)\('([a-zA-Z0-9_-]+)'/g)].map((m) => m[1])
);
const missingJsCss = [...jsClasses].filter((c) => !css.includes('.' + c));

console.log('elements referenced in app.js but missing from index.html:', missingIds.length ? missingIds : 'none');
console.log('html classes without a css rule:', missingCss.length ? missingCss : 'none');
console.log('js-toggled classes without a css rule:', missingJsCss.length ? missingJsCss : 'none');

const bad = missingIds.length + missingCss.length + missingJsCss.length;
process.exit(bad ? 1 : 0);
