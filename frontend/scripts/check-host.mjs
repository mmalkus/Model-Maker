// Guards the host abstraction (src/host.ts): API URLs are built only in
// api.ts (with host.apiBase), and browser-only calls go through host.ts.
// oxlint's no-restricted-globals catches bare alert/confirm/prompt; this
// catches the window.-qualified forms and raw '/api' URLs it can't.
import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

const SRC = new URL('../src/', import.meta.url).pathname
const rules = [
  { re: /['"`]\/api[/'"`]/, allow: ['api.ts'], why: "build API URLs in api.ts (they need host.apiBase)" },
  { re: /window\.(alert|confirm|prompt)\(|\blocalStorage\b|\.download\s*=/, allow: ['host.ts'], why: 'use host.* (VS Code webviews block it)' },
]

const failures = []
for (const name of readdirSync(SRC)) {
  if (!/\.(ts|tsx)$/.test(name)) continue
  readFileSync(join(SRC, name), 'utf8')
    .split('\n')
    .forEach((line, i) => {
      if (line.trim().startsWith('//')) return
      for (const { re, allow, why } of rules) {
        if (re.test(line) && !allow.includes(name)) failures.push(`src/${name}:${i + 1}: ${why}\n    ${line.trim()}`)
      }
    })
}
if (failures.length) {
  console.error(failures.join('\n'))
  process.exit(1)
}
