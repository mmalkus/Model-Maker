// Bundles the extension into dist/extension.js (vscode stays external: the
// host provides it). --tests compiles the unit tests, which import only
// vscode-free modules, to out/unit/ for `node --test`; the integration tests
// to out/integration/ for vscode-test.
import * as esbuild from 'esbuild'
import { readdirSync } from 'node:fs'

const args = new Set(process.argv.slice(2))
const common = { bundle: true, platform: 'node', target: 'node20', format: 'cjs', external: ['vscode'], logLevel: 'info' }

if (args.has('--tests')) {
  const entries = (dir) => readdirSync(dir).filter((f) => f.endsWith('.test.ts')).map((f) => `${dir}/${f}`)
  await esbuild.build({ ...common, entryPoints: entries('test/unit'), outdir: 'out/unit', sourcemap: true })
  await esbuild.build({ ...common, entryPoints: entries('test/integration'), outdir: 'out/integration', external: ['vscode', 'mocha'] })
} else {
  const options = {
    ...common,
    entryPoints: ['src/extension.ts'],
    outfile: 'dist/extension.js',
    minify: args.has('--production'),
    sourcemap: !args.has('--production'),
  }
  if (args.has('--watch')) await (await esbuild.context(options)).watch()
  else await esbuild.build(options)
}
