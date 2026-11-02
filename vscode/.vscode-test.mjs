import { defineConfig } from '@vscode/test-cli'

// Integration tests: a real VS Code with this extension loaded, opening the
// demo project (written by `modelmaker-demo`, see the CI vscode job).
export default defineConfig({
  files: 'out/integration/**/*.test.js',
  workspaceFolder: process.env.MODELMAKER_TEST_WORKSPACE ?? '.vscode-test/workspace',
  mocha: { timeout: 180000 },
})
