import assert from 'node:assert/strict'
import { deflateRawSync } from 'node:zlib'
import { test } from 'node:test'
import { pickUvWheel, UV_VERSION, type WheelFile } from '../../src/uv'
import { extractEntry } from '../../src/zip'

// uv 0.13.2's files on PyPI, as the JSON API lists them.
const FILENAMES: string[] = [
  "uv-0.13.2-py3-none-linux_armv6l.whl",
  "uv-0.13.2-py3-none-macosx_10_12_x86_64.whl",
  "uv-0.13.2-py3-none-macosx_11_0_arm64.whl",
  "uv-0.13.2-py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.musllinux_1_1_aarch64.whl",
  "uv-0.13.2-py3-none-manylinux_2_17_armv7l.manylinux2014_armv7l.musllinux_1_1_armv7l.whl",
  "uv-0.13.2-py3-none-manylinux_2_17_armv7l.manylinux2014_armv7l.whl",
  "uv-0.13.2-py3-none-manylinux_2_17_i686.manylinux2014_i686.whl",
  "uv-0.13.2-py3-none-manylinux_2_17_ppc64le.manylinux2014_ppc64le.whl",
  "uv-0.13.2-py3-none-manylinux_2_17_s390x.manylinux2014_s390x.whl",
  "uv-0.13.2-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
  "uv-0.13.2-py3-none-manylinux_2_28_aarch64.whl",
  "uv-0.13.2-py3-none-manylinux_2_31_riscv64.musllinux_1_1_riscv64.whl",
  "uv-0.13.2-py3-none-manylinux_2_31_riscv64.whl",
  "uv-0.13.2-py3-none-musllinux_1_1_i686.whl",
  "uv-0.13.2-py3-none-musllinux_1_1_x86_64.whl",
  "uv-0.13.2-py3-none-win32.whl",
  "uv-0.13.2-py3-none-win_amd64.whl",
  "uv-0.13.2-py3-none-win_arm64.whl",
  "uv-0.13.2.tar.gz"
]
const FILES: WheelFile[] = FILENAMES.map((filename) => ({ filename, url: `https://files.example/${filename}`, digests: { sha256: '' } }))

const pick = (platform: string, arch: string, musl = false) =>
  pickUvWheel(FILES, platform, arch, musl).filename.replace(`uv-${UV_VERSION}-py3-none-`, '')

test('picks the wheel for each platform VS Code runs on', () => {
  assert.equal(pick('win32', 'x64'), 'win_amd64.whl')
  assert.equal(pick('win32', 'arm64'), 'win_arm64.whl')
  assert.equal(pick('darwin', 'x64'), 'macosx_10_12_x86_64.whl')
  assert.equal(pick('darwin', 'arm64'), 'macosx_11_0_arm64.whl')
  assert.equal(pick('linux', 'x64'), 'manylinux_2_17_x86_64.manylinux2014_x86_64.whl')
  assert.equal(pick('linux', 'x64', true), 'musllinux_1_1_x86_64.whl')
  // The lowest glibc baseline, for the oldest distributions.
  assert.equal(pick('linux', 'arm64'), 'manylinux_2_17_aarch64.manylinux2014_aarch64.musllinux_1_1_aarch64.whl')
  assert.equal(pick('linux', 'arm64', true), 'manylinux_2_17_aarch64.manylinux2014_aarch64.musllinux_1_1_aarch64.whl')
})

test('an unsupported platform fails clearly', () => {
  assert.throws(() => pickUvWheel(FILES, 'freebsd', 'x64'), /no build for freebsd\/x64/)
  assert.throws(() => pickUvWheel([], 'linux', 'x64'), /no build/)
})

// A minimal zip writer, to test the reader without fixtures.
function zip(entries: Array<{ name: string; data: Buffer; deflate: boolean }>): Buffer {
  const locals: Buffer[] = []
  const centrals: Buffer[] = []
  let offset = 0
  for (const { name, data, deflate } of entries) {
    const body = deflate ? deflateRawSync(data) : data
    const nameBuf = Buffer.from(name)
    const local = Buffer.alloc(30)
    local.writeUInt32LE(0x04034b50, 0)
    local.writeUInt16LE(deflate ? 8 : 0, 8)
    local.writeUInt32LE(body.length, 18)
    local.writeUInt32LE(data.length, 22)
    local.writeUInt16LE(nameBuf.length, 26)
    const central = Buffer.alloc(46)
    central.writeUInt32LE(0x02014b50, 0)
    central.writeUInt16LE(deflate ? 8 : 0, 10)
    central.writeUInt32LE(body.length, 20)
    central.writeUInt32LE(data.length, 24)
    central.writeUInt16LE(nameBuf.length, 28)
    central.writeUInt32LE(offset, 42)
    locals.push(local, nameBuf, body)
    centrals.push(central, nameBuf)
    offset += 30 + nameBuf.length + body.length
  }
  const dir = Buffer.concat(centrals)
  const eocd = Buffer.alloc(22)
  eocd.writeUInt32LE(0x06054b50, 0)
  eocd.writeUInt16LE(entries.length, 8)
  eocd.writeUInt16LE(entries.length, 10)
  eocd.writeUInt32LE(dir.length, 12)
  eocd.writeUInt32LE(offset, 16)
  return Buffer.concat([...locals, dir, eocd])
}

test('takes one file out of a zip, stored or deflated', () => {
  const binary = Buffer.from('\x7fELF'.repeat(5000))
  const z = zip([
    { name: 'uv/__init__.py', data: Buffer.from('# uv'), deflate: false },
    { name: `uv-${UV_VERSION}.data/scripts/uv`, data: binary, deflate: true },
  ])
  assert.deepEqual(extractEntry(z, `uv-${UV_VERSION}.data/scripts/uv`), binary)
  assert.equal(extractEntry(z, 'uv/__init__.py').toString(), '# uv')
  assert.throws(() => extractEntry(z, 'missing'), /not found/)
  assert.throws(() => extractEntry(Buffer.from('not a zip at all, no'), 'x'), /not a zip/)
})
