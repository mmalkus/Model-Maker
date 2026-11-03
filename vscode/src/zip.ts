import { inflateRawSync } from 'node:zlib'

// Just enough of the zip format to take one file out of a wheel (the uv
// binary, see uv.ts): the central directory to find it, its local header to
// skip, and stored or deflated data. No zip64 -- a wheel is far below 4 GB.

const EOCD = 0x06054b50
const CENTRAL = 0x02014b50
const LOCAL = 0x04034b50

export function extractEntry(zip: Buffer, name: string): Buffer {
  // The end-of-central-directory record sits in the last 22 bytes, plus an
  // optional comment of up to 64 KiB.
  let eocd = -1
  for (let i = zip.length - 22; i >= Math.max(0, zip.length - 22 - 0xffff); i--) {
    if (zip.readUInt32LE(i) === EOCD) {
      eocd = i
      break
    }
  }
  if (eocd < 0) throw new Error('not a zip file')
  const count = zip.readUInt16LE(eocd + 10)
  let at = zip.readUInt32LE(eocd + 16)
  for (let n = 0; n < count; n++) {
    if (zip.readUInt32LE(at) !== CENTRAL) throw new Error('corrupt zip central directory')
    const method = zip.readUInt16LE(at + 10)
    const compressedSize = zip.readUInt32LE(at + 20)
    const size = zip.readUInt32LE(at + 24)
    const nameLength = zip.readUInt16LE(at + 28)
    const extraLength = zip.readUInt16LE(at + 30)
    const commentLength = zip.readUInt16LE(at + 32)
    const localOffset = zip.readUInt32LE(at + 42)
    const entryName = zip.toString('utf8', at + 46, at + 46 + nameLength)
    if (entryName === name) {
      if (zip.readUInt32LE(localOffset) !== LOCAL) throw new Error('corrupt zip entry')
      const start = localOffset + 30 + zip.readUInt16LE(localOffset + 26) + zip.readUInt16LE(localOffset + 28)
      const raw = zip.subarray(start, start + compressedSize)
      const data = method === 0 ? Buffer.from(raw) : method === 8 ? inflateRawSync(raw) : null
      if (!data) throw new Error(`unsupported zip compression method ${method}`)
      if (data.length !== size) throw new Error(`${name}: expected ${size} bytes, got ${data.length}`)
      return data
    }
    at += 46 + nameLength + extraLength + commentLength
  }
  throw new Error(`${name} not found in the zip`)
}
