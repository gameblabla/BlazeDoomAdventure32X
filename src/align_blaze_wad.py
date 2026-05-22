#!/usr/bin/env python3
import struct, sys

ALIGN = 4

def be16(b): return struct.unpack('>H', b)[0]
def be32(b): return struct.unpack('>I', b)[0]
def pbe16(x): return struct.pack('>H', x & 0xffff)
def pbe32(x): return struct.pack('>I', x & 0xffffffff)

def find_iwad(rom):
    for off in range(0, len(rom)-12, 4):
        if rom[off:off+4] in (b'IWAD', b'PWAD'):
            count = be32(rom[off+4:off+8])
            table = be32(rom[off+8:off+12])
            if 0 < count <= 2048 and 0x0c <= table < 0x400000 and off + table + count*16 <= len(rom):
                return off, count, table
    raise SystemExit('IWAD/PWAD header not found')

def rom_checksum(rom):
    # Genesis checksum covers 16-bit words from 0x200 to end-of-ROM.
    n = len(rom) & ~1
    s = 0
    for i in range(0x200, n, 2):
        s = (s + be16(rom[i:i+2])) & 0xffff
    return s

def parse_lumps(rom, iwad, count, table):
    lumps = []
    for i in range(count):
        e = iwad + table + i*16
        ptr = be32(rom[e:e+4])
        size = be32(rom[e+4:e+8])
        name = rom[e+8:e+16]
        lumps.append({'ptr': ptr, 'size': size, 'name': name})
    return lumps

def next_data_ptr(lumps, cur_ptr, table_ptr):
    candidates = [l['ptr'] for l in lumps if l['ptr'] > cur_ptr and l['ptr'] < table_ptr]
    return min(candidates) if candidates else table_ptr

def align_up(x, a=ALIGN):
    return (x + a - 1) & ~(a - 1)

def fix(in_path, out_path):
    rom = bytearray(open(in_path, 'rb').read())
    iwad, count, table_ptr = find_iwad(rom)
    lumps = parse_lumps(rom, iwad, count, table_ptr)

    # Header and code before the embedded WAD remain byte-identical.
    out = bytearray(rom[:iwad])
    out += rom[iwad:iwad+12]  # IWAD/PWAD, count, placeholder table pointer
    cur = 12
    new_entries = []
    copied_ranges = {}
    padding_added = 0

    for idx, lump in enumerate(lumps):
        ptr, size, name = lump['ptr'], lump['size'], lump['name']
        stored_len = 0
        if ptr and ptr < table_ptr and size:
            # For compressed map lumps, table size is uncompressed size. The actual stored byte length is
            # determined by the next distinct data pointer or by the directory pointer.
            npt = next_data_ptr(lumps, ptr, table_ptr)
            stored_len = max(0, npt - ptr)
        elif ptr and ptr < table_ptr and size == 0:
            stored_len = 0

        if stored_len > 0:
            aligned = align_up(cur)
            if aligned > cur:
                out += b'\xff' * (aligned - cur)
                padding_added += aligned - cur
                cur = aligned
            new_ptr = cur
            src0 = iwad + ptr
            src1 = src0 + stored_len
            out += rom[src0:src1]
            cur += stored_len
        else:
            # Zero-length labels/markers do not require storage. Use the current aligned location so any code
            # that still looks at the pointer won't receive an odd address.
            new_ptr = align_up(cur) if ptr else 0

        new_entries.append((new_ptr, size, name))

    table_new = align_up(cur)
    if table_new > cur:
        out += b'\xff' * (table_new - cur)
        padding_added += table_new - cur
        cur = table_new

    # Patch embedded WAD header table pointer.
    out[iwad+8:iwad+12] = pbe32(table_new)

    for ptr, size, name in new_entries:
        out += pbe32(ptr) + pbe32(size) + name

    if len(out) > len(rom):
        raise SystemExit(f'new ROM exceeds original size: {len(out)} > {len(rom)}')
    out += b'\xff' * (len(rom) - len(out))

    # Preserve ROM size/end fields and repair checksum.
    out[0x1a4:0x1a8] = pbe32(len(out)-1)
    out[0x18e:0x190] = b'\x00\x00'
    out[0x18e:0x190] = pbe16(rom_checksum(out))
    open(out_path, 'wb').write(out)

    # Report remaining misalignment, excluding ptr==0 markers.
    i2, c2, t2 = find_iwad(out)
    l2 = parse_lumps(out, i2, c2, t2)
    odd = [(i, e['name'], e['ptr'], e['size']) for i, e in enumerate(l2) if e['ptr'] and e['ptr'] % 2]
    mod4 = [(i, e['name'], e['ptr'], e['size']) for i, e in enumerate(l2) if e['ptr'] and e['ptr'] % 4]
    print(f'IWAD at 0x{iwad:x}; old table=0x{table_ptr:x}; new table=0x{table_new:x}')
    print(f'padding added: {padding_added} bytes')
    print(f'odd pointers remaining: {len(odd)}; non-4-byte pointers remaining: {len(mod4)}')
    print(f'checksum: 0x{be16(out[0x18e:0x190]):04x}')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('usage: align_blaze_wad.py in.32x out.32x')
    fix(sys.argv[1], sys.argv[2])
