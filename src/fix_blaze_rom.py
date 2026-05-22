from pathlib import Path

BASE = Path('dooma.32x')
TARGET = Path('blaze.32x')
OUT = Path('blaze_spritefix.32x')
CHECKSUM_OUT = Path('blaze_checksumfix_only.32x')

SPRITES_TO_RESTORE = [
    'SMBTA0','SMBTB0','SMBTC0','SMBTD0',
    'SMRTA0','SMRTB0','SMRTC0','SMRTD0',
]

def find_iwad(buf):
    found = None
    for i in range(0, len(buf) - 12, 4):
        if buf[i:i+4] in (b'IWAD', b'PWAD'):
            count = int.from_bytes(buf[i+4:i+8], 'big')
            table_ptr = int.from_bytes(buf[i+8:i+12], 'big')
            if 0 < count <= 2048 and 0x0c <= table_ptr < 0x400000:
                found = (i, count, table_ptr)
    if found is None:
        raise RuntimeError('No valid 32X WAD directory found')
    return found

def parse_lumps(buf):
    iwad, count, table_ptr = find_iwad(buf)
    lumps = []
    for idx in range(count):
        off = iwad + table_ptr + idx * 16
        ptr = int.from_bytes(buf[off:off+4], 'big')
        size = int.from_bytes(buf[off+4:off+8], 'big')
        name = buf[off+8:off+16].split(b'\0', 1)[0].decode('latin1')
        lumps.append({'idx': idx, 'ptr': ptr, 'size': size, 'name': name, 'diroff': off})
    return iwad, count, table_ptr, lumps

def fix_checksum(buf):
    calc = sum(int.from_bytes(buf[i:i+2], 'big') for i in range(0x200, len(buf), 2)) & 0xffff
    buf[0x18e:0x190] = calc.to_bytes(2, 'big')
    return calc

def checksum_status(buf):
    stored = int.from_bytes(buf[0x18e:0x190], 'big')
    calc = sum(int.from_bytes(buf[i:i+2], 'big') for i in range(0x200, len(buf), 2)) & 0xffff
    return stored, calc

def copy_lump_pair(src_buf, dst_buf, src_iwad, dst_iwad, src_lumps, dst_lumps, name):
    src = next(l for l in src_lumps if l['name'] == name)
    dst = next(l for l in dst_lumps if l['name'] == name)
    # 32X sprite uses two consecutive lumps: key/header named e.g. SMBTB0, then raw pixel data named ".".
    src_pair = [src, src_lumps[src['idx'] + 1]]
    dst_pair = [dst, dst_lumps[dst['idx'] + 1]]
    if src_pair[1]['name'] != '.' or dst_pair[1]['name'] != '.':
        raise RuntimeError(f'{name}: expected following raw sprite lump named .')
    for s, d in zip(src_pair, dst_pair):
        if s['size'] > d['size']:
            raise RuntimeError(f'{name}: source {s["name"]} lump is larger than target slot')
        # Copy stock compressed data into the existing Blaze slot.
        dst_buf[dst_iwad + d['ptr']:dst_iwad + d['ptr'] + s['size']] = src_buf[src_iwad + s['ptr']:src_iwad + s['ptr'] + s['size']]
        # Neutralize old trailing data so accidental over-read is less likely to decode stale Blaze data.
        if d['size'] > s['size']:
            dst_buf[dst_iwad + d['ptr'] + s['size']:dst_iwad + d['ptr'] + d['size']] = b'\xff' * (d['size'] - s['size'])
        # Update directory size, keep existing pointer so no directory relocation is needed.
        dst_buf[d['diroff'] + 4:d['diroff'] + 8] = s['size'].to_bytes(4, 'big')
        # Preserve existing lump name bytes.

# Checksum-only output.
target = bytearray(TARGET.read_bytes())
cs = fix_checksum(target)
CHECKSUM_OUT.write_bytes(target)
print(f'wrote {CHECKSUM_OUT} with checksum 0x{cs:04X}')

# Sprite-fix output.
base = BASE.read_bytes()
target = bytearray(TARGET.read_bytes())
base_iwad, _, _, base_lumps = parse_lumps(base)
target_iwad, _, _, target_lumps = parse_lumps(target)
for sprite in SPRITES_TO_RESTORE:
    copy_lump_pair(base, target, base_iwad, target_iwad, base_lumps, target_lumps, sprite)
cs = fix_checksum(target)
OUT.write_bytes(target)
stored, calc = checksum_status(target)
print(f'wrote {OUT} with checksum 0x{stored:04X}; computed 0x{calc:04X}')
