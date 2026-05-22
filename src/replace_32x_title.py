#!/usr/bin/env python3
"""
Replace the Sega 32X DOOM startup title image.

This targets the original 32X title-loader asset, not the embedded IWAD
TITLEPIC lump. The game stores a 320x200 8-bit indexed image compressed with
a small custom RLE/LZ back-reference format. The image is displayed with a
12-pixel black border above and below on a 320x224 NTSC frame.

Default usage:
    python3 replace_32x_title.py blaze_aligned.32x titlescreen.png blaze_titlepatched.32x

The converter uses the ROM's existing 32X palette at 0x16f38 so later title/menu
screens keep their expected colors. It patches the compressed image block at
0x17138 and fixes the Genesis header checksum at 0x18e.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Pillow is required: pip install pillow") from exc

WIDTH = 320
STORED_HEIGHT = 200
DISPLAY_HEIGHT = 224
DISPLAY_Y_OFFSET = 12

# Locations in the unmodified DOOM 32X / Blaze ROM family.
PAL_OFFSET_DEFAULT = 0x16F38       # 256 big-endian RGB555 words, copied to 32X CRAM
TITLE_OFFSET_DEFAULT = 0x17138     # compressed title image block
OLD_COMPRESSED_LEN_DEFAULT = 0x7604 # consumed by original decoder block, including 4-byte size
CHECKSUM_OFFSET = 0x18E


def rgb555_to_rgb888(word: int) -> Tuple[int, int, int]:
    # 32X CRAM layout used here: bit 0..4 R, 5..9 G, 10..14 B.
    def sc(v: int) -> int:
        return (v << 3) | (v >> 2)
    return sc(word & 31), sc((word >> 5) & 31), sc((word >> 10) & 31)


def rgb888_to_rgb555(rgb: Tuple[int, int, int]) -> int:
    r, g, b = rgb
    return (r >> 3) | ((g >> 3) << 5) | ((b >> 3) << 10)


def read_rom_palette(rom: bytes, pal_offset: int) -> List[Tuple[int, int, int]]:
    if pal_offset < 0 or pal_offset + 512 > len(rom):
        raise ValueError(f"palette offset 0x{pal_offset:x} is outside the ROM")
    words = struct.unpack(">256H", rom[pal_offset:pal_offset + 512])
    return [rgb555_to_rgb888(w) for w in words]


def make_pillow_palette(colors: Sequence[Tuple[int, int, int]]) -> Image.Image:
    pal_img = Image.new("P", (1, 1))
    flat: List[int] = []
    for r, g, b in colors:
        flat.extend([r, g, b])
    flat.extend([0] * (768 - len(flat)))
    pal_img.putpalette(flat)
    return pal_img


def fit_image(img: Image.Image, mode: str) -> Image.Image:
    """Return RGB 320x200 title payload image."""
    img = img.convert("RGB")
    if mode == "stretch":
        return img.resize((WIDTH, STORED_HEIGHT), Image.Resampling.LANCZOS)

    src_w, src_h = img.size
    if src_w <= 0 or src_h <= 0:
        raise ValueError("input image has invalid dimensions")

    if mode == "contain":
        scale = min(WIDTH / src_w, STORED_HEIGHT / src_h)
        new_w = max(1, round(src_w * scale))
        new_h = max(1, round(src_h * scale))
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        out = Image.new("RGB", (WIDTH, STORED_HEIGHT), (0, 0, 0))
        out.paste(resized, ((WIDTH - new_w) // 2, (STORED_HEIGHT - new_h) // 2))
        return out

    if mode == "cover":
        scale = max(WIDTH / src_w, STORED_HEIGHT / src_h)
        new_w = max(1, round(src_w * scale))
        new_h = max(1, round(src_h * scale))
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (new_w - WIDTH) // 2
        top = (new_h - STORED_HEIGHT) // 2
        return resized.crop((left, top, left + WIDTH, top + STORED_HEIGHT))

    raise ValueError(f"unknown fit mode: {mode}")


def quantize_to_palette(img: Image.Image, palette: Sequence[Tuple[int, int, int]], dither: bool) -> Image.Image:
    pal_img = make_pillow_palette(palette)
    return img.quantize(palette=pal_img, dither=Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE)


def make_preview(indexed_payload: Image.Image, palette: Sequence[Tuple[int, int, int]]) -> Image.Image:
    out = Image.new("P", (WIDTH, DISPLAY_HEIGHT), 0)
    out.putpalette(make_pillow_palette(palette).getpalette())
    out.paste(indexed_payload, (0, DISPLAY_Y_OFFSET))
    return out.convert("RGB")


def decompress_title_block(comp: bytes) -> bytes:
    """Decoder matching the original 68k routine at ROM offset 0x25e4."""
    if len(comp) < 4:
        raise ValueError("compressed block is too short")
    out_size = struct.unpack(">I", comp[:4])[0]
    ci = 4
    out = bytearray()
    bits_left = 1
    key = 0
    while len(out) < out_size:
        bits_left -= 1
        if bits_left == 0:
            if ci >= len(comp):
                raise ValueError("compressed stream ended while reading a flag byte")
            key = comp[ci]
            ci += 1
            bits_left = 8
        carry = bool(key & 0x80)
        key = (key << 1) & 0xFF
        if not carry:
            if ci >= len(comp):
                raise ValueError("compressed stream ended while reading a literal")
            out.append(comp[ci])
            ci += 1
            continue
        if ci >= len(comp):
            raise ValueError("compressed stream ended while reading a command")
        d3 = comp[ci]
        ci += 1
        if d3 < 0x80:
            if ci >= len(comp):
                raise ValueError("compressed stream ended while reading an RLE value")
            count = (d3 & 0x7F) + 3
            value = comp[ci]
            ci += 1
            out.extend([value] * min(count, out_size - len(out)))
        else:
            if ci >= len(comp):
                raise ValueError("compressed stream ended while reading an LZ offset")
            d4 = comp[ci]
            ci += 1
            offset = ((d4 << 3) | ((d3 >> 4) & 7)) & 0x7FF
            count = (d3 & 0x0F) + 3
            src = len(out) - offset
            if offset <= 0 or src < 0:
                raise ValueError(f"invalid LZ offset {offset} at output 0x{len(out):x}")
            for _ in range(count):
                if len(out) >= out_size:
                    break
                out.append(out[src])
                src += 1
    return bytes(out)


def emit_tokens(tokens: Iterable[Tuple[str, int, int | None]]) -> bytes:
    """Pack tokens into the original MSB-first flag-byte layout."""
    out = bytearray()
    key_pos = None
    key = 0
    bit_count = 0

    def start_key_if_needed() -> None:
        nonlocal key_pos, key, bit_count
        if key_pos is None or bit_count == 8:
            if key_pos is not None:
                out[key_pos] = key
            key_pos = len(out)
            out.append(0)
            key = 0
            bit_count = 0

    def put_bit(bit: int) -> None:
        nonlocal key, bit_count
        start_key_if_needed()
        if bit:
            key |= 0x80 >> bit_count
        bit_count += 1

    for kind, a, b in tokens:
        if kind == "lit":
            put_bit(0)
            out.append(a & 0xFF)
        elif kind == "rle":
            count = a
            value = b
            if value is None or not (3 <= count <= 130):
                raise ValueError("invalid RLE token")
            put_bit(1)
            out.append((count - 3) & 0x7F)
            out.append(value & 0xFF)
        elif kind == "lz":
            offset = a
            count = b
            if count is None or not (1 <= offset <= 0x7FF) or not (3 <= count <= 18):
                raise ValueError("invalid LZ token")
            put_bit(1)
            out.append(0x80 | ((offset & 7) << 4) | (count - 3))
            out.append((offset >> 3) & 0xFF)
        else:
            raise ValueError(f"unknown token kind {kind}")

    if key_pos is not None:
        out[key_pos] = key
    return bytes(out)


def find_lz_match(data: bytes, pos: int, max_offset: int = 0x7FF, max_len: int = 18) -> Tuple[int, int]:
    end = min(len(data), pos + max_len)
    best_len = 0
    best_off = 0
    if pos >= len(data):
        return 0, 0
    window_start = max(0, pos - max_offset)
    # A simple bounded search is fast enough for 64 KiB and gives good compression.
    first = data[pos]
    candidates = []
    search = data[window_start:pos]
    rel = search.rfind(bytes([first]))
    while rel != -1 and len(candidates) < 96:
        cand = window_start + rel
        candidates.append(cand)
        rel = search.rfind(bytes([first]), 0, rel)
    for cand in candidates:
        ln = 0
        while pos + ln < end and data[cand + ln] == data[pos + ln]:
            # Overlapping copies are valid on the original decoder. Direct comparison
            # against the final payload is sufficient because the target bytes must
            # be periodic whenever the source runs into the region being produced.
            ln += 1
        if ln > best_len:
            best_len = ln
            best_off = pos - cand
            if ln == max_len:
                break
    if best_len < 3:
        return 0, 0
    return best_off, best_len


def compress_title_payload(payload: bytes) -> bytes:
    if len(payload) != WIDTH * STORED_HEIGHT:
        raise ValueError(f"payload must be {WIDTH * STORED_HEIGHT} bytes, got {len(payload)}")
    tokens: List[Tuple[str, int, int | None]] = []
    i = 0
    n = len(payload)
    while i < n:
        # RLE command: best for long runs and cheaper/equal for runs >= 3.
        run = 1
        while i + run < n and payload[i + run] == payload[i] and run < 130:
            run += 1
        if run >= 3:
            tokens.append(("rle", run, payload[i]))
            i += run
            continue

        off, ln = find_lz_match(payload, i)
        if ln >= 3:
            tokens.append(("lz", off, ln))
            i += ln
            continue

        tokens.append(("lit", payload[i], None))
        i += 1
    return struct.pack(">I", len(payload)) + emit_tokens(tokens)


def md_checksum(rom: bytes) -> int:
    s = 0
    end = len(rom) & ~1
    for i in range(0x200, end, 2):
        s = (s + ((rom[i] << 8) | rom[i + 1])) & 0xFFFF
    return s


def patch_checksum(rom: bytearray) -> int:
    rom[CHECKSUM_OFFSET:CHECKSUM_OFFSET + 2] = b"\x00\x00"
    chk = md_checksum(rom)
    rom[CHECKSUM_OFFSET:CHECKSUM_OFFSET + 2] = struct.pack(">H", chk)
    return chk


def make_ips(original: bytes, modified: bytes) -> bytes:
    if len(original) != len(modified):
        raise ValueError("IPS maker requires same-size files")
    out = bytearray(b"PATCH")
    i = 0
    n = len(original)
    while i < n:
        if original[i] == modified[i]:
            i += 1
            continue
        start = i
        chunk = bytearray()
        while i < n and original[i] != modified[i] and len(chunk) < 0xFFFF:
            chunk.append(modified[i])
            i += 1
        out += start.to_bytes(3, "big")
        out += len(chunk).to_bytes(2, "big")
        out += chunk
    out += b"EOF"
    return bytes(out)


def patch_rom(args: argparse.Namespace) -> None:
    rom_path = Path(args.rom)
    png_path = Path(args.png)
    out_path = Path(args.output)
    rom = bytearray(rom_path.read_bytes())
    original = bytes(rom)

    palette = read_rom_palette(rom, args.palette_offset)
    fitted = fit_image(Image.open(png_path), args.fit)
    indexed = quantize_to_palette(fitted, palette, args.dither)
    payload = indexed.tobytes()
    comp = compress_title_payload(payload)

    # Verify compressor before writing.
    decoded = decompress_title_block(comp)
    if decoded != payload:
        raise RuntimeError("internal compressor verification failed")

    if len(comp) > args.max_len:
        raise RuntimeError(
            f"new compressed title is too large: 0x{len(comp):x} bytes > slot 0x{args.max_len:x}. "
            "Try --fit contain, avoid dithering, or use a flatter image."
        )

    start = args.title_offset
    end = start + args.max_len
    if start < 0 or end > len(rom):
        raise ValueError("title block range is outside the ROM")
    rom[start:start + len(comp)] = comp
    # The old decoder stops after the declared output size, but filling the unused tail
    # with 0xff makes the patch deterministic and avoids stale data confusion.
    rom[start + len(comp):end] = b"\xFF" * (args.max_len - len(comp))

    chk = patch_checksum(rom)
    out_path.write_bytes(rom)

    preview_path = Path(args.preview) if args.preview else out_path.with_suffix(out_path.suffix + ".preview.png")
    make_preview(indexed, palette).save(preview_path)

    if args.ips:
        Path(args.ips).write_bytes(make_ips(original, bytes(rom)))

    print(f"input ROM:          {rom_path}")
    print(f"input image:        {png_path}")
    print(f"output ROM:         {out_path}")
    print(f"preview PNG:        {preview_path}")
    if args.ips:
        print(f"IPS patch:          {args.ips}")
    print(f"title offset:       0x{start:x}")
    print(f"palette offset:     0x{args.palette_offset:x}")
    print(f"compressed size:    0x{len(comp):x} / 0x{args.max_len:x}")
    print(f"header checksum:    0x{chk:04x}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replace the startup title image in Sega 32X DOOM-family ROMs.")
    ap.add_argument("rom", help="input .32x ROM")
    ap.add_argument("png", help="replacement image, normally 320x224 PNG")
    ap.add_argument("output", help="output patched .32x ROM")
    ap.add_argument("--fit", choices=["contain", "cover", "stretch"], default="contain",
                    help="how to fit the source image into the stored 320x200 title payload; default: contain")
    ap.add_argument("--dither", action="store_true", help="use Floyd-Steinberg dithering when quantizing to the ROM palette")
    ap.add_argument("--title-offset", type=lambda s: int(s, 0), default=TITLE_OFFSET_DEFAULT)
    ap.add_argument("--palette-offset", type=lambda s: int(s, 0), default=PAL_OFFSET_DEFAULT)
    ap.add_argument("--max-len", type=lambda s: int(s, 0), default=OLD_COMPRESSED_LEN_DEFAULT,
                    help="available compressed title slot length; default is the original consumed block length")
    ap.add_argument("--preview", help="where to write a 320x224 preview PNG")
    ap.add_argument("--ips", help="optional IPS patch path against the input ROM")
    args = ap.parse_args(argv)
    patch_rom(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
