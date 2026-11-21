#!/usr/bin/env python3
"""news-image-20261121: one image summing up Saturday 21 November 2026.

Pure standard library. Reads the hero photograph (hero.png, shipped in the
repository), draws a newspaper-style front page around it -- masthead with
the date, a lead story, and four more stories -- quantizes the result to a
256-colour palette PNG and writes it to $MAKEFUN_OUTPUT_DIR/entry.png
(or /output/entry.png). Fully deterministic: two runs write identical bytes.
"""

import math
import os
import struct
import sys
import zlib

W, H = 1024, 1024

# ---------------------------------------------------------------- PNG I/O

def _chunk(ctype, data):
    out = ctype + data
    return struct.pack(">I", len(data)) + out + struct.pack(">I", zlib.crc32(out) & 0xFFFFFFFF)


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def decode_png(path):
    """Decode an 8-bit non-interlaced PNG. Returns (width, height, rgb bytes)."""
    raw = open(path, "rb").read()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    pos, width = 8, 0
    height = bitdepth = ctype = 0
    palette, idat = None, b""
    while pos < len(raw):
        size = struct.unpack(">I", raw[pos:pos + 4])[0]
        tag = raw[pos + 4:pos + 8]
        data = raw[pos + 8:pos + 8 + size]
        pos += 12 + size
        if tag == b"IHDR":
            width, height, bitdepth, ctype = struct.unpack(">IIBB", data[:10])
            assert bitdepth == 8, "only 8-bit PNGs supported"
        elif tag == b"PLTE":
            palette = data
        elif tag == b"IDAT":
            idat += data
        elif tag == b"IEND":
            break
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    stride = width * channels
    scan = zlib.decompress(idat)
    out = bytearray(width * height * 3)
    prev = bytearray(stride)
    p = 0
    for y in range(height):
        f = scan[p]
        p += 1
        line = bytearray(scan[p:p + stride])
        p += stride
        if f == 1:
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif f == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif f == 3:
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif f == 4:
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                line[i] = (line[i] + _paeth(a, b, c)) & 0xFF
        if ctype == 0:
            for x in range(width):
                g = line[x]
                o = (y * width + x) * 3
                out[o] = out[o + 1] = out[o + 2] = g
        elif ctype == 2:
            out[y * width * 3:(y + 1) * width * 3] = line
        elif ctype == 3:
            for x in range(width):
                idx = line[x] * 3
                o = (y * width + x) * 3
                out[o:o + 3] = palette[idx:idx + 3]
        elif ctype == 4:
            for x in range(width):
                g = line[x * 2]
                o = (y * width + x) * 3
                out[o] = out[o + 1] = out[o + 2] = g
        elif ctype == 6:
            for x in range(width):
                o = (y * width + x) * 3
                out[o:o + 3] = line[x * 4:x * 4 + 3]
        prev = line
    return width, height, bytes(out)


def encode_png_palette(rgb, width, height, palette):
    """Encode RGB bytes as a palette PNG (filter 0 rows, zlib level 9)."""
    lut = _build_lut(palette)
    body = bytearray()
    for y in range(height):
        body.append(0)
        base = y * width * 3
        for x in range(width):
            o = base + x * 3
            body.append(lut[((rgb[o] >> 4) << 8) | ((rgb[o + 1] >> 4) << 4) | (rgb[o + 2] >> 4)])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0)
    plte = bytes(palette)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"PLTE", plte)
            + _chunk(b"IDAT", zlib.compress(bytes(body), 9)) + _chunk(b"IEND", b""))


def _build_lut(palette):
    """Map every 4-bit-per-channel colour cube cell to nearest palette index."""
    n = len(palette) // 3
    lut = [0] * 4096
    for cell in range(4096):
        r = ((cell >> 8) & 0xF) * 17
        g = ((cell >> 4) & 0xF) * 17
        b = (cell & 0xF) * 17
        best, bestd = 0, 1 << 62
        for i in range(n):
            dr = r - palette[i * 3]
            dg = g - palette[i * 3 + 1]
            db = b - palette[i * 3 + 2]
            d = dr * dr + dg * dg + db * db
            if d < bestd:
                bestd, best = d, i
        lut[cell] = best
    return lut


def median_cut_palette(rgb, width, height, count=256):
    """Median-cut palette from sampled pixels (deterministic)."""
    px = []
    step = 3
    for y in range(0, height, step):
        base = y * width * 3
        for x in range(0, width, step):
            o = base + x * 3
            px.append((rgb[o], rgb[o + 1], rgb[o + 2]))
    boxes = [px]
    while len(boxes) < count:
        bi, best_range = -1, -1
        for i, box in enumerate(boxes):
            if len(box) < 2:
                continue
            for ch in range(3):
                vals = [p[ch] for p in box]
                r = max(vals) - min(vals)
                if r > best_range:
                    best_range, bi, bch = r, i, ch
        if bi < 0:
            break
        box = boxes.pop(bi)
        box.sort(key=lambda p: p[bch])
        mid = len(box) // 2
        boxes.append(box[:mid])
        boxes.append(box[mid:])
    pal = bytearray()
    for box in boxes:
        n = len(box)
        pal += bytes((sum(p[0] for p in box) // n,
                      sum(p[1] for p in box) // n,
                      sum(p[2] for p in box) // n))
    return bytes(pal)

# ---------------------------------------------------------------- 5x7 font
# Each glyph is 7 rows of 5 cells; '#' is ink. Only capitals, digits and
# punctuation are needed: all text is upper-cased before drawing.

FONT = {
    'A': ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    'B': ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    'C': ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    'D': ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    'E': ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    'F': ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    'G': ("01111", "10000", "10000", "10111", "10001", "10001", "01110"),
    'H': ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    'I': ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    'J': ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    'K': ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    'L': ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    'M': ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    'N': ("10001", "11001", "11001", "10101", "10011", "10011", "10001"),
    'O': ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    'P': ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    'Q': ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    'R': ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    'S': ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    'T': ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    'U': ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    'V': ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    'W': ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
    'X': ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    'Y': ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    'Z': ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    '0': ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    '1': ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    '2': ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    '3': ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    '4': ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    '5': ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    '6': ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    '7': ("11111", "00001", "00010", "00100", "00100", "00100", "00100"),
    '8': ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    '9': ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    ' ': ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    '.': ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ',': ("00000", "00000", "00000", "00000", "00100", "00100", "01000"),
    "'": ("00100", "00100", "00000", "00000", "00000", "00000", "00000"),
    '-': ("00000", "00000", "00000", "01110", "00000", "00000", "00000"),
    ':': ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    ';': ("00000", "01100", "01100", "00000", "01100", "01100", "01000"),
    '(': ("00011", "00100", "01000", "01000", "01000", "00100", "00011"),
    ')': ("11000", "00100", "00010", "00010", "00010", "00100", "11000"),
    '%': ("11001", "11010", "00010", "00100", "01000", "01011", "10011"),
    '+': ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    '/': ("00001", "00001", "00010", "00100", "01000", "10000", "10000"),
    '&': ("01100", "10010", "10010", "01000", "10101", "10010", "01101"),
    '!': ("00100", "00100", "00100", "00100", "00100", "00000", "00100"),
    '?': ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
    '"': ("01010", "01010", "00000", "00000", "00000", "00000", "00000"),
    '=': ("00000", "00000", "11111", "00000", "11111", "00000", "00000"),
}

CHAR_W = 6  # 5 px glyph + 1 px gap, times scale


def text_width(text, scale):
    return len(text) * CHAR_W * scale - scale


def draw_text(cv, x, y, text, scale, color):
    """Draw upper-case text left-aligned at (x, y); returns width drawn."""
    text = text.upper()
    cx = x
    for ch in text:
        glyph = FONT.get(ch, FONT[' '])
        for row in range(7):
            bits = glyph[row]
            for col in range(5):
                if bits[col] == '1':
                    fill_rect(cv, cx + col * scale, y + row * scale,
                              scale, scale, color)
        cx += CHAR_W * scale
    return cx - x - scale


def draw_text_centred(cv, cx, y, text, scale, color):
    w = text_width(text.upper(), scale)
    draw_text(cv, cx - w // 2, y, text, scale, color)


def wrap_text(text, max_chars):
    """Greedy word wrap; returns a list of lines, each <= max_chars."""
    words = text.upper().split(' ')
    lines, cur = [], ''
    for word in words:
        trial = word if not cur else cur + ' ' + word
        if len(trial) <= max_chars:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            while len(word) > max_chars:
                lines.append(word[:max_chars])
                word = word[max_chars:]
            cur = word
    if cur:
        lines.append(cur)
    return lines

# ---------------------------------------------------------------- canvas

def new_canvas(color):
    return bytearray(bytes(color) * (W * H))


def fill_rect(cv, x, y, w, h, color):
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    row = bytes(color) * (x1 - x0)
    for yy in range(y0, y1):
        o = (yy * W + x0) * 3
        cv[o:o + (x1 - x0) * 3] = row


def draw_line(cv, x0, y0, x1, y1, color, thick=1):
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        fill_rect(cv, x0 - thick // 2, y0 - thick // 2, thick, thick, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def fill_circle(cv, cx, cy, r, color):
    for dy in range(-r, r + 1):
        half = int((r * r - dy * dy) ** 0.5)
        fill_rect(cv, cx - half, cy + dy, half * 2 + 1, 1, color)


def fill_triangle(cv, x0, y0, x1, y1, x2, y2, color):
    ys = sorted(((x0, y0), (x1, y1), (x2, y2)), key=lambda p: p[1])
    (ax, ay), (bx, by), (cx, cy) = ys
    for y in range(ay, cy + 1):
        if y < by:
            t1 = (y - ay) / max(1, by - ay)
            xa = ax + (bx - ax) * t1
        else:
            t1 = (y - by) / max(1, cy - by)
            xa = bx + (cx - bx) * t1
        t2 = (y - ay) / max(1, cy - ay)
        xb = ax + (cx - ax) * t2
        fill_rect(cv, int(min(xa, xb)), y, int(abs(xb - xa)) + 1, 1, color)


def blit(cv, src, sw, sh, dx, dy, dw, dh, darken=0.0):
    """Nearest-neighbour scale of an RGB image onto the canvas, optionally
    darkened (darken 0..1) for a text band."""
    if sw == dw and sh == dh and not darken:
        for y in range(dh):
            o = ((dy + y) * W + dx) * 3
            cv[o:o + dw * 3] = src[y * sw * 3:(y + 1) * sw * 3]
        return
    for y in range(dh):
        sy = min(sh - 1, y * sh // dh)
        for x in range(dw):
            sx = min(sw - 1, x * sw // dw)
            o = (sy * sw + sx) * 3
            r, g, b = src[o], src[o + 1], src[o + 2]
            if darken:
                r = int(r * (1 - darken))
                g = int(g * (1 - darken))
                b = int(b * (1 - darken))
            oo = ((dy + y) * W + dx + x) * 3
            cv[oo] = r
            cv[oo + 1] = g
            cv[oo + 2] = b


# ---------------------------------------------------------------- icons

NAVY = (16, 28, 46)
NAVY2 = (24, 44, 70)
RED = (178, 34, 34)
GOLD = (214, 158, 46)
PAPER = (245, 241, 232)
PANEL_A = (253, 252, 248)
PANEL_B = (235, 230, 217)
INK = (24, 24, 24)
GREY = (92, 92, 92)
WHITE = (255, 255, 255)
BLUE = (30, 64, 140)


def icon_markets(cv, x, y):
    fill_rect(cv, x, y, 110, 110, (246, 236, 232))
    fill_rect(cv, x + 12, y + 88, 86, 3, NAVY)
    fill_rect(cv, x + 12, y + 14, 3, 77, NAVY)
    bars = [(20, 34), (38, 46), (56, 58), (74, 74)]
    for bx, bh in bars:
        fill_rect(cv, x + bx, y + 88 - bh, 12, bh, RED)
    draw_line(cv, x + 16, y + 30, x + 36, y + 44, RED, 3)
    draw_line(cv, x + 36, y + 44, x + 56, y + 58, RED, 3)
    draw_line(cv, x + 56, y + 58, x + 78, y + 74, RED, 3)
    draw_line(cv, x + 78, y + 74, x + 94, y + 82, RED, 3)


def icon_court(cv, x, y):
    fill_rect(cv, x, y, 110, 110, (236, 239, 244))
    fill_triangle(cv, x + 12, y + 34, x + 98, y + 34, x + 55, y + 12, NAVY2)
    fill_circle(cv, x + 55, y + 27, 5, GOLD)
    for cx in (22, 45, 68):
        fill_rect(cv, x + cx, y + 38, 12, 44, NAVY2)
    fill_rect(cv, x + 12, y + 82, 86, 6, NAVY2)
    fill_rect(cv, x + 8, y + 90, 94, 5, NAVY2)


def icon_quake(cv, x, y):
    fill_rect(cv, x, y, 110, 110, (238, 234, 226))
    fill_rect(cv, x + 16, y + 52, 26, 34, (120, 110, 96))
    fill_rect(cv, x + 46, y + 42, 26, 44, (140, 130, 116))
    fill_rect(cv, x + 76, y + 58, 20, 28, (120, 110, 96))
    draw_line(cv, x + 10, y + 88, x + 34, y + 78, (70, 60, 48), 4)
    draw_line(cv, x + 34, y + 78, x + 52, y + 92, (70, 60, 48), 4)
    draw_line(cv, x + 52, y + 92, x + 74, y + 80, (70, 60, 48), 4)
    draw_line(cv, x + 74, y + 80, x + 100, y + 90, (70, 60, 48), 4)
    pts = [(14, 26), (26, 26), (32, 14), (40, 30), (48, 18), (56, 28),
           (64, 16), (72, 28), (80, 20), (90, 26), (98, 24)]
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        draw_line(cv, x + ax, y + ay, x + bx, y + by, RED, 3)


def icon_flags(cv, x, y):
    fill_rect(cv, x, y, 110, 110, (236, 239, 244))
    # Union Jack, simplified: blue field, white saltire + cross, red cross.
    fx, fy = x + 8, y + 34
    fill_rect(cv, fx, fy, 44, 30, (18, 36, 98))
    draw_line(cv, fx, fy, fx + 44, fy + 30, WHITE, 5)
    draw_line(cv, fx + 44, fy, fx, fy + 30, WHITE, 5)
    fill_rect(cv, fx + 19, fy, 6, 30, WHITE)
    fill_rect(cv, fx, fy + 12, 44, 6, WHITE)
    fill_rect(cv, fx + 21, fy, 3, 30, RED)
    fill_rect(cv, fx, fy + 14, 44, 3, RED)
    # EU flag: blue field, ring of gold stars.
    ex, ey = x + 58, y + 34
    fill_rect(cv, ex, ey, 44, 30, (0, 51, 153))
    for i in range(8):
        ang = math.pi * 2 * i / 8 - math.pi / 2
        sx = int(ex + 22 + 11 * math.cos(ang))
        sy = int(ey + 15 + 9 * math.sin(ang))
        fill_circle(cv, sx, sy, 2, GOLD)
    fill_circle(cv, x + 55, y + 82, 9, GOLD)
    fill_rect(cv, x + 53, y + 76, 4, 12, NAVY)
    fill_rect(cv, x + 49, y + 80, 12, 4, NAVY)

# ---------------------------------------------------------------- layout

STORIES = [
    ("MARKETS IN FREEFALL",
     "S&P 500 and Dow enter correction as AI sell-off deepens. Nasdaq sheds 5.3%.",
     icon_markets),
    ("GENOCIDE VERDICT AT THE HAGUE",
     "UN court finds Myanmar responsible for genocide against the Rohingya - a first.",
     icon_court),
    ("QUAKE SHAKES PERU",
     "Magnitude 6.8 strikes Junin; tremors felt 320 km away in Lima.",
     icon_quake),
    ("BRITAIN AND EU INK LANDMARK DEAL",
     "Biggest post-Brexit pact: youth visas, food checks scrapped, Erasmus+ return.",
     icon_flags),
]


def draw_panel(cv, x, y, headline, caption, icon):
    bg = PANEL_A if ((x == 0) == (y < 800)) else PANEL_B
    fill_rect(cv, x, y, 512, 231, bg)
    icon(cv, x + 22, y + 22)
    lines = wrap_text(headline, 19)
    ty = y + 30
    for line in lines[:2]:
        draw_text(cv, x + 150, ty, line, 3, NAVY)
        ty += 26
    draw_line(cv, x + 150, y + 96, x + 488, y + 96, RED, 3)
    clines = wrap_text(caption, 38)
    cy = y + 142
    for line in clines[:3]:
        draw_text(cv, x + 24, cy, line, 2, INK)
        cy += 18
    draw_text(cv, x + 24, y + 204, "21 NOV 2026", 2, GREY)


def build():
    here = os.path.dirname(os.path.abspath(__file__))
    hw, hh, hero = decode_png(os.path.join(here, "hero.png"))
    cv = new_canvas(PAPER)

    # Masthead.
    fill_rect(cv, 0, 0, W, 116, NAVY)
    draw_text_centred(cv, W // 2, 16, "THE WORLD IN ONE IMAGE", 2, GOLD)
    draw_text_centred(cv, W // 2, 44, "SATURDAY 21 NOVEMBER 2026", 5, WHITE)
    fill_rect(cv, 24, 102, W - 48, 4, RED)

    # Lead photograph with headline band.
    blit(cv, hero, hw, hh, 0, 116, W, 442)
    fill_rect(cv, 24, 132, 128, 34, RED)
    draw_text(cv, 38, 141, "UKRAINE", 2, WHITE)
    band_y = 470
    dark = bytes(int(v * 0.35) for v in range(256))
    for yy in range(band_y, 558):
        o = yy * W * 3
        cv[o:o + W * 3] = bytes(cv[o:o + W * 3]).translate(dark)
    draw_text(cv, 24, 482, "RUSSIA'S BIGGEST STRIKE ON UKRAINE", 4, WHITE)
    draw_text(cv, 24, 522, "400+ DRONES AND 110 MISSILES - TWO LAND IN POLAND", 2, GOLD)

    # Divider and four-story grid.
    fill_rect(cv, 0, 558, W, 4, RED)
    positions = [(0, 562), (512, 562), (0, 793), (512, 793)]
    for (headline, caption, icon), (px, py) in zip(STORIES, positions):
        draw_panel(cv, px, py, headline, caption, icon)
    fill_rect(cv, 510, 562, 4, 462, NAVY)
    fill_rect(cv, 0, 791, W, 4, NAVY)

    palette = median_cut_palette(bytes(cv), W, H, 256)
    return encode_png_palette(bytes(cv), W, H, palette)


def main():
    data = build()
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "entry.png")
    with open(path, "wb") as f:
        f.write(data)
    print("wrote %s (%d bytes)" % (path, len(data)))


if __name__ == "__main__":
    main()
