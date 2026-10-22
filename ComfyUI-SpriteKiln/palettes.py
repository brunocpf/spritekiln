"""Built-in palettes plus any Lospec-style .hex files dropped into ./palettes/."""
import os

from .sk_core import parse_hex_palette

BUILTIN = {
    "PICO-8": "000000 1d2b53 7e2553 008751 ab5236 5f574f c2c3c7 fff1e8 ff004d ffa300 ffec27 00e436 29adff 83769c ff77a8 ffccaa",
    "Game Boy (DMG)": "0f380f 306230 8bac0f 9bbc0f",
    "Game Boy Pocket (grey)": "000000 555555 aaaaaa ffffff",
    "1-bit": "000000 ffffff",
    "CGA (mode 4, high)": "000000 55ffff ff55ff ffffff",
    "Commodore 64": "000000 ffffff 68372b 70a4b2 6f3d86 588d43 352879 b8c76f 6f4f25 433900 9a6759 444444 6c6c6c 9ad284 6c5eb5 959595",
    "Virtual Boy": "000000 550000 aa0000 ff0000",
    "Sweetie 16": "1a1c2c 5d275d b13e53 ef7d57 ffcd75 a7f070 38b764 257179 29366f 3b5dc9 41a6f6 73eff7 f4f4f4 94b0c2 566c86 333c57",
    "Endesga 16": "e4a672 b86f50 743f39 3f2832 9e2835 e53b44 fb922b ffe762 63c64d 327345 193d3f 4f6781 afbfd2 ffffff 2ce8f4 0484d1",
    "Endesga 32": "be4a2f d77643 ead4aa e4a672 b86f50 733e39 3e2731 a22633 e43b44 f77622 feae34 fee761 63c74d 3e8948 265c42 193c3e 124e89 0099db 2ce8f5 ffffff c0cbdc 8b9bb4 5a6988 3a4466 262b44 181425 ff0044 68386c b55088 f6757a e8b796 c28569",
    "SLSO8": "0d2b45 203c56 544e68 8d697a d08159 ffaa5e ffd4a3 ffecd6",
    "Nyx8": "08141e 0f2a3f 20394f f6d6bd c3a38a 997577 816271 4e495f",
    "Ammo-8": "040c06 112318 1e3a29 305d42 4d8061 89a257 bedc7f eeffcc",
    "Twilight 5": "fbbbad ee8695 4a7a96 333f58 292831",
    "Hollow": "0f0f1b 565a75 c6b7be fafbf6",
}

PALETTE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "palettes")


def all_palettes():
    pals = dict(BUILTIN)
    if os.path.isdir(PALETTE_DIR):
        for fn in sorted(os.listdir(PALETTE_DIR)):
            if fn.lower().endswith((".hex", ".txt", ".gpl")):
                try:
                    with open(os.path.join(PALETTE_DIR, fn), encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except OSError:
                    continue
                if fn.lower().endswith(".gpl"):  # GIMP palette: "R G B name" lines
                    rows = []
                    for line in text.splitlines():
                        parts = line.split()
                        if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
                            rows.append("%02x%02x%02x" % tuple(int(p) for p in parts[:3]))
                    text = " ".join(rows)
                pals[os.path.splitext(fn)[0]] = text
    return pals


def palette_names():
    return list(all_palettes().keys())


def get_palette(name):
    return parse_hex_palette(all_palettes().get(name, ""))
