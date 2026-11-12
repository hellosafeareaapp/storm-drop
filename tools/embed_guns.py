"""Embed models/guns/*.glb into index.html as base64, so the game stays a single
self-contained file that works when opened straight from disk.

Usage:  python tools/embed_guns.py
Re-run it whenever the gun models are regenerated with tools/make_guns.py.
"""
import base64
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
GUNS = ["ar", "shotgun", "smg", "sniper", "rocket"]

data = {g: base64.b64encode((ROOT / "models" / "guns" / f"{g}.glb").read_bytes()).decode("ascii") for g in GUNS}
block = "<script id=\"gun-data\">window.GUN_GLB = " + json.dumps(data) + ";</script>"

html_path = ROOT / "index.html"
html = html_path.read_text(encoding="utf-8")
pattern = re.compile(r'<script id="gun-data">.*?</script>', re.S)
if not pattern.search(html):
    raise SystemExit('index.html has no <script id="gun-data"> placeholder')
html = pattern.sub(lambda _: block, html, count=1)
html_path.write_text(html, encoding="utf-8", newline="\n")
print(f"embedded {len(GUNS)} guns, {sum(len(v) for v in data.values()) / 1e6:.2f} MB of base64")
