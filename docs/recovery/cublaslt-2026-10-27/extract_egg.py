"""Recover include_str! egglog files from preserved Rust archive sections.

This reads archived ELF objects without executing the recovered code.
Section bytes are preserved; the manifest records their origin and SHA-256.
"""

import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = (
    Path(sys.argv[1]).resolve()
    if len(sys.argv) > 1
    else Path(__file__).resolve().parent
)
ANCHORS = {
    "approved_constructors.egg": b"; Complete production vocabulary.",
    "approved_rules.egg": b"; Production proof relations.",
    "approved_whitelist.egg": b"; Complete approved configurations.",
    "separate_destination_call.egg": b"; Independent C and D storage.",
    "accumulate_in_place_call.egg": b"; One read-modify-write C operand.",
    "strided_batch_rules.egg": b"; One explicit batch axis.",
    "element_stride_constructors.egg": b"(datatype ElementStride\n",
    "element_stride_rules.egg": b"; Relations preserve independently proved readings.",
    "element_stride_discovery.egg": b"; Separately scheduled experiment.",
}


def members(data):
    assert data.startswith(b"!<arch>\n")
    offset = 8
    while offset + 60 <= len(data):
        header = data[offset : offset + 60]
        size = int(header[48:58])
        name = header[:16].decode().strip()
        start = offset + 60
        yield name, data[start : start + size]
        offset = start + size + size % 2


def sections(data):
    if not data.startswith(b"\x7fELF\x02\x01"):
        return
    offset = struct.unpack_from("<Q", data, 40)[0]
    width, count, names_index = struct.unpack_from("<HHH", data, 58)
    entries = [
        struct.unpack_from("<IIQQQQIIQQ", data, offset + i * width)
        for i in range(count)
    ]
    names_entry = entries[names_index]
    names = data[names_entry[4] : names_entry[4] + names_entry[5]]
    for entry in entries:
        if entry[1] != 1 or not entry[5]:
            continue
        name = names[entry[0] :].split(b"\0", 1)[0].decode(errors="replace")
        if not name.startswith(".rodata"):
            continue
        yield name, data[entry[4] : entry[4] + entry[5]]


archives = sorted((ROOT / "artifacts").glob("*.rlib"))
if not archives:
    raise SystemExit(f"No preserved .rlib archives found in {ROOT / 'artifacts'}")

manifest = []
for archive in archives:
    destination = ROOT / "recovered-egg" / archive.stem
    destination.mkdir(parents=True, exist_ok=True)
    found = set()
    for member, body in members(archive.read_bytes()):
        for section, content in sections(body):
            for filename, anchor in ANCHORS.items():
                if not content.startswith(anchor) or b"\n" not in content:
                    continue
                digest = hashlib.sha256(content).hexdigest()
                key = (filename, digest)
                if key in found:
                    continue
                found.add(key)
                target = destination / filename
                if target.exists() and target.read_bytes() != content:
                    target = destination / (digest[:12] + "-" + filename)
                target.write_bytes(content)
                record = {
                    "archive": str(archive.relative_to(ROOT)),
                    "member": member,
                    "section": section,
                    "file": str(target.relative_to(ROOT)),
                    "bytes": len(content),
                    "sha256": digest,
                    "starts_with_anchor": content.startswith(anchor),
                    "ends_with_newline": content.endswith(b"\n"),
                    "contains_nul": b"\0" in content,
                }
                manifest.append(record)
                print(json.dumps(record), flush=True)
(ROOT / "egg-recovery-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
