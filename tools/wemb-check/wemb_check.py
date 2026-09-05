#!/usr/bin/env python3
"""Offline checker for the int8 embedding tables in a marian binary model.

Two jobs, both about the D1/D2 change (Wemb stays int8 in memory):

  items   print the item table (name / type / shape / data length) so it is
          obvious which items carry "Wemb" in their name -- the real tables and
          the 1x1 "<name>_QuantMultA" alpha items, which look like tables but
          are not.

  requant emulate, in float32, the round trip the FP32 output-layer path does:

              r      = 1 / q                       (unquantizeWemb)
              fp[i]  = int8[i] * r
              maxabs = max |fp|                    (MaxAbsolute)
              q'     = 127 / maxabs                (QuantMultRuyNodeOp)
              out[i] = sat_int8(rint_even(fp[i] * q'))   (NEON quantize)

          and report whether q' == q bit for bit and whether out == int8 for
          every element. Needs numpy for the elementwise part; without it the
          command still prints the multipliers.

The engine carries the same check against its *production* kernels behind
BERGAMOT_WEMB_CHECK=1; this script is the independent second opinion and needs
no build.

Usage:
  python3 tools/wemb-check/wemb_check.py items   <model.bin> [<model.bin> ...]
  python3 tools/wemb-check/wemb_check.py requant <model.bin> [<model.bin> ...]
"""

import struct
import sys

TYPES = {
    0x0101: "int8", 0x0102: "int16", 0x0104: "int32", 0x0108: "int64",
    0x0201: "uint8", 0x0202: "uint16", 0x0204: "uint32", 0x0208: "uint64",
    0x0402: "float16", 0x0404: "float32", 0x0408: "float64",
    0x0802: "packed16", 0x1801: "packed8avx2", 0x2801: "packed8avx512",
    0x4101: "intgemm8", 0x4102: "intgemm16",
}
SIZEOF = {"int8": 1, "int16": 2, "int32": 4, "int64": 8, "uint8": 1, "uint16": 2,
          "uint32": 4, "uint64": 8, "float16": 2, "float32": 4, "float64": 8,
          "intgemm8": 1, "intgemm16": 2}


def load_items(path):
    """Mirror of marian::io::binary::loadItems, without any of the conversions."""
    with open(path, "rb") as handle:
        buf = handle.read()
    off = 0
    version, = struct.unpack_from("<Q", buf, off); off += 8
    count, = struct.unpack_from("<Q", buf, off); off += 8
    headers = []
    for _ in range(count):
        headers.append(struct.unpack_from("<QQQQ", buf, off)); off += 32
    names = []
    for name_len, _, _, _ in headers:
        names.append(buf[off:off + name_len - 1].decode("utf-8")); off += name_len
    shapes = []
    for _, _, shape_len, _ in headers:
        shapes.append(list(struct.unpack_from("<" + "i" * shape_len, buf, off)))
        off += 4 * shape_len
    padding, = struct.unpack_from("<Q", buf, off); off += 8 + padding
    items = []
    for i, (_, type_id, _, data_len) in enumerate(headers):
        items.append({
            "name": names[i],
            "type": TYPES.get(type_id, hex(type_id)),
            "shape": shapes[i],
            "data": buf[off:off + data_len],
        })
        off += data_len
    return version, items


def elements(item):
    total = 1
    for dim in item["shape"]:
        total *= dim
    return total


def tail_quant_mult(item):
    """The float marian appends after the last int8 of an intgemm tensor."""
    offset = elements(item) * SIZEOF[item["type"]]
    value, = struct.unpack_from("<f", item["data"], offset)
    bits, = struct.unpack_from("<I", item["data"], offset)
    return value, bits


def cmd_items(paths):
    for path in paths:
        version, items = load_items(path)
        print("=" * 104)
        print("%s  (binary version %d, %d items)" % (path, version, len(items)))
        for item in items:
            size = SIZEOF.get(item["type"])
            count = elements(item)
            extra = len(item["data"]) - count * size if size else -1
            marker = " <-- Wemb" if "Wemb" in item["name"] else ""
            print("%-46s %-10s %-16s data=%-10d elems=%-9d pad=%d%s"
                  % (item["name"], item["type"], item["shape"], len(item["data"]),
                     count, extra, marker))


def cmd_requant(paths):
    try:
        import numpy as np
    except ImportError:
        np = None
        print("numpy not available: printing multipliers only", file=sys.stderr)

    for path in paths:
        _, items = load_items(path)
        print("=" * 104)
        print(path)
        for item in items:
            if "Wemb" not in item["name"] or item["type"] != "intgemm8":
                continue
            count = elements(item)
            q, q_bits = tail_quant_mult(item)
            if count <= 1:
                print("%-16s shape=%-14s alpha item, dequantised value = %.17g"
                      % (item["name"], item["shape"],
                         struct.unpack_from("<b", item["data"], 0)[0] * (1.0 / q)))
                continue
            if np is None:
                print("%-16s elems=%-9d q=%.17g (0x%08x)" % (item["name"], count, q, q_bits))
                continue

            f32 = np.float32
            orig = np.frombuffer(item["data"][:count], dtype=np.int8)
            r = f32(1.0) / f32(q)
            fp = orig.astype(np.float32) * r
            maxabs = np.max(np.abs(fp))
            q_prime = f32(127.0) / f32(maxabs)
            q_prime_bits = struct.unpack("<I", struct.pack("<f", float(q_prime)))[0]
            requant = np.clip(np.rint((fp * q_prime).astype(np.float32)), -128, 127).astype(np.int8)
            bad = int((requant != orig).sum())
            print("%-16s elems=%-9d max|int8|=%-4d q=%.17g (0x%08x) q'=%.17g (0x%08x) "
                  "q_bits_equal=%s requant_mismatches=%d"
                  % (item["name"], count, int(np.abs(orig.astype(np.int32)).max()),
                     q, q_bits, float(q_prime), q_prime_bits,
                     q_bits == q_prime_bits, bad))


def main(argv):
    if len(argv) < 3 or argv[1] not in ("items", "requant"):
        print(__doc__)
        return 2
    (cmd_items if argv[1] == "items" else cmd_requant)(argv[2:])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
