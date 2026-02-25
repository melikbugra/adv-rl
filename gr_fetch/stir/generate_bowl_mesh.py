import numpy as np
import struct
import os

OUT = "/home/melik/Gymnasium-Robotics/gymnasium_robotics/envs/assets/stls/fetch/stir_bowl.stl"

R_OUT     = 0.135
R_IN_BOT  = 0.085
R_IN_TOP  = 0.110
Z_BOT     = -0.010
Z_BASE    =  0.010
Z_TOP     =  0.150
N         = 48

angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
tris = []


def quad(v0, v1, v2, v3):
    tris.extend([(v0, v1, v2), (v0, v2, v3)])


def tri(v0, v1, v2):
    tris.append((v0, v1, v2))


def rin(z):
    return R_IN_BOT + (R_IN_TOP - R_IN_BOT) * (z - Z_BASE) / (Z_TOP - Z_BASE)


for i in range(N):
    a0, a1 = angles[i], angles[(i + 1) % N]
    c0, s0 = np.cos(a0), np.sin(a0)
    c1, s1 = np.cos(a1), np.sin(a1)

    # Outer wall (normals outward)
    quad(
        [R_OUT * c0, R_OUT * s0, Z_BOT],
        [R_OUT * c1, R_OUT * s1, Z_BOT],
        [R_OUT * c1, R_OUT * s1, Z_TOP],
        [R_OUT * c0, R_OUT * s0, Z_TOP],
    )

    # Inner wall (normals inward — reverse winding)
    ri_base = rin(Z_BASE)
    ri_top  = rin(Z_TOP)
    quad(
        [ri_top * c1, ri_top * s1, Z_TOP],
        [ri_base * c1, ri_base * s1, Z_BASE],
        [ri_base * c0, ri_base * s0, Z_BASE],
        [ri_top * c0, ri_top * s0, Z_TOP],
    )

    # Bottom disk (normals down)
    tri(
        [0.0, 0.0, Z_BOT],
        [R_OUT * c1, R_OUT * s1, Z_BOT],
        [R_OUT * c0, R_OUT * s0, Z_BOT],
    )

    # Base annulus top (normals up)
    quad(
        [R_IN_BOT * c0, R_IN_BOT * s0, Z_BASE],
        [R_IN_BOT * c1, R_IN_BOT * s1, Z_BASE],
        [R_OUT * c1, R_OUT * s1, Z_BASE],
        [R_OUT * c0, R_OUT * s0, Z_BASE],
    )

    # Top rim (normals up)
    quad(
        [R_OUT * c0, R_OUT * s0, Z_TOP],
        [R_OUT * c1, R_OUT * s1, Z_TOP],
        [R_IN_TOP * c1, R_IN_TOP * s1, Z_TOP],
        [R_IN_TOP * c0, R_IN_TOP * s0, Z_TOP],
    )

os.makedirs(os.path.dirname(OUT), exist_ok=True)

with open(OUT, "wb") as f:
    header = b"stir_bowl" + b"\x00" * 71
    f.write(header)
    f.write(struct.pack("<I", len(tris)))
    for v0, v1, v2 in tris:
        v0 = np.array(v0, dtype=np.float32)
        v1 = np.array(v1, dtype=np.float32)
        v2 = np.array(v2, dtype=np.float32)
        n = np.cross(v1 - v0, v2 - v0)
        nl = np.linalg.norm(n)
        if nl > 0:
            n = n / nl
        n = n.astype(np.float32)
        f.write(struct.pack("<fff", *n))
        f.write(v0.tobytes())
        f.write(v1.tobytes())
        f.write(v2.tobytes())
        f.write(struct.pack("<H", 0))

print(f"Wrote {len(tris)} triangles to {OUT}")
