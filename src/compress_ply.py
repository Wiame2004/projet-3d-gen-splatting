import argparse
from pathlib import Path
import numpy as np
from plyfile import PlyData


def ply_to_splat(ply_path: Path, splat_path: Path) -> dict:
    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]

    # Positions (x, y, z)
    positions = np.stack(
        [vertex["x"], vertex["y"], vertex["z"]], axis=1
    ).astype(np.float32)

    # Opacité (sigmoid)
    opacity_raw = np.asarray(vertex["opacity"], dtype=np.float32)
    opacities = 1.0 / (1.0 + np.exp(-opacity_raw))

    # Couleur de base (SH DC f_dc_0..2 -> RGB)
    sh_c0 = 0.28209479177387814
    colors = np.stack(
        [
            0.5 + sh_c0 * np.asarray(vertex["f_dc_0"], dtype=np.float32),
            0.5 + sh_c0 * np.asarray(vertex["f_dc_1"], dtype=np.float32),
            0.5 + sh_c0 * np.asarray(vertex["f_dc_2"], dtype=np.float32),
        ],
        axis=1,
    )
    rgba = np.clip(
        np.stack([colors[:, 0], colors[:, 1], colors[:, 2], opacities], axis=1)
        * 255.0,
        0,
        255,
    ).astype(np.uint8)

    # Échelles (exp)
    scales = np.exp(
        np.stack(
            [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1
        ).astype(np.float32)
    )

    # Rotation (quaternion normalisé encodé sur uint8)
    quats = np.stack(
        [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]],
        axis=1,
    ).astype(np.float32)
    norms = np.linalg.norm(quats, axis=1, keepdims=True) + 1e-8
    quats_u8 = np.clip((quats / norms) * 128.0 + 128.0, 0, 255).astype(np.uint8)

    # Tri par volume décroissant (bonne pratique .splat)
    volumes = scales[:, 0] * scales[:, 1] * scales[:, 2]
    order = np.argsort(-volumes)

    positions = positions[order]
    scales = scales[order]
    rgba = rgba[order]
    quats_u8 = quats_u8[order]

    splat_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splat_path, "wb") as f:
        for i in range(len(positions)):
            f.write(positions[i].tobytes())  # 12 octets
            f.write(scales[i].tobytes())     # 12 octets
            f.write(rgba[i].tobytes())       # 4 octets
            f.write(quats_u8[i].tobytes())   # 4 octets

    orig_bytes = ply_path.stat().st_size
    comp_bytes = splat_path.stat().st_size
    reduction_pct = (1.0 - (comp_bytes / orig_bytes)) * 100.0

    return {
        "num_gaussians": int(len(positions)),
        "ply_size_mb": round(orig_bytes / (1024 * 1024), 2),
        "splat_size_mb": round(comp_bytes / (1024 * 1024), 2),
        "size_reduction_pct": round(reduction_pct, 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Chemin vers point_cloud.ply")
    parser.add_argument("--output", required=True, help="Chemin de sortie model.splat")
    args = parser.parse_args()

    stats = ply_to_splat(Path(args.input), Path(args.output))
    print("Résultats de compression :", stats)