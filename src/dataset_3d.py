from pathlib import Path
import json
import numpy as np
import trimesh


def mesh_to_normalized_pointcloud(mesh_path: Path, num_points: int = 2048) -> np.ndarray:
    """Échantillonne num_points sur la surface d'un maillage et normalise dans la sphère unité."""
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    points = points.astype(np.float32)

    # Centrage à l'origine
    points -= points.mean(axis=0, keepdims=True)

    # Mise à l'échelle dans la sphère unité
    max_norm = np.linalg.norm(points, axis=1).max()
    if max_norm > 0:
        points /= max_norm

    return points


def prepare_dataset(
    raw_dir: str = "data",
    out_dir: str = "data/processed",
    categories: list[str] | None = None,
    num_points: int = 2048,
    max_total_objects: int = 2000,
    seed: int = 42,
):
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if categories is None:
        categories = ["bed", "chair", "sofa", "table"]

    rng = np.random.default_rng(seed)
    samples = []

    for cat in categories:
        cat_dir = raw_path / cat
        if not cat_dir.exists():
            continue
        files = sorted(
            list(cat_dir.rglob("*.off"))
            + list(cat_dir.rglob("*.obj"))
            + list(cat_dir.rglob("*.ply"))
        )
        for f in files:
            samples.append((cat, f))

    rng.shuffle(samples)
    samples = samples[:max_total_objects]

    all_points = []
    all_labels = []
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}

    for cat, file_path in samples:
        try:
            pts = mesh_to_normalized_pointcloud(file_path, num_points=num_points)
            all_points.append(pts)
            all_labels.append(cat_to_idx[cat])
        except Exception as exc:
            print(f"[AVERTISSEMENT] Ignoré {file_path} : {exc}")

    X = np.stack(all_points, axis=0)
    y = np.array(all_labels, dtype=np.int64)

    n_total = len(X)
    indices = rng.permutation(n_total)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }

    for split_name, idx in splits.items():
        np.save(out_path / f"points_{split_name}.npy", X[idx])
        np.save(out_path / f"labels_{split_name}.npy", y[idx])

    meta = {
        "num_points": num_points,
        "categories": categories,
        "cat_to_idx": cat_to_idx,
        "counts": {k: int(len(v)) for k, v in splits.items()},
        "total_saved": int(n_total),
    }
    with open(out_path / "metadata.json", "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2, ensure_ascii=False)

    print("Terminé :", meta)


if __name__ == "__main__":
    prepare_dataset(
        raw_dir="data",
        out_dir="data/processed",
        categories=["bed", "chair", "sofa", "table"],
        num_points=2048,
        max_total_objects=1500,
    )