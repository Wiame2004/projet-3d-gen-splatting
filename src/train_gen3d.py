from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.pointcloud_model import PointCloudDataset, PointNetAE, chamfer_distance


def save_comparison_figure(real_pts: np.ndarray, pred_pts: np.ndarray, out_path: Path):
    """Sauvegarde une vue 3D côte à côte (Réel vs Généré/Reconstruit)."""
    fig = plt.figure(figsize=(10, 4.5))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(real_pts[:, 0], real_pts[:, 1], real_pts[:, 2], s=2, c="tab:blue")
    ax1.set_title("Forme réelle (2048 pts)")
    ax1.set_axis_off()

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(pred_pts[:, 0], pred_pts[:, 1], pred_pts[:, 2], s=2, c="tab:orange")
    ax2.set_title("Forme reconstruite / générée")
    ax2.set_axis_off()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def train_and_evaluate(
    data_dir: str = "data/processed",
    results_dir: str = "results",
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    latent_dim: int = 256,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_path = Path(results_dir)
    (res_path / "figures").mkdir(parents=True, exist_ok=True)
    (res_path / "tables").mkdir(parents=True, exist_ok=True)

    train_ds = PointCloudDataset(f"{data_dir}/points_train.npy", f"{data_dir}/labels_train.npy")
    val_ds = PointCloudDataset(f"{data_dir}/points_val.npy", f"{data_dir}/labels_val.npy")
    test_ds = PointCloudDataset(f"{data_dir}/points_test.npy", f"{data_dir}/labels_test.npy")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    num_points = train_ds.points.shape[1]
    model = PointNetAE(num_points=num_points, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for pts, _ in train_loader:
            pts = pts.to(device)
            optimizer.zero_grad()
            recon, _ = model(pts)
            loss = chamfer_distance(recon, pts)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(pts)
        tr_loss /= len(train_ds)
        train_losses.append(tr_loss)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for pts, _ in val_loader:
                pts = pts.to(device)
                recon, _ = model(pts)
                va_loss += chamfer_distance(recon, pts).item() * len(pts)
        va_loss /= len(val_ds)
        val_losses.append(va_loss)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train CD: {tr_loss:.6f} | Val CD: {va_loss:.6f}")

    # Évaluation finale sur le jeu de test
    model.eval()
    test_loss = 0.0
    sample_real, sample_pred = None, None
    with torch.no_grad():
        for pts, _ in test_loader:
            pts = pts.to(device)
            recon, _ = model(pts)
            test_loss += chamfer_distance(recon, pts).item() * len(pts)
            if sample_real is None:
                sample_real = pts[0].cpu().numpy()
                sample_pred = recon[0].cpu().numpy()
    test_loss /= len(test_ds)

    max_vram_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    print(f"Test Chamfer Distance: {test_loss:.6f} | Max VRAM: {max_vram_gb:.2f} Go")

    # Courbe d'apprentissage
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Chamfer Distance")
    plt.plot(range(1, epochs + 1), val_losses, label="Val Chamfer Distance")
    plt.xlabel("Époque")
    plt.ylabel("Chamfer Distance")
    plt.title("Évolution de la Chamfer Distance (PointNetAE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(res_path / "figures" / "pointnet_ae_loss.png", dpi=150)
    plt.close()

    # Exemple visuel de reconstruction
    save_comparison_figure(sample_real, sample_pred, res_path / "figures" / "sample_reconstruction.png")

    # Génération de nouvelles formes par interpolation / bruit dans l'espace latent
    with torch.no_grad():
        z_rand = torch.randn(1, latent_dim, device=device) * 0.5
        gen_pts = model.decode(z_rand)[0].cpu().numpy()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(gen_pts[:, 0], gen_pts[:, 1], gen_pts[:, 2], s=2, c="tab:green")
    ax.set_title("Nuage de points généré (espace latent)")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(res_path / "figures" / "sample_generated.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    train_and_evaluate(epochs=15, batch_size=16)