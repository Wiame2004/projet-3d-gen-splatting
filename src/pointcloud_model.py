import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    def __init__(self, points_path: str, labels_path: str | None = None):
        self.points = torch.from_numpy(np.load(points_path)).float()  # (N, 2048, 3)
        self.labels = (
            torch.from_numpy(np.load(labels_path)).long()
            if labels_path is not None
            else None
        )

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.points[idx], self.labels[idx]
        return self.points[idx]


class PointNetAE(nn.Module):
    """AutoEncoder léger sur nuages de points (B, N, 3) -> reconstruction (B, N, 3)."""

    def __init__(self, num_points: int = 2048, latent_dim: int = 256):
        super().__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim

        self.enc = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, latent_dim, 1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
        )

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_points * 3),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.enc(x.transpose(1, 2))
        return torch.max(feat, dim=2)[0]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z).view(-1, self.num_points, 3)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
    """Chamfer Distance symétrique moyenne par blocs pour limiter l'empreinte VRAM."""
    _, n_pred, _ = pred.shape
    n_targ = target.shape[1]

    min_p2t_list = []
    for i in range(0, n_pred, chunk_size):
        p_chunk = pred[:, i : i + chunk_size, :]
        dist_pt = torch.cdist(p_chunk, target, p=2) ** 2
        min_p2t_list.append(dist_pt.min(dim=2)[0])
    min_p2t = torch.cat(min_p2t_list, dim=1).mean(dim=1)

    min_t2p_list = []
    for j in range(0, n_targ, chunk_size):
        t_chunk = target[:, j : j + chunk_size, :]
        dist_tp = torch.cdist(t_chunk, pred, p=2) ** 2
        min_t2p_list.append(dist_tp.min(dim=2)[0])
    min_t2p = torch.cat(min_t2p_list, dim=1).mean(dim=1)

    return (min_p2t + min_t2p).mean()