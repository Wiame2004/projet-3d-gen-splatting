# Projet 6 — Génération de formes 3D et reconstruction de scène (Gaussian Splatting)

## Présentation
Ce projet combine la génération de formes 3D simples (nuages de points / voxels) et la reconstruction d'une scène réelle photographiée au smartphone via **3D Gaussian Splatting**, avec une étude de compression et de performances sur GPU (≤ 12 Go VRAM, Google Colab / Kaggle).

## Structure du dépôt
- `notebooks/` : notebooks numérotés (données, génération 3D, splatting, compression/évaluation, démo)
- `src/` : code Python modulaire et réutilisable
- `configs/` : paramètres des expériences
- `data/` : scripts de préparation et fiche des données (sans données lourdes)
- `results/` : figures, tableaux chiffrés et exemples générés
- `demo/` : application interactive et vidéo de démonstration
- `report/` : rapport final (6 à 8 pages)
- `docs/` : licences, fiche du modèle et réflexion éthique

## Installation
```bash
git clone <URL_DU_DEPOT>
cd projet-3d-gen-splatting
pip install -r requirements.txt