# Fiche des Modèles, Licences et Réflexion Éthique (Model Card)

**Projet :** Génération de Formes 3D et Reconstruction de Scènes par 3D Gaussian Splatting  
**Auteures :** Wiame El Hadraoui – Douae Benabou  
**Dépôt GitHub :** [https://github.com/Wiame2004/projet-3d-gen-splatting](https://github.com/Wiame2004/projet-3d-gen-splatting)  
**Date :** Novembre 2026  

---

## 1. Sources et Licences des Données

### 1.1. Jeu de données — Génération de formes 3D (`data/`)
* **Origine :** Sous-ensemble extrait de **ModelNet10 / ShapeNetCore** (4 catégories d'objets du quotidien : `bed`, `chair`, `sofa`, `table`).
* **Prétraitement :** Conversion et normalisation en nuages de points 3D centrés et mis à l'échelle (stockés dans `data/processed/`).
* **Licence et usage :** Données issues de jeux de données académiques publics, utilisées strictement à des fins de recherche et d'enseignement. Conformément aux bonnes pratiques Git, les fichiers volumineux bruts sont exclus du suivi de version.

### 1.2. Jeu de données — Reconstruction 3D (`splatting_data/images/`)
* **Origine :** Jeu de données original composé de **9 photographies** d'un objet réel (une petite boîte blanche ornée de motifs floraux posée sur une surface claire), capturées directement par les membres de l'équipe.
* **Caractéristiques techniques :** Images hétérogènes multi-résolutions (`960 × 1280`, `1280 × 960` et `3120 × 4160` pixels) acquises sans calibration matérielle préalable.
* **Licence et droits :** Photographies originales réalisées par Wiame El Hadraoui et Douae Benabou, libres de droits dans le cadre de ce projet académique.

---

## 2. Fiches Techniques des Modèles (Model Cards)

### 2.1. Modèle 1 : Auto-Encodeur de Nuages de Points (`PointNetAE`)
* **Rôle :** Encodage et reconstruction/génération de formes géométriques 3D simples à partir de nuages de points.
* **Architecture :** Encodeur inspiré de PointNet (extraction de caractéristiques invariantes aux permutations par convolutions 1D et *Max Pooling*) couplé à un décodeur entièrement connecté reconstituant les coordonnées $(x, y, z)$.
* **Fonction de perte :** *Chamfer Distance* (mesure de la distance bidirectionnelle moyenne entre les points générés et les points cibles).
* **Cas d'usage prévu :** Apprentissage de représentations latentes 3D compactes et génération de mobilier synthétique simple.

### 2.2. Modèle 2 : Reconstruction de Scène par 3D Gaussian Splatting (`3DGS`)
* **Rôle :** Synthèse de nouvelles vues photoréalistes en temps réel d'une scène réelle à partir d'images 2D.
* **Pipeline :**
  1. Calibration multi-caméras (`single_camera=0`, modèle `SIMPLE_PINHOLE`) et reconstruction éparse par Structure-from-Motion via **COLMAP** (9/9 images recalées, 113 points initiaux, erreur de reprojection de `2,20133 px`).
  2. Optimisation des gaussiennes 3D anisotropes (position, covariance/rotation, opacité, harmoniques sphériques) sur **3 000 itérations** avec facteur de sous-échantillonnage `-r 4`.
* **Environnement matériel :** Exécution sur Google Colab avec GPU **NVIDIA Tesla T4** ($\le 12\text{ Go}$ de VRAM, CUDA 13.0, architecture `TORCH_CUDA_ARCH_LIST="7.5"`).
* **Artefact produit :** Nuage de gaussiennes `point_cloud.ply` (`5,1 Mo`, perte finale `0,0105611`), exclu du dépôt Git via `.gitignore`.

---

## 3. Limites Techniques et Biais

* **Parcimonie et hétérogénéité des vues (3DGS) :** L'utilisation de seulement 9 photographies aux résolutions variées laisse certains angles faiblement contraints géométriquement, ce qui engendre un léger flou gaussien sur les contours périphériques.
* **Compromis résolution / mémoire :** L'application du facteur de sous-échantillonnage `-r 4` et l'arrêt à 3 000 itérations (nécessaires pour respecter la limite de $12\text{ Go}$ de VRAM et maintenir un temps d'entraînement court d'environ 1 min 10 s) limitent la restitution des micro-textures par rapport à un entraînement complet (7 000 à 30 000 itérations).
* **Généralisation géométrique (`PointNetAE`) :** Le modèle est entraîné uniquement sur 4 catégories rigides (`bed`, `chair