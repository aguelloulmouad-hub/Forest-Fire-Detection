# 🔥 Système de Détection Temps Réel des Feux de Forêt et de la Fumée

<div align="center">
  
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Video%20Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-green?style=for-the-badge)

**Détection automatique du feu et de la fumée par réseaux de neurones convolutionnels (CNN) et transfert d'apprentissage**

*Projet d'IA — Année Universitaire 2025/2026*

[📄 Rapport](#-rapport) · [🚀 Démarrage Rapide](#-démarrage-rapide) · [📊 Résultats](#-résultats) · [🎬 Démos Vidéo](#-démos-vidéo)

</div>

---

## 📋 Table des Matières

- [Aperçu du Projet](#-aperçu-du-projet)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Modèles](#-modèles)
  - [InceptionResNetV2](#1-inceptionresnetv2)
  - [MobileNetV2](#2-mobilenetv2)
- [Résultats](#-résultats)
- [Structure du Projet](#-structure-du-projet)
- [Démarrage Rapide](#-démarrage-rapide)
- [Détection Vidéo Temps Réel](#-détection-vidéo-temps-réel)
- [Démos Vidéo](#-démos-vidéo)
- [Technologies Utilisées](#-technologies-utilisées)
- [Rapport](#-rapport)
- [Références](#-références)
- [Auteur](#-auteur)

---

## 🌍 Aperçu du Projet

Chaque année, des millions d'hectares de forêts sont détruits par les incendies, aggravant le réchauffement climatique et menaçant la biodiversité. Face aux limites des systèmes traditionnels (capteurs physiques coûteux et peu fiables en extérieur), ce projet propose un **système de détection automatique en temps réel** du feu et de la fumée basé sur le **Deep Learning** et le **transfert d'apprentissage**.

### Objectifs

- ✅ **Reproduire** l'approche de l'article de référence [1] avec **InceptionResNetV2**
- ✅ **Proposer** une alternative plus légère avec **MobileNetV2** et fine-tuning en deux phases
- ✅ **Développer** un système de détection vidéo temps réel fonctionnel
- ✅ **Comparer** les performances et l'efficacité des deux architectures

---

## 🏗 Architecture

Le pipeline complet se compose des étapes suivantes :

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌────────────────┐
│  Image /    │───▶│ Preprocessing │───▶│  CNN Backbone   │───▶│ Classification │
│  Frame vidéo│     │  224 × 224   │     │  (Transfer      │     │  fire / smoke  │
│             │     │  + Normalize │     │   Learning)     │     │  / non-fire    │
└─────────────┘     └──────────────┘     └─────────────────┘     └────────────────┘
```

Les deux modèles utilisent un **backbone pré-entraîné sur ImageNet** dont les couches convolutionnelles sont gelées (ou partiellement dégelées pour le fine-tuning), suivi d'une **tête de classification personnalisée**.

Un **seuil de confiance τ = 0.043** est appliqué : si la probabilité maximale est inférieure à ce seuil, la prédiction est classée **"Non-fire"**.

---

## 📁 Dataset

| Caractéristique | Détail |
|---|---|
| **Source** | [Forest Fire Dataset (Kaggle)](https://www.kaggle.com/datasets/kutaykutlu/forest-fire) |
| **Classes** | 🔥 Fire (1 102 images) — 💨 Smoke (1 102 images) |
| **Total** | 2 204 images équilibrées |
| **Split** | 80% Train (1 760) — 10% Validation (222) — 10% Test (222) |
| **Taille des images** | Redimensionnées à **224 × 224** pixels |
| **Augmentation** | Zoom, rotation, flips, décalages, variation de luminosité |

---

## 🧠 Modèles

### 1. InceptionResNetV2

> Reproduction fidèle de l'article de référence [1]

| Paramètre | Valeur |
|---|---|
| **Backbone** | InceptionResNetV2 (pré-entraîné ImageNet) |
| **Stratégie** | Backbone entièrement gelé |
| **Tête de classification** | GlobalAveragePooling2D → Dropout(0.45) → Dense(2, softmax) |
| **Optimiseur** | Adam (lr = 1e-3) |
| **Batch size** | 55 |
| **Époques** | 10 (avec EarlyStopping, patience = 4) |
| **Augmentation** | Zoom ±10%, flip horizontal/vertical |
| **Paramètres totaux** | ~55.8 M |
| **Taille du modèle** | ~205 Mo |

### 2. MobileNetV2

> Alternative légère avec fine-tuning en deux phases et augmentation agressive

| Paramètre | Valeur |
|---|---|
| **Backbone** | MobileNetV2 (pré-entraîné ImageNet) |
| **Stratégie** | Phase 1 : gelé → Phase 2 : dégel des 30 dernières couches |
| **Tête de classification** | GAP → Dense(256)+BN+Drop(0.3) → Dense(128)+BN+Drop(0.3) → Dense(2, softmax) |
| **Optimiseur** | Adam (Phase 1 : lr=1e-3, Phase 2 : lr=1e-4) |
| **Batch size** | 55 |
| **Époques** | 10 par phase (avec EarlyStopping, patience = 5) |
| **Augmentation** | Rotation ±20°, zoom ±15%, shifts ±10%, luminosité, flips |
| **Paramètres totaux** | ~3.4 M |
| **Taille du modèle** | ~25 Mo |

---

## 📊 Résultats

### Performances sur l'ensemble de test

| Métrique | InceptionResNetV2 | MobileNetV2 |
|---|:---:|:---:|
| **Accuracy** | 97.75% | 97.75% |
| **Précision** | 96.49% | 97.75% |
| **Rappel** | 99.10% | 97.75% |
| **F1-Score** | 97.78% | 97.75% |
| **AUC** | 99.56% | 99.50% |
| **Meilleure val_accuracy** | 99.55% | 99.55% |

### Matrice de confusion

<table>
<tr><th></th><th colspan="2" align="center">InceptionResNetV2</th><th colspan="2" align="center">MobileNetV2</th></tr>
<tr><td></td><td><b>Prédit Fire</b></td><td><b>Prédit Smoke</b></td><td><b>Prédit Fire</b></td><td><b>Prédit Smoke</b></td></tr>
<tr><td><b>Réel Fire</b></td><td>✅ 110</td><td>❌ 1</td><td>✅ 108</td><td>❌ 3</td></tr>
<tr><td><b>Réel Smoke</b></td><td>❌ 4</td><td>✅ 107</td><td>❌ 2</td><td>✅ 109</td></tr>
</table>

### Performances en inférence vidéo (CPU)

| Métrique | InceptionResNetV2 | MobileNetV2 |
|---|:---:|:---:|
| **Latence par frame** | 0.008 – 0.01 s | 0.004 – 0.005 s |
| **FPS estimés** | ~100-125 FPS | ~200-250 FPS |
| **Temps réel** | ✅ Oui | ✅ Oui |
| **Taille fichier .keras** | ~205 Mo | ~25 Mo (16× plus léger) |

> 💡 **MobileNetV2** est **16× plus léger** et **2× plus rapide** qu'InceptionResNetV2, tout en maintenant des performances quasi identiques. C'est le candidat idéal pour un déploiement embarqué (Raspberry Pi, drone, smartphone).

---

## 📂 Structure du Projet

```
Fire_detection/
│
├── 📓 Fire_Detection_InceptionResNetV2.ipynb   # Notebook : entraînement & évaluation InceptionResNetV2
├── 📓 Fire_Detection_MobileNetV2.ipynb         # Notebook : entraînement & évaluation MobileNetV2
│
├── 📄 rapport_mp_PFA.pdf                       # Rapport complet du projet (63 pages)
│
├── 🎬 video_demo_1.mp4                         # Vidéo de démonstration n°1
├── 🎬 video_demo_2.mp4                         # Vidéo de démonstration n°2
│
├── 📁 Application_video/                       # Scripts de détection vidéo temps réel
│   ├── 🐍 detection_vid_InceptionResNetV2.py   # Détection vidéo avec InceptionResNetV2
│   └── 🐍 detection_vid_mobilenetv2.py         # Détection vidéo avec MobileNetV2
│
└── 📄 README.md                                # Ce fichier
```

---

## 🚀 Démarrage Rapide

### Prérequis

```bash
pip install tensorflow numpy opencv-python matplotlib seaborn scikit-learn
```

### Entraînement des modèles

Les notebooks sont conçus pour **Google Colab** avec GPU (T4) :

1. **Télécharger** le [dataset Forest Fire](https://www.kaggle.com/datasets/kutaykutlu/forest-fire) sur Google Drive
2. **Ouvrir** le notebook souhaité dans Google Colab :
   - `Fire_Detection_InceptionResNetV2.ipynb` pour le modèle InceptionResNetV2
   - `Fire_Detection_MobileNetV2.ipynb` pour le modèle MobileNetV2
3. **Exécuter** toutes les cellules séquentiellement
4. Le modèle entraîné sera sauvegardé au format `.keras`

---

## 🎥 Détection Vidéo Temps Réel

### Utilisation

1. Placer le modèle `.keras` et le fichier `class_indices.json` dans le dossier `Application_video/`
2. Modifier le chemin de la vidéo dans le script si nécessaire
3. Lancer le script :

```bash
# Avec InceptionResNetV2
cd Application_video
python detection_vid_InceptionResNetV2.py

# Avec MobileNetV2
cd Application_video
python detection_vid_mobilenetv2.py
```

### Fonctionnement

- 📹 Chaque **frame** de la vidéo est analysée individuellement
- 🏷️ Le label prédit (**fire**, **smoke** ou **Non-fire**) et la probabilité sont affichés en overlay
- 🎨 Code couleur :
  - 🔴 **Rouge** → Feu détecté
  - 🟠 **Orange** → Fumée détectée
  - 🟢 **Vert** → Pas de détection
- ⏹️ Appuyer sur **`q`** pour arrêter la détection
- 📊 Les statistiques finales (nombre de détections) sont affichées à la fin

### Configuration

Les scripts utilisent les paramètres suivants (modifiables en tête de fichier) :

```python
MODEL_PATH = "fire_smoke_model.keras"      # Chemin vers le modèle
CLASS_INDICES_PATH = "class_indices.json"   # Mapping des classes
VIDEO_PATH = "video_test_1.mp4"            # Vidéo d'entrée
IMG_SIZE = (224, 224)                      # Taille d'entrée du modèle
THRESHOLD_NON_FIRE = 0.043                 # Seuil de confiance τ
```

---

## 🎬 Démos Vidéo

Le projet inclut **deux vidéos de démonstration** (`video_demo_1.mp4` et `video_demo_2.mp4`) montrant le système en action sur des séquences vidéo réelles. Ces vidéos illustrent :

- La détection du **feu** et de la **fumée** frame par frame
- L'affichage en temps réel des labels et probabilités
- Le compteur de frames traités
- La robustesse du modèle sur des scènes variées

---

## 🛠 Technologies Utilisées

| Technologie | Utilisation |
|---|---|
| **Python 3.8+** | Langage principal |
| **TensorFlow / Keras** | Construction et entraînement des modèles CNN |
| **OpenCV** | Traitement vidéo et affichage temps réel |
| **scikit-learn** | Métriques d'évaluation (accuracy, precision, recall, F1, ROC, AUC) |
| **Matplotlib / Seaborn** | Visualisation des résultats (courbes, matrices de confusion) |
| **Google Colab** | Environnement d'entraînement avec GPU |
| **NumPy** | Manipulation de tableaux et calculs numériques |

---

## 📄 Rapport

Le rapport complet du projet (`rapport_mp_PFA.pdf`, 63 pages) couvre :

1. **Introduction** — Contexte et problématique des feux de forêt
2. **État de l'art** — Réseaux de neurones convolutionnels et transfert d'apprentissage
3. **Méthodologie** — Pipeline complet, préparation des données, architectures
4. **Résultats InceptionResNetV2** — Métriques, courbes ROC, précision-rappel, matrice de confusion
5. **Comparaison avec l'article** — Validation de la reproductibilité
6. **Résultats MobileNetV2** — Fine-tuning en deux phases, comparaison des performances
7. **Conclusion et perspectives** — Limites et axes d'amélioration

---

## 📚 Références

1. R. K. Mohammed, *"A real-time forest fire and smoke detection system using deep learning"*, Int. J. Nonlinear Anal. Appl., vol. 13, no. 1, Jan. 2022, doi: 10.22075/ijnaa.2022.5899.
2. C. Szegedy, S. Ioffe, V. Vanhoucke, and A. Alemi, *"Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning"*, Proc. AAAI Conf. Artif. Intell., vol. 31, no. 1, Feb. 2017.
3. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, *"MobileNetV2: Inverted Residuals and Linear Bottlenecks"*, CVPR 2018, pp. 4510–4520.
4. Dataset : [Forest Fire — Kaggle](https://www.kaggle.com/datasets/kutaykutlu/forest-fire)

---

## 👤 Auteur

**AGUELLOUL Mouad**

- 🎓 Projet d'IA — Année Universitaire 2025/2026
- 👨‍🏫 Encadré par **Pr. NACHAOUI Mourad**

---

<div align="center">

**Made with ❤️ for DL & Fire detection**

</div>
