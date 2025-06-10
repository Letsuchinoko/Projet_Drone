# 🛩️ Projet Drone 2025 - Vision par ordinateur embarquée (Parrot Bebop 2)

Ce dépôt regroupe une partie du projet de traitement vidéo embarqué réalisé sur un drone **Parrot Bebop 2**.  
Il s'agit ici **de la branche "Vision"**, dédiée à la récupération du flux vidéo du drone et à l'application de **détection temps réel** avec **zoom adaptatif intelligent** (notamment d'un gant de couleur rouge/orangé) via **OpenCV**.

---

## 🧱 Structure du projet global

```bash
Projet_Drone/Code/Code_Drone/
│
├── Code_Prototypage_Data_Analysis/    # Code principal optimisé avec zoom adaptatif
│   │
│   └── main.py                                 # Détection intelligente avec zoom adaptatif 🆕
│   └── README.md                               # Ce fichier (ce dépôt)
│
├── Code_Flux_Avec_Detection/          # Code principal pour communication, traitement, détection
│   │
│   └── Detection_Live.py                       # Détection gant en direct (version de base)
│
├── Code_Flux_Video/                   # Code tests pour communication, récupération du flux
│   │
│   └── Test_Flux_Video.py                      # Code de test pour lecture flux ffmpeg + affichage
│
├── Tests_Gants/                       # Code tests detection gant
│   │
│   ├── Capture_Gant.py                          # Script principal de traitement d'images
│   ├── images/                                  # 📥 Images à analyser
│   ├── detection/                               # 📤 Images annotées avec gant détecté
│   ├── masks/                                   # 📤 Gants détourés (sur fond noir)
│   └── redzones/                                # 📤 Zones rouges détectées (debug HSV)
├── README.md                          # Ce fichier (ce dépôt)
└── 
```

---

## 🎯 Objectifs (vision)

- Connexion WiFi au drone Parrot Bebop 2
- Récupération du flux vidéo via `ffmpeg` optimisé (via fichier SDP fourni par le SDK pyparrot)
- Traitement des frames temps réel avec OpenCV et **zoom adaptatif intelligent**
- Détection d'un gant rouge à **longue distance (3m+)** grâce au système de zoom
- Affichage fluide avec interface utilisateur enrichie
- **Système de zoom automatique** basé sur la distance estimée de l'objet

---

## ✅ Fonctionnalités

### 🔍 **Nouvelles fonctionnalités - Zoom Adaptatif**
- 🎯 **Zoom automatique intelligent** (1.0x à 4.0x) basé sur la taille de l'objet détecté
- 📏 **Détection longue distance** optimisée pour 3m+ avec zoom progressif
- 🔄 **Lissage du zoom** pour éviter les oscillations
- 🎪 **Zone de recherche adaptative** qui se concentre sur la dernière position détectée
- 📊 **Interface enrichie** avec barre de zoom, statistiques en temps réel et historique
- ⚡ **Performance optimisée** avec morphologie adaptative selon le niveau de zoom

### 📡 **Fonctionnalités de base**
- 📡 Connexion automatique au drone (via pyparrot)
- 🎥 Flux vidéo live via pipeline FFmpeg optimisé
- 🧠 Détection d'objet (gant) sur plage HSV définie avec exclusions intelligentes
- 🖍 Affichage temps réel avec surlignage coloré selon la distance
- 🚀 Performant sous Windows + Python 3.11 + Anaconda
- 🎮 **Contrôle drone intégré** (décollage, atterrissage, mouvements)

---

## 🔁 Historique des Implémentations

### 1. **Olympe SDK (officiel Parrot)** ❌  
- Linux uniquement, complexe à installer
- Trop lourd pour le projet → abandonné

### 2. **pyparrot + DroneVision** ✅  
- Stable sous Windows
- Utilise `ffmpeg` pour transformer le flux en images PNG
- Facilement intégrable avec OpenCV

### 3. **Pipeline FFmpeg direct** ✅ 🆕
- **Pipeline temps réel** sans fichiers intermédiaires
- Lecture directe du buffer FFmpeg dans OpenCV
- **Performances optimales** pour le zoom adaptatif

### 4. **Système de zoom adaptatif** ✅ 🆕
- **Auto-zoom basé sur l'aire** de l'objet détecté
- **Zones de recherche intelligentes** 
- **Lissage temporel** pour éviter les variations brutales

---

## 🎯 Détection de gant avec zoom adaptatif

### 🎨 **Détection couleur avancée**
La détection s'appuie sur :
- **Plages HSV multiples** optimisées pour différentes conditions d'éclairage :
  - Orange principal : `[12-20, 160-255, 160-255]`
  - Orange lumineux : `[10-18, 180-255, 180-255]`
  - Orange ombré : `[14-19, 120-200, 140-220]`
  - Rouge (deux plages) : `[0-6, 160-255, 160-255]` et `[174-180, 160-255, 160-255]`
- **Exclusions intelligentes** (peau, bordures)
- **Ajustement automatique** des seuils selon le niveau de zoom

### 🔍 **Système de zoom intelligent**
- **Zone 1 (Très loin)** : Aire < 800px → Zoom 3.5x
- **Zone 2 (Loin)** : Aire < 1500px → Zoom 2.5x  
- **Zone 3 (Normal)** : Aire < 3000px → Zoom 1.8x
- **Zone 4 (Proche)** : Aire < 6000px → Zoom 1.3x
- **Zone 5 (Très proche)** : Aire > 6000px → Zoom 1.0x

### 🧠 **Filtrage avancé**
- **Morphologie adaptative** selon le niveau de zoom
- **Filtrage géométrique** (aire, ratio, solidité)
- **Score de position** favorisant le centre de l'image
- **Historique de détection** pour la stabilisation

---

## 🧠 Exécution

### 🚀 **Version zoom adaptatif (recommandée)**

```bash
python Code_Prototypage_Data_Analysis/main.py
```

### 🧪 **Version de base**

```bash
python Code_Flux_Avec_Detection/Detection_Live.py
```

> ⚠️ Se connecter d'abord au WiFi du drone (ex: `Bebop2-XXXXXX`)

### 🔧 Prérequis

- Python ≥ 3.8 (Anaconda recommandé, Python 3.11 testé)
- **ffmpeg dans le PATH** (obligatoire)
- pip install:
  ```bash
  pip install opencv-python numpy pyparrot
  ```

---

## 🎮 Contrôles

### 🖱️ **Interface vidéo**
- `q` ou `Échap` : Quitter
- `s` : Screenshot avec informations de zoom
- `r` : Reset complet du détecteur
- `z` : Reset du zoom à 1.0x
- `+` / `=` : Zoom manuel +0.5x
- `-` : Zoom manuel -0.5x
- `d` : Affichage debug détaillé

### 🚁 **Contrôle drone** (dans le terminal)
- `t` : Décollage | `l` : Atterrissage | `e` : Quitter
- `f/b/g/d` : Avant/Arrière/Gauche/Droite
- `h/m` : Haut/Bas
- `a/c` : Rotation gauche/droite

---

## 📊 Interface utilisateur enrichie

### 🎯 **Affichage principal**
- **Statut de détection** avec indicateur de zoom
- **Contour coloré** selon la distance estimée :
  - 🟢 Vert : Proche (> 4000px²)
  - 🟡 Jaune : Moyen (1500-4000px²)  
  - 🟠 Orange : Loin (< 1500px²)

### 📈 **Informations temps réel**
- **Barre de zoom visuelle** avec niveau actuel
- **Statistiques de performance** (FPS, taux de détection)
- **Historique de détection** (●○●○●)
- **Zone de recherche** visualisée quand le zoom est actif

### 📸 **Screenshots enrichis**
Les captures incluent automatiquement :
- Niveau de zoom actuel
- Numéro de frame
- Informations de détection

---

## 🔧 Optimisations techniques

### ⚡ **Performances**
- **Pipeline FFmpeg optimisé** sans fichiers intermédiaires
- **Skip de frames adaptatif** pour maintenir le temps réel
- **Morphologie adaptative** selon le niveau de zoom
- **Zone de recherche restreinte** en mode zoom élevé

### 🧠 **Algorithmes intelligents**
- **Lissage temporel du zoom** (facteur 0.1) pour éviter les oscillations
- **Historique de détection** pour la stabilisation
- **Score composite** combinant aire, forme et position
- **Fallback progressif** : zoom out si pas de détection

### 📝 **Logging avancé**
- **Logs structurés** avec horodatage
- **Statistiques détaillées** en fin de session
- **Informations de debug** sur demande
- **Sauvegarde automatique** des logs

---

## 📈 Métriques et performances

Le système génère automatiquement :
- **Taux de détection** global (détections/frames)
- **Statistiques de zoom** (ajustements, niveau moyen)
- **Performance FPS** temps réel
- **Aire moyenne** des objets détectés
- **Durée de session** complète

---

## 🔧 Limitations & pistes d'amélioration

### ⚠️ **Limitations actuelles**
- Le zoom adaptatif introduit un léger délai de stabilisation
- Performance dépendante de la qualité du flux WiFi
- Détection optimisée pour gants rouge/orange uniquement

### 🚀 **Améliorations futures**
- **Multi-objet** : détection simultanée de plusieurs gants
- **IA embarquée** : modèle de deep learning optimisé
- **Prédiction de trajectoire** pour anticipation du zoom
- **Calibration automatique** des couleurs selon l'éclairage
- **Mode nuit** avec détection infrarouge

---

## 🧩 Modules à venir (hors scope ici)

- 🧭 **Navigation autonome** basée sur la détection
- 🎯 **Suivi actif** avec contrôle des mouvements du drone
- 🎒 **Capture de dataset** automatique pour entraînement IA
- 📡 **Streaming multi-clients** avec zoom partagé
- 🤖 **Mode autonome** avec recherche intelligente

---

## 📊 Résultats typiques

Sur une session de test standard :
- **Détection** : 85-95% de réussite selon les conditions
- **Performance** : 15-25 FPS selon la configuration
- **Portée** : Détection efficace jusqu'à 5+ mètres
- **Zoom** : Adaptation automatique en 1-2 secondes

---

## 👤 Auteur

Projet encadré — Université Paris-Saclay - IUT de Cachan 2025  
Réalisé par Rayan

Encadrant : Mr.Mininger

---

## 📄 Licence

Ce projet est sous licence **Creative Commons Attribution - NonCommercial 4.0 International (CC BY-NC 4.0)**.  
Vous êtes libre de :
- **Partager** — copier, distribuer et communiquer le matériel par tous moyens et sous tous formats
- **Adapter** — remixer, transformer et créer à partir du matériel

Sous les conditions suivantes :
- **Attribution** — Vous devez créditer le projet, fournir un lien vers la licence, et indiquer si des modifications ont été effectuées.
- **Pas d'utilisation commerciale** — Vous ne pouvez pas faire un usage commercial de ce contenu.

📖 [Consulter la licence complète](https://creativecommons.org/licenses/by-nc/4.0/)