
# 🛩️ Projet Drone 2025 - Vision par ordinateur embarquée (Parrot Bebop 2)

Ce dépôt regroupe une partie du projet de traitement vidéo embarqué réalisé sur un drone **Parrot Bebop 2**.  
Il s'agit ici **de la branche "Vision"**, dédiée à la récupération du flux vidéo du drone et à l’application de **détection temps réel** (notamment d’un gant de couleur rouge/orangé) via **OpenCV**.

---

## 🧱 Structure du projet global

```bash
Projet_Drone/
│
├── Code/Code_Drone/Code_Flux_Avec_Detection/   # Code principal pour communication, traitement, détection
│   ├── Test_Flux_Video.py                      # Code de test pour lecture flux ffmpeg + affichage
│   └── README.md                               # Ce fichier (ce dépôt)
└── 
```

---

## 🎯 Objectifs (vision)

- Connexion WiFi au drone Parrot Bebop 2
- Récupération du flux vidéo via `ffmpeg` (via fichier SDP fourni par le SDK pyparrot)
- Traitement des frames temps réel avec OpenCV
- Détection d’un gant rouge sur base couleur + forme
- Affichage fluide avec mise en valeur
- Épuration continue des images pour limiter l’usage disque

---

## ✅ Fonctionnalités

- 📡 Connexion automatique au drone (via pyparrot)
- 🎥 Flux vidéo live récupéré sous forme de fichiers image
- 🧠 Détection d’objet (gant) sur plage HSV définie manuellement
- 🧼 Nettoyage automatique des images trop anciennes
- 🖍 Affichage temps réel avec surlignage en vert de l’objet détecté
- 🚀 Performant sous Windows + Python 3.11 + Anaconda

---

## 🔁 Historique des Implémentations

### 1. **Olympe SDK (officiel Parrot)** ❌  
- Linux uniquement, complexe à installer
- Trop lourd pour le projet → abandonné

### 2. **pyparrot + DroneVision** ✅  
- Stable sous Windows
- Utilise `ffmpeg` pour transformer le flux en images PNG
- Facilement intégrable avec OpenCV

### 3. **VLC / RTSP** ⚠️  
- Tenté mais instable
- Lecture directe du flux impossible sur certaines machines

### 4. **ffmpeg vers image + OpenCV** ✅  
- Pipeline retenu
- ffmpeg enregistre en temps réel
- OpenCV lit les images dès qu'elles sont disponibles

---

## 🖼️ Détection de gant

La détection s’appuie sur :
- **Plage HSV ajustée** en fonction des couleurs connues du gant :
  - `#89190D`, `#C0584E`, `#B95009`, `#DB8D3F`, `#C97168`, `#D6955E`
- **Nettoyage du masque** (morphologie)
- **Filtrage des contours** selon :
  - Aire
  - Ratio de forme
  - Solidité (remplit bien son enveloppe convexe)
  - Complexité
- **Fallbacks intelligents** pour ne pas rater une détection plausible

---

## 🧠 Exécution

### 🧪 Version démo (détection uniquement)

```bash
python Code_Drone/Detection_Live.py
```

> ⚠️ Se connecter d’abord au WiFi du drone (ex: `Bebop2-XXXXXX`)

### 🔧 Prérequis

- Python ≥ 3.8 (Anaconda recommandé, Python 3.11 testé)
- ffmpeg dans le PATH
- pip install:
  ```bash
  pip install opencv-python numpy pyparrot
  ```

---

## 🧼 Nettoyage mémoire / fichiers

Un thread secondaire supprime automatiquement les anciennes images (> 10 dernières) toutes les X secondes, sans bloquer la détection.

---

## 📸 Exemple (à ajouter plus tard)

```text
[IMG] Affichage flux + gant entouré en vert
```

---

## 🔧 Limitations & pistes d’amélioration

- La capture par image fixe (vs flux) introduit un léger délai (buffer + sauvegarde disque)
- Parfois, `ffmpeg` perd des paquets → erreurs de décodage H264
- Amélioration possible :
  - Passer à `cv2.VideoCapture` si support natif du flux
  - Détection IA (modèle entraîné sur gants réels)

---

## 🧩 Modules à venir (hors scope ici)

- 🧭 Contrôle du drone avec l'utilisation de L'IA
- 🎯 Suivi du gant dans l’image
- 🎒 Capture de dataset automatique
- 📡 Streaming + affichage en direct

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