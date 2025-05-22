# 🛩️ Projet Drone 2025 – Drone autonome piloté par IA & détection de gestes

Projet réalisé à l’IUT de Cachan – Université Paris-Saclay  
**Année universitaire 2024–2025**

---

## 🧠 Objectif du projet

Développer un **drone autonome piloté par reconnaissance gestuelle via IA**, capable de se déplacer dans des **zones difficiles d’accès** pour effectuer des **mesures sonores** (niveau en dB) à proximité d'une enceinte.

### 🔍 Fonctionnement global :
- 🎮 L’utilisateur dirige le drone à l’aide de **gestes manuels**, détectés par la **caméra du drone**.
- 🧠 Une **IA traite les mouvements** pour piloter le drone.
- 📡 Le drone embarque une carte STM32 avec **capteurs GPS et niveau sonore**.
- 📲 Une **application Android** affiche les données récupérées via **Bluetooth**.

---

## 🗂️ Arborescence du dépôt

```
Projet_Drone/
│
├── Carte_Drone/                 # ⚙️ Conception de la carte électronique (KiCad)
│
├── Code/
│   ├── Code_Drone/              # 🎥 Vision embarquée (détection main + OpenCV)
│   ├── Code_API_Drone/          # 📱 App Android pour affichage données capteurs
│   └── Code_general/            # 🔧 (Extensions futures / archives)
│
├── Documents/                   # 📄 Carte mentale, matériel, visuels, images
│   ├── Carte mentale.drawio
│   ├── Liste matériel.pdf
│   └── Projet Drone.png / .gan
│
└── README.md                    # Ce fichier
```

---

## 📌 Sous-projets

### 1. 🎥 Détection de gestes (Vision embarquée – `Code_Drone/`)

Traitement temps réel du **flux vidéo du drone Parrot Bebop 2**, détection d’un **gant rouge** porté par l’utilisateur, IA (TensorFlow / HuggingFace) pour classification des gestes.

- Détection couleur (HSV)
- Classification IA via Teachable Machine
- Contrôle du drone par position de la main
- Nettoyage automatique du cache image

👉 [Voir README dédié](./Code/Code_Drone/README.md)

---

### 2. ⚙️ Carte électronique embarquée (`Carte_Drone/`)

Conception et soudure d’une **carte sur KiCad** embarquant plusieurs capteurs :

- Microcontrôleur STM32F303K8T6 (Nucleo-32)
- GPS PMOD
- Capteur de son (DFRobot V2.2)
- Module Bluetooth (ref à confirmer)
- Transmission Bluetooth vers l’app Android

---

### 3. 📱 Application Android (`Code_API_Drone/`)

Développement d’une **application mobile (Android Studio)** permettant de :
- Se connecter en Bluetooth au drone
- Lire les données (GPS + son)
- Afficher ces valeurs en temps réel
- Tester sur plusieurs plateformes (tablette & téléphone)

---

### 4. 🧠 IA pour reconnaissance gestuelle

Utilisation de **TensorFlow**, **HuggingFace** et **Teachable Machine** pour :
- Créer un dataset de mouvements
- Entraîner un modèle de classification
- Prédire des gestes simples
- Contrôler le drone avec ces gestes

---

## 👥 Équipe & responsabilités

| Prénom           | Pseudo GitHub       | Rôle principal                                      | Tâches principales |
|------------------|---------------------|-----------------------------------------------------|---------------------|
| **Bouna**        | `weeduck12`         | Capteurs & instrumentation                          | Configuration capteurs, acquisition GPS, mesure dB, visualisation |
| **Baptiste**     | `Letsuchinoko`      | Carte électronique                                  | Design KiCad, soudure, tests continuité, liaison Bluetooth |
| **Abderrahmane** | `Abderra-boutka`    | Application mobile (API)                            | Dev Android Studio, réception Bluetooth, affichage, tests multiplateforme |
| **Rayan**        | `RayKill`           | IA & pilotage du drone                              | Détection main, reconnaissance de geste, entraînement IA, contrôle drone |

---

## ⚙️ Technologies utilisées

- **Drone** : Parrot Bebop 2 + SDK PyParrot
- **Vision / IA** : Python, OpenCV, ffmpeg, TensorFlow, HuggingFace
- **Carte embarquée** : STM32 Nucleo-32 (C / STMCubeIDE)
- **Mobile** : Android Studio (Java/Kotlin)
- **Électronique** : KiCad
- **Transmission** : Bluetooth

---

## 📅 Avancement

| Module                    | État              |
|---------------------------|-------------------|
| Détection gant + flux     | ✅ Fonctionnel    |
| Carte électronique        | 🛠️ En test        |
| App mobile Android        | 🔧 En cours       |
| Contrôle IA du drone      | 🔜 En intégration |

---

## 🔐 Licence

Ce projet est sous licence **Creative Commons Attribution – NonCommercial 4.0 International (CC BY-NC 4.0)**  
📖 [Lire la licence](https://creativecommons.org/licenses/by-nc/4.0/)

---

## 🏁 Remerciements

Projet encadré par **Mr. Mininger**  
Formation LP MECSE GEII – IUT de Cachan – Université Paris-Saclay – Promo 2025
