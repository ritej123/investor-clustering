# Segmentation des investisseurs par clustering

Projet de **clustering non supervisé** : regrouper 3 866 investisseurs selon leur **capacité** (revenu, patrimoine, logement, âge) et leur **volonté** (attitude face au risque, habitudes d’épargne) de prendre des risques. Les profils obtenus servent de base à un conseil automatisé ou à une segmentation de clientèle.

Le dépôt contient :
- le **notebook** d’analyse complet (EDA, préparation, ACP, choix de k, 4 modèles, validation) ;
- une **application web Streamlit** qui présente tous les résultats ;
- les fichiers de **déploiement** (Docker, intégration continue).

---

## Résultats principaux

**Modèle retenu : K-Means avec k = 5**

| Groupe | Profil | Effectif |
|---|---|---|
| C0 | Jeunes locataires à faible patrimoine | 859 |
| C1 | Seniors aisés propriétaires | 807 |
| C2 | Jeunes aisés, tolérants au risque | 823 |
| C3 | Jeunes propriétaires, revenu moyen, prudents | 779 |
| C4 | Seniors modestes, très prudents | 598 |

**Comparaison des 4 modèles**

| Modèle | Groupes | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|---|---|---|
| **K-Means** | 5 | **0,132** | 2,069 | **585,5** |
| Hiérarchique (Ward) | 5 | 0,103 | 2,319 | 444,2 |
| GMM | 5 | 0,126 | 2,126 | 448,3 |
| DBSCAN | 6 | 0,107 | 1,879 | 336,8 |

- **Stabilité de K-Means** : ARI moyen de 0,735 sur 20 relances (minimum 0,574).
- Les silhouettes sont faibles (< 0,2) : les groupes se chevauchent, ce qui est courant pour des données de comportement humain.
- Les métriques de DBSCAN sont calculées **hors bruit** (400 personnes, 10,3 %) : elles ne sont pas strictement comparables aux autres modèles.

---

## Données

Fichier `ProcessedData.xlsx` : 3 866 individus × 13 colonnes, toutes numériques (codes ordinaux), sans valeur manquante ni doublon.

| Variable | Sens (d’après la base américaine SCF, à vérifier) |
|---|---|
| ID | identifiant (retiré avant la modélisation) |
| AGE | tranche d’âge (1 à 6) |
| EDUC | niveau d’études (1 à 4) |
| MARRIED | 1 = en couple, 2 = seul |
| KIDS | nombre d’enfants |
| LIFECL | étape du cycle de vie (1 à 6) |
| OCCAT | catégorie professionnelle (1 à 4) |
| RISK | volonté de prendre des risques (1 = élevée, 4 = nulle) |
| HHOUSES | propriétaire du logement (0/1) |
| WSAVED | épargne par rapport aux revenus (1 à 3) |
| SPENDMOR | tendance à dépenser plus que les autres (1 à 5) |
| NWCAT | catégorie de patrimoine net (1 à 5) |
| INCCL | catégorie de revenu (1 à 5) |

---

## Méthode

1. **Chargement et contrôle** : types, valeurs manquantes, doublons.
2. **Exploration (EDA)** : distributions et matrice de corrélation (AGE ↔ LIFECL = 0,76).
3. **Préparation** : suppression de `ID`, standardisation (`StandardScaler`), test de la redondance AGE/LIFECL.
4. **ACP** : visualisation en 2D (44 % de la variance avec 2 composantes).
5. **Choix de k** : méthode du coude et silhouette, combinés à l’interprétabilité métier.
6. **Modèles** : K-Means, classification hiérarchique (Ward), GMM, DBSCAN.
7. **Validation** : silhouette, Davies-Bouldin, Calinski-Harabasz, accord entre modèles (ARI), stabilité.
8. **Interprétation** : profils, boîtes à moustaches, graphique capacité contre volonté.

---

## Structure du dépôt

> À adapter selon le contenu réel de votre dossier.

```
.
├── app.py                      # Application Streamlit
├── segmentation.py             # Entraînement et logique métier
├── requirements.txt            # Dépendances Python
├── Dockerfile                  # Image Docker
├── tests/                      # Tests de la logique métier
├── .github/workflows/          # Intégration continue GitHub
├── notebooks/
│   └── Clustering_Investisseurs.ipynb
├── data/
│   └── ProcessedData.xlsx
└── README.md
```

---

## Installation et lancement

```bash
git clone https://github.com/<votre-compte>/investor-segmentation-clustering.git
cd investor-segmentation-clustering

python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Le modèle s’entraîne automatiquement au premier lancement (environ 30 secondes).

### Pages de l’application

| Page | Contenu |
|---|---|
| Synthèse | Indicateurs clés et répartition des 5 profils |
| Profils des clusters | Heatmap, description et piste d’orientation par groupe |
| Cartographie | Projection ACP et graphique capacité contre volonté |
| Simulateur | Saisie d’un individu : groupe et proximité avec chaque profil |
| Comparaison des modèles | Métriques, ARI, stabilité, choix de k |
| Données et export | Résultats filtrables et téléchargement CSV |
| Méthodologie | Pipeline, limites, dictionnaire des variables |

### Notebook

Ouvrez `notebooks/Clustering_Investisseurs.ipynb` dans Jupyter, VS Code ou Google Colab, puis faites *Run All*. Adaptez le chemin du fichier de données dans la cellule de chargement :

```python
df = pd.read_excel("data/ProcessedData.xlsx")
```

---

## Déploiement

### Option 1 : Streamlit Community Cloud (gratuit)

1. Poussez le dépôt sur GitHub (voir ci-dessous).
2. Allez sur [share.streamlit.io](https://share.streamlit.io) et connectez votre compte GitHub.
3. Cliquez sur **New app**, choisissez le dépôt, la branche `main` et le fichier principal `app.py`.
4. Cliquez sur **Deploy**.

### Option 2 : Docker

```bash
docker build -t segmentation .
docker run -p 8501:8501 segmentation
```

L’application est alors disponible sur `http://localhost:8501`.

---

## Publier sur GitHub

```bash
git init
git add .
git commit -m "Projet de segmentation des investisseurs"
git branch -M main
git remote add origin https://github.com/<votre-compte>/investor-segmentation-clustering.git
git push -u origin main
```

Fichiers à ne pas publier : ajoutez-les à un `.gitignore`.

```
.venv/
__pycache__/
.ipynb_checkpoints/
*.pyc
```

---

## Limites

- Variables ordinales traitées comme numériques.
- Silhouettes faibles : les groupes se chevauchent et leurs frontières restent floues.
- Données de sondage, sans validation externe.
- DBSCAN : paramètres choisis par un critère de sélection arbitraire ; les données forment un continuum sans zones denses séparées, donc ce modèle est peu adapté.
- La signification des codes vient de la documentation SCF et doit être confirmée.
- Les pistes d’orientation par profil sont indicatives : ce n’est **pas** un conseil en investissement.

---

## Technologies

Python, pandas, NumPy, scikit-learn, SciPy, matplotlib, seaborn, Streamlit, Docker, GitHub Actions.
