# Segmentation des investisseurs : application de déploiement

Application web (Streamlit) qui présente et exploite les résultats du projet de clustering :
regrouper des investisseurs selon leur **capacité** et leur **volonté** de prendre des risques.

- 3 866 individus, 12 variables ordinales
- Modèle retenu : **K-Means, k = 5** (comparé à HAC, DBSCAN et GMM)
- Silhouette 0,132 · stabilité (ARI moyen sur 20 relances) 0,735

## Fonctionnalités

| Page | Contenu |
|---|---|
| Synthèse | Indicateurs clés, répartition des 5 profils |
| Profils des clusters | Heatmap des profils, description et piste d'orientation par cluster |
| Cartographie | Projection ACP, graphique capacité vs volonté |
| Simulateur | Saisie d'un individu → cluster, proximité relative, comparaison au profil moyen |
| Comparaison des modèles | Métriques des 4 modèles, accord (ARI), stabilité, choix de k |
| Données et export | Résultats filtrables, téléchargement CSV |
| Méthodologie | Pipeline, limites, dictionnaire des variables |

## Structure

```
app.py                 interface Streamlit
segmentation.py        entraînement, profils, prédiction (sans dépendance Streamlit)
train_model.py         script d'entraînement → models/
data/ProcessedData.xlsx
tests/                 tests unitaires et test de fumée de l'interface
Dockerfile · requirements.txt · .github/workflows/ci.yml
```

## Lancer en local

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python train_model.py              # génère models/ (≈ 30 s ; fait aussi automatiquement au 1er lancement)
streamlit run app.py               # http://localhost:8501
```

## Tests

```bash
pip install pytest
pytest -q
```

## Déploiement

**Option 1 : Streamlit Community Cloud (gratuit, le plus simple)**
1. Poussez le dossier sur un dépôt GitHub.
2. Sur share.streamlit.io : *New app* → choisissez le dépôt, la branche, le fichier `app.py`.
3. *Deploy*. Au premier chargement, le modèle s'entraîne automatiquement.

**Option 2 : Docker (serveur, cloud, Render, Azure, AWS…)**
```bash
docker build -t segmentation-investisseurs .
docker run -p 8501:8501 segmentation-investisseurs
```

**Option 3 : Hugging Face Spaces**
Créez un Space de type *Streamlit*, déposez les fichiers du projet.

## Limites et précautions

- Les silhouettes sont faibles : les profils se chevauchent (continuum plutôt que groupes isolés).
- DBSCAN ne trouve pas de groupes nettement séparés ; ses métriques (hors bruit) ne sont pas comparables aux autres modèles.
- Les libellés des modalités viennent de la documentation de la base américaine SCF : à confirmer avec la source.
- Les pistes d'orientation sont indicatives : **pas un conseil en investissement**.
- Données personnelles : le jeu fourni est anonymisé (identifiant numérique). Ne déployez pas de données nominatives sans cadre juridique adapté.
