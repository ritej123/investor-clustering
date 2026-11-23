"""Logique métier du projet : entraînement, nommage des profils et prédiction.

Ce module ne dépend pas de Streamlit : il peut être testé et réutilisé seul
(notebook, API, script batch).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ProcessedData.xlsx"
MODEL_DIR = ROOT / "models"

RANDOM_STATE = 42
K = 5                                   # nombre de clusters retenu dans le notebook
DBSCAN_EPS, DBSCAN_MIN_SAMPLES = 2.0, 12  # paramètres retenus dans le notebook
FEATURES = ["AGE", "EDUC", "MARRIED", "KIDS", "LIFECL", "OCCAT",
            "RISK", "HHOUSES", "WSAVED", "SPENDMOR", "NWCAT", "INCCL"]

# --------------------------------------------------------------------------
# Dictionnaire des variables (libellés d'après la documentation SCF : à vérifier)
# --------------------------------------------------------------------------
VARIABLES = {
    "AGE": {"label": "Tranche d'âge", "options": {
        1: "1 · moins de 35 ans", 2: "2 · 35 à 44 ans", 3: "3 · 45 à 54 ans",
        4: "4 · 55 à 64 ans", 5: "5 · 65 à 74 ans", 6: "6 · 75 ans et plus"}},
    "EDUC": {"label": "Niveau d'études", "options": {
        1: "1 · sans diplôme du secondaire", 2: "2 · diplôme du secondaire",
        3: "3 · études supérieures sans diplôme", 4: "4 · diplôme universitaire"}},
    "MARRIED": {"label": "Situation de couple", "options": {
        1: "1 · en couple", 2: "2 · seul(e)"}},
    "KIDS": {"label": "Nombre d'enfants", "options": {i: str(i) for i in range(0, 9)}},
    "LIFECL": {"label": "Étape du cycle de vie", "options": {
        1: "1 · moins de 55 ans, seul, sans enfant", 2: "2 · moins de 55 ans, en couple, sans enfant",
        3: "3 · moins de 55 ans, en couple, avec enfants", 4: "4 · moins de 55 ans, seul, avec enfants",
        5: "5 · 55 ans et plus, en activité", 6: "6 · 55 ans et plus, retraité / inactif"}},
    "OCCAT": {"label": "Catégorie professionnelle", "options": {
        1: "1 · cadre / profession libérale", 2: "2 · technicien, vente, services",
        3: "3 · ouvrier / production / autre", 4: "4 · sans emploi / retraité"}},
    "RISK": {"label": "Volonté de prendre des risques", "options": {
        1: "1 · risques importants pour des gains importants", 2: "2 · risques supérieurs à la moyenne",
        3: "3 · risques moyens", 4: "4 · aucune prise de risque"}},
    "HHOUSES": {"label": "Propriétaire de son logement", "options": {0: "0 · non", 1: "1 · oui"}},
    "WSAVED": {"label": "Épargne par rapport aux revenus", "options": {
        1: "1 · dépenses supérieures aux revenus", 2: "2 · dépenses égales aux revenus",
        3: "3 · dépenses inférieures aux revenus"}},
    "SPENDMOR": {"label": "Tendance à dépenser (échelle 1 à 5)", "options": {
        1: "1 · très faible", 2: "2 · faible", 3: "3 · moyenne", 4: "4 · élevée", 5: "5 · très élevée"}},
    "NWCAT": {"label": "Patrimoine net (catégorie)", "options": {
        1: "1 · très faible", 2: "2 · faible", 3: "3 · moyen", 4: "4 · élevé", 5: "5 · très élevé"}},
    "INCCL": {"label": "Revenu (catégorie)", "options": {
        1: "1 · très faible", 2: "2 · faible", 3: "3 · moyen", 4: "4 · élevé", 5: "5 · très élevé"}},
}

# --------------------------------------------------------------------------
# Profils d'investisseurs (nommage par règles, donc stable et explicable)
# --------------------------------------------------------------------------
PROFILE_INFO = {
    "jeune_locataire": {
        "name": "Jeunes locataires à faible patrimoine",
        "capacity": "Faible", "willingness": "Faible",
        "description": "Ménages jeunes, presque aucun propriétaire, patrimoine et revenus faibles, "
                       "attitude prudente face au risque.",
        "orientation": "Constitution d'une épargne de précaution, produits simples et liquides, "
                       "éducation financière."},
    "jeune_proprietaire": {
        "name": "Jeunes propriétaires, revenu moyen, prudents",
        "capacity": "Moyenne", "willingness": "Faible",
        "description": "Jeunes ménages propriétaires de leur logement, revenu et patrimoine moyens, "
                       "peu enclins à prendre des risques.",
        "orientation": "Épargne régulière et progressive, supports diversifiés à profil prudent."},
    "jeune_aise": {
        "name": "Jeunes aisés, tolérants au risque",
        "capacity": "Élevée", "willingness": "Élevée",
        "description": "Jeunes ménages très diplômés, revenus les plus élevés, propriétaires, "
                       "et les plus disposés à prendre des risques.",
        "orientation": "Allocation dynamique à long terme (actifs de croissance diversifiés)."},
    "senior_aise": {
        "name": "Seniors aisés propriétaires",
        "capacity": "Élevée", "willingness": "Moyenne à élevée",
        "description": "Ménages âgés, propriétaires, patrimoine et revenus élevés, "
                       "tolérance au risque supérieure à la moyenne.",
        "orientation": "Diversification, préservation du capital, revenus complémentaires, transmission."},
    "senior_modeste": {
        "name": "Seniors modestes, très prudents",
        "capacity": "Faible", "willingness": "Faible",
        "description": "Ménages âgés au patrimoine et revenus limités, les plus averses au risque.",
        "orientation": "Sécurisation du capital, revenus stables, produits à capital garanti."},
}


def archetype(means: pd.Series) -> str:
    """Associe un centre de cluster (valeurs moyennes d'origine) à un profil."""
    if means["AGE"] < 3.5:
        if means["HHOUSES"] < 0.5:
            return "jeune_locataire"
        if means["RISK"] <= 2.8 and means["INCCL"] >= 4.0:
            return "jeune_aise"
        return "jeune_proprietaire"
    return "senior_aise" if means["NWCAT"] >= 3.5 else "senior_modeste"


# --------------------------------------------------------------------------
@dataclass
class Artifacts:
    scaler: StandardScaler
    kmeans: KMeans
    pca: PCA
    meta: dict
    results: pd.DataFrame


def _metrics(X: np.ndarray, lab: np.ndarray) -> dict:
    m = lab != -1                                   # DBSCAN : bruit exclu
    n_cl = len(set(lab[m]))
    out = {"Nb clusters": n_cl, "% bruit": round(100 * float((~m).mean()), 1)}
    if n_cl > 1:
        out.update({"Silhouette": float(silhouette_score(X[m], lab[m])),
                    "Davies-Bouldin": float(davies_bouldin_score(X[m], lab[m])),
                    "Calinski-Harabasz": float(calinski_harabasz_score(X[m], lab[m]))})
    return out


def train_and_save(model_dir: Path = MODEL_DIR, data_path: Path = DATA_PATH,
                   full: bool = True) -> Artifacts:
    """Entraîne le modèle, calcule tous les résultats et sauvegarde les artefacts.

    full=False saute la comparaison des 4 modèles et les courbes de choix de k
    (utile pour des tests rapides).
    """
    df = pd.read_excel(data_path)
    missing = [c for c in FEATURES + ["ID"] if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier : {missing}")
    if df[FEATURES].isna().any().any():
        raise ValueError("Valeurs manquantes détectées : nettoyez le fichier avant l'entraînement.")

    X_df = df[FEATURES]
    scaler = StandardScaler().fit(X_df)
    X = scaler.transform(X_df)

    km = KMeans(K, n_init=10, random_state=RANDOM_STATE).fit(X)
    labels = km.labels_
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X)
    coords = pca.transform(X)

    # --- profils ---------------------------------------------------------
    means = df[FEATURES].groupby(labels).mean()
    profiles, seen = [], {}
    for cid, row in means.iterrows():
        key = archetype(row)
        info = PROFILE_INFO[key]
        name = info["name"]
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name} (variante C{cid})"
        size = int((labels == cid).sum())
        profiles.append({"id": int(cid), "key": key, "name": name, "size": size,
                         "pct": round(100 * size / len(df), 1),
                         "capacity": info["capacity"], "willingness": info["willingness"],
                         "description": info["description"], "orientation": info["orientation"],
                         "means": {c: float(round(row[c], 3)) for c in FEATURES}})
    name_of = {p["id"]: p["name"] for p in profiles}

    # --- tableau de résultats (une ligne par individu) --------------------
    z = lambda s: (s - s.mean()) / s.std()
    results = df[["ID"] + FEATURES].copy()
    results["Cluster"] = labels
    results["Profil"] = results["Cluster"].map(name_of)
    results["PC1"], results["PC2"] = coords[:, 0], coords[:, 1]
    results["Capacite"] = (z(df["NWCAT"]) + z(df["INCCL"]) + z(df["HHOUSES"])) / 3
    results["Volonte"] = -z(df["RISK"])

    meta = {
        "n": int(len(df)), "k": K, "features": FEATURES,
        "silhouette_kmeans": float(silhouette_score(X, labels)),
        "pca_variance": float(pca.explained_variance_ratio_.sum()),
        "profiles": profiles,
        "ranges": {c: [int(df[c].min()), int(df[c].max())] for c in FEATURES},
        "medians": {c: int(df[c].median()) for c in FEATURES},
    }

    if full:
        ks = list(range(2, 9))
        fits = [KMeans(k, n_init=10, random_state=RANDOM_STATE).fit(X) for k in ks]
        meta["k_selection"] = {"k": ks, "inertia": [float(f.inertia_) for f in fits],
                               "silhouette": [float(silhouette_score(X, f.labels_)) for f in fits]}

        all_labels = {
            "K-Means": labels,
            "HAC (Ward)": AgglomerativeClustering(n_clusters=K, linkage="ward").fit(X).labels_,
            "DBSCAN": DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(X),
            "GMM": GaussianMixture(K, covariance_type="full", n_init=5,
                                   random_state=RANDOM_STATE).fit(X).predict(X)}
        meta["comparison"] = {n: _metrics(X, l) for n, l in all_labels.items()}
        names = list(all_labels)
        ari = {}
        for a in names:
            ari[a] = {}
            for b in names:
                m = (all_labels[a] != -1) & (all_labels[b] != -1)
                ari[a][b] = float(adjusted_rand_score(all_labels[a][m], all_labels[b][m]))
        meta["ari"] = ari
        ref = KMeans(K, n_init=10, random_state=0).fit(X).labels_
        aris = [adjusted_rand_score(ref, KMeans(K, n_init=1, random_state=s).fit(X).labels_)
                for s in range(1, 21)]
        meta["stability"] = {"mean": float(np.mean(aris)), "min": float(np.min(aris))}

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, model_dir / "scaler.joblib")
    joblib.dump(km, model_dir / "kmeans.joblib")
    joblib.dump(pca, model_dir / "pca.joblib")
    results.to_csv(model_dir / "results.csv", index=False)
    (model_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return Artifacts(scaler, km, pca, meta, results)


def load_artifacts(model_dir: Path = MODEL_DIR) -> Artifacts:
    """Charge les artefacts ; les ré-entraîne s'ils sont absents ou illisibles
    (par exemple après un changement de version de scikit-learn)."""
    model_dir = Path(model_dir)
    try:
        return Artifacts(
            scaler=joblib.load(model_dir / "scaler.joblib"),
            kmeans=joblib.load(model_dir / "kmeans.joblib"),
            pca=joblib.load(model_dir / "pca.joblib"),
            meta=json.loads((model_dir / "meta.json").read_text(encoding="utf-8")),
            results=pd.read_csv(model_dir / "results.csv"))
    except Exception:
        return train_and_save(model_dir)


def validate_profile(profile: dict, meta: dict) -> None:
    """Vérifie qu'un profil contient toutes les variables, dans les plages observées."""
    missing = [c for c in FEATURES if c not in profile]
    if missing:
        raise ValueError(f"Variables manquantes : {missing}")
    for c in FEATURES:
        lo, hi = meta["ranges"][c]
        if not (lo <= profile[c] <= hi):
            raise ValueError(f"{c}={profile[c]} hors de la plage observée [{lo}, {hi}]")


def predict(art: Artifacts, profile: dict) -> dict:
    """Rattache un nouvel individu au cluster dont le centre est le plus proche."""
    validate_profile(profile, art.meta)
    z = art.scaler.transform(pd.DataFrame([{c: profile[c] for c in FEATURES}]))
    dist = art.kmeans.transform(z)[0]
    cid = int(dist.argmin())
    inv = 1.0 / np.maximum(dist, 1e-9)
    share = inv / inv.sum()
    prof = next(p for p in art.meta["profiles"] if p["id"] == cid)
    order = np.sort(dist)
    return {"cluster": cid, "profile": prof, "distances": dist.tolist(),
            "proximity": share.tolist(),
            "margin": float(order[1] - order[0])}   # écart avec le 2e cluster le plus proche
