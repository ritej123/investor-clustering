"""Application Streamlit : segmentation des investisseurs selon leur capacité
et leur volonté de prendre des risques."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import segmentation as seg

st.set_page_config(page_title="Segmentation des investisseurs", page_icon="📊", layout="wide")

PAGES = ["Synthèse", "Profils des clusters", "Cartographie", "Simulateur",
         "Comparaison des modèles", "Données et export", "Méthodologie"]
CMAP = plt.get_cmap("tab10")
DISCLAIMER = ("Outil pédagogique : les pistes proposées sont indicatives et ne constituent "
              "pas un conseil en investissement.")


@st.cache_resource(show_spinner="Chargement du modèle (entraînement initial si nécessaire)…")
def get_artifacts():
    return seg.load_artifacts()


def color(cid):
    return CMAP(int(cid) % 10)


def profile_label(p):
    return f"C{p['id']} · {p['name']}"


# ---------------------------------------------------------------- graphiques
def fig_sizes(art):
    profs = art.meta["profiles"]
    fig, ax = plt.subplots(figsize=(8, 0.9 + 0.55 * len(profs)))
    ys = np.arange(len(profs))
    ax.barh(ys, [p["size"] for p in profs], color=[color(p["id"]) for p in profs])
    ax.set_yticks(ys)
    ax.set_yticklabels([profile_label(p) for p in profs])
    ax.invert_yaxis()
    for y, p in zip(ys, profs):
        ax.text(p["size"] + 10, y, f"{p['size']} ({p['pct']} %)", va="center", fontsize=9)
    ax.set_xlabel("Nombre d'individus")
    ax.set_xlim(0, max(p["size"] for p in profs) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_heatmap(art):
    profs = art.meta["profiles"]
    prof = pd.DataFrame([p["means"] for p in profs], index=[f"C{p['id']}" for p in profs])[seg.FEATURES]
    ref = art.results[seg.FEATURES]
    z = (prof - ref.mean()) / ref.std()
    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.6 * len(prof)))
    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(seg.FEATURES)))
    ax.set_xticklabels(seg.FEATURES, rotation=45, ha="right")
    ax.set_yticks(range(len(prof)))
    ax.set_yticklabels(prof.index)
    for i in range(prof.shape[0]):
        for j in range(prof.shape[1]):
            ax.text(j, i, f"{prof.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Écart à la moyenne (écarts-types)")
    fig.tight_layout()
    return fig


def fig_scatter(art, x, y, xlabel, ylabel, title):
    res = art.results
    fig, ax = plt.subplots(figsize=(8, 6))
    for p in art.meta["profiles"]:
        sub = res[res["Cluster"] == p["id"]]
        ax.scatter(sub[x], sub[y], s=10, alpha=0.45, color=color(p["id"]), label=profile_label(p))
        ax.scatter(sub[x].mean(), sub[y].mean(), s=220, marker="X", color=color(p["id"]),
                   edgecolor="black", linewidth=1.2)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best", markerscale=2)
    fig.tight_layout()
    return fig


def fig_ari(art):
    ari = pd.DataFrame(art.meta["ari"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(ari.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(ari)))
    ax.set_xticklabels(ari.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(ari)))
    ax.set_yticklabels(ari.index)
    for i in range(len(ari)):
        for j in range(len(ari)):
            ax.text(j, i, f"{ari.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if ari.values[i, j] > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def fig_k_selection(art):
    ks = art.meta["k_selection"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(ks["k"], ks["inertia"], "o-")
    ax[0].set_title("Méthode du coude (inertie)")
    ax[0].set_xlabel("k")
    ax[1].plot(ks["k"], ks["silhouette"], "o-", color="green")
    ax[1].set_title("Score de silhouette")
    ax[1].set_xlabel("k")
    for a in ax:
        a.axvline(art.meta["k"], color="red", ls="--", lw=1)
    fig.tight_layout()
    return fig


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# -------------------------------------------------------------------- pages
def page_summary(art):
    m = art.meta
    st.title("📊 Segmentation des investisseurs")
    st.caption("Regroupement des individus selon leur capacité et leur volonté de prendre des risques")
    c = st.columns(4)
    c[0].metric("Individus analysés", f"{m['n']:,}".replace(",", " "))
    c[1].metric("Clusters (K-Means)", m["k"])
    c[2].metric("Silhouette", f"{m['silhouette_kmeans']:.3f}")
    stab = m.get("stability")
    c[3].metric("Stabilité (ARI moyen)", f"{stab['mean']:.2f}" if stab else "n/d")

    st.subheader("Répartition des profils")
    show(fig_sizes(art))

    st.subheader("Les 5 profils en un coup d'œil")
    cols = st.columns(len(m["profiles"]))
    for col, p in zip(cols, m["profiles"]):
        with col:
            with st.container(border=True):
                st.markdown(f"**C{p['id']} · {p['name']}**")
                st.caption(f"{p['size']} individus ({p['pct']} %)")
                st.write(f"Capacité : **{p['capacity']}**")
                st.write(f"Volonté : **{p['willingness']}**")
    st.info("Les silhouettes sont faibles (autour de 0,13) : les profils se chevauchent. "
            "Il s'agit d'un continuum de comportements plutôt que de groupes parfaitement séparés.")
    st.caption(DISCLAIMER)


def page_profiles(art):
    st.title("Profils des clusters")
    st.write("Couleur : écart du profil moyen par rapport à la moyenne globale "
             "(rouge = plus élevé, bleu = plus bas). Attention : pour `RISK`, une valeur élevée "
             "signifie une **faible** volonté de prendre des risques.")
    show(fig_heatmap(art))
    for p in art.meta["profiles"]:
        with st.expander(f"{profile_label(p)}  ·  {p['size']} individus", expanded=False):
            a, b = st.columns([2, 1])
            with a:
                st.write(p["description"])
                st.markdown(f"**Piste d'orientation :** {p['orientation']}")
            with b:
                st.write(f"Capacité : **{p['capacity']}**")
                st.write(f"Volonté : **{p['willingness']}**")
            tbl = pd.DataFrame({"Variable": [seg.VARIABLES[f]["label"] for f in seg.FEATURES],
                                "Moyenne": [p["means"][f] for f in seg.FEATURES]})
            st.dataframe(tbl, hide_index=True, use_container_width=True)
    st.caption(DISCLAIMER)


def page_map(art):
    st.title("Cartographie des clusters")
    st.subheader("Projection ACP")
    st.caption(f"Les deux axes résument {art.meta['pca_variance'] * 100:.0f} % de l'information : "
               "les groupes paraissent plus mélangés à l'écran qu'ils ne le sont en réalité.")
    show(fig_scatter(art, "PC1", "PC2", "PC1", "PC2", "Clusters K-Means (projection ACP)"))
    st.subheader("Capacité vs volonté de prendre des risques")
    st.caption("Capacité = moyenne standardisée de NWCAT, INCCL et HHOUSES. "
               "Volonté = RISK inversé (plus c'est haut, plus l'individu accepte le risque). "
               "Les croix marquent le centre de chaque cluster.")
    show(fig_scatter(art, "Capacite", "Volonte", "Capacité", "Volonté",
                     "Capacité vs volonté de prendre des risques"))


def page_simulator(art):
    st.title("Simulateur de profil")
    st.write("Renseignez les caractéristiques d'un individu : il est rattaché au cluster "
             "dont le centre est le plus proche.")
    meds = art.meta["medians"]
    values = {}
    with st.form("simulateur"):
        cols = st.columns(3)
        for i, f in enumerate(seg.FEATURES):
            spec = seg.VARIABLES[f]
            opts = spec["options"]
            keys = list(opts.keys())
            default = keys.index(meds[f]) if meds[f] in keys else 0
            with cols[i % 3]:
                values[f] = st.selectbox(spec["label"], options=keys, index=default,
                                         format_func=lambda k, o=opts: o[k], key=f"in_{f}")
        submitted = st.form_submit_button("Identifier le profil", type="primary")
    if submitted:
        out = seg.predict(art, values)
        p = out["profile"]
        st.success(f"Profil identifié : **C{p['id']} · {p['name']}**")
        a, b = st.columns([1, 1])
        with a:
            st.write(p["description"])
            st.write(f"Capacité : **{p['capacity']}** · Volonté : **{p['willingness']}**")
            st.markdown(f"**Piste d'orientation :** {p['orientation']}")
            if out["margin"] < 0.3:
                st.warning("Ce profil est proche de la frontière entre deux clusters : "
                           "l'affectation est peu tranchée.")
        with b:
            prox = pd.DataFrame({"Proximité relative (%)": np.round(np.array(out["proximity"]) * 100, 1)},
                                index=[f"C{q['id']}" for q in art.meta["profiles"]])
            st.bar_chart(prox)
        cmp_tbl = pd.DataFrame({"Variable": [seg.VARIABLES[f]["label"] for f in seg.FEATURES],
                                "Individu saisi": [values[f] for f in seg.FEATURES],
                                "Moyenne du cluster": [p["means"][f] for f in seg.FEATURES]})
        st.dataframe(cmp_tbl, hide_index=True, use_container_width=True)
    st.caption(DISCLAIMER)


def page_models(art):
    m = art.meta
    st.title("Comparaison des modèles")
    comp = pd.DataFrame(m["comparison"]).T.round(3)
    st.dataframe(comp, use_container_width=True)
    st.markdown(
        "- **K-Means** est retenu : meilleures silhouette et Calinski-Harabasz, groupes équilibrés et lisibles.\n"
        "- Les métriques de **DBSCAN** sont calculées **hors bruit** et avec un nombre de clusters différent : "
        "elles ne sont pas strictement comparables aux autres modèles. Il ne trouve pas de zones denses bien "
        "séparées et sert surtout à repérer les cas atypiques.\n"
        "- Silhouette (plus haut = mieux), Davies-Bouldin (plus bas = mieux), Calinski-Harabasz (plus haut = mieux).")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Accord entre modèles (ARI)")
        show(fig_ari(art))
    with c2:
        st.subheader("Stabilité de K-Means")
        s = m["stability"]
        st.metric("ARI moyen sur 20 relances", f"{s['mean']:.3f}")
        st.metric("ARI minimum", f"{s['min']:.3f}")
        st.caption("Les groupes sont globalement reproductibles ; les individus situés aux "
                   "frontières peuvent changer de cluster d'une initialisation à l'autre.")
    st.subheader("Choix du nombre de clusters")
    show(fig_k_selection(art))
    st.caption("Les métriques hésitent entre 2 et 5 clusters et restent faibles : k = 5 est un compromis "
               "entre critères statistiques et interprétabilité (il distingue deux types de jeunes aisés).")


def page_data(art):
    st.title("Données et export")
    res = art.results
    sel = st.multiselect("Filtrer par profil", options=[profile_label(p) for p in art.meta["profiles"]],
                         default=[profile_label(p) for p in art.meta["profiles"]])
    ids = [int(s.split(" · ")[0][1:]) for s in sel]
    view = res[res["Cluster"].isin(ids)]
    st.write(f"{len(view)} individus affichés")
    st.dataframe(view, hide_index=True, use_container_width=True)
    st.download_button("Télécharger la sélection (CSV)", data=view.to_csv(index=False).encode("utf-8"),
                       file_name="resultats_segmentation.csv", mime="text/csv")


def page_method(art):
    st.title("Méthodologie")
    st.markdown("""
**Objectif.** Regrouper des individus selon leur capacité et leur volonté de prendre des risques,
à partir de variables démographiques et financières.

**Données.** 3 866 individus, 12 variables ordinales (identifiant exclu), sans valeur manquante ni doublon.

**Pipeline.** Standardisation (StandardScaler) → K-Means (k = 5, 10 initialisations, graine fixée) →
profilage des clusters → comparaison avec HAC (Ward), DBSCAN et GMM → validation (silhouette,
Davies-Bouldin, Calinski-Harabasz, ARI, stabilité sur 20 relances).

**Limites.**
- Variables ordinales traitées comme numériques.
- Silhouettes faibles : les profils se chevauchent.
- Données d'enquête, sans validation externe ; le sens exact des codes doit être confirmé
  avec la documentation officielle du jeu de données.
- Les pistes d'orientation sont indicatives.
""")
    st.subheader("Dictionnaire des variables")
    rows = [{"Variable": f, "Libellé": v["label"],
             "Modalités": " | ".join(v["options"].values()) if f != "KIDS" else "0 à 8"}
            for f, v in seg.VARIABLES.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Libellés d'après la documentation de la base américaine SCF : à vérifier.")
    st.caption(DISCLAIMER)


# --------------------------------------------------------------------- main
def main():
    art = get_artifacts()
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Page", PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption(f"Modèle : K-Means, k = {art.meta['k']} · n = {art.meta['n']}")
    st.sidebar.caption(DISCLAIMER)
    {"Synthèse": page_summary, "Profils des clusters": page_profiles, "Cartographie": page_map,
     "Simulateur": page_simulator, "Comparaison des modèles": page_models,
     "Données et export": page_data, "Méthodologie": page_method}[page](art)


main()
