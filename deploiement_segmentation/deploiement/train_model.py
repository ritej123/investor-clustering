"""Entraîne le modèle et génère les artefacts dans models/.

Usage :  python train_model.py
"""
import segmentation as seg

if __name__ == "__main__":
    art = seg.train_and_save()
    print(f"Modèle entraîné : K={art.meta['k']}, n={art.meta['n']}, "
          f"silhouette={art.meta['silhouette_kmeans']:.3f}")
    for p in art.meta["profiles"]:
        print(f"  C{p['id']} · {p['name']} ({p['size']} individus, {p['pct']} %)")
    s = art.meta.get("stability")
    if s:
        print(f"Stabilité (ARI sur 20 relances) : moyenne {s['mean']:.3f}, min {s['min']:.3f}")
    print("Artefacts enregistrés dans", seg.MODEL_DIR)
