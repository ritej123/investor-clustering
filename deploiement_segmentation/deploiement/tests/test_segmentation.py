"""Tests de la logique métier (lancer avec : pytest)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import segmentation as seg  # noqa: E402


@pytest.fixture(scope="module")
def art(tmp_path_factory):
    return seg.train_and_save(tmp_path_factory.mktemp("models"), full=False)


def test_all_individuals_clustered(art):
    assert len(art.results) == art.meta["n"] == 3866
    assert sum(p["size"] for p in art.meta["profiles"]) == art.meta["n"]
    assert art.results["Cluster"].nunique() == seg.K


def test_profile_names_are_unique(art):
    names = [p["name"] for p in art.meta["profiles"]]
    assert len(names) == len(set(names))


def test_predict_matches_training_labels(art):
    sample = art.results.sample(200, random_state=0)
    for _, row in sample.iterrows():
        out = seg.predict(art, {f: int(row[f]) for f in seg.FEATURES})
        assert out["cluster"] == int(row["Cluster"])


def test_proximity_sums_to_one(art):
    out = seg.predict(art, dict(art.meta["medians"]))
    assert pytest.approx(sum(out["proximity"]), abs=1e-9) == 1.0


def test_missing_variable_rejected(art):
    bad = dict(art.meta["medians"])
    bad.pop("RISK")
    with pytest.raises(ValueError):
        seg.predict(art, bad)


def test_out_of_range_rejected(art):
    bad = dict(art.meta["medians"])
    bad["AGE"] = 99
    with pytest.raises(ValueError):
        seg.predict(art, bad)


def test_save_and_reload(art, tmp_path):
    seg.train_and_save(tmp_path, full=False)
    loaded = seg.load_artifacts(tmp_path)
    assert loaded.meta["k"] == seg.K
    assert np.allclose(loaded.kmeans.cluster_centers_, art.kmeans.cluster_centers_)
