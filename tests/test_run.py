"""
Tests de los nodos de preparación de datos (data_cleaning y data_transformation).
Se ejecutan con: pytest
"""
import pandas as pd
import pytest

from evaluacion2_machine.pipelines.data_cleaning.nodes import (
    deduplicate_tracks,
    filter_zero_popularity,
    remove_invalid_values,
)
from evaluacion2_machine.pipelines.data_transformation.nodes import (
    create_targets,
    extract_features,
    scale_numerical,
    split_data,
)


@pytest.fixture
def tracks():
    return pd.DataFrame({
        "track_id": ["a", "a", "b", "c", "d"],
        "artists": ["X", "X", "Y;Z", "X", "W"],
        "track_name": ["s1", "s1", "s2", "s1", "s3"],
        "popularity": [40, 45, 60, 30, 0],
        "tempo": [120.0, 120.0, 0.0, 100.0, 90.0],
        "time_signature": [4, 4, 4, 4, 3],
        "duration_ms": [180_000, 180_000, 200_000, 240_000, 120_000],
        "explicit": [False, False, True, False, False],
        "track_genre": ["pop", "rock", "pop", "pop", "jazz"],
    })


class TestDataCleaning:
    def test_remove_invalid_values(self, tracks):
        result = remove_invalid_values(tracks, ["tempo", "time_signature", "duration_ms"])
        assert (result["tempo"] > 0).all()
        assert len(result) == 4

    def test_deduplicate_keeps_most_popular_copy(self, tracks):
        result = deduplicate_tracks(tracks, dedupe_by_name_artist=False)
        assert result["track_id"].is_unique
        assert result.loc[result["track_id"] == "a", "popularity"].iloc[0] == 45

    def test_deduplicate_by_name_and_artist(self, tracks):
        result = deduplicate_tracks(tracks, dedupe_by_name_artist=True)
        assert not result.duplicated(["track_name", "artists"]).any()

    def test_filter_zero_popularity(self, tracks):
        assert (filter_zero_popularity(tracks, True)["popularity"] > 0).all()
        assert len(filter_zero_popularity(tracks, False)) == len(tracks)


class TestDataTransformation:
    def test_extract_features(self, tracks):
        result = extract_features(tracks)
        assert "duration_ms" not in result.columns
        assert result["duration_min"].iloc[0] == 3.0
        assert result.loc[2, "n_artists"] == 2
        assert result.loc[2, "is_collab"] == 1
        assert result["explicit"].dtype.kind == "i"

    def test_create_targets(self, tracks):
        result = create_targets(tracks, hit_threshold=50)
        assert result["is_hit"].tolist() == [0, 0, 1, 0, 0]

    def test_split_excludes_targets_and_scaler_fits_on_train(self):
        df = pd.DataFrame({
            "energy": range(100),
            "popularity": range(100),
            "is_hit": [0] * 70 + [1] * 30,
        }).astype(float)
        X_train, X_test, y_train_clf, _, y_train_reg, _ = split_data(df, {"test_size": 0.2, "random_state": 42})

        assert {"popularity", "is_hit"}.isdisjoint(X_train.columns)
        assert len(X_train) == len(y_train_clf) == len(y_train_reg) == 80
        assert y_train_clf["is_hit"].mean() == pytest.approx(0.3)

        X_train_s, X_test_s, _ = scale_numerical(X_train, X_test, ["energy"])
        assert X_train_s["energy"].mean() == pytest.approx(0, abs=1e-9)
        assert X_test_s["energy"].mean() != pytest.approx(0, abs=1e-9)
