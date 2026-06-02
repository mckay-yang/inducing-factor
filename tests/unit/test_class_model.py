import tempfile
import os

import numpy as np
import pandas as pd
import pytest

from inducing_factor.core.class_model import DisturbanceClassEngine
from inducing_factor.core.models import InducingFactor


def _make_factor(
    factor_id: str = "IF_001",
    name: str = "test_factor",
    items: tuple = ("slope=high",),
) -> InducingFactor:
    return InducingFactor(
        factor_id=factor_id,
        name=name,
        semantics="test",
        items=items,
        support_event=0.5,
        support_background=0.1,
        relative_risk=5.0,
        stability=0.9,
        score=1.0,
    )


def _make_label_df(n_rows: int, factor: InducingFactor) -> pd.DataFrame:
    data = {"slope": ["high"] * n_rows}
    return pd.DataFrame(data)


def _make_training_samples(n_rows: int, class_vals: list) -> pd.DataFrame:
    return pd.DataFrame({"disturbance_class": class_vals[:n_rows]})


class TestDisturbanceClassEngine:
    def test_initial_status_is_untrained(self):
        engine = DisturbanceClassEngine()
        assert engine.model_status == "untrained"
        assert engine.is_available() is False
        assert engine.available_classes() == []

    def test_unavailable_on_all_zero_class_labels(self):
        engine = DisturbanceClassEngine()
        factor = _make_factor()
        label_df = _make_label_df(10, factor)
        training_samples = _make_training_samples(10, [0] * 10)

        engine.fit(label_df, training_samples, [factor])

        assert engine.model_status == "unavailable"
        assert engine.is_available() is False
        assert engine.available_classes() == []
        assert engine.model is None

    def test_unavailable_on_no_1_to_4_class_labels(self):
        engine = DisturbanceClassEngine()
        factor = _make_factor()
        label_df = _make_label_df(10, factor)
        training_samples = _make_training_samples(10, [5, 5, 5, 5, 5, 6, 6, 6, 6, 6])

        engine.fit(label_df, training_samples, [factor])

        assert engine.model_status == "unavailable"
        assert engine.is_available() is False

    def test_trained_on_valid_labels(self):
        engine = DisturbanceClassEngine()
        factor = _make_factor()
        n = 40
        label_df = _make_label_df(n, factor)
        classes = [1] * 20 + [2] * 20
        np.random.default_rng(42).shuffle(classes)
        training_samples = _make_training_samples(n, classes)

        engine.fit(label_df, training_samples, [factor])

        assert engine.model_status == "trained"
        assert engine.is_available() is True
        assert sorted(engine.available_classes()) == [1, 2]
        assert engine.model is not None

    def test_fallback_on_single_class(self):
        engine = DisturbanceClassEngine()
        factor = _make_factor()
        n = 20
        label_df = _make_label_df(n, factor)
        training_samples = _make_training_samples(n, [1] * n)

        engine.fit(label_df, training_samples, [factor])

        assert engine.model_status == "fallback_single_class"
        assert engine.is_available() is True
        assert engine.available_classes() == [1]

    def test_predict_proba_when_untrained_returns_empty(self):
        engine = DisturbanceClassEngine()
        factor = _make_factor()
        label_df = _make_label_df(5, factor)
        result = engine.predict_proba(label_df, [factor])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_predict_proba_when_unavailable_returns_empty(self):
        engine = DisturbanceClassEngine()
        engine.model_status = "unavailable"
        factor = _make_factor()
        label_df = _make_label_df(5, factor)
        result = engine.predict_proba(label_df, [factor])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_predict_proba_when_trained_returns_probs(self):
        engine = DisturbanceClassEngine()
        factor = _make_factor()
        n = 40
        label_df = _make_label_df(n, factor)
        classes = [1] * 20 + [2] * 20
        np.random.default_rng(42).shuffle(classes)
        training_samples = _make_training_samples(n, classes)
        engine.fit(label_df, training_samples, [factor])
        assert engine.model_status == "trained"

        test_label_df = _make_label_df(10, factor)
        result = engine.predict_proba(test_label_df, [factor])

        assert not result.empty
        assert len(result) == 10
        assert "class_probability_1" in result.columns
        assert "class_probability_2" in result.columns

    def test_save_load_preserves_model_status(self):
        engine = DisturbanceClassEngine()
        factor = _make_factor()
        n = 40
        label_df = _make_label_df(n, factor)
        classes = [1] * 20 + [2] * 20
        np.random.default_rng(42).shuffle(classes)
        training_samples = _make_training_samples(n, classes)
        engine.fit(label_df, training_samples, [factor])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            engine.save(path)

            engine2 = DisturbanceClassEngine()
            engine2.load(path)

            assert engine2.model_status == "trained"
            assert engine2.is_available() is True
            assert sorted(engine2.available_classes()) == [1, 2]

    def test_save_load_preserves_unavailable_status(self):
        engine = DisturbanceClassEngine()
        engine.model_status = "unavailable"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            engine.save(path)

            engine2 = DisturbanceClassEngine()
            engine2.load(path)

            assert engine2.model_status == "unavailable"
            assert engine2.is_available() is False

    def test_available_classes_empty_when_unavailable(self):
        engine = DisturbanceClassEngine()
        factor = _make_factor()
        label_df = _make_label_df(10, factor)
        training_samples = _make_training_samples(10, [0] * 10)
        engine.fit(label_df, training_samples, [factor])

        assert engine.available_classes() == []
