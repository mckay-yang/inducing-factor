import numpy as np
import pandas as pd
import pytest

from inducing_factor.core.class_conditional_analysis import FactorClassAnalyzer, FactorClassProfile
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


class DummyDiscretizer:
    def __init__(self, feature_columns):
        self.features_ = feature_columns
        self.feature_meta_ = {}
        for f in feature_columns:
            self.feature_meta_[f] = {"kind": "categorical", "categories": []}

    def transform(self, df, as_labels=False):
        result = pd.DataFrame(index=df.index)
        for col in df.columns:
            result[col] = df[col].astype(str)
        return result


def test_factor_class_profile_creation():
    profile = FactorClassProfile(
        factor_id="IF_TEST",
        factor_name="test_factor",
        factor_items=["slope=high"],
        count_in_fire=500, count_in_logging=10, count_in_geo_hazard=50,
        count_in_road_construction=40,
        count_in_unknown=4000, count_in_background=10000,
        total_events_with_factor=4600, total_labeled_with_factor=600,
        total_events=50000, total_labeled=49889,
        prob_fire_given_factor=500 / 600, prob_logging_given_factor=10 / 600,
        prob_geo_hazard_given_factor=50 / 600,
        prob_road_given_factor=40 / 600,
        support_in_fire=500 / 42228, support_in_logging=10 / 432,
        support_in_geo_hazard=50 / 2453, support_in_road=40 / 4647,
        prior_lift_fire=5.0, prior_lift_logging=1.0,
        prior_lift_geo=1.0, prior_lift_road=1.0,
        fisher_p_fire=1e-20, fisher_p_logging=1.0,
        fisher_p_geo=1.0, fisher_p_road=1.0,
    )
    assert profile.prior_lift_fire == 5.0
    assert profile.prob_fire_given_factor > 0.5


def test_compute_class_conditional_basic():
    rng = np.random.RandomState(42)
    n = 1000
    feature_columns = ["slope", "landuse"]

    data = {
        "disturbance_flag": np.ones(n, dtype=int),
        "disturbance_class": np.zeros(n, dtype=int),
        "slope": np.empty(n, dtype=object),
        "landuse": np.empty(n, dtype=object),
    }

    for i in range(n):
        if i < 400:
            data["disturbance_class"][i] = 1  # fire
            data["slope"][i] = "high"
            data["landuse"][i] = "forest" if i < 300 else "grass"
        elif i < 500:
            data["disturbance_class"][i] = 3  # geo_hazard
            data["slope"][i] = "medium"
            data["landuse"][i] = "grass"
        elif i < 600:
            data["disturbance_class"][i] = 4  # road
            data["slope"][i] = "low"
            data["landuse"][i] = "grass"
        else:
            data["disturbance_class"][i] = 0  # unknown
            data["slope"][i] = rng.choice(["low", "medium", "high"])
            data["landuse"][i] = rng.choice(["forest", "grass"])

    label_df = pd.DataFrame(data)
    discretizer = DummyDiscretizer(feature_columns)

    factors = [
        _make_factor("IF_001", "slope=high & landuse=forest", ("slope=high", "landuse=forest")),
        _make_factor("IF_002", "slope=medium & landuse=grass", ("slope=medium", "landuse=grass")),
    ]

    analyzer = FactorClassAnalyzer(min_count_per_class=1)
    profiles = analyzer.compute_class_conditional(
        factors=factors,
        label_df=label_df,
        feature_columns=feature_columns,
        discretizer=discretizer,
    )

    assert len(profiles) == 2
    p1 = profiles[0]
    assert p1.factor_id == "IF_001"
    assert p1.count_in_fire == 300
    assert p1.count_in_geo_hazard == 0
    assert p1.count_in_road_construction == 0
    assert p1.total_labeled_with_factor == 300
    assert p1.prob_fire_given_factor == pytest.approx(1.0, abs=1e-6)

    p2 = profiles[1]
    assert p2.factor_id == "IF_002"
    assert p2.count_in_fire == 0
    assert p2.count_in_geo_hazard == 100
    assert p2.prob_geo_hazard_given_factor == pytest.approx(1.0, abs=1e-6)


def test_find_top_class_specific_factors():
    profile_fire = FactorClassProfile(
        factor_id="IF_FIRE", factor_name="fire_factor",
        factor_items=["slope=high"],
        count_in_fire=500, count_in_logging=5, count_in_geo_hazard=10,
        count_in_road_construction=5,
        count_in_unknown=1000, count_in_background=5000,
        total_events_with_factor=1520, total_labeled_with_factor=520,
        total_events=50000, total_labeled=40000,
        prob_fire_given_factor=0.96, prob_logging_given_factor=0.01,
        prob_geo_hazard_given_factor=0.02, prob_road_given_factor=0.01,
        support_in_fire=0.01, support_in_logging=0.001,
        support_in_geo_hazard=0.004, support_in_road=0.001,
        prior_lift_fire=3.0, prior_lift_logging=0.5,
        prior_lift_geo=0.3, prior_lift_road=0.1,
        fisher_p_fire=1e-10, fisher_p_logging=1.0,
        fisher_p_geo=1.0, fisher_p_road=1.0,
    )
    profile_geo = FactorClassProfile(
        factor_id="IF_GEO", factor_name="geo_factor",
        factor_items=["slope=medium", "rainfall=high"],
        count_in_fire=20, count_in_logging=5, count_in_geo_hazard=200,
        count_in_road_construction=10,
        count_in_unknown=500, count_in_background=3000,
        total_events_with_factor=735, total_labeled_with_factor=235,
        total_events=50000, total_labeled=40000,
        prob_fire_given_factor=0.085, prob_logging_given_factor=0.021,
        prob_geo_hazard_given_factor=0.851, prob_road_given_factor=0.043,
        support_in_fire=0.004, support_in_logging=0.01,
        support_in_geo_hazard=0.008, support_in_road=0.002,
        prior_lift_fire=0.5, prior_lift_logging=1.0,
        prior_lift_geo=5.0, prior_lift_road=0.5,
        fisher_p_fire=1.0, fisher_p_logging=1.0,
        fisher_p_geo=1e-15, fisher_p_road=1.0,
    )

    analyzer = FactorClassAnalyzer(min_count_per_class=1)
    result = analyzer.find_top_class_specific_factors(
        [profile_fire, profile_geo], min_prior_lift=2.0, top_n=20,
    )

    assert len(result["fire"]) == 1
    assert result["fire"][0].factor_id == "IF_FIRE"
    assert len(result["geo_hazard"]) == 1
    assert result["geo_hazard"][0].factor_id == "IF_GEO"


def test_min_count_filter():
    profile_low = FactorClassProfile(
        factor_id="IF_LOW", factor_name="low_count_factor",
        factor_items=["slope=low"],
        count_in_fire=3, count_in_logging=1, count_in_geo_hazard=2,
        count_in_road_construction=1,
        count_in_unknown=50, count_in_background=200,
        total_events_with_factor=57, total_labeled_with_factor=7,
        total_events=50000, total_labeled=40000,
        prob_fire_given_factor=0.43, prob_logging_given_factor=0.14,
        prob_geo_hazard_given_factor=0.29, prob_road_given_factor=0.14,
        support_in_fire=0.001, support_in_logging=0.002,
        support_in_geo_hazard=0.001, support_in_road=0.0002,
        prior_lift_fire=3.0, prior_lift_logging=1.0,
        prior_lift_geo=2.5, prior_lift_road=1.0,
        fisher_p_fire=0.01, fisher_p_logging=1.0,
        fisher_p_geo=0.05, fisher_p_road=1.0,
    )

    analyzer = FactorClassAnalyzer(min_count_per_class=10)
    result = analyzer.find_top_class_specific_factors(
        [profile_low], min_prior_lift=2.0, top_n=20, min_count=10,
    )

    assert len(result["fire"]) == 0
    assert len(result["geo_hazard"]) == 0