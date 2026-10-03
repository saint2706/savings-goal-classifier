from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from savings_goal import api

ONBOARDING: dict[str, Any] = {
    "annual_income": 120000,
    "household_size": 5,
    "age_dependents": 2,
    "head_age": 45,
    "max_adult_education": 10,
    "outstanding_debt": 20000,
    "occupation": "Salaried",
    "area_type": "Other_Urban",
    "caste_group": "OBC",
    "religion": "Hindu",
}
ORIGINAL_LOAD_TIER = api.load_tier
SHARES = {"Groceries": 0.5, "Utilities": 0.1, "Healthcare": 0.1, "Miscellaneous": 0.3}


@pytest.fixture(scope="module")
def client(
    features: pd.DataFrame,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch_module: pytest.MonkeyPatch,
) -> TestClient:
    artifacts = tmp_path_factory.mktemp("artifacts")
    scores = api.train_tiers(features, artifacts)
    assert set(scores) == {"tier1 held-out ROC-AUC", "tier2 held-out ROC-AUC"}
    monkeypatch_module.setattr(api, "ARTIFACTS", artifacts)
    api.load_tier.cache_clear()
    original = api.load_tier.__wrapped__

    def load(tier: int, artifacts_dir: Path = artifacts) -> Any:
        return original(tier, artifacts_dir)

    monkeypatch_module.setattr(api, "load_tier", load)
    return TestClient(api.create_app())


@pytest.fixture(scope="module")
def monkeypatch_module() -> Any:
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_onboarding_score(client: TestClient) -> None:
    r = client.post("/score/onboarding", json=ONBOARDING)
    assert r.status_code == 200
    body = r.json()
    assert body["tier"] == 1 and 0 <= body["p_on_track"] <= 1
    assert body["at_risk_score"] == pytest.approx(1 - body["p_on_track"])
    minimal = {"annual_income": 50000, "household_size": 3, "age_dependents": 0}
    assert client.post("/score/onboarding", json=minimal).status_code == 200


def test_onboarding_rejects_bad_input(client: TestClient) -> None:
    assert (
        client.post("/score/onboarding", json={**ONBOARDING, "annual_income": -1}).status_code
        == 422
    )
    assert (
        client.post("/score/onboarding", json={**ONBOARDING, "age_dependents": 9}).status_code
        == 422
    )
    bad_cat = client.post("/score/onboarding", json={**ONBOARDING, "occupation": "Astronaut"})
    assert bad_cat.status_code == 422


def test_tier2_requires_prior_shares(client: TestClient) -> None:
    ok = {
        **ONBOARDING,
        "shares": SHARES,
        "shares_observed_through": "2024-03-31",
        "prediction_window_start": "2024-04-01",
    }
    r = client.post("/score/with-spending", json=ok)
    assert r.status_code == 200 and r.json()["tier"] == 2
    same_period = {**ok, "shares_observed_through": "2024-04-01"}
    assert client.post("/score/with-spending", json=same_period).status_code == 422
    bad_sum = {**ok, "shares": {"Groceries": 0.5}}
    assert client.post("/score/with-spending", json=bad_sum).status_code == 422
    unknown = {**ok, "shares": {**SHARES, "Yachts": 0.0}}
    assert client.post("/score/with-spending", json=unknown).status_code == 422


def test_missing_artifacts_is_503(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(tier: int) -> Any:
        raise FileNotFoundError("no model")

    monkeypatch.setattr(api, "load_tier", missing)
    r = TestClient(api.create_app()).post("/score/onboarding", json=ONBOARDING)
    assert r.status_code == 503


def test_load_tier_without_artifacts(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="sgc train"):
        ORIGINAL_LOAD_TIER.__wrapped__(1, tmp_path)
