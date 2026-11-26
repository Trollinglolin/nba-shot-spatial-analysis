"""Quick sanity checks: python -W ignore tests_smoke.py"""
import numpy as np
import features as F
import ingest


def test_geometry():
    assert F.is_three(0, 240) and not F.is_three(0, 200)
    assert F.is_three(225, 0) and not F.is_three(215, 0)
    z = F.assign_zone([0, 0, 230, 0, 0], [10, 100, 20, 250, 180], F.is_three([0, 0, 230, 0, 0], [10, 100, 20, 250, 180]))
    assert list(z) == ["Restricted Area", "Paint (Non-RA)", "Corner 3", "Above-Break 3", "Mid-Range"]


def test_heaves_removed():
    df = ingest.generate_synthetic(n_players=3, seasons=[2019])
    out = ingest.clean(df)
    assert out["shot_distance"].max() <= 40


def test_nba_api_path(tmp=None):
    """Fake ShotChartDetail responses: exercises fetch (cache + retry) and normalization offline."""
    import tempfile
    import pandas as pd
    from nba_api.stats.endpoints import shotchartdetail as scd
    calls = {"n": 0}

    class Fake:
        def __init__(self, team_id, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise TimeoutError("simulated")          # first retry path
            self.d = pd.DataFrame({
                "GAME_ID": ["0022300001"], "GAME_EVENT_ID": [1], "PLAYER_ID": [1], "PLAYER_NAME": ["A B"],
                "TEAM_ID": [team_id], "TEAM_NAME": ["T"], "PERIOD": [1], "MINUTES_REMAINING": [5],
                "SECONDS_REMAINING": [10], "EVENT_TYPE": ["Made Shot"], "ACTION_TYPE": ["Jump Shot"],
                "ACTION_TYPE": ["Pullup Jump shot"], "SHOT_TYPE": ["3PT Field Goal"], "SHOT_DISTANCE": [25], "LOC_X": [10], "LOC_Y": [250],
                "SHOT_ATTEMPTED_FLAG": [1], "SHOT_MADE_FLAG": [1], "GAME_DATE": ["20231024"]})

        def get_data_frames(self):
            return [self.d.copy()]

    orig, ingest.time.sleep = scd.ShotChartDetail, lambda *_: None
    scd.ShotChartDetail = Fake
    try:
        d = tempfile.mkdtemp()
        raw = ingest.fetch_nba_api(["2023-24"], cache_dir=d, pause=0)
        assert len(raw) == 30                            # 30 teams, one retried
        n = calls["n"]
        ingest.fetch_nba_api(["2023-24"], cache_dir=d, pause=0)
        assert calls["n"] == n                           # second run is fully cached
        out = ingest.clean(ingest.normalize_nba_api(raw))
        assert list(out.columns) == ingest.COLUMNS and out["shot_made_flag"].eq(1).all()
        assert ingest.normalize_nba_api(raw, flip_x=True)["loc_x"].iloc[0] == -10
    finally:
        scd.ShotChartDetail = orig


if __name__ == "__main__":
    test_nba_api_path()
    test_geometry(); test_heaves_removed(); print("ok")
