import pandas as pd
from src.main_gradcam_classifier import to_log_dataframe


def test_to_log_dataframe_handles_scalar_logs():
    logs = {"loss": 0.5, "acc": 0.8, "acc_cnt": 4, "total_cnt": 5}
    df = to_log_dataframe(logs)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 4)
    assert df.loc[0, "loss"] == 0.5


def test_to_log_dataframe_handles_epoch_logs():
    logs = {"loss": [0.5, 0.6], "acc": [0.8, 0.9], "acc_cnt": [4, 5], "total_cnt": [5, 6]}
    df = to_log_dataframe(logs)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 4)
    assert df.loc[1, "acc"] == 0.9
