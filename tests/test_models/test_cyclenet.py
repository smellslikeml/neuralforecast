import numpy as np
import pandas as pd
import pytest
import torch

from neuralforecast import NeuralForecast
from neuralforecast.models import CycleNet


def _make_panel(period=7, n_series=2, n_steps=120):
    dates = pd.date_range("2020-01-01", periods=n_steps, freq="D")
    t = np.arange(n_steps)
    frames = []
    for uid in range(n_series):
        y = 10.0 + 2.0 * np.sin(2 * np.pi * t / period) + 0.05 * t + 0.1 * uid
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": dates,
                    "y": y,
                    "cycle_index": t % period,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _small_cyclenet(**kwargs):
    config = dict(
        h=12,
        input_size=24,
        cycle_len=7,
        max_steps=2,
        batch_size=2,
        windows_batch_size=8,
        inference_windows_batch_size=8,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    config.update(kwargs)
    return CycleNet(**config)


def test_cyclenet_registers_cycle_col():
    model = _small_cyclenet(cycle_col="position")
    assert model.EXOGENOUS_FUTR
    assert "position" in model.futr_exog_list
    assert model.cycle_col_idx == model.futr_exog_list.index("position")
    # A custom cycle_col is appended next to (not replacing) other futr exogs
    model = _small_cyclenet(futr_exog_list=["trend"], cycle_col="position")
    assert model.futr_exog_list == ["trend", "position"]


@pytest.mark.parametrize("model_type", ["linear", "mlp"])
def test_cyclenet_fit_predict(model_type):
    period = 7
    df = _make_panel(period=period)
    model = _small_cyclenet(model_type=model_type, d_model=16)
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=df)
    futr_df = nf.make_future_dataframe()
    offsets = (futr_df["ds"] - df["ds"].min()).dt.days
    futr_df["cycle_index"] = offsets % period
    predictions = nf.predict(futr_df=futr_df)

    assert len(predictions) == 2 * 12
    forecasts = predictions.filter(like="CycleNet")
    assert not forecasts.empty
    assert np.isfinite(forecasts.to_numpy()).all()


def test_cyclenet_save_load(tmp_path):
    period = 7
    df = _make_panel(period=period)
    model = _small_cyclenet()
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=df)
    nf.save(str(tmp_path))
    nf2 = NeuralForecast.load(str(tmp_path))

    horizon_dates = pd.date_range(
        df["ds"].max() + pd.Timedelta(days=1), periods=12, freq="D"
    )
    futr_df = pd.DataFrame(
        {
            "unique_id": np.repeat(sorted(df["unique_id"].unique()), 12),
            "ds": np.tile(horizon_dates, 2),
        }
    )
    offsets = (futr_df["ds"] - df["ds"].min()).dt.days
    futr_df["cycle_index"] = offsets % period
    predictions = nf2.predict(futr_df=futr_df)

    assert len(predictions) == 2 * 12
    forecasts = predictions.filter(like="CycleNet")
    assert not forecasts.empty
    assert np.isfinite(forecasts.to_numpy()).all()


@pytest.mark.parametrize("use_revin", [True, False])
def test_cyclenet_loss_decreases(use_revin):
    torch.manual_seed(0)
    batch_size, input_size, h, cycle_len = 8, 24, 12, 7
    t = torch.arange(input_size + h, dtype=torch.float32)
    cycle_index = torch.remainder(t, cycle_len).unsqueeze(0).repeat(batch_size, 1)
    y = 10.0 + 2.0 * torch.sin(2 * np.pi * t / cycle_len) + 0.05 * t
    y = y.unsqueeze(0).repeat(batch_size, 1)
    windows_batch = dict(
        insample_y=y[:, :input_size],
        insample_mask=torch.ones(batch_size, input_size),
        futr_exog=cycle_index.unsqueeze(-1),
        hist_exog=None,
        stat_exog=None,
    )
    model = _small_cyclenet(use_revin=use_revin)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()
    target = y[:, input_size:]
    losses = []
    for _ in range(2):
        optimizer.zero_grad()
        forecast = model(windows_batch)
        loss = criterion(forecast.squeeze(-1), target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[1] < losses[0]


def test_cyclenet_rejects_unsupported_config():
    with pytest.raises(Exception, match="model_type must be"):
        _small_cyclenet(model_type="transformer")
    with pytest.raises(Exception, match="scaler_type="):
        _small_cyclenet(scaler_type="robust")
