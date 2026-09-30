from neuralforecast import NeuralForecast
from neuralforecast.common._model_checks import check_model
from neuralforecast.losses.pytorch import MAE, MQLoss
from neuralforecast.models import SRSNet
from neuralforecast.utils import AirPassengersPanel


def test_srsnet_model(suppress_warnings):
    check_model(SRSNet, ["airpassengers"])


def _air_passengers_split(h=12):
    Y_train_df = AirPassengersPanel[
        AirPassengersPanel.ds < AirPassengersPanel["ds"].values[-h]
    ].reset_index(drop=True)
    Y_test_df = AirPassengersPanel[
        AirPassengersPanel.ds >= AirPassengersPanel["ds"].values[-h]
    ].reset_index(drop=True)
    return Y_train_df, Y_test_df


def test_srsnet_smoke_forecast():
    # CPU smoke test: fit 2 steps on a small panel and forecast the full horizon
    h = 12
    Y_train_df, Y_test_df = _air_passengers_split(h=h)

    model = SRSNet(
        h=h,
        input_size=104,
        patch_len=16,
        stride=8,
        d_model=64,
        hidden_size=32,
        dropout=0.2,
        alpha=2.0,
        pos=True,
        loss=MAE(),
        max_steps=2,
        enable_progress_bar=False,
        enable_model_summary=False,
        val_check_steps=2,
        batch_size=8,
        windows_batch_size=8,
        valid_batch_size=8,
        inference_windows_batch_size=8,
    )
    fcst = NeuralForecast(models=[model], freq="ME")
    fcst.fit(df=Y_train_df)
    forecasts = fcst.predict(futr_df=Y_test_df)

    n_series = Y_test_df["unique_id"].nunique()
    assert forecasts.shape[0] == n_series * h
    assert "SRSNet" in forecasts.columns
    assert forecasts["SRSNet"].notna().all()


def test_srsnet_smoke_quantile_forecast():
    # the flatten head predicts h * outputsize_multiplier outputs,
    # check the quantile loss wiring end to end
    h = 12
    Y_train_df, Y_test_df = _air_passengers_split(h=h)

    model = SRSNet(
        h=h,
        input_size=104,
        patch_len=16,
        stride=8,
        d_model=64,
        hidden_size=32,
        loss=MQLoss(),
        max_steps=2,
        enable_progress_bar=False,
        enable_model_summary=False,
        val_check_steps=2,
        batch_size=8,
        windows_batch_size=8,
        valid_batch_size=8,
        inference_windows_batch_size=8,
    )
    fcst = NeuralForecast(models=[model], freq="ME")
    fcst.fit(df=Y_train_df)
    forecasts = fcst.predict(futr_df=Y_test_df, level=[80])

    n_series = Y_test_df["unique_id"].nunique()
    model_cols = [col for col in forecasts.columns if col.startswith("SRSNet")]
    assert forecasts.shape[0] == n_series * h
    assert len(model_cols) > 1  # point forecast + interval bounds
    for col in model_cols:
        assert forecasts[col].notna().all()
