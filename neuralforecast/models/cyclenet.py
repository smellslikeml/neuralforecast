"""CycleNet: periodic residual forecasting.

Ported from the official implementation at
https://github.com/ACAT-SCUT/CycleNet (Apache License 2.0).
Lin, Shengsheng, et al. "CycleNet: Enhancing Time Series Forecasting through
Modeling Periodic Patterns." NeurIPS 2024. https://arxiv.org/abs/2409.18479
"""



__all__ = ['RecurrentCycle', 'CycleNet']


from typing import Optional, Union

import torch
import torch.nn as nn

from ..common._base_model import BaseModel
from ..losses.pytorch import MAE


class RecurrentCycle(nn.Module):
    """
    Learnable global cyclic pattern of length `cycle_len`, indexed by cyclic position.
    """

    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = nn.Parameter(torch.zeros(cycle_len, channel_size))

    def forward(self, index):
        # index: integer tensor of cyclic positions (in [0, cycle_len)),
        # returns self.data[index] with the shape of index plus the channel dim
        return self.data[torch.remainder(index.long(), self.cycle_len)]


class CycleNet(BaseModel):
    """CycleNet

    CycleNet is a simple and efficient univariate architecture based on Residual
    Cycle Forecasting: a learnable global cyclic pattern of length `cycle_len`
    is subtracted from the (optionally instance-normalized) input, a shared
    Linear (or MLP) backbone forecasts the residual for the horizon, and the
    cyclic pattern is added back on the forecasts.

    The model requires the cyclic positions of the series as a future exogenous
    column: add an integer column (named `cycle_index` by default, configurable
    through `cycle_col`) with the position of each timestamp within the cycle,
    for example `df["cycle_index"] = np.arange(len(df)) % cycle_len`, or the
    day of the week for a daily series with `cycle_len=7`. The column must be
    available both in the train data and in the prediction horizon (`futr_df`
    in `NeuralForecast.predict`).

    Ported from the official implementation in
    [ACAT-SCUT/CycleNet](https://github.com/ACAT-SCUT/CycleNet) (Apache-2.0).

    Args:
        h (int): forecast horizon.
        input_size (int): maximum sequence length for truncated train backpropagation.
        cycle_len (int): length of the cycle to model, e.g. 24 for hourly data with daily periodicity or 168 for hourly data with weekly periodicity. It must match the period of the `cycle_col` positions.
        model_type (str): backbone applied to the residual series, one of `"linear"` (default) or `"mlp"`.
        d_model (int): hidden dimension of the MLP backbone, ignored when `model_type="linear"`.
        use_revin (bool): if True, applies instance normalization (RevIN) to the input and reverses it on the forecasts.
        cycle_col (str): name of the future exogenous column containing the cyclic positions (integers in `[0, cycle_len)`), automatically appended to `futr_exog_list`.
        stat_exog_list (str list): static exogenous columns.
        hist_exog_list (str list): historic exogenous columns.
        futr_exog_list (str list): future exogenous columns.
        exclude_insample_y (bool): the model skips the autoregressive features y[t-input_size:t] if True.
        loss (PyTorch module): instantiated train loss class from [losses collection](./losses.pytorch.html).
        valid_loss (PyTorch module): instantiated valid loss class from [losses collection](./losses.pytorch.html).
        max_steps (int): maximum number of training steps.
        learning_rate (float): Learning rate between (0, 1).
        num_lr_decays (int): Number of learning rate decays, evenly distributed across max_steps.
        early_stop_patience_steps (int): Number of validation iterations before early stopping.
        val_monitor (str): metric to monitor for early stopping. Valid options: "ptl/val_loss", "valid_loss", "train_loss". Default: "ptl/val_loss".
        val_check_steps (int): Number of training steps between every validation loss check.
        batch_size (int): number of different series in each batch.
        valid_batch_size (int): number of different series in each validation and test batch, if None uses batch_size.
        windows_batch_size (int): number of windows to sample in each training batch, default uses all.
        inference_windows_batch_size (int): number of windows to sample in each inference batch.
        start_padding_enabled (bool): if True, the model will pad the time series with zeros at the beginning, by input size.
        training_data_availability_threshold (Union[float, List[float]]): minimum fraction of valid data points required for training windows. Single float applies to both insample and outsample; list of two floats specifies [insample_fraction, outsample_fraction]. Default 0.0 allows windows with only 1 valid data point (current behavior).
        step_size (int): step size between each window of temporal data.
        scaler_type (str): type of scaler for temporal inputs normalization see [temporal scalers](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/common/_scalers.py). CycleNet reads `cycle_col` as raw integer positions, so temporal scalers that transform exogenous columns are not supported: keep the default "identity".
        random_seed (int): random_seed for pytorch initializer and numpy generators.
        drop_last_loader (bool): if True `TimeSeriesDataLoader` drops last non-full batch.
        alias (str): optional,  Custom name of the model.
        optimizer (Subclass of 'torch.optim.Optimizer'): optional, user specified optimizer instead of the default choice (Adam).
        optimizer_kwargs (dict): optional, list of parameters used by the user specified `optimizer`.
        lr_scheduler (Subclass of 'torch.optim.lr_scheduler.LRScheduler'): optional, user specified lr_scheduler instead of the default choice (StepLR).
        lr_scheduler_kwargs (dict): optional, list of parameters used by the user specified `lr_scheduler`.
        dataloader_kwargs (dict): optional, list of parameters passed into the PyTorch Lightning dataloader by the `TimeSeriesDataLoader`.
        **trainer_kwargs (int):  keyword trainer arguments inherited from [PyTorch Lightning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).

    References:
        - [Lin, Shengsheng, et al. "CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns." Advances in Neural Information Processing Systems 37 (2024).](https://arxiv.org/abs/2409.18479)
    """

    # Class attributes
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    MULTIVARIATE = False  # If the model produces multivariate forecasts (True) or univariate (False)
    RECURRENT = (
        False  # If the model produces forecasts recursively (True) or direct (False)
    )

    def __init__(
        self,
        h: int,
        input_size: int,
        cycle_len: int,
        stat_exog_list=None,
        hist_exog_list=None,
        futr_exog_list=None,
        exclude_insample_y=False,
        model_type: str = "linear",
        d_model: int = 512,
        use_revin: bool = True,
        cycle_col: str = "cycle_index",
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 5000,
        learning_rate: float = 1e-4,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_monitor: str = "ptl/val_loss",
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size=1024,
        inference_windows_batch_size=1024,
        start_padding_enabled=False,
        training_data_availability_threshold=0.0,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        drop_last_loader: bool = False,
        alias: Optional[str] = None,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs
    ):
        # The cyclic positions are read from a future exogenous column
        futr_exog_list = list(futr_exog_list) if futr_exog_list is not None else []
        if cycle_col not in futr_exog_list:
            futr_exog_list = futr_exog_list + [cycle_col]

        super(CycleNet, self).__init__(
            h=h,
            input_size=input_size,
            stat_exog_list=stat_exog_list,
            hist_exog_list=hist_exog_list,
            futr_exog_list=futr_exog_list,
            exclude_insample_y=exclude_insample_y,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_monitor=val_monitor,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            windows_batch_size=windows_batch_size,
            valid_batch_size=valid_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            training_data_availability_threshold=training_data_availability_threshold,
            step_size=step_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            drop_last_loader=drop_last_loader,
            alias=alias,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs
        )

        # Architecture
        if model_type not in ["linear", "mlp"]:
            raise Exception(f"model_type must be 'linear' or 'mlp', got '{model_type}'")
        if scaler_type not in [None, "identity"]:
            raise Exception(
                f"CycleNet reads cyclic positions from the '{cycle_col}' future "
                "exogenous column as raw integers, so temporal scalers that "
                f"transform exogenous columns are not supported: scaler_type="
                f"'{scaler_type}' would corrupt them. Use scaler_type='identity'."
            )

        self.cycle_len = cycle_len
        self.model_type = model_type
        self.use_revin = use_revin
        self.cycle_col = cycle_col
        self.cycle_col_idx = self.futr_exog_list.index(cycle_col)

        self.c_out = self.loss.outputsize_multiplier
        self.output_attention = False
        self.enc_in = 1
        self.dec_in = 1

        # Learnable cyclic pattern, shared across series
        self.cycleQueue = RecurrentCycle(cycle_len=cycle_len, channel_size=1)

        # Backbone on the residual series, shared across series
        self.backbone: Union[nn.Linear, nn.Sequential]
        if model_type == "linear":
            self.backbone = nn.Linear(
                self.input_size, self.loss.outputsize_multiplier * h, bias=True
            )
        else:  # "mlp"
            self.backbone = nn.Sequential(
                nn.Linear(self.input_size, d_model),
                nn.ReLU(),
                nn.Linear(d_model, self.loss.outputsize_multiplier * h),
            )

    def forward(self, windows_batch):
        # Parse windows_batch
        insample_y = windows_batch["insample_y"].squeeze(-1)
        futr_exog = windows_batch["futr_exog"]  # [batch, input_size + h, F]

        # Parse inputs
        batch_size = len(insample_y)

        # Cyclic positions from the cycle_col channel: insample span + horizon
        cycle_index = futr_exog[:, :, self.cycle_col_idx]
        cycle = self.cycleQueue(cycle_index)  # [batch, input_size + h, 1]
        cycle_insample = cycle[:, : self.input_size, :]
        cycle_outsample = cycle[:, self.input_size :, :]

        x = insample_y.unsqueeze(-1)

        # Instance normalization (RevIN)
        if self.use_revin:
            seq_mean = x.mean(dim=1, keepdim=True)
            seq_var = x.var(dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # Remove the cyclic pattern from the input
        x = x - cycle_insample

        # Backbone on the residual series
        forecast = self.backbone(x.permute(0, 2, 1)).permute(0, 2, 1)
        forecast = forecast.reshape(batch_size, self.h, self.loss.outputsize_multiplier)

        # Add the cyclic pattern back on the horizon
        forecast = forecast + cycle_outsample

        # Instance denormalization
        if self.use_revin:
            forecast = forecast * torch.sqrt(seq_var) + seq_mean

        return forecast
