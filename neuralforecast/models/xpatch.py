# XPatch: dual-stream forecaster ported from https://github.com/stitsyuk/xPatch
# (Apache-2.0, Copyright 2024 Artyom Stitsyuk).

__all__ = ['XPatch']


from typing import Optional

import torch
from torch import nn

from ..common._base_model import BaseModel
from ..losses.pytorch import MAE


class RevIN(nn.Module):
    """
    Reversible Instance Normalization.
    """

    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) block to highlight the trend of time series.

    Device-agnostic port: the weights tensor derives its device from the input.
    """

    def __init__(self, alpha):
        super(EMA, self).__init__()
        self.alpha = alpha

    # Optimized implementation with O(1) time complexity
    def forward(self, x):
        # x: [Batch, Input, Channel]
        _, t, _ = x.shape
        powers = torch.flip(torch.arange(t, dtype=torch.double), dims=(0,))
        weights = torch.pow((1 - self.alpha), powers).to(x.device)
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        x = torch.cumsum(x * weights, dim=1)
        x = torch.div(x, divisor)
        return x.to(torch.float32)


class DEMA(nn.Module):
    """
    Double Exponential Moving Average (DEMA) block to highlight the trend of
    time series.

    Device-agnostic port: the smoothing factors are kept as plain python floats.
    """

    def __init__(self, alpha, beta):
        super(DEMA, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        # x: [Batch, Input, Channel]
        s_prev = x[:, 0, :]
        b = x[:, 1, :] - s_prev
        res = [s_prev.unsqueeze(1)]
        for t in range(1, x.shape[1]):
            xt = x[:, t, :]
            s = self.alpha * xt + (1 - self.alpha) * (s_prev + b)
            b = self.beta * (s - s_prev) + (1 - self.beta) * b
            s_prev = s
            res.append(s.unsqueeze(1))
        return torch.cat(res, dim=1)


class DECOMP(nn.Module):
    """
    Series decomposition block.
    """

    def __init__(self, ma_type, alpha, beta):
        super(DECOMP, self).__init__()
        if ma_type == "ema":
            self.ma = EMA(alpha)
        elif ma_type == "dema":
            self.ma = DEMA(alpha, beta)

    def forward(self, x):
        # x: [Batch, Input, Channel]
        moving_average = self.ma(x)
        res = x - moving_average
        return res, moving_average


class Network(nn.Module):
    """
    Dual-stream network: a non-linear patch + CNN stream for the seasonality
    and a linear MLP stream for the trend.
    """

    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, c_out=1):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.c_out = c_out

        # Non-linear Stream
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == "end":  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        # Patch Embedding
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        # CNN Depthwise
        self.conv1 = nn.Conv1d(
            self.patch_num,
            self.patch_num,
            patch_len,
            patch_len,
            groups=self.patch_num,
        )
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual Stream
        self.fc2 = nn.Linear(self.dim, patch_len)

        # CNN Pointwise
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # Flatten Head
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # Linear Stream
        # MLP
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # Streams Concatination
        self.fc8 = nn.Linear(pred_len * 2, pred_len * c_out)

    def forward(self, s, t):
        # s: [Batch, Input, Channel] - seasonality
        # t: [Batch, Input, Channel] - trend

        s = s.permute(0, 2, 1)  # to [Batch, Channel, Input]
        t = t.permute(0, 2, 1)  # to [Batch, Channel, Input]

        # Channel split for channel independence
        B = s.shape[0]  # Batch size
        C = s.shape[1]  # Channel size
        I = s.shape[2]  # Input size
        s = torch.reshape(s, (B * C, I))  # [Batch and Channel, Input]
        t = torch.reshape(t, (B * C, I))  # [Batch and Channel, Input]

        # Non-linear Stream
        # Patching
        if self.padding_patch == "end":
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: [Batch and Channel, Patch_num, Patch_len]

        # Patch Embedding
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        # CNN Depthwise
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        # Residual Stream
        res = self.fc2(res)
        s = s + res

        # CNN Pointwise
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        # Flatten Head
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        # Linear Stream
        # MLP
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t)

        # Streams Concatination
        x = torch.cat((s, t), dim=1)
        x = self.fc8(x)

        # Channel concatination
        x = torch.reshape(x, (B, C, self.pred_len * self.c_out))
        # x: [Batch, Channel, Output]

        x = x.permute(0, 2, 1)  # to [Batch, Output, Channel]

        return x


class XPatch(BaseModel):
    """XPatch

    The XPatch model is a dual-stream forecaster for univariate time series. It
    decomposes the input into a seasonal and a trend component with exponential
    moving averages (EMA/DEMA); the seasonality flows through a non-linear
    patching + CNN stream while the trend flows through a linear MLP stream.
    Both streams are concatenated and projected to the forecast horizon.

    * Requirements: windows of `input_size > stride` and even horizon `h`.

    Ported from the reference implementation by Artyom Stitsyuk.

    Args:
        h (int): forecast horizon.
        input_size (int): autorregresive inputs size, y=[1,2,3,4] input_size=2 -> y_[t-2:t]=[1,2].
        stat_exog_list (str list): static exogenous columns.
        hist_exog_list (str list): historic exogenous columns.
        futr_exog_list (str list): future exogenous columns.
        exclude_insample_y (bool): the model skips the autoregressive features y[t-input_size:t] if True.
        patch_len (int): length of patch. Note: patch_len = min(patch_len, input_size + stride).
        stride (int): stride of patch.
        ma_type (str): moving average for the seasonal-trend decomposition, one of ['ema', 'dema', 'reg'] ('reg' skips decomposition and feeds both streams with the input).
        alpha (float): smoothing factor of the EMA/DEMA decomposition.
        beta (float): trend smoothing factor of the DEMA decomposition.
        revin (bool): bool to use RevIn.
        revin_affine (bool): bool to use affine in RevIn.
        revin_subtract_last (bool): bool to use subtract last in RevIn.
        loss (PyTorch module): instantiated train loss class from [losses collection](./losses.pytorch.html).
        valid_loss (PyTorch module): instantiated valid loss class from [losses collection](./losses.pytorch.html).
        max_steps (int): maximum number of training steps.
        learning_rate (float): learning rate between (0, 1).
        num_lr_decays (int): number of learning rate decays, evenly distributed across max_steps.
        early_stop_patience_steps (int): number of validation iterations before early stopping.
        val_monitor (str): metric to monitor for early stopping. Valid options: "ptl/val_loss", "valid_loss", "train_loss". Default: "ptl/val_loss".
        val_check_steps (int): number of training steps between every validation loss check.
        batch_size (int): number of different series in each batch.
        valid_batch_size (int): number of different series in each validation and test batch, if None uses batch_size.
        windows_batch_size (int): number of windows to sample in each training batch, default uses all.
        inference_windows_batch_size (int): number of windows to sample in each inference batch.
        start_padding_enabled (bool): if True, the model will pad the time series with zeros at the beginning, by input size.
        training_data_availability_threshold (Union[float, List[float]]): minimum fraction of valid data points required for training windows. Single float applies to both insample and outsample; list of two floats specifies [insample_fraction, outsample_fraction]. Default 0.0 allows windows with only 1 valid data point (current behavior).
        step_size (int): step size between each window of temporal data.
        scaler_type (str): type of scaler for temporal inputs normalization see [temporal scalers](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/common/_scalers.py).
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
        - [Stitsyuk, A. (2024). "xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition"](https://arxiv.org/abs/2412.17323)
    """

    # Class attributes
    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    MULTIVARIATE = False  # If the model produces multivariate forecasts (True) or univariate (False)
    RECURRENT = (
        False  # If the model produces forecasts recursively (True) or direct (False)
    )

    def __init__(
        self,
        h,
        input_size,
        stat_exog_list=None,
        hist_exog_list=None,
        futr_exog_list=None,
        exclude_insample_y=False,
        patch_len: int = 16,
        stride: int = 8,
        ma_type: str = "ema",
        alpha: float = 0.3,
        beta: float = 0.3,
        revin: bool = True,
        revin_affine: bool = False,
        revin_subtract_last: bool = True,
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
        inference_windows_batch_size: int = 1024,
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
        super(XPatch, self).__init__(
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
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
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

        # Enforce correct patch_len, regardless of user input
        patch_len = min(input_size + stride, patch_len)

        c_out = self.loss.outputsize_multiplier

        # Fixed hyperparameters
        c_in = 1  # Always univariate
        padding_patch = "end"  # Padding at the end

        # Normalization
        self.revin = revin
        if revin:
            self.revin_layer = RevIN(
                c_in, affine=revin_affine, subtract_last=revin_subtract_last
            )

        # Moving Average decomposition
        self.ma_type = ma_type
        self.decomp = DECOMP(ma_type=ma_type, alpha=alpha, beta=beta)

        # Dual-stream network
        self.model = Network(
            seq_len=input_size,
            pred_len=h,
            patch_len=patch_len,
            stride=stride,
            padding_patch=padding_patch,
            c_out=c_out,
        )

    def forward(self, windows_batch):

        # Parse windows_batch
        x = windows_batch["insample_y"]  # x: [Batch, Input, 1]

        # Normalization
        if self.revin:
            x = self.revin_layer(x, "norm")

        if self.ma_type == "reg":  # No decomposition: input to both streams
            x = self.model(x, x)
        else:
            seasonal_init, trend_init = self.decomp(x)
            x = self.model(seasonal_init, trend_init)

        # Denormalization
        if self.revin:
            x = self.revin_layer(x, "denorm")

        forecast = x.reshape(x.shape[0], self.h, -1)  # x: [Batch, h, c_out]

        return forecast
