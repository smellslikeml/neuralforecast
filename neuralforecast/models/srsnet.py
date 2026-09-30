

# SRSNet ported with attribution from the MIT-licensed reference implementation
# https://github.com/decisionintelligence/SRSNet (Copyright (c) 2024 Huawei Technologies):
# the `SRS` layer (ts_benchmark/baselines/srsnet/layers/SRS.py) and the model
# architecture (ts_benchmark/baselines/srsnet/models/srsnet_model.py:
# RevIN -> SRS patch embedding -> FlattenHead -> RevIN) are ported faithfully.
# The reference's TFB training harness (ts_benchmark config, scripts and wrapper)
# is intentionally NOT ported: neuralforecast's BaseModel owns the training loop,
# data windowing, scaling and loss, and neuralforecast's own RevIN is reused.

__all__ = ["PositionalEmbedding", "SRS", "FlattenHead", "SRSNet_backbone", "SRSNet"]


import math
from typing import Optional

import torch
import torch.nn as nn

from ..common._base_model import BaseModel
from ..common._modules import RevIN
from ..losses.pytorch import MAE


class PositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for the patch (representation) dimension.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class SRS(nn.Module):
    """
    Selective Representation Spaces (SRS) patch embedding.

    Instead of a single fixed patch grid, the padded sequence is encoded into two
    representation spaces: the `original` view given by the fixed-stride patching,
    and a `reconstruction` view whose patches are dynamically selected among all
    stride-one candidate patches (learned scorer) and reassembled in a learned order
    (shuffle scorer). Both views are linearly embedded and combined with a learnable
    adaptive weight (element-wise sigmoid of `alpha`), plus a positional embedding.
    """

    def __init__(self, d_model, patch_len, stride, seq_len, dropout, hidden_size,
                 alpha=2.0, pos=True):
        super(SRS, self).__init__()

        self.patch_len = patch_len
        self.stride = stride
        self.seq_len = seq_len

        self.patch_num = math.ceil((self.seq_len - self.patch_len) / self.stride) + 1
        self.padding = (
            self.patch_len + (self.patch_num - 1) * self.stride - self.seq_len
        )
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.padding))
        self.scorer_select = nn.Sequential(
            nn.Linear(self.patch_len, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.patch_num),
        )

        self.scorer_shuffle = nn.Sequential(
            nn.Linear(self.patch_len, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding_org = nn.Linear(patch_len, d_model, bias=False)
        self.value_embedding_rec = nn.Linear(patch_len, d_model, bias=False)
        # Positional embedding
        if pos:
            self.position_embedding = PositionalEmbedding(d_model)

        self.pos = pos

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Adaptive weight between Original View and Reconstruction View
        self.alpha = nn.Parameter(torch.ones(self.patch_num, d_model) * alpha)

    def _origin_view(self, x):
        # [batch_size, n_vars, patch_num, patch_size]
        x_origin = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # [batch_size * n_vars, patch_num, patch_size]
        origin_patches = x_origin.reshape(
            x_origin.shape[0] * x_origin.shape[1], x_origin.shape[2], x_origin.shape[3]
        )
        return origin_patches

    def _rec_view(self, x):
        # [batch_size, n_vars, seq_len - patch_size + 1, patch_size]
        x_rec = x.unfold(dimension=-1, size=self.patch_len, step=1)
        # [batch_size, n_vars, patch_num, patch_size]
        selected_patches = self._select(x_rec)
        # [batch_size, n_vars, patch_num, patch_size]
        shuffled_patches = self._shuffle(selected_patches)
        # [batch_size * n_vars, patch_num, patch_size]
        rec_patches = shuffled_patches.reshape(
            shuffled_patches.shape[0] * shuffled_patches.shape[1],
            shuffled_patches.shape[2],
            shuffled_patches.shape[3],
        )
        return rec_patches

    def _select(self, x_rec):
        # [batch_size, n_vars, seq_len - patch_size + 1, select_num]  select_num = original patch_num
        scores = self.scorer_select(x_rec)
        # [batch_size, n_vars, 1, select_num]
        indices = torch.argmax(scores, dim=-2, keepdim=True)
        # [batch_size, n_vars, 1, select_num]
        max_scores = torch.gather(input=scores, dim=-2, index=indices)
        non_zero_mask = max_scores != 0
        inv = (1 / max_scores[non_zero_mask]).detach()

        # [batch_size, n_vars, select_num, patch_size]
        x_rec_indices = indices.repeat(1, 1, self.patch_len, 1).permute(0, 1, 3, 2)
        # [batch_size, n_vars, select_num, patch_size]
        selected_patches = torch.gather(input=x_rec, index=x_rec_indices, dim=-2)

        # Normalize the selected (max) scores to 1 in the forward pass while
        # keeping their gradients, i.e. s * (1 / s).detach()
        max_scores[non_zero_mask] *= inv
        # [batch_size, n_vars, select_num, patch_size]
        selected_patches = max_scores.permute(0, 1, 3, 2) * selected_patches

        return selected_patches

    def _shuffle(self, selected_patches):
        # [batch_size, n_vars, patch_num, 1]
        shuffle_scores = self.scorer_shuffle(selected_patches)
        # [batch_size, n_vars, patch_num, 1]
        shuffle_indices = torch.argsort(
            input=shuffle_scores, dim=-2, descending=True
        )
        # [batch_size, n_vars, patch_num, 1]
        shuffled_scores = torch.gather(
            input=shuffle_scores, index=shuffle_indices, dim=-2
        )
        non_zero_mask = shuffled_scores != 0
        inv = (1 / shuffled_scores[non_zero_mask]).detach()

        # [batch_size, n_vars, patch_num, patch_size]
        shuffle_patch_indices = shuffle_indices.repeat(1, 1, 1, self.patch_len)
        # [batch_size, n_vars, patch_num, patch_size]
        shuffled_patches = torch.gather(
            input=selected_patches, index=shuffle_patch_indices, dim=-2
        )
        # Normalize the shuffled scores to 1 in the forward pass while keeping
        # their gradients, i.e. s * (1 / s).detach()
        shuffled_scores[non_zero_mask] *= inv
        # [batch_size, n_vars, patch_num, patch_size]
        shuffled_patches = shuffled_scores * shuffled_patches

        return shuffled_patches

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        # padding for the original stride
        x = self.padding_patch_layer(x)

        # [batch_size * n_vars, patch_num, patch_size]
        rec_repr_space = self._rec_view(x)
        # [batch_size * n_vars, patch_num, patch_size]
        original_repr_space = self._origin_view(x)
        # The adaptive weight between the two views
        weight = torch.sigmoid(self.alpha)
        # [batch_size * n_vars, patch_num, d_model]
        embedding = (weight * self.value_embedding_org(original_repr_space)
                     + (1 - weight) * self.value_embedding_rec(rec_repr_space))

        if self.pos:
            position_embedding = self.position_embedding(original_repr_space)
            embedding = embedding + position_embedding

        return self.dropout(embedding), n_vars


class FlattenHead(nn.Module):
    """
    FlattenHead

    Flattens the `[bs x nvars x d_model x patch_num]` patch representations and
    projects them to the target window.
    """

    def __init__(self, n_vars, nf, target_window, head_dropout=0, mode="linear"):
        super(FlattenHead, self).__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        if mode == "linear":
            self.head = nn.Linear(nf, target_window)
        else:
            self.head = nn.Sequential(nn.Linear(nf, nf // 2), nn.SiLU(),
                                      nn.Linear(nf // 2, target_window))
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.head(x)
        x = self.dropout(x)
        return x


class SRSNet_backbone(nn.Module):
    """
    SRSNet_backbone

    Channel-independent backbone: RevIN -> SRS patch embedding -> FlattenHead -> RevIN.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        input_size: int,
        h: int,
        patch_len: int,
        stride: int,
        d_model: int = 512,
        dropout: float = 0.2,
        hidden_size: int = 128,
        alpha: float = 2.0,
        pos: bool = True,
        head_mode: str = "linear",
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
    ):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Selective representation space patch embedding
        self.patch_embedding = SRS(
            d_model,
            patch_len,
            stride,
            input_size,
            dropout,
            hidden_size,
            alpha,
            pos,
        )

        # Prediction Head
        # (the head flattens d_model x patch_num representations, hence the reference's
        # head_nf = d_model * (ceil((input_size - patch_len) / stride) + 1))
        self.head_nf = d_model * self.patch_embedding.patch_num
        self.head = FlattenHead(
            c_in,
            self.head_nf,
            h * c_out,  # target_window: h point forecasts per loss output size
            head_dropout=dropout,
            mode=head_mode,
        )

    def forward(self, z):  # z: [bs x nvars x seq_len]
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)

        # do selective patching and embedding
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(z)

        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x h * c_out]

        # denorm
        if self.revin:
            dec_out = dec_out.permute(0, 2, 1)
            dec_out = self.revin_layer(dec_out, "denorm")
            dec_out = dec_out.permute(0, 2, 1)
        return dec_out


class SRSNet(BaseModel):
    """SRSNet

    The SRSNet model is a channel-independent patch-based model for univariate time series
    forecasting. Like PatchTST, it segments the lookback window into patches and predicts
    through a linear flatten head, but it replaces the fixed patch grid with a Selective
    Representation Spaces (SRS) module: a learned patch *selection* over all stride-one
    candidate patches and a dynamic *reassembly* of the selected patches. Both the original
    (fixed-stride) and the reconstructed representation spaces are embedded and combined
    with a learnable adaptive weight, so the model attends to the most informative patch
    representations rather than a fixed grid.

    Ported with attribution from the MIT-licensed reference implementation
    [decisionintelligence/SRSNet](https://github.com/decisionintelligence/SRSNet), adapted
    to neuralforecast (neuralforecast's `BaseModel` owns the training loop, data windowing,
    scaling and loss; neuralforecast's own `RevIN` is reused).

    Args:
        h (int): forecast horizon.
        input_size (int): autorregresive inputs size, y=[1,2,3,4] input_size=2 -> y_[t-2:t]=[1,2].
        stat_exog_list (str list): static exogenous columns.
        hist_exog_list (str list): historic exogenous columns.
        futr_exog_list (str list): future exogenous columns.
        exclude_insample_y (bool): the model skips the autoregressive features y[t-input_size:t] if True.
        patch_len (int): length of patch. Note: patch_len = min(patch_len, input_size).
        stride (int): stride of the original (fixed-stride) patch view.
        d_model (int): units of the patch value embeddings (and of the flatten head input).
        hidden_size (int): units of the SRS scorers' hidden layer (patch selection and reassembly).
        dropout (float): dropout rate for the SRS patch embedding and the flatten head.
        alpha (float): initial value of the adaptive weight between the original and the
            reconstructed representation spaces.
        pos (bool): bool to add a sinusoidal positional embedding on top of the patch embeddings.
        head_mode (str): flatten head type from ['linear', 'mlp'].
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
        - [Enhancing Time Series Forecasting through Selective Representation Spaces: A Patch Perspective](https://arxiv.org/abs/2510.14510)
        - [decisionintelligence/SRSNet reference implementation (MIT license)](https://github.com/decisionintelligence/SRSNet)
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
        patch_len: int = 24,
        stride: int = 24,
        d_model: int = 512,
        hidden_size: int = 128,
        dropout: float = 0.2,
        alpha: float = 2.0,
        pos: bool = True,
        head_mode: str = "linear",
        revin: bool = True,
        revin_affine: bool = True,
        revin_subtract_last: bool = False,
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
        super(SRSNet, self).__init__(
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
        # (patch_len > input_size would yield an empty patch grid)
        patch_len = min(input_size, patch_len)

        c_out = self.loss.outputsize_multiplier

        # Fixed hyperparameters
        c_in = 1  # Always univariate (channel-independent, like PatchTST)

        self.model = SRSNet_backbone(
            c_in=c_in,
            c_out=c_out,
            input_size=input_size,
            h=h,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            dropout=dropout,
            hidden_size=hidden_size,
            alpha=alpha,
            pos=pos,
            head_mode=head_mode,
            revin=revin,
            affine=revin_affine,
            subtract_last=revin_subtract_last,
        )

    def forward(self, windows_batch):  # x: [batch, input_size]

        # Parse windows_batch
        x = windows_batch["insample_y"]

        x = x.permute(0, 2, 1)  # x: [Batch, 1, input_size]
        x = self.model(x)
        forecast = x.reshape(x.shape[0], self.h, -1)  # x: [Batch, h, c_out]

        return forecast
