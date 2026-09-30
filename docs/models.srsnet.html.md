---
description: >-
  SRSNet: channel-independent patch model with Selective Representation Spaces (learnable patch selection and dynamic reassembly) for time series forecasting.
output-file: models.srsnet.html
title: SRSNet
---

SRSNet is a channel-independent patch model, like PatchTST, that replaces the fixed patch grid with a Selective Representation Spaces (SRS) module: a learned patch *selection* over all stride-one candidate patches and a dynamic *reassembly* of the selected patches, so the model attends to the most informative patch representations. The original (fixed-stride) and the reconstructed representation spaces are embedded and combined with a learnable adaptive weight.

## 1. SRSNet

::: neuralforecast.models.srsnet.SRSNet
    options:
      members:
        - fit
        - predict
      heading_level: 3

### Usage example

```python
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import SRSNet
from neuralforecast.utils import AirPassengersPanel
from neuralforecast.losses.pytorch import MAE

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]].reset_index(drop=True)
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True)

model = SRSNet(h=12,
               input_size=104,
               patch_len=16,
               stride=8,
               d_model=64,
               hidden_size=128,
               dropout=0.2,
               alpha=2.0,
               pos=True,
               loss=MAE(),
               max_steps=500,
               early_stop_patience_steps=3,
               batch_size=32)

fcst = NeuralForecast(models=[model], freq='ME')
fcst.fit(df=Y_train_df)
forecasts = fcst.predict(futr_df=Y_test_df)

fig, ax = plt.subplots(1, 1, figsize = (20, 7))
Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id=='Airline1'].drop('unique_id', axis=1)
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['SRSNet'], c='blue', label='Forecast')
ax.set_title('AirPassengers Forecast', fontsize=22)
ax.set_ylabel('Monthly Passengers', fontsize=20)
ax.set_xlabel('Year', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()
```

## 2. Auxiliary functions

::: neuralforecast.models.srsnet.PositionalEmbedding
    options:
      members: []

::: neuralforecast.models.srsnet.SRS
    options:
      members: []

::: neuralforecast.models.srsnet.FlattenHead
    options:
      members: []

::: neuralforecast.models.srsnet.SRSNet_backbone
    options:
      members: []

## 3. References

- [Enhancing Time Series Forecasting through Selective Representation Spaces: A Patch Perspective (arXiv:2510.14510)](https://arxiv.org/abs/2510.14510)
- [decisionintelligence/SRSNet reference implementation (MIT license)](https://github.com/decisionintelligence/SRSNet)
