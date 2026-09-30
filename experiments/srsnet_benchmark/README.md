# SRSNet Accuracy Benchmark Plan (ETT / Weather long-horizon)

This note describes the accuracy-parity plan for the `SRSNet` port
(`neuralforecast/models/srsnet.py`). It is a **plan**, not results: the parity
run is a human/GPU step that must happen before upstreaming. No benchmark
numbers are claimed here.

## Motivation

SRSNet (*Enhancing Time Series Forecasting through Selective Representation
Spaces: A Patch Perspective*, [arXiv:2510.14510](https://arxiv.org/abs/2510.14510))
is a channel-independent patch model that replaces PatchTST's fixed patch grid
with a Selective Representation Spaces (SRS) module (learnable patch selection
+ dynamic reassembly). The paper reports SOTA-competitive long-horizon results;
this port must confirm it **beats or matches PatchTST on at least one standard
setting** before being proposed upstream.

## Datasets

Standard long-horizon benchmarks, all with `input_size=96` lookback
(512 in the paper's main table; run both if budget allows):

- ETTh1, ETTh2, ETTm1, ETTm2 (7 series each, train/val/test splits per the
  standard protocol)
- Weather (21 meteorological series, WTK not included)

## Protocol

- Horizons: `h ∈ {96, 192, 336, 720}`
- Metrics: MSE and MAE on the test split, per dataset × horizon
- Models:
  - `SRSNet` (this port), paper hyperparameters as defaults:
    `patch_len=24, stride=24, d_model=512, hidden_size=128, dropout=0.2,
    alpha=2.0, pos=True, revin_affine=True, revin_subtract_last=False`
  - `PatchTST` (in-library baseline) with the library's recommended
    long-horizon configuration for the same lookback
  - PatchMixer is not implemented in neuralforecast; take its published
    numbers from the paper/reference repo for context only
- Training: fixed seed, early stopping on validation loss, learning rate 1e-4
  (paper default) with the library's default scheduler settings; identical
  `windows_batch_size`/`batch_size` across models where possible
- Deliverable: a per-dataset × horizon MSE/MAE table (SRSNet vs PatchTST),
  plus training time/parameter count, appended below when the run completes

## Acceptance

- SRSNet beats or matches PatchTST (within one standard error) on at least one
  dataset × horizon setting, with no catastrophic regression (>10% MSE) across
  the grid — otherwise revisit the SRS-layer port before upstreaming.

## Status

Not run. This plan is executable via `@remyx-ai validate` once the fork is
App-wired, or manually on a GPU host with the neuralforecast long-horizon
experiment scripts (`experiments/long_horizon/`).
