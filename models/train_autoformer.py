"""
models/train_autoformer.py

Provides:
  - finrl_to_wide_df(df)              pivot FinRL long-format → Autoformer wide-format
  - train_autoformer(df, ...)         train a fresh Exp_Main from a wide-format df
  - train_autoformer_from_finrl(...)  train an existing Exp_Main from FinRL DataFrames
"""

import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from configs.conf import get_train_config
from exp.exp_main import Exp_Main
from utils.timefeatures import time_features
from utils.tools import EarlyStopping, adjust_learning_rate


# --------------------------------------------------------------------------- #
#  FinRL → Autoformer format conversion                                        #
# --------------------------------------------------------------------------- #

def finrl_to_wide_df(df: pd.DataFrame, features=None) -> pd.DataFrame:
    """
    Pivot a FinRL long-format DataFrame to the wide format expected by
    AutoformerInMemoryDataset.

    FinRL long-format (input):
        columns: ['date', 'tic', 'open', 'high', 'low', 'close', 'volume', ...indicators]
        one row per (date, ticker)

    Output wide-format:
        columns: ['date', '<tic>_<feature>', ...]
        one row per date, NaN rows dropped

    Parameters
    ----------
    df : pd.DataFrame
        FinRL long-format DataFrame with at least ['date', 'tic'] columns.
    features : list[str] | None
        Feature columns to include.  Defaults to ['close'].
    """
    if features is None:
        features = ['close']

    pivoted = df.pivot(index='date', columns='tic', values=features)
    # After pivot: MultiIndex columns (feature, tic) → flatten to "<tic>_<feature>"
    pivoted.columns = [f"{tic}_{feat}" for feat, tic in pivoted.columns]
    pivoted = pivoted.reset_index()

    # Drop rows with NaN that can appear at indicator boundaries
    pivoted = pivoted.dropna()

    return pivoted


# --------------------------------------------------------------------------- #
#  In-memory dataset                                                           #
# --------------------------------------------------------------------------- #

class AutoformerInMemoryDataset(Dataset):
    """
    Mirrors Dataset_Yahoo_Downloader / Dataset_Custom but reads directly from a
    DataFrame instead of a CSV file.

    Expected df format:
        - 'date' column (parseable by pd.to_datetime)
        - one or more numeric feature columns

    The data is split into train / val / test with the ratios given.
    The StandardScaler is always fitted on the training slice only; pass a
    pre-fitted ``scaler`` when constructing val / test datasets so that the
    same normalisation is applied.

    Columns that contain any NaN values are mean-imputed and a warning is
    printed so the caller is aware of the imputation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        flag: str = 'train',
        seq_len: int = 96,
        label_len: int = 48,
        pred_len: int = 24,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        scale: bool = True,
        freq: str = 'd',
        scaler: StandardScaler = None,
    ):
        assert flag in ('train', 'val', 'test'), \
            f"flag must be 'train', 'val', or 'test'; got '{flag}'"

        self.seq_len   = seq_len
        self.label_len = label_len
        self.pred_len  = pred_len
        self.scale     = scale
        self.freq      = freq

        set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self._build(df, set_type, train_ratio, val_ratio, scaler)

    # ------------------------------------------------------------------ #

    def _build(self, df, set_type, train_ratio, val_ratio, scaler):
        n        = len(df)
        n_train  = int(n * train_ratio)
        n_test   = int(n * (1.0 - train_ratio - val_ratio))
        n_val    = n - n_train - n_test

        border1s = [0,       n_train - self.seq_len,   n - n_test - self.seq_len]
        border2s = [n_train, n_train + n_val,           n]
        b1, b2   = border1s[set_type], border2s[set_type]

        feature_cols = [c for c in df.columns if c != 'date']
        raw = df[feature_cols].values.astype(np.float32)

        # --- NaN check and mean imputation -------------------------------- #
        for i, col in enumerate(feature_cols):
            col_data = raw[:, i]
            if np.isnan(col_data).any():
                mean_val = np.nanmean(col_data)
                n_nans = int(np.isnan(col_data).sum())
                print(
                    f"[AutoformerInMemoryDataset] Column '{col}' has {n_nans} NaN "
                    f"value(s) — imputing with column mean ({mean_val:.4f})."
                )
                col_data[np.isnan(col_data)] = mean_val
                raw[:, i] = col_data

        if self.scale:
            if scaler is None:
                self.scaler = StandardScaler()
                self.scaler.fit(raw[border1s[0]:border2s[0]])
            else:
                self.scaler = scaler
            data = self.scaler.transform(raw)
        else:
            self.scaler = None
            data = raw

        # time features → shape (n, n_time_feats)
        dates = pd.to_datetime(df['date'].values)
        stamp = time_features(dates, freq=self.freq).T

        self.data_x     = data[b1:b2]
        self.data_y     = data[b1:b2]
        self.data_stamp = stamp[b1:b2]

    # ------------------------------------------------------------------ #

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end   = s_begin + self.seq_len
        r_begin = s_end   - self.label_len
        r_end   = r_begin + self.label_len + self.pred_len

        return (
            self.data_x[s_begin:s_end],
            self.data_y[r_begin:r_end],
            self.data_stamp[s_begin:s_end],
            self.data_stamp[r_begin:r_end],
        )

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# --------------------------------------------------------------------------- #
#  Training function                                                           #
# --------------------------------------------------------------------------- #

def train_autoformer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seq_len: int = 96,
    label_len: int = 48,
    pred_len: int = 24,
    train_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    patience: int = 3,
    freq: str = 'd',
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    checkpoint_dir: str = './autoformer_checkpoints',
) -> Exp_Main:
    """
    Train an Autoformer model on the provided DataFrame.

    Parameters
    ----------
    train_df : pd.DataFrame
        Must contain a 'date' column (parseable by pd.to_datetime) plus at
        least one numeric feature column.
        Requires at least seq_len + label_len + pred_len rows total.
    seq_len : int
        Encoder look-back window length.
    label_len : int
        Decoder context (overlap) length.
    pred_len : int
        Prediction horizon.
    train_epochs : int
        Maximum number of training epochs.
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Adam learning rate.
    patience : int
        Early-stopping patience (epochs without val-loss improvement).
    freq : str
        Time-feature frequency passed to `time_features` ('d', 'h', 't', …).
    train_ratio : float
        Fraction of rows used for training.
    val_ratio : float
        Fraction of rows used for validation (remainder becomes test set).
    checkpoint_dir : str
        Root directory where the best checkpoint is saved.

    Returns
    -------
    exp : Exp_Main
        Trained experiment object with best weights loaded.
        Access the raw model via ``exp.model``.
        The fitted scaler is available as ``exp.scaler``.
    """
    assert 'date' in train_df.columns, "train_df must have a 'date' column"

    feature_cols = [c for c in train_df.columns if c != 'date']
    feature_dim  = len(feature_cols)

    min_rows = seq_len + label_len + pred_len
    assert len(train_df) >= min_rows, (
        f"train_df has {len(train_df)} rows but needs at least {min_rows} "
        f"(seq_len={seq_len} + label_len={label_len} + pred_len={pred_len})"
    )

    # ---- datasets -------------------------------------------------------- #
    train_ds = AutoformerInMemoryDataset(
        train_df, flag='train',
        seq_len=seq_len, label_len=label_len, pred_len=pred_len,
        train_ratio=train_ratio, val_ratio=val_ratio, freq=freq,
    )
    val_ds = AutoformerInMemoryDataset(
        train_df, flag='val',
        seq_len=seq_len, label_len=label_len, pred_len=pred_len,
        train_ratio=train_ratio, val_ratio=val_ratio, freq=freq,
        scaler=train_ds.scaler,
    )
    test_ds = AutoformerInMemoryDataset(
        train_df, flag='test',
        seq_len=seq_len, label_len=label_len, pred_len=pred_len,
        train_ratio=train_ratio, val_ratio=val_ratio, freq=freq,
        scaler=train_ds.scaler,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=0)

    # ---- config ---------------------------------------------------------- #
    args, _ = get_train_config()
    args.enc_in        = feature_dim
    args.dec_in        = feature_dim
    args.c_out         = feature_dim
    args.seq_len       = seq_len
    args.label_len     = label_len
    args.pred_len      = pred_len
    args.train_epochs  = train_epochs
    args.batch_size    = batch_size
    args.learning_rate = learning_rate
    args.patience      = patience
    args.freq          = freq
    args.checkpoints   = checkpoint_dir
    args.load_params   = False
    args.save_params   = True

    if torch.cuda.is_available():
        args.use_gpu = True
        args.gpu     = 0
    else:
        args.use_gpu = False
        args.gpu     = None

    setting = (
        f"Autoformer_ft{feature_dim}"
        f"_sl{seq_len}_ll{label_len}_pl{pred_len}"
        f"_dm{args.d_model}_nh{args.n_heads}"
        f"_el{args.e_layers}_dl{args.d_layers}"
    )

    # ---- build model ----------------------------------------------------- #
    exp = Exp_Main(args)
    exp.scaler = train_ds.scaler

    # ---- training loop --------------------------------------------------- #
    # We implement the loop here directly rather than calling exp.train() with
    # custom loaders, because exp.train() assumes train_data (a Dataset object
    # with a .scaler attribute) is always available alongside train_loader.

    path = os.path.join(checkpoint_dir, setting)
    os.makedirs(path, exist_ok=True)

    optimizer  = exp._select_optimizer()
    criterion  = exp._select_criterion()
    early_stop = EarlyStopping(patience=patience, verbose=False)

    for epoch in range(train_epochs):
        exp.model.train()
        train_loss = []
        t0 = time.time()

        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            batch_x      = batch_x.float().to(exp.device)
            batch_y      = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)

            optimizer.zero_grad()
            outputs, target = exp._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        train_loss_avg = np.mean(train_loss)
        val_loss       = exp.vali(val_ds,  val_loader,  criterion)
        test_loss      = exp.vali(test_ds, test_loader, criterion)

        print(
            f"Epoch {epoch + 1}/{train_epochs}  "
            f"({time.time() - t0:.1f}s)  "
            f"train: {train_loss_avg:.6f}  "
            f"val: {val_loss:.6f}  "
            f"test: {test_loss:.6f}"
        )

        early_stop(val_loss, exp.model, path)
        if early_stop.early_stop:
            print("Early stopping triggered.")
            break

        adjust_learning_rate(optimizer, epoch + 1, args)

    # load best checkpoint saved by EarlyStopping
    best_ckpt = os.path.join(path, 'checkpoint.pth')
    exp.model.load_state_dict(torch.load(best_ckpt, map_location=exp.device))
    exp.model.eval()

    return exp


# --------------------------------------------------------------------------- #
#  Train an existing Exp_Main model from pre-split FinRL DataFrames            #
# --------------------------------------------------------------------------- #

def evaluate(model, val_loader, criterion):
    model.model.eval()
    val_loss = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in val_loader:
            batch_x      = batch_x.float().to(model.device)
            batch_y      = batch_y.float().to(model.device)
            batch_x_mark = batch_x_mark.float().to(model.device)
            batch_y_mark = batch_y_mark.float().to(model.device)

            outputs, target = model._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

            directional_accuracy = (np.sign(np.diff(target.detach().cpu().numpy())) == np.sign(np.diff(outputs.detach().cpu().numpy()))).mean()
            MAE = torch.mean(torch.abs(outputs - target)).item()

            loss = criterion(outputs, target)
            val_loss.append(loss.item())
    return np.mean(val_loss), directional_accuracy, MAE

def train_autoformer_from_finrl(
    model,
    train_finrl_df: pd.DataFrame,
    val_finrl_df: pd.DataFrame = None,
    features=None,
    setting: str = 'autoformer_finrl',
):
    """
    Train an already-created Exp_Main (Autoformer / Transformer) dynamics model
    using FinRL-format DataFrames produced by get_data() in run_experiment.py.

    This is the entry point called by train_dynamics_model() in run_experiment.py.
    The model's architecture and hyper-parameters (seq_len, pred_len, enc_in, …)
    are read from model.args — no separate config is needed.

    Parameters
    ----------
    model : Exp_Main
        Dynamics model created by get_model().
    train_finrl_df : pd.DataFrame
        FinRL long-format training data (columns: date, tic, close, …).
    val_finrl_df : pd.DataFrame | None
        FinRL long-format validation data.  If None, train_finrl_df is reused
        as a validation proxy (no early-stopping signal from held-out data).
    features : list[str] | None
        FinRL columns to use as model input features.  Defaults to ['close'].
    setting : str
        Checkpoint subdirectory suffix.

    Returns
    -------
    model  (trained in-place, returned for convenience)
    """
    args = model.args

    # --- 1. Convert FinRL long format → Autoformer wide format -------------- #
    wide_train = finrl_to_wide_df(train_finrl_df, features=features)
    wide_val   = finrl_to_wide_df(
        val_finrl_df if val_finrl_df is not None else train_finrl_df,
        features=features,
    )

    n_features = len(wide_train.columns) - 1   # exclude 'date'
    assert n_features == args.enc_in, (
        f"Feature count mismatch: pivoted FinRL data has {n_features} columns "
        f"but model was built with enc_in={args.enc_in}.  "
        f"Check that assets_per_ep matches feature_dim in dynamics_kwargs."
    )

    # --- 2. Build datasets (full split = use the whole passed-in DataFrame) - #
    train_ds = AutoformerInMemoryDataset(
        wide_train, flag='train',
        seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
        train_ratio=1.0, val_ratio=0.0, freq=args.freq,
    )
    val_ds = AutoformerInMemoryDataset(
        wide_val, flag='train',
        seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
        train_ratio=1.0, val_ratio=0.0, freq=args.freq,
        scaler=train_ds.scaler,   # reuse train scaler — no data leakage
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  drop_last=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, drop_last=False, num_workers=0)

    # Expose the scaler so callers can invert-transform predictions later
    model.scaler = train_ds.scaler

    # --- 3. Training loop --------------------------------------------------- #
    path = os.path.join(args.checkpoints, f"{args.model}_{setting}")
    os.makedirs(path, exist_ok=True)

    optimizer  = model._select_optimizer()
    criterion  = model._select_criterion()
    early_stop = EarlyStopping(patience=args.patience, verbose=False)

    for epoch in range(args.train_epochs):
        model.model.train()
        train_loss = []
        t0 = time.time()

        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            batch_x      = batch_x.float().to(model.device)
            batch_y      = batch_y.float().to(model.device)
            batch_x_mark = batch_x_mark.float().to(model.device)
            batch_y_mark = batch_y_mark.float().to(model.device)

            optimizer.zero_grad()
            outputs, target = model._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        val_loss, directional_accuracy, MAE = evaluate(model, val_loader, criterion)

        print(
            f"Epoch {epoch + 1}/{args.train_epochs}  "
            f"({time.time() - t0:.1f}s)  "
            f"train: {np.mean(train_loss):.6f} val: {val_loss:.6f}  directional_Acc: {directional_accuracy:.4f} MAE: {MAE:.4f}"
        )

        early_stop(val_loss, model.model, path)
        if early_stop.early_stop:
            print("Early stopping triggered.")
            break

        adjust_learning_rate(optimizer, epoch + 1, args)

    # Load best weights saved by EarlyStopping
    best_ckpt = os.path.join(path, 'checkpoint.pth')
    model.model.load_state_dict(torch.load(best_ckpt, map_location=model.device))
    model.model.eval()

    return model