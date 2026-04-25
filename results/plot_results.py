
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os
import random

save_graphs = 'results/graphs'
save_df = 'results/df.csv'

colours = list(mcolors.TABLEAU_COLORS)

# --------------------------------------------------------------------------- #
# Paper-quality style                                                           #
# --------------------------------------------------------------------------- #

# Colorblind-safe palette (Wong 2011)
PALETTE = [
    '#0173B2',  # blue
    '#DE8F05',  # orange
    '#029E73',  # green
    '#D55E00',  # vermilion
    '#CC78BC',  # purple
    '#CA9161',  # brown
    '#FBAFE4',  # pink
    '#949494',  # gray
]

PAPER_RC = {
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.framealpha': 0.85,
    'lines.linewidth': 1.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'grid.color': '#444444',
}

SAVE_DPI = 300


def _apply_paper_style():
    plt.rcParams.update(PAPER_RC)


# --------------------------------------------------------------------------- #
# Metric computation helpers                                                    #
# --------------------------------------------------------------------------- #

def _portfolio_values_from_states(states: np.ndarray, features: int, stock_dim: int | None = None) -> np.ndarray:
    """Reconstruct portfolio value at every timestep from the state array.

    State layout: [cash, price_0..price_n, shares_0..shares_n, tech_indicators...]

    stock_dim overrides features when the experiment used a sampled subset of
    the full ticker universe (assets_per_ep < len(all_tickers)).
    """
    n = stock_dim if stock_dim is not None else features
    balances = states[:, 0]
    prices = states[:, 1:n + 1]
    holdings = states[:, n + 1: n * 2 + 1]
    return balances + np.sum(holdings * prices, axis=1)


def _compute_financial_metrics(portfolio_values: np.ndarray, trading_days: int = 252) -> dict:
    """Compute a comprehensive set of financial performance metrics.

    Metrics included
    ----------------
    Standard measures  : Total Return, Annualised Return, Annualised Volatility
    Risk-adjusted      : Sharpe Ratio, Sortino Ratio, Calmar Ratio, Omega Ratio
    Drawdown           : Maximum Drawdown, Average Drawdown
    Tail risk          : VaR 95%, CVaR 95% (Expected Shortfall)
    Trade statistics   : Win Rate, Profit Factor

    Additional suggestions beyond the requested metrics
    ---------------------------------------------------
    - Omega Ratio     : captures the full return distribution (no normality assumption)
    - Average Drawdown: complements max drawdown with a persistence measure
    - CVaR 95%        : more conservative and coherent than plain VaR
    """
    pv = np.asarray(portfolio_values, dtype=float)
    ret = np.diff(pv) / np.where(pv[:-1] != 0, pv[:-1], 1e-12)

    n = len(ret)
    total_ret = (pv[-1] - pv[0]) / pv[0] if pv[0] != 0 else 0.0
    ann_ret = (1 + total_ret) ** (trading_days / n) - 1 if n > 0 else 0.0
    ann_vol = ret.std(ddof=1) * np.sqrt(trading_days) if n > 1 else 0.0

    # Sharpe
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Sortino (downside deviation)
    neg = ret[ret < 0]
    sortino_denom = neg.std(ddof=1) * np.sqrt(trading_days) if len(neg) > 1 else 1e-12
    sortino = ann_ret / sortino_denom

    # Drawdown series
    cum = np.cumprod(1 + ret)
    running_max = np.maximum.accumulate(cum)
    dd_series = (cum - running_max) / np.where(running_max != 0, running_max, 1e-12)
    max_dd = dd_series.min()
    avg_dd = dd_series[dd_series < 0].mean() if (dd_series < 0).any() else 0.0

    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    # VaR / CVaR at 95%
    var_95 = float(-np.percentile(ret, 5))
    tail = ret[ret <= -var_95]
    cvar_95 = float(-tail.mean()) if len(tail) > 0 else var_95

    # Win rate & profit factor
    win_rate = float((ret > 0).mean()) * 100
    pos_sum = ret[ret > 0].sum()
    neg_sum = abs(ret[ret < 0].sum())
    profit_factor = pos_sum / neg_sum if neg_sum > 0 else float('inf')

    # Omega Ratio (threshold = 0)
    gain = ret[ret > 0].sum()
    loss = abs(ret[ret < 0].sum())
    omega = gain / loss if loss > 0 else float('inf')

    return {
        'Total Return (%)':      round(total_ret * 100, 2),
        'Ann. Return (%)':       round(ann_ret * 100, 2),
        'Ann. Volatility (%)':   round(ann_vol * 100, 2),
        'Sharpe Ratio':          round(sharpe, 3),
        'Sortino Ratio':         round(sortino, 3),
        'Max Drawdown (%)':      round(max_dd * 100, 2),
        'Avg Drawdown (%)':      round(avg_dd * 100, 2),
        'Calmar Ratio':          round(calmar, 3),
        'VaR 95% (%)':           round(var_95 * 100, 3),
        'CVaR 95% (%)':          round(cvar_95 * 100, 3),
        'Win Rate (%)':          round(win_rate, 1),
        'Profit Factor':         round(profit_factor, 3),
        'Omega Ratio':           round(omega, 3),
    }


def _compute_prediction_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    """Compute time-series forecast evaluation metrics.

    Metrics included
    ----------------
    Standard errors  : MAE, RMSE, MAPE
    Shape / rank     : R², Directional Accuracy

    Additional suggestions
    ----------------------
    - MASE (Mean Absolute Scaled Error): scale-free, robust to near-zero prices;
      uses the naïve random-walk as the baseline scaler.
    - Theil's U2: compares the model to a naïve forecast — U2 < 1 means
      the model beats the random walk.
    """
    p = np.asarray(predictions, dtype=float)
    a = np.asarray(actuals, dtype=float)

    mae  = float(np.mean(np.abs(p - a)))
    rmse = float(np.sqrt(np.mean((p - a) ** 2)))

    mask = a != 0
    mape = float(np.mean(np.abs((p[mask] - a[mask]) / a[mask])) * 100) if mask.any() else float('nan')

    # R²
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

    # Directional accuracy
    if a.ndim == 1:
        pred_dir   = np.sign(np.diff(p))
        actual_dir = np.sign(np.diff(a))
    else:
        pred_dir   = np.sign(np.diff(p, axis=0))
        actual_dir = np.sign(np.diff(a, axis=0))
    dir_acc = float(np.mean(pred_dir == actual_dir) * 100)

    # MASE (naïve baseline = previous value)
    naive_err = np.mean(np.abs(np.diff(a.flatten()))) if a.size > 1 else 1e-12
    mase = mae / naive_err if naive_err > 0 else float('nan')

    # Theil's U2
    p_flat = p.flatten()
    a_flat = a.flatten()
    naïve_fc = a_flat[:-1]
    theil_num = np.sqrt(np.mean((p_flat[:-1] - a_flat[1:]) ** 2))
    theil_den = np.sqrt(np.mean((naïve_fc - a_flat[1:]) ** 2))
    theil_u2 = theil_num / theil_den if theil_den > 0 else float('nan')

    return {
        'MAE':                    round(mae, 4),
        'RMSE':                   round(rmse, 4),
        'MAPE (%)':               round(mape, 2),
        'R²':                     round(r2, 4),
        'Directional Acc. (%)':   round(dir_acc, 1),
        'MASE':                   round(mase, 4),
        "Theil's U2":             round(theil_u2, 4),
    }


# --------------------------------------------------------------------------- #
# Saving helpers                                                                #
# --------------------------------------------------------------------------- #

def _save_metrics_table(df_metrics: pd.DataFrame, dataset_name: str, out_dir: str):
    """Save metrics as both CSV and a publication-quality matplotlib table figure."""
    csv_path = os.path.join(out_dir, f'{dataset_name}_metrics.csv')
    df_metrics.to_csv(csv_path)

    agents = df_metrics.columns.tolist()
    metrics = df_metrics.index.tolist()
    cell_text = []

    for metric in metrics:
        row = []
        for agent in agents:
            val = df_metrics.loc[metric, agent]
            row.append(f'{val:.3g}' if isinstance(val, float) else str(val))
        cell_text.append(row)

    # Highlight best value per row
    higher_is_better = {
        'Total Return (%)', 'Ann. Return (%)', 'Sharpe Ratio', 'Sortino Ratio',
        'Calmar Ratio', 'Win Rate (%)', 'Profit Factor', 'Omega Ratio',
        'R²', 'Directional Acc. (%)',
    }
    lower_is_better = {
        'Ann. Volatility (%)', 'Max Drawdown (%)', 'Avg Drawdown (%)',
        'VaR 95% (%)', 'CVaR 95% (%)', 'MAE', 'RMSE', 'MAPE (%)',
        'MASE', "Theil's U2",
    }

    cell_colors = [['white'] * len(agents) for _ in metrics]
    for r, metric in enumerate(metrics):
        try:
            vals = [df_metrics.loc[metric, a] for a in agents]
            numeric = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
            if not numeric:
                continue
            if metric in higher_is_better:
                best = max(numeric)
            elif metric in lower_is_better:
                best = min(numeric)
            else:
                continue
            for c, v in enumerate(vals):
                if v == best:
                    cell_colors[r][c] = '#d4edda'   # light green
        except Exception:
            pass

    n_rows = len(metrics)
    fig_h = max(3.0, 0.38 * n_rows + 1.2)
    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(agents) + 2), fig_h))
    ax.axis('off')

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=metrics,
        colLabels=agents,
        cellColours=cell_colors,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    # Header row style
    for (row, col), cell in tbl.get_celld().items():
        if row == 0 or col == -1:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')

    fig.suptitle(
        f'Performance Metrics — {dataset_name}',
        fontsize=12, fontweight='bold', y=0.98,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, f'{dataset_name}_metrics_table.png'),
        dpi=SAVE_DPI, bbox_inches='tight',
    )
    fig.savefig(
        os.path.join(out_dir, f'{dataset_name}_metrics_table.pdf'),
        bbox_inches='tight',
    )
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Plot: portfolio net value                                                     #
# --------------------------------------------------------------------------- #

def _plot_portfolio_performance(
    results: dict,
    trials: list,
    dataset_name: str,
    agents: list,
    features: int,
    out_dir: str,
):
    """Two-panel figure: (top) portfolio net value, (bottom) rolling drawdown."""
    _apply_paper_style()

    fig = plt.figure(figsize=(9, 5))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax_val = fig.add_subplot(gs[0])
    ax_dd  = fig.add_subplot(gs[1], sharex=ax_val)

    initial_value = None

    for i, agent in enumerate(agents):
        color = PALETTE[i % len(PALETTE)]
        all_pv = []

        for trial in trials:
            try:
                entry = results[trial][dataset_name][agent]
                stock_dim = entry.get('stock_dim')
                for ep_states in entry['states']:
                    pv = _portfolio_values_from_states(np.array(ep_states), features, stock_dim)
                    all_pv.append(pv)
            except (KeyError, IndexError):
                continue

        if not all_pv:
            continue

        # Align lengths
        min_len = min(len(x) for x in all_pv)
        all_pv = np.array([x[:min_len] for x in all_pv])

        if initial_value is None:
            initial_value = all_pv[0, 0]

        # Normalise so all agents start at 100 (index value)
        normalised = all_pv / all_pv[:, [0]] * 100

        mean_pv = normalised.mean(axis=0)
        std_pv  = normalised.std(axis=0)

        t = np.arange(min_len)

        # Individual trials (faint)
        for pv_trial in normalised:
            ax_val.plot(t, pv_trial, color=color, alpha=0.12, linewidth=0.8)

        # Mean
        ax_val.plot(t, mean_pv, color=color, label=agent, linewidth=2.0, zorder=3)

        # ± 1 std band
        if len(trials) > 1:
            ax_val.fill_between(t, mean_pv - std_pv, mean_pv + std_pv,
                                color=color, alpha=0.15)

        # Drawdown
        ret = np.diff(mean_pv) / mean_pv[:-1]
        cum = np.cumprod(1 + ret)
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / np.where(running_max > 0, running_max, 1e-12) * 100
        ax_dd.fill_between(np.arange(len(dd)), dd, 0, color=color, alpha=0.35)
        ax_dd.plot(np.arange(len(dd)), dd, color=color, linewidth=0.8)

    ax_val.axhline(100, color='black', linestyle=':', linewidth=0.9, alpha=0.5)
    ax_val.set_ylabel('Portfolio Value (indexed, start = 100)')
    ax_val.legend(loc='upper left', frameon=True)
    ax_val.set_title(f'Portfolio Performance — {dataset_name}', fontweight='bold')
    plt.setp(ax_val.get_xticklabels(), visible=False)

    ax_dd.set_ylabel('Drawdown (%)')
    ax_dd.set_xlabel('Trading Day')
    ax_dd.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))

    fig.savefig(
        os.path.join(out_dir, f'{dataset_name}_portfolio.png'),
        dpi=SAVE_DPI, bbox_inches='tight',
    )
    fig.savefig(
        os.path.join(out_dir, f'{dataset_name}_portfolio.pdf'),
        bbox_inches='tight',
    )
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Plot: prediction vs actual                                                    #
# --------------------------------------------------------------------------- #

def _plot_predictions_vs_actual(
    results: dict,
    trials: list,
    dataset_name: str,
    agents: list,
    features: int,
    ticker_names: list,
    out_dir: str,
    n_assets: int = 4,
    max_steps: int | None = None,
):
    """Grid of subplots comparing each agent's price predictions to actuals.

    One row per sampled asset; one column per agent that has predictions.
    """
    _apply_paper_style()

    # Only include agents that have non-empty prediction lists
    pred_agents = [
        a for a in agents
        if any(
            len(results[t][dataset_name].get(a, {}).get('predictions', [])) > 0
            for t in trials
        )
    ]
    if not pred_agents:
        return

    # Determine actual stock_dim from the prediction arrays (may be smaller than
    # the full universe `features` because tickers are sampled each trial).
    stock_dim = features
    for t in trials:
        for a in pred_agents:
            preds = results[t][dataset_name].get(a, {}).get('predictions', [])
            if len(preds) > 0:
                ep = np.array(preds[0])
                if ep.ndim == 4:
                    stock_dim = ep.shape[3]
                elif ep.ndim == 3:
                    stock_dim = ep.shape[2]
                else:
                    stock_dim = ep.shape[-1]
                break
        else:
            continue
        break

    # Use stored ticker names for the sampled subset when available.
    sampled_ticker_names = None
    for t in trials:
        for a in pred_agents:
            stored = results[t][dataset_name].get(a, {}).get('tickers')
            if stored is not None:
                sampled_ticker_names = stored
                break
        if sampled_ticker_names is not None:
            break
    if sampled_ticker_names is None:
        sampled_ticker_names = ticker_names[:stock_dim]

    n_sample = min(n_assets, stock_dim)
    sampled_idx = random.sample(range(stock_dim), n_sample)

    n_rows = n_sample
    n_cols = len(pred_agents)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 2.4 * n_rows),
        squeeze=False,
        sharey=False,
    )
    fig.subplots_adjust(hspace=0.45, wspace=0.3)

    for col, agent in enumerate(pred_agents):
        agent_color = PALETTE[col % len(PALETTE)]

        for row, asset_idx in enumerate(sampled_idx):
            ax = axes[row][col]
            ticker = sampled_ticker_names[asset_idx] if asset_idx < len(sampled_ticker_names) else str(asset_idx)

            # Collect actual & prediction over trials; each is a list of per-episode arrays
            all_preds = []
            actuals_ref = None
            for trial in trials:
                try:
                    ep_preds = results[trial][dataset_name][agent]['predictions']
                    ep_acts  = results[trial][dataset_name][agent]['actuals']
                    pred = np.concatenate(ep_preds, axis=0)
                    act  = np.concatenate(ep_acts,  axis=0)
                    # squeeze batch dim + take step-0 → (T, stock_dim)
                    if pred.ndim == 4:
                        pred = pred[:, 0, 0, :]
                    elif pred.ndim == 3:
                        pred = pred[:, 0, :]
                    all_preds.append(pred[:, asset_idx])
                    if actuals_ref is None:
                        actuals_ref = act[:, asset_idx] if act.ndim == 2 else act
                except (KeyError, IndexError, ValueError):
                    continue

            if actuals_ref is None or not all_preds:
                ax.set_visible(False)
                continue

            t_end = max_steps if max_steps else len(actuals_ref)
            t     = np.arange(t_end)

            all_preds_arr = np.array([p[:t_end] for p in all_preds])
            mean_pred = all_preds_arr.mean(axis=0)
            std_pred  = all_preds_arr.std(axis=0)

            actual_vals = actuals_ref[:t_end]
            ax.plot(t, actual_vals, color='#2c3e50', linewidth=1.4,
                    label='Actual', zorder=4)
            ax.plot(t, mean_pred, color=agent_color, linewidth=1.4,
                    linestyle='--', label=f'{agent} prediction', zorder=3)
            if len(trials) > 1:
                ax.fill_between(t, mean_pred - std_pred, mean_pred + std_pred,
                                color=agent_color, alpha=0.2)

            # Pin y-axis to ±30 % of actual price range so off-scale predictions
            # don't crush the actual price line
            a_min, a_max = float(np.nanmin(actual_vals)), float(np.nanmax(actual_vals))
            margin = max((a_max - a_min) * 0.30, a_max * 0.05)
            ax.set_ylim(a_min - margin, a_max + margin)

            ax.set_title(f'{ticker}  [{agent}]', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if row == n_rows - 1:
                ax.set_xlabel('Trading Day')
            if col == 0:
                ax.set_ylabel('Price ($)')

    # Shared legend — collect unique entries from all axes
    handles, labels = [], []
    for row_axes in axes:
        for ax in row_axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=max(1, len(labels)),
                   frameon=True, bbox_to_anchor=(0.5, 1.0), fontsize=9)

    fig.suptitle(
        f'Price Predictions vs Actual — {dataset_name}',
        fontsize=12, fontweight='bold', y=1.02,
    )
    fig.savefig(
        os.path.join(out_dir, f'{dataset_name}_predictions.png'),
        dpi=SAVE_DPI, bbox_inches='tight',
    )
    fig.savefig(
        os.path.join(out_dir, f'{dataset_name}_predictions.pdf'),
        bbox_inches='tight',
    )
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Rolling Sharpe                                                                #
# --------------------------------------------------------------------------- #

def _compute_rolling_sharpe(portfolio_values: np.ndarray, window: int = 30, trading_days: int = 252) -> np.ndarray:
    pv = np.asarray(portfolio_values, dtype=float)
    ret = np.diff(pv) / np.where(pv[:-1] != 0, pv[:-1], 1e-12)
    sharpe = np.full(len(ret), np.nan)
    for i in range(window - 1, len(ret)):
        r = ret[i - window + 1: i + 1]
        std = r.std(ddof=1)
        sharpe[i] = (r.mean() / std * np.sqrt(trading_days)) if std > 0 else 0.0
    return sharpe


def _plot_rolling_sharpe(
    results: dict, trials: list, dataset_name: str, agents: list,
    features: int, out_dir: str, window: int = 30,
):
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.97)

    for i, agent in enumerate(agents):
        color = PALETTE[i % len(PALETTE)]
        all_sharpe = []
        for trial in trials:
            try:
                stock_dim = results[trial][dataset_name][agent].get('stock_dim')
                for ep_states in results[trial][dataset_name][agent]['states']:
                    pv = _portfolio_values_from_states(np.array(ep_states), features, stock_dim)
                    all_sharpe.append(_compute_rolling_sharpe(pv, window))
            except (KeyError, IndexError):
                continue
        if not all_sharpe:
            continue
        min_len = min(len(s) for s in all_sharpe)
        arr = np.array([s[:min_len] for s in all_sharpe])
        mean_s = np.nanmean(arr, axis=0)
        std_s  = np.nanstd(arr, axis=0)
        t = np.arange(min_len)
        ax.plot(t, mean_s, color=color, label=agent)
        ax.fill_between(t, mean_s - std_s, mean_s + std_s, color=color, alpha=0.15)

    ax.axhline(0, color='black', linestyle=':', linewidth=0.9, alpha=0.5)
    ax.set_xlabel('Trading Day')
    ax.set_ylabel(f'{window}-day Rolling Sharpe')
    ax.set_title(f'Rolling Sharpe Ratio — {dataset_name}', fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'{dataset_name}_rolling_sharpe.{ext}'),
                    dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Return distribution                                                           #
# --------------------------------------------------------------------------- #

def _plot_return_distribution(
    results: dict, trials: list, dataset_name: str, agents: list,
    features: int, out_dir: str,
):
    _apply_paper_style()

    all_returns = {}
    for agent in agents:
        rets = []
        for trial in trials:
            try:
                stock_dim = results[trial][dataset_name][agent].get('stock_dim')
                for ep_states in results[trial][dataset_name][agent]['states']:
                    pv = _portfolio_values_from_states(np.array(ep_states), features, stock_dim)
                    r  = np.diff(pv) / np.where(pv[:-1] != 0, pv[:-1], 1e-12)
                    rets.extend(r.tolist())
            except (KeyError, IndexError):
                continue
        if rets:
            all_returns[agent] = np.array(rets) * 100   # percent

    if not all_returns:
        return

    n = len(all_returns)
    fig, ax = plt.subplots(figsize=(max(5, 1.4 * n), 4))
    fig.subplots_adjust(top=0.88, bottom=0.14, left=0.09, right=0.97)

    positions  = list(range(n))
    agent_list = list(all_returns.keys())

    parts = ax.violinplot(
        [all_returns[a] for a in agent_list],
        positions=positions, widths=0.6,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for pc, color in zip(parts['bodies'], PALETTE):
        pc.set_facecolor(color)
        pc.set_alpha(0.5)

    bp = ax.boxplot(
        [all_returns[a] for a in agent_list],
        positions=positions, widths=0.18,
        patch_artist=True, notch=False,
        medianprops=dict(color='black', linewidth=1.5),
        boxprops=dict(facecolor='white', linewidth=1),
        whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1),
        flierprops=dict(marker='.', markersize=2, alpha=0.3),
    )
    for patch, color in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0, color='black', linestyle=':', linewidth=0.9, alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(agent_list, rotation=20, ha='right')
    ax.set_ylabel('Daily Return (%)')
    ax.set_title(f'Daily Return Distribution — {dataset_name}', fontweight='bold')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'{dataset_name}_return_dist.{ext}'),
                    dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Prediction error over time                                                    #
# --------------------------------------------------------------------------- #

def _plot_prediction_error(
    results: dict, trials: list, dataset_name: str, agents: list,
    features: int, ticker_names: list, out_dir: str,
    n_assets: int = 3, window: int = 30,
):
    pred_agents = [
        a for a in agents
        if any(len(results[t][dataset_name].get(a, {}).get('predictions', [])) > 0
               for t in trials)
    ]
    if not pred_agents:
        return

    stock_dim_ref = None
    for t in trials:
        for a in pred_agents:
            sd = results[t][dataset_name].get(a, {}).get('stock_dim')
            if sd is not None:
                stock_dim_ref = sd
                break
        if stock_dim_ref:
            break
    n_assets = min(n_assets, stock_dim_ref or features)
    sampled_idx = random.sample(range(stock_dim_ref or features), n_assets)

    fig, axes = plt.subplots(
        n_assets, len(pred_agents),
        figsize=(3.5 * len(pred_agents), 2.5 * n_assets),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.45, wspace=0.3)

    for col, agent in enumerate(pred_agents):
        agent_color = PALETTE[col % len(PALETTE)]
        for row, asset_idx in enumerate(sampled_idx):
            ax = axes[row][col]
            ticker = ticker_names[asset_idx] if asset_idx < len(ticker_names) else str(asset_idx)
            all_mae = []
            for trial in trials:
                try:
                    preds   = results[trial][dataset_name][agent]['predictions']
                    actuals = results[trial][dataset_name][agent]['actuals']
                    p = np.concatenate(preds, axis=0)
                    a = np.concatenate(actuals, axis=0)
                    if p.ndim == 4: p = p[:, 0, 0, :]
                    elif p.ndim == 3: p = p[:, 0, :]
                    mae_series = np.abs(p[:, asset_idx] - a[:, asset_idx])
                    # rolling window
                    rolling = np.convolve(mae_series, np.ones(window) / window, mode='valid')
                    all_mae.append(rolling)
                except (KeyError, IndexError, ValueError):
                    continue
            if not all_mae:
                ax.set_visible(False)
                continue
            min_len = min(len(m) for m in all_mae)
            arr  = np.array([m[:min_len] for m in all_mae])
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            t = np.arange(min_len)
            ax.plot(t, mean, color=agent_color, linewidth=1.4)
            ax.fill_between(t, mean - std, mean + std, color=agent_color, alpha=0.2)
            ax.set_title(f'{ticker}  [{agent}]', fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if row == n_assets - 1: ax.set_xlabel('Trading Day')
            if col == 0: ax.set_ylabel(f'{window}d Rolling MAE')

    fig.suptitle(f'Prediction Error Over Time — {dataset_name}',
                 fontsize=12, fontweight='bold', y=1.02)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'{dataset_name}_pred_error.{ext}'),
                    dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Predicted vs actual return scatter                                            #
# --------------------------------------------------------------------------- #

def _plot_pred_scatter(
    results: dict, trials: list, dataset_name: str, agents: list,
    features: int, ticker_names: list, out_dir: str, n_assets: int = 3,
):
    pred_agents = [
        a for a in agents
        if any(len(results[t][dataset_name].get(a, {}).get('predictions', [])) > 0
               for t in trials)
    ]
    if not pred_agents:
        return

    stock_dim_ref = None
    for t in trials:
        for a in pred_agents:
            sd = results[t][dataset_name].get(a, {}).get('stock_dim')
            if sd is not None:
                stock_dim_ref = sd; break
        if stock_dim_ref: break
    n_assets = min(n_assets, stock_dim_ref or features)
    sampled_idx = random.sample(range(stock_dim_ref or features), n_assets)

    fig, axes = plt.subplots(
        n_assets, len(pred_agents),
        figsize=(3.0 * len(pred_agents), 2.8 * n_assets),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.5, wspace=0.35)

    for col, agent in enumerate(pred_agents):
        agent_color = PALETTE[col % len(PALETTE)]
        for row, asset_idx in enumerate(sampled_idx):
            ax = axes[row][col]
            ticker = ticker_names[asset_idx] if asset_idx < len(ticker_names) else str(asset_idx)
            all_pred_ret, all_act_ret = [], []
            for trial in trials:
                try:
                    preds   = results[trial][dataset_name][agent]['predictions']
                    actuals = results[trial][dataset_name][agent]['actuals']
                    p = np.concatenate(preds, axis=0)
                    a = np.concatenate(actuals, axis=0)
                    if p.ndim == 4: p = p[:, 0, 0, :]
                    elif p.ndim == 3: p = p[:, 0, :]
                    # 1-step predicted return vs actual return
                    p_col = p[:, asset_idx]
                    a_col = a[:, asset_idx]
                    # shift: pred at t vs actual change from t to t+1
                    pred_ret = np.diff(p_col)
                    act_ret  = np.diff(a_col)
                    all_pred_ret.append(pred_ret)
                    all_act_ret.append(act_ret)
                except (KeyError, IndexError, ValueError):
                    continue
            if not all_pred_ret:
                ax.set_visible(False)
                continue

            pr = np.concatenate(all_pred_ret)
            ar = np.concatenate(all_act_ret)
            T  = len(pr)
            colors_t = plt.cm.Blues(np.linspace(0.25, 0.9, T))

            ax.scatter(pr, ar, c=colors_t, s=4, alpha=0.6, linewidths=0)

            # R² and directional accuracy annotation
            ss_res = np.sum((ar - pr) ** 2)
            ss_tot = np.sum((ar - ar.mean()) ** 2)
            r2     = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
            dir_acc = np.mean(np.sign(pr) == np.sign(ar)) * 100
            ax.text(0.05, 0.93, f'R²={r2:.2f}  Dir={dir_acc:.0f}%',
                    transform=ax.transAxes, fontsize=7.5, va='top')

            # y = x reference line
            lim = max(abs(pr).max(), abs(ar).max()) * 1.1
            ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=0.8, alpha=0.4)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_title(f'{ticker}  [{agent}]', fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if row == n_assets - 1: ax.set_xlabel('Predicted Δ')
            if col == 0: ax.set_ylabel('Actual Δ')

    fig.suptitle(f'Predicted vs Actual Price Change — {dataset_name}',
                 fontsize=12, fontweight='bold', y=1.02)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'{dataset_name}_pred_scatter.{ext}'),
                    dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Actions vs price (per agent, dual y-axis)                                     #
# --------------------------------------------------------------------------- #

def _plot_actions_vs_price(
    results: dict,
    trials: list,
    dataset_name: str,
    agents: list,
    features: int,
    ticker_names: list,
    out_dir: str,
    n_samples: int = 3,
):
    """One figure per agent: n_samples rows, each showing action (left y) and price (right y)."""
    _apply_paper_style()

    for agent in agents:
        # Collect concatenated episodes from first available trial
        concat_actions = None
        concat_prices  = None
        stock_dim      = features
        stored_tickers = None

        for trial in trials:
            entry = results[trial][dataset_name].get(agent, {})
            if not entry.get('actions') or not entry.get('states'):
                continue
            stock_dim = entry.get('stock_dim', features)
            if stored_tickers is None:
                stored_tickers = entry.get('tickers')
            acts_list   = [np.array(a) for a in entry['actions']]
            states_list = [np.array(s) for s in entry['states']]
            concat_actions = np.concatenate(acts_list, axis=0)     # (T_total, stock_dim)
            concat_prices  = np.concatenate(
                [s[:, 1:stock_dim + 1] for s in states_list], axis=0
            )                                                         # (T_total, stock_dim)
            break  # use first trial with data

        if concat_actions is None:
            continue

        if stored_tickers is None:
            stored_tickers = ticker_names[:stock_dim]

        n_sample    = min(n_samples, stock_dim)
        sampled_idx = random.sample(range(stock_dim), n_sample)

        fig, axes = plt.subplots(n_sample, 1, figsize=(11, 3.5 * n_sample), squeeze=False)
        fig.subplots_adjust(hspace=0.55)

        T = len(concat_actions)
        t = np.arange(T)

        action_color = PALETTE[0]   # blue
        price_color  = PALETTE[1]   # orange

        for row, asset_idx in enumerate(sampled_idx):
            ax_act   = axes[row][0]
            ax_price = ax_act.twinx()

            ticker = stored_tickers[asset_idx] if asset_idx < len(stored_tickers) else str(asset_idx)

            act   = concat_actions[:, asset_idx] if concat_actions.ndim == 2 else concat_actions
            price = concat_prices[:, asset_idx]

            # Actions as semi-transparent bars so buy/sell is immediately visible
            ax_act.bar(t, act, color=action_color, alpha=0.35, width=1.0, label='Action')
            ax_act.axhline(0, color=action_color, linewidth=0.7, linestyle=':')
            ax_act.set_ylabel('Action (−1 sell → +1 buy)', color=action_color, fontsize=9)
            ax_act.tick_params(axis='y', labelcolor=action_color)
            ax_act.spines['top'].set_visible(False)

            ax_price.plot(t, price, color=price_color, linewidth=1.6, label='Price ($)', zorder=3)
            ax_price.set_ylabel('Price ($)', color=price_color, fontsize=9)
            ax_price.tick_params(axis='y', labelcolor=price_color)
            ax_price.spines['top'].set_visible(False)

            ax_act.set_title(f'{ticker}', fontsize=10, pad=4)
            if row == n_sample - 1:
                ax_act.set_xlabel('Trading Day')

            h1, l1 = ax_act.get_legend_handles_labels()
            h2, l2 = ax_price.get_legend_handles_labels()
            ax_price.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8, frameon=True)

        agent_label = agent.replace('_', ' ').title()
        fig.suptitle(
            f'Actions vs Price — {agent_label}  [{dataset_name}]',
            fontsize=12, fontweight='bold',
        )

        agent_safe = agent.replace('/', '_')
        for ext in ('png', 'pdf'):
            fig.savefig(
                os.path.join(out_dir, f'{dataset_name}_actions_price_{agent_safe}.{ext}'),
                dpi=SAVE_DPI, bbox_inches='tight',
            )
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Model-free vs model-based comparison                                          #
# --------------------------------------------------------------------------- #

# Groups of related agents — base (model-free) first, then model-based variants
_AGENT_FAMILIES = {
    'TD3': ['td3', 'dense_td3', 'autoformer_td3'],
    'PPO': ['ppo', 'dense_ppo', 'autoformer_ppo'],
    'DDPG': ['ddpg', 'dense_ddpg', 'autoformer_ddpg'],
}

# Human-readable labels for display
_AGENT_LABELS = {
    'td3': 'TD3 (model-free)',
    'dense_td3': 'Dense TD3',
    'autoformer_td3': 'Autoformer TD3',
    'ppo': 'PPO (model-free)',
    'dense_ppo': 'Dense PPO',
    'autoformer_ppo': 'Autoformer PPO',
    'ddpg': 'DDPG (model-free)',
    'dense_ddpg': 'Dense DDPG',
    'autoformer_ddpg': 'Autoformer DDPG',
}


def _plot_model_free_vs_model_based(
    results: dict,
    trials: list,
    dataset_name: str,
    agents: list,
    features: int,
    out_dir: str,
):
    """Side-by-side panels comparing portfolio value for each algorithm family."""
    _apply_paper_style()

    families_to_plot = {
        name: [a for a in members if a in agents]
        for name, members in _AGENT_FAMILIES.items()
    }
    # Keep only families with ≥ 2 present agents
    families_to_plot = {k: v for k, v in families_to_plot.items() if len(v) >= 2}

    if not families_to_plot:
        return

    n_fam = len(families_to_plot)
    fig, axes = plt.subplots(1, n_fam, figsize=(7 * n_fam, 5), squeeze=False)
    fig.subplots_adjust(wspace=0.32, top=0.88, bottom=0.12)

    for col, (family_name, family_agents) in enumerate(families_to_plot.items()):
        ax = axes[0][col]

        for i, agent in enumerate(family_agents):
            color   = PALETTE[i % len(PALETTE)]
            all_pv  = []

            for trial in trials:
                try:
                    entry     = results[trial][dataset_name][agent]
                    stock_dim = entry.get('stock_dim')
                    for ep_states in entry['states']:
                        pv = _portfolio_values_from_states(np.array(ep_states), features, stock_dim)
                        all_pv.append(pv)
                except (KeyError, IndexError):
                    continue

            if not all_pv:
                continue

            min_len    = min(len(x) for x in all_pv)
            arr        = np.array([x[:min_len] for x in all_pv])
            normalised = arr / arr[:, [0]] * 100
            mean_pv    = normalised.mean(axis=0)
            std_pv     = normalised.std(axis=0)
            t          = np.arange(min_len)

            label = _AGENT_LABELS.get(agent, agent)
            ax.plot(t, mean_pv, color=color, label=label, linewidth=2.0, zorder=3)
            if len(all_pv) > 1:
                ax.fill_between(t, mean_pv - std_pv, mean_pv + std_pv, color=color, alpha=0.15)

        ax.axhline(100, color='black', linestyle=':', linewidth=0.9, alpha=0.5)
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Portfolio Value (indexed, start = 100)')
        ax.set_title(f'{family_name} Family', fontweight='bold')
        ax.legend(loc='upper left', frameon=True)

    fig.suptitle(
        f'Model-Free vs Model-Based — {dataset_name}',
        fontsize=13, fontweight='bold',
    )
    for ext in ('png', 'pdf'):
        fig.savefig(
            os.path.join(out_dir, f'{dataset_name}_model_comparison.{ext}'),
            dpi=SAVE_DPI, bbox_inches='tight',
        )
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Action heatmap                                                                #
# --------------------------------------------------------------------------- #

def _plot_action_heatmap(
    results: dict, trials: list, dataset_name: str, agents: list,
    features: int, ticker_names: list, out_dir: str,
):
    _apply_paper_style()

    # Collect agents that have non-empty actions
    act_agents = [
        a for a in agents
        if any(len(results[t][dataset_name].get(a, {}).get('actions', [])) > 0
               for t in trials)
    ]
    if not act_agents:
        return

    n_cols = len(act_agents)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.0 * n_cols, 4.5), squeeze=False)
    fig.subplots_adjust(wspace=0.35, top=0.88, bottom=0.12)

    for col, agent in enumerate(act_agents):
        ax = axes[0][col]
        all_actions = []
        stock_dim = None
        for trial in trials:
            try:
                entry    = results[trial][dataset_name][agent]
                stock_dim = entry.get('stock_dim', features)
                for ep_actions in entry['actions']:
                    ep_arr = np.array(ep_actions)
                    if ep_arr.ndim == 1:
                        ep_arr = ep_arr.reshape(-1, 1)
                    all_actions.append(ep_arr)
            except (KeyError, IndexError):
                continue
        if not all_actions:
            ax.set_visible(False)
            continue

        action_mat = np.concatenate(all_actions, axis=0).T   # (stock_dim, T)
        n = stock_dim or action_mat.shape[0]
        action_mat = action_mat[:n, :]
        labels = [ticker_names[i] if i < len(ticker_names) else str(i) for i in range(n)]

        im = ax.imshow(
            action_mat, aspect='auto', cmap='RdYlGn',
            vmin=-1, vmax=1, interpolation='nearest',
        )
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel('Trading Day')
        ax.set_title(agent, fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03,
                     label='Action (−1=sell, +1=buy)')

    fig.suptitle(f'Action Heatmap — {dataset_name}', fontsize=12, fontweight='bold')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'{dataset_name}_action_heatmap.{ext}'),
                    dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main report function                                                          #
# --------------------------------------------------------------------------- #

def create_performance_report(
    results: dict,
    tickers: dict,
    output_dir: str = 'results/final_results',
):
    """Generate a complete performance report: metrics tables + paper-quality figures.

    Parameters
    ----------
    results : dict
        Nested dict ``results[trial][dataset_name][agent]`` where each leaf
        contains at minimum ``'states'`` (T × state_dim ndarray) and optionally
        ``'predictions'``, ``'actuals'``, and ``'rewards'``.
    tickers : dict
        ``tickers[dataset_name]`` → list of ticker strings for that dataset.
    output_dir : str
        Directory to write all outputs.  Created if absent.

    Outputs (per dataset)
    ---------------------
    ``{dataset}_metrics.csv``          – raw metrics table
    ``{dataset}_metrics_table.{png,pdf}`` – formatted table figure
    ``{dataset}_portfolio.{png,pdf}``     – net value + drawdown chart
    ``{dataset}_predictions.{png,pdf}``   – prediction vs actual grid (if available)
    """
    os.makedirs(output_dir, exist_ok=True)

    trials  = list(results.keys())
    if not trials:
        raise ValueError('results dict is empty')

    datasets = list(results[trials[0]].keys())
    agents   = list(results[trials[0]][datasets[0]].keys())

    for dataset_name in datasets:
        features    = len(tickers[dataset_name])
        ticker_names = list(tickers[dataset_name])

        # ------------------------------------------------------------------ #
        # 1. Compute metrics for every agent (averaged over trials)           #
        # ------------------------------------------------------------------ #
        fin_rows  = {}   # agent → financial metric dict
        pred_rows = {}   # agent → prediction metric dict

        for agent in agents:
            trial_fin  = []
            trial_pred = []

            for trial in trials:
                entry = results[trial][dataset_name].get(agent, {})

                # states is a list of per-episode arrays (T, obs_dim)
                states = entry.get('states')
                if states is not None:
                    stock_dim = entry.get('stock_dim')
                    for ep_states in states:
                        ep_states = np.array(ep_states)
                        if ep_states.ndim == 2 and len(ep_states) > 1:
                            pv = _portfolio_values_from_states(ep_states, features, stock_dim)
                            trial_fin.append(_compute_financial_metrics(pv))

                # predictions/actuals are lists of per-episode arrays — concatenate
                preds   = entry.get('predictions')
                actuals = entry.get('actuals')
                if preds is not None and actuals is not None and len(preds) > 0:
                    p = np.concatenate(preds, axis=0)
                    a = np.concatenate(actuals, axis=0)
                    # predictions may be (T, 1, pred_len, stock_dim) — squeeze batch
                    # dim and take step-0 (next-day) to align with actuals (T, stock_dim)
                    if p.ndim == 4:
                        p = p[:, 0, 0, :]
                    elif p.ndim == 3:
                        p = p[:, 0, :]
                    trial_pred.append(_compute_prediction_metrics(p, a))

            if trial_fin:
                # Average each metric across trials
                all_keys = trial_fin[0].keys()
                fin_rows[agent] = {
                    k: round(float(np.mean([d[k] for d in trial_fin if np.isfinite(d[k])])), 3)
                    for k in all_keys
                }

            if trial_pred:
                all_keys = trial_pred[0].keys()
                pred_rows[agent] = {
                    k: round(float(np.mean([d[k] for d in trial_pred if np.isfinite(d[k])])), 4)
                    for k in all_keys
                }

        if not fin_rows:
            print(f'[create_performance_report] No state data found for {dataset_name}, skipping.')
            continue

        # Build combined DataFrame (financial rows first, then prediction rows)
        df_fin  = pd.DataFrame(fin_rows)
        df_pred = pd.DataFrame(pred_rows) if pred_rows else None
        df_all  = pd.concat([df_fin, df_pred]) if df_pred is not None else df_fin

        # ------------------------------------------------------------------ #
        # 2. Save metrics table                                               #
        # ------------------------------------------------------------------ #
        _save_metrics_table(df_all, dataset_name, output_dir)

        # ------------------------------------------------------------------ #
        # 3. Portfolio net value + drawdown chart                             #
        # ------------------------------------------------------------------ #
        _plot_portfolio_performance(
            results, trials, dataset_name, agents, features, output_dir
        )

        # ------------------------------------------------------------------ #
        # 4. Prediction vs actual (only if any agent has predictions)         #
        # ------------------------------------------------------------------ #
        _plot_predictions_vs_actual(
            results, trials, dataset_name, agents, features, ticker_names, output_dir
        )

        # ------------------------------------------------------------------ #
        # 5. Rolling Sharpe ratio                                             #
        # ------------------------------------------------------------------ #
        _plot_rolling_sharpe(
            results, trials, dataset_name, agents, features, output_dir
        )

        # ------------------------------------------------------------------ #
        # 6. Daily return distribution                                        #
        # ------------------------------------------------------------------ #
        _plot_return_distribution(
            results, trials, dataset_name, agents, features, output_dir
        )

        # ------------------------------------------------------------------ #
        # 7. Prediction error over time                                       #
        # ------------------------------------------------------------------ #
        _plot_prediction_error(
            results, trials, dataset_name, agents, features, ticker_names, output_dir
        )

        # ------------------------------------------------------------------ #
        # 8. Predicted vs actual return scatter                               #
        # ------------------------------------------------------------------ #
        _plot_pred_scatter(
            results, trials, dataset_name, agents, features, ticker_names, output_dir
        )

        # ------------------------------------------------------------------ #
        # 9. Action heatmap                                                   #
        # ------------------------------------------------------------------ #
        _plot_action_heatmap(
            results, trials, dataset_name, agents, features, ticker_names, output_dir
        )

        # ------------------------------------------------------------------ #
        # 10. Actions vs price (per agent, 3 sampled assets)                  #
        # ------------------------------------------------------------------ #
        _plot_actions_vs_price(
            results, trials, dataset_name, agents, features, ticker_names, output_dir,
            n_samples=3,
        )

        # ------------------------------------------------------------------ #
        # 11. Model-free vs model-based portfolio comparison                  #
        # ------------------------------------------------------------------ #
        _plot_model_free_vs_model_based(
            results, trials, dataset_name, agents, features, output_dir
        )

        print(f'[create_performance_report] Saved outputs for "{dataset_name}" → {output_dir}')


# --------------------------------------------------------------------------- #
# Original quick-look plot (kept for backwards compatibility)                  #
# --------------------------------------------------------------------------- #

def plot_results(results, tickers):

    datasets = list(results[1].keys())
    trials = list(results.keys())
    agents = list(results[1][datasets[0]].keys())
    categories = list(results[1][datasets[0]][agents[0]].keys())

    for data_name in datasets:

        features = len(tickers[data_name])

        plt.figure(figsize=(12, 6), dpi=120)
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.title(f"Porfolio Net Value On {data_name} Dataset On Each Strategy")
        plt.xlabel('Timestep')
        plt.ylabel('Value')

        for i, agent in enumerate(agents):
            balances = []
            prices = []
            holdings = []
            for trial in trials:
                state = np.array(results[trial][data_name][agent]['states'])
                balances.append(state[:, 0])
                prices.append(state[:, 1:features+1])
                holdings.append(state[:, features+1:features*2+1])

            holdings = np.array(holdings)
            prices = np.array(prices)
            balances = np.array(balances)

            balances = balances.reshape(balances.shape[0], balances.shape[1], 1)
            states = np.concatenate([holdings*prices,balances], axis = -1)
            states = np.sum(states, axis = -1)

            net = np.array(states).mean(axis=0)

            for j in range(len(trials)):
                plt.plot(states[j].flatten(), alpha = 0.25, color = colours[i])

            plt.plot(net, label = f'{agent} avarage', color = colours[i])

        plt.legend()
        plt.savefig(save_graphs +'/'+ data_name + '_state_graph.png' )
        plt.close()

        plt.style.use("seaborn-v0_8-whitegrid")

        max_graphs = 4

        fig, axes = plt.subplots(min(max_graphs, features), 1, figsize = (16,8), sharex = False, sharey=False)
        if min(max_graphs, features) == 1:
             axes = [axes]
        else:
            axes = axes.flatten()

        plt.title(f'Agent predicitons on dataset {data_name} for an example asset')
        plt.ylabel('Price')
        plt.xlabel('Timestep')

        for i, sample in enumerate(random.sample(list(range(features)), min(3, features))):

            ax = axes[i]
            actual = np.array(results[trials[0]][data_name][agents[0]]['actuals'])

            for j, agent in enumerate(agents):
                trial = list(random.sample(trials, 1))[0]

                if agent == 'Buy And Hold':
                    continue

                prediction = np.array(results[trial][data_name][agent]['predictions'])

                assert prediction.shape == actual.shape, f'Shapes different {prediction.shape}, {actual.shape}'

                ax.plot(prediction[ :, sample ], label = f'{agent} prediciton')

            ax.plot(actual[:, sample], label = f'actual')

            ax.set_title(f'Predictions on {tickers[data_name][sample]} ')
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(alpha=0.25)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False)

        fig.suptitle("Agent Prediction vs Actual (Sampled assets Graphs)",
                    fontsize=16,
                    weight="bold")

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        plt.savefig(save_graphs +'/'+ data_name + '_predicitons.png' )
        plt.close()
