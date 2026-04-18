
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
    'figure.constrained_layout.use': True,
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

    fig = plt.figure(figsize=(9, 6))
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

    # Only include agents that store predictions
    pred_agents = [
        a for a in agents
        if any(
            'predictions' in results[t][dataset_name].get(a, {})
            for t in trials
        )
    ]
    if not pred_agents:
        return

    n_sample = min(n_assets, features)
    sampled_idx = random.sample(range(features), n_sample)

    n_rows = n_sample
    n_cols = len(pred_agents)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 2.8 * n_rows),
        squeeze=False,
        sharey='row',
    )

    trial0 = trials[0]

    for col, agent in enumerate(pred_agents):
        agent_color = PALETTE[col % len(PALETTE)]

        for row, asset_idx in enumerate(sampled_idx):
            ax = axes[row][col]
            ticker = ticker_names[asset_idx] if asset_idx < len(ticker_names) else str(asset_idx)

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

            ax.plot(t, actuals_ref[:t_end], color='#2c3e50', linewidth=1.4,
                    label='Actual', zorder=4)
            ax.plot(t, mean_pred, color=agent_color, linewidth=1.4,
                    linestyle='--', label=f'{agent} prediction', zorder=3)
            if len(trials) > 1:
                ax.fill_between(t, mean_pred - std_pred, mean_pred + std_pred,
                                color=agent_color, alpha=0.2)

            ax.set_title(f'{ticker}', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if row == n_rows - 1:
                ax.set_xlabel('Trading Day')
            if col == 0:
                ax.set_ylabel('Price')

    # Shared legend at top — collect from all axes in case row 0 col 0 was skipped
    handles, labels = [], []
    for row_axes in axes:
        for ax in row_axes:
            h, l = ax.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=max(1, len(labels)),
                   frameon=True, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(
        f'Price Predictions vs Actual — {dataset_name}',
        fontsize=12, fontweight='bold', y=1.04,
    )
    plt.tight_layout()
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
