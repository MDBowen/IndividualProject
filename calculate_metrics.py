import numpy as np
import pandas as pd
from typing import Dict, Any
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class MetricsCalculator:
    """Calculate performance metrics for trading agents across trials."""
    
    def __init__(self, initial_amount: float, risk_free_rate: float = 0.02):
        """
        Args:
            initial_amount: Initial portfolio amount
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.initial_amount = initial_amount
        self.risk_free_rate = risk_free_rate
    
    def calculate_profit(self, start_worth, end_worth) -> float:
        """Calculate total profit from rewards."""

        # Handle case where rewards might be appended incorrectly
        # total_reward = sum([r for r in rewards if isinstance(r, (int, float))])

        return end_worth - start_worth
    
    def calculate_cumulative_return(self, start_worth, end_worth) -> float:
        """Calculate cumulative return percentage."""
        
        total_reward = self.calculate_profit(start_worth, end_worth)
        ccr = (total_reward / start_worth) * 100
        print(f'Cumulative return: {ccr:.3f}%')

        return ccr
    
    def calculate_sharpe_ratio(self, rewards: list, periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            rewards: List of rewards at each step
            periods_per_year: Number of trading periods per year (default 252 for daily)
        """
        
        # Convert rewards to returns
        # rewards_array = np.array([r for r in rewards if isinstance(r, (int, float))])

        rewards_array = np.array(rewards)
        
        
        # Calculate daily returns
        mean_return = np.mean(rewards_array)
        std_return = np.std(rewards_array)
        
        # Annualize
        daily_sharpe = mean_return / std_return if std_return > 0 else 0
        annual_sharpe = daily_sharpe * np.sqrt(periods_per_year)
        print('Sharpe ratio:', annual_sharpe)
        return annual_sharpe
    
    def calculate_prediction_accuracy(self, predictions: np.array, actuals: np.array) -> float:
        """
        Calculate prediction accuracy - directional accuracy (up/down).
        
        Args:
            predictions: List of predicted prices
            actuals: List of actual prices
        """
        
        assert predictions.shape == actuals.shape, f'Shapes of predictions and actuals must match for accuracy calculation. Got {predictions.shape} and {actuals.shape}.'
        
        correct = 0
        total = 0

        # changes: sign prediction - actual[i] == sign actual[i+1] - actual[i]


        pred_change = predictions[1:] - actuals[:-1]  # predicted change
        actual_change = actuals[1:] - actuals[:-1] # actual[i+1] - actual[i]

        print('Pred change shape:', pred_change.shape)
        print('Actual change shape:', actual_change.shape)

        correct = np.sum(np.sign(pred_change) == np.sign(actual_change))
        
        total = pred_change.shape[0] * pred_change.shape[1]  # total predictions made (number of time steps * number of assets)

        acc = (correct / total) * 100 if total > 0 else 0.0
        print(f'Prediction accuracy: {acc:.2f}% ({correct}/{total} correct predictions)')
        
        # assert False
        return acc

        for i in range(len(predictions) - 1):
            try:
                # Get change direction
                pred_change = predictions[i+1] - predictions[i]
                actual_change = actuals[i+1] - actuals[i]
                
                # Check if direction matches
                if (pred_change > 0 and actual_change > 0) or (pred_change < 0 and actual_change < 0):
                    correct += 1
                total += 1
            except:
                continue
        
        if total == 0:
            return 0.0
        
        return (correct / total) * 100
    
    def calculate_mse(self, predictions: list, actuals: list) -> float:
        """
        Calculate Mean Squared Error between predictions and actuals.
        
        Args:
            predictions: List of predicted prices/values
            actuals: List of actual prices/values
        """        
        assert predictions.shape == actuals.shape, f'Shapes of predictions and actuals must match for MSE calculation. Got {predictions.shape} and {actuals.shape}.'
        
        min = actuals.min()
        max = actuals.max()

        print(max, min)
        print('Predictions shape before scaling:', predictions.shape)

        # assert min.shape == max.shape == predictions.shape[1:] == actuals.shape[1:], f'Shapes of min, max, and predictions must match. Got {min.shape}, {max.shape}, and {predictions.shape[1:]}.'
        norm_preds = (predictions - min)/(max - min + 1e-8) # add small epsilon to avoid division by zero
        norm_act = (actuals - min)/(max - min + 1e-8)
        assert np.all(norm_preds <= 1) and np.all(norm_act <= 1), f"Normalized predictions and actuals should be <= 1 {norm_preds}, {norm_act}"
        mse = np.mean((norm_preds - norm_act) ** 2)
        print(f'Mean Squared Error: {mse:.4f}')
        return mse
        
    
    def calculate_max_drawdown(self, rewards: list) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            rewards: List of rewards at each step
        """
        if len(rewards) < 2:
            return 0.0
        
        cumulative_rewards = np.cumsum(rewards)
        running_max = np.maximum.accumulate(cumulative_rewards)
        drawdowns = (cumulative_rewards - running_max) / np.maximum(running_max, self.initial_amount)
        print(f'Max drawdown: {np.min(drawdowns) * 100:.2f}%')
        return np.min(drawdowns) * 100
    
    def calculate_win_rate(self, rewards: list) -> float:
        """
        Calculate win rate - percentage of positive reward steps.
        
        Args:
            rewards: List of rewards at each step
        """
        temp = np.zeros_like(rewards)
        temp[rewards > 0] = 1
        positive_rewards = np.sum(temp)
        print(f'Win rate: {positive_rewards}/{rewards.shape[0]} positive reward steps')
        return (positive_rewards / rewards.shape[0]) * 100
    
    def calculate_calmar_ratio(self, rewards: list, annual_return: float) -> float:
        """
        Calculate Calmar ratio = Annual Return / Max Drawdown.
        
        Args:
            rewards: List of rewards at each step
            annual_return: Annual return percentage
        """
        max_dd = abs(self.calculate_max_drawdown(rewards))
        
        if max_dd == 0:
            return 0.0
        
        ratio = annual_return / max_dd if max_dd > 0 else 0.0
        
        print(f'Calmar ratio: {ratio}')
        
        return ratio
    
    def calculate_sortino_ratio(self, rewards: list, periods_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio (similar to Sharpe but only penalizes downside volatility).
        
        Args:
            rewards: List of rewards at each step
            periods_per_year: Number of trading periods per year
        """
        if len(rewards) < 2:
            return 0.0
        
        rewards_array = np.array([r for r in rewards if isinstance(r, (int, float))])
        
        if len(rewards_array) < 2:
            return 0.0
        
        mean_return = np.mean(rewards_array)
        negative_returns = rewards_array[rewards_array < 0]
        
        if len(negative_returns) == 0:
            downside_std = 0
        else:
            downside_std = np.std(negative_returns)
        
        if downside_std == 0:
            return 0.0
        
        daily_sortino = mean_return / downside_std
        annual_sortino = daily_sortino * np.sqrt(periods_per_year)

        print('Sortino ratio:', annual_sortino)
        
        return annual_sortino
    
    def calculate_all_metrics(self, data: Dict[str, Any], agent = '') -> Dict[str, float]:
        """
        Calculate all metrics for a single trial result.
        
        Args:
            data: Dictionary with keys 'rewards', 'predictions', 'actuals'
        
        Returns:
            Dictionary of all calculated metrics
        """
        # rewards = data.get('rewards', [])
        # predictions = data.get('predictions', [])
        # actuals = data.get('actuals', [])

        print('Calculating metrics for data...')

        rewards = np.array(data['rewards'])

        predictions = np.array(data['predictions'])
        actuals = np.array(data['actuals'])

        if agent != 'Buy And Hold':
            temp_pred = predictions.copy()

            mask = np.all(temp_pred != 0, axis=-1)  # Only keep rows where all features have non-zero predictions

            predictions = predictions[mask]
            actuals = actuals[mask]

            print('Predictions:', np.array(predictions).shape)
            print('Actuals:', np.array(actuals).shape)
            print('Temp: ',temp_pred.shape, mask.shape)

        print('Rewards:', np.array(rewards).shape)

        intial_worth = data['initial_worth']
        end_worth = data['final_worth']

        
        cumulative_return = self.calculate_cumulative_return(intial_worth, end_worth)
        max_dd = self.calculate_max_drawdown(rewards)
        annual_return = cumulative_return  # Simplified - adjust for actual period if needed
        
        metrics = {
            'start_worth': intial_worth,
            'end_worth': end_worth,
            'profit': self.calculate_profit(intial_worth, end_worth),
            'cumulative_return_%': cumulative_return,
            'sharpe_ratio': self.calculate_sharpe_ratio(rewards),
            'sortino_ratio': self.calculate_sortino_ratio(rewards),
            'max_drawdown_%': max_dd,
            'win_rate_%': self.calculate_win_rate(rewards),
            'calmar_ratio': self.calculate_calmar_ratio(rewards, annual_return),
            'prediction_accuracy_%': 0 if agent == 'Buy And Hold' else self.calculate_prediction_accuracy(predictions, actuals),
            'mse': 0 if agent == 'Buy And Hold' else self.calculate_mse(predictions, actuals),
            'num_trades': len([a for a in data.get('actions', []) if a is not None]),
        }
        
        return metrics


def aggregate_metrics_across_trials(trials: Dict[int, Dict[str, Dict[str, Any]]], 
                                    calculator: MetricsCalculator) -> pd.DataFrame:
    """
    Aggregate metrics across all trials.
    
    Args:
        trials: Dictionary structure {trial_num: {dataset: {agent: data}}}
        calculator: MetricsCalculator instance
    
    Returns:
        DataFrame with aggregated metrics
    """
    all_metrics = {}

    print('Agregating metrics across trials...')
    
    # Calculate metrics for each trial
    trial_metrics = {}
    for trial_num, datasets in trials.items():
        trial_metrics[trial_num] = {}
        for dataset, agents in datasets.items():
            trial_metrics[trial_num][dataset] = {}
            for agent, data in agents.items():
                print(f'Calculating metrics for trial {trial_num}, dataset {dataset}, agent {agent}...')
                trial_metrics[trial_num][dataset][agent] = calculator.calculate_all_metrics(data, agent)
            print(f'Calculated metrics for trial {trial_num} on dataset {dataset}')
    
    print('Calculated metrics for all trials. Now aggregating...')
    # Aggregate across trials
    results = []
    
    # Get unique datasets and agents
    datasets = list(trials[1].keys())
    agents = list(trials[1][datasets[0]].keys())
    
    for dataset in datasets:
        for agent in agents:
            row = {'Dataset': dataset, 'Agent': agent}
            
            # Get metrics from all trials
            metrics_per_trial = {}
            for trial_num in trials.keys():
                metrics_per_trial[trial_num] = trial_metrics[trial_num][dataset][agent]
            
            # Calculate mean and std across trials
            metric_names = list(metrics_per_trial[1].keys())
            
            for metric_name in metric_names:
                values = [metrics_per_trial[t][metric_name] for t in trials.keys()]
                row[f'{metric_name}_mean'] = np.mean(values, dtype=np.float64) if all(isinstance(v, (int, float)) for v in values) else 0
                row[f'{metric_name}_std'] = np.std(values, dtype=np.float64) if all(isinstance(v, (int, float)) for v in values) else 0
            
            results.append(row)

    print('Aggregated metrics for Datasets:', datasets)
    
    return pd.DataFrame(results)


def print_metrics_summary(results_df: pd.DataFrame):
    """Print a nicely formatted summary of metrics."""
    
    # Define which metrics to display
    display_metrics = [
        'profit',
        'cumulative_return_%',
        'sharpe_ratio',
        'sortino_ratio',
        'max_drawdown_%',
        'win_rate_%',
        'prediction_accuracy_%',
        'mse',
        'num_trades'
    ]
    
    print("\n" + "="*120)
    print("TRADING AGENT PERFORMANCE METRICS (Averaged over 3 trials)")
    print("="*120)
    
    for dataset in results_df['Dataset'].unique():
        print(f"\n{'Dataset: ' + dataset:^120}")
        print("-"*120)
        
        dataset_df = results_df[results_df['Dataset'] == dataset]
        
        for _, row in dataset_df.iterrows():
            print(f"\n  {row['Agent']}")
            print(f"  {'-'*50}")
            
            for metric in display_metrics:
                mean_col = f'{metric}_mean'
                std_col = f'{metric}_std'
                
                if mean_col in row:
                    mean_val = row[mean_col]
                    std_val = row[std_col]
                    
                    if 'ratio' in metric:
                        print(f"    {metric:.<40} {mean_val:>10.4f} ± {std_val:>8.4f}")
                    elif '%' in metric:
                        print(f"    {metric:.<40} {mean_val:>10.2f}% ± {std_val:>8.2f}%")
                    else:
                        print(f"    {metric:.<40} {mean_val:>10.2f} ± {std_val:>8.2f}")


def save_metrics_to_csv(results_df: pd.DataFrame, filename: str = 'metrics_summary.csv'):
    """Save metrics to CSV file."""
    results_df.to_csv(filename, index=False)
    print(f"\nMetrics saved to {filename}")


def save_metrics_to_pickle(results_df: pd.DataFrame, filename: str = 'metrics_summary.pkl'):
    """Save metrics to pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(results_df, f)
    print(f"Metrics saved to {filename}")


def create_matplotlib_table(results_df: pd.DataFrame, figsize: tuple = (20, 12), 
                           save_path: str = 'results/table.png', show: bool = True):
    """
    Create a matplotlib table visualization of metrics.
    
    Args:
        results_df: DataFrame with metrics
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
        show: Whether to display the figure
    """
    
    # Define metrics to display
    display_metrics = [
        'profit',
        'cumulative_return_%',
        'sharpe_ratio',
        'sortino_ratio',
        'max_drawdown_%',
        'win_rate_%',
        'prediction_accuracy_%',
        'mse',
        'num_trades'
    ]
    
    fig, axes = plt.subplots(len(results_df['Dataset'].unique()), 1, 
                            figsize=figsize)
    
    # Handle single dataset case
    if len(results_df['Dataset'].unique()) == 1:
        axes = [axes]
    
    for idx, dataset in enumerate(sorted(results_df['Dataset'].unique())):
        ax = axes[idx]
        dataset_df = results_df[results_df['Dataset'] == dataset]
        
        # Prepare table data
        table_data = []
        agents = dataset_df['Agent'].values
        
        # Header row
        header = ['Agent'] + [m.replace('_', '\n') for m in display_metrics]
        table_data.append(header)
        
        # Data rows
        for _, row in dataset_df.iterrows():
            row_data = [row['Agent']]
            for metric in display_metrics:
                mean_col = f'{metric}_mean'
                std_col = f'{metric}_std'
                
                if mean_col in row:
                    mean_val = row[mean_col]
                    std_val = row[std_col]
                    
                    if 'ratio' in metric:
                        cell_text = f'{mean_val:.2f}±{std_val:.2f}'
                    elif '%' in metric:
                        cell_text = f'{mean_val:.2f}%±{std_val:.2f}%'
                    else:
                        cell_text = f'{mean_val:.0f}±{std_val:.0f}'
                else:
                    cell_text = 'N/A'
                
                row_data.append(cell_text)
            
            table_data.append(row_data)
        
        # Create table
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.12] + [0.125]*len(display_metrics))
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)
        
        # Style header row
        for i in range(len(header)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(len(header)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('#ffffff')
                import re
                # Highlight best performing cells (highest values)
                if j > 0:  # Skip agent name column
                    metric = display_metrics[j - 1]
                    try:
                        values = [float(re.sub(r'[^0-9]' ,'' ,str(td[j])).split('±')[0]) 
                                for td in table_data[1:]]
                    except ValueError:
                        print(f"Could not convert values for metric '{metric}' to float for highlighting. Skipping highlighting for this metric.")
                        print(f"Values were: {[td[j] for td in table_data[1:]]}")
                    
                        raise ValueError(f"Could not convert values for metric '{metric}' to float for highlighting. Skipping highlighting for this metric.")
                    try:
                        max_val = max(values)
                        current_val = values[i - 1]
                        if current_val == max_val and current_val > 0:
                            table[(i, j)].set_facecolor('#FFF59D')
                    except:
                        pass
        
        ax.set_title(f'Dataset: {dataset}', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Table saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig, axes


if __name__ == '__main__':
    """
    Example usage - call this after running predictor_experiments.py:
    
    from calculate_metrics import MetricsCalculator, aggregate_metrics_across_trials, print_metrics_summary, create_matplotlib_table
    from predictor_experiments import trials  # Assuming trials is saved/accessible
    
    calculator = MetricsCalculator(initial_amount=100_000)
    results_df = aggregate_metrics_across_trials(trials, calculator)
    
    print_metrics_summary(results_df)
    
    # Create matplotlib table
    create_matplotlib_table(results_df, save_path='metrics_table.png', show=True)
    
    save_metrics_to_csv(results_df)
    save_metrics_to_pickle(results_df)
    """
    print("Metrics calculation module loaded. Import and use with your trials data.")
