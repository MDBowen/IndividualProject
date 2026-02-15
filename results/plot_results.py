
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import os
import random
save_graphs = 'results/graphs'
save_df = 'results/df.csv'

colours = list(colors.TABLEAU_COLORS)

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

        # plt.figure(figsize=(12, 6), dpi=120)
        plt.style.use("seaborn-v0_8-whitegrid")

        fig, axes = plt.subplots(3, 1, figsize = (14,8), sharex = False, sharey=False)

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
                   # clean look
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(alpha=0.25) 

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False)

        fig.suptitle("Agent Prediction vs Actual (Sampled assets Graphs)",
                    fontsize=16,
                    weight="bold")

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        # plt.show()
        
        plt.savefig(save_graphs +'/'+ data_name + '_predicitons.png' )
        plt.close()






                

    
