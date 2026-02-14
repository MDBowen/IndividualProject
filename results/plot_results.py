
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import os

save_graphs = 'results/graphs'
save_df = 'results/df.csv'

colours = list(colors.TABLEAU_COLORS)

def plot_results(results = None, features = 3):



    print('Colours',colours)
    datasets = list(results[1].keys())
    print('Sets :',datasets)
    trials = list(results.keys())
    agents = list(results[1][datasets[0]].keys())
    categories = list(results[1][datasets[0]][agents[0]].keys())
    print('agents:',agents)
    print('trials:',trials)
    print('categories', categories)
    # Plot Net Values

    # return

    results = results[1][datasets[0]]['Buy And Hold']
    print(list(results.keys()))
    actions = np.array(results['actions'])
    print('actions shape', actions.shape)
    states = np.array(results['states'])
    print('states shape:',states.shape)
    balances = states[0, :, 0]
    prices = states[0, :, 1:features+1]
    holdings = states[0, :, features+1: features*2 +1]
    print('prices shape',prices.shape)
    print('holdings shape',holdings.shape)

    print('first state', states[0][0] )


    for data_name in datasets:

        plt.title(f"Net Values on {data_name} dataset")
        plt.xlabel('Timestep')
        plt.ylabel('Net Values')
        for i, agent in enumerate(agents):
            print('Agent:', agent)
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

            print('states ', np.array(results[trial][data_name][agent]['states'])[-10:, :])
            print('actions ', np.array(results[trial][data_name][agent]['actions'])[-10:, :])

            print('holdings:', holdings.shape)
            print('prices:',prices.shape)
            print('balances:',balances.shape)

            # print('holdings:',holdings)
            # print('balances',balances)

            balances = balances.reshape(balances.shape[0], balances.shape[1], 1)
            states = np.concatenate([holdings*prices,balances], axis = -1)
            states = np.sum(states, axis = -1)
            print('states:',states.shape)

            net = np.array(states).mean(axis=0)

            print('Net',net.shape)

            for j in range(len(trials)):
                plt.plot(states[j].flatten(), alpha = 0.25, color = colours[i])

            plt.plot(net, label = f'{agent} avarage', color = colours[i])

        plt.legend()
        plt.savefig(save_graphs +'/'+ data_name + '_state_graph.png' )
        plt.close()

        

        plt.title(f'Agent predicitons on dataset {data_name} for an example asset')
        plt.ylabel('Price')
        plt.xlabel('Timestep')

        for i, agent in enumerate(agents):
            for trial in trials:

                if agent == 'Buy And Hold':
                    break

                prediction = np.array(results[trial][data_name][agent]['predictions'])
                actual = np.array(results[trial][data_name][agent]['actual'])

                assert prediction.shape == actual.shape
                
                plt.plot(prediction[ :, 0 ], label = f'{agent} prediciton')
                plt.plot(actual[:, 0], label = f'{agent} actual')
            
        plt.legend()
        plt.savefig(save_graphs +'/'+ data_name + '_predicitons.png' )
        plt.close()






                

    
