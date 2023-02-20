import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import pickle
from scipy.linalg import block_diag
import networkx as nx

import sys, os
sys.path.append(os.getcwd() + "/../..")
import matplotlib.pyplot as plt
from src.estimators import *
from src.cross_validation import naive_cv
from simulations.examples import SmoothStair
from simulations.sample_data import gaussian_sample
import networkx as nx
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


parser = argparse.ArgumentParser()
parser.add_argument('--state', type=str, default="")
parser.add_argument('--county', type=str, default="")
parser.add_argument('--var', type=str, default="cases")
parser.add_argument('--shuffle', type=str, default="False")
parser.add_argument('--temporal_reg', type=int, default=1)
parser.add_argument('--dim', type=int, default=2)
parser.add_argument('--ego_net_size', type=int, default=2)
parser.add_argument('--predict', type=str, default="")
parser.add_argument('--free_beta', type=int, default=0)
args = parser.parse_args()


# cols = ['diff_TM21', 'diff_TM20', 'diff_TM19', 'diff_TM18',
#         'diff_TM17', 'diff_TM16', 'diff_TM15', 'TM14']
cols = ['TM21', 'TM20', 'TM19', 'TM18',
                'TM17', 'TM16', 'TM15', 'TM14']
if args.dim == 3:
    cols = ['TM21', 'TM18', 'TM14']

dd   = args.county + "_tiny_egonet_subset_" + str(args.ego_net_size) + "_" + args.var + "_dim"+ str(len(cols))
if args.shuffle == "True":
    shuffle = True
    dd += "shuffle"
else:
    shuffle = False
    dd += "no_shuffle"
if args.temporal_reg == 1:
    dd = dd + "_temp_reg"
if args.free_beta == 1:
    dd = dd + "_free_beta"



print(args.temporal_reg, dd)
file_name = "/Users/cdonnat/Dropbox/network_regularisation/data/COVID/county_adjacency.txt"
file1 = open(file_name, "r")
str1=file1.read()
file1.close()
all_lines= str1.split("\n")
if args.var == "death":
    variable = '_log_deaths'
    dd = dd + '_log_deaths'
else:
    variable = "_log_cases"
    dd = dd + '_log_cases'

county_adjacency = {}
county_names = {}
for k, line in enumerate(all_lines):
    if len(line)>0:
        s = line.replace('"', '').split("\t")
        if s[3] not in county_names.keys():
            county_names[s[3]] = s[2]
        if len(s[0]) > 0 :
            if s[1] not in county_names.keys():
                county_names[s[1]] = s[0]
            prev_county = s[0]
            county_adjacency[prev_county] = [s[2]]
        else:
            county_adjacency[prev_county] += [s[2]]

county_id = {county_names[v].replace('"', ''): v for k, v in enumerate(county_names)}


all_states = np.unique( [s.split(", ")[1] for s in county_id.keys()])
counties_per_state = {}
adjacencies = {}


state = args.state
if True:

    counties_per_state[state] = [k for k in county_id.keys() if k.split(", ")[1] == state]
    adjacencies[state] = pd.DataFrame(np.zeros((len(counties_per_state[state]), len(counties_per_state[state]))),
                                      index=counties_per_state[state],
                                      columns=counties_per_state[state])
    for c in counties_per_state[state]:
        try:
            for cc in county_adjacency[c]:
                adjacencies[state].loc[c][cc] = 1.
        except:
            print(c)
    if (adjacencies[state]==adjacencies[state].T).mean().mean() != 1:
        adjacencies[state] = adjacencies[state] + adjacencies[state].T
        adjacencies[state].values[adjacencies[state].values>1] =1
    print(state, (adjacencies[state]==adjacencies[state].T).mean().mean())


state =args.state
predictions = {}
predictions_test = {}
fitted_clfs = {}
test_metrics = {}
train_metrics = {}
full_metrics = {}
r2 = {}
r2_full = {}
GRID1_SMALL = {'l1': [ 0.01, 0.05, 0.1, 0.5, 1, 2, 10, 15, 20],
               'l2': [0.01, 0.05, 0.1, 0.5, 1, 2, 10, 15, 20]}


if True:
    #X_train = pd.read_csv("/Users/cdonnat/Downloads/new_train_data" +state  +".csv").fillna(0)
    #X_test = pd.read_csv("/Users/cdonnat/Downloads/new_test_data" +state + ".csv").fillna(0)
    X_full_data = pd.read_csv("/Users/cdonnat/Downloads/fulldata" +state + ".csv").fillna(0)
    ###### Bootstrap the time series:
    index_data = range(X_full_data.shape[0])
    ordered_counties= np.sort(adjacencies[state].columns)
    #print(ordered_counties)
    list_counties = [u.replace('Municipality', '').replace('County', '').split(', '+ str(state))[0].strip() for u in ordered_counties ]
    #print(list_counties)
    # G = block_diag(*[adjacencies[state][ordered_counties].loc[ordered_counties].values] * 6)
    # for i in range(len(ordered_counties)):
    #         for t in range(1, 5):
    #             G[i + len(ordered_counties) * t, i + len(ordered_counties) * (t+1)] = 1
    #             G[i + len(ordered_counties) * t, i + len(ordered_counties) * (t-1)] = 1
    #             G[i + len(ordered_counties) * (t+1), i + len(ordered_counties) * t] = 1
    #             G[i + len(ordered_counties) * (t-1), i + len(ordered_counties) * t] = 1
    #         for t in range(2, 4):
    #             G[i + len(ordered_counties) * t, i + len(ordered_counties) * (t+2)] = 1
    #             G[i + len(ordered_counties) * t, i + len(ordered_counties) * (t-2)] = 1
    #             G[i + len(ordered_counties) * (t+2), i + len(ordered_counties) * t] = 1
    #             G[i + len(ordered_counties) * (t-2), i + len(ordered_counties) * t] = 1
    # D = nx.incidence_matrix(nx.from_numpy_matrix(G), oriented=True).todense().T
    # D = np.hstack([np.zeros((D.shape[0],1)), D])

    # D = nx.incidence_matrix(nx.from_numpy_matrix(G), oriented=True).todense().T
    # D = np.hstack([np.zeros((D.shape[0],1)), D])
    #D = np.hstack([D, np.zeros((D.shape[0],2))]) ## for the last two columns
    #for county in [args.county]:#list_counties:
    for county in ["San Francisco", "Santa Clara", "San Mateo", "Santa Cruz",
    "Alameda", "Contra Costa", "San Benito", "Solano", "Napa", "Sonoma", "Los Angeles",
    "San Diego", "Riverside", "Marin", "Placer", "Stanislaus", "Yolo", "Fresno", "Sutter",
    "Merced", "Butte", "Santa Barbara", "Yuba", "Kings", "Kern",
     "Sacramento", "Ventura", "Orange", "Riverside", "San Joaquin"]:#[args.county]: #list_counties:
        hat_beta = {}
        i = np.where(np.array(list_counties)==county)[0][0]
        print("here i:", i )

        graph = nx.ego_graph(nx.from_pandas_adjacency(adjacencies[state][ordered_counties].loc[ordered_counties]),
                           n=ordered_counties[i], radius=args.ego_net_size)
        print("the egonet includes:", list(graph.nodes()))
        colnames = [u.replace('Municipality', '').replace('County', '').split(', '+ str(state))[0].strip() +'_' + v
                    for v in cols
                    for u in list(graph.nodes()) ]
                    #if u != ordered_counties[i]] #+ [county + '_TM21_sq', county + '_TM28_sq' ]
        #x_train = np.hstack([np.ones((X_train.shape[0],1)), X_train[colnames].values])
        # x_train = X_train[colnames].values
        # y_train = X_train[str(county) + variable]
        # mu = y_train.mean()
        # #x_test = np.hstack([np.ones((X_test.shape[0],1)), X_test[colnames].values])
        # x_test = X_test[colnames].values
        # y_test = X_test[str(county) + variable]
        # y_test = y_test - mu
        #
        # #x_full = np.hstack([np.ones((X_full_data.shape[0],1)), X_full_data[colnames].values])
        # x_full =  X_full_data[colnames].values
        # x_test = X_test[colnames].values
        # x_train = np.vstack([x_train, x_test])
        # y_train = np.concatenate([y_train, y_test])
        # print("Size of dataset: ", x_train.shape)
        # y_full= X_full_data[str(county) + variable]
        print(colnames)
        print("And we're predicting: ", str(county) + variable)
        # x_train = X_train[colnames].values
        # #x_train = np.hstack([np.ones((X_train.shape[0],1)), X_train[colnames].values])
        # y_train = X_train[str(county) + variable].values
        # mu = y_train.mean()
        # #y_train = y_train #- mu
        # #x_test = np.hstack([np.ones((X_test.shape[0],1)), X_test[colnames].values])
        # x_test = X_test[colnames].values
        # y_test = X_test[str(county) + variable].values
        #y_test = y_test #- mu
        #x_train = np.vstack([x_train, x_test])
        # #y_train = np.concatenate([y_train, y_test])
        # print("X train size:", x_train.shape)
        # print("X testsize:", x_test.shape)


        x_full = np.hstack([np.ones((X_full_data.shape[0],1)), X_full_data[colnames].values])
        #x_full =  X_full_data[colnames].values
        print(X_full_data[colnames].head())
        #x_test = X_test[colnames].values
        y_full = X_full_data[str(county) +variable].values


        GG = nx.adjacency_matrix(graph).todense()
        print("list nodes")
        print(np.array(list(graph.nodes())))
        index= np.where(np.array(list(graph.nodes())) == ordered_counties[i])[0][0]
        print("making sure we have a match", index, list(graph.nodes())[index])
        # GG = np.delete(GG, index, 0)
        # GG = np.delete(GG, index, 1)
        if args.free_beta == 1:
             GG[index, :] = 0
             GG[:, index] = 0

        G = block_diag(*[GG]*len(cols))
        length_block = GG.shape[0]
        # #print(GG.shape, len(cols), G.shape)
        # print("yooo")
        if args.temporal_reg == 1 and args.ego_net_size >1:
            for u in range(len(list(graph.nodes()))):
                for t in range(1, len(cols)-1):
                    G[u + length_block * t, u + length_block  * (t+1)] = 1
                    G[u + length_block * t, u +length_block  * (t-1)] = 1
                    G[u + length_block * (t+1), u + length_block  * t] = 1
                    G[u + length_block  * (t-1), u + length_block  * t] = 1

        #### Delete the dependency of the county_id


        # for i in range(len(ordered_counties)):
        #         for t in range(1, 5):
        #             G[i + len(ordered_counties) * t, i + len(ordered_counties) * (t+1)] = 1
        #             G[i + len(ordered_counties) * t, i + len(ordered_counties) * (t-1)] = 1
        #             G[i + len(ordered_counties) * (t+1), i + len(ordered_counties) * t] = 1
        #             G[i + len(ordered_counties) * (t-1), i + len(ordered_counties) * t] = 1
        #         G[i + len(ordered_counties) * 1, i + len(ordered_counties) * 3 ] = 1
        #         G[i + len(ordered_counties) * 2, i + len(ordered_counties) * 0] = 1
        #         G[i + len(ordered_counties) * 3, i + len(ordered_counties) * 1] = 1
        #         G[i + len(ordered_counties) * 0, i + len(ordered_counties) * 2] = 1
        # for i in range(len(ordered_counties)):
        #         for t in range(1, 3):
        #             G[i + len(ordered_counties) * t, i + len(ordered_counties) * (t+1)] = 1
        #             G[i + len(ordered_counties) * t, i + len(ordered_counties) * (t-1)] = 1
        #             G[i + len(ordered_counties) * (t+1), i + len(ordered_counties) * t] = 1
        #             G[i + len(ordered_counties) * (t-1), i + len(ordered_counties) * t] = 1
                # G[i + len(ordered_counties) * 1, i + len(ordered_counties) * 3 ] = 1
                # G[i + len(ordered_counties) * 2, i + len(ordered_counties) * 0] = 1
                # G[i + len(ordered_counties) * 3, i + len(ordered_counties) * 1] = 1
                # G[i + len(ordered_counties) * 0, i + len(ordered_counties) * 2] = 1
        # G[np.where(ordered_counties == county), :] = 0
        # G[:, np.where(ordered_counties == county)] = 0
        D = nx.incidence_matrix(nx.from_numpy_matrix(G), oriented=True).todense().T
        D = np.hstack([np.zeros((D.shape[0],1)), D])
        print("Size of D: ", D.shape)



        #hat_beta[(state, county, exp)] = {}
        #predictions[(state, county, exp)] = {}
        index4training = np.arange(0, x_full.shape[0], 7)
        kf = KFold(n_splits=7)
        exp = 0
        for ind_train_index, ind_test_index in kf.split(index4training):
            train_index = index4training[ind_train_index]
            test_index = index4training[ind_test_index]
            print("test_index", test_index)
            #fitted_clfs[(state, county, exp)] = {}
            test_metrics[(state, county, exp)] = {}
            train_metrics[(state, county, exp)] = {}
            full_metrics[(state, county, exp)] = {}
            r2[(state, county, exp)] = {}
            r2_full[(state, county, exp)] = {}
            #np.random.shuffle(blocks)
            new_x_full = x_full #np.concatenate([x_full[blocks[i]:(blocks[i]+27),:] for i in range(len(blocks))])
            new_y_full = y_full #np.concatenate([y_full[blocks[i]:(blocks[i]+27)] for i in range(len(blocks))])
            x_train = new_x_full[train_index, :]
            y_train = new_y_full[train_index]
            #mu = y_train.mean()
            #y_train = y_train
            x_test = new_x_full[test_index,:]
            y_test = new_y_full[test_index]
            print("Checking  train shapes", x_train.shape, x_test.shape)

            print("x_train shape", x_train.shape)
            for estimator in ['no_neighbours_naive', 'SL', 'FL', 'EN', 'GenEN', 'naive', 'lasso'] :
                print(state, county, estimator)
                #try:
                if True:
                    if estimator == 'no_neighbours_naive':
                        colnames_county = [u.replace('Municipality', '').replace('County', '').split(', '+ str(state))[0].strip() +'_' + v
                                    for v in cols
                                    for u in [county] ] #+ [county + '_TM21_sq', county + '_TM28_sq' ]
                        # x_train_no_neighbour =  X_train[colnames_county].values
                        # x_test_no_neighbour  =  X_test[colnames_county].values
                        #x_train_no_neighbour = np.vstack([x_train_no_neighbour, x_test_no_neighbour])
                        #x_full_no_neighbour  = X_full_data[colnames_county].values
                        #x_train_no_neighbour =  X_train[colnames_county].values
                        #x_test_no_neighbour  =  X_test[colnames_county].values
                        # x_train_no_neighbour = np.hstack([np.ones((X_train.shape[0],1)),
                        #                                   X_train[colnames_county].values])
                        # x_test_no_neighbour = np.hstack([np.ones((X_test.shape[0],1)),
                        #                                  X_test[colnames_county].values])
                        #x_train_no_neighbour = np.vstack([x_train_no_neighbour, x_test_no_neighbour])
                        x_full_no_neighbour  = np.hstack([np.ones((X_full_data.shape[0],1)),
                                                          X_full_data[colnames_county].values])
                        #new_x_full_no_neighbour = np.concatenate([x_full_no_neighbour[blocks[i]:(blocks[i]+27),:] for i in range(len(blocks))])
                        new_x_full_no_neighbour = x_full_no_neighbour
                        print("Checking shapes", new_x_full_no_neighbour.shape, new_x_full.shape, new_y_full.shape)
                        x_train_no_neighbour = new_x_full_no_neighbour[train_index,:]
                        x_test_no_neighbour = new_x_full_no_neighbour[test_index,:]

                        res = naive_cv(NaiveEstimator, x_train_no_neighbour, y_train, D, grid=GRID1_SMALL)
                        clf = NaiveEstimator(l1=res[0]['l1'], l2=res[0]['l2'],
                                                             D =D, family='normal')
                    a, b = 0, 0

                    if estimator == 'naive':
                        res = naive_cv(NaiveEstimator, x_train, y_train, D, shuffle=shuffle, grid=GRID1_SMALL)
                        clf = NaiveEstimator(l1=res[0]['l1'], l2=res[0]['l2'],
                                                 D =D, family='normal')
                        a, b = res[0]['l1'], res[0]['l2']
                    elif estimator == 'lasso':
                        res = naive_cv(LassoEstimator, x_train, y_train, D, shuffle=shuffle, grid=GRID1_SMALL)
                        clf = LassoEstimator(l1=res[0]['l1'], l2=res[0]['l2'],
                                                 D =D, family='normal')
                        a, b = res[0]['l1'], res[0]['l2']
                    elif estimator == 'SL':
                        res = naive_cv(SmoothedLassoEstimator,  x_train, y_train, D,shuffle=shuffle, grid=GRID1_SMALL )
                        clf = SmoothedLassoEstimator(l1=res[0]['l1'], l2=res[0]['l2'],
                                                    D =D, family='normal')
                        a, b = res[0]['l1'], res[0]['l2']
                    elif estimator == 'FL':
                        res = naive_cv(FusedLassoEstimator, x_train,  y_train, D, shuffle=shuffle, grid=GRID1_SMALL )
                        clf = FusedLassoEstimator(l1=res[0]['l1'], l2=res[0]['l2'],
                                                    D =D, family='normal')
                        a, b = res[0]['l1'], res[0]['l2']
                    elif estimator == 'EN':
                        res = naive_cv(ElasticNetEstimator, x_train,  y_train, D, shuffle=shuffle, grid=GRID1_SMALL )
                        clf = ElasticNetEstimator(l1=res[0]['l1'], l2=res[0]['l2'],
                                                    D =D, family='normal')
                        a, b = res[0]['l1'], res[0]['l2']
                    elif estimator == 'GenEN':
                        res = naive_cv(GenElasticNetEstimator,  x_train,  y_train, D,shuffle=shuffle, grid=GRID1_SMALL
                                      )
                        clf = GenElasticNetEstimator(l1=res[0]['l1'], l2=res[0]['l2'],
                                                    D =D, family='normal')
                        a, b = res[0]['l1'], res[0]['l2']
                    elif estimator == 'GenEN_cgd':
                        res = naive_cv(GenElasticNetEstimator, x_train,  y_train, D, shuffle=shuffle,
                                       grid=GRID1_SMALL, solver='cgd')
                        clf = GenElasticNetEstimator(l1=res[0]['l1'], l2=res[0]['l2'],
                                                    D =D, family='normal', solver='cgd')
                        a, b = res[0]['l1'], res[0]['l2']

                    if estimator !="no_neighbours_naive":
                        try:
                            print(r2[(state, county, exp)])
                            clf.fit(x_train, y_train, 500)
                            y_hat_residuals_pred_test = clf.predict(x_test)
                            y_hat_residuals_pred_train = clf.predict(x_train)
                            y_hat_residuals_pred_full = clf.predict(new_x_full)
                            hat_beta[(state, county, exp, estimator)] = clf.beta
                            #test_metrics[(state, county, exp)][estimator]  = mean_squared_error(y_test, clf.predict(x_test))
                            test_metrics[(state, county, exp)][estimator] = mean_squared_error(y_test, clf.predict(x_test))
                            r2[(state, county, exp)][estimator]  = r2_score(y_train, clf.predict(x_train)) #1 - np.sum(np.square(y_tt - y_hat_residuals_pred_test))/ np.sum(np.square(y_test - y_test.mean()))

                            train_metrics[(state, county, exp)][estimator] = mean_squared_error(y_train, clf.predict(x_train))
                            full_metrics[(state, county, exp)][estimator] = mean_squared_error(new_y_full, y_hat_residuals_pred_full)
                            predictions[(state, county, exp, estimator)] = y_hat_residuals_pred_full
                            predictions_test [(state, county, exp, estimator)] = y_hat_residuals_pred_test
                            r2_full[(state, county, exp)][estimator] = 1 - np.sum(np.square(new_y_full - y_hat_residuals_pred_full))/ np.sum(np.square(new_y_full - y_full.mean()))
                            # #r2_score(y_full, y_hat_residuals_pred_full)
                        except:
                            pass
                        # plt.figure()
                        # plt.plot(range(len(y_hat_residuals_pred_full)), y_hat_residuals_pred_full, c="red")
                        # plt.plot(range(len(y_hat_residuals_pred_full)), new_y_full, c="black")
                        # plt.show()
                        # #
                        # plt.figure()
                        # plt.scatter(residuals_full, clf.predict(x_full), c="blue")
                        # plt.plot(residuals_full, residuals_full, c="red")
                        # plt.show()

                    else:
                        clf.fit(x_train_no_neighbour, y_train, 500)
                        #hat_beta[(state, county, estimator)] = clf.beta
                        test_metrics[(state, county, exp)][estimator] = mean_squared_error(y_test, clf.predict(x_test_no_neighbour))
                        r2[(state, county, exp)][estimator] = r2_score(y_train, clf.predict(x_train_no_neighbour))
                        train_metrics[(state, county, exp)][estimator] = mean_squared_error(y_train, clf.predict(x_train_no_neighbour))
                        y_hat  = clf.predict(new_x_full_no_neighbour)
                        y_hat_train  = clf.predict(x_train_no_neighbour)
                        y_hat_test  = clf.predict(x_test_no_neighbour)
                        full_metrics[(state, county,estimator)] = mean_squared_error(new_y_full, y_hat)
                        r2_full[(state, county, exp)][estimator] = r2_score(new_y_full, y_hat)
                        print("heya", r2[(state, county, exp)])

                        # residuals_full = y_full - y_hat
                        # residuals_train = y_train - y_hat_train
                        # residuals_test = y_test - y_hat_test
                        # predictions[(state, county, exp, estimator)] = y_hat
                        # predictions_test[(state, county, exp, estimator)] = y_hat_test
                        # plt.figure()
                        # plt.plot(range(len(y_hat)), y_hat, c="red")
                        # plt.plot(range(len(new_y_full)), new_y_full, c="black")
                        # plt.show()
                        #
                        # plt.figure()
                        # plt.plot(X_full_data["date"], residuals_full, c="red")
                        # plt.show()

                        #r2_full[(state, county, exp)][estimator] = r2_score(y_full, y_hat)
                    #print("Size of hat_beta", hat_beta.shape)
                    print(r2[(state, county, exp)][estimator])


                    print(estimator, a, b)

                    with open("results4/" + state + dd + 'clfs.pickle', 'wb') as handle:
                        pickle.dump(fitted_clfs, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    pd.DataFrame.from_dict(test_metrics).T.to_csv("results4/" + state + dd + "test_metrics.csv")
                    pd.DataFrame.from_dict(train_metrics).T.to_csv("results4/" + state +  dd + "train_metrics.csv")
                    pd.DataFrame.from_dict(full_metrics).T.to_csv("results4/" + state +  dd + "full_metrics.csv")
                    pd.DataFrame.from_dict(hat_beta).T.to_csv("results4/" + state +  dd + county + "hat_beta.csv")
                    pd.DataFrame.from_dict(predictions).T.to_csv("results4/" + state +  dd +  "predictions.csv")
                    pd.DataFrame.from_dict(r2).T.to_csv("results4/" + state +  dd + "r2.csv")
                    pd.DataFrame.from_dict(r2_full).T.to_csv("results4/" + state +  dd + "r2_full.csv")
            exp +=1
                # except:
                #     pass
