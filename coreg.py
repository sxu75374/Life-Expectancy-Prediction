import numpy as np
from time import time
from scipy.spatial.distance import minkowski
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from data_utils import load_data


class Coreg():
    """
    Instantiates a CoReg regressor.
    """
    def __init__(self, k1=4, k2=4, p1=1, p2=2, max_iters=100, pool_size=80):
        self.k1, self.k2 = k1, k2 # number of neighbors
        self.p1, self.p2 = p1, p2 # distance metrics
        self.max_iters = max_iters
        self.pool_size = pool_size
        self.h1 = KNeighborsRegressor(n_neighbors=self.k1, p=self.p1)
        self.h2 = KNeighborsRegressor(n_neighbors=self.k2, p=self.p2)
        self.h1_temp = KNeighborsRegressor(n_neighbors=self.k1, p=self.p1)
        self.h2_temp = KNeighborsRegressor(n_neighbors=self.k2, p=self.p2)
        self.Plot_train1 = list()
        self.Plot_test1 = list()
        self.Plot_train2 = list()
        self.Plot_test2 = list()
        self.Plot_train = list()
        self.Plot_test = list()
        self.visual_predict = list()
        self.finalh1 = list()
        self.finalh2 = list()

    def add_data(self, X, y):
        """
        Adds data and splits into labeled and unlabeled.
        """
        self.X, self.y = load_data(X, y)

    def run_trials(self, num_train=732, trials=3, verbose=False):
        """
        Runs multiple trials of training.
        """
        self.num_train = num_train
        self.num_trials = trials
        self._initialize_metrics()
        self.trial = 0
        while self.trial < self.num_trials:
            t0 = time()
            print('Starting trial {}:'.format(self.trial + 1))            
            self.train(random_state=(self.trial+self.num_train),
                num_labeled=self.num_train, num_test=549, verbose=verbose,
                store_results=True)
            print('Finished trial {}: {:0.2f}s elapsed\n'.format(
                self.trial + 1, time() - t0))
            self.trial += 1

    def train(self, random_state=-1, num_labeled=1464, num_test=549,
        verbose=False, store_results=False):
        """
        Trains the CoReg regressor.
        """
        t0 = time()
        self._split_data(random_state, num_labeled, num_test)
        self._fit_and_evaluate(verbose)
        if store_results: self._store_results(0)
        self._get_pool()
        if verbose: print('Initialized h1, h2: {:0.2f}s\n'.format(time()-t0))
        for t in range(1, self.max_iters+1):
            stop_training = self._run_iteration(t, t0, verbose, store_results)
            if stop_training:
                if verbose:
                    print('Done in {} iterations: {:0.2f}s'.format(t, time()-t0))
                break
        if verbose: print('Finished {} iterations: {:0.2f}s'.format(t, time()-t0))

    def _run_iteration(self, t, t0, verbose=False, store_results=False):
        """
        Run t-th iteration of co-training, returns stop_training=True if
        no more unlabeled points are added to label sets.
        """
        stop_training = False
        if t == 100:
            self.finalh1 = self.h1
            print(mean_squared_error(self.y_test,self.h1.predict(self.X_test)))
            self.finalh2 = self.h2
            print(mean_squared_error(self.y_test, self.h2.predict(self.X_test)))
        if verbose: print('Started iteration {}: {:0.2f}s'.format(t, time()-t0))
        self._find_points_to_add()
        added = self._add_points()
        if added:
            self._fit_and_evaluate(verbose)
            if store_results:
                self._store_results(t)
            self._remove_from_unlabeled()
            self._get_pool()
        else:
            stop_training = True
        return stop_training

    def _add_points(self):
        """
        Adds new examples to training sets.
        """
        added = False
        if self.to_add['x1'] is not None:
            self.L2_X = np.vstack((self.L2_X, self.to_add['x1']))
            self.L2_y = np.vstack((self.L2_y, self.to_add['y1']))
            added = True
        if self.to_add['x2'] is not None:
            self.L1_X = np.vstack((self.L1_X, self.to_add['x2']))
            self.L1_y = np.vstack((self.L1_y, self.to_add['y2']))
            added = True
        return added

    def _compute_delta(self, omega, L_X, L_y, h, h_temp):
        """
        Computes the improvement in MSE among the neighbors of the point being
        evaluated.
        """
        delta = 0
        for idx_o in omega:
            delta += (L_y[idx_o].reshape(1, -1) -
                      h.predict(L_X[idx_o].reshape(1, -1))) ** 2
            delta -= (L_y[idx_o].reshape(1, -1) -
                      h_temp.predict(L_X[idx_o].reshape(1, -1))) ** 2
        return delta

    def _compute_deltas(self, L_X, L_y, h, h_temp):
        """
        Computes the improvements in local MSE for all points in pool.
        """
        deltas = np.zeros((self.U_X_pool.shape[0],))
        for idx_u, x_u in enumerate(self.U_X_pool):
            # Make prediction
            x_u = x_u.reshape(1, -1)
            y_u_hat = h.predict(x_u).reshape(1, -1)
            # Compute neighbors
            omega = h.kneighbors(x_u, return_distance=False)[0]
            # Retrain regressor after adding unlabeled point
            X_temp = np.vstack((L_X, x_u))
            y_temp = np.vstack((L_y, y_u_hat)) # use predicted y_u_hat
            h_temp.fit(X_temp, y_temp)
            delta = self._compute_delta(omega, L_X, L_y, h, h_temp)
            deltas[idx_u] = delta
        return deltas

    def _evaluate_metrics(self, verbose):
        """
        Evaluates KNN regressors on training and test data.
        """
        train1_hat = self.h1.predict(self.X_labeled)
        train2_hat = self.h2.predict(self.X_labeled)
        train_hat = 0.5 * (train1_hat + train2_hat)
        test1_hat = self.h1.predict(self.X_test)
        test2_hat = self.h2.predict(self.X_test)
        test_hat = 0.5 * (test1_hat + test2_hat)
        self.mse1_train = mean_squared_error(train1_hat, self.y_labeled)
        self.mse1_test = mean_squared_error(test1_hat, self.y_test)
        self.mse2_train = mean_squared_error(train2_hat, self.y_labeled)
        self.mse2_test = mean_squared_error(test2_hat, self.y_test)
        self.mse_train = mean_squared_error(train_hat, self.y_labeled)
        self.mse_test = mean_squared_error(test_hat, self.y_test)

        self.visual_predict.append(train_hat)
        self.Plot_train1.append(self.mse1_train)
        self.Plot_test1.append(self.mse1_test)
        self.Plot_train2.append(self.mse2_train)
        self.Plot_test2.append(self.mse2_test)
        self.Plot_train.append(self.mse_train)
        self.Plot_test.append(self.mse_test)

        if verbose:
            print('MSEs:')
            print('  KNN1:')
            print('    Train: {:0.4f}'.format(self.mse1_train))
            print('    Test : {:0.4f}'.format(self.mse1_test))
            print('  KNN2:')
            print('    Train: {:0.4f}'.format(self.mse2_train))
            print('    Test : {:0.4f}'.format(self.mse2_test))
            print('  Combined:')
            print('    Train: {:0.4f}'.format(self.mse_train))
            print('    Test : {:0.4f}\n'.format(self.mse_test))

    def _find_points_to_add(self):
        """
        Finds unlabeled points (if any) to add to training sets.
        """
        self.to_add = {'x1': None, 'y1': None, 'idx1': None,
                       'x2': None, 'y2': None, 'idx2': None}
        # Keep track of added idxs
        added_idxs = []
        for idx_h in [1, 2]:
            if idx_h == 1:
                h = self.h1
                h_temp = self.h1_temp
                L_X, L_y = self.L1_X, self.L1_y
            elif idx_h == 2:
                h = self.h2
                h_temp = self.h2_temp
                L_X, L_y = self.L2_X, self.L2_y
            deltas = self._compute_deltas(L_X, L_y, h, h_temp)
            # Add largest delta (improvement)
            sort_idxs = np.argsort(deltas)[::-1] # max to min
            max_idx = sort_idxs[0]
            if max_idx in added_idxs: max_idx = sort_idxs[1]
            if deltas[max_idx] > 0:
                added_idxs.append(max_idx)
                x_u = self.U_X_pool[max_idx].reshape(1, -1)
                y_u_hat = h.predict(x_u).reshape(1, -1)
                self.to_add['x' + str(idx_h)] = x_u
                self.to_add['y' + str(idx_h)] = y_u_hat
                self.to_add['idx' + str(idx_h)] = self.U_idx_pool[max_idx]

    def _fit_and_evaluate(self, verbose):
        """
        Fits h1 and h2 and evaluates metrics.
        """
        self.h1.fit(self.L1_X, self.L1_y)
        self.h2.fit(self.L2_X, self.L2_y)
        self._evaluate_metrics(verbose)

    def _get_pool(self):
        """
        Gets unlabeled pool and indices of unlabeled.
        """
        self.U_X_pool, self.U_y_pool, self.U_idx_pool = shuffle(
            self.U_X, self.U_y, range(self.U_y.size))
        self.U_X_pool = self.U_X_pool[:self.pool_size]
        self.U_y_pool = self.U_y_pool[:self.pool_size]
        self.U_idx_pool = self.U_idx_pool[:self.pool_size]

    def _initialize_metrics(self):
        """
        Sets up metrics to be stored.
        """
        initial_metrics = np.full((self.num_trials, self.max_iters+1), np.inf)
        self.mses1_train = np.copy(initial_metrics)
        self.mses1_test = np.copy(initial_metrics)
        self.mses2_train = np.copy(initial_metrics)
        self.mses2_test = np.copy(initial_metrics)
        self.mses_train = np.copy(initial_metrics)
        self.mses_test = np.copy(initial_metrics)

    def _remove_from_unlabeled(self):
        # Remove added examples from unlabeled
        to_remove = []
        if self.to_add['idx1'] is not None:
            to_remove.append(self.to_add['idx1'])
        if self.to_add['idx2'] is not None:
            to_remove.append(self.to_add['idx2'])
        self.U_X = np.delete(self.U_X, to_remove, axis=0)
        self.U_y = np.delete(self.U_y, to_remove, axis=0)

    def _split_data(self, random_state=-1, num_labeled=1464, num_test=549):
        """
        Shuffles data and splits it into train, test, and unlabeled sets.
        """
        if random_state >= 0:
            self.X_shuffled, self.y_shuffled, self.shuffled_indices = shuffle(
                self.X, self.y, range(self.y.size), random_state=random_state)
        else:
            self.X_shuffled = self.X[:]
            self.y_shuffled = self.y[:]
            self.shuffled_indices = range(self.y.size)
        # Initial labeled, test, and unlabeled sets
        test_end = num_labeled + num_test
        self.X_labeled = self.X_shuffled[:num_labeled]
        self.y_labeled = self.y_shuffled[:num_labeled]
        self.X_test = self.X_shuffled[num_labeled:test_end]
        print('Xall', self.X_shuffled)
        print('Xtest ', self.X_test)
        self.y_test = self.y_shuffled[num_labeled:test_end]
        self.X_unlabeled = self.X_shuffled[test_end:]
        self.y_unlabeled = self.y_shuffled[test_end:]
        # Up-to-date training sets and unlabeled set
        self.L1_X = self.X_labeled[:]
        self.L1_y = self.y_labeled[:]
        self.L2_X = self.X_labeled[:]
        self.L2_y = self.y_labeled[:]
        self.U_X = self.X_unlabeled[:]
        self.U_y = self.y_unlabeled[:]

    def _store_results(self, iteration):
        """
        Stores current MSEs.
        """

        self.mses1_train[self.trial,iteration] = self.mse1_train
        self.mses1_test[self.trial,iteration] = self.mse1_test
        self.mses2_train[self.trial,iteration] = self.mse2_train
        self.mses2_test[self.trial,iteration] = self.mse2_test
        self.mses_train[self.trial,iteration] = self.mse_train
        self.mses_test[self.trial,iteration] = self.mse_test

    def plot_mse(self):
        import matplotlib.pyplot as plt
        xrange = np.arange(101)+1
        plt.figure()
        plt.plot(xrange, self.Plot_train1, color='blue', linestyle='dotted', label='train MSE of KNN regression model 1')
        plt.plot(xrange, self.Plot_train2, color='blue', linestyle='dashed', label='train MSE of KNN regression model 2')
        plt.plot(xrange, self.Plot_train, color='blue', label='train MSE of KNN Co-training regression')
        plt.scatter(np.argmin(self.Plot_train) + 1, self.Plot_train[np.argmin(self.Plot_train)], marker='o',
                    label='best co-training train MSE score')

        plt.plot(xrange, self.Plot_test1, color='orange', linestyle='dotted', label='test MSE of KNN regression model 1')
        plt.plot(xrange, self.Plot_test2, color='orange', linestyle='dashed', label='test MSE of KNN regression model 2')
        plt.plot(xrange, self.Plot_test, color='orange', label='test MSE of KNN Co-training regression')
        plt.scatter(np.argmin(self.Plot_test)+1, self.Plot_test[np.argmin(self.Plot_test)], marker='o', color='orange',
                    label='best co-training test MSE score')
        plt.title('Iterations vs train and test MSE of KNN model 1, model 2 and Co-training regression')
        plt.xlabel('Iterations')
        plt.ylabel('MSE score')
        plt.legend()
        plt.show()
        self.finalh1 = self.h1
        print(mean_squared_error(self.y_test, self.h1.predict(self.X_test)))
        self.finalh2 = self.h2
        print(mean_squared_error(self.y_test, self.h2.predict(self.X_test)))
        print(self.X_test)
        return self.Plot_train1, self.Plot_train2, self.Plot_train, self.Plot_train[np.argmin(self.Plot_train)], \
               self.Plot_test1, self.Plot_test2, self.Plot_test, self.Plot_test[np.argmin(self.Plot_test)], self.finalh1,\
               self.finalh2

    def visualization(self, LA1, LA2, Xlabel, Ylabel, Title):
        import matplotlib.pyplot as plt
        x_min_cart2 = np.min(self.X)
        x_max_cart2 = np.max(self.X)
        xrange_cart2 = np.arange(x_min_cart2, x_max_cart2, 1)
        plt.figure()
        k=(self.h1.predict(xrange_cart2.reshape(-1,1)) + self.h2.predict(xrange_cart2.reshape(-1,1)))/2
        plt.scatter(self.X, self.y, label=LA1)
        plt.plot(xrange_cart2, k, color='red', label=LA2)
        plt.xlabel(Xlabel)
        plt.ylabel(Ylabel)
        plt.title(Title)
        plt.legend()
        plt.show()