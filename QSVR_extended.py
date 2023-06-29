#%%
# -*- coding: utf-8 -*-
"""
@author: Edoardo Pasetto
@edited by: Tómas Laufdal, Þorsteinn Elí, Þorsteinn Ingólfsson

"""

import os
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


import utility as utility

import importlib
importlib.reload(utility)

from datetime import datetime
from pathlib import Path
import logging
from dwave.system import LeapHybridSampler
from dimod import ConstrainedQuadraticModel

from dwave.system import LeapHybridCQMSampler


CURRENT_DIR = Path(__file__).parent
DATASET_DIR = CURRENT_DIR / "Datasets"
RESULTS_DIR = CURRENT_DIR / "Results"

#logging.basicConfig(filename=CURRENT_DIR/"debug.log", encoding="utf-8", level=logging.DEBUG)
logging.basicConfig(filename=CURRENT_DIR/"debug.log", level=logging.DEBUG)


class QSVR():

    def __init__(self, reuse_sampler: bool = True, sampler = None) -> None:
        
        self.sampler = sampler
        self.reuse_sampler = reuse_sampler
        self.hybridsampler = None
        pass

    def reset_sampler(self) -> None:
        self.sampler = None

    def set_reuse_sampler(self, reuse_sampler: bool) -> None:
        self.reuse_sampler = reuse_sampler
    
  
    def goodParams(self, C: float = None, gamma: float = None) -> None:
        """
        Calculates the best C and gamma by optimizing these hyperparameters with classical SVR on a reduced dataset.
        Splits the validation set in half in to a train/test split. The hyperparameters are selected from the SVR model that has the lowest MSE.
        """
        
        ##################################
        # Data values are changed to the logarithmic domain
        if self.change_to_logarithmic:
            self.dataset_train_X = np.log(self.dataset_train_X)
            self.dataset_train_Y = np.log(self.dataset_train_Y)
        ##############################

        # change the random seed to change how the dataseta are randomly chosen

        exponents_array = np.array(list(range(self.K)))
        exponents_array = exponents_array-self.k0

        # C_max is the minimum value that C can take
        C_max = sum(np.power(self.B, exponents_array))

        # Increase C_max in case of K=1 which often resulted in errors in estimation of parameter b,
        # as no alphas were in the range 0<aplpha<C.
        if self.K == 1:
            C_max += 1

        # The arrays gamma_values and C_values are used for the hyperparameters validation
        if C is None:   C_values = [C_max, 2*C_max, 5*C_max, 10*C_max, 20*C_max, 50*C_max]
        else:           C_values = [C]
        
        if gamma is None:   gamma_values = [0.1, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 20, 50]
        else:               gamma_values = [gamma]

        # If you don't want to optimize the value of C and gamma or if you want to optimize it outside of QSVR
        # You can select gamma and C in the fit function and set n_samples_for_gamma_and_C_optimization as 0.
        if self.n_samples_for_gamma_and_C_optimizations == 0:
            if gamma is None or C is None: 
                raise ValueError("Gamma and C values must be set if validation set is of size 0")
            
            self.X_train = self.dataset_train_X
            self.Y_train = self.dataset_train_Y
            self.best_gamma = gamma
            self.best_C = C
            return

        # The train and test dataset are selected with the built-in function
        # it is posible to vary the dataset generation process by modyfing the random seed generator
        self.X_train, X_validation_pool, self.Y_train, Y_validation_pool = train_test_split(
            self.dataset_train_X, self.dataset_train_Y, train_size=self.N, random_state=self.random_seed)

        
        # from the test set 100 samples are selected that are then used for finding the best hyperparameters (gamma and epsilon) for the SVR
        # the hyperparameters of the SVR are also used for its quantum implementation
        X_validation, _, Y_validation, _ = train_test_split(
            X_validation_pool, Y_validation_pool, train_size=self.n_samples_for_gamma_and_C_optimizations, random_state=self.random_seed)

        # the samples for validation are then divided into two datasets that are then used for the hyperparameter optimization
        # for each combination of values for gamma and C the SVR is trained using those hyperparameters and are then tested on the X_val_test dataset
        # the hyperparameters combination that achieves the best MSE is chosen

        X_val, X_val_test, Y_val, Y_val_test = train_test_split(
            X_validation, Y_validation, test_size=0.5, random_state=self.random_seed)

        # the hyperparameter matrix stores the MSE values for all the combination of gamma and C parameters
        hyperparameter_matrix, self.best_gamma, self.best_C, _ = utility.hyperparameters_validation(
            X_val, Y_val, X_val_test, Y_val_test, gamma_values, C_values, [self.epsilon])

        

    def fit(self,
            dataset_train_X,
            dataset_train_Y,
            epsilon: float,
            K: int,
            B: float,
            n_samples: int = 30,
            beta: float = 0,
            xi: float = 0,
            k0: float = 0,
            C: float = None,
            gamma: float = None, 
            num_reads: int = 2500,
            anneal_time: float = 20,
            n_samples_for_gamma_and_C_optimizations: int = 100,
            sampler = None,
            change_to_logarithmic: bool = False,
            random_seed = 10,
            use_custom_chainstrength: bool = False,
            chain_mult: float = 1,
            result_percentage_used: float = 0.004,
            hybrid: bool = False,
            only_hybrid: bool = False,
            hybrid_time = 4
            ):
        """
        Fits the alpha values to the given dataset for 7 several scoring algorithms.

        Utilizes n_samples_for_gamma_and_C_optimizations samples from dataset to optimize some hyperparameters classically.
        Additionally utilizes n_samples samples from dataset to train the alpha values on a Quantum Annealer.
        These splits are determined from the random_seed.

        Parameters:
        -----------
            dataset_train_X: 
                    Training dataset of at least size n_samples + n_samples_for_gamma_and_C_optimization.
            dataset_train_Y: 
                    Training target dataset corresponding to dataset_train_X.
            n_samples : `int`
                    Number of samples to incorporate into the QUBO.
            n_samples_for_gamma_and_C_optimizations: `int`
                    Number of samples to extract from the training set to optimize `γ` & `C` classically before introduction to quantum annealing.
            num_reads : `int`
                    Number of reads on the Quantum Annealer (default: 2500) (range: [1, 10000])
            sampler:
                    Either `None` or EmbeddingComposite of a DwaveSampler which can reuse an embedding. ???
            anneal_time : `float`
                    Actual annealing time for each sample on the QA in µs. Resolution is 0.01µs. (default: 20) (range: [0.5, 2000])
            change_to_logarithmic : `bool`
                    Boolean representing if the logarithm of the data should be used to fit the model and consequently respected when predicting/scoring. 
            random_seed : `int` or None
                    Seed to be used for training data splits when optimizing for C & γ and when extracting n_samples for QUBO.
                    Use `None` for random splits (default: 10).
            use_custom_chainstrength : `bool`
                    Whether or not to use the custom integrated chain strength function in utility.dwave_run.
                    If `True`, the sampler will not use the default chain strength, otherwise calculated with dwave.embedding.chain_strength.uniform_torque_compensation().
                    (default: False)
            chain_mult : `float`
                    The chain strength multiplier to use if `use_custom_chainstrength` is set to `True`. (default: 1)
            result_percentage_used : `float`
                    Proportion of resulting solutions from the quantum annealer to compound into a single solution. (default: 0.004)
            hybrid : `bool`
                    Whether or not to also train on a quantum hybrid machine. (default: False)
            only_hybrid : `bool`
                    Whether or not to only train on a quantum hybrid machine. Will skip all other quantum approaches except hybrid (specified by the `hybrid` variable). (default: False)
            hybrid_time : `float`
                    Time limit in seconds for the hybrid sampler to reach an optimal solution. Must be larger than the minimum time required to solve the hybrid problem. (default: 3)

        Hyperparameters:
        ----------------
            C : `float`
                    Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
            gamma : `float`
                    Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. 
            epsilon : `float`
                    Hyperparameter, margin of tolerance.
            K : `int`
                    Hyperparameter, size of kernel.
            B : `float`
                    Hyperparameter controlling the construction of the exponential vector ???
            beta : `float`
                    Hyperparameter, --INSERT DESCRIPTION HERE--
            xi : `float`
                    Hyperparameter, --INSERT DESCRIPTION HERE--
            k0 : `float`
                    Hyperparameter, compliments B in construction of exponential vector ???

        """
        if hasattr(self, "N") and hasattr(self, "K"):
            can_use_same_embedding = self.N * self.K == n_samples * K
        else:
            can_use_same_embedding = True
        self.N = n_samples
        self.B = B
        self.K = K
        self.xi = xi
        self.k0 = k0
        self.beta = beta
        self.epsilon = epsilon
        self.num_reads = num_reads
        self.change_to_logarithmic = change_to_logarithmic  # If training should happen on a logarithm of the data.
        self.n_samples_for_gamma_and_C_optimizations = n_samples_for_gamma_and_C_optimizations
        self.random_seed = random_seed  # None for "true" random
        self.hybrid = hybrid
        self.only_hybrid = only_hybrid

        self.dataset_train_X = dataset_train_X
        self.dataset_train_Y = dataset_train_Y
        
        self.X_train = dataset_train_X
        self.Y_train = dataset_train_Y

        self.best_gamma = gamma
        self.best_C = C
        #self.goodParams(C = C, gamma = gamma)
        
        # %%% Quantum part
        
        

        self.Q = utility.gen_svm_qubos(
            self.X_train, self.Y_train, self.B, self.K, self.xi, self.best_gamma, self.epsilon, self.beta, self.k0)
        self.MAXRESULTS = max(int(result_percentage_used*self.num_reads), 2)

        if self.hybrid: 
            if self.hybridsampler is None:
                self.hybridsampler = LeapHybridSampler()
            sampleset = self.hybridsampler.sample_qubo(Q=self.Q, time_limit=hybrid_time)
            sample = sampleset.first.sample
        
            N_for_decode = len(sample) // K
            N_samples = N_for_decode // 2
            B = float(B)
            Bvec = B ** (np.arange(K)-k0)
            
            #print(sample)
            adj_sample = "".join(str(x) for x in (list(sample.values())))
            self.hybrid_alphas = np.fromiter(adj_sample, float).reshape(N_for_decode, K) @ Bvec

            self.hybrid_alphas_1 = self.hybrid_alphas[0:N_samples]
            self.hybrid_alphas_2 = self.hybrid_alphas[N_samples:]

        # %% solutions combination

        # X_comb and Y_comb are defined to be the training and test datasets for the solutions combination techniques
        self.X_comb = self.X_train
        self.Y_comb = self.Y_train
        
        if self.change_to_logarithmic:
            self.X_comb_original = np.exp(self.X_comb)
            self.Y_comb_original = np.exp(self.Y_comb)
        else:
            self.X_comb_original = self.X_comb
            self.Y_comb_original = self.Y_comb


        self.X_train_reshaped = utility.modify_dataset(self.X_train)
        X_comb_reshaped = utility.modify_dataset(self.X_comb)

        if self.only_hybrid:
            return 
        
        if self.reuse_sampler and self.sampler is not None and can_use_same_embedding:
            self.alphas, results, self.sampler, self.response = utility.dwave_run(
                self.Q, self.B, self.K, self.MAXRESULTS, self.k0, sampler=self.sampler, num_reads=self.num_reads, anneal_time=anneal_time, use_custom_chainstrength=use_custom_chainstrength)
        else:
            self.alphas, results, self.sampler, self.response = utility.dwave_run(
                self.Q, self.B, self.K, self.MAXRESULTS, self.k0, sampler=sampler, num_reads=self.num_reads, anneal_time=anneal_time, use_custom_chainstrength=use_custom_chainstrength, chain_mult=chain_mult)
        

        # the first N alphas refer to the alpha_n variables, whereas the last N alphas refer to the alpha_hat_n variables
        # please refer to the description of the implementation of the QSVR for the details of the formulation

        alphas_1 = self.alphas[:, 0:self.N]
        alphas_2 = self.alphas[:, self.N:]

        alphas_1_mean = np.mean(alphas_1, axis=0)
        alphas_2_mean = np.mean(alphas_2, axis=0)


        # q_val_scores_m is a matrix that is used to create the solutions combination.
        # The prediction is stored at the i-th row and j-th column.
        # Where i represents the i-th set of alpha coefficients.
        # And j represents the j-th element of the X_comb dataset.
        self.q_val_scores_m = np.zeros((self.alphas.shape[0], self.X_comb.shape[0]))

        alphas_val_norm_1 = np.zeros((alphas_1.shape[1]))
        alphas_val_norm_2 = np.zeros((alphas_1.shape[1]))
        alphas_val_norm_lc_1 = np.zeros((alphas_1.shape[1]))
        alphas_val_softmax_lc_1 = np.zeros((alphas_1.shape[1]))

        alphas_val_softmax_1 = np.zeros((alphas_1.shape[1]))
        alphas_val_softmax_2 = np.zeros((alphas_1.shape[1]))
        alphas_val_norm_lc_2 = np.zeros((alphas_1.shape[1]))
        alphas_val_softmax_lc_2 = np.zeros((alphas_1.shape[1]))

        alphas_best_1 = np.zeros((alphas_1.shape[1]))
        alphas_best_2 = np.zeros((alphas_2.shape[1]))
        
        alphas_r2_softmax_1=np.zeros((alphas_1.shape[1]))
        alphas_r2_softmax_2=np.zeros((alphas_2.shape[1]))
        
        for i in range(self.alphas.shape[0]):
            alp_1 = alphas_1[i, :]
            alp_2 = alphas_2[i, :]
            res = utility.predict(X_comb_reshaped, self.X_train_reshaped, self.Y_train, alp_1, alp_2, self.B, self.K, self.epsilon, self.best_gamma, self.best_C)
            res = np.reshape(res, (self.X_comb.shape[0],))
            self.q_val_scores_m[i, :] = res

        if self.change_to_logarithmic:
            self.q_val_scores_original = np.exp(self.q_val_scores_m)
        else:
            self.q_val_scores_original = self.q_val_scores_m
        
        
        # the array validation_coefficients store the mean square error calculated for each set of alphas on the validation set
        # the same procedure applies to log_cosh_coefficients but the log-cosh error function is used
        validation_coefficients = np.zeros(self.alphas.shape[0])
        for i in range(self.q_val_scores_m.shape[0]):
            validation_coefficients[i] = mean_squared_error(
                self.Y_comb_original, self.q_val_scores_original[i, :])

        log_cosh_coefficients = np.zeros(self.alphas.shape[0])
        for i in range(self.q_val_scores_m.shape[0]):
            logging.info(f"Y_comb_original: {self.Y_comb_original}")
            logging.info(f"q_val_scores_original[i,:]: {self.q_val_scores_original[i,:]}")
            log_cosh_coefficients[i] = utility.log_cosh(
                self.Y_comb_original, self.q_val_scores_original[i, :])

        
        r2_coefficients = np.zeros(self.alphas.shape[0])
        for i in range(self.q_val_scores_m.shape[0]):
            validation_coefficients[i] = r2_score(
                self.Y_comb_original, self.q_val_scores_original[i, :])
        
        
        # the score assigned to each solution is then calculated by considering the multiplicative inverse of the validation coefficients

        validation_coefficients = 1/validation_coefficients
        log_cosh_coefficients = 1/log_cosh_coefficients

        ###################################################

        # for both mse and log-cosh the coefficients of the weighted average are calculated
        # they both use a normalization where each score is divided by the sum of all the scores or a softmax
        # by the choice of the loss functions and the combination methods they both yield weights
        # that can be used for a weighted average
        # the variable best_alpha stores the index of the best solution based on the prediction on the validation set

        # Perhaps more stable 
        validation_coefficients_softmax = np.exp(validation_coefficients-np.max(validation_coefficients))/sum(np.exp(validation_coefficients-np.max(validation_coefficients)))
        validation_coefficients_norm = validation_coefficients / sum(validation_coefficients)
        
        r2_coefficients_softmax=np.exp(r2_coefficients)/sum(np.exp(r2_coefficients))

        logging.info(f"log_cosh_coefficients: {log_cosh_coefficients}")

        log_cosh_coefficients_n = log_cosh_coefficients / sum(log_cosh_coefficients)
        log_cosh_coefficients_s = np.exp(log_cosh_coefficients)/sum(np.exp(log_cosh_coefficients))

        best_alpha = np.argmax(validation_coefficients)
        validation_coefficients_ba = np.zeros((self.MAXRESULTS,))
        validation_coefficients_ba[best_alpha] = 1

        # for the method that selects the best alpha all the weights are set to 0 except for the one that stores the best alpha



        for i in range(alphas_1.shape[1]):
            alphas_val_norm_1[i] = np.average(alphas_1[:, i], weights=validation_coefficients_norm)
            alphas_val_norm_2[i] = np.average(alphas_2[:, i], weights=validation_coefficients_norm)

            alphas_val_softmax_1[i] = np.average(alphas_1[:, i], weights=validation_coefficients_softmax)
            alphas_val_softmax_2[i] = np.average(alphas_2[:, i], weights=validation_coefficients_softmax)

            alphas_val_norm_lc_1[i] = np.average(alphas_1[:, i], weights=log_cosh_coefficients_n)
            alphas_val_norm_lc_2[i] = np.average(alphas_2[:, i], weights=log_cosh_coefficients_n)

            alphas_val_softmax_lc_1[i] = np.average(alphas_1[:, i], weights=log_cosh_coefficients_s)
            alphas_val_softmax_lc_2[i] = np.average(alphas_2[:, i], weights=log_cosh_coefficients_s)

            alphas_best_1[i] = np.average(alphas_1[:, i], weights=validation_coefficients_ba)
            alphas_best_2[i] = np.average(alphas_2[:, i], weights=validation_coefficients_ba)
            
            alphas_r2_softmax_1[i] = np.average(alphas_1[:, i], weights=r2_coefficients_softmax)
            alphas_r2_softmax_2[i] = np.average(alphas_2[:, i], weights=r2_coefficients_softmax)
        

        self.all_alphas = np.ndarray(shape=(7, 2, self.N))
        self.all_alphas[0, :, :] = alphas_val_norm_1, alphas_val_norm_2
        self.all_alphas[1, :, :] = alphas_val_softmax_1, alphas_val_softmax_2
        self.all_alphas[2, :, :] = alphas_val_norm_lc_1, alphas_val_norm_lc_2
        self.all_alphas[3, :, :] = alphas_val_softmax_lc_1, alphas_val_softmax_lc_2
        self.all_alphas[4, :, :] = alphas_best_1, alphas_best_2
        self.all_alphas[5, :, :] = alphas_1_mean, alphas_2_mean
        self.all_alphas[6, :, :] = alphas_r2_softmax_1, alphas_r2_softmax_2
        
        """
        ^^^ This determines the order of the scoring algorithms. ^^^
        """
            

        date_str = self.get_date_string()
        dir_path = RESULTS_DIR / date_str
        os.makedirs(dir_path, exist_ok=True)

        # IMPORTANT
        # the following lines of code saves the datasets that are already in the logarithmic domain
        #np.save(dir_path / 'X_train.npy', self.X_train)
        #np.save(dir_path / 'Y_train.npy', self.Y_train)
        #np.save(dir_path / 'Q.npy', self.Q)
        #import pickle
        #with open(dir_path / "response.pkl", "wb") as handle:
        #    pickle.dump(self.response, handle)
        #with open(dir_path / "results.pkl", "wb") as handle:
        #    pickle.dump(results, handle)

        
    def fit_cqm(self, X_train, y_train, B, K, k0, gamma, epsilon, C, limit=10):
        
        N=X_train.shape[0]
        self.N=N
        self.B=B
        self.K=K
        self.gamma=gamma
        self.k0=k0
        self.X_train=X_train
        self.X_train_reshaped=utility.modify_dataset(X_train)
        self.y_train=y_train
        self.time_limit=limit
        self.epsilon=epsilon
        self.C=C
        
        #be careful with the time limit because the CQM consumes a lot of compute time
        
        
        model_cqm=utility.get_cqm(X=X_train, Y=y_train, B=B, K=K, k0=k0, gamma=gamma, epsilon=epsilon)
        cqm_sampler=LeapHybridCQMSampler()
        
        cqm_sampleset=cqm_sampler.sample_cqm(model_cqm, time_limit=limit)
        cqm_sample=cqm_sampleset.record.sample 
        cqm_sample=cqm_sample.ravel()
        
        cqm_alphas=utility.decode(cqm_sample,B, K, k0)
        self.cqm_alphas= cqm_alphas
        self.cqm_alphas_1=self.cqm_alphas[0:N]
        self.cqm_alphas_2=self.cqm_alphas[N:]
        
    def predict_cqm(self,X_test):
        X_test_reshaped=utility.modify_dataset(X_test)
        cqm_predictions=utility.predict(X_test=X_test_reshaped,
                                    X_train=self.X_train_reshaped,
                                    Y_train=self.y_train,
                                    alphas=self.cqm_alphas_1,
                                    alphas_2=self.cqm_alphas_2,
                                    B=self.B,
                                    K=self.K,
                                    epsilon=self.epsilon,
                                    gamma=self.gamma,
                                    C=self.C
                                    )
                                        
        
        self.cqm_predictions=cqm_predictions
        return cqm_predictions
        
        
    
    
    def predict(self, X_values):
        """
        Predict Y values for the given X values.

        Parameters:
        -----------
        X_values : Array-like
                A n*m two-dimensional array of training samples with n samples and m features.

        Returns:
        --------
        Y_pred : Array-like
                A one-dimensional array-like of length n containing the predicted values corresponding to `X_values`.
        """

        if self.change_to_logarithmic:
            X_values = np.log(X_values)

        X_values_reshaped = utility.modify_dataset(X_values)

        Y_scores = np.ndarray(shape=(7, X_values.shape[0]))
        for i in range(Y_scores.shape[0]):
            temp_output = utility.predict(
                X_test = X_values_reshaped, 
                X_train = self.X_train_reshaped, 
                Y_train = self.Y_train,
                alphas = self.all_alphas[i, 0, :],
                alphas_2 = self.all_alphas[i, 1, :],
                B = self.B,
                K = self.K,
                epsilon = self.epsilon,
                gamma = self.best_gamma,
                C = self.best_C
                )
            Y_scores[i, :] = np.reshape(temp_output,(temp_output.shape[0],))

            if self.change_to_logarithmic:
                Y_scores[i, :] = np.exp(Y_scores[i, :])
        
        return Y_scores
        
    
    def score(
        self, 
        X_test,
        Y_test,
        compare_to_classical: bool=True
        ):
        """
        Returns the coefficient of determination (R^2) of the prediction.
        Writes the score to file. 

        Parameters:
        -----------
        X_test : array-like of shape (n_samples, n_features)
                Test samples.
        Y_test : array-like of shape (n_samples,)
                True values for `X_test`.
        compare_to_classical : `bool`
                A boolean representing whether or not to print out a classical comparison of SVR mse & R^2 score.
        """

        hybridscore = None
        # Y_test are converted back to the original domain

        Y_test_original = Y_test
        X_test_original = X_test
        if self.change_to_logarithmic:
            X_test = np.log(X_test)
            Y_test = np.log(Y_test)
        
        
        SVR_score = -1
        if compare_to_classical:
            # Here the SVR is initialized using the hyperparameters determined in the previous 
            svr = SVR(kernel='rbf', C=self.best_C, epsilon=self.epsilon, gamma=self.best_gamma)
            svr.fit(self.X_train, self.Y_train)
            Y_predicted = svr.predict(X_test)
            Y_predicted_original = Y_predicted

            # predicted test samples in the logarithmic domain
            if self.change_to_logarithmic:
                Y_predicted_original = np.exp(Y_predicted)

            SVR_mse = mean_squared_error(Y_test_original, Y_predicted_original)
            print(f'SVR mse: \n{SVR_mse}')
            SVR_train_score = svr.score(self.X_train, self.Y_train)
            SVR_score = svr.score(X_test, Y_test)
            print(f"SVR score: \n{SVR_score}")
        if self.hybrid:
            #Hybrid only returns one solution so a seperate function must be called to get a prediction. 
            hybridscore = self.prediction_from_single_sample(self.hybrid_alphas_1, self.hybrid_alphas_2, X_test, Y_test)
            print(f"Hybrid score: {hybridscore}")

        if self.only_hybrid:
            return None, SVR_train_score, SVR_score, hybridscore


        mse_final_results = np.zeros((7,))
        scores_final_results = np.zeros((7,))

        Y_scores = self.predict(X_values=X_test_original)
        
        for i in range(Y_scores.shape[0]):
            
            mse_final_results[i] = mean_squared_error(Y_test_original, Y_scores[i, :])
            scores_final_results[i] = 1 - ((Y_test_original - Y_scores[i, :])**2).sum() / ((Y_test_original - Y_test_original.mean())**2).sum()
            

        best_method = np.argmin(mse_final_results)

        methods = {0: (Y_scores[0,:], 'scores norm'),
                   1: (Y_scores[1,:], 'scores softmax'),
                   2: (Y_scores[2,:], 'scores lc norm'),
                   3: (Y_scores[3,:], 'scores lc softmax'),
                   4: (Y_scores[4,:], 'best set of alphas'),
                   5: (Y_scores[5,:], 'simple mean'),
                   6: (Y_scores[6,:], 'r2-based scores')}

        mse_best_method = mse_final_results[best_method]

        
        methods_array = np.asarray([methods[i][0] for i in range(len(methods))])

        date_str = self.get_date_string()
        dir_path = RESULTS_DIR / date_str
        os.makedirs(dir_path, exist_ok=True)
        
        # IMPORTANT
        # The following lines save the datasets in the original domain.
        np.save(dir_path / 'X_test.npy', X_test)
        np.save(dir_path / 'Y_test.npy', Y_test)

        if compare_to_classical:
            np.save(dir_path / 'SVR_predictions.npy', Y_predicted_original)

        np.save(dir_path / 'QSVR_predictions.npy', methods_array)

        if compare_to_classical:
            compare = np.vstack((methods_array, Y_predicted_original, Y_test_original))
            np.save(dir_path / "compare.npy", compare)

        print('QSVR mse: ')
        print(mse_final_results)
        print('QSVR scores: ')
        print(scores_final_results)
        from scatter import plot_scatters, plot_scatters_mult_graphs
        plot_scatters_mult_graphs(date_str)
        postprocessed_result = self.post_processing_idea(X_test, Y_test)
        print("Postprocessed:", postprocessed_result)

        scores_final_results[6] = postprocessed_result
        return scores_final_results, SVR_train_score, SVR_score, hybridscore
    
    def get_response(self):
        return self.response
    
    def get_qubo(self):
        return self.Q
    
    def get_sampler(self):
        return self.sampler
   
    # Predict from a single sample, currently only returns the R^2 score. 
    def prediction_from_single_sample(self, alphas_1, alphas_2, X_test, Y_test):
        """
        Predict and score a set of datapoints with given alpha-lists, e.g. from a single sample.
        """
        
        Y_test_original = Y_test

        if self.change_to_logarithmic:
            Y_test_original = np.exp(Y_test)

        X_test_reshaped = utility.modify_dataset(X_test)
        res = utility.predict(X_test_reshaped, self.X_train_reshaped, self.Y_train, alphas_1, alphas_2, self.B, self.K, self.epsilon, self.best_gamma, self.best_C)
        
        pred = np.reshape(res,(res.shape[0],))

        if self.change_to_logarithmic:
            pred = np.exp(pred)

        scores_final_results = 1 - ((Y_test_original - pred)**2).sum() / ((Y_test_original - Y_test_original.mean())**2).sum()
            
        return scores_final_results

    def post_processing_idea(self, X_test, Y_test):
        import numpy.lib.recfunctions as rfn
        response = self.response

        samples = np.array([''.join(map(str, sample)) for sample in response.record[
        'sample']])  # NOTE: it would be safer to use the labeling from record.data() for the qubit variable order
        unique_samples, unique_idx, unique_counts = np.unique(samples, return_index=True, return_counts=True)  # unfortunately, num_occurrences seems not to be added up after unembedding

        unique_records = response.record[unique_idx]
        result = rfn.merge_arrays((unique_samples, unique_records['energy'], unique_counts))
        #####
        #print("result: ", result)
        result = result[np.argsort(result['f1'])]

        num_results = 50
        running_window_size = 5
        scores_single_train = np.zeros((num_results))
        scores_running_train = np.zeros((num_results))
        X_comb_reshaped = utility.modify_dataset(self.X_train)


        def bla_train(functionality_type, generic_score_array, num_maxresults, begin_index=0):
            def alpha_extract(alphas):
                # alphas = np.array([utility.decode(sample, self.B, self.K, self.k0) for sample in result['f0'][begin:num]])
                alphas_1 = alphas[:, 0:self.N]
                alphas_2 = alphas[:, self.N:]
                q_val_scores_m = np.zeros((alphas.shape[0], self.N))
                for i in range(alphas.shape[0]):
                    alp_1 = alphas_1[i, :]
                    alp_2 = alphas_2[i, :]
                    res = utility.predict(X_comb_reshaped, self.X_train_reshaped, self.Y_train, alp_1, alp_2, self.B, self.K, self.epsilon, self.best_gamma, self.best_C)
                    res = np.reshape(res, (self.X_comb.shape[0],))
                    q_val_scores_m[i, :] = res

                q_val_scores_original = q_val_scores_m

                log_cosh_coefficients = np.zeros(alphas.shape[0])
                for i in range(q_val_scores_m.shape[0]):
                    log_cosh_coefficients[i] = utility.log_cosh(self.Y_comb_original, q_val_scores_original[i, :])
                
                log_cosh_coefficients = 1/log_cosh_coefficients

                alphas_val_norm_lc_1 = np.zeros((alphas_1.shape[1]))
                alphas_val_norm_lc_2 = np.zeros((alphas_1.shape[1]))

                log_cosh_coefficients_n = log_cosh_coefficients / sum(log_cosh_coefficients)

                for i in range(alphas_1.shape[1]):
                    alphas_val_norm_lc_1[i] = np.average(alphas_1[:, i], weights=log_cosh_coefficients_n)
                    alphas_val_norm_lc_2[i] = np.average(alphas_2[:, i], weights=log_cosh_coefficients_n)
                return alphas_val_norm_lc_1, alphas_val_norm_lc_2

            if functionality_type == 3:                                                     # Single window test
                alphas = np.array([utility.decode(sample, self.B, self.K, self.k0) for sample in result['f0'][max(begin_index+1,0):begin_index+running_window_size+1]])
                #print(f"result.shape: {result.shape} ")
                #print(f"result['f0'].shape: {result['f0'].shape} ")

                #print(alphas)
                #print("shape: ",alphas.shape)
                alphas_val_norm_lc_1, alphas_val_norm_lc_2 = alpha_extract(alphas)
                return self.prediction_from_single_sample(alphas_val_norm_lc_1, alphas_val_norm_lc_2, X_test, Y_test)
                
            for num in range(1,num_maxresults+1):
                if functionality_type == 0:      begin = num - 1                            # Single
                elif functionality_type == 1:    begin = 0                                  # Cumulative
                elif functionality_type == 2:    begin = max(num - running_window_size, 0)  # Running window
                
                alphas = np.array([utility.decode(sample, self.B, self.K, self.k0) for sample in result['f0'][begin:num]])
                if functionality_type != 0: 
                    alphas_val_norm_lc_1, alphas_val_norm_lc_2 = alpha_extract(alphas)
                else: 
                    alphas_val_norm_lc_1 = alphas[:, 0:self.N]
                    alphas_val_norm_lc_2 = alphas[:, self.N:]
                (generic_score_array)[num-1] = self.prediction_from_single_sample(alphas_val_norm_lc_1, alphas_val_norm_lc_2, self.X_train, self.Y_train)
                
            return generic_score_array
            
        scores_single_train = bla_train(0, scores_single_train, num_results)
        result = result[np.argsort(-scores_single_train[:])]
        scores_running_train = bla_train(2, scores_running_train, num_results)
        best_index = np.argmax(scores_running_train)
        print(f"best index was: ", best_index, "with the score: ", np.max(scores_running_train))
        # scores_running = np.zeros((1,))
        
        return bla_train(3, None, best_index, best_index-5)
        

    #Creates a string of the current date and time. 
    def get_date_string(self):
        now = datetime.now()
        date_time = str(now)
        date_time = date_time[0:19]
        date_array = []

        for i in range(len(date_time)):
            date_array.append(date_time[i])
            if date_array[i] == ' ' or date_array[i] == '-' or date_array[i] == ':':
                date_array[i] = '_'

        date_str = ''
        for j in date_array:
            date_str = date_str+j
        return date_str


if __name__ == "__main__":
    
    pass

# %%
