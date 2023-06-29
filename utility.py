from time import sleep
import numpy as np
import numpy.lib.recfunctions as rfn
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite, LazyFixedEmbeddingComposite
from dimod import BinaryQuadraticModel
from dimod import ConstrainedQuadraticModel
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import os
import sys

from pathlib import Path
import logging

CURRENT_DIR = Path(__file__).parent

# logging.basicConfig(filename=CURRENT_DIR/"utility_debug.log", encoding="utf-8", level=logging.INFO)
logging.basicConfig(filename=CURRENT_DIR/"utility_debug.log", level=logging.INFO)


def kernel(xn, xm, gamma=-1): # here (xn.shape: NxD, xm.shape: ...xD) -> Nx...
    if gamma == -1:
        return xn @ xm.T
    xn = np.atleast_2d(xn)
    xm = np.atleast_2d(xm)
    return np.exp(-gamma * np.sum((xn[:,None] - xm[None,:])**2, axis=-1)) # (N,1,D) - (1,...,D) -> (N,...,D) -> (N,...); see Hsu guide.pdf for formula


# decode binary -> alpha
def decode(binary, B=10, K=3, k0=0):
    N = len(binary) // K
    B = float(B)
    Bvec = B ** (np.arange(K)-k0)
    return np.fromiter(binary,float).reshape(N,K) @ Bvec


def log_cosh_base_2(x):
    """
    Calculate log_2(cosh(x)) with numerically stable methods
    """
    # ~Courtesey of https://stackoverflow.com/questions/57785222/avoiding-overflow-in-logcoshx
    # s always has real part >= 0
    return np.logaddexp2(x, -x) - 1
    s = np.sign(x) * x
    p = np.exp(-2 * s)
    return s + np.log1p(p) - np.log(2)

def log_cosh(Y_true, Y_pred): # the inputs Y_tue and Y_pred are given as arrays of the same dimension
    res=0
    for i in range(Y_true.shape[0]):
        d=Y_true[i]-Y_pred[i]
        # c=np.cosh(d)
        # l=np.log2(c)
        l = log_cosh_base_2(d)  # "Numerically stable"
        res+=l
        
    return res



###################################################################################################
    




def eval_regressor(x, alphas, alphas_2, data, gamma, b=0):  #the reshaped version of X_train and X_test are needed
    
    alphas_d=alphas-alphas_2
    res=[]
    #the x provided is X_test and data is X_train
    for i in range(len(x)):
        temp=0
        for j in range(len(data)):
            k=kernel(x[i],data[j],gamma)
            temp+=alphas_d[j]*kernel(x[i],data[j],gamma)
        temp+=b
        res.append(temp)
        
    res=np.array(res)
    #print('VALUE FOR B IS',b)
    return res
    


def modify_dataset(X):
    #change the structure of the dataset in order to be properly used by the function gen_svm_qubos
    #given a dataset X NxD it turns into a list of length N where each element is a np.array of size D
    X_new=[]
    for i in range(X.shape[0]):
        X_new.append(X[i,:])
    
    return X_new

                 

def predict(X_test, X_train, Y_train, alphas, alphas_2, B, K, epsilon, gamma, C):

    
    b = eval_offset_avg(alphas, alphas_2, X_train, Y_train, epsilon, gamma, C) #requires the reshaped version of X_train

    scoretest = eval_regressor(X_test, alphas, alphas_2, X_train, gamma,  b) #requires the reshaped version of X_train and X_test

    return scoretest
   
def gen_svm_qubos(X, Y, B, K, xi, gamma, epsilon, beta, k0):
    N = len(X)
    
    Q = np.zeros((2*K*N, 2*K*N))
    print(f'Creating the QUBO Q matrix of size {Q.shape}')
    for n in range(N):
        for m in range(N):
            for i in range(K):
                for j in range(K):
                    Q[K * n + i, K * m + j] = 0.5 * B ** (i + j - 2*k0) * (kernel(X[n], X[m], gamma)) + B**(i+j-2*k0)*xi
                    Q[K*(N+n)+i, K*(N+m)+j] = 0.5 * B ** (i + j - 2*k0) * (kernel(X[n], X[m], gamma)) + B**(i+j-2*k0)*xi
                    Q[K * n + i, K*(N+m)+j] =-0.5 * B ** (i + j -2*k0) * (kernel(X[n], X[m], gamma)) - B**(i+j-2*k0)*xi
                    Q[K*(N+n)+i, K * m + j] =-0.5 * B ** (i + j -2*k0) * (kernel(X[n], X[m], gamma)) - B**(i+j-2*k0)*xi
    
    for n in range(N):
        for i in range(K):
            Q[K*n+i,K*n+i]+= epsilon* B**(i-k0) - B**(i-k0) * Y[n] # B**i * (epsilon-Y[n])
            Q[K*(N+n)+i,K*(N+n)+i] += epsilon* B**(i-k0) + B**(i-k0) * Y[n] # B**i * (epsilon+Y[n])
            
    
    for n in range(N):
        for i in range(K):
            for j in range(K):
                Q[K*n +i, K*(N+n) +j]+=beta*B**(i+j-2*k0)
            

    Q = np.triu(Q) + np.tril(Q, -1).T  # turn the symmetric matrix into upper triangular
    return Q



def eval_offset_avg(alphas, alphas_2, data, target, epsilon, gamma, C, useavgforb=True):
    #to obtain an estimate of b is necessary to average all the estimate of from the support vectors
    #requires the reshaped X_train
    alpha_d=alphas-alphas_2
    support_vectors_1=[]
    support_vectors_2=[]
    b_estimates_1=[]
    b_estimates_2=[]
    
    #to estimate b only the 0<alphas<C are considered
    
    #the indexes corresponding to the support vectors are calculated

    logging.info(f"alphas: {alphas}")
    logging.info(f"alphas_2: {alphas_2}")
    while len(support_vectors_1) == 0 or len(support_vectors_2) == 0:
        for i in range(len(alphas)):                        #alphas and alphas_2 have the same length
            if not np.isclose(alphas[i],0) and alphas[i]<C:
                support_vectors_1.append(i)
            if not np.isclose(alphas_2[i],0) and alphas_2[i]<C:
                support_vectors_2.append(i)
        if len(support_vectors_1) == 0 or len(support_vectors_2) == 0:
            C = C+0.1
            support_vectors_1=[]
            support_vectors_2=[]
            print("SUPPORT VECTOR NOT FOUND, TRYING WITH A LARGER VALUE OF C")
    
    logging.info(f"target: {target}")
    logging.info(f"support_vectors_1: {support_vectors_1}")
    logging.info(f"support_vectors_2: {support_vectors_2}")

        
    for i in support_vectors_1:
        temp=0
        for j in range(len(alphas)):
            temp+=alpha_d[j]*kernel(data[i],data[j],gamma)
        
        estimate=target[i]-epsilon-temp
        b_estimates_1.append(estimate)
    
    for i in support_vectors_2:
        temp=0
        for j in range(len(alphas)):
            temp+=alpha_d[j]*kernel(data[i],data[j],gamma)
        
        estimate=target[i]-epsilon-temp
        b_estimates_2.append(estimate)
    
    logging.info(f"b_estimates_1: {b_estimates_1}")
    logging.info(f"b_estimates_2: {b_estimates_2}")
    b_avg=(np.sum(b_estimates_1)+np.sum(b_estimates_2))/(len(b_estimates_1)+len(b_estimates_2))
    
    
    return b_avg




####################################################################################


def dwave_run(Q, B, K, m, k0=0, sampler=None, num_reads=2500, anneal_time=20, use_custom_chainstrength: bool=False, chain_mult: float=1):

    MAXRESULTS = m

    print(f'Extracting nodes and couplers from Q')

    # qubo_couplers = np.asarray([[n, m, Q[n, m]] for n in range(len(Q)) for m in range(n + 1, len(Q)) if not np.isclose(Q[n, m], 0)])
    qubo_couplers = np.asarray([[n, m, Q[n, m]] for n in range(len(Q)) for m in range(n + 1, len(Q))])
    qubo_couplers = qubo_couplers[np.argsort(-np.abs(qubo_couplers[:, 2]))]

    qubo_nodes = np.asarray([[n, n, Q[n, n]] for n in range(len(Q))])  # if not np.isclose(Q[n,n],0)]) NOTE: removed due to variable order!

    #qubo_nodes = np.array([[i, i, (qubo_nodes[qubo_nodes[:, 0] == i, 2][0] if i in qubo_nodes[:, 0] else 0.)] for i in
    #                       np.arange(np.concatenate((qubo_nodes, qubo_couplers))[:, [0,
    #                                                                                 1]].max() + 1)])  # to make sure every (i,i) occurs in the qubo in increasing order such that the variable order in BinaryQuadraticModel is consistent (see locate wrongenergies-* github issue)

    print(f'The problem has {len(qubo_nodes)} nodes and {len(qubo_couplers)} couplers')

    is_clique = True
    if len(qubo_couplers) < (len(qubo_nodes)*(len(qubo_nodes)-1))/2:
        is_clique = False
        sampler = None


    maxcouplers = len(qubo_couplers)  ## POSSIBLE INPUT if len(sys.argv) <= 2 else int(sys.argv[2])
    couplerslist = [maxcouplers]

    # Are these values still valid for the new QA advantage system with PEGASUS ????
    for trycouplers in [2500, 2000, 1800, 1600, 1400, 1200, 1000, 500]:
        if maxcouplers > trycouplers:
            couplerslist += [trycouplers]

    if sampler is None:
        sampler = LazyFixedEmbeddingComposite(DWaveSampler())
        
    for couplers in couplerslist:  # try to reduce as little couplers as necessary to find an embedding
        Q_np = Q
        Q = {(q[0], q[1]): q[2] for q in np.vstack((qubo_nodes, qubo_couplers[:couplers]))}

        print(f'Running with {len(qubo_nodes)} nodes and {couplers} couplers')
        
        #ordering = np.array(list(BinaryQuadraticModel.from_qubo(Q,offset=0).iter_variables()))
        ordering = np.array(list(iter(BinaryQuadraticModel.from_qubo(Q).variables)))
        if not (ordering == np.arange(len(ordering), dtype=ordering.dtype)).all():
            print(f'WARNING: variables are not correctly ordered! ordering={ordering}')
        #sleep(2)  # Trying to avert remote end closing the connection without response
        
        try:
            if use_custom_chainstrength:
                chain_strength = chain_mult * int(np.ceil(np.max([np.abs(np.min(Q_np)),np.abs(np.max(Q_np))])))
                response = sampler.sample_qubo(Q, num_reads=num_reads, annealing_time=anneal_time, chain_strength=chain_strength) #10000 max # maybe some more postprocessing can be specified here ...
            else:
                response = sampler.sample_qubo(Q, num_reads=num_reads, annealing_time=anneal_time)
        except ValueError as v:
            print(f' -- no embedding found, trying less couplers')
            continue
        except ConnectionError as e:
            print("ConnectionError encountered:")
            print(e)
            print("Retrying...")
            sleep(5)
            try:
                if use_custom_chainstrength:
                    response = sampler.sample_qubo(Q, num_reads=num_reads, annealing_time=anneal_time, chain_strength=chain_strength) #10000 max # maybe some more postprocessing can be specified here ...
                else:
                    response = sampler.sample_qubo(Q, num_reads=num_reads, annealing_time=anneal_time)
            except ValueError as v:
                print(f' -- no embedding found, trying less couplers')
                continue
        break

    samples = np.array([''.join(map(str, sample)) for sample in response.record[
        'sample']])  # NOTE: it would be safer to use the labeling from record.data() for the qubit variable order
    unique_samples, unique_idx, unique_counts = np.unique(samples, return_index=True, return_counts=True)  # unfortunately, num_occurrences seems not to be added up after unembedding
   
    unique_records = response.record[unique_idx]
    try:
        result = rfn.merge_arrays(
            (unique_samples, unique_records['energy'], unique_counts, unique_records['chain_break_fraction']))
        #print(unique_records['chain_break_fraction'])
        print(f'Chain Breaks mean {unique_records["chain_break_fraction"].mean()}')
    except ValueError as e:
        print(e) 
        print('->trying with: result = rfn.merge_arrays((unique_samples, unique_records["energy"], unique_counts))')
        result = rfn.merge_arrays(
            (unique_samples, unique_records['energy'], unique_counts))
    #####
    #print("result: ", result)
    result = result[np.argsort(result['f1'])]
    #np.savetxt(pathsub + 'result.dat', result[:MAXRESULTS], fmt='%s', delimiter='\t',
    #           header='\t'.join(response.record.dtype.names),
    #           comments='')  # load with np.genfromtxt(..., dtype=['<U2000',float,int,float], names=True, encoding=None)
    #print("sample[f0]: ", result["f0"][0])
    #print("sample[f1]: ", result["f1"][0])
    
    alphas = np.array([decode(sample, B, K, k0) for sample in result['f0'][:MAXRESULTS]])
    #print(f"MAXRESULTS: {MAXRESULTS}")
    #print("alphas: ", alphas)
    return alphas, result, sampler if is_clique else None, response #the function returns all the alphas coefficients as a single array, they are then separated into alphas and aplhas_2 in the main through slicing



    
def hyperparameters_validation(X_val, Y_val, X_test, Y_test, gamma_values, C_values, epsilon_values):
    best_gamma = 0
    best_C = 0
    best_epsilon = 0
    best_mse = np.Inf

    mse_matrix = np.zeros((len(gamma_values),len(C_values)))

    for gamma_ind in range(len(gamma_values)):
        for C_ind in range(len(C_values)):
            for epsilon_ind in range(len(epsilon_values)):
                svr = SVR(kernel='rbf', C=C_values[C_ind], gamma=gamma_values[gamma_ind], epsilon=epsilon_values[epsilon_ind])
                svr.fit(X_val, Y_val)
                Y_pred = svr.predict(X_test)
                current_mse = mean_squared_error(Y_test, Y_pred)

                mse_matrix[gamma_ind,C_ind] = current_mse

                if current_mse < best_mse:
                    best_mse = current_mse
                    best_gamma = gamma_values[gamma_ind]
                    best_C = C_values[C_ind]
                    best_epsilon = epsilon_values[epsilon_ind]

    print(f'best value for gamma is {best_gamma}, best value for C is {best_C}')

    return mse_matrix, best_gamma, best_C, best_epsilon
    



def rscore_hyperparameters_validation(X_val, Y_val, X_test, Y_test, gamma_values, C_values, epsilon_values, verbosity_high=False):
    best_gamma=0
    best_C=0
    best_epsilon=0
    best_r_squared=-np.Inf


    r_squared_matrix=np.zeros((len(gamma_values),len(C_values)))

    for gamma_ind in range(len(gamma_values)):
        for C_ind in range(len(C_values)):
            for epsilon_ind in range(len(epsilon_values)):
                svr = SVR(kernel='rbf', C=C_values[C_ind], gamma=gamma_values[gamma_ind], epsilon=epsilon_values[epsilon_ind])
                svr.fit(X_val, Y_val)
                current_r_squared = svr.score(X_test, Y_test)

                r_squared_matrix[gamma_ind,C_ind] = current_r_squared

                if current_r_squared > best_r_squared:
                    best_r_squared = current_r_squared
                    best_gamma = gamma_values[gamma_ind]
                    best_C = C_values[C_ind]
                    best_epsilon = epsilon_values[epsilon_ind]

    if verbosity_high:
        print(f'best value for gamma is {best_gamma}, best value for C is {best_C}')

    return r_squared_matrix, best_gamma, best_C, best_epsilon


def define_sampler_from_embedding(embedding_file_name, region, solver):
    # First check if there is embedding:
    if os.path.isfile(embedding_file_name):
        embedding_file = open(embedding_file_name, 'r')
        try:
            print('Reading embedding...') 
            embedding = eval(embedding_file.read())
            aux_sampler = DWaveSampler(region=region, solver=solver)
            sampler = FixedEmbeddingComposite(aux_sampler, embedding=embedding)
        except Exception as ex:
            print('Embedding file reading error')
            print(ex)
            return None
        finally:
            embedding_file.close()
    return sampler



def get_Q_dict(Q_matrix):
    qubo_nodes=np.array([[i, i, Q_matrix[i,i]] for i in range(len(Q_matrix))])
    qubo_couplers=np.array([[i,j, Q_matrix[i,j]] for i in range(len(Q_matrix)) for j in range(i+1,len(Q_matrix)) if not np.isclose(Q_matrix[i,j],0)])             

    qubo_list=np.concatenate((qubo_nodes, qubo_couplers))
    
    Q={(q[0],q[1]):q[2] for q in qubo_list}
    
    return Q   #, qubo_nodes, qubo_couplers



def get_cqm(X, Y, B, K, k0, gamma, epsilon, beta=0, xi=0 ):
    N = len(X)
    
    Q = np.zeros((2*K*N, 2*K*N))
    print(f'Creating the QUBO Q matrix of size {Q.shape}')
    for n in range(N):
        for m in range(N):
            for i in range(K):
                for j in range(K):
                    Q[K * n + i, K * m + j] = 0.5 * B ** (i + j - 2*k0) * (kernel(X[n], X[m], gamma)) #+ B**(i+j-2*k0)*xi
                    Q[K*(N+n)+i, K*(N+m)+j] = 0.5 * B ** (i + j - 2*k0) * (kernel(X[n], X[m], gamma)) #+ B**(i+j-2*k0)*xi
                    Q[K * n + i, K*(N+m)+j] =-0.5 * B ** (i + j - 2*k0) * (kernel(X[n], X[m], gamma)) #- B**(i+j-2*k0)*xi
                    Q[K*(N+n)+i, K * m + j] =-0.5 * B ** (i + j - 2*k0) * (kernel(X[n], X[m], gamma)) #- B**(i+j-2*k0)*xi              
    
    
    
    
    for n in range(N):
        for i in range(K):
            Q[K*n+i,K*n+i]+= epsilon* B**(i-k0) - B**(i-k0) * Y[n] # B**i * (epsilon-Y[n])
            Q[K*(N+n)+i,K*(N+n)+i] += epsilon* B**(i-k0) + B**(i-k0) * Y[n] # B**i * (epsilon+Y[n])
            
    
    Q = np.triu(Q) + np.tril(Q, -1).T  # turn the symmetric matrix into upper triangular
    problem_dict=get_Q_dict(Q)
    problem_bqm=BinaryQuadraticModel.from_qubo(problem_dict)
    
    cqm=ConstrainedQuadraticModel.from_bqm(problem_bqm)
    
 
            
#%% xi constraint
    
    
    Q_xi=np.zeros((2*K*N, 2*K*N))
    for n in range(N):
        for m in range(N):
            for i in range(K):
                for j in range(K):
                    Q_xi[K * n + i, K * m + j] =   B**(i+j-2*k0)
                    Q_xi[K*(N+n)+i, K*(N+m)+j] =   B**(i+j-2*k0)
                    Q_xi[K * n + i, K*(N+m)+j] = - B**(i+j-2*k0)
                    Q_xi[K*(N+n)+i, K * m + j] = - B**(i+j-2*k0)
    
    Q_xi = np.triu(Q_xi) + np.tril(Q_xi, -1).T
    
    xi_dict=get_Q_dict(Q_xi)
    xi_bqm=BinaryQuadraticModel.from_qubo(xi_dict)
    cqm.add_constraint_from_model(xi_bqm, sense='==', rhs=0, label='xi constraint')
    
    return cqm


#%% beta constraint
  
    # Q_beta=np.zeros((2*K*N, 2*K*N))
    # for n in range(N):
    #     for i in range(K):
    #         for j in range(K):
    #             Q_beta[K*n +i, K*(N+n) +j]+=B**(i+j-2*k0)
    
    # Q_beta = np.triu(Q_beta) + np.tril(Q_beta, -1).T
    

#%%

    
   