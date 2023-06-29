"""

@author: Edoardo Pasetto
"""

import numpy as np



import pandas as pd
import dwave_networkx as dnx
from minorminer import busclique
from dimod.binary import BinaryQuadraticModel
from dwave.system import DWaveSampler
from dwave.system import DWaveCliqueSampler
from dwave_networkx import pegasus_graph
from dwave.embedding.pegasus import find_clique_embedding
from dwave.embedding import is_valid_embedding
import networkx as nx

from minorminer import find_embedding


#%%

def get_clique_emebedding(dim,region=None, solver=None):
    '''
    

    Parameters
    ----------
    dim : int
    dimension of the clique
    region : TYPE, optional
        DESCRIPTION. The default is None.
    solver : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    embedding : TYPE
        DESCRIPTION.

    Important: region and solver, if given, must be given together!!!
    to do: write exceptions to handle this issue
    '''
    var_list=list(range(dim))
    
    if region==None and solver==None:
        sampler=DWaveSampler()
    else:
        sampler=DWaveSampler(region=region, solver=solver)
        
        
    topology=sampler.properties['topology']
    active_qubits=sampler.properties['qubits']
    active_couplers=sampler.properties['couplers']
    
    new_pegasus=pegasus_graph(topology['shape'][0], node_list=active_qubits, edge_list=active_couplers)   
    embedding=find_clique_embedding(k=var_list, target_graph= new_pegasus)
    
    return embedding
    

def load_embedding(path):
    r=open(path)
    lines=r.readlines()

    r.close()

    recovered_embedding={}
    for l in range(len(lines)):
        data=np.fromstring(lines[l], sep=',').astype('int32')
        data=list(data)
        key=data[0]
        value=data[1:]
        
        recovered_embedding[key]=value
        
   
    
    return recovered_embedding
    
    
    


def check_clique_embedding(emb):
    
    sampler=DWaveSampler()
    topology=sampler.properties['topology']
    active_qubits=sampler.properties['qubits']
    active_couplers=sampler.properties['couplers']
    
    new_pegasus=pegasus_graph(topology['shape'][0], node_list=active_qubits, edge_list=active_couplers) 
    
    var_list=list(emb.keys())
    complete_graph= nx.complete_graph(var_list)
    
    res=is_valid_embedding(emb, complete_graph, new_pegasus)
    
    
    return res


def check_embedding(embedding,bqm):
    sampler=DWaveSampler()
    topology=sampler.properties['topology']
    active_qubits=sampler.properties['qubits']
    active_couplers=sampler.properties['couplers']
    
    new_pegasus=pegasus_graph(topology['shape'][0], node_list=active_qubits, edge_list=active_couplers) 
    
    var_list=list(bqm.variables)
    
    G=nx.Graph()
    G.add_nodes_from(var_list)
    G.add_edges_from(list(bqm.quadratic.keys()))
    
    res=is_valid_embedding(embedding, G, new_pegasus)
    
    
    return res

    


def save_embedding(emb, path):
    writing_path=path+'embedding.txt'
    f=open(writing_path,'w+')
    f.write('{')
    for k in emb.keys():
        f.write(str(k))
        f.write(' :(')
        
        for val in emb[k]:
            f.write(str(val))
            f.write(',')
            
        f.write('),')    
        f.write('\n')
        
    f.write('}')
    f.close()