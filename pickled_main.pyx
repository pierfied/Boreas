import numpy
import pyximport; pyximport.install(setup_args={"include_dirs":numpy.get_include()})
import numpy as np
import pickle

sampler = pickle.load(open('save.p','rb'))

chain,logp = sampler.sample(1000,100,1000,0.01)

np.save(open('chain.npy','wb'),chain,allow_pickle=False)
np.save(open('likelihoods.npy','wb'),logp,allow_pickle=False)