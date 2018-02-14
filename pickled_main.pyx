import numpy
import pyximport; pyximport.install(setup_args={"include_dirs":numpy.get_include()})
import numpy as np
import pickle
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sampler = pickle.load(open('save.p','rb'))

chain,logp = sampler.sample(10000,10,0,1/512.)

np.save(open('chain.npy','wb'),chain,allow_pickle=False)
np.save(open('likelihoods.npy','wb'),logp,allow_pickle=False)

plt.plot(range(len(logp)),logp)
plt.xlabel('Step Number')
plt.ylabel('Log-Likelihood')
plt.savefig('logp.png')