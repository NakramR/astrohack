import numpy as np

N = 78859

Xg = np.load('Xg-'+str(N) +'-stephan-preproc-test.npy')

Xg3 = np.zeros((N, 224,224,3))

Xg3[:,:,:,:] = Xg.reshape(N,224,224,1)

Xg = np.save('Xg3-'+str(N) +'-stephan-preproc-test.npy')
print('done')
