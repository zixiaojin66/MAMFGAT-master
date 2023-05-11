import numpy as np
for i in range(10):
    a=np.loadtxt('fold '+str(i)+' score.txt')
    b=np.loadtxt('fold '+str(i)+' test.txt')
    np.savetxt('fold '+str(i)+'.txt',np.concatenate((np.expand_dims(a,axis=1),np.expand_dims(b,axis=1)),axis=1),delimiter='\t')