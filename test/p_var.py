import matplotlib.pyplot as plt
import numpy
import time
from mlearning import mnist, dropout
import sys
np=numpy

#load data
# Test the effect of the dropout prob variation
train_set,train_label= mnist.load_mnist(dataset="train")
test_set, test_label = mnist.load_mnist(dataset ="test")

prng = np.random.RandomState(20141216)
train_shuffle=np.arange(train_set.shape[0])
test_shuffle=np.arange(test_set.shape[0])
prng.shuffle(train_shuffle)
prng.shuffle(test_shuffle)

train_size =5000
test_size = 1000

train_x= train_set[train_shuffle[:train_size]]
train_y = train_label[train_shuffle[:train_size]]
test_x= test_set[test_shuffle[:test_size]]
test_y= test_label[test_shuffle[:test_size]]

# test input with fixed probability of 0.5 for all other layer
#parameters
activation='relu'
test_txt="test"
train_txt="train"
l2_reg=  numpy.logspace(np.log10(1./10),np.log10(1./100000),5)
learning_rate=1
learning_decay=0.1
max_epoch=400
batch_size=100
monit={test_txt: (test_x, test_y)}
n_in= train_x.shape[1]
n_out=np.unique(train_y).shape[0]

n_hid=[1024,1024,2048]
fixednhid= [256,256,512]
# dropout value test

prob_value=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
ptest_error=[]
ptrain_error=[]

nptest_error=[]
nptrain_error=[]

for p in prob_value:
    h_dropout=[p]*len(n_hid)
    sgd=dropout.SGDTraining(dropout.MLP(n_in, n_hid, n_out, activation, [0.] + h_dropout))
    sgd.train(train_x, train_y, learning_rate, max_epoch, batch_size, decay=learning_decay, monitoring_data=monit)
    sgd2=dropout.SGDTraining(dropout.MLP(n_in, [int(n/p) for n in fixednhid], n_out, activation, [0.] + h_dropout))
    sgd2.train(train_x, train_y, learning_rate, max_epoch, batch_size, decay=learning_decay, monitoring_data=monit)

    ptest_error.append(sgd.error_curves[test_txt][-1])
    ptrain_error.append(sgd.error_curves[train_txt][-1])

    nptest_error.append(sgd2.error_curves[test_txt][-1])
    nptrain_error.append(sgd2.error_curves[train_txt][-1])

    
plt.plot(prob_value, ptrain_error, '-b.')
plt.plot(prob_value, ptest_error, '-g.')
plt.xlabel("p")
plt.ylabel("Erreur")
plt.legend(["train", "test"], loc="best")
plt.savefig("dropouteffect", format='eps', dpi=600)
plt.clf()

plt.plot(prob_value, nptrain_error, '-b.')
plt.plot(prob_value, nptest_error, '-g.')
plt.xlabel("p")
plt.ylabel("Erreur")
plt.legend(["train", "test"], loc="best")
plt.savefig("dropouteffectfixed", format='eps', dpi=600)