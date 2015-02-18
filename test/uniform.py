import matplotlib.pyplot as plt
import numpy
import time
from mlearning import mnist, dropout
import sys
np=numpy

#load data
train_set,train_label= mnist.load_mnist(dataset="train")
test_set, test_label = mnist.load_mnist(dataset ="test")

# Test the effect of a random uniform picked dropout
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

#randomnly picked 
test_error={}
train_error={}
dropouts=np.random.uniform(size=(5,len(n_hid))).tolist()
ind=0
sgd=None
for drop in dropouts:
    sgd=dropout.SGDTraining(dropout.MLP(n_in, n_hid, n_out, activation, [0.] + drop))
    sgd.train(train_x, train_y, learning_rate, max_epoch, batch_size, decay=learning_decay, monitoring_data=monit)
    test_error["%s"%ind]=sgd.error_curves[test_txt]
    train_error["%s"%ind]=sgd.error_curves[train_txt]
    ind+=1

epochs=sgd.epochs
dropout.plot_training_curves(epochs, train_error, title=u"Courbe d'apprentissage %s - \nErreur de prediction sur train", ylabel="Erreur", path="uniformtrain")
dropout.plot_training_curves(epochs, test_error, title=u"Courbe d'apprentissage %s - \nErreur de prediction sur test", ylabel="Erreur", path="uniformtest")
