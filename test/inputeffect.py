import matplotlib.pyplot as plt
import numpy
import time
from mlearning import mnist, dropout
import sys
np=numpy

#load data
# Test the effect of dropout on input variation
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

n_hid=[512,512,1024]
drop_input= [0., 0.2, 0.5, 0.7, 0.9]
imlabel= ["p=0", "p=0.2", "p=0.5", "p=0.7", "p=0.9"]
input_effect= [dropout.MLP(n_in, n_hid, n_out, activation, [p]+[0.5]*len(n_hid)) for p in drop_input]

cost={}
test_error={}
train_error={}
epochs=[]
ind=0
sgd=None
for model in input_effect:
    sgd= dropout.SGDTraining(model)
    sgd.train(train_x, train_y, learning_rate, max_epoch, batch_size, decay=learning_decay, monitoring_data=monit)
    cost[imlabel[ind]]= sgd.cost_curves[train_txt]
    train_error[imlabel[ind]]= sgd.error_curves[train_txt]
    test_error[imlabel[ind]]= sgd.error_curves[test_txt]
    ind+=1

epochs=sgd.epochs

dropout.plot_training_curves(epochs, cost, title=u"Courbe d'apprentissage %s - \nFonction de perte", ylabel="Perte", path="inputeffectcost")
dropout.plot_training_curves(epochs, train_error, title=u"Courbe d'apprentissage %s - \nErreur de prediction sur train", ylabel="Erreur", path="inputeffecttrain")
dropout.plot_training_curves(epochs, test_error, title=u"Courbe d'apprentissage %s - \nErreur de prediction sur test", ylabel="Erreur", path="inputeffecttest")
