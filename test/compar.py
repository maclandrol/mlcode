import matplotlib.pyplot as plt
import numpy
import time
from mlearning import mnist, dropout
import sys
np=numpy

#load data
# Compare dropout to L2 regulation
train_set,train_label= mnist.load_mnist(dataset="train")
test_set, test_label = mnist.load_mnist(dataset ="test")

prng = np.random.RandomState(20141216)
train_shuffle=np.arange(train_set.shape[0])
test_shuffle=np.arange(test_set.shape[0])
prng.shuffle(train_shuffle)
prng.shuffle(test_shuffle)

train_size =5000
valid_size=1000

train_x= train_set[train_shuffle[:train_size]]
train_y = train_label[train_shuffle[:train_size]]
valid_x= train_set[train_shuffle[train_size:train_size+valid_size]]
valid_y= train_label[train_shuffle[train_size:train_size+valid_size]]
test_x= test_set[:valid_size]
test_y= test_label[:valid_size]

#parameters
activation='relu'
test_txt="test"
valid_txt="validation"
train_txt="train"
l2_reg=  numpy.logspace(np.log10(1./1000),np.log10(1./100000),3)
learning_rate=1
learning_decay=0.1
max_epoch=400
batch_size=100
monit={valid_txt: (valid_x, valid_y)}

n_hid=[512, 512]
dropout=[0., 0.5, 0.5]
n_in= train_x.shape[1]
n_out=np.unique(train_y).shape[0]

#dropout vs dropout+weight_decay vs weight_decay
best_drop_l2, best_error_drop_l2= None, np.inf
best_l2, best_error_l2= None, np.inf

for l in l2_reg:
    mlp_drop_l2 = dropout.MLP(n_in, n_hid, n_out, activation, dropout, l2=l)      
    sgd_drop_l2= dropout.SGDTraining(mlp_drop_l2)
    sgd_drop_l2.train(train_x, train_y, learning_rate, max_epoch, batch_size, decay=learning_decay, monitoring_data=monit)
   
    mlp = dropout.MLP(n_in, n_hid, n_out, activation, 0.0, l2=l, use_dropout=False)
    sgd_l2= dropout.SGDTraining(mlp)
    sgd_l2.train(train_x, train_y, learning_rate, max_epoch, batch_size, decay=learning_decay, monitoring_data=monit)
    
    if(sgd_drop_l2.error_curves[valid_txt][-1]<best_error_drop_l2):
        best_drop_l2 = sgd_drop_l2
        best_error_drop_l2= sgd_drop_l2.error_curves[valid_txt][-1]

    if(sgd_l2.error_curves[valid_txt][-1]<best_error_l2):
        best_l2 = sgd_l2
        best_error_l2= sgd_l2.error_curves[valid_txt][-1]

# dropout only
mlp_drop= dropout.MLP(n_in, n_hid, n_out, activation, dropout, l2=0.0)
best_drop= dropout.SGDTraining(mlp_drop)
best_drop.train(train_x, train_y, learning_rate, max_epoch, batch_size, decay=learning_decay, monitoring_data=monit)

# plot variable
cost_curves={}
test_error_curves={}
train_error_curves={}
epochs=best_drop.epochs

cost_curves["dropout"] = best_drop.cost_curves[train_txt]
cost_curves["L2_reg (lambda= %s)"%best_l2.mlp.l2[0]] = best_l2.cost_curves[train_txt]
cost_curves["dropout + L2_reg (lambda= %s)"%best_drop_l2.mlp.l2[0]] = best_drop_l2.cost_curves[train_txt]

test_error_curves["dropout"] = best_drop.cost_curves[valid_txt]
test_error_curves["L2_reg (lambda= %s)"%best_l2.mlp.l2[0]] = best_l2.error_curves[valid_txt]
test_error_curves["dropout + L2_reg (lambda= %s)"%best_drop_l2.mlp.l2[0]] = best_drop_l2.cost_curves[valid_txt]

train_error_curves["dropout"] = best_drop.cost_curves[train_txt]
train_error_curves["L2_reg (lambda= %s)"%best_l2.mlp.l2[0]] = best_l2.error_curves[train_txt]
train_error_curves["dropout + L2_reg (lambda= %s)"%best_drop_l2.mlp.l2[0]] = best_drop_l2.cost_curves[train_txt]


dropout.plot_training_curves(epochs, cost_curves, title=u"Courbe d'apprentissage %s - \nFonction de perte", ylabel="Perte", path="cost")
dropout.plot_training_curves(epochs, train_error_curves, title=u"Courbe d'apprentissage %s - \nErreur de prediction sur train", ylabel="Erreur", path="train")
dropout.plot_training_curves(epochs, test_error_curves, title=u"Courbe d'apprentissage %s - \nErreur de prediction sur validation", ylabel="Erreur",path="valid")

#prediction

erreur_drop=best_drop.mlp.compute_error(test_x, test_y)
erreur_l2=best_l2.mlp.compute_error(test_x, test_y)
erreur_dropl2=best_drop_l2.mlp.compute_error(test_x, test_y)

print "Erreur: \n", "dropout=%s"%erreur_drop,  "L2=%s"%erreur_l2,  "L2 + dropout=%s"%erreur_dropl2, "\n"