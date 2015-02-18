import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy
import time
from mlearning import mnist, dropout
import sys
np=numpy

#load data
train_set,train_label= mnist.load_mnist(dataset="train")
test_set, test_label = mnist.load_mnist(dataset ="test")

prng = np.random.RandomState(20141216)
train_shuffle=np.arange(train_set.shape[0])
test_shuffle=np.arange(test_set.shape[0])
prng.shuffle(train_shuffle)
prng.shuffle(test_shuffle)

train_size =10000
test_size = 5000

train_x= train_set[train_shuffle[:train_size]]
train_y = train_label[train_shuffle[:train_size]]
test_x= test_set[test_shuffle[:test_size]]
test_y= test_label[test_shuffle[:test_size]]

# test input with fixed probability of 0.5 for all other layer
# parameters
# Test the effect of weight variation
activation='relu'
test_txt="test"
train_txt="train"
l2_reg=  numpy.logspace(np.log10(1./10),np.log10(1./100000),5)
learning_rate=1
learning_decay=0.05
max_epoch=300
batch_size=100
monit={test_txt: (test_x, test_y)}
n_in= train_x.shape[1]
n_out=np.unique(train_y).shape[0]

mlp_drops= [dropout.MLP(n_in, [256], n_out, activation, [0., 0.5]),\
            dropout.MLP(n_in, [256], n_out, activation, [0.5, 0.5]),\
            dropout.MLP(n_in, [256], n_out, activation, [0.9, 0.5]),\
            dropout.MLP(n_in, [256], n_out, activation, [0.5, 0.5], l2= l2_reg[-1]),\
            dropout.MLP(n_in, [256], n_out, activation, [0., 0.], use_dropout=False)]

imlabels=["dropout: (0, 0.5)","dropout: (0.5, 0.5)","dropout: (0.9, 0.5)",\
        "dropout: (0.5, 0.5 with reg)", "without dropout"]

ind=0
for mlpd in mlp_drops:
    sgd= dropout.SGDTraining(mlpd)
    features= sgd.train(train_x, train_y, learning_rate, 100, batch_size, decay=learning_decay)

    plt.gray()
    fig = plt.figure(1, (16., 16.))
    grid = ImageGrid(fig, 111, # similar to subplot(111)
                nrows_ncols = (16, 16), # creates 2x2 grid of axes
                axes_pad=0.001, # pad between axes in inch.
                )

    for i in range(features.shape[1]):
        ax=grid[i]
        ax.imshow(features[:,i].reshape(28,28)) # The AxesGrid object work as a list of axes.
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.title(imlabels[ind])
    plt.savefig("weight%s"%ind, format='eps', dpi=600)
    plt.clf()
    ind+=1
