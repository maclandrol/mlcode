# -*- coding:utf-8 -*-

import itertools
from collections import OrderedDict
import numpy as np

import theano
import theano.tensor as T
from theano import function

import matplotlib.pyplot as plt


def cast_data(data):
    """
    Cast les données en floatX et int32
    X : floatX
    y : int32

    Returns
    -------
    liste de tuples representant (x, y)
    """
    return [(x.astype(theano.config.floatX), y.astype('int32')) for x, y in data]



def _dropout_layer(layer, p, rng=None):
    """
    dropout unit in the layer with probability p

    """
    if(rng is None):
        rng = np.random.RandomState([2014, 12, 15])

    srng= T.shared_randomstreams.RandomStreams(rng.randint(999999))
    # we set this to 1-p because p is the prob to drop and not keep
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    output=layer*T.cast(mask, theano.config.floatX)
    return output


class SGDTraining(object):
    """
    Entraînement par descente de gradient stochastique.
    (Stochastic Gradient Descent)

    La classe elle même est abstraite et doit être hérité d'un
    modèle pour faire son entraînement.

    Attributes
    ---------
    message_frequency: float
        La fréquence à laquelle le message est mis à jour 
        (Pourcentage du nombre maximal d'époque)
    epochs: list of int
        Liste des époques où il y a eu enregistrement du coût
        et de la perte du modèle
    cost_curves: list of float
        Liste de la perte du modèle à chaque mise à jour du message
    error_curves
        Liste du coût du modèle à chaque mise à jour du message

    Methods
    -------
    train(train_data, learning_rate, max_epoch, train_labels=None, 
          batch_size=128, stopping_rule=None, monitoring_data=None)
        Entraine le modèle jusqu'à max_epoch ou jusqu'à ce que stopping_rule
        renvoie True

    """
    def __init__(self, mlp=None, message_frequency=0.01):
        """
        Parameters
        ----------
        message_frequency: float
            La fréquence à laquelle le message est mis à jour 
            (Pourcentage du nombre maximal d'époque)
        """
        self.mlp=mlp
        self.message_frequency = message_frequency

    def train(self,train_data, train_labels, learning_rate, max_epoch, batch_size=128, decay=0.1, stopping_rule=None, monitoring_data=None):
        """
        Entraînement le modèle. La méthode doit être hérité d'un modèle 
        pour être utilisé.

        Parameters
        ---------
        train_data: ndarray
            Matrice d'exemples de format (n,d) où n est le nombre d'exemple et 
            d la dimension
        train_labels: ndarray
            Vecteur de cibles, de format n
        learning_rate: float
            taux d'apprentissage pour l'entraînement
        max_epoch: int
            nombre maximal d'époque pendant l'entraînement
        batch_size: int, default is 128
            nombre d'exemples par mini-batch
        stopping_rule: object, default to None
            object avec méthode __call__ qui prend en paramètre train_data et 
            train_labels et renvoie True si le critère d'arrêt est atteint,
            False sinon
        monitoring_data: dict
            dictionnaire d'ensemble de données pour le monitoring, chaque
            valeur (value) doit être un tuple (data, labels)
        """

        assert hasattr(self.mlp, "update"), ("Le modèle n'a pas de méthode "
            "'update' pour mettre à jour les poids")

        assert hasattr(self.mlp, "compute_error"), ("Le modèle n'a pas de méthode "
            "'compute_error' pour calculer le coût sur un ensemble")

        # Engregistre la valeur du taux d'apprentissage pour l'appel de la 
        # fonction self.update (au cas où le taux d'apprentissage serait 
        # changeant d'une époque à l'autre)
        self.mlp.learning_rate.set_value(learning_rate)

        data_keys = [u"train"]
        if monitoring_data is not None:
            data_keys += monitoring_data.keys()

        self.epochs = []
        self.cost_curves = OrderedDict((name, []) for name in data_keys)
        self.error_curves = OrderedDict((name, []) for name in data_keys)

        last_msg_epoch = 0
    
        for epoch in xrange(max_epoch):

            for mini_batch in xrange(0, train_data.shape[0], batch_size):
                if train_labels is not None:
                    self.mlp.update(train_data[mini_batch:mini_batch+batch_size], train_labels[mini_batch:mini_batch+batch_size])
                
                else:
                    self.mlp.update(train_data[mini_batch:mini_batch+batch_size])

            learning= self.mlp.decay_learning(epoch, decay)

            # Calcule les coûts et pertes et mets à jour le message
            if (epoch-last_msg_epoch)/float(max_epoch) > self.message_frequency:

                error, cost, errors, costs = self._compute_monitoring(
                        train_data, train_labels, monitoring_data)

                for name in errors.keys():
                    self.cost_curves[name].append(costs[name])
                    self.error_curves[name].append(errors[name])

                self.epochs.append(epoch)

                print "\r%3d%% : époque %d : perte = %f, learning_rate =%f" % \
                    (int(100*epoch/float(max_epoch)), epoch, cost, learning),

                last_msg_epoch = epoch

        # change de ligne pour compenser le signe '\r'
        print ""

        return self.mlp.get_first_feature()

    def _compute_monitoring(self, train_data, train_labels, monitoring_data):

        errors = {}
        costs = {}
        error = self.mlp.compute_error(train_data, train_labels)
        cost = self.mlp.compute_cost(train_data, train_labels)

        errors[u"train"] = error
        costs[u"train"] = cost

        if monitoring_data is not None:
            for name, [data, labels] in monitoring_data.items():
                errors[name] = self.mlp.compute_error(data, labels)
                costs[name] = self.mlp.compute_cost(data, labels)
            
        return error, cost, errors, costs



class NNLayer(object):
    """
    Couche d'un réseau de neurones feedforward

    Attributes
    ---------
    non_linearities: dict
        dictionnaire de non-linéarités (peut être augmenté avec tanh et relu)
    activation: object
        non-linéarité de la couche. La fonction est extraite du dictionnaire
        de non-linéarités
    W: theano.shared
        paramètres W de la couche, enregistré comme variable «partagée» theano
    b: theano.shard
        paramètres b de la couche, enregistré comme variable «partagée» theano
    params: list
        liste des paramètres W et b de la couche

    Methods
    -------
    fprop(state_below) 
        Applique la transformation linéaire puis la non-linéarité.
    get_l1()
        calcule le coût de norme L1 pour W
    get_l2()
        calcule le coût de norme L2 pour W
    """

    non_linearities = {
        "linear": lambda input: input, 
        "sigmoid": lambda input: T.nnet.sigmoid(input), 
        "softmax": lambda input: T.nnet.softmax(input),
        "tanh": lambda input: T.tanh(input),
        "rl" : lambda input : (input>=0.0)*input,
        "relu" :  lambda input :  T.maximum(0.0, input)
    }
    
    def __init__(self, name, n_in, n_out, activation, rng=None,  W=None, b=None):
        """
        Parameters
        ----------
        name: string
            Nom de la couche. Ce nom sera présent dans le nom des 
            variables W et b et sera pratique pour le débugage du graphe
            theano
        n_in: int
            nombre d'unité en entrée
        n_out: int
            le nombre d'unité en sortie
        activation: string
            le nom de la non-linéarité. Doit être présent comme clé
            dans le dictionnaire des non-linéarités (non_linearities)
        rng: numpy.random.RandomState or None
            Un objet pour échantillioner les valeurs initiales de W
        """

        assert activation in self.non_linearities.keys(), \
            "La non-linéarité n'est pas supportée : %s" % str(activation)

        assert isinstance(n_in, int) and n_in > 0 and \
               isinstance(n_out, int) and n_out > 0

        if rng is None:
            # Crée un générateur de nombre aléatoire avec un germe précis
            rng = np.random.RandomState([2014, 12, 15])

        irange = 1. /np.sqrt(n_in)
       
        if W is None:
            W = np.asarray(rng.uniform(-irange, irange, size=(n_in, n_out)),
                       dtype=theano.config.floatX)
            W = theano.shared(W, name = "%s_W" % name)

        if b is None:
            b = np.asarray(np.zeros(n_out), 
                       dtype=theano.config.floatX)
        
            b = theano.shared(b, name = "%s_b" % name)

        self.W=W
        self.b=b
        self.rng=rng
        self.params = [self.W, self.b]

        self.activation = self.non_linearities[activation]


    def fprop(self, state_below):
        """
        Calcul la phase de propagation avant; transformation linéare et 
        non-linéarité.

        Parameters:
        state_below: theano.Variable
            Variable theano, peut être une entrée X où la sortie d'une
            couche précédente

        Returns
        -------
        theano.Variable
            Retourne une variable theano de format (batch_size, n_out)
        """
        return self.activation(T.dot(state_below, self.W) + self.b)


    def get_l1(self):
        """
        Calcule le coût de norme L1 pour W
 
        Returns
        -------
        theano.Scalar
            Retounr un scalaire theano
        """
        return T.abs_(self.W).sum()


    def get_l2(self):
        """
        Calcule le coût de norme L2 pour W
 
        Returns
        -------
        theano.Scalar
            Retounr un scalaire theano
        """
        return T.sqr(self.W).sum()



class DropNNLayer(NNLayer):
    """
    Couche d'un réseau de neurones feedforward avec dropout

    Attributes
    ---------
    non_linearities: dict
        dictionnaire de non-linéarités (peut être augmenté avec tanh et relu)
    activation: object
        non-linéarité de la couche. La fonction est extraite du dictionnaire
        de non-linéarités
    W: theano.shared
        paramètres W de la couche, enregistré comme variable «partagée» theano
    b: theano.shard
        paramètres b de la couche, enregistré comme variable «partagée» theano
    params: list
        liste des paramètres W et b de la couche

    Methods
    -------
    fprop(state_below) 
        Applique la transformation linéaire puis la non-linéarité.
    get_l1()
        calcule le coût de norme L1 pour W
    get_l2()
        calcule le coût de norme L2 pour W
    """


    def __init__(self, name, n_in, n_out, activation, p=0, rng=None,  W=None, b=None):
        super(DropNNLayer, self).__init__(name, n_in, n_out, activation, rng,  W, b)
        self.p= p



    def fprop(self, state_below):
        """
        Calcul la phase de propagation avant; transformation linéare et 
        non-linéarité.

        Parameters:
        state_below: theano.Variable
            Variable theano, peut être une entrée X où la sortie d'une
            couche précédente

        Returns
        -------
        theano.Variable
            Retourne une variable theano de format (batch_size, n_out)
        """
        return _dropout_layer(super(DropNNLayer, self).fprop(state_below), self.p, self.rng)



class MLP():
    """
    Réseau de neurones de type feedforward

    Attributes
    ----------
    n_in: int 
        Nombre d'unité en entrée
    n_hids: list of int
        Nombre d'unité cachée pour chaque couche cachée
        (une liste vide donne un modèle de régression logistique)
    n_out: in
        Nombre d'unité de sortie (nombre de classes)
    non_linearities: list of int or string
        Non-linéarité des couches cachées. Tous identiques si seulement 
        défini par une string et non une liste
    l1: list of float or float
        Coût de norme L1 appliqué aux W
    l2:
        Coût de norme L2 appliqué aux W
    layers: list of object
        Couche du réseau, comprend toutes les couches cachées et la couche
        de sortie
    params: list of theano.shared
        Liste de variable «partagée» theano représentant les W et b du réseau
    rng: numpy.random.RandomState or None
        Un objet pour échantillioner les valeurs initiales de W

    Methods
    -------

    compute_predictions(test_x)
        Calcule les prédictions du modèle du l'ensemble "test_x"
    compute_cost(data, labels)
        Calcule la fonction de perte sur l'ensemble "data" avec les 
        cibles "labels"
    compute_error(data, labels)
        Calcule l'erreur de classification sur l'ensemble "data" avec les 
        cibles "labels"
    """

    def __init__(self, n_in, n_hids, n_out, non_linearities, dropout, l1=0., l2=0., rng=None, use_dropout=True):
        """
        Parameters
        ----------
        n_in: int 
            Nombre d'unité en entrée
        n_hids: list of int
            Nombre d'unité cachée pour chaque couche cachée
            (une liste vide donne un modèle de régression logistique)
        n_out: in
            Nombre d'unité de sortie (nombre de classes)
        non_linearities: list of int or string
            Non-linéarité des couches cachées. Tous identiques si seulement 
            défini par une string et non une liste

        dropout: list of float or float
            Dropout on each layer
        l1: list of float or float
            Coût de norme L1 appliqué aux W
        l2:
            Coût de norme L2 appliqué aux W
        rng: numpy.random.RandomState or None
            Un objet pour échantillioner les valeurs initiales de W
        """

        if rng is None:
            # Crée un générateur de nombre aléatoire avec un germe précis
            self.rng = np.random.RandomState([2014, 10, 26])

        if isinstance(non_linearities, str):
            non_linearities = [non_linearities] * len(n_hids)

        assert len(non_linearities) == len(n_hids), \
            ("Nombre de non-linéarité inégale au nombre de couches cachées : "
             "%d vs %d" % (len(non_linearities), len(n_hids)))

        if isinstance(l1, float):
            l1 = [l1] * (len(n_hids) + 1)
        
        if isinstance(l2, float):
            l2 = [l2] * (len(n_hids) + 1)


        if isinstance(dropout, float):
            dropout = [dropout] * (len(n_hids)+1)
        
        assert len(l1) == len(n_hids) + 1
        assert len(l2) == len(n_hids) + 1
        assert len(dropout) == len(n_hids)+1
        for p in dropout:
            assert isinstance(p, float) and 0<=p and p<=1, "les valeurs de dropout doivent etre des probabilites"
        for l in l1:
            assert isinstance(l, float) and l >= 0, "l1 < 0 n'a pas de sens!"
        for l in l2:
            assert isinstance(l, float) and l >= 0, "l2 < 0 n'a pas de sens!"

        self.layers = [] #layers pour dropout
        self.predict_layers=[] #layers pour la prediction. Ici, on garde toutes les entrées. 
        #Seule les poids sont multipliés par (1-p)
        # Crée chacune des couches du réseaux
        for i, [layer_n_in, layer_n_out, layer_nonlin, drop_p] in \
                enumerate(itertools.izip([n_in] + n_hids, n_hids, 
                                         non_linearities, dropout[1:])):
            layer = DropNNLayer("h_%d" % i, layer_n_in, layer_n_out, 
                                              layer_nonlin, drop_p, self.rng)

            pred_layer = NNLayer("hpred_%d" % i, layer_n_in, layer_n_out, 
                                              layer_nonlin, self.rng, W=layer.W*(1-drop_p), b=layer.b)

            self.layers.append(layer)
            self.predict_layers.append(pred_layer)

        # Crée la dernière couche qui est toujours présente peut importe les 
        # couches cachées
        last_layer_in = n_hids[-1] if n_hids else n_in
        last_layer = NNLayer("y", last_layer_in, n_out, "softmax", rng)
        self.layers.append(last_layer)

        # also add an output layer for the prediction layer
        last_pred_layer= NNLayer("ypred", n_hids[-1] if n_hids else n_in,  n_out, "softmax", rng,  W=last_layer.W*(1-drop_p), b=last_layer.b)
        self.predict_layers.append(last_pred_layer)

        self.dropout_input= dropout[0]
        self.n_in = n_in
        self.n_hids = n_hids
        self.n_out = n_out
        self.rng = rng
        self.non_linearities = non_linearities
        self.l1 = l1
        self.l2 = l2
        self.use_dropout=use_dropout
        
        self._build_theano_graph()

    @property
    def params(self):
        """ 
        Liste de variable «partagée» theano représentant les W et b du réseau
        """

        return sum((layer.params for layer in self.layers), []) 

    def _build_theano_graph(self):
        """
        Construit le modèle et compile les fonctions
        """
        
        X = T.matrix()
        y = T.ivector()
        epoch=T.scalar()
        decay=T.scalar()

        # La valeur 0 est donné pour s'assurer qu'elle sera changé à 
        # l'entraînement. Si elle n'est pas changé, ça ne fonctionnera pas.
        self.learning_rate = theano.shared(
                np.cast[theano.config.floatX](0.), 
                name="learning_rate")

        state_below = _dropout_layer(X, self.dropout_input, self.rng)
        pred_state_below = X

        for layer in self.layers:
            state_below = layer.fprop(state_below)

        p_y_given_x = state_below

        for pred_layer in self.predict_layers:
            pred_state_below = pred_layer.fprop(pred_state_below)

        pred_p_y_given_x= pred_state_below

        y_pred = T.argmax(p_y_given_x, axis=1)
        
        test_y_pred = T.argmax(pred_p_y_given_x, axis=1)

        pred_cost= -T.mean(T.log(pred_p_y_given_x)[T.arange(y.shape[0]), y])

        cost = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])

        # on itère sur toutes les couches pour calculer le coût pour 
        # tous les W
        for l1, layer in zip(self.l1, self.layers):
            if l1 > 0.:
                cost += l1*layer.get_l1()
                pred_cost+=l1*layer.get_l1()

        for l2, layer in zip(self.l2, self.layers):
            if l2 > 0.:
                cost += l2*layer.get_l2()
                pred_cost+=l2*layer.get_l2()

        # Le gradient est calculé par rapport à tout les paramètres
        # grads est une liste contenant le gradient par rapport à 
        # chaque paramètre, dans le même ordre que dans self.params
        grads = T.grad(cost if self.use_dropout else pred_cost, self.params) #calcul du gradient uniquement avec les données de dropout

        updates = OrderedDict()

        # On itère sur self.params et grads en même temps 
        # Les paramètres sont tous mis à jour de la même façon
        for param, grad in itertools.izip(self.params, grads):
            updates[param] = param - self.learning_rate * grad

        # La fonction update fera la propagation avant puis la 
        # rétro-propagation pour mettre à jour les paramètres
        self.update = function([X, y], y_pred if self.use_dropout else test_y_pred, updates=updates)

        self.decay_learning_rate = function(inputs=[epoch, decay], outputs=self.learning_rate,  updates={self.learning_rate: self.learning_rate *((1+epoch*decay)/(1+ decay*(epoch+1)))})

        # une fonction qui renvoie les prédictions du modèle", la prediction se fait toujours avec les layers qui ne sont pas issu de dropout
        self.predict = function([X], test_y_pred)

        # une fonction qui renvoie le taux d'erreur du modèle",
        self.compute_error_rate = function([X, y], T.mean(T.neq(test_y_pred, y)))

        # une fonction qui renvoie le résultat de la fonction de perte
        self.compute_cost = function([X, y], cost if self.use_dropout else pred_cost)


    def compute_predictions(self, test_x):
        """
        Calcule les prédictions du modèle du l'ensemble "test_x"
        """
        return self.predict(test_x)

    def compute_cost(self, data, labels):
        """
        Calcule la fonction de perte sur l'ensemble "data" avec les 
        """
        return float(self.compute_cost(data, labels))

    def compute_error(self, data, labels): 
        """
        Calcule l'erreur de classification sur l'ensemble "data" avec les 
        """
        return float(self.compute_error_rate(data, labels))*100

    def decay_learning(self, epoch, decay):
        """
        Compute next learning value
        """
        return self.decay_learning_rate(epoch, decay)

    def get_first_feature(self):
        return self.layers[0].W.get_value()


def plot_training_curves(epochs, learning_curves, title, ylabel,  xlabel=u"Époques", xlog=False, ylog=False, path=None):
    """
    Parameters
    ----------
    epochs: list
        Liste représentant les époques
    learning_curves: dict
        Dictionnaire de courbe. Chaque clé sera utilisé pour identifier la 
        courbe dans la légende
    title: string
        Titre du graphique
    ylabel: string
        Nom de l'axe y
    xlabel: string
        Nom de l'axe x
    xlog: bool, default to False
        L'axe x est affiché sous format logarithmique si True
    ylog: bool, default to False
        L'axe y est affiché sous format logarithmique si True
    """

    figure = plt.figure()#figsize=(8,6))
    axes = plt.subplot(111)

    handlers = []

    if isinstance(learning_curves, dict):
        for name, curve in learning_curves.items():
            handler = axes.plot(epochs, curve)[0]
            handlers.append(handler)
    else :
        axes.plot(epochs, learning_curves)[0]

    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    if xlog:
        axes.set_xscale('log')
    if ylog:
        axes.set_yscale('log')

    if isinstance(learning_curves, dict):
        axes.legend(handlers, learning_curves.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.title(title)

    if path is None:
        path= "".join(title.split())
    plt.savefig(path, format='eps', dpi=600)


if __name__ == '__main__':
    
    #load data
    path = "data/"
    train_x = np.loadtxt(path+"train_images.txt", delimiter=",")
    train_y = np.loadtxt(path+"train_labels.txt", delimiter=",").argmax(1)
    test_x= np.loadtxt(path+"test_images.txt", delimiter=",")
    test_y = np.loadtxt(path+"test_labels.txt", delimiter=",").argmax(1)
    data = [(train_x, train_y), (test_x, test_y)]
    data= cast_data(data)
    (train_x, train_y), (test_x, test_y) = data
    #parameters
    activation='relu'
    test_txt="test"
    train_txt="train"
    l2_reg=  0.01
    learning_rate=1
    learning_decay=0.1
    max_epoch=100
    batch_size=100
    monit={test_txt: (test_x, test_y)}
    n_in= train_x.shape[1]
    n_out=np.unique(train_y).shape[0]
    n_hid=[512]
    drop = [0., 0.5] # not dropout on input layer

    test_error={}
    train_error={}
    sgd=SGDTraining(MLP(n_in, n_hid, n_out, activation, drop))
    sgd.train(train_x, train_y, learning_rate, max_epoch, batch_size, decay=learning_decay, monitoring_data=monit)
    test_error= sgd.error_curves[test_txt]
    train_error = sgd.error_curves[train_txt]
    epochs=sgd.epochs
    plot_training_curves(epochs, train_error, title=u"Courbe d'apprentissage %s - \nErreur de prediction sur train", ylabel="Erreur", path="train.eps")
    plot_training_curves(epochs, test_error, title=u"Courbe d'apprentissage %s - \nErreur de prediction sur test", ylabel="Erreur", path="test.eps")
