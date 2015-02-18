import numpy
import time
np=numpy

class PCA(object):
    
    def __init__(self, data):

        # data : n x d
    
        # Centrer les donnees
        self.mean_data = np.mean(data, axis=0)
        self.c_data = data - self.mean_data
        
        # Calculer la matrice de covariance
        self.cov = (1./len(data)) * np.dot(self.c_data.T, self.c_data)
        
        #Calculer les valeurs et vecteurs propres
        # Chaque colonne de eigvec est un vecteur propre
        # eigval : d
        # eigvec : d x d
        eigval, eigvec = np.linalg.eigh(self.cov) # Each column is an eigenvector
        
        #trie des valeurs propres
        order = np.argsort(-eigval) 
        self.eigval = eigval[order]
        self.eigvec = eigvec[:,order]

    def backward(self, data=None, M=None, is_centered=True):
        if data == None:
            data = self.c_data
        if M == None:
            M = len(self.eigval)
            print M
        assert M<=len(self.eigval)
        self.U = self.eigvec[:,:M]
        return (np.dot(self.U.T, (data - (1-is_centered)*self.mean_data).T)).T
    
    def forward(self, data=None, M=None, is_centered=True):
        Z = self.backward(data, M, is_centered)
        return (np.dot(self.U, Z.T)).T + (1-is_centered)*self.mean_data



if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    path = "data/"
    train_x = np.loadtxt(path+"train_images.txt", delimiter=",")
    train_y = np.loadtxt(path+"train_labels.txt", delimiter=",").argmax(1)
    test_x= np.loadtxt(path+"test_images.txt", delimiter=",")
    test_y = np.loadtxt(path+"test_labels.txt", delimiter=",").argmax(1)
    which_classes = np.unique(train_y.tolist())# On utilise seulement les classes 0 et 1, pas mini-mnist au complet
    which_indices = np.nonzero(np.in1d(train_y, which_classes))
    mnist_data = train_x[which_indices]
    mnist_labels = train_y[which_indices]
    mnist_pca = PCA(mnist_data)
    mnist_Z = mnist_pca.backward(M=2)
    plt.scatter(mnist_Z[:,0], mnist_Z[:,1], c = mnist_labels, marker = '.', s=100)
    plt.title('All train data together')
    plt.savefig('traindata.eps', format='eps', dpi=600)

    print "Visualation des donnees de test en 2D par PCA"
    which_indices = np.nonzero(np.in1d(test_y, which_classes[1]))
    test_data = test_x[which_indices]
    test_labels = test_y[which_indices]
    Z = mnist_pca.backward(test_data, M=2)
    plt.scatter( Z[:,0], Z[:,1], marker = '.', s=75)
    plt.title(u'Test data, classe 1');
    plt.savefig('class1data.eps', format='eps', dpi=600)
