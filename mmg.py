import numpy 
import numpy.random
import numpy.linalg
import time
from scipy.misc import logsumexp
np=numpy
logpi2 = np.log(2*numpy.pi)
SMALL = 0.001


def plotGaussian(mean, covariance, c):
    """ Plot a 2d gaussian with given mean and covariance. """
    import matplotlib.pyplot as plt

    t = numpy.arange(-numpy.pi, numpy.pi, 0.01)
    k = len(t)
    x = numpy.sin(t)[:, numpy.newaxis]
    y = numpy.cos(t)[:, numpy.newaxis]

    D, V = numpy.linalg.eigh(covariance)
    A = numpy.real(numpy.dot(V, numpy.diag(numpy.sqrt(D))).T)
    z = numpy.dot(numpy.hstack([x, y]), A)

    plt.hold('on')
    plt.plot(z[:,0]+mean[0], z[:,1]+mean[1], linewidth=2, color=c)
    plt.plot(numpy.array([mean[0]]), numpy.array([mean[1]]), "kx")
    plt.hold('off')
  
class MMG(object):
    """ Mixture of Gaussians.
  
        Use em to train the model, use compute_posteriors to get the posterior 
        probabilities over states given data, use log_probs to get the log-probability that 
        the model assigns to data. 
    """

    def __init__(self, data, n_center, rand_init=False):
        self.data = data.T
        self.dim, self.n= self.data.shape
        self.n_center = n_center
        #Parametres du models :
        #K-vecteur probabilite a priori:  n_center 
        #D vecteur uk, moyenne d'un etat ==> dim*n_center
        #K matrice de covariance D*D pour chaque etat donc n_center*dim**2
        self.numparams = n_center + self.dim*n_center + n_center*(self.dim**2)

        #vecteur de paramettre a optimiser
        self.params = numpy.zeros(self.numparams, 'double')

        #priors p(zk=1 ) pour chaque centre
        self.priors = self.params[:self.n_center]
        self.priors[:] = numpy.ones(self.n_center, 'double')/numpy.float(self.n_center) #priors, chaque centre a la meme probabilite au debut

        #probabilite a posteriori
        self.posteriors = numpy.zeros((self.n_center, self.n), 'double')
        
        #uk vector, moyenne de chaque etat. ==> les dim*n_center variables qui suivent dans les parametres. reshaped correctement
        self.means = self.params[self.n_center:self.n_center+self.dim*self.n_center].reshape(self.dim, self.n_center) #definie a 0 au depart
        
        #matrices de covariances ( n_center * dim *dim)
        self.covs = self.params[-self.dim**2*n_center:].reshape(self.dim, self.dim, self.n_center)
        self.covs[:,:,:] = numpy.zeros((self.dim, self.dim, self.n_center), 'double')
        for s in range(self.n_center):
            self.covs[:, :, s] = numpy.identity(self.dim)

  
        if(rand_init):
            rand_ind=numpy.random.choice(self.n, self.n_center, replace=False)

            self.means=self.data[:, rand_ind]
            #self.means[:,:] = self.data.mean(1)[:, numpy.newaxis] + numpy.random.randn(self.dim, self.n_center)*0.1*self.data.var()
            for s in xrange(self.n_center):
                a = self.data - self.means[:, s][:, numpy.newaxis]
                self.covs[:, :, s] = numpy.dot(a, a.T)
            self.covs /= self.n


    def em(self, numiter=100):
        """Train for numiter EM iterations."""

        lastLogL = -numpy.inf
        for iter in range(numiter):

            #E-Step:
            self.compute_posteriors(self.data)

            #M-Step:
            qsums = self.posteriors.sum(1)
            for s in range(self.n_center):
                self.priors[s] = qsums[s]/self.n
                self.means[:, s] = numpy.sum(self.posteriors[s, :][numpy.newaxis, :]*self.data, 1)/qsums[s]
                self.covs[:, :, s] =  numpy.dot(
                           self.posteriors[s, :][numpy.newaxis, :]*
                              (self.data-self.means[:, s][:, numpy.newaxis]), 
                              (self.data-self.means[:, s][:, numpy.newaxis]).T
                                      ) / qsums[s]

            #threshold eigenvals:
            for s in range(self.n_center):
                D, V = numpy.linalg.eigh(self.covs[:, :, s])
                self.covs[:, :, s] = numpy.dot(numpy.dot(V, numpy.diag(D + SMALL)), V.T)

            newLogL = self.logprob()
            if (newLogL - lastLogL)<0:
                #print 'Likelihood not decreasing anymore....'
                break
            lastLogL = newLogL
        #print 'likelihood :', lastLogL


    def logprob(self):
        return self.logpx(self.data).sum()

    def logpx(self, data):
        log_probs = numpy.zeros((self.n_center, data.shape[1]), 'double')
        #calculons le log de p(x) puis retournons le logsumexp de cette valeur.
        for s in range(self.n_center):
            logc = -0.5*self.dim*logpi2-0.5*numpy.log(numpy.linalg.det(self.covs[:, :, s]))
            cov_inv = numpy.linalg.pinv(self.covs[:, :, s])
            a = data - self.means[:, s][:, numpy.newaxis]
            log_probs[s, :] = -0.5 * numpy.diag(numpy.dot(numpy.dot(a.T, cov_inv), a))+logc+numpy.log(self.priors[s])
        return logsumexp(log_probs, 0)

    
    def compute_posteriors(self, data):
        for s in range(self.n_center):
            logc = -0.5*self.dim*logpi2-0.5*numpy.log(numpy.linalg.det(self.covs[:, :, s]))
            cov_inv = numpy.linalg.pinv(self.covs[:, :, s])
            a = data - self.means[:, s][:, numpy.newaxis]
            self.posteriors[s, :] = -0.5*numpy.diag(numpy.dot(numpy.dot(a.T, cov_inv), a))+logc+numpy.log(self.priors[s])
        self.posteriors -= logsumexp(self.posteriors, 0)
        self.posteriors[:, :] = numpy.exp(self.posteriors)
        return self.posteriors


if __name__ == '__main__':
	import matplotlib.pyplot as plt

	faithful=np.loadtxt('data/faithful.txt')
	colors =np.array(['b','g','r','y','m','c'])
	for k in [2,3,5]:
	    print "n_center = ", k
	    mixgau=MMG(faithful, k, False)
	    mixgau.em(100)
	    print "Likelihood : ", mixgau.logprob()
	    
	for k in [2,3,5]:
	    print "n_center = ", k
	    best_logLk=-np.inf
	    best_model=None
	    
	    # pour chaque nombre de centre, choisir k points au hasard
	    for i in xrange(100):
	        mixgau=MMG(faithful, k, True)
	        mixgau.em(1000)
	        if(mixgau.logprob()>best_logLk):
	            best_model=mixgau
	            best_logLk=mixgau.logprob()
	            
	    plt.scatter(faithful[:,0], faithful[:,1], c=colors[numpy.argmax(best_model.compute_posteriors(best_model.data),0)])
	    plt.title(u'LogLikelihood : %s'%best_logLk)
	    for center in xrange(k):
	        plotGaussian(best_model.means[:,center], best_model.covs[:,:,center], colors[center]);
	
	    plt.savefig('mmg%s.eps'%k, format='eps', dpi=600)