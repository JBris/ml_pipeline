"""
Machine Learning Pipeline Custom Estimators

A library of custom estimators for the machine learning pipeline.
"""

##########################################################################################################
### Imports  
##########################################################################################################

# External
from pyod.models.copod import COPOD
from pyod.models.lmdd import LMDD
from pyod.models.loci import LOCI
from pyod.models.loda import LODA
from sklearn.cluster import MiniBatchKMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import GammaRegressor, SGDRegressor, TweedieRegressor
from sklearn.mixture import GaussianMixture as GaussianMixtureBase, BayesianGaussianMixture as BayesianGaussianMixtureBase

##########################################################################################################
### Library  
##########################################################################################################

##########################################################################################################
### Regression  
##########################################################################################################

CUSTOM_REGRESSORS = {
    "gamma": GammaRegressor,
    "gp": GaussianProcessRegressor,
    "sgd": SGDRegressor,
    "tweedie": TweedieRegressor
}

##########################################################################################################
### Classification  
##########################################################################################################

CUSTOM_CLASSIFIERS = {}

##########################################################################################################
### Anomaly Detection  
##########################################################################################################

CUSTOM_ANOMALY_DETECTION = {
    "copod": COPOD,
    "lmdd": LMDD,
    "loci": LOCI,
    "loda": LODA,
}

##########################################################################################################
### Clustering  
##########################################################################################################

class GaussianMixture(GaussianMixtureBase):
    def __init__(self, n_clusters = 4, *, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        super().__init__(
            n_components=n_clusters, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)
        self.n_clusters = n_clusters
        
    def fit(self, X, y=None):
        super().fit(X, y)
        self.labels_ = self.predict(X)

class BayesianGaussianMixture(BayesianGaussianMixtureBase):
    def __init__(self, n_clusters = 4, *, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        super().__init__(
            n_components=n_clusters, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)
        self.n_clusters = n_clusters
        
    def fit(self, X, y=None):
        super().fit(X, y)
        self.labels_ = self.predict(X)

CUSTOM_CLUSTERING = {
    "mbkmeans": MiniBatchKMeans,
    "gmm": GaussianMixture,
    "bgmm": BayesianGaussianMixture
}