'''
    Apply Dimensionality Reduction to dataset
'''
from sklearn.decomposition import PCA

pca = PCA(n_components=784, whiten = False, random_state = 2019)

def apply_pca(x, t):
    # reconst = pca.inverse_transform(X_pca)
    pca.fit_transform(t)
    return pca.transform(x)