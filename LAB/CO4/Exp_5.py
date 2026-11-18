import numpy as np
from sklearn.decomposition import FastICA, DictionaryLearning
rng = np.random.default_rng(42)

# ---------- ICA: separate mixed non-Gaussian signals ----------
t = np.linspace(0, 8, 2000)
S = np.c_[np.sign(np.sin(3*t)), np.sin(5*t)] + 0.05*rng.standard_normal((2000,2))  # true sources
A = np.array([[1, 0.5],[0.4, 1]])                                                  # mixing matrix
mixed = S @ A.T                                                                    # observed mixtures

ica = FastICA(n_components=2, random_state=0)
S_hat = ica.fit_transform(mixed)                                                   # estimated sources
corr = np.abs(np.corrcoef(S.T, S_hat.T)[:2, 2:])
print("|corr(true, estimated)|:\n", corr.round(3))

# ---------- Sparse coding: min ||x - Dh||² + λ||h||₁ ----------
L = np.array([[0.8,0.1],[0.7,0.2],[0.2,0.8],[0.1,0.9]])
psi = 1 - (L**2).sum(axis=1)
X = rng.standard_normal((500,2)) @ L.T + rng.standard_normal((500,4))*np.sqrt(psi)

dl = DictionaryLearning(n_components=6, alpha=0.5,
                        transform_algorithm="lasso_lars", transform_alpha=0.5,
                        max_iter=200, random_state=0)
H = dl.fit_transform(X)                      # sparse codes h
print("\nFraction of zeros in code :", round((H == 0).mean(), 3))
print("Reconstruction MSE        :", round(((X - H @ dl.components_)**2).mean(), 3))
print("Dictionary D shape        :", dl.components_.shape)
