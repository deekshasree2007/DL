import numpy as np
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.model_selection import train_test_split
np.set_printoptions(precision=3, suppress=True)

# --- simulate data from the factor model ---
rng = np.random.default_rng(42)
L = np.array([[0.8,0.1],[0.7,0.2],[0.2,0.8],[0.1,0.9]])
psi = 1 - (L**2).sum(axis=1)
X = rng.standard_normal((2000,2)) @ L.T + rng.standard_normal((2000,4))*np.sqrt(psi)
Xtr, Xte = train_test_split(X, test_size=0.3, random_state=0)

# --- Factor Analysis: diagonal noise Ψ (a separate variance for each variable) ---
fa = FactorAnalysis(n_components=2, rotation="varimax", random_state=0).fit(Xtr)
print("FA loadings:\n", fa.components_.T)
print("FA noise variances (Psi):", fa.noise_variance_)

# --- PPCA: isotropic noise σ²I (sklearn's PCA uses the PPCA likelihood) ---
pp = PCA(n_components=2).fit(Xtr)
print("PPCA noise variance (sigma^2):", round(pp.noise_variance_, 3))

# --- Compare average log-likelihood on unseen customers ---
print("Held-out log-lik  FA  :", round(fa.score(Xte), 3))
print("Held-out log-lik  PPCA:", round(pp.score(Xte), 3))

# Model covariances
print("\nFA   cov = LL' + Psi:\n", fa.get_covariance())
print("PPCA cov = WW' + s2*I:\n", pp.get_covariance())
