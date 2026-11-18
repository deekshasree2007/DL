import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(42)

L = np.array([[0.8,0.1],[0.7,0.2],[0.2,0.8],[0.1,0.9]])
psi = 1 - (L**2).sum(axis=1); Psi = np.diag(psi)
Sigma = L @ L.T + Psi
n = 2000
idx = np.tril_indices(4, -1)
rmsr = lambda R: np.sqrt((R[idx]**2).mean())

# ---------- 1. Usefulness: residual matrix and RMSR ----------
X = rng.standard_normal((n,2)) @ L.T + rng.standard_normal((n,4))*np.sqrt(psi)
S = np.cov(X.T)
R = S - Sigma
print("Residual S - (LL'+Psi):\n", R)
print("RMSR:", round(rmsr(R), 4), "| average communality:", (L**2).sum(1).mean().round(2))

# ---------- 2. Limitation: factors really correlated (r = 0.5) ----------
Phi = np.array([[1,0.5],[0.5,1]])
Fc = rng.multivariate_normal([0,0], Phi, size=n)
Xc = Fc @ L.T + rng.standard_normal((n,4))*np.sqrt(psi)
Sc = np.cov(Xc.T)
print("\nCorrelated factors -> RMSR (orthogonal model):", round(rmsr(Sc - Sigma), 4))

# ---------- 3. Improvement: allow correlated factors  Σ = ΛΦΛᵀ + Ψ ----------
Sigma_obl = L @ Phi @ L.T + Psi
print("Correlated factors -> RMSR (oblique model)   :", round(rmsr(Sc - Sigma_obl), 4))

# ---------- 4. Choosing number of factors: scree, parallel analysis, BIC ----------
ev = np.sort(np.linalg.eigvalsh(np.corrcoef(X.T)))[::-1]
sim = np.mean([np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n,4)).T)))[::-1]
               for _ in range(100)], axis=0)
print("\nEigenvalues (data)  :", ev)
print("Eigenvalues (random):", sim)

for k in (1, 2):
    f = FactorAnalysis(n_components=k, random_state=0).fit(X)
    ll = f.score(X) * n
    p = 4*k - k*(k-1)//2 + 4                 # number of free parameters
    print(f"k={k}: logL={ll:.1f}  BIC={-2*ll + p*np.log(n):.1f}")

plt.plot([1,2,3,4], ev, "o-", label="Data")
plt.plot([1,2,3,4], sim, "s--", label="Random (parallel analysis)")
plt.axhline(1, color="grey", lw=.7)
plt.xlabel("Factor"); plt.ylabel("Eigenvalue"); plt.title("Scree plot"); plt.legend(); plt.show()
