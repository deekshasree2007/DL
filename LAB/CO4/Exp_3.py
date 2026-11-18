import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis

L = np.array([[0.8,0.1],[0.7,0.2],[0.2,0.8],[0.1,0.9]])
vars_ = ["X1 Purchase frequency","X2 Monthly spending","X3 Product variety","X4 Website visits"]

communality = (L**2).sum(axis=1)       # λ1² + λ2²
unique = 1 - communality               # unique variance (standardised variables)

print(f"{'Variable':25s}{'F1':>5s}{'F2':>5s}  Strongest  Comm.  Unique")
for v, row, c, u in zip(vars_, L, communality, unique):
    print(f"{v:25s}{row[0]:5.1f}{row[1]:5.1f}  F{row.argmax()+1} ({row.max():.1f})  {c:5.2f}  {u:5.2f}")
print("Average communality:", communality.mean().round(2))

# Check: recover the loadings from simulated data
rng = np.random.default_rng(42)
n = 2000
X = rng.standard_normal((n,2)) @ L.T + rng.standard_normal((n,4))*np.sqrt(unique)
fa = FactorAnalysis(n_components=2, rotation="varimax", random_state=0).fit(X)
print("\nEstimated loadings (order/sign of factors may differ):\n", fa.components_.T.round(3))

# Heat map of loadings
plt.imshow(np.abs(L), cmap="Blues", vmin=0, vmax=1)
plt.xticks([0,1], ["F1","F2"]); plt.yticks(range(4), ["X1","X2","X3","X4"])
for i in range(4):
    for j in range(2):
        plt.text(j, i, L[i,j], ha="center", va="center")
plt.title("Loading matrix"); plt.colorbar(); plt.show()
