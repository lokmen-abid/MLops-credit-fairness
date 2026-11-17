import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate

SEED = 42

# --- Même préparation que baseline.py ---
df = fetch_openml(name="credit-g", version=1, as_frame=True).frame
y = (df["class"] == "good").astype(int)
X = df.drop(columns=["class"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
cat_cols = X.select_dtypes(include=["category", "object"]).columns
num_cols = X.select_dtypes(exclude=["category", "object"]).columns
prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols),
])
model = Pipeline([
    ("prep", prep),
    ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
])
model.fit(X_train, y_train)

# --- Nouveau : analyse par groupe ---
y_pred = model.predict(X_test)  # 1 = accepté, 0 = refusé (seuil 0,5)

group = np.where(X_test["age"] < 25, "moins_25", "25_et_plus")
print(pd.Series(group).value_counts())

mf = MetricFrame(
    metrics={
        "taux_acceptation": selection_rate,
        "taux_bons_acceptes": true_positive_rate,
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=group,
)
print(mf.by_group)

sel = mf.by_group["taux_acceptation"]
tpr = mf.by_group["taux_bons_acceptes"]
print("Disparate impact :", sel["moins_25"] / sel["25_et_plus"])
print("Equal opportunity (rapport) :", tpr["moins_25"] / tpr["25_et_plus"])
print("Equal opportunity (écart)   :", tpr["25_et_plus"] - tpr["moins_25"])

# --- Bootstrap : incertitude due à l'échantillon de test ---
rng = np.random.default_rng(SEED)
n = len(y_test)
y_true = y_test.to_numpy()
is_young = (X_test["age"] < 25).to_numpy()

def fairness_ratios(idx):
    yt, yp, g = y_true[idx], y_pred[idx], is_young[idx]
    di = yp[g].mean() / yp[~g].mean()
    eo = yp[g & (yt == 1)].mean() / yp[~g & (yt == 1)].mean()
    return di, eo

B = 2000
results = np.array([fairness_ratios(rng.integers(0, n, n)) for _ in range(B)])

lo, hi = np.percentile(results, [2.5, 97.5], axis=0)
print(f"DI  : IC 95 % = [{lo[0]:.2f} ; {hi[0]:.2f}]")
print(f"EO  : IC 95 % = [{lo[1]:.2f} ; {hi[1]:.2f}]")

from sklearn.model_selection import cross_val_predict, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
y_pred_cv = cross_val_predict(model, X, y, cv=cv)  # chaque personne prédite sans l'avoir vue

y_all = y.to_numpy()
young_all = (X["age"] < 25).to_numpy()
print("Moins de 25 ans :", young_all.sum(), "sur", len(y_all))

def ratios(yt, yp, g):
    di = yp[g].mean() / yp[~g].mean()
    eo = yp[g & (yt == 1)].mean() / yp[~g & (yt == 1)].mean()
    return di, eo

print("DI, EO (rapport) :", ratios(y_all, y_pred_cv, young_all))

rng = np.random.default_rng(SEED)
n = len(y_all)
res = []
for _ in range(2000):
    i = rng.integers(0, n, n)
    res.append(ratios(y_all[i], y_pred_cv[i], young_all[i]))
lo, hi = np.percentile(np.array(res), [2.5, 97.5], axis=0)
print(f"DI : IC 95 % = [{lo[0]:.2f} ; {hi[0]:.2f}]")
print(f"EO : IC 95 % = [{lo[1]:.2f} ; {hi[1]:.2f}]")