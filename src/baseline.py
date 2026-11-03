from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold


SEED = 42

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

print("AUC train :", roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
print("AUC test  :", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
print("AUC validation croisée :", scores.round(3))
print("Moyenne :", scores.mean().round(3), "écart-type :", scores.std().round(3))