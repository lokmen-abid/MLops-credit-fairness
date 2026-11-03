from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

SEED = 42

df = fetch_openml(name="credit-g", version=1, as_frame=True).frame

y = (df["class"] == "good").astype(int)
X = df.drop(columns=["class"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print(X_train.shape, X_test.shape)
print("Part de bons payeurs - train :", y_train.mean(), "test :", y_test.mean())
print("Moins de 25 ans dans le test :", (X_test["age"] < 25).sum())