from sklearn.datasets import fetch_openml

data = fetch_openml(name="credit-g", version=1, as_frame=True)
df = data.frame

print(df.shape)
print(df.head())
print(df.dtypes)
print(df["class"].value_counts())
print(df["age"].describe())
print("Moins de 25 ans :", (df["age"] < 25).sum())