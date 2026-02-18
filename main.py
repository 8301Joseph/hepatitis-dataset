from sklearn.model_selection import train_test_split
from data_prep import load_and_clean_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# load processed data
X, y = load_and_clean_data()

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


model = Pipeline([
    ("scale", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000))
])

model.fit(X_train, y_train)
print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))