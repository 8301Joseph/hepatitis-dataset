from sklearn.model_selection import train_test_split
from data_prep import load_and_clean_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

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
    ("scale", StandardScaler()), #standardise values
    ("lr", LogisticRegression(max_iter=2000)) #predicts probabilities for a binary outcome
])

model.fit(X_train, y_train)
print("Train accuracy:", model.score(X_train, y_train)) # ~ 0.911
print("Test accuracy:", model.score(X_test, y_test)) # ~ 0.839
# not overfitting



#rank IVs by importance
coefs = model.named_steps["lr"].coef_[0] #stores coeffs from fitted logistic regression model
importance = pd.Series(np.abs(coefs), index=X.columns).sort_values(ascending=False) #take absolute value of coefs and display as labeled Pandas series, sorted descending order
print(importance.head(10))
# all predictors are standardised --> comparing coeffs is valid

