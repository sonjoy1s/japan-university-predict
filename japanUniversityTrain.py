import numpy as np
import pandas as pd

from logging import log
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score



df = pd.read_csv("japan_universities_2026.csv")


X = df.drop("National_Rank",axis=1)
y = df["National_Rank"]


numerical_cols = X.select_dtypes(include=np.number).columns

print("Numerical Columns =",numerical_cols)

catagorical_cols = X.select_dtypes(exclude=np.number).columns

print("Catagorical Columns =",catagorical_cols)


numerical_pipe = Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p)),
        ("scaler",RobustScaler())
    ]
)



catagorical_pipe = Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown='ignore'))
    ]
)



combine_pipe = ColumnTransformer(
    transformers=[
        ("num",numerical_pipe,numerical_cols),
        ("cat",catagorical_pipe,catagorical_cols)
    ]
)


from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2 , random_state=42)


# Base Learners Models
lr = LinearRegression()
dt = DecisionTreeRegressor()
rf = RandomForestRegressor()
gb = GradientBoostingRegressor()
sv = SVR()
knn = KNeighborsRegressor() 



# Stacking regressor 
stacking_rg = StackingRegressor(
    estimators=[
        ("lr",lr),
        ("rf",rf),
        ("gb",gb)
        
]
)


stacking_pipeline = Pipeline(
    [
        ('preprocessor', combine_pipe),
        ('model', stacking_rg)
    ]
)

# Train the stacking pipeline
stacking_pipeline.fit(X_train, y_train)
y_pred = stacking_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Stacking Regressor Performance:")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")

