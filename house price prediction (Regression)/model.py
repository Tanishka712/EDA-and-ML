#%%
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
#%%
df= pd.read_csv(r"C:\Users\taish\OneDrive\Documents\mypythonprojects\house price prediction\data\housing.csv")
df.info()
# %%
df.isnull().sum()
df.drop(columns=['total_bedrooms'], inplace=True)
# %%
df['ocean_proximity'].value_counts().plot(kind='bar')
#%%
target = "median_house_value"
numeric_cols = [col for col in df.select_dtypes(include=["int64", "float64"]).columns if col != target]

for col in numeric_cols:
    
    df[f"{col}_bin"] = pd.qcut(df[col], q=10, duplicates='drop')

    grouped = df.groupby(f"{col}_bin")[target].mean().reset_index()

    plt.figure(figsize=(8, 4))
    sns.barplot(data=grouped, x=f"{col}_bin", y=target, palette="viridis")
    plt.title(f"Average {target} by Binned {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    df.drop(columns=[f"{col}_bin"], inplace=True)

# %%
# conversion of categorical data
df = pd.get_dummies(df, drop_first=True).astype(int)
# %%
df.drop(columns=['ocean_proximity_ISLAND'], inplace=True)
# %%
# scaling
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']
# %%
numerical_features_to_scale = X.select_dtypes(include=['number'])
numerical_scaling_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())                    
])
X_scaled_array = numerical_scaling_pipeline.fit_transform(numerical_features_to_scale)
X_scaled_df = pd.DataFrame(X_scaled_array, columns=numerical_features_to_scale.columns)
#%%
X_scaled_df
# %%
# ml training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model= LinearRegression()
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2) Score: {r2:.4f}")
# %%
print(" Actual vs. Predicted Values:")
predictions_df = pd.DataFrame({'Actual': y_test.head, 'Predicted': y_pred})
predictions_df
# %%
