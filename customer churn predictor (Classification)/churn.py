# %%
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
# %%
df = pd.read_csv(r"C:\Users\taish\OneDrive\Documents\mypythonprojects\customer churn predictor\data\Customer Churn.csv")
df.sample(5)
#%%
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
#%%
df=df.drop(columns=['TotalCharges', 'gender', 'customerID'])
df
#%%
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()
# %%
categorical_cols = df.select_dtypes(include='object').columns
#%%
for col in categorical_cols:
    sns.countplot(y=col, data=df)
    plt.title(f"Distribution of {col}")
    plt.show()
# %%
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
#%%
for col in categorical_cols:
    churn_rate = df.groupby(col)['Churn'].mean().sort_values(ascending=False)
    churn_rate.plot(kind='bar', title=f"Churn Rate by {col}")
    plt.ylabel("Churn Rate")
    plt.show()

# %%
sns.kdeplot(data=df[df['Churn'] == 0], x='tenure', label='Stayed', fill=True)
sns.kdeplot(data=df[df['Churn'] == 1], x='tenure', label='Left (Churned)', fill=True, color='red')
plt.title("Distribution of Tenure by Churn")
plt.xlabel("Tenure (months)")
plt.ylabel("Density")
plt.legend()
plt.show()

#%%
df=df.drop(columns=['MultipleLines','PhoneService'])
#%%
df.info()
# %%
# one-hot encoding

categorical_cols = df.select_dtypes(include='object').columns
categorical_cols
# %%
df_encoded = pd.get_dummies(df, drop_first=True).astype(int)
# %%
df_encoded
# %%
correlation_matrix = df_encoded.corr()
correlation_matrix
# %%
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, linewidths=0.5, center=0)
plt.title("Correlation Heatmap of Encoded Customer Churn Dataset", fontsize=16)
plt.show()
#%%
# feature engineering

# %%
df_encoded['Tenure*MonthlyCharges'] = df_encoded['tenure'] * df_encoded['MonthlyCharges']
df_encoded
# %%
df_encoded['NumServices'] = (
    df_encoded[['OnlineSecurity_Yes', 'TechSupport_Yes', 'StreamingTV_Yes', 
        'StreamingMovies_Yes', 'OnlineBackup_Yes']].sum(axis=1)
)
df_encoded
# %%
df_encoded['HighCharges'] = (df_encoded['MonthlyCharges'] > df_encoded['MonthlyCharges'].median()).astype(int)

df_encoded
# %%
# model training
from sklearn.model_selection import train_test_split
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)  # increase max_iter if needed
model.fit(X_train, y_train)

# %%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# %%
# randon forest
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
# %%
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# %%
# XGboost
from xgboost import XGBClassifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
# %%
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# %%
# saving models
import joblib
joblib.dump(model, 'logistic_model.pkl')

joblib.dump(rf_model, 'random_forest_model.pkl')

joblib.dump(xgb_model, 'xgboost_model.pkl')

# %%
