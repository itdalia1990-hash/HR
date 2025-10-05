import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
uploaded = files.upload()  # upload HR.csv

df = pd.read_csv("HR.csv")

print(df.shape)
print(df.columns)
print(df.dtypes.head())
print(df.isnull().sum().head())

df = df.drop_duplicates()
df = df.dropna()

print(df.describe())

mean_age = df["Age"].mean()
median_age = df["Age"].median()
var_age = df["Age"].var()
q_age = df["Age"].quantile([0.2,0.4,0.6,0.8])

print("Mean Age:", mean_age)
print("Median Age:", median_age)
print("Variance Age:", var_age)
print("Quintiles:\n", q_age)

print(df["Department"].value_counts())

corr = df.select_dtypes(include=np.number).corr()
print(corr.head())

plt.figure(figsize=(7,5))
plt.hist(df["Age"], bins=15, color='lightblue', edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(7,5))
sns.boxplot(x="Attrition", y="MonthlyIncome", data=df, palette="pastel")
plt.title("Monthly Income by Attrition")
plt.xlabel("Attrition")
plt.ylabel("Monthly Income")
plt.show()

plt.figure(figsize=(7,5))
sns.scatterplot(x="Age", y="MonthlyIncome", data=df, hue="Attrition", alpha=0.7)
plt.title("Age vs Monthly Income")
plt.xlabel("Age")
plt.ylabel("Monthly Income")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()
