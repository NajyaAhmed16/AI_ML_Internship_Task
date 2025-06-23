import pandas as pd

df = pd.read_csv(r"C:\Users\najya\.kaggle\Titanic-Dataset.csv")


df.head()
df.info()
df.describe()
df.isnull().sum()  # shows missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])  # Male=1, Female=0

# One-Hot Encoding
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot
sns.boxplot(x=df['Fare'])
plt.show()

# Removing extreme outliers
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df['Fare'] >= lower) & (df['Fare'] <= upper)]

