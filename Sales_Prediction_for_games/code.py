import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


df = pd.read_csv(r'C:\Users\HP\Desktop\python\selling prediction\vgsales.csv')
print(df.isnull().sum())

new_df = df.dropna()

# print(new_df.to_string())
print(new_df.isnull().sum())

df = new_df.drop_duplicates()

# # print(df.head())

label_encoder = LabelEncoder()

df['Genre'] = label_encoder.fit_transform(df['Genre'])
df['Publisher'] = label_encoder.fit_transform(df['Publisher'])

print(df.head())

X = df.drop(columns=['Name', 'Platform', 'Global_Sales'])

Y = df[['Global_Sales']]

# # x = np.array(X['Name'])
# # y = np.array(df['Global_Sales'])


# # plt.scatter(x, y)
# # plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

model = RandomForestRegressor()

model.fit(X_train,Y_train.values.ravel())

predictions = predictions = model.predict(X_test)
print(predictions)

mse = mean_squared_error(Y_test, predictions)
print(f"Mean Squared Error: {mse}")

# df['Year'] = pd.to_datetime(df['Year'])

# print(df.to_string())

# print(df.info()) 

# df = df.dropna(inplace = True)

# print(df.info()) 

# print(df.to_string())
# print(df)
# print(df.shape)
# c = df.describe()
# print(c)

# Y = df['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']
# score = accuracy_score(Y_test,predictions)
# print(predictions)