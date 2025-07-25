import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae

df = pd.read_csv(r'C:\Users\HP\Desktop\python\Weather_prediction\testset.csv')

# print(df.head())
date = "%d%m%Y-%I:%M"

df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
# df = pd.to_datetime(df["datetime_utc"], format = date)

df['year'] = df['datetime_utc'].dt.year
df["month"] = df["datetime_utc"].dt.month
df["day"] = df["datetime_utc"].dt.day

# print(df.head())
# print(df.isnull().sum())

# print(df.duplicated())
# df.info()
# # Categorical columns
# cat_col = [col for col in df.columns if df[col].dtype == 'object']
# print('Categorical columns :',cat_col)
# # Numerical columns
# num_col = [col for col in df.columns if df[col].dtype != 'object']
# print('Numerical columns :',num_col)
# print(df[cat_col].nunique())
# df[''].unique()[:50]
df1 = df.drop(columns=['condition', 'wind_direction'])
# print(df1.shape)
# newdf = df1.isnull()
# print(newdf)
# print(round((df1.isnull().sum()/df1.shape[0])*100,2))
df['pressure'] = df['pressure'].fillna(df['pressure'].mean())
df['humidity'] = df['humidity'].fillna(df['humidity'].mean())
df['wind_speed'] = df['wind_speed'].fillna(df['wind_speed'].mean())
df['temp'] = df['temp'].fillna(df['temp'].mean())
# print(df.isnull().sum())

# print(df.head())


features = df[['day', 'month', 'year', 'pressure', 'humidity', 'wind_speed']]
target = df['temp']

X= features
Y = target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=53)

model = RandomForestRegressor()

model.fit(X_train,Y_train)

pred = model.predict(X_test)
# print(pred)

error = mae(Y_test, pred)
print("Mean absolute error : " + str(error))



day = int(input("Enter the day of the year (1-365): "))
month = int(input("Enter the month (1-12): "))
year = int(input("Enter the year: "))
pressure = float(input("Enter the atmospheric pressure: "))
humidity = float(input("Enter the humidity in (%): "))
wind_speed = float(input("Enter the wind speed in (m/s): "))


future_data = pd.DataFrame({
    'day': [day],
    'month': [month],
    'year': [year],
    'pressure': [pressure],
    'humidity': [humidity],
    'wind_speed': [wind_speed]
})

temperature = model.predict(future_data)

print(f"Predicted Temperature: {temperature[0]:.2f} C")