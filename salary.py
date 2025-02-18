import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


data = {
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [38000, 43000, 48000, 54000, 59000, 64000, 69000, 74000, 79000, 84000]
}

df = pd.DataFrame(data)


X = df[['Experience']]
y = df['Salary']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title("Experience vs Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Step 9: Make a prediction for a new value
new_experience = np.array([[5]])  # Example: 5 years of experience
predicted_salary = model.predict(new_experience)
print(f"Predicted Salary for 5 years of experience: {predicted_salary[0]}")
