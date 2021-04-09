from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pandas import read_csv
import os

# Load data
data_path = os.path.join(os.getcwd(), "blood-pressure.txt")
dataset = read_csv(data_path, delim_whitespace=True)

# We have 30 entries in our dataset and four features. The first feature is the ID of the entry.
# The second feature is always 1. The third feature is the age and the last feature is the blood pressure.
# We will now drop the ID and One feature for now, as this is not important.
dataset = dataset.drop(['ID', 'One'], axis=1)



# Our data
X = dataset[['Age']]
y = dataset[['Pressure']]

regr = LinearRegression()
regr.fit(X, y)

# Plot outputs
plt.xlabel('Age')
plt.ylabel('Pressure')

plt.scatter(X, y,  color='black')
plt.plot(X, regr.predict(X), color='blue')

plt.show()
plt.gcf().clear()

print( 'Predicted blood pressure at 30 y.o.   = ', regr.predict([[30]]) )
print( 'Predicted blood pressure at 35 y.o.   = ', regr.predict([[35]]) )
print( 'Predicted blood pressure at 40 y.o.   = ', regr.predict([[40]]) )
print( 'Predicted blood pressure at 44.5 y.o. = ', regr.predict([[44.5]]) )
print( 'Predicted blood pressure at 71 y.o.   = ', regr.predict([[71]]) )

