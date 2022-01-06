import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('Basics-1-intro-to-sklearn-classification/data/car-prices.csv')

# Converting yes = 1 and no = 0
change = {
    'yes': 1,
    'no' : 0
}
data['sold'] = data.sold.map(change)

# Now we're going to use the model year to get the car's age
current_year = datetime.today().year
data['models_age'] = current_year - data.model_year

# And now converting miles to km cuz, you know, it's superior
data['km_per_year'] = data.mileage_per_year * 1.60934

# Now let's get rid of the unwanted columns
data = data.drop(columns = ["Unnamed: 0", "mileage_per_year", "model_year"], axis = 1)

x = data[['price', 'models_age', 'km_per_year']]
y = data['sold']

SEED = 15
np.random.seed(SEED)
raw_train_x, raw_test_x, train_y, test_y = train_test_split(x, y, test_size = 0.25, stratify = y)
print("We will train with %d elements and test with %d elements" % (len(raw_train_x), len(raw_test_x)))

model = DecisionTreeClassifier(max_depth=2) # Max depth = 2 is to the graph is't stupidously big
model.fit(raw_train_x, train_y)
predictions = model.predict(raw_test_x)

accuracy = accuracy_score(test_y, predictions) * 100
print("The model's accuracy was %.2f%%" % accuracy)
