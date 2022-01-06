import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.read_csv("Basics-1-intro-to-sklearn-classification/data/projects.csv")

# Let's change the "unfinished" column to "finished" so it's easier to understand
change = {
    0: 1,
    1: 0
}
data['finished'] = data.unfinished.map(change)

# # Let's see the graph
# sns.scatterplot(x="expected_hours", y="price", hue="finished", data=data)

# # Seeing it seperately
# sns.relplot(x="expected_hours", y="price", hue="finished", col="finished", data=data)

x = data[['expected_hours', 'price']]
y = data['finished']


SEED = 15
np.random.seed(SEED)
raw_train_x, raw_test_x, train_y, test_y = train_test_split(x, y, test_size = 0.25, stratify = y)

print("We will train with %d elements and test with %d elements" % (len(raw_train_x), len(raw_test_x)))

# Rescaling our axis
scaler = StandardScaler()
scaler.fit(raw_train_x)
train_x = scaler.transform(raw_train_x)
test_x = scaler.transform(raw_test_x)

model = SVC()
model.fit(train_x, train_y)
predictions = model.predict(test_x)

accuracy = accuracy_score(test_y, predictions) * 100
print("The model's accuracy was %.2f%%" % accuracy)
