import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('Basics-1-intro-to-sklearn-classification/data/tracking.csv')

x = data[["home", "how_it_works", "contact"]]
y = data["bought"]


# Since train_test_split chooses the sets randomly, we'll set a fixed seed so numbers won't change everytime the code is run 
SEED = 30 # Any number

train_x, test_x, train_y, test_y = train_test_split(x, y, 
                                                        random_state = SEED, test_size=0.25, # Here we're setting the test size to be 25% of the whole dataset, leaving 75% for training
                                                        stratify = y) # Now we're stratifying properly
print("We'll train with %d elements and test with %d elements" % (len(train_x), len(test_x)))

# Creating the model
model = LinearSVC()
model.fit(train_x, train_y)

# Making predictions
predictions = model.predict(test_x)

# Evaluating the model's accuracy
accuracy = accuracy_score(test_y, predictions) * 100

print("The accuracy was %.2f%%" % accuracy)