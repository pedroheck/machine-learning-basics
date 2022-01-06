
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Features (1 = yes, 0 = no)

# Does it have long hair?
# Does it have short legs?
# Does it bark?
pig_1 = [0, 1, 0]
pig_2 = [0, 1, 1]
pig_3 = [1, 1, 0]

dog_1 = [0, 1, 1]
dog_2 = [1, 0, 1]
dog_3 = [1, 1, 1]

# Labels: 1 = pig, 0 = dog
train_x = [pig_1, pig_2, pig_3, dog_1, dog_2, dog_3]
train_y = [1, 1, 1, 0, 0, 0] # Labels


model = LinearSVC()
model.fit(train_x, train_y) # Here we're training the model with our training set (including the labels, which is train_y)

mysterious_1 = [1, 1, 1]
mysterious_2 = [1, 1, 0]
mysterious_3 = [0, 1, 1]

test_x = [mysterious_1, mysterious_2, mysterious_3]
test_y = [0, 1, 1]

# We now have a variable called predictions that contains the predictions for the test data.
predictions = model.predict(test_x)


# We can use this to see how well our model is doing.
accuracy = accuracy_score(test_y, predictions)

print("Accuracy: %.2f" % (accuracy * 100) + "%")