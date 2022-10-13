import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

seed = 42

#Read the dataset.
iris_df = pd.read_csv('inputs/iris.csv')
iris_df.sample(frac = 1, random_state=seed)

#Highlight the features and the target.
features = iris_df.iloc[:, :4]
target = iris_df.iloc[:, 4]

X_train, X_test, y_train, y_test = train_test_split(features, target, random_state = seed)


#Create an instance of the random forest classifier.
clf = RandomForestClassifier(n_estimators = 100)

#train the classifier.
clf.fit(X_train, y_train)

# predict on the test set.
y_pred = clf.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

#Save the model.
joblib.dump(clf, "models/rf_model.sav")
