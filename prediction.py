import joblib

def predict(data):
    clf = joblib.load('models/rf_model.sav')
    return clf.predict(data)