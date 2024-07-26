import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print("Accuracy score = ", accuracy_score(y_test, y_pred))
    print("Classification report = ", classification_report(y_test, y_pred))

    return model, scaler

def get_clean_data():
    data = pd.read_csv("data/data.csv")
    # Drop the Unnamed column and id because Unnamed has NaN values and id is not required
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    # Encode the diagnosis data
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def main():
    data = get_clean_data()
    model, scaler = create_model(data)

    # Save the model and scaler using joblib
    joblib.dump(model, 'model/model.joblib')
    joblib.dump(scaler, 'model/scaler.joblib')

if __name__ == '__main__':
    main()
