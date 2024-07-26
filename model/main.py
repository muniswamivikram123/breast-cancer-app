import pandas as pd 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import pickle5 as pickle
def create_model(data):
    X = data.drop(['diagnosis'],axis = 1)
    y = data['diagnosis']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

# train the model
    model = LogisticRegression()
    model.fit(X_train,y_train)

    #test the model
    y_pred = model.predict(X_test)
    print("accuracy score = ",accuracy_score(y_test,y_pred))
    print("classification report = ",classification_report(y_test,y_pred))

    return model,scaler



def get_clean_data():
    data = pd.read_csv("data\data.csv")
    #print(data.head())
    # droping the Unnamed and id because unnamed have Nan value and id is not required
    data = data.drop(['Unnamed: 32','id'], axis=1)
    # now we have to encode the data of diagnosis
    data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
    #print(data)
    #print(data)
    return data

def main():
    data = get_clean_data()
    model,scaler = create_model(data)

    #print(data.head())

    # we want to export the model for tha we want pickele file
    # pip install pickle5
    with open('model/model.pkl', 'wb') as f: # w for write b for binary file,f for file
        pickle.dump(model , f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    main()