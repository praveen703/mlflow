import mlflow
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub

dagshub.init(repo_owner='praveen703', repo_name='mlflow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/praveen703/mlflow.mlflow")

#Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

#Spliting the DataSet into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Define the parameters for the random forest model
max_depth = 1

#apply ML FLow

mlflow.set_experiment("isis-dt")

with mlflow.start_run():

    dt = DecisionTreeClassifier(max_depth=max_depth)

    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy',accuracy)

    mlflow.log_param('max_depth', max_depth)

    # Create a confusion matrix plot
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    #Save the plot as an artifact
    plt.savefig("confusion_matrix.png")

    # mlflow code
    mlflow.log_artifact("confusion_matrix.png")

    #mlflow code
    mlflow.log_artifact(__file__)

    mlflow.sklearn.log_model(dt, "decision tree")

    mlflow.set_tag('author','praveen')
    mlflow.set_tag('model','decision tree')

    print('accuracy', accuracy)