import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt

data_set = pd.read_csv("C:\\Users\\ASUS\\Downloads\\archive.zip")
X = data_set.iloc[:, [1, 2, 3]].values  # Features
Y = data_set.iloc[:, 7].values  # Target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifiers = [
    ('SVC', SVC(kernel='rbf', random_state=0)),
    ('Decision Tree', DecisionTreeClassifier(random_state=0)),
    ('KNN', KNeighborsClassifier(n_neighbors=3))
]

def evaluate_model(classifier, X_train, Y_train, X_test, Y_test):
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    return {
        'Accuracy': accuracy_score(Y_test, Y_pred),
        'F1 Score': f1_score(Y_test, Y_pred, average='macro', zero_division=1),
        'Recall': recall_score(Y_test, Y_pred, average='macro', zero_division=1),
        'Precision': precision_score(Y_test, Y_pred, average='macro', zero_division=1)
    }

results = {name: evaluate_model(model, X_train, Y_train, X_test, Y_test) for name, model in classifiers}

for name, metrics_values in results.items():
    print(f"{name} - Accuracy: {metrics_values['Accuracy']}")
    print(f"  F1 Score: {metrics_values['F1 Score']}")
    print(f"  Recall: {metrics_values['Recall']}")
    print(f"  Precision: {metrics_values['Precision']}")
    print()

metrics = ['Accuracy', 'F1 Score', 'Recall', 'Precision']
data = np.array([list(metrics_values.values()) for metrics_values in results.values()])

fig, ax = plt.subplots()
x = np.arange(len(metrics))
width = 0.2

for i, (name, _) in enumerate(classifiers):
    ax.bar(x + i * width - width, data[i], width, label=name)

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Algorithms Performance')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

plt.show()
