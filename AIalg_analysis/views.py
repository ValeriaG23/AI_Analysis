from django.shortcuts import render
from .forms import DatasetUploadForm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import time
from sklearn.preprocessing import LabelEncoder

def home(request):
    return render(request, 'home.html')

def train_models(X_train, y_train):
    models = {
        'J48 (C4.5 Decision Tree)': DecisionTreeClassifier(),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(),
        'Logistic Regression': LogisticRegression()
    }

    results = {}

    for name, model in models.items():
        start_time = time.time()

        model.fit(X_train, y_train)

        if name != 'K-Means Clustering':
            y_pred = model.predict(X_train)

            precision = precision_score(y_train, y_pred, average='weighted', zero_division=1)
            recall = recall_score(y_train, y_pred, average='weighted', zero_division=1)
            f1 = f1_score(y_train, y_pred, average='weighted', zero_division=1)
            accuracy = accuracy_score(y_train, y_pred)
        else:
            precision, recall, f1, accuracy = None, None, None, None

        execution_time = time.time() - start_time

        print(f"{name} - Precision: {precision}, Recall: {recall}, F1: {f1}, Accuracy: {accuracy}, Time: {execution_time}")

        results[name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'execution_time': execution_time
        }

    return results


def preprocess_data(df):
    label_encoder = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = label_encoder.fit_transform(df[column])

    df = df.fillna(df.mean())

    return df

def upload_dataset(request):
    if request.method == 'POST' and request.FILES['dataset']:
        uploaded_file = request.FILES['dataset']
        try:
            df = pd.read_csv(uploaded_file)

            df = preprocess_data(df)

            request.session['dataset'] = df.to_json()

            return render(request, 'upload_success.html', {'dataset': df.head()})
        except Exception as e:
            return render(request, 'upload_failed.html', {'error': str(e)})
    return render(request, 'upload.html', {'form': DatasetUploadForm()})

def evaluate_algorithms(request):
    if 'dataset' not in request.session:
        return render(request, 'upload_failed.html', {'error': 'Nu a fost încărcat niciun dataset.'})

    df = pd.read_json(request.session['dataset'])
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    if y.isnull().any():
        y = y.fillna(y.mode()[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = train_models(X_train, y_train)

    performance_data = {
        'algorithms': list(results.keys()),
        'precision': [f"{results[algo]['precision'] * 100:.2f}%" for algo in results],
        'recall': [f"{results[algo]['recall'] * 100:.2f}%" for algo in results],
        'f1_score': [f"{results[algo]['f1_score'] * 100:.2f}%" for algo in results],
        'accuracy': [f"{results[algo]['accuracy'] * 100:.2f}%" for algo in results],
        'execution_time': [f"{results[algo]['execution_time']:.2f} sec" for algo in results]
    }
    algorithms_results = {}
    for algo in results:
        algorithms_results[algo] = {
            'precision': f"{results[algo]['precision'] * 100:.2f}%",
            'recall': f"{results[algo]['recall'] * 100:.2f}%",
            'f1_score': f"{results[algo]['f1_score'] * 100:.2f}%",
            'accuracy': f"{results[algo]['accuracy'] * 100:.2f}%",
            'execution_time': f"{results[algo]['execution_time']:.2f} sec"
        }

    return render(request, 'evaluate_algorithms.html', {
        'performance_data': algorithms_results
    })
