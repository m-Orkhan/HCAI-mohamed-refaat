import io
import os
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from django.conf import settings
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix
from .forms import CSVUploadForm, FeatureSelectForm, ScatterSelectForm, TrainForm, PredictForm

MODEL_PATH = os.path.join(settings.MEDIA_ROOT, 'trained_model.joblib')

def save_plot(fig, filename):
    path = os.path.join(settings.MEDIA_ROOT, filename)
    fig.savefig(path)
    plt.close(fig)
    return f"{settings.MEDIA_URL}{filename}?v={int(time.time() * 1000)}"

def read_csv(data):
    return pd.read_csv(io.StringIO(data), sep=None, engine='python')

def make_scatter(df, feature_x, feature_y, label, problem_type):
    fig, ax = plt.subplots()
    if problem_type == 'classification':
        for cls in df[label].unique():
            subset = df[df[label] == cls]
            ax.scatter(subset[feature_x], subset[feature_y], label=str(cls))
        ax.legend()
    else:
        if feature_y == label:
            ax.scatter(df[feature_x], df[label], color='steelblue', alpha=0.7, label='Data points')
            m, b = np.polyfit(df[feature_x], df[label], 1)
            x_line = pd.Series(sorted(df[feature_x]))
            ax.plot(x_line, m * x_line + b, color='red', linewidth=2, label='Trend line')
            ax.legend()
        else:
            sc = ax.scatter(df[feature_x], df[feature_y], c=df[label], cmap='viridis')
            fig.colorbar(sc, ax=ax, label=label)
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_title(f'{feature_x} vs {feature_y}')
    return save_plot(fig, 'scatter.png')

def make_histogram(df, feature, label, problem_type):
    fig, ax = plt.subplots()
    if problem_type == 'classification':
        for cls in df[label].unique():
            subset = df[df[label] == cls]
            ax.hist(subset[feature], alpha=0.6, label=str(cls))
        ax.legend()
    else:
        ax.hist(df[feature], bins=20, color='steelblue', alpha=0.8)
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    ax.set_title(f'Histogram of {feature}')
    return save_plot(fig, 'histogram.png')

def make_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    tick_marks = range(len(np.unique(y_test)))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
    return save_plot(fig, 'confusion_matrix.png')

def make_actual_vs_predicted(y_test, y_pred, label):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color='steelblue', alpha=0.7)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=2, label='Perfect prediction')
    ax.set_xlabel(f'Actual {label}')
    ax.set_ylabel(f'Predicted {label}')
    ax.set_title('Actual vs Predicted')
    ax.legend()
    return save_plot(fig, 'actual_vs_predicted.png')

def get_summary(df, label, problem_type):
    features = list(df.columns[:-1])
    stats = df[features].describe().round(2).to_html(classes='table', index=True)
    missing = df.isnull().sum()
    missing_dict = {col: int(missing[col]) for col in df.columns if missing[col] > 0}
    if problem_type == 'classification':
        target_info = f"{df[label].nunique()} classes: {', '.join(str(c) for c in df[label].unique())}"
    else:
        target_info = f"Min: {df[label].min():.2f}, Max: {df[label].max():.2f}, Mean: {df[label].mean():.2f}"
    return stats, missing_dict, target_info

def train_model(df, label, problem_type, model_name, test_size):
    features = list(df.columns[:-1])
    X = df[features].to_numpy()
    y = df[label].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=42
    )

    if problem_type == 'classification':
        model = DecisionTreeClassifier() if model_name == 'decision_tree' else SVC()
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    y_pred = model.predict(X_test)

    if problem_type == 'classification':
        score = round(accuracy_score(y_test, y_pred) * 100, 2)
        score_label = f'Accuracy: {score}%'
        result_plot_url = make_confusion_matrix(y_test, y_pred)
    else:
        score = round(r2_score(y_test, y_pred), 4)
        score_label = f'R² Score: {score}'
        result_plot_url = make_actual_vs_predicted(y_test, y_pred, label)

    return score_label, result_plot_url

def build_context(df, label, problem_type, features,
                  feature_x=None, feature_y=None, hist_feature=None, extra=None):
    stats, missing_dict, target_info = get_summary(df, label, problem_type)

    if feature_x is None:
        feature_x = features[0]
    if feature_y is None:
        feature_y = features[1] if len(features) > 1 else label
    if hist_feature is None:
        hist_feature = features[0]

    scatter_url = make_scatter(df, feature_x, feature_y, label, problem_type)
    hist_url = make_histogram(df, hist_feature, label, problem_type)

    context = {
        'form': CSVUploadForm(),
        'features': features,
        'label': label,
        'shape': df.shape,
        'preview': df.head(10).to_html(classes='table', index=False),
        'stats': stats,
        'missing_dict': missing_dict,
        'target_info': target_info,
        'problem_type': problem_type,
        'scatter_url': scatter_url,
        'hist_url': hist_url,
        'selected_feature': hist_feature,
        'feature_form': FeatureSelectForm(features, initial={'feature': hist_feature}),
        'scatter_form': ScatterSelectForm(
            features, initial={'feature_x': feature_x, 'feature_y': feature_y}
        ),
        'train_form': TrainForm(problem_type),
    }
    if extra:
        context.update(extra)
    return context

def get_selections(request, features, label):
    feature_x = request.session.get('selected_x')
    feature_y = request.session.get('selected_y')
    hist_feature = request.session.get('selected_hist')
    if feature_x not in features:
        feature_x = features[0]
    if feature_y not in list(features) + [label]:
        feature_y = features[1] if len(features) > 1 else label
    if hist_feature not in features:
        hist_feature = features[0]
    return feature_x, feature_y, hist_feature

def index(request):
    if request.method != 'POST':
        return render(request, 'project1/index.html', {'form': CSVUploadForm()})

    context = {'form': CSVUploadForm()}

    if 'file' in request.FILES:
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            decoded = request.FILES['file'].read().decode('utf-8')
            df = read_csv(decoded)
            problem_type = form.cleaned_data['problem_type']

            for key in ('csv_data', 'problem_type', 'selected_hist', 'selected_x',
                        'selected_y', 'features', 'model_name', 'score_label',
                        'result_plot_url'):
                request.session.pop(key, None)

            request.session['csv_data'] = decoded
            request.session['problem_type'] = problem_type

            features = list(df.columns[:-1])
            label = df.columns[-1]
            context = build_context(df, label, problem_type, features)
        return render(request, 'project1/index.html', context)

    csv_data = request.session.get('csv_data')
    problem_type = request.session.get('problem_type')
    if not csv_data:
        return render(request, 'project1/index.html', context)

    df = read_csv(csv_data)
    features = list(df.columns[:-1])
    label = df.columns[-1]
    feature_x, feature_y, hist_feature = get_selections(request, features, label)

    trained_features = request.session.get('features')
    extra = {
        'score_label': request.session.get('score_label'),
        'result_plot_url': request.session.get('result_plot_url'),
        'model_name': request.session.get('model_name'),
    }
    if trained_features:
        extra['predict_form'] = PredictForm(trained_features)

    if 'predict' in request.POST:
        pf_features = trained_features or features
        predict_form = PredictForm(pf_features, request.POST)
        extra['predict_form'] = predict_form
        if predict_form.is_valid():
            input_data = np.array(
                [[predict_form.cleaned_data[f] for f in pf_features]], dtype=float
            )
            model = joblib.load(MODEL_PATH)
            prediction = model.predict(input_data)[0]
            if problem_type == 'classification':
                extra['prediction_result'] = f'Predicted class: {prediction}'
            else:
                extra['prediction_result'] = f'Predicted value: {round(float(prediction), 4)}'

    elif 'model' in request.POST:
        model_name = request.POST['model']
        test_size = request.POST['split']
        score_label, result_plot_url = train_model(
            df, label, problem_type, model_name, test_size
        )
        request.session['features'] = features
        request.session['model_name'] = model_name
        request.session['score_label'] = score_label
        request.session['result_plot_url'] = result_plot_url
        extra.update({
            'score_label': score_label,
            'result_plot_url': result_plot_url,
            'model_name': model_name,
            'predict_form': PredictForm(features),
        })

    elif 'feature_x' in request.POST:
        feature_x = request.POST['feature_x']
        feature_y = request.POST['feature_y']
        request.session['selected_x'] = feature_x
        request.session['selected_y'] = feature_y

    elif 'feature' in request.POST:
        hist_feature = request.POST['feature']
        request.session['selected_hist'] = hist_feature

    context = build_context(df, label, problem_type, features,
                            feature_x, feature_y, hist_feature, extra)
    return render(request, 'project1/index.html', context)