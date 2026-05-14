import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from django.conf import settings
from django.shortcuts import render
from palmerpenguins import load_penguins
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def save_plot(fig, filename):
    path = os.path.join(settings.MEDIA_ROOT, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return settings.MEDIA_URL + filename

def load_and_prepare():
    df_raw = load_penguins().dropna().reset_index(drop=True)
    df_encoded = df_raw.copy()
    df_encoded['island'] = df_encoded['island'].astype('category').cat.codes
    df_encoded['sex'] = df_encoded['sex'].astype('category').cat.codes
    feature_cols = ['island', 'bill_length_mm', 'bill_depth_mm',
                    'flipper_length_mm', 'body_mass_g', 'sex', 'year']
    X = df_encoded[feature_cols].values
    y = df_encoded['species'].values
    return X, y, feature_cols, df_raw, df_encoded

def train_all_trees(X_train, X_test, y_train, y_test):
    results = []
    for max_leaves in range(2, 31):
        clf = DecisionTreeClassifier(max_leaf_nodes=max_leaves, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append({
            'clf': clf,
            'accuracy': round(acc * 100, 2),
            'n_leaves': clf.get_n_leaves(),
        })
    return results

def train_all_logistic(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    results = []
    for C in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]:
        clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        coef_norm = round(float(np.sum(np.abs(clf.coef_))), 4)
        results.append({
            'clf': clf,
            'scaler': scaler,
            'accuracy': round(acc * 100, 2),
            'complexity': coef_norm,
            'C': C,
        })
    return results

def select_best_tree(results, lam):
    max_leaves = max(r['n_leaves'] for r in results)
    best = min(results, key=lambda r: (
        (1 - r['accuracy'] / 100) + lam * (r['n_leaves'] / max_leaves),
        -r['n_leaves']
    ))
    return best

def select_best_logistic(results, lam):
    max_complexity = max(r['complexity'] for r in results)
    best = min(results, key=lambda r: (
        (1 - r['accuracy'] / 100) + lam * (r['complexity'] / max_complexity),
        -r['complexity']
    ))
    return best

def make_tree_plot(clf, feature_cols):
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(clf, feature_names=feature_cols,
              class_names=['Adelie', 'Chinstrap', 'Gentoo'],
              filled=True, ax=ax, fontsize=8)
    return save_plot(fig, 'tree_plot.png')

def make_logistic_coef_plot(clf, feature_cols):
    fig, ax = plt.subplots(figsize=(10, 5))
    coefs = clf.coef_
    classes = clf.classes_
    x = np.arange(len(feature_cols))
    width = 0.25
    for i, cls in enumerate(classes):
        ax.bar(x + i * width, coefs[i], width, label=cls)
    ax.set_xticks(x + width)
    ax.set_xticklabels(feature_cols, rotation=25, ha='right')
    ax.set_ylabel('Coefficient value')
    ax.set_title('Logistic Regression Coefficients')
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.8)
    return save_plot(fig, 'logistic_coef_plot.png')
NUMERICAL_FEATURES = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

def compute_pdp(clf, X, feature_cols, feature_name, scaler=None):
    feature_idx = feature_cols.index(feature_name)
    feature_values = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), 50)
    classes = ['Adelie', 'Chinstrap', 'Gentoo']
    pdp_values = {cls: [] for cls in classes}

    for val in feature_values:
        X_modified = X.copy()
        X_modified[:, feature_idx] = val
        if scaler is not None:
            X_modified_scaled = scaler.transform(X_modified)
            probs = clf.predict_proba(X_modified_scaled)
        else:
            probs = clf.predict_proba(X_modified)
        mean_probs = probs.mean(axis=0)
        for i, cls in enumerate(classes):
            pdp_values[cls].append(mean_probs[i])

    return feature_values, pdp_values

def compute_ale(clf, X, feature_cols, feature_name, scaler=None, n_bins=20):
    feature_idx = feature_cols.index(feature_name)
    feature_vals = X[:, feature_idx]
    classes = ['Adelie', 'Chinstrap', 'Gentoo']

    percentiles = np.percentile(feature_vals, np.linspace(0, 100, n_bins + 1))
    percentiles = np.unique(percentiles)
    bin_centers = (percentiles[:-1] + percentiles[1:]) / 2

    ale_values = {cls: np.zeros(len(bin_centers)) for cls in classes}

    for b in range(len(bin_centers)):
        low = percentiles[b]
        high = percentiles[b + 1]

        mask = (feature_vals >= low) & (feature_vals <= high)
        if mask.sum() == 0:
            continue

        X_low = X[mask].copy()
        X_high = X[mask].copy()
        X_low[:, feature_idx] = low
        X_high[:, feature_idx] = high

        if scaler is not None:
            probs_low = clf.predict_proba(scaler.transform(X_low))
            probs_high = clf.predict_proba(scaler.transform(X_high))
        else:
            probs_low = clf.predict_proba(X_low)
            probs_high = clf.predict_proba(X_high)

        local_effect = (probs_high - probs_low).mean(axis=0)
        for i, cls in enumerate(classes):
            ale_values[cls][b] = local_effect[i]


    for cls in classes:
        ale_values[cls] = np.cumsum(ale_values[cls])
        ale_values[cls] -= ale_values[cls].mean()

    return bin_centers, ale_values

def make_pdp_plot(feature_values, pdp_values, feature_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'Adelie': 'blue', 'Chinstrap': 'orange', 'Gentoo': 'green'}
    for cls, vals in pdp_values.items():
        ax.plot(feature_values, vals, label=cls, color=colors[cls])
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Average predicted probability')
    ax.set_title(f'PDP — {feature_name}')
    ax.legend()
    return save_plot(fig, 'pdp_plot.png')

def make_ale_plot(bin_centers, ale_values, feature_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'Adelie': 'blue', 'Chinstrap': 'orange', 'Gentoo': 'green'}
    for cls, vals in ale_values.items():
        ax.plot(bin_centers, vals, label=cls, color=colors[cls])
    ax.set_xlabel(feature_name)
    ax.set_ylabel('ALE')
    ax.set_title(f'ALE — {feature_name}')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.legend()
    return save_plot(fig, 'ale_plot.png')

def get_penguin_options(df_raw):
    options = []
    for i, row in df_raw.iterrows():
        label = f"#{i} — {row['species']}, {row['island']}, {row['sex']}, bill={row['bill_length_mm']}mm"
        options.append((i, label))
    return options

def generate_counterfactuals(x_original, target_class, clf, feature_cols, df_encoded_features, N=10000, k=3, scaler=None):
    categorical_indices = [feature_cols.index('island'), feature_cols.index('sex'), feature_cols.index('year')]
    samples = []
    for _ in range(N):
        new_point = x_original.copy().astype(float)
        for j, col in enumerate(feature_cols):
            if j in categorical_indices:
                unique_vals = df_encoded_features.iloc[:, j].unique()
                new_point[j] = np.random.choice(unique_vals)
            else:
                std = df_encoded_features.iloc[:, j].std()
                new_point[j] = x_original[j] + np.random.normal(0, std * 0.5)
        samples.append(new_point)

    samples = np.array(samples)

    if scaler is not None:
        preds = clf.predict(scaler.transform(samples))
    else:
        preds = clf.predict(samples)

    matching = samples[preds == target_class]
    if len(matching) == 0:
        return None

    mad = np.median(np.abs(df_encoded_features.values - np.median(df_encoded_features.values, axis=0)), axis=0)
    mad = np.where(mad == 0, 1, mad)
    distances = np.sum(np.abs(matching - x_original) / mad, axis=1)
    top_k_idx = np.argsort(distances)[:k]
    return matching[top_k_idx]

def index(request):
    X, y, feature_cols, df_raw, df_encoded = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_type = request.GET.get('model_type', 'tree')
    lam = float(request.GET.get('lam', 0.0))

    best_clf = None
    best_scaler = None

    context = {
        'model_type': model_type,
        'lam': lam,
        'penguin_options': get_penguin_options(df_raw),
        'classes': list(df_raw['species'].unique()),
        'feature_cols': feature_cols,
    }

    if model_type == 'tree':
        results = train_all_trees(X_train, X_test, y_train, y_test)
        best = select_best_tree(results, lam)
        best_clf = best['clf']
        tree_url = make_tree_plot(best_clf, feature_cols)
        context.update({
            'tree_url': tree_url,
            'accuracy': best['accuracy'],
            'n_leaves': best['n_leaves'],
        })
    else:
        results = train_all_logistic(X_train, X_test, y_train, y_test)
        best = select_best_logistic(results, lam)
        best_clf = best['clf']
        best_scaler = best['scaler']
        coef_url = make_logistic_coef_plot(best_clf, feature_cols)
        context.update({
            'coef_url': coef_url,
            'accuracy': best['accuracy'],
            'complexity': best['complexity'],
            'C': best['C'],
        })

    if request.method == 'POST':
        penguin_idx = int(request.POST.get('penguin_idx'))
        target_class = request.POST.get('target_class')
        k = int(request.POST.get('k', 3))

        x_original = df_encoded[feature_cols].values[penguin_idx]

        counterfactuals = generate_counterfactuals(
            x_original, target_class, best_clf, feature_cols,
            df_encoded[feature_cols], k=k, scaler=best_scaler
        )

        if counterfactuals is not None:
            original_display = [round(float(x_original[j]), 2) for j in range(len(feature_cols))]
            cf_table = []
            for cf in counterfactuals:
                row = []
                for j in range(len(feature_cols)):
                    original_val = round(float(x_original[j]), 2)
                    cf_val = round(float(cf[j]), 2)
                    changed = abs(original_val - cf_val) > 0.01
                    row.append({'value': cf_val, 'changed': changed})
                cf_table.append(row)

            context.update({
                'original_display': original_display,
                'original_species': df_raw.iloc[penguin_idx]['species'],
                'cf_table': cf_table,
                'target_class': target_class,
                'penguin_idx': penguin_idx,
                'k': k,
            })
        else:
            context['cf_error'] = 'No counterfactuals found. Try a different penguin or target class.'
    
    selected_feature = request.GET.get('selected_feature', 'bill_length_mm')
    feature_values, pdp_values = compute_pdp(best_clf, X, feature_cols, selected_feature, scaler=best_scaler)
    bin_centers, ale_values = compute_ale(best_clf, X, feature_cols, selected_feature, scaler=best_scaler)
    pdp_url = make_pdp_plot(feature_values, pdp_values, selected_feature)
    ale_url = make_ale_plot(bin_centers, ale_values, selected_feature)

    context.update({
        'pdp_url': pdp_url,
        'ale_url': ale_url,
        'selected_feature': selected_feature,
        'numerical_features': NUMERICAL_FEATURES,
    })
    return render(request, 'project2/index.html', context)