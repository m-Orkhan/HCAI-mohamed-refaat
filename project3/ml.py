import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score

LABELS = ['World', 'Sports', 'Business', 'Sci/Tech']
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    dataset = load_dataset('fancyzhx/ag_news')
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']
    return (
        np.array(train_texts), np.array(train_labels),
        np.array(test_texts), np.array(test_labels)
    )

def train_baseline(train_texts, train_labels, test_texts, test_labels):
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, train_labels)

    train_acc = round(accuracy_score(train_labels, clf.predict(X_train)) * 100, 2)
    test_acc = round(accuracy_score(test_labels, clf.predict(X_test)) * 100, 2)

    joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'vectorizer.joblib'))
    joblib.dump(clf, os.path.join(MODEL_DIR, 'baseline_clf.joblib'))

    return clf, vectorizer, train_acc, test_acc

def simulate_expert(labels):
    expert_preds = []
    for label in labels:
        if label == 0 or label == 1: 
            expert_preds.append(label)
        else:  
            expert_preds.append(np.random.randint(0, 4))
    return np.array(expert_preds)

def evaluate_expert(test_labels):
    expert_preds = simulate_expert(test_labels)
    overall_acc = round(accuracy_score(test_labels, expert_preds) * 100, 2)

    per_class_acc = {}
    for i, label_name in enumerate(LABELS):
        mask = test_labels == i
        acc = round(accuracy_score(test_labels[mask], expert_preds[mask]) * 100, 2)
        per_class_acc[label_name] = acc

    return overall_acc, per_class_acc, expert_preds
def train_deferral_model(train_texts, train_labels, test_texts, test_labels):
    vectorizer = joblib.load(os.path.join(MODEL_DIR, 'vectorizer.joblib'))
    clf = joblib.load(os.path.join(MODEL_DIR, 'baseline_clf.joblib'))

    X_train = vectorizer.transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    clf_test_preds = clf.predict(X_test)
    expert_test_preds = simulate_expert(test_labels)

    defer_train = np.isin(train_labels, [0, 1]).astype(int)
    defer_test = np.isin(test_labels, [0, 1]).astype(int)

    deferral_clf = LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced'
    )
    deferral_clf.fit(X_train, defer_train)

    joblib.dump(deferral_clf, os.path.join(MODEL_DIR, 'deferral_clf.joblib'))

    defer_preds_test = deferral_clf.predict(X_test)

    final_preds = []
    for i in range(len(test_labels)):
        if defer_preds_test[i] == 1:
            final_preds.append(expert_test_preds[i])
        else:
            final_preds.append(clf_test_preds[i])
    final_preds = np.array(final_preds)

    system_acc = round(accuracy_score(test_labels, final_preds) * 100, 2)
    deferral_acc = round(accuracy_score(defer_test, defer_preds_test) * 100, 2)
    defer_rate = round(defer_preds_test.mean() * 100, 2)

    per_class_system_acc = {}
    for i, label_name in enumerate(LABELS):
        mask = test_labels == i
        acc = round(accuracy_score(test_labels[mask], final_preds[mask]) * 100, 2)
        per_class_system_acc[label_name] = acc

    return system_acc, deferral_acc, defer_rate, per_class_system_acc


AL_STRATEGIES = ['random', 'entropy', 'least_confidence', 'margin', 'competence_uncertainty']

def get_uncertainty_scores(probs, strategy):
    if strategy == 'random':
        return np.random.rand(probs.shape[0])
    elif strategy == 'entropy':
        return -np.sum(probs * np.log(probs + 1e-10), axis=1)
    elif strategy == 'least_confidence':
        return 1 - probs.max(axis=1)
    elif strategy == 'margin':
        sorted_probs = np.sort(probs, axis=1)
        return -(sorted_probs[:, -1] - sorted_probs[:, -2])

def run_active_learning(X_train, train_labels, X_test, test_labels, train_probs,
                        clf_test_preds, strategy, n_rounds=10, batch_size=100, seed=42):
    np.random.seed(seed)
    expert_test_preds = simulate_expert(test_labels)
    expert_test_correct = (expert_test_preds == test_labels).astype(int)

    queried_indices = []
    expert_correct_collected = {}
    competence_model = None

    query_counts, accuracies, coverages, competence_scores = [], [], [], []

    for round_i in range(n_rounds):
        remaining = np.setdiff1d(np.arange(len(train_labels)),
                                 np.array(queried_indices, dtype=int))

        if strategy == 'competence_uncertainty':
            if competence_model is None:
                scores = np.random.rand(len(remaining))
            else:
                comp_probs = competence_model.predict_proba(X_train[remaining])[:, 1]
                scores = -np.abs(comp_probs - 0.5)
        else:
            scores = get_uncertainty_scores(train_probs[remaining], strategy)

        top_batch = remaining[np.argsort(scores)[-batch_size:]]

        for idx in top_batch:
            expert_pred = simulate_expert(np.array([train_labels[idx]]))[0]
            expert_correct_collected[idx] = int(expert_pred == train_labels[idx])
            queried_indices.append(idx)

        q_idx = np.array(queried_indices)
        q_labels = np.array([expert_correct_collected[i] for i in q_idx])

        if len(np.unique(q_labels)) < 2:
            competence_model = None
            defer_test = np.zeros(len(test_labels), dtype=int)
            comp_score = 50.0
        else:
            competence_model = LogisticRegression(max_iter=1000, class_weight='balanced')
            competence_model.fit(X_train[q_idx], q_labels)
            defer_test = competence_model.predict(X_test)
            comp_score = balanced_accuracy_score(expert_test_correct, defer_test) * 100

        final_preds = np.where(defer_test == 1, expert_test_preds, clf_test_preds)

        query_counts.append(len(queried_indices))
        accuracies.append(accuracy_score(test_labels, final_preds) * 100)
        coverages.append((1 - defer_test.mean()) * 100)
        competence_scores.append(comp_score)

    queried_categories = train_labels[np.array(queried_indices)]
    return query_counts, accuracies, coverages, competence_scores, queried_categories

def compare_strategies(train_texts, train_labels, test_texts, test_labels, seeds=(42, 43, 44)):
    vectorizer = joblib.load(os.path.join(MODEL_DIR, 'vectorizer.joblib'))
    clf = joblib.load(os.path.join(MODEL_DIR, 'baseline_clf.joblib'))
    X_train = vectorizer.transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    train_probs = clf.predict_proba(X_train)
    clf_test_preds = clf.predict(X_test)

    np.random.seed(0)
    expert_test_preds = simulate_expert(test_labels)
    oracle_defer = np.isin(test_labels, [0, 1])
    oracle_preds = np.where(oracle_defer, expert_test_preds, clf_test_preds)

    baselines = {
        'classifier': round(accuracy_score(test_labels, clf_test_preds) * 100, 2),
        'expert': round(accuracy_score(test_labels, expert_test_preds) * 100, 2),
        'full_information': round(accuracy_score(test_labels, oracle_preds) * 100, 2),
    }

    results = {}
    for strategy in AL_STRATEGIES:
        acc_runs, cov_runs, comp_runs = [], [], []
        cat_counts = np.zeros(4)
        for seed in seeds:
            qc, accs, covs, comps, qcats = run_active_learning(
                X_train, train_labels, X_test, test_labels,
                train_probs, clf_test_preds, strategy, seed=seed
            )
            acc_runs.append(accs)
            cov_runs.append(covs)
            comp_runs.append(comps)
            for c in range(4):
                cat_counts[c] += np.sum(qcats == c)

        results[strategy] = {
            'queries': qc,
            'acc_mean': np.mean(acc_runs, axis=0),
            'acc_std': np.std(acc_runs, axis=0),
            'cov_mean': np.mean(cov_runs, axis=0),
            'cov_std': np.std(cov_runs, axis=0),
            'comp_mean': np.mean(comp_runs, axis=0),
            'comp_std': np.std(comp_runs, axis=0),
            'category_counts': cat_counts / len(seeds),
        }

    return results, baselines

def make_al_plots(results, baselines, media_root, media_url):
    colors = {
        'random': 'gray',
        'entropy': 'red',
        'least_confidence': 'blue',
        'margin': 'green',
        'competence_uncertainty': 'purple',
    }
    urls = {}

    fig, ax = plt.subplots(figsize=(9, 6))
    for s, d in results.items():
        ax.errorbar(d['queries'], d['acc_mean'], yerr=d['acc_std'],
                    label=s, color=colors[s], marker='o', markersize=3, capsize=2)
    ax.axhline(baselines['classifier'], color='black', linestyle='--', linewidth=1, label='classifier alone')
    ax.axhline(baselines['expert'], color='brown', linestyle=':', linewidth=1, label='expert alone')
    ax.axhline(baselines['full_information'], color='teal', linestyle='-.', linewidth=1, label='full information deferral')
    ax.set_xlabel('Number of expert queries')
    ax.set_ylabel('System accuracy (%)')
    ax.set_title('System Accuracy vs Expert Queries')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = os.path.join(media_root, 'al_accuracy.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    urls['acc'] = media_url + 'al_accuracy.png'

    fig, ax = plt.subplots(figsize=(9, 6))
    for s, d in results.items():
        ax.errorbar(d['queries'], d['cov_mean'], yerr=d['cov_std'],
                    label=s, color=colors[s], marker='o', markersize=3, capsize=2)
    ax.set_xlabel('Number of expert queries')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Coverage vs Expert Queries')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = os.path.join(media_root, 'al_coverage.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    urls['cov'] = media_url + 'al_coverage.png'

    fig, ax = plt.subplots(figsize=(9, 6))
    for s, d in results.items():
        ax.errorbar(d['queries'], d['comp_mean'], yerr=d['comp_std'],
                    label=s, color=colors[s], marker='o', markersize=3, capsize=2)
    ax.set_xlabel('Number of expert queries')
    ax.set_ylabel('Competence model balanced accuracy (%)')
    ax.set_title('Expert Competence Discovery')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = os.path.join(media_root, 'al_competence.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    urls['comp'] = media_url + 'al_competence.png'

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(4)
    width = 0.15
    for i, (s, d) in enumerate(results.items()):
        ax.bar(x + i * width, d['category_counts'], width, label=s, color=colors[s])
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel('Average queries per category')
    ax.set_title('Query Distribution by Category')
    ax.legend(fontsize=8)
    path = os.path.join(media_root, 'al_distribution.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    urls['dist'] = media_url + 'al_distribution.png'

    return urls