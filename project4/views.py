import random
import time
import numpy as np
from django.shortcuts import render, redirect
from django.utils.crypto import get_random_string
from .models import Participant, ElicitationResponse, WorkloadResponse, ValidationResponse
from .features import load_movies, feature_matrix, movie_display
from .model import fit_preferences, predict_pairwise

BLOCK_SECONDS = 300
RANKING_SIZE = 10
VALIDATION_TASKS = 20
VALIDATION_SEED = 20260101
VALIDATION_POOL = 200

_CACHE = {}

def get_data():
    if 'df' not in _CACHE:
        df, feature_cols, genres = load_movies()
        _CACHE['df'] = df
        _CACHE['feature_cols'] = feature_cols
        _CACHE['X'] = feature_matrix(df, feature_cols)
    return _CACHE['df'], _CACHE['feature_cols'], _CACHE['X']

def current_condition(state):
    other = 'ranking' if state['first'] == 'pairwise' else 'pairwise'
    return state['first'] if state['block'] == 1 else other

def get_participant(state):
    return Participant.objects.get(participant_id=state['pid'])

def landing(request):
    return render(request, 'project4/landing.html')

def consent(request):
    if request.method == 'POST':
        if not (request.POST.get('consent_participation') and
                request.POST.get('consent_data_storage')):
            return redirect('project4:consent')

        df, cols, X = get_data()
        pid = get_random_string(8).upper()
        first = random.choice(['pairwise', 'ranking'])

        Participant.objects.create(
            participant_id=pid,
            consent_participation=True,
            consent_data_storage=True,
            first_condition=first,
        )

        fixed = random.Random(VALIDATION_SEED)
        all_idx = list(range(len(df)))
        val_pool = fixed.sample(all_idx, VALIDATION_POOL)
        val_set = set(val_pool)
        rest = [i for i in all_idx if i not in val_set]
        random.shuffle(rest)
        half = len(rest) // 2

        request.session['p4'] = {
            'pid': pid,
            'first': first,
            'block': 1,
            'task_index': 0,
            'val_index': 0,
            'pool_a': rest[:half],
            'pool_b': rest[half:],
            'pool_val': val_pool,
            'deadline': time.time() + BLOCK_SECONDS,
            'current': None,
            'started': time.time(),
        }
        return redirect('project4:task')

    return render(request, 'project4/consent.html')

def task(request):
    state = request.session.get('p4')
    if not state:
        return redirect('project4:index')

    df, cols, X = get_data()
    condition = current_condition(state)

    if request.method == 'POST':
        ranked = request.POST.get('ranked_indices', '')
        ranked_list = [int(v) for v in ranked.split(',') if v != '']
        ElicitationResponse.objects.create(
            participant=get_participant(state),
            condition=condition,
            block=state['block'],
            task_index=state['task_index'],
            movie_indices=state['current'],
            ranked_indices=ranked_list,
            seconds_taken=round(time.time() - state['started'], 2),
        )
        state['task_index'] += 1
        request.session['p4'] = state
        if time.time() >= state['deadline']:
            return redirect('project4:workload')
        return redirect('project4:task')

    if time.time() >= state['deadline'] and state['task_index'] > 0:
        return redirect('project4:workload')

    pool = state['pool_a'] if state['block'] == 1 else state['pool_b']
    n = 2 if condition == 'pairwise' else RANKING_SIZE
    movies = random.sample(pool, n)
    state['current'] = movies
    state['started'] = time.time()
    request.session['p4'] = state

    movie_data = [dict(movie_display(df.iloc[i]), idx=i) for i in movies]
    remaining = max(0, int(state['deadline'] - time.time()))
    template = ('project4/task_pairwise.html' if condition == 'pairwise'
                else 'project4/task_ranking.html')

    return render(request, template, {
        'movies': movie_data,
        'block_number': state['block'],
        'task_number': state['task_index'] + 1,
        'remaining': remaining,
    })

def workload(request):
    state = request.session.get('p4')
    if not state:
        return redirect('project4:index')
    condition = current_condition(state)

    if request.method == 'POST':
        WorkloadResponse.objects.create(
            participant=get_participant(state),
            condition=condition,
            mental_demand=int(request.POST['mental_demand']),
            effort=int(request.POST['effort']),
            frustration=int(request.POST['frustration']),
            satisfaction=int(request.POST['satisfaction']),
        )
        if state['block'] == 1:
            return redirect('project4:break')
        return redirect('project4:validation')

    return render(request, 'project4/workload.html', {'condition': condition})

def block_break(request):
    state = request.session.get('p4')
    if not state:
        return redirect('project4:index')

    if request.method == 'POST':
        state['block'] = 2
        state['task_index'] = 0
        state['deadline'] = time.time() + BLOCK_SECONDS
        request.session['p4'] = state
        return redirect('project4:task')

    next_condition = 'ranking' if state['first'] == 'pairwise' else 'pairwise'
    return render(request, 'project4/break.html', {'next_condition': next_condition})

def validation(request):
    state = request.session.get('p4')
    if not state:
        return redirect('project4:index')

    df, cols, X = get_data()
    rng = random.Random(VALIDATION_SEED + 1)
    pairs = [rng.sample(state['pool_val'], 2) for _ in range(VALIDATION_TASKS)]
    idx = state.get('val_index', 0)

    if request.method == 'POST':
        ValidationResponse.objects.create(
            participant=get_participant(state),
            task_index=idx,
            movie_indices=pairs[idx],
            chosen_index=int(request.POST['chosen']),
        )
        state['val_index'] = idx + 1
        request.session['p4'] = state
        if state['val_index'] >= VALIDATION_TASKS:
            return redirect('project4:debrief')
        return redirect('project4:validation')

    if idx >= VALIDATION_TASKS:
        return redirect('project4:debrief')

    movies = [dict(movie_display(df.iloc[i]), idx=i) for i in pairs[idx]]
    return render(request, 'project4/validation.html', {
        'movies': movies,
        'task_number': idx + 1,
        'total': VALIDATION_TASKS,
    })

def debrief(request):
    state = request.session.get('p4')
    if not state:
        return redirect('project4:index')

    df, cols, X = get_data()
    participant = get_participant(state)
    participant.completed = True
    participant.save()

    validations = list(participant.validation.all())
    results = {}

    for condition in ('pairwise', 'ranking'):
        responses = list(participant.responses.filter(condition=condition))
        rankings = []
        for r in responses:
            ranked = r.ranked_indices
            if len(ranked) < 2:
                continue
            unranked = [i for i in r.movie_indices if i not in ranked]
            rankings.append((X[ranked + unranked], len(ranked)))
        if not rankings:
            continue

        w = fit_preferences(rankings, X.shape[1])
        total_seconds = sum(r.seconds_taken for r in responses)

        correct = 0
        for v in validations:
            a, b = v.movie_indices
            chosen = v.chosen_index
            other = b if chosen == a else a
            if predict_pairwise(w, X[chosen], X[other]) > 0.5:
                correct += 1
        accuracy = 100.0 * correct / len(validations) if validations else 0.0

        top = sorted(zip(cols, w), key=lambda t: -abs(t[1]))[:6]
        results[condition] = {
            'n_tasks': len(rankings),
            'minutes': round(total_seconds / 60, 1),
            'accuracy': round(accuracy, 1),
            'per_minute': round(accuracy / (total_seconds / 60), 1) if total_seconds > 0 else 0,
            'top_weights': [
                (c.replace('genre_', '').replace('_z', '').replace('rating_', 'rated '),
                 round(float(v), 2))
                for c, v in top
            ],
        }

    return render(request, 'project4/debrief.html', {
        'pid': state['pid'],
        'results': results,
    })

def withdraw(request):
    state = request.session.get('p4')
    if state:
        Participant.objects.filter(participant_id=state['pid']).delete()
        del request.session['p4']
    return render(request, 'project4/withdrawn.html')