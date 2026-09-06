import os
import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'movie_metadata.csv')

NUMERIC_FEATURES = ['title_year', 'duration', 'imdb_score', 'log_budget']
RATING_BUCKETS = ['rating_family', 'rating_teen', 'rating_mature', 'rating_other']

def bucket_rating(value):
    if value in ('G', 'PG'):
        return 'rating_family'
    if value == 'PG-13':
        return 'rating_teen'
    if value in ('R', 'NC-17', 'X'):
        return 'rating_mature'
    return 'rating_other'

def load_movies(min_votes=25000, n_genres=15):
    df = pd.read_csv(DATA_PATH)

    df['movie_title'] = df['movie_title'].str.replace('\xa0', '', regex=False).str.strip()
    df = df.drop_duplicates(subset='movie_title')

    df = df[df['country'] == 'USA']
    df = df.dropna(subset=['movie_title', 'genres', 'title_year',
                           'duration', 'imdb_score', 'budget'])
    df = df[df['num_voted_users'] >= min_votes]
    df = df.reset_index(drop=True)

    df['log_budget'] = np.log10(df['budget'])

    genre_lists = df['genres'].str.split('|')
    all_genres = pd.Series([g for lst in genre_lists for g in lst])
    top_genres = list(all_genres.value_counts().head(n_genres).index)

    genre_cols = []
    for g in top_genres:
        col = 'genre_' + g.lower().replace('-', '_')
        df[col] = genre_lists.apply(lambda lst: int(g in lst))
        genre_cols.append(col)

    buckets = df['content_rating'].apply(bucket_rating)
    for b in RATING_BUCKETS:
        df[b] = (buckets == b).astype(int)

    numeric_cols = []
    for col in NUMERIC_FEATURES:
        z = col + '_z'
        df[z] = (df[col] - df[col].mean()) / df[col].std()
        numeric_cols.append(z)

    feature_cols = genre_cols + numeric_cols + RATING_BUCKETS
    return df, feature_cols, top_genres

def feature_matrix(df, feature_cols):
    return df[feature_cols].to_numpy(dtype=float)

def movie_display(row):
    return {
        'title': row['movie_title'],
        'year': int(row['title_year']),
        'genres': row['genres'].replace('|', ', '),
        'duration': int(row['duration']),
        'imdb_score': float(row['imdb_score']),
        'director': row['director_name'] if pd.notna(row['director_name']) else 'Unknown',
    }