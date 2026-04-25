from django import forms

PROBLEM_TYPES = [
    ('classification', 'Group data into categories'),
    ('regression', 'Predict a number'),
]

SPLIT_CHOICES = [
    ('0.3', '70% train / 30% test'),
    ('0.2', '80% train / 20% test'),
    ('0.1', '90% train / 10% test'),
]

MODEL_CHOICES_CLASSIFICATION = [
    ('decision_tree', 'Decision Tree'),
    ('svm', 'SVM'),
]

MODEL_CHOICES_REGRESSION = [
    ('linear_regression', 'Linear Regression'),
]

class CSVUploadForm(forms.Form):
    file = forms.FileField(
        label='Upload your CSV file',
        help_text='Make sure the last column is the label'
    )
    problem_type = forms.ChoiceField(
        label='What do you want to do?',
        choices=PROBLEM_TYPES,
    )

class FeatureSelectForm(forms.Form):
    feature = forms.ChoiceField(label='Select a feature to plot', choices=[])

    def __init__(self, features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['feature'].choices = [(f, f) for f in features]

class ScatterSelectForm(forms.Form):
    feature_x = forms.ChoiceField(label='X axis', choices=[])
    feature_y = forms.ChoiceField(label='Y axis', choices=[])

    def __init__(self, features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['feature_x'].choices = [(f, f) for f in features]
        self.fields['feature_y'].choices = [(f, f) for f in features]

class TrainForm(forms.Form):
    model = forms.ChoiceField(label='Choose a model', choices=[])
    split = forms.ChoiceField(
        label='How much data should be used for training?',
        choices=SPLIT_CHOICES,
    )

    def __init__(self, problem_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if problem_type == 'classification':
            self.fields['model'].choices = MODEL_CHOICES_CLASSIFICATION
        else:
            self.fields['model'].choices = MODEL_CHOICES_REGRESSION

class PredictForm(forms.Form):
    def __init__(self, features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for feature in features:
            self.fields[feature] = forms.FloatField(
                label=feature,
                widget=forms.NumberInput(attrs={'step': 'any'})
            )