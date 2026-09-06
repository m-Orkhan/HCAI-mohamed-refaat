from django.db import models

class Participant(models.Model):
    participant_id = models.CharField(max_length=12, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    consent_participation = models.BooleanField(default=False)
    consent_data_storage = models.BooleanField(default=False)
    first_condition = models.CharField(max_length=10)
    completed = models.BooleanField(default=False)

    def __str__(self):
        return self.participant_id

class ElicitationResponse(models.Model):
    participant = models.ForeignKey(Participant, on_delete=models.CASCADE, related_name='responses')
    condition = models.CharField(max_length=10)
    block = models.IntegerField()
    task_index = models.IntegerField()
    movie_indices = models.JSONField()
    ranked_indices = models.JSONField()
    seconds_taken = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

class WorkloadResponse(models.Model):
    participant = models.ForeignKey(Participant, on_delete=models.CASCADE, related_name='workload')
    condition = models.CharField(max_length=10)
    mental_demand = models.IntegerField()
    effort = models.IntegerField()
    frustration = models.IntegerField()
    satisfaction = models.IntegerField()

class ValidationResponse(models.Model):
    participant = models.ForeignKey(Participant, on_delete=models.CASCADE, related_name='validation')
    task_index = models.IntegerField()
    movie_indices = models.JSONField()
    chosen_index = models.IntegerField()