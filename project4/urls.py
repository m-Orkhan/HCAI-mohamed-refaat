from django.urls import path
from . import views

app_name = 'project4'

urlpatterns = [
    path('', views.landing, name='index'),
    path('study/consent/', views.consent, name='consent'),
    path('study/task/', views.task, name='task'),
    path('study/workload/', views.workload, name='workload'),
    path('study/break/', views.block_break, name='break'),
    path('study/validation/', views.validation, name='validation'),
    path('study/debrief/', views.debrief, name='debrief'),
    path('study/withdraw/', views.withdraw, name='withdraw'),
]