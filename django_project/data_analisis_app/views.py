import os
from django.conf import settings
from django.shortcuts import render
import pandas as pd

'''
creating webpage backend
'''
# Create your views here.

def home(request):
    #funcion que recibe al usuario
    return render(request, 'home.html')

def dashboard(request):
    # obtener direccion del csv
    csv_path = os.path.join(settings.BASE_DIR, 'data_analisis_app', 'csv', 'heart_failure_clinical_records.csv')
    # funcion que contiene el dash
    csv_df = pd.read_csv(csv_path)

    df_html = csv_df.to_html(classes='table table-striped')
    
    # Pasa el HTML del DataFrame al contexto
    context = {'df_html': df_html}

    return render(request, 'dashboard.html', context)