import os
from django.conf import settings
from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, classification_report

'''
creating webpage backend
'''
# Create your views here.

def home(request):
    #funcion que recibe al usuario
    return render(request, 'home.html')

def dashboard(request):
    scaler = StandardScaler()

    # obtener direccion del csv
    csv_path = os.path.join(settings.BASE_DIR, 'data_analisis_app', 'csv', 'heart_failure_clinical_records.csv')
    # lectura del csv
    csv_df = pd.read_csv(csv_path)
    # obtencion de 7 registros aleatorios para la visualizacion de datos
    dscrpt_df = csv_df.sample(7)
    # preparacion del df para la visualizacion en el template
    df_html = dscrpt_df.to_dict(orient='records')
    # obtencion de datos NaN
    df_nan = csv_df.isna().sum()
    df_html_nan = df_nan.to_frame(name='missing_values').to_html()
    # separara variables objetivos de variables caracteristicas
    X = csv_df.drop(columns=['DEATH_EVENT'])
    y = csv_df['DEATH_EVENT']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizar las características numéricas
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(random_state=42)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    y_pred_prob = model.predict_proba(X_test_scaled)

    # Preparar los datos para la plantilla
    prob_data = []
    for i, (prob_0, prob_1) in enumerate(y_pred_prob):
        prob_data.append({
            'index': i,
            'prob_class_0': prob_0,
            'prob_class_1': prob_1,
            'predicted_class': np.argmax([prob_0, prob_1]),
            'actual_class': y_test.iloc[i]
        })
    # Limitar a las primeras 10 predicciones para no sobrecargar la página
    prob_data = prob_data[:10]

    report = classification_report(y_test, y_pred, output_dict=True)

    # Preparar los datos para la plantilla
    report_data = []
    for class_name, metrics in report.items():
        if isinstance(metrics, dict):
            metrics['class'] = class_name
            metrics['f1_score'] = metrics.pop('f1-score', None)
            report_data.append(metrics)

    # Pasa el HTML del DataFrame al contexto
    context = {'df_html': df_html,
                'df_html_nan': df_html_nan,
                'accuracy': accuracy,
                'precission': precision,
                'prob_data': prob_data,
                'report_data': report_data,
                'report_accuracy': report['accuracy']
            }

    return render(request, 'dashboard.html', context)