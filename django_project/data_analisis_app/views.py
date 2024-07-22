from django.shortcuts import render
from django.conf import settings
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import json
def home(request):
    scaler = StandardScaler()

    # Obtener dirección del CSV
    csv_path = os.path.join(settings.BASE_DIR, 'data_analisis_app', 'csv', 'heart_failure_clinical_records.csv')
    # Lectura del CSV
    csv_df = pd.read_csv(csv_path)

    # SECCIÓN 1: Análisis inicial y preparación de datos
    dscrpt_df = csv_df.sample(7)
    df_html = dscrpt_df.to_dict(orient='records')
    df_nan = csv_df.isna().sum()
    df_nan = df_nan.items()

    # SECCIÓN 2: Introducción de valores NaN y técnicas de imputación
    # Reemplazar 10% de los valores con NaN
    df_with_nan = csv_df.mask(np.random.random(csv_df.shape) < 0.1)
    
    # Imputación simple con la media
    imputer_mean = SimpleImputer(strategy='mean')
    df_imputed_mean = pd.DataFrame(imputer_mean.fit_transform(df_with_nan), columns=csv_df.columns)
    
    # Imputación por la mediana
    imputer_median = SimpleImputer(strategy='median')
    df_imputed_median = pd.DataFrame(imputer_median.fit_transform(df_with_nan), columns=csv_df.columns)
    
    # Imputación iterativa
    imputer_iterative = IterativeImputer(random_state=0)
    df_imputed_iterative = pd.DataFrame(imputer_iterative.fit_transform(df_with_nan), columns=csv_df.columns)


    # SECCIÓN 4: Preparación de datos para visualizaciones (usando imputación por la mediana)
    features = csv_df.columns.drop('DEATH_EVENT')
    
    feature_translation = {
        'age': 'Edad', 'anaemia': 'Anemia', 'creatinine_phosphokinase': 'Creatina Fosfoquinasa',
        'diabetes': 'Diabetes', 'ejection_fraction': 'Fracción de Eyección',
        'high_blood_pressure': 'Presión Arterial Alta', 'platelets': 'Plaquetas',
        'serum_creatinine': 'Creatinina Sérica', 'serum_sodium': 'Sodio Sérico',
        'sex': 'Mujer/Hombre', 'smoking': 'Fumador', 'time': 'Tiempo en estudio'
    }
    scaler = MinMaxScaler()

    # no imputation
    df_normalized = pd.DataFrame(scaler.fit_transform(csv_df.drop('DEATH_EVENT', axis=1)), columns=features)
    df_normalized['DEATH_EVENT'] = csv_df['DEATH_EVENT']
    df_death_norm = df_normalized[df_normalized['DEATH_EVENT'] == 1]
    df_no_death_norm = df_normalized[df_normalized['DEATH_EVENT'] == 0]
    
    chart_data = {
        'features': [feature_translation.get(feature, feature) for feature in features],
        'no_death': [df_no_death_norm[feature].mean() for feature in features],
        'death': [df_death_norm[feature].mean() for feature in features]
    }

    # mean imputation
    df_normalized2 = pd.DataFrame(scaler.fit_transform(df_imputed_mean.drop('DEATH_EVENT', axis=1)), columns=features)
    df_normalized2['DEATH_EVENT'] = df_imputed_mean['DEATH_EVENT']
    df_death_norm2 = df_normalized2[df_normalized2['DEATH_EVENT'] == 1]
    df_no_death_norm2 = df_normalized2[df_normalized2['DEATH_EVENT'] == 0]
    
    chart_data2 = {
        'features': [feature_translation.get(feature, feature) for feature in features],
        'no_death': [df_no_death_norm2[feature].mean() for feature in features],
        'death': [df_death_norm2[feature].mean() for feature in features]
    }

    # median imputation
    df_normalized3 = pd.DataFrame(scaler.fit_transform(df_imputed_median.drop('DEATH_EVENT', axis=1)), columns=features)
    df_normalized3['DEATH_EVENT'] = df_imputed_median['DEATH_EVENT']
    df_death_norm3 = df_normalized3[df_normalized3['DEATH_EVENT'] == 1]
    df_no_death_norm3 = df_normalized3[df_normalized3['DEATH_EVENT'] == 0]
    
    chart_data3 = {
        'features': [feature_translation.get(feature, feature) for feature in features],
        'no_death': [df_no_death_norm3[feature].mean() for feature in features],
        'death': [df_death_norm3[feature].mean() for feature in features]
    }

    # median imputation
    df_normalized4 = pd.DataFrame(scaler.fit_transform(df_imputed_iterative.drop('DEATH_EVENT', axis=1)), columns=features)
    df_normalized4['DEATH_EVENT'] = df_imputed_iterative['DEATH_EVENT']
    df_death_norm4 = df_normalized4[df_normalized4['DEATH_EVENT'] == 1]
    df_no_death_norm4 = df_normalized4[df_normalized4['DEATH_EVENT'] == 0]
    
    chart_data4 = {
        'features': [feature_translation.get(feature, feature) for feature in features],
        'no_death': [df_no_death_norm4[feature].mean() for feature in features],
        'death': [df_death_norm4[feature].mean() for feature in features]
    }




    # SECCIÓN 5: Preparación de datos adicionales para visualizaciones

    ## Dashboard2
    # Matriz de correlación (usando imputación de media)
    correlation_matrix = csv_df.corr().round(2).to_dict() # no imputated df 
    correlation_matrix_mean = df_imputed_mean.corr().round(2).to_dict()
    
    # Crear rangos de edad (usando imputación por la media)
    bins = [0, 40, 60, 80, 100]
    labels = ['<40', '40-60', '60-80', '80+']

    csv_df_copy = csv_df.copy()

    csv_df_copy['age_range'] = pd.cut(csv_df['age'], bins=bins, labels=labels) # no imputed df
    df_imputed_median['age_range'] = pd.cut(df_imputed_median['age'], bins=bins, labels=labels) # imputed df

    
    ## dash 3
    # Datos para gráfico de fracción de eyección por rango de edad con mediana
    # no imputation
    ejection_fraction_by_age_range = csv_df_copy.groupby('age_range', observed=True)['ejection_fraction'].apply(list)
    age_ranges = ejection_fraction_by_age_range.index.tolist()
    ejection_fraction_data = ejection_fraction_by_age_range.tolist()

    # Datos para gráfico de tendencia
    age_midpoints = [0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)]
    ejection_fraction = [csv_df_copy[csv_df_copy['age_range'] == label]['ejection_fraction'].mean() for label in labels]

    # df imputated
    ejection_fraction_by_age_range = df_imputed_median.groupby('age_range', observed=True)['ejection_fraction'].apply(list)
    age_ranges_median = ejection_fraction_by_age_range.index.tolist()
    ejection_fraction_data_median = ejection_fraction_by_age_range.tolist()
    
    # Datos para gráfico de tendencia
    age_midpoints_median = [0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)]
    ejection_fraction_median = [df_imputed_median[df_imputed_median['age_range'] == label]['ejection_fraction'].mean() for label in labels]



    ## dashboard 4
    # Datos para gráfico de dispersión (usando imputación iterativa)
    scatter_data = csv_df_copy[['serum_creatinine', 'serum_sodium', 'DEATH_EVENT']].to_dict('records')
    scatter_data_iterative = df_imputed_iterative[['serum_creatinine', 'serum_sodium', 'DEATH_EVENT']].to_dict('records')

    context = {
        ## dashboard 1 (home)
        'df_html': df_html,
        'df_nan': df_nan,
        'chart_data': json.dumps(chart_data),
        'chart_data_mean': json.dumps(chart_data2),
        'chart_data_median': json.dumps(chart_data3),
        'chart_data_iterative': json.dumps(chart_data4),


        ## dashboard 2
        # no imputed data
        'correlation_matrix': json.dumps(correlation_matrix),
        # imputed data
        'correlation_matrix_mean': json.dumps(correlation_matrix_mean),
        'features': json.dumps(list(csv_df.columns)),

        ## dashboard 3
        # no imputed data
        'age_ranges': json.dumps(age_ranges),
        'ejection_fraction_data': json.dumps(ejection_fraction_data),
        'age_midpoints': json.dumps(age_midpoints),
        'ejection_fraction': json.dumps(ejection_fraction),

        # imputed data
        'age_ranges_median': json.dumps(age_ranges_median),
        'ejection_fraction_data_median': json.dumps(ejection_fraction_data_median),
        'age_midpoints_median': json.dumps(age_midpoints_median),
        'ejection_fraction_median': json.dumps(ejection_fraction_median),

        ## dashboard 4
        'scatter_data': json.dumps(scatter_data),

        'scatter_data_iterative': json.dumps(scatter_data_iterative)
    }

    return render(request, 'home.html', context)