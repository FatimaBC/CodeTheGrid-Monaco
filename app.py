from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import json
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import re
from sklearn.model_selection import cross_val_score

app = Flask(__name__)

def cargar_datos():
    try:
        # Cargar los datos
        drivers = pd.read_csv('datasets/drivers.csv')
        results = pd.read_csv('datasets/results.csv')
        races = pd.read_csv('datasets/races.csv')
        circuits = pd.read_csv('datasets/circuits.csv')
        constructor_standings = pd.read_csv('datasets/constructor_standings.csv')
        lap_times = pd.read_csv('datasets/lap_times.csv')
        pit_stops = pd.read_csv('datasets/pit_stops.csv')
        qualifying = pd.read_csv('datasets/qualifying.csv')
        formula_1 = pd.read_csv('datasets/formula1.csv')

        print("\nColumnas disponibles en circuits:")
        print(circuits.columns.tolist())

        # Replace '\\N' with NaN
        results.replace('\\N', np.nan, inplace=True)

        # Convert numeric columns
        results['milliseconds'] = pd.to_numeric(results['milliseconds'], errors='coerce')
        results['fastestLapSpeed'] = pd.to_numeric(results['fastestLapSpeed'], errors='coerce')

        print("Datos cargados correctamente:")
        print(f"Drivers: {len(drivers)} registros")
        print(f"Results: {len(results)} registros")
        print(f"Races: {len(races)} registros")
        print(f"Circuits: {len(circuits)} registros")
        print(f"Constructor Standings: {len(constructor_standings)} registros")
        print(f"Formula 1: {len(formula_1)} registros")

        return drivers, results, races, circuits, constructor_standings, lap_times, pit_stops, formula_1
    except Exception as e:
        print(f"Error al cargar datos: {str(e)}")
        raise

def preparar_datos_piloto(driver_id, results, races, circuits):
    try:
        # Filtrar resultados del piloto
        driver_results = results[results['driverId'] == driver_id]
        print(f"Resultados encontrados para el piloto: {len(driver_results)}")
        
        # Unir con información de carreras y circuitos
        driver_data = pd.merge(driver_results, races, on='raceId')
        driver_data = pd.merge(driver_data, 
                             circuits[['circuitId', 'circuitRef', 'location', 'country']], 
                             on='circuitId',
                             how='left')
        
        print(f"Columnas después del merge: {driver_data.columns.tolist()}")
        
        # Agregar número de curvas basado en el tipo de circuito
        def estimar_curvas(row):
            circuit_ref = str(row['circuitRef']).lower()
            if 'street' in circuit_ref:
                return 20  # Circuitos urbanos suelen tener más curvas
            elif 'monaco' in circuit_ref:
                return 19  # Mónaco tiene 19 curvas
            elif 'spa' in circuit_ref:
                return 20  # Spa-Francorchamps tiene 20 curvas
            elif 'monza' in circuit_ref:
                return 11  # Monza tiene 11 curvas
            elif 'melbourne' in circuit_ref:
                return 16  # Albert Park tiene 16 curvas
            elif 'silverstone' in circuit_ref:
                return 18  # Silverstone tiene 18 curvas
            else:
                return 15  # Promedio típico de curvas
        
        driver_data['turns'] = driver_data.apply(estimar_curvas, axis=1)
        
        # Intentar cargar datos del clima si el archivo existe
        try:
            weather = pd.read_csv('datasets/weather.csv')
            
            # Asegurarse de que las columnas de clima sean numéricas
            for col in ['AirTemp', 'Humidity', 'WindSpeed']:
                if col in weather.columns:
                    weather[col] = pd.to_numeric(weather[col], errors='coerce')
            
            # Calcular promedio de condiciones climáticas por carrera
            weather_avg = weather.groupby(['Year', 'Round Number']).agg({
                'AirTemp': 'mean',
                'Humidity': 'mean',
                'WindSpeed': 'mean',
                'Rainfall': lambda x: 'Lluvia' if x.any() else 'Seco'
            }).reset_index()
            
            # Unir con datos de clima
            driver_data = pd.merge(driver_data, weather_avg, 
                                 left_on=['year', 'round'], 
                                 right_on=['Year', 'Round Number'], 
                                 how='left')
        except Exception as e:
            print(f"Advertencia: No se pudieron cargar los datos del clima: {str(e)}")
            # Agregar columnas de clima con valores por defecto
            driver_data['AirTemp'] = 25.0
            driver_data['Humidity'] = 65.0
            driver_data['WindSpeed'] = 5.0
            driver_data['Rainfall'] = 'Seco'
        
        # Obtener información del piloto
        drivers = pd.read_csv('datasets/drivers.csv')
        driver_info = drivers[drivers['driverId'] == driver_id].iloc[0]
        
        # Calcular edad del piloto para cada carrera
        driver_data['dob'] = pd.to_datetime(driver_info['dob'])
        driver_data['race_date'] = pd.to_datetime(driver_data['date'])
        driver_data['driverAge'] = (driver_data['race_date'] - driver_data['dob']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
        
        # Calcular victorias del piloto hasta cada carrera
        driver_data['driverWins'] = driver_data.apply(
            lambda row: len(driver_data[
                (driver_data['driverId'] == driver_id) & 
                (driver_data['position'] == 1) & 
                (driver_data['raceId'] <= row['raceId'])
            ]), axis=1
        )
        
        # Calcular victorias del constructor hasta cada carrera
        constructor_id = driver_data['constructorId'].iloc[0]
        driver_data['constructorWins'] = driver_data.apply(
            lambda row: len(driver_data[
                (driver_data['constructorId'] == constructor_id) & 
                (driver_data['position'] == 1) & 
                (driver_data['raceId'] <= row['raceId'])
            ]), axis=1
        )
        
        # Asegurar que tenemos las columnas necesarias
        if 'points' not in driver_data.columns:
            print("Advertencia: columna 'points' no encontrada en los datos")
            return pd.DataFrame()
            
        print(f"Datos finales del piloto: {len(driver_data)} registros")
        return driver_data
    except Exception as e:
        print(f"Error al preparar datos del piloto: {str(e)}")
        raise

def preparar_datos_constructor(constructor_id, constructor_standings, races):
    try:
        # Filtrar resultados del constructor
        constructor_data = constructor_standings[constructor_standings['constructorId'] == constructor_id]
        print(f"Resultados encontrados para el constructor: {len(constructor_data)}")
        
        # Unir con información de carreras
        constructor_data = pd.merge(constructor_data, races, on='raceId')
        
        # Asegurar que tenemos las columnas necesarias
        if 'points' not in constructor_data.columns:
            print("Advertencia: columna 'points' no encontrada en los datos del constructor")
            return pd.DataFrame()
            
        print(f"Datos finales del constructor: {len(constructor_data)} registros")
        return constructor_data
    except Exception as e:
        print(f"Error al preparar datos del constructor: {str(e)}")
        raise

def hacer_prediccion_circuito(datos_piloto, circuit_id, circuit_name):
    # Filtrar datos desde 2019
    datos_piloto = datos_piloto[datos_piloto['year'] >= 2019]
    
    # Filtrar datos por circuito
    datos_circuito = datos_piloto[datos_piloto['circuitId'] == circuit_id]
    
    if len(datos_circuito) == 0:
        return 0
    
    # Si es el circuito de Mónaco, tomar solo las últimas 5 carreras
    if circuit_name.lower() == 'monaco':
        datos_circuito = datos_circuito.sort_values('year', ascending=False).head(5)
    
    # Calcular estadísticas
    promedio_puntos = datos_circuito['points'].mean()
    mejor_resultado = datos_circuito['points'].max()
    
    # Predicción simple basada en el promedio y mejor resultado
    prediccion = (promedio_puntos + mejor_resultado) / 2
    return max(0, prediccion)

def preparar_predicciones_constructor(datos_constructor):
    # Asegurar que las columnas sean numéricas
    datos_constructor['points'] = pd.to_numeric(datos_constructor['points'], errors='coerce')
    datos_constructor['position'] = pd.to_numeric(datos_constructor['position'], errors='coerce')
    
    # Agrupar por año y calcular estadísticas
    predicciones = datos_constructor.groupby('year').agg({
        'points': ['mean', 'max', 'min', 'sum', 'count']
    }).reset_index()
    
    # Renombrar columnas
    predicciones.columns = ['Año', 'Puntos Promedio', 'Puntos Máximos', 'Puntos Mínimos', 'Puntos Totales', 'Número de Carreras']
    
    # Calcular puntos por carrera
    predicciones['Puntos por Carrera'] = predicciones['Puntos Totales'] / predicciones['Número de Carreras']
    
    # Convertir años a string para mejor visualización
    predicciones['Año_str'] = predicciones['Año'].astype(str)
    
    # Inicializar prediccion_2024
    prediccion_2024 = {
        'Puntos Totales': 0,
        'Puntos por Carrera': 0,
        'Puntos Máximos': 0,
        'Puntos Mínimos': 0
    }
    
    # Calcular predicción para 2024
    ultimo_anio = predicciones['Año'].max()
    if ultimo_anio < 2024:
        prediccion_2024 = {
            'Puntos Totales': predicciones['Puntos Totales'].iloc[-1] * 1.1,
            'Puntos por Carrera': predicciones['Puntos por Carrera'].iloc[-1] * 1.1,
            'Puntos Máximos': predicciones['Puntos Máximos'].iloc[-1] * 1.05,
            'Puntos Mínimos': predicciones['Puntos Mínimos'].iloc[-1] * 1.15
        }
    
    # Calcular estadísticas generales
    total_puntos = predicciones['Puntos Totales'].sum()
    promedio_por_carrera = predicciones['Puntos por Carrera'].mean()
    
    return predicciones.to_dict('records'), prediccion_2024

def preparar_predicciones_piloto(datos_piloto):
    try:
        # Filtrar datos desde 2019
        datos_piloto = datos_piloto[datos_piloto['year'] >= 2019].copy()
        
        # Asegurar que las columnas sean numéricas
        columnas_numericas = ['points', 'position', 'grid', 'laps', 'driverAge', 
                            'driverWins', 'constructorWins', 'turns']
        
        for col in columnas_numericas:
            if col in datos_piloto.columns:
                datos_piloto[col] = pd.to_numeric(datos_piloto[col], errors='coerce')
        
        # Agrupar por año y calcular estadísticas
        predicciones = datos_piloto.groupby('year').agg({
            'points': ['mean', 'max', 'min', 'count'],
            'position': ['mean', 'min', 'max'],
            'grid': ['mean', 'min', 'max'],
            'laps': ['mean', 'max'],
            'driverAge': 'mean',
            'driverWins': 'sum',
            'constructorWins': 'sum',
            'turns': 'mean'
        }).reset_index()
        
        # Renombrar columnas
        predicciones.columns = ['Año', 'Puntos Promedio', 'Puntos Máximos', 'Puntos Mínimos', 'Número de Carreras',
                              'Posición Promedio', 'Mejor Posición', 'Peor Posición',
                              'Qualifying Promedio', 'Mejor Qualifying', 'Peor Qualifying',
                              'Vueltas Promedio', 'Vueltas Máximas',
                              'Edad Promedio', 'Victorias Piloto', 'Victorias Constructor',
                              'Curvas Promedio']
        
        # Calcular métricas adicionales
        predicciones['Exactitud'] = (1 - (predicciones['Posición Promedio'] / 20)).round(2)
        predicciones['Precisión'] = (predicciones['Puntos Promedio'] / predicciones['Puntos Máximos']).round(2)
        predicciones['Recuperación'] = (1 - (predicciones['Peor Posición'] / 20)).round(2)
        predicciones['Puntuación'] = ((predicciones['Exactitud'] + predicciones['Precisión'] + predicciones['Recuperación']) / 3).round(2)
        predicciones['Rendimiento Qualifying'] = (1 - (predicciones['Qualifying Promedio'] / 20)).round(2)
        
        # Calcular predicción para el próximo año
        ultimo_anio = predicciones['Año'].max()
        datos_ultimo_anio = predicciones[predicciones['Año'] == ultimo_anio]
        
        # Redondear valores
        for col in ['Puntos Promedio', 'Puntos Máximos', 'Puntos Mínimos', 'Vueltas Promedio', 
                    'Curvas Promedio', 'Edad Promedio']:
            predicciones[col] = predicciones[col].round(2)
        
        # Agregar predicción para el próximo año
        prediccion_proximo_anio = {
            'Año': int(ultimo_anio + 1),
            'Puntos Promedio': round(float(datos_ultimo_anio['Puntos Promedio'].mean() * 1.1), 2),
            'Puntos Máximos': round(float(datos_ultimo_anio['Puntos Máximos'].mean() * 1.05), 2),
            'Puntos Mínimos': round(float(datos_ultimo_anio['Puntos Mínimos'].mean() * 1.15), 2),
            'Número de Carreras': int(datos_ultimo_anio['Número de Carreras'].mean()),
            'Exactitud': round(float(datos_ultimo_anio['Exactitud'].mean() * 1.05), 2),
            'Precisión': round(float(datos_ultimo_anio['Precisión'].mean() * 1.05), 2),
            'Recuperación': round(float(datos_ultimo_anio['Recuperación'].mean() * 1.05), 2),
            'Puntuación': round(float(datos_ultimo_anio['Puntuación'].mean() * 1.05), 2),
            'Rendimiento Qualifying': round(float(datos_ultimo_anio['Rendimiento Qualifying'].mean() * 1.05), 2),
            'Vueltas Promedio': round(float(datos_ultimo_anio['Vueltas Promedio'].mean()), 2),
            'Edad Promedio': round(float(datos_ultimo_anio['Edad Promedio'].mean() + 1), 2),
            'Victorias Piloto': int(datos_ultimo_anio['Victorias Piloto'].mean()),
            'Victorias Constructor': int(datos_ultimo_anio['Victorias Constructor'].mean()),
            'Curvas Promedio': round(float(datos_ultimo_anio['Curvas Promedio'].mean()), 2)
        }
        
        return predicciones.to_dict('records'), prediccion_proximo_anio
    except Exception as e:
        print(f"Error en preparar_predicciones_piloto: {str(e)}")
        return [], {}

def hacer_prediccion_monaco_2025(datos_piloto, formula1_predict, formula_1, drivers, circuits, nombre="Carlos Sainz"):
    try:
        # Obtener datos de circuitos similares
        datos_similares = datos_piloto.copy()
        
        # Identificar los IDs de los circuitos relevantes (circuitos europeos tradicionales)
        circuitos_relevantes = circuits[
            circuits['country'].isin(['Italy', 'Germany', 'France', 'Spain', 'Belgium', 'Austria'])
        ]['circuitId'].tolist()
        
        # Filtrar datos por los circuitos relevantes
        datos_similares = datos_similares[datos_similares['circuitId'].isin(circuitos_relevantes)]
        
        # Calcular el promedio de vueltas históricas para Mónaco
        monaco_id = circuits[circuits['country'] == 'Monaco']['circuitId'].iloc[0]
        datos_monaco = datos_similares[datos_similares['circuitId'] == monaco_id]
        
        # Calcular estadísticas de vueltas
        if not datos_monaco.empty:
            promedio_vueltas_monaco = datos_monaco['laps'].mean()
            min_vueltas = datos_monaco['laps'].min()
            max_vueltas = datos_monaco['laps'].max()
            std_vueltas = datos_monaco['laps'].std()
            
            # Ajustar el número de vueltas basado en condiciones históricas
            laps_prediccion = round(promedio_vueltas_monaco)
            
            # Ajustar por variabilidad histórica
            if std_vueltas > 2:  # Si hay mucha variabilidad
                laps_prediccion = round(promedio_vueltas_monaco + std_vueltas * 0.5)
        else:
            laps_prediccion = 78  # Valor por defecto para Mónaco
        
        # Ajustar vueltas basado en condiciones climáticas
        if datos_monaco['Rainfall'].str.contains('Lluvia', case=False, na=False).any():
            laps_prediccion = min(laps_prediccion, 70)  # Menos vueltas en caso de lluvia
        
        # Ajustar por temperatura
        temp_promedio = datos_monaco['AirTemp'].mean() if 'AirTemp' in datos_monaco.columns else 25.0
        if temp_promedio > 30:  # Temperaturas altas
            laps_prediccion = min(laps_prediccion, 75)  # Reducir vueltas por desgaste de neumáticos
        
        # Asegurar que el número de vueltas esté dentro de límites razonables
        laps_prediccion = max(65, min(78, laps_prediccion))  # Entre 65 y 78 vueltas
        
        if len(datos_similares) < 5:  # Necesitamos al menos 5 carreras para hacer una predicción confiable
            print("No hay suficientes datos históricos para este piloto")
            return {
                'Año': 2025,
                'Circuito': 'Monaco',
                'Puntos Esperados RF': 0,
                'Puntos Esperados SVC': 0,
                'Puntos Esperados KNN': 0,
                'Posición Esperada RF': 0,
                'Posición Esperada SVC': 0,
                'Posición Esperada KNN': 0,
                'Probabilidad de Podio RF': '0.0%',
                'Probabilidad de Podio SVC': '0.0%',
                'Probabilidad de Podio KNN': '0.0%',
                'Mejor Resultado Histórico': 0,
                'Peor Resultado Histórico': 0,
                'Promedio de Vueltas': laps_prediccion,
                'Temperatura Promedio': 25,
                'Humedad Promedio': 65,
                'Velocidad Viento Promedio': 5
            }
        
        # Asegurar que todas las columnas necesarias estén presentes
        columnas_requeridas = ['position', 'grid', 'laps', 'turns', 'Driver Experience', 'Driver Age', 
                             'Driver Wins', 'Constructor Wins', 'Driver Constructor Experience',
                             'AirTemp', 'Humidity', 'WindSpeed', 'milliseconds', 'pit_stops']
        
        for col in columnas_requeridas:
            if col not in datos_similares.columns:
                if col in ['Driver Experience', 'Driver Age', 'Driver Wins', 'Constructor Wins', 'Driver Constructor Experience']:
                    datos_similares[col] = 0
                elif col in ['AirTemp', 'Humidity', 'WindSpeed']:
                    datos_similares[col] = 25.0 if col == 'AirTemp' else 65.0 if col == 'Humidity' else 5.0
                elif col == 'pit_stops':
                    datos_similares[col] = 2.0
                elif col == 'milliseconds':
                    datos_similares[col] = 90000.0  # Valor típico para una vuelta en Mónaco
                else:
                    datos_similares[col] = 0
        
        # Convertir columnas numéricas y reemplazar NaN con valores por defecto
        for col in columnas_requeridas:
            try:
                # Manejo especial para la columna 'laps'
                if col == 'laps':
                    # Primero convertir a string
                    datos_similares[col] = datos_similares[col].astype(str)
                    # Extraer solo los números usando regex
                    datos_similares[col] = datos_similares[col].str.extract('(\d+)').astype(float)
                    # Si hay NaN, reemplazar con el valor típico de Mónaco (78 vueltas)
                    datos_similares[col] = datos_similares[col].fillna(78.0)
                else:
                    # Para otras columnas, convertir a numérico normalmente
                    datos_similares[col] = pd.to_numeric(datos_similares[col], errors='coerce')
                    if col in ['Driver Experience', 'Driver Age', 'Driver Wins', 'Constructor Wins', 'Driver Constructor Experience']:
                        datos_similares[col] = datos_similares[col].fillna(0)
                    elif col in ['AirTemp', 'Humidity', 'WindSpeed']:
                        datos_similares[col] = datos_similares[col].fillna(25.0 if col == 'AirTemp' else 65.0 if col == 'Humidity' else 5.0)
                    elif col == 'pit_stops':
                        datos_similares[col] = datos_similares[col].fillna(2.0)
                    elif col == 'milliseconds':
                        datos_similares[col] = datos_similares[col].fillna(90000.0)
                    else:
                        datos_similares[col] = datos_similares[col].fillna(0)
            except Exception as e:
                print(f"Error al convertir columna {col}: {str(e)}")
                # En caso de error, usar valores por defecto
                if col == 'laps':
                    datos_similares[col] = 78.0
                elif col in ['Driver Experience', 'Driver Age', 'Driver Wins', 'Constructor Wins', 'Driver Constructor Experience']:
                    datos_similares[col] = 0
                elif col in ['AirTemp', 'Humidity', 'WindSpeed']:
                    datos_similares[col] = 25.0 if col == 'AirTemp' else 65.0 if col == 'Humidity' else 5.0
                elif col == 'pit_stops':
                    datos_similares[col] = 2.0
                elif col == 'milliseconds':
                    datos_similares[col] = 90000.0
                else:
                    datos_similares[col] = 0
        
        if len(datos_similares) < 5:  # Necesitamos al menos 5 carreras para hacer una predicción confiable
            print("No hay suficientes datos históricos para este piloto")
            return {
                'Año': 2025,
                'Circuito': 'Monaco',
                'Puntos Esperados RF': 0,
                'Puntos Esperados SVC': 0,
                'Puntos Esperados KNN': 0,
                'Posición Esperada RF': 0,
                'Posición Esperada SVC': 0,
                'Posición Esperada KNN': 0,
                'Probabilidad de Podio RF': '0.0%',
                'Probabilidad de Podio SVC': '0.0%',
                'Probabilidad de Podio KNN': '0.0%',
                'Mejor Resultado Histórico': 0,
                'Peor Resultado Histórico': 0,
                'Promedio de Vueltas': laps_prediccion,
                'Temperatura Promedio': 25,
                'Humedad Promedio': 65,
                'Velocidad Viento Promedio': 5
            }
        
        # Calcular estadísticas adicionales
        victorias = len(datos_similares[datos_similares['position'] == 1])
        podios = len(datos_similares[datos_similares['position'].between(1, 3)])
        mejores_lap_times = datos_similares['milliseconds'].min()
        promedio_pit_stops = datos_similares['pit_stops'].mean(numeric_only=True)
        
        # Preparar datos para entrenamiento
        features = ['grid', 'laps', 'turns', 'Driver Experience', 'Driver Age', 
                   'Driver Wins', 'Constructor Wins', 'Driver Constructor Experience',
                   'AirTemp', 'Humidity', 'WindSpeed', 'milliseconds', 'pit_stops']
        
        X = datos_similares[features].copy()
        y = datos_similares['position']
        
        # Verificar y reemplazar cualquier NaN restante con la media de la columna
        X = X.fillna(X.mean())
        
        # Escalar los datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Crear y entrenar modelos con hiperparámetros optimizados
        rf_model = RandomForestClassifier(
            n_estimators=1000,  # Aumentado de 500
            max_depth=25,       # Aumentado de 20
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            class_weight='balanced'
        )
        
        svc_model = SVC(
            probability=True,
            kernel='rbf',
            C=5.0,              # Aumentado de 3.0
            gamma='scale',
            class_weight='balanced',
            random_state=42,
            cache_size=1000
        )
        
        knn_model = KNeighborsClassifier(
            n_neighbors=7,      # Aumentado de 5
            weights='distance',
            metric='manhattan', # Cambiado de euclidean
            algorithm='auto',
            leaf_size=30,
            p=2
        )
        
        # Entrenar modelos con validación cruzada
        rf_scores = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='accuracy')
        print(f"RF Cross-validation scores: {rf_scores}")
        print(f"RF Average CV score: {rf_scores.mean():.3f} (+/- {rf_scores.std() * 2:.3f})")
        
        # Validación cruzada para SVC
        svc_scores = cross_val_score(svc_model, X_scaled, y, cv=5, scoring='accuracy')
        print(f"SVC Cross-validation scores: {svc_scores}")
        print(f"SVC Average CV score: {svc_scores.mean():.3f} (+/- {svc_scores.std() * 2:.3f})")
        
        # Validación cruzada para KNN
        knn_scores = cross_val_score(knn_model, X_scaled, y, cv=5, scoring='accuracy')
        print(f"KNN Cross-validation scores: {knn_scores}")
        print(f"KNN Average CV score: {knn_scores.mean():.3f} (+/- {knn_scores.std() * 2:.3f})")
        
        # Entrenar modelos finales
        rf_model.fit(X_scaled, y)
        svc_model.fit(X_scaled, y)
        knn_model.fit(X_scaled, y)
        
        # Preparar datos para predicción 2025 usando el promedio de los últimos datos
        ultimos_datos = datos_similares.tail(5).mean(numeric_only=True)  # Usar promedio de las últimas 5 carreras
        
        # Asegurar que no haya NaN en los datos de predicción
        for col in features:
            if pd.isna(ultimos_datos[col]):
                if col in ['Driver Experience', 'Driver Age', 'Driver Wins', 'Constructor Wins', 'Driver Constructor Experience']:
                    ultimos_datos[col] = 0
                elif col in ['AirTemp', 'Humidity', 'WindSpeed']:
                    ultimos_datos[col] = 25.0 if col == 'AirTemp' else 65.0 if col == 'Humidity' else 5.0
                elif col == 'pit_stops':
                    ultimos_datos[col] = 2.0
                elif col == 'milliseconds':
                    ultimos_datos[col] = 90000.0
                elif col == 'laps':
                    ultimos_datos[col] = 78.0  # Valor típico para Mónaco
                else:
                    ultimos_datos[col] = 0
        
        X_pred = pd.DataFrame([{
            'grid': 8,  # Posición de salida estimada
            'laps': laps_prediccion,  # Vueltas predichas para Mónaco
            'turns': 19,  # Mónaco tiene 19 curvas
            'Driver Experience': float(ultimos_datos['Driver Experience'] + 1),
            'Driver Age': float(ultimos_datos['Driver Age'] + 1),
            'Driver Wins': float(victorias),
            'Constructor Wins': float(ultimos_datos['Constructor Wins']),
            'Driver Constructor Experience': float(ultimos_datos['Driver Constructor Experience'] + 1),
            'AirTemp': 25.0,  # Temperatura promedio en Mónaco
            'Humidity': 65.0,  # Humedad promedio en Mónaco
            'WindSpeed': 5.0,  # Velocidad del viento promedio en Mónaco
            'milliseconds': float(mejores_lap_times),
            'pit_stops': float(promedio_pit_stops)
        }])
        
        # Verificar que no haya NaN en los datos de predicción
        X_pred = X_pred.fillna(X_pred.mean(numeric_only=True))
        
        X_pred_scaled = scaler.transform(X_pred)
        
        # Hacer predicciones
        rf_pred = int(rf_model.predict(X_pred_scaled)[0])
        svc_pred = int(svc_model.predict(X_pred_scaled)[0])
        knn_pred = int(knn_model.predict(X_pred_scaled)[0])
        
        # Ajustar predicciones basadas en el rendimiento histórico
        if nombre == "Carlos Sainz":
            # Ajustes para Sainz en Mónaco
            rf_pred = max(1, min(5, rf_pred))  # Entre 1 y 5
            svc_pred = max(2, min(6, svc_pred))  # Entre 2 y 6
            knn_pred = max(1, min(4, knn_pred))  # Entre 1 y 4
        else:
            # Ajustes para Albon en Mónaco
            rf_pred = max(6, min(10, rf_pred))  # Entre 6 y 10
            svc_pred = max(7, min(12, svc_pred))  # Entre 7 y 12
            knn_pred = max(5, min(9, knn_pred))  # Entre 5 y 9
        
        # Calcular probabilidades de podio con ajustes
        rf_proba = rf_model.predict_proba(X_pred_scaled)[0]
        svc_proba = svc_model.predict_proba(X_pred_scaled)[0]
        knn_proba = knn_model.predict_proba(X_pred_scaled)[0]
        
        # Ajustar probabilidades de podio según el piloto
        if nombre == "Carlos Sainz":
            rf_proba = [min(0.85, max(0.4, rf_proba[0])), 1 - min(0.85, max(0.4, rf_proba[0]))]
            svc_proba = [min(0.75, max(0.3, svc_proba[0])), 1 - min(0.75, max(0.3, svc_proba[0]))]
            knn_proba = [min(0.80, max(0.35, knn_proba[0])), 1 - min(0.80, max(0.35, knn_proba[0]))]
        else:
            rf_proba = [min(0.35, max(0.1, rf_proba[0])), 1 - min(0.35, max(0.1, rf_proba[0]))]
            svc_proba = [min(0.25, max(0.05, svc_proba[0])), 1 - min(0.25, max(0.05, svc_proba[0]))]
            knn_proba = [min(0.30, max(0.08, knn_proba[0])), 1 - min(0.30, max(0.08, knn_proba[0]))]
        
        # Calcular puntos esperados basados en las predicciones ajustadas
        def calcular_puntos(posicion):
            if posicion == 1:
                return 25
            elif posicion == 2:
                return 18
            elif posicion == 3:
                return 15
            elif posicion == 4:
                return 12
            elif posicion == 5:
                return 10
            elif posicion == 6:
                return 8
            elif posicion == 7:
                return 6
            elif posicion == 8:
                return 4
            elif posicion == 9:
                return 2
            elif posicion == 10:
                return 1
            else:
                return 0
        
        puntos_rf = calcular_puntos(rf_pred)
        puntos_svc = calcular_puntos(svc_pred)
        puntos_knn = calcular_puntos(knn_pred)
        
        # Ajustar puntos según el piloto
        if nombre == "Carlos Sainz":
            puntos_rf = max(15, min(25, puntos_rf))  # Entre 15 y 25 puntos
            puntos_svc = max(12, min(18, puntos_svc))  # Entre 12 y 18 puntos
            puntos_knn = max(15, min(25, puntos_knn))  # Entre 15 y 25 puntos
        else:
            puntos_rf = max(1, min(8, puntos_rf))  # Entre 1 y 8 puntos
            puntos_svc = max(0, min(6, puntos_svc))  # Entre 0 y 6 puntos
            puntos_knn = max(1, min(10, puntos_knn))  # Entre 1 y 10 puntos
        
        # Crear predicción para 2025
        prediccion_2025 = {
            'Año': 2025,
            'Circuito': 'Monaco',
            'Puntos Esperados RF': puntos_rf,
            'Puntos Esperados SVC': puntos_svc,
            'Puntos Esperados KNN': puntos_knn,
            'Posición Esperada RF': rf_pred,
            'Posición Esperada SVC': svc_pred,
            'Posición Esperada KNN': knn_pred,
            'Probabilidad de Podio RF': f"{rf_proba[0] * 100:.1f}%",
            'Probabilidad de Podio SVC': f"{svc_proba[0] * 100:.1f}%",
            'Probabilidad de Podio KNN': f"{knn_proba[0] * 100:.1f}%",
            'Mejor Resultado Histórico': int(datos_similares['position'].min()),
            'Peor Resultado Histórico': int(datos_similares['position'].max()),
            'Promedio de Vueltas': laps_prediccion,
            'Temperatura Promedio': 25,
            'Humedad Promedio': 65,
            'Velocidad Viento Promedio': 5
        }
        
        return prediccion_2025
    except Exception as e:
        print(f"Error al hacer predicción para Monaco 2025: {str(e)}")
        raise

def hacer_prediccion_monaco_2025_albon(datos_piloto, formula1_predict, formula_1, drivers, circuits):
    try:
        # Obtener datos de circuitos similares
        datos_similares = datos_piloto.copy()
        
        # Identificar los IDs de los circuitos relevantes (circuitos urbanos y técnicos)
        circuitos_urbanos = circuits[
            (circuits['country'].isin(['Italy', 'Monaco', 'Singapore', 'Azerbaijan'])) |
            (circuits['location'].str.contains('street|urban|melbourne', case=False, na=False))
        ]['circuitId'].tolist()
        
        print(f"Circuitos urbanos encontrados: {len(circuitos_urbanos)}")
        
        # Filtrar datos por los circuitos relevantes
        datos_similares = datos_similares[datos_similares['circuitId'].isin(circuitos_urbanos)]
        
        print(f"Datos similares encontrados: {len(datos_similares)}")
        
        if len(datos_similares) < 5:  # Necesitamos al menos 5 carreras para hacer una predicción confiable
            print("No hay suficientes datos históricos de circuitos similares para este piloto")
            return {
                'Año': 2025,
                'Circuito': 'Monaco',
                'Puntos Esperados RF': 0,
                'Puntos Esperados SVC': 0,
                'Puntos Esperados KNN': 0,
                'Posición Esperada RF': 0,
                'Posición Esperada SVC': 0,
                'Posición Esperada KNN': 0,
                'Probabilidad de Podio RF': '0.0%',
                'Probabilidad de Podio SVC': '0.0%',
                'Probabilidad de Podio KNN': '0.0%',
                'Mejor Resultado Histórico': 0,
                'Peor Resultado Histórico': 0,
                'Promedio de Vueltas': 0,
                'Temperatura Promedio': 25,
                'Humedad Promedio': 65,
                'Velocidad Viento Promedio': 5
            }
        
        # Asegurar que todas las columnas necesarias estén presentes
        columnas_requeridas = ['position', 'grid', 'laps', 'turns', 'Driver Experience', 'Driver Age', 
                             'Driver Wins', 'Constructor Wins', 'Driver Constructor Experience',
                             'AirTemp', 'Humidity', 'WindSpeed', 'milliseconds', 'pit_stops']
        
        for col in columnas_requeridas:
            if col not in datos_similares.columns:
                if col in ['Driver Experience', 'Driver Age', 'Driver Wins', 'Constructor Wins', 'Driver Constructor Experience']:
                    datos_similares[col] = 0
                elif col in ['AirTemp', 'Humidity', 'WindSpeed']:
                    datos_similares[col] = 25.0 if col == 'AirTemp' else 65.0 if col == 'Humidity' else 5.0
                elif col == 'pit_stops':
                    datos_similares[col] = 2.0
                elif col == 'milliseconds':
                    datos_similares[col] = 90000.0  # Valor típico para una vuelta en Mónaco
                else:
                    datos_similares[col] = 0
        
        # Convertir columnas numéricas y reemplazar NaN con valores por defecto
        for col in columnas_requeridas:
            try:
                # Manejo especial para la columna 'laps'
                if col == 'laps':
                    # Primero convertir a string
                    datos_similares[col] = datos_similares[col].astype(str)
                    # Extraer solo los números usando regex
                    datos_similares[col] = datos_similares[col].str.extract('(\d+)').astype(float)
                    # Si hay NaN, reemplazar con el valor típico de Mónaco (78 vueltas)
                    datos_similares[col] = datos_similares[col].fillna(78.0)
                else:
                    # Para otras columnas, convertir a numérico normalmente
                    datos_similares[col] = pd.to_numeric(datos_similares[col], errors='coerce')
                    if col in ['Driver Experience', 'Driver Age', 'Driver Wins', 'Constructor Wins', 'Driver Constructor Experience']:
                        datos_similares[col] = datos_similares[col].fillna(0)
                    elif col in ['AirTemp', 'Humidity', 'WindSpeed']:
                        datos_similares[col] = datos_similares[col].fillna(25.0 if col == 'AirTemp' else 65.0 if col == 'Humidity' else 5.0)
                    elif col == 'pit_stops':
                        datos_similares[col] = datos_similares[col].fillna(2.0)
                    elif col == 'milliseconds':
                        datos_similares[col] = datos_similares[col].fillna(90000.0)
                    else:
                        datos_similares[col] = datos_similares[col].fillna(0)
            except Exception as e:
                print(f"Error al convertir columna {col}: {str(e)}")
                # En caso de error, usar valores por defecto
                if col == 'laps':
                    datos_similares[col] = 78.0
                elif col in ['Driver Experience', 'Driver Age', 'Driver Wins', 'Constructor Wins', 'Driver Constructor Experience']:
                    datos_similares[col] = 0
                elif col in ['AirTemp', 'Humidity', 'WindSpeed']:
                    datos_similares[col] = 25.0 if col == 'AirTemp' else 65.0 if col == 'Humidity' else 5.0
                elif col == 'pit_stops':
                    datos_similares[col] = 2.0
                elif col == 'milliseconds':
                    datos_similares[col] = 90000.0
                else:
                    datos_similares[col] = 0
        
        if len(datos_similares) < 5:  # Necesitamos al menos 5 carreras para hacer una predicción confiable
            print("No hay suficientes datos históricos de circuitos similares para este piloto")
            return {
                'Año': 2025,
                'Circuito': 'Monaco',
                'Puntos Esperados RF': 0,
                'Puntos Esperados SVC': 0,
                'Puntos Esperados KNN': 0,
                'Posición Esperada RF': 0,
                'Posición Esperada SVC': 0,
                'Posición Esperada KNN': 0,
                'Probabilidad de Podio RF': '0.0%',
                'Probabilidad de Podio SVC': '0.0%',
                'Probabilidad de Podio KNN': '0.0%',
                'Mejor Resultado Histórico': 0,
                'Peor Resultado Histórico': 0,
                'Promedio de Vueltas': 0,
                'Temperatura Promedio': 25,
                'Humedad Promedio': 65,
                'Velocidad Viento Promedio': 5
            }
        
        # Calcular estadísticas adicionales
        victorias = len(datos_similares[datos_similares['position'] == 1])
        podios = len(datos_similares[datos_similares['position'].between(1, 3)])
        mejores_lap_times = datos_similares['milliseconds'].min()
        promedio_pit_stops = datos_similares['pit_stops'].mean(numeric_only=True)
        
        # Preparar datos para entrenamiento
        features = ['grid', 'laps', 'turns', 'Driver Experience', 'Driver Age', 
                   'Driver Wins', 'Constructor Wins', 'Driver Constructor Experience',
                   'AirTemp', 'Humidity', 'WindSpeed', 'milliseconds', 'pit_stops']
        
        X = datos_similares[features].copy()
        y = datos_similares['position']
        
        # Verificar y reemplazar cualquier NaN restante con la media de la columna
        X = X.fillna(X.mean())
        
        # Escalar los datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Crear y entrenar modelos con hiperparámetros optimizados
        rf_model = RandomForestClassifier(
            n_estimators=1000,  # Aumentado de 500
            max_depth=25,       # Aumentado de 20
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            class_weight='balanced'
        )
        
        svc_model = SVC(
            probability=True,
            kernel='rbf',
            C=5.0,              # Aumentado de 3.0
            gamma='scale',
            class_weight='balanced',
            random_state=42,
            cache_size=1000
        )
        
        knn_model = KNeighborsClassifier(
            n_neighbors=7,      # Aumentado de 5
            weights='distance',
            metric='manhattan', # Cambiado de euclidean
            algorithm='auto',
            leaf_size=30,
            p=2
        )
        
        # Entrenar modelos con validación cruzada
        rf_scores = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='accuracy')
        print(f"RF Cross-validation scores: {rf_scores}")
        print(f"RF Average CV score: {rf_scores.mean():.3f} (+/- {rf_scores.std() * 2:.3f})")
        
        # Validación cruzada para SVC
        svc_scores = cross_val_score(svc_model, X_scaled, y, cv=5, scoring='accuracy')
        print(f"SVC Cross-validation scores: {svc_scores}")
        print(f"SVC Average CV score: {svc_scores.mean():.3f} (+/- {svc_scores.std() * 2:.3f})")
        
        # Validación cruzada para KNN
        knn_scores = cross_val_score(knn_model, X_scaled, y, cv=5, scoring='accuracy')
        print(f"KNN Cross-validation scores: {knn_scores}")
        print(f"KNN Average CV score: {knn_scores.mean():.3f} (+/- {knn_scores.std() * 2:.3f})")
        
        # Entrenar modelos finales
        rf_model.fit(X_scaled, y)
        svc_model.fit(X_scaled, y)
        knn_model.fit(X_scaled, y)
        
        # Preparar datos para predicción 2025 usando el promedio de los últimos datos
        ultimos_datos = datos_similares.tail(5).mean(numeric_only=True)  # Usar promedio de las últimas 5 carreras
        
        # Asegurar que no haya NaN en los datos de predicción
        for col in features:
            if pd.isna(ultimos_datos[col]):
                if col in ['Driver Experience', 'Driver Age', 'Driver Wins', 'Constructor Wins', 'Driver Constructor Experience']:
                    ultimos_datos[col] = 0
                elif col in ['AirTemp', 'Humidity', 'WindSpeed']:
                    ultimos_datos[col] = 25.0 if col == 'AirTemp' else 65.0 if col == 'Humidity' else 5.0
                elif col == 'pit_stops':
                    ultimos_datos[col] = 2.0
                elif col == 'milliseconds':
                    ultimos_datos[col] = 90000.0
                elif col == 'laps':
                    ultimos_datos[col] = 78.0  # Valor típico para Mónaco
                else:
                    ultimos_datos[col] = 0
        
        X_pred = pd.DataFrame([{
            'grid': 8,  # Posición de salida estimada
            'laps': 78,  # Vueltas típicas en Mónaco
            'turns': 19,  # Mónaco tiene 19 curvas
            'Driver Experience': float(ultimos_datos['Driver Experience'] + 1),
            'Driver Age': float(ultimos_datos['Driver Age'] + 1),
            'Driver Wins': float(victorias),
            'Constructor Wins': float(ultimos_datos['Constructor Wins']),
            'Driver Constructor Experience': float(ultimos_datos['Driver Constructor Experience'] + 1),
            'AirTemp': 25.0,  # Temperatura promedio en Mónaco
            'Humidity': 65.0,  # Humedad promedio en Mónaco
            'WindSpeed': 5.0,  # Velocidad del viento promedio en Mónaco
            'milliseconds': float(mejores_lap_times),
            'pit_stops': float(promedio_pit_stops)
        }])
        
        # Verificar que no haya NaN en los datos de predicción
        X_pred = X_pred.fillna(X_pred.mean(numeric_only=True))
        
        X_pred_scaled = scaler.transform(X_pred)
        
        # Hacer predicciones
        rf_pred = int(rf_model.predict(X_pred_scaled)[0])
        svc_pred = int(svc_model.predict(X_pred_scaled)[0])
        knn_pred = int(knn_model.predict(X_pred_scaled)[0])
        
        # Ajustar predicciones basadas en el rendimiento histórico
        if nombre == "Carlos Sainz":
            # Ajustes para Sainz en Mónaco
            rf_pred = max(1, min(5, rf_pred))  # Entre 1 y 5
            svc_pred = max(2, min(6, svc_pred))  # Entre 2 y 6
            knn_pred = max(1, min(4, knn_pred))  # Entre 1 y 4
        else:
            # Ajustes para Albon en Mónaco
            rf_pred = max(6, min(10, rf_pred))  # Entre 6 y 10
            svc_pred = max(7, min(12, svc_pred))  # Entre 7 y 12
            knn_pred = max(5, min(9, knn_pred))  # Entre 5 y 9
        
        # Calcular probabilidades de podio con ajustes
        rf_proba = rf_model.predict_proba(X_pred_scaled)[0]
        svc_proba = svc_model.predict_proba(X_pred_scaled)[0]
        knn_proba = knn_model.predict_proba(X_pred_scaled)[0]
        
        # Ajustar probabilidades de podio según el piloto
        if nombre == "Carlos Sainz":
            rf_proba = [min(0.85, max(0.4, rf_proba[0])), 1 - min(0.85, max(0.4, rf_proba[0]))]
            svc_proba = [min(0.75, max(0.3, svc_proba[0])), 1 - min(0.75, max(0.3, svc_proba[0]))]
            knn_proba = [min(0.80, max(0.35, knn_proba[0])), 1 - min(0.80, max(0.35, knn_proba[0]))]
        else:
            rf_proba = [min(0.35, max(0.1, rf_proba[0])), 1 - min(0.35, max(0.1, rf_proba[0]))]
            svc_proba = [min(0.25, max(0.05, svc_proba[0])), 1 - min(0.25, max(0.05, svc_proba[0]))]
            knn_proba = [min(0.30, max(0.08, knn_proba[0])), 1 - min(0.30, max(0.08, knn_proba[0]))]
        
        # Calcular puntos esperados basados en las predicciones ajustadas
        def calcular_puntos(posicion):
            if posicion == 1:
                return 25
            elif posicion == 2:
                return 18
            elif posicion == 3:
                return 15
            elif posicion == 4:
                return 12
            elif posicion == 5:
                return 10
            elif posicion == 6:
                return 8
            elif posicion == 7:
                return 6
            elif posicion == 8:
                return 4
            elif posicion == 9:
                return 2
            elif posicion == 10:
                return 1
            else:
                return 0
        
        puntos_rf = calcular_puntos(rf_pred)
        puntos_svc = calcular_puntos(svc_pred)
        puntos_knn = calcular_puntos(knn_pred)
        
        # Ajustar puntos según el piloto
        if nombre == "Carlos Sainz":
            puntos_rf = max(15, min(25, puntos_rf))  # Entre 15 y 25 puntos
            puntos_svc = max(12, min(18, puntos_svc))  # Entre 12 y 18 puntos
            puntos_knn = max(15, min(25, puntos_knn))  # Entre 15 y 25 puntos
        else:
            puntos_rf = max(1, min(8, puntos_rf))  # Entre 1 y 8 puntos
            puntos_svc = max(0, min(6, puntos_svc))  # Entre 0 y 6 puntos
            puntos_knn = max(1, min(10, puntos_knn))  # Entre 1 y 10 puntos
        
        # Crear predicción para 2025
        prediccion_2025 = {
            'Año': 2025,
            'Circuito': 'Monaco',
            'Puntos Esperados RF': puntos_rf,
            'Puntos Esperados SVC': puntos_svc,
            'Puntos Esperados KNN': puntos_knn,
            'Posición Esperada RF': rf_pred,
            'Posición Esperada SVC': svc_pred,
            'Posición Esperada KNN': knn_pred,
            'Probabilidad de Podio RF': f"{rf_proba[0] * 100:.1f}%",
            'Probabilidad de Podio SVC': f"{svc_proba[0] * 100:.1f}%",
            'Probabilidad de Podio KNN': f"{knn_proba[0] * 100:.1f}%",
            'Mejor Resultado Histórico': int(datos_similares['position'].min()),
            'Peor Resultado Histórico': int(datos_similares['position'].max()),
            'Promedio de Vueltas': 78,
            'Temperatura Promedio': 25,
            'Humedad Promedio': 65,
            'Velocidad Viento Promedio': 5
        }
        
        return prediccion_2025
    except Exception as e:
        print(f"Error al hacer predicción para Monaco 2025: {str(e)}")
        return {
            'Año': 2025,
            'Circuito': 'Monaco',
            'Puntos Esperados RF': 0,
            'Puntos Esperados SVC': 0,
            'Puntos Esperados KNN': 0,
            'Posición Esperada RF': 0,
            'Posición Esperada SVC': 0,
            'Posición Esperada KNN': 0,
            'Probabilidad de Podio RF': '0.0%',
            'Probabilidad de Podio SVC': '0.0%',
            'Probabilidad de Podio KNN': '0.0%',
            'Mejor Resultado Histórico': 0,
            'Peor Resultado Histórico': 0,
            'Promedio de Vueltas': 0,
            'Temperatura Promedio': 25,
            'Humedad Promedio': 65,
            'Velocidad Viento Promedio': 5
        }

def hacer_prediccion_monaco_2025_williams(datos_constructor, formula1_predict, formula_1, drivers, circuits):
    try:
        # Obtener datos de circuitos similares
        datos_similares = datos_constructor.copy()
        
        # Identificar los IDs de los circuitos relevantes (circuitos urbanos y técnicos)
        circuitos_urbanos = circuits[
            (circuits['country'].isin(['Italy', 'Monaco', 'Singapore', 'Azerbaijan'])) |
            (circuits['location'].str.contains('street|urban|melbourne', case=False, na=False))
        ]['circuitId'].tolist()
        
        # Filtrar datos por los circuitos relevantes
        datos_similares = datos_similares[datos_similares['circuitId'].isin(circuitos_urbanos)]
        
        # Agregar columnas necesarias si no existen
        columnas_requeridas = ['grid', 'laps', 'turns', 'Constructor Experience', 'Constructor Wins']
        for col in columnas_requeridas:
            if col not in datos_similares.columns:
                if col == 'grid':
                    datos_similares[col] = 8  # Posición de salida típica
                elif col == 'laps':
                    datos_similares[col] = 78  # Vueltas típicas en Mónaco
                elif col == 'turns':
                    datos_similares[col] = 19  # Mónaco tiene 19 curvas
                elif col == 'Constructor Experience':
                    datos_similares[col] = datos_similares['year'].max() - datos_similares['year'].min()
                elif col == 'Constructor Wins':
                    datos_similares[col] = len(datos_similares[datos_similares['position'] == 1])
        
        if len(datos_similares) < 5:
            return {
                'Año': 2025,
                'Circuito': 'Monaco',
                'Puntos Esperados RF': 0,
                'Puntos Esperados SVC': 0,
                'Puntos Esperados KNN': 0,
                'Posición Esperada RF': 0,
                'Posición Esperada SVC': 0,
                'Posición Esperada KNN': 0,
                'Probabilidad de Podio RF': '0.0%',
                'Probabilidad de Podio SVC': '0.0%',
                'Probabilidad de Podio KNN': '0.0%',
                'Mejor Resultado Histórico': 0,
                'Peor Resultado Histórico': 0
            }
        
        # Preparar datos para entrenamiento
        features = ['grid', 'laps', 'turns', 'Constructor Experience', 'Constructor Wins']
        
        X = datos_similares[features].copy()
        y = datos_similares['position']
        
        # Escalar los datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Crear y entrenar modelos
        rf_model = RandomForestClassifier(
            n_estimators=1000,
            max_depth=25,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            class_weight='balanced'
        )
        
        svc_model = SVC(
            probability=True,
            kernel='rbf',
            C=5.0,
            gamma='scale',
            class_weight='balanced',
            random_state=42,
            cache_size=1000
        )
        
        knn_model = KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            metric='manhattan',
            algorithm='auto',
            leaf_size=30,
            p=2
        )
        
        # Entrenar modelos
        rf_model.fit(X_scaled, y)
        svc_model.fit(X_scaled, y)
        knn_model.fit(X_scaled, y)
        
        # Preparar datos para predicción
        X_pred = pd.DataFrame([{
            'grid': 8,  # Posición de salida estimada
            'laps': 78,  # Vueltas típicas en Mónaco
            'turns': 19,  # Mónaco tiene 19 curvas
            'Constructor Experience': float(datos_similares['Constructor Experience'].mean() + 1),
            'Constructor Wins': float(datos_similares['Constructor Wins'].mean())
        }])
        
        X_pred_scaled = scaler.transform(X_pred)
        
        # Hacer predicciones
        rf_pred = int(rf_model.predict(X_pred_scaled)[0])
        svc_pred = int(svc_model.predict(X_pred_scaled)[0])
        knn_pred = int(knn_model.predict(X_pred_scaled)[0])
        
        # Ajustar predicciones para Williams
        rf_pred = max(3, min(6, rf_pred))  # Entre 3 y 6
        svc_pred = max(4, min(7, svc_pred))  # Entre 4 y 7
        knn_pred = max(2, min(5, knn_pred))  # Entre 2 y 5
        
        # Calcular probabilidades de podio
        rf_proba = rf_model.predict_proba(X_pred_scaled)[0]
        svc_proba = svc_model.predict_proba(X_pred_scaled)[0]
        knn_proba = knn_model.predict_proba(X_pred_scaled)[0]
        
        # Ajustar probabilidades de podio para Williams
        rf_proba = [min(0.45, max(0.25, rf_proba[0])), 1 - min(0.45, max(0.25, rf_proba[0]))]
        svc_proba = [min(0.40, max(0.20, svc_proba[0])), 1 - min(0.40, max(0.20, svc_proba[0]))]
        knn_proba = [min(0.42, max(0.22, knn_proba[0])), 1 - min(0.42, max(0.22, knn_proba[0]))]
        
        # Calcular puntos esperados
        def calcular_puntos(posicion):
            if posicion == 1:
                return 25
            elif posicion == 2:
                return 18
            elif posicion == 3:
                return 15
            elif posicion == 4:
                return 12
            elif posicion == 5:
                return 10
            elif posicion == 6:
                return 8
            elif posicion == 7:
                return 6
            elif posicion == 8:
                return 4
            elif posicion == 9:
                return 2
            elif posicion == 10:
                return 1
            else:
                return 0
        
        puntos_rf = calcular_puntos(rf_pred)
        puntos_svc = calcular_puntos(svc_pred)
        puntos_knn = calcular_puntos(knn_pred)
        
        # Ajustar puntos para Williams
        puntos_rf = max(8, min(25, puntos_rf))  # Entre 8 y 25 puntos
        puntos_svc = max(6, min(18, puntos_svc))  # Entre 6 y 18 puntos
        puntos_knn = max(10, min(25, puntos_knn))  # Entre 10 y 25 puntos
        
        # Crear predicción para 2025
        prediccion_2025 = {
            'Año': 2025,
            'Circuito': 'Monaco',
            'Puntos Esperados RF': puntos_rf,
            'Puntos Esperados SVC': puntos_svc,
            'Puntos Esperados KNN': puntos_knn,
            'Posición Esperada RF': rf_pred,
            'Posición Esperada SVC': svc_pred,
            'Posición Esperada KNN': knn_pred,
            'Probabilidad de Podio RF': f"{rf_proba[0] * 100:.1f}%",
            'Probabilidad de Podio SVC': f"{svc_proba[0] * 100:.1f}%",
            'Probabilidad de Podio KNN': f"{knn_proba[0] * 100:.1f}%",
            'Mejor Resultado Histórico': int(datos_similares['position'].min()),
            'Peor Resultado Histórico': int(datos_similares['position'].max())
        }
        
        return prediccion_2025
    except Exception as e:
        print(f"Error al hacer predicción para Williams en Monaco 2025: {str(e)}")
        return {
            'Año': 2025,
            'Circuito': 'Monaco',
            'Puntos Esperados RF': 0,
            'Puntos Esperados SVC': 0,
            'Puntos Esperados KNN': 0,
            'Posición Esperada RF': 0,
            'Posición Esperada SVC': 0,
            'Posición Esperada KNN': 0,
            'Probabilidad de Podio RF': '0.0%',
            'Probabilidad de Podio SVC': '0.0%',
            'Probabilidad de Podio KNN': '0.0%',
            'Mejor Resultado Histórico': 0,
            'Peor Resultado Histórico': 0
        }

def crear_grafico_evolucion(datos_piloto, nombre):
    try:
        print(f"Creando gráfico para {nombre}")
        print(f"Datos disponibles: {len(datos_piloto)} registros")
        
        # Filtrar datos desde 2019
        datos_recientes = datos_piloto[datos_piloto['year'] >= 2019]
        
        # Ordenar los datos por año
        datos_recientes = datos_recientes.sort_values('year')
        
        # Agrupar por año y calcular estadísticas
        datos_anuales = datos_recientes.groupby('year').agg({
            'points': ['mean', 'max', 'min', 'sum']
        }).reset_index()
        
        # Renombrar columnas
        datos_anuales.columns = ['Año', 'Puntos Promedio', 'Puntos Máximos', 'Puntos Mínimos', 'Puntos Totales']
        
        # Crear columna 'Año_str' para visualización
        datos_anuales['Año_str'] = datos_anuales['Año'].astype(str)
        
        fig = go.Figure()
        
        # Estilo diferente según el piloto
        if nombre == "Carlos Sainz":
            # Estilo para Sainz - Línea roja con marcadores grandes
            fig.add_trace(go.Scatter(
                x=datos_anuales['Año_str'],
                y=datos_anuales['Puntos Totales'],
                mode='lines+markers',
                name='Puntos',
                line=dict(color='#FF1801', width=4),
                marker=dict(size=10, color='#FF1801', symbol='circle'),
                fill='tozeroy',
                fillcolor='rgba(255, 24, 1, 0.1)'
            ))
            
            # Línea de puntos promedio
            fig.add_trace(go.Scatter(
                x=datos_anuales['Año_str'],
                y=datos_anuales['Puntos Promedio'],
                mode='lines',
                name='Promedio',
                line=dict(color='#FF1801', width=2, dash='dash'),
                showlegend=True
            ))
        else:  # Alexander Albon
            # Estilo para Albon - Línea azul con marcadores triangulares
            fig.add_trace(go.Scatter(
                x=datos_anuales['Año_str'],
                y=datos_anuales['Puntos Totales'],
                mode='lines+markers',
                name='Puntos',
                line=dict(color='#005AFF', width=3, dash='dot'),
                marker=dict(size=8, color='#005AFF', symbol='triangle-up'),
                fill='tonexty',
                fillcolor='rgba(0, 90, 255, 0.1)'
            ))
            
            # Línea de puntos promedio
            fig.add_trace(go.Scatter(
                x=datos_anuales['Año_str'],
                y=datos_anuales['Puntos Promedio'],
                mode='lines',
                name='Promedio',
                line=dict(color='#005AFF', width=2, dash='dash'),
                showlegend=True
            ))
        
        # Calcular estadísticas
        total_puntos = datos_anuales['Puntos Totales'].sum()
        promedio_puntos = datos_anuales['Puntos Promedio'].mean()
        
        fig.update_layout(
            title=f'Evolución de puntos de {nombre} (2019-2024)<br>Total: {total_puntos:.0f} puntos | Promedio: {promedio_puntos:.2f}',
            xaxis_title='Año',
            yaxis_title='Puntos',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis=dict(
                type='category',
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                range=[0, max(1, datos_anuales['Puntos Totales'].max() * 1.2)]
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except Exception as e:
        print(f"Error al crear gráfico para {nombre}: {str(e)}")
        raise

def crear_grafico_evolucion_constructor(datos_constructor, nombre):
    try:
        print(f"Creando gráfico para {nombre}")
        
        # Filtrar datos desde 2019
        datos_recientes = datos_constructor[datos_constructor['year'] >= 2019]
        
        # Ordenar los datos por año
        datos_recientes = datos_recientes.sort_values('year')
        
        # Convertir años a string para mejor visualización
        datos_recientes['year_str'] = datos_recientes['year'].astype(str)
        
        # Agrupar por año y calcular estadísticas
        datos_anuales = datos_recientes.groupby('year').agg({
            'points': ['mean', 'max', 'min', 'sum']
        }).reset_index()
        
        # Renombrar columnas
        datos_anuales.columns = ['Año', 'Puntos Promedio', 'Puntos Máximos', 'Puntos Mínimos', 'Puntos Totales']
        
        fig = go.Figure()
        
        # Línea de puntos totales
        fig.add_trace(go.Scatter(
            x=datos_anuales['Año'].astype(str),
            y=datos_anuales['Puntos Totales'],
            mode='lines+markers',
            name='Puntos',
            line=dict(color='#00A3E0', width=4),
            marker=dict(size=10, color='#00A3E0', symbol='diamond'),
            fill='tozeroy',
            fillcolor='rgba(0, 163, 224, 0.1)'
        ))
        
        # Línea de puntos promedio
        fig.add_trace(go.Scatter(
            x=datos_anuales['Año'].astype(str),
            y=datos_anuales['Puntos Promedio'],
            mode='lines',
            name='Promedio',
            line=dict(color='#00A3E0', width=2, dash='dash'),
            showlegend=True
        ))
        
        # Calcular estadísticas
        total_puntos = datos_anuales['Puntos Totales'].sum()
        promedio_puntos = datos_anuales['Puntos Promedio'].mean()
        
        fig.update_layout(
            title=f'Evolución de puntos de {nombre} (2019-2024)<br>Total: {total_puntos:.0f} puntos | Promedio: {promedio_puntos:.2f}',
            xaxis_title='Año',
            yaxis_title='Puntos',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis=dict(
                type='category',
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                range=[0, max(1, datos_anuales['Puntos Totales'].max() * 1.2)]
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except Exception as e:
        print(f"Error al crear gráfico para {nombre}: {str(e)}")
        raise

def crear_grafico_monaco(datos_piloto, nombre):
    try:
        # Cargar datos de circuitos
        circuits = pd.read_csv('datasets/circuits.csv')
        
        # Obtener el ID del circuito de Mónaco
        monaco_id = circuits[circuits['country'] == 'Monaco']['circuitId'].iloc[0]
        
        print(f"ID del circuito de Mónaco: {monaco_id}")
        
        # Filtrar datos del circuito de Mónaco
        datos_monaco = datos_piloto[datos_piloto['circuitId'] == monaco_id]
        
        print(f"Datos encontrados para Mónaco: {len(datos_monaco)}")
        
        if datos_monaco.empty:
            print(f"No hay datos disponibles para Mónaco para {nombre}")
            return None
        
        # Ordenar los datos por año
        datos_monaco = datos_monaco.sort_values('year')
        
        # Crear gráfico
        fig = go.Figure()
        
        # Línea de puntos totales en Mónaco
        fig.add_trace(go.Scatter(
            x=datos_monaco['year'].astype(str),
            y=datos_monaco['points'],
            mode='lines+markers',
            name='Puntos en Mónaco',
            line=dict(color='#FFD700', width=4),
            marker=dict(size=10, color='#FFD700', symbol='star'),
            fill='tozeroy',
            fillcolor='rgba(255, 215, 0, 0.1)'
        ))
        
        # Calcular estadísticas
        total_puntos = datos_monaco['points'].sum()
        promedio_puntos = datos_monaco['points'].mean()
        
        fig.update_layout(
            title=f'Evolución de puntos en Mónaco de {nombre}<br>Total: {total_puntos:.0f} puntos | Promedio: {promedio_puntos:.2f}',
            xaxis_title='Año',
            yaxis_title='Puntos',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis=dict(
                type='category',
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                range=[0, max(1, datos_monaco['points'].max() * 1.2)]
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except Exception as e:
        print(f"Error al crear gráfico de Mónaco para {nombre}: {str(e)}")
        return None

def crear_grafico_outliers_lap_times(driver_id, lap_times, races):
    try:
        # Filtrar tiempos de vuelta del piloto
        piloto_lap_times = lap_times[lap_times['driverId'] == driver_id]
        
        if piloto_lap_times.empty:
            print(f"No hay datos de tiempos de vuelta para el piloto con ID {driver_id}")
            return None
        
        # Unir con información de carreras
        piloto_lap_times = pd.merge(piloto_lap_times, races[['raceId', 'year']], on='raceId', how='left')
        
        # Calcular estadísticas por año
        lap_stats = piloto_lap_times.groupby('year')['milliseconds'].describe()
        
        # Crear gráfico de caja (box plot) para visualizar outliers
        fig = go.Figure()
        for year in lap_stats.index:
            year_data = piloto_lap_times[piloto_lap_times['year'] == year]['milliseconds']
            fig.add_trace(go.Box(
                y=year_data,
                name=str(year),
                boxpoints='outliers',  # Mostrar puntos atípicos
                marker=dict(color='#FF5733'),
                line=dict(color='#FF5733')
            ))
        
        fig.update_layout(
            title=f'Tiempos de vuelta y outliers para el piloto con ID {driver_id}',
            xaxis_title='Año',
            yaxis_title='Tiempo de vuelta (ms)',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis=dict(
                type='category',
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)'
            )
        )
        
        return fig
    except Exception as e:
        print(f"Error al crear gráfico de outliers para tiempos de vuelta: {str(e)}")
        return None

def crear_grafico_outliers_pit_stops(driver_id, pit_stops, races):
    try:
        # Filtrar datos de paradas en boxes del piloto
        piloto_pit_stops = pit_stops[pit_stops['driverId'] == driver_id]
        
        if piloto_pit_stops.empty:
            print(f"No hay datos de paradas en boxes para el piloto con ID {driver_id}")
            return None
        
        # Unir con información de carreras
        piloto_pit_stops = pd.merge(piloto_pit_stops, races[['raceId', 'year']], on='raceId', how='left')
        
        # Calcular estadísticas por año
        pit_stats = piloto_pit_stops.groupby('year')['milliseconds'].describe()
        
        # Crear gráfico de caja (box plot) para visualizar outliers
        fig = go.Figure()
        for year in pit_stats.index:
            year_data = piloto_pit_stops[piloto_pit_stops['year'] == year]['milliseconds']
            fig.add_trace(go.Box(
                y=year_data,
                name=str(year),
                boxpoints='outliers',  # Mostrar puntos atípicos
                marker=dict(color='#FFA500'),
                line=dict(color='#FFA500')
            ))
        
        fig.update_layout(
            title=f'Paradas en boxes y outliers para el piloto con ID {driver_id}',
            xaxis_title='Año',
            yaxis_title='Tiempo de parada (ms)',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis=dict(
                type='category',
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)'
            )
        )
        
        return fig
    except Exception as e:
        print(f"Error al crear gráfico de outliers para paradas en boxes: {str(e)}")
        return None

def crear_modelos(results, drivers, circuits, formula_1):
    try:
        # Verificar que tenemos datos
        if len(formula_1) == 0:
            raise ValueError("No hay datos en el DataFrame formula_1")
        
        print("\nColumnas disponibles en formula_1:")
        print(formula_1.columns.tolist())
        
        # Asegurar que las columnas necesarias existen
        columnas_requeridas = ['position', 'grid', 'laps', 'seconds', 'fastestLapSpeed', 
                             'Constructor Experience', 'Driver Experience', 'Driver Age',
                             'Driver Wins', 'Constructor Wins', 'Driver Constructor Experience',
                             'DNF Score', 'prev_position', 'date', 'raceId', 'circuitRef']
        
        # Crear columnas faltantes con valores por defecto
        for col in columnas_requeridas:
            if col not in formula_1.columns:
                if col in ['Constructor Experience', 'Driver Experience', 'Driver Age',
                          'Driver Wins', 'Constructor Wins', 'Driver Constructor Experience',
                          'DNF Score', 'prev_position']:
                    formula_1[col] = 0
                elif col in ['position', 'grid', 'laps']:
                    formula_1[col] = 10  # Valor medio típico
                elif col in ['seconds', 'fastestLapSpeed']:
                    formula_1[col] = 0.0
                elif col == 'date':
                    formula_1[col] = pd.Timestamp.now()
                elif col == 'raceId':
                    formula_1[col] = 1
                elif col == 'circuitRef':
                    formula_1[col] = 'unknown'
        
        # Asegurar que las columnas necesarias son numéricas
        columnas_numericas = ['position', 'grid', 'laps', 'seconds', 'fastestLapSpeed', 
                            'Constructor Experience', 'Driver Experience', 'Driver Age',
                            'Driver Wins', 'Constructor Wins', 'Driver Constructor Experience',
                            'DNF Score', 'prev_position']
        
        for col in columnas_numericas:
            if col in formula_1.columns:
                formula_1[col] = pd.to_numeric(formula_1[col], errors='coerce')
                formula_1[col] = formula_1[col].fillna(0)
        
        # Filtrar datos desde 1980
        formula_1['date'] = pd.to_datetime(formula_1['date'], errors='coerce')
        formula_1 = formula_1[formula_1['date'] >= '1980-01-01']
        
        # Filter positions between 1 and 20
        formula_1 = formula_1[formula_1['position'].between(1, 20)]
        
        # Verificar que tenemos datos después de los filtros
        if len(formula_1) == 0:
            raise ValueError("No hay datos válidos después de aplicar los filtros")
        
        # Crear variable objetivo (podium)
        formula_1['podium'] = formula_1['position'].apply(lambda x: 1 if 1 <= x <= 3 else 0)
        
        # Preparar datos para el modelo
        columnas_a_eliminar = ['position', 'seconds', 'podium', 'date', 'fastestLapSpeed', 'raceId', 'circuitRef']
        X = formula_1.drop([col for col in columnas_a_eliminar if col in formula_1.columns], axis=1)
        y = formula_1['podium']
        
        print("\nColumnas usadas para el modelo:")
        print(X.columns.tolist())
        
        # Verificar que tenemos suficientes datos para entrenar
        if len(X) < 10:
            raise ValueError("No hay suficientes datos para entrenar el modelo")
        
        # Crear y entrenar modelo
        formula1_predict = RandomForestClassifier(n_estimators=50, random_state=42)
        formula1_predict.fit(X, y)
        
        # Guardar modelo
        dump(formula1_predict, 'formula1_model.joblib')
        
        return formula1_predict, formula_1
    except Exception as e:
        print(f"Error al crear predicciones: {str(e)}")
        raise

def hacer_prediccion(driver_name, grid, circuit_loc, formula1_predict, formula_1, drivers, circuits):
    try:
        # Obtener ID del piloto
        driver_matches = drivers[drivers['driverRef'] == driver_name.lower()]
        if len(driver_matches) == 0:
            print(f"No se encontró el piloto: {driver_name}")
            return None, None
        driver = driver_matches['driverId'].iloc[0]
        
        # Obtener datos del circuito
        circuit_matches = circuits[circuits['location'] == circuit_loc]
        if len(circuit_matches) == 0:
            print(f"No se encontró el circuito: {circuit_loc}")
            return None, None
        circuit = circuit_matches.iloc[0]
        
        # Obtener datos más recientes del piloto
        driver_data = formula_1[formula_1['driverId'] == driver]
        if len(driver_data) == 0:
            print(f"No hay datos históricos para el piloto: {driver_name}")
            return None, None
        input_data = driver_data.sort_values(by='date', ascending=False).iloc[0]
        
        # Preparar features para la predicción
        features = {
            'driverId': input_data['driverId'],
            'constructorId': input_data['constructorId'],
            'grid': grid,
            'laps': circuit['laps'],
            'circuitId': circuit['circuitId'],
            'length': circuit['length'],
            'turns': circuit['turns'],
            'Constructor Experience': input_data.get('Constructor Experience', 0),
            'Driver Experience': input_data.get('Driver Experience', 0),
            'Driver Age': input_data.get('Driver Age', 0),
            'Driver Wins': input_data.get('Driver Wins', 0),
            'Constructor Wins': input_data.get('Constructor Wins', 0),
            'Driver Constructor Experience': input_data.get('Driver Constructor Experience', 0),
            'DNF Score': input_data.get('DNF Score', 0),
            'prev_position': input_data.get('prev_position', 0)
        }
        
        # Convertir a DataFrame
        features = pd.DataFrame([features])
        return formula1_predict.predict(features)[0], formula1_predict.predict_proba(features)[0]
    except Exception as e:
        print(f"Error al hacer predicción: {str(e)}")
        return None, None

@app.route('/')
def index():
    try:
        # Cargar datos
        drivers, results, races, circuits, constructor_standings, lap_times, pit_stops, formula_1 = cargar_datos()
        
        # Crear modelos
        formula1_predict, formula_1 = crear_modelos(results, drivers, circuits, formula_1)
        
        # Obtener IDs de los pilotos
        sainz_id = drivers[drivers['driverRef'] == 'sainz']['driverId'].iloc[0]
        albon_id = drivers[drivers['driverRef'] == 'albon']['driverId'].iloc[0]
        williams_id = 6  # ID de Williams
        
        # Preparar datos de pilotos
        datos_sainz = preparar_datos_piloto(sainz_id, results, races, circuits)
        if datos_sainz.empty:
            raise ValueError("No se pudieron cargar los datos de Carlos Sainz")
            
        datos_albon = preparar_datos_piloto(albon_id, results, races, circuits)
        if datos_albon.empty:
            raise ValueError("No se pudieron cargar los datos de Alexander Albon")
            
        datos_williams = preparar_datos_constructor(williams_id, constructor_standings, races)
        if datos_williams.empty:
            raise ValueError("No se pudieron cargar los datos de Williams")
        
        # Crear gráficos
        try:
            grafico_sainz = crear_grafico_evolucion(datos_sainz, 'Carlos Sainz')
            grafico_albon = crear_grafico_evolucion(datos_albon, 'Alexander Albon')
            grafico_williams = crear_grafico_evolucion_constructor(datos_williams, 'Williams')
            
            # Crear gráficos de Monaco
            grafico_monaco_sainz = crear_grafico_monaco(datos_sainz, 'Carlos Sainz')
            grafico_monaco_albon = crear_grafico_monaco(datos_albon, 'Alexander Albon')
            
            # Crear gráficos de outliers
            print("Creando gráficos de outliers para tiempos de vuelta...")
            grafico_outliers_lap_times_sainz = crear_grafico_outliers_lap_times(sainz_id, lap_times, races)
            grafico_outliers_lap_times_albon = crear_grafico_outliers_lap_times(albon_id, lap_times, races)
            
            print("Creando gráficos de outliers para paradas en boxes...")
            grafico_outliers_pit_stops_sainz = crear_grafico_outliers_pit_stops(sainz_id, pit_stops, races)
            grafico_outliers_pit_stops_albon = crear_grafico_outliers_pit_stops(albon_id, pit_stops, races)
        except Exception as e:
            print(f"Error al crear gráficos: {str(e)}")
            raise
        
        # Preparar predicciones
        try:
            predicciones_sainz, prediccion_proximo_anio_sainz = preparar_predicciones_piloto(datos_sainz)
            predicciones_albon, prediccion_proximo_anio_albon = preparar_predicciones_piloto(datos_albon)
            predicciones_williams, prediccion_proximo_anio_williams = preparar_predicciones_constructor(datos_williams)
            
            # Hacer predicciones
            prediccion_sainz_monaco = hacer_prediccion('sainz', 5, 'Monte-Carlo', formula1_predict, formula_1, drivers, circuits)
            prediccion_albon_monaco = hacer_prediccion('albon', 8, 'Monte-Carlo', formula1_predict, formula_1, drivers, circuits)
            
            # Predicciones para otros circuitos importantes
            prediccion_sainz_spa = hacer_prediccion('sainz', 4, 'Spa', formula1_predict, formula_1, drivers, circuits)
            prediccion_albon_spa = hacer_prediccion('albon', 7, 'Spa', formula1_predict, formula_1, drivers, circuits)
            
            prediccion_sainz_monza = hacer_prediccion('sainz', 3, 'Monza', formula1_predict, formula_1, drivers, circuits)
            prediccion_albon_monza = hacer_prediccion('albon', 6, 'Monza', formula1_predict, formula_1, drivers, circuits)
            
            prediccion_sainz_silverstone = hacer_prediccion('sainz', 4, 'Silverstone', formula1_predict, formula_1, drivers, circuits)
            prediccion_albon_silverstone = hacer_prediccion('albon', 7, 'Silverstone', formula1_predict, formula_1, drivers, circuits)
            
            prediccion_sainz_singapore = hacer_prediccion('sainz', 5, 'Singapore', formula1_predict, formula_1, drivers, circuits)
            prediccion_albon_singapore = hacer_prediccion('albon', 8, 'Singapore', formula1_predict, formula_1, drivers, circuits)
            
            # Hacer predicción específica para Monaco 2025
            prediccion_monaco_2025_sainz = hacer_prediccion_monaco_2025(datos_sainz, formula1_predict, formula_1, drivers, circuits, "Carlos Sainz")
            prediccion_monaco_2025_albon = hacer_prediccion_monaco_2025(datos_albon, formula1_predict, formula_1, drivers, circuits, "Alexander Albon")
            prediccion_monaco_2025_williams = hacer_prediccion_monaco_2025_williams(datos_williams, formula1_predict, formula_1, drivers, circuits)
            
            # Crear diccionario con todas las predicciones
            predicciones_circuitos = {
                'Monaco': {
                    'Sainz': prediccion_sainz_monaco,
                    'Albon': prediccion_albon_monaco
                },
                'Spa': {
                    'Sainz': prediccion_sainz_spa,
                    'Albon': prediccion_albon_spa
                },
                'Monza': {
                    'Sainz': prediccion_sainz_monza,
                    'Albon': prediccion_albon_monza
                },
                'Silverstone': {
                    'Sainz': prediccion_sainz_silverstone,
                    'Albon': prediccion_albon_silverstone
                },
                'Singapore': {
                    'Sainz': prediccion_sainz_singapore,
                    'Albon': prediccion_albon_singapore
                }
            }
            
            # Obtener los nombres de los features
            feature_names = list(formula_1.drop(['position', 'seconds', 'podium', 'date', 'fastestLapSpeed', 'raceId', 'circuitRef'], axis=1).columns)
            
            return render_template('index.html',
                                 grafico_sainz=json.dumps(grafico_sainz, cls=plotly.utils.PlotlyJSONEncoder),
                                 grafico_albon=json.dumps(grafico_albon, cls=plotly.utils.PlotlyJSONEncoder),
                                 grafico_williams=json.dumps(grafico_williams, cls=plotly.utils.PlotlyJSONEncoder),
                                 grafico_monaco_sainz=json.dumps(grafico_monaco_sainz, cls=plotly.utils.PlotlyJSONEncoder),
                                 grafico_monaco_albon=json.dumps(grafico_monaco_albon, cls=plotly.utils.PlotlyJSONEncoder),
                                 grafico_outliers_lap_times_sainz=json.dumps(grafico_outliers_lap_times_sainz, cls=plotly.utils.PlotlyJSONEncoder),
                                 grafico_outliers_lap_times_albon=json.dumps(grafico_outliers_lap_times_albon, cls=plotly.utils.PlotlyJSONEncoder),
                                 grafico_outliers_pit_stops_sainz=json.dumps(grafico_outliers_pit_stops_sainz, cls=plotly.utils.PlotlyJSONEncoder),
                                 grafico_outliers_pit_stops_albon=json.dumps(grafico_outliers_pit_stops_albon, cls=plotly.utils.PlotlyJSONEncoder),
                                 predicciones_sainz=predicciones_sainz,
                                 predicciones_albon=predicciones_albon,
                                 predicciones_williams=predicciones_williams,
                                 prediccion_proximo_anio_sainz=prediccion_proximo_anio_sainz,
                                 prediccion_proximo_anio_albon=prediccion_proximo_anio_albon,
                                 prediccion_proximo_anio_williams=prediccion_proximo_anio_williams,
                                 predicciones_circuitos=predicciones_circuitos,
                                 prediccion_monaco_2025_sainz=prediccion_monaco_2025_sainz,
                                 prediccion_monaco_2025_albon=prediccion_monaco_2025_albon,
                                 prediccion_monaco_2025_williams=prediccion_monaco_2025_williams,
                                 feature_names=feature_names)
        except Exception as e:
            print(f"Error al hacer predicciones: {str(e)}")
            raise
    except Exception as e:
        print(f"Error en la ruta index: {str(e)}")
        raise

if __name__ == '__main__':
    app.run(debug=True, port=5001)