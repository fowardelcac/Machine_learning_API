# API de Procesamiento y Entrenamiento de Modelos con FastAPI

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-teal.svg) ![License](https://img.shields.io/badge/licencia-MIT-green.svg)

Este proyecto es una API desarrollada con **FastAPI** que permite subir archivos CSV, procesarlos, dividirlos en conjuntos de entrenamiento y prueba, entrenar un modelo de regresión (RandomForestRegressor), y realizar predicciones con evaluación de métricas. La API almacena los datos procesados y el modelo entrenado en memoria para su uso posterior, y está acompañada de pruebas unitarias para garantizar su correcto funcionamiento.

## Propósito

El objetivo de este proyecto es proporcionar una interfaz RESTful para:
- Cargar y procesar archivos CSV.
- Dividir los datos en conjuntos de entrenamiento y prueba.
- Entrenar un modelo de regresión basado en Random Forest.
- Realizar predicciones y evaluar el rendimiento del modelo con métricas como error máximo, error absoluto medio y R².

El proyecto incluye un flujo completo de aprendizaje automático y está diseñado para ser escalable y fácil de probar.


### Estructura de Archivos

```
machine-learning-api/
├── main.py              # Lógica principal de la API con endpoints y funciones
├── test_main.py         # Pruebas unitarias para los endpoints
└── README.md            # Documentación del proyecto
```


### Dependencias

El proyecto depende de las siguientes bibliotecas:
- `fastapi>=0.100.0`: Framework para construir la API.
- `uvicorn`: Servidor ASGI para ejecutar FastAPI.
- `pandas`: Manipulación de datos.
- `scikit-learn`: Modelos y métricas de aprendizaje automático.
- `pydantic`: Validación de datos.

## Uso

### Endpoints de la API

La API ofrece los siguientes endpoints:

1. **GET `/`**  
   - **Descripción**: Retorna un mensaje de bienvenida ("Hello World").
   - **Respuesta**: `{"Hello": "World"}`.

2. **GET `/data/{key}`**  
   - **Descripción**: Recupera un elemento específico almacenado en `stored_data` basado en la clave proporcionada.
   - **Parámetros**: `key` (str) - Clave del dato a recuperar.
   - **Respuesta**: `{key: value}` o error 404 si el valor es `None`.

3. **POST `/uploadfile/`**  
   - **Descripción**: Sube un archivo CSV, lo procesa y almacena los datos.
   - **Cuerpo**: Archivo CSV (`file`).
   - **Respuesta**: `{status, filename, content_type, data}` o error 415 (solo CSV) o 500 (error de procesamiento).
   - **Ejemplo de uso**:
     ```bash
     curl -X POST "http://127.0.0.1:8000/uploadfile/" -F "file=@/ruta/a/tu/archivo.csv"
     ```

4. **POST `/split/`**  
   - **Descripción**: Divide los datos cargados en conjuntos de entrenamiento y prueba.
   - **Cuerpo**: JSON con `columns` (lista), `target` (str), `test_size` (float, default 0.3).
   - **Respuesta**: `{status, x_train, x_test, y_train, y_test}` o error 404 (sin datos) o 500 (error de división).
   - **Ejemplo de uso**:
     ```bash
     curl -X POST "http://127.0.0.1:8000/split/" -H "Content-Type: application/json" -d '{"columns": ["age", "sex", "bmi"], "target": "Target", "test_size": 0.3}'
     ```

5. **POST `/train/`**  
   - **Descripción**: Entrena un modelo RandomForestRegressor con los datos de entrenamiento.
   - **Respuesta**: `{status, model info}` o error 404 (sin datos) o 500 (error de entrenamiento).

6. **POST `/predict/`**  
   - **Descripción**: Realiza predicciones con el modelo entrenado y evalúa las métricas.
   - **Respuesta**: `{status, metrics}` (max_error, mae, r2) o error 404 (sin datos/modelo) o 500 (error de predicción).

### Flujo de Uso

1. Sube un archivo CSV usando `/uploadfile/`.
2. Divide los datos con `/split/` especificando columnas y objetivo.
3. Entrena el modelo con `/train/`.
4. Realiza predicciones con `/predict/`.

### Pruebas

Ejecuta las pruebas unitarias para verificar la funcionalidad:
```bash
pytest test_main.py -v
```

Las pruebas cubren:
- Respuesta del endpoint raíz (`/`).
- Comportamiento de `/data/{key}` sin datos.
- Subida de archivos válidos e inválidos (`/uploadfile/`).
- División de datos (`/split/`).
- Entrenamiento del modelo (`/train/`).
- Predicciones y métricas (`/predict/`).

Asegúrate de ajustar las rutas de los archivos CSV en `test_main.py` según tu entorno (por ejemplo, `/home/fowardelcac/mlApp/API/Dataset/Diabetes.csv`).

## Características

- **Validación de Archivos**: Solo acepta archivos CSV.
- **Gestión de Datos**: Almacena datos procesados y modelos en memoria.
- **Entrenamiento y Predicción**: Utiliza RandomForestRegressor con evaluación de métricas.
- **Manejo de Errores**: Responde con códigos HTTP adecuados (404, 415, 500).

