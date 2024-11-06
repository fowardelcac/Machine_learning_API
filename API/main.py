from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional

from io import StringIO
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import max_error, mean_absolute_error, r2_score


class StoredData(BaseModel):
    data: Optional[dict] = None
    columns: Optional[dict] = None
    x_train: Optional[dict] = None
    x_test: Optional[dict] = None
    y_train: Optional[dict] = None
    y_test: Optional[dict] = None
    sklearn_model: Optional[object] = None


class SplitParams(BaseModel):
    columns: Optional[list] = None
    target: Optional[str] = None
    test_size: float = 0.3


app = FastAPI()
stored_data = StoredData()


@app.get("/")
async def root():
    return {"Hello": "World"}


@app.get("/data/{key}")
async def get_data(key: str):
    """
    Retrieve a specific data item based on the provided key.

    Args:
        key (str): The key representing the item to retrieve from `stored_data`.

    Returns:
        dict: A dictionary containing the key and its associated value from `stored_data`.

    Raises:
        HTTPException: If the value associated with the provided key is not found
        or is None, a 404 HTTP exception is raised with a detail message "Item is empty".
    """
    # getattr(object, attribute_name, default_value)
    value = getattr(stored_data, key)

    if value is None:
        raise HTTPException(status_code=404, detail="Item is empty")

    return {key: value}


@app.post("/uploadfile/")
async def upload__and_split_file(file: UploadFile):
    """
    Uploads a CSV file, processes its content, and stores the data in a structured format.

    Args:
        file (UploadFile): The CSV file to be uploaded. The file is expected to be in
                            CSV format.

    Returns:
        dict: A response containing the status of the upload, the filename,
              content type, and the parsed data.

    Raises:
        HTTPException:
            - 415 if the uploaded file is not a CSV (based on file extension).
            - 500 if an error occurs during the file processing or CSV parsing.
    """

    content = await file.read()

    if not file.filename.endswith("csv"):
        raise HTTPException(status_code=415, detail="Only CSV files are allowed.")

    try:
        # Decode the content to UTF-8 and parse the CSV
        decoded_count = content.decode("utf-8")
        df = pd.read_csv(StringIO(decoded_count))
        # Store the data in a structured format
        stored_data.data = df.to_dict(orient="records")
        stored_data.columns = df.columns.tolist()

        return {
            "status": "Success.",
            "filename": file.filename,
            "content_type": file.content_type,
            "data": stored_data.data,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during the upload. More about: {e}"
        )


@app.post("/split/")
def split_model(input_params: SplitParams):
    """
    Splits the dataset stored in `stored_data` into training and test sets for features (X)
    and target (y) columns as specified by the user.

    Args:
        input_params (SplitParams): The parameters used to split the dataset. This includes:
            - `columns`: List of column names to be used as features (X).
            - `target`: The column name to be used as the target variable (y).
            - `test_size`: The proportion of the dataset to be used as the test set (between 0 and 1).

    Returns:
        dict: A response containing the status of the split operation, along with the
              resulting training and test sets for both features and target variables:
            - `x_train`: Training set for features (X).
            - `x_test`: Test set for features (X).
            - `y_train`: Training set for target (y).
            - `y_test`: Test set for target (y).

    Raises:
        HTTPException:
            - 404 if `stored_data.data` is `None`, indicating no data has been uploaded yet.
            - 500 if an error occurs during the dataset splitting process (e.g., invalid columns or data issues).
    """
    if stored_data.data is None:
        raise HTTPException(status_code=404, detail="Item not found")
    try:
        df = pd.DataFrame(stored_data.data)
        X = df[input_params.columns]
        y = df[input_params.target]

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=input_params.test_size, random_state=42
        )

        stored_data.x_train = x_train.to_dict(orient="records")
        stored_data.x_test = x_test.to_dict(orient="records")
        stored_data.y_train = y_train.tolist()
        stored_data.y_test = y_test.tolist()
        return {
            "status": "Success",
            "x_train": stored_data.x_train,
            "x_test": stored_data.x_test,
            "y_train": stored_data.y_train,
            "y_test": stored_data.y_test,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during the dataset split. More about: {e}"
        )


@app.post("/train/")
def train_model():
    """
    Trains a machine learning model (RandomForestRegressor) using the stored training data
    (`x_train` and `y_train`) and saves the trained model in `stored_data`.

    This endpoint verifies if the training data has been preprocessed and stored in `stored_data`.

    Returns:
        dict: A response containing the status of the training operation and the model parameters.
            - `status`: Indicates the training status (e.g., "Trained").
            - `model info`: The parameters of the trained model.

    Raises:
        HTTPException:
            - 404 if `data` and `x_train` and `y_train` are missing from `stored_data`.
            - 500 if an error occurs during the training process.
    """
    if verify_store_data("data", "x_train", "y_train"):
        raise HTTPException(status_code=404, detail="Missing training data")
    try:
        x_train = pd.DataFrame(stored_data.x_train)
        y_train = stored_data.y_train

        stored_data.sklearn_model = RandomForestRegressor().fit(x_train, y_train)
        params = stored_data.sklearn_model.get_params()
        return {"status": "Trained", "model info": params}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during the training. More about: {e}"
        )


@app.post("/predict/")
def predict_model():
    """
    Makes predictions using the trained machine learning model (`sklearn_model`)
    and evaluates the predictions using regression metrics.

    The function uses the test dataset (`x_test`) and the trained model (`sklearn_model`)
    stored in `stored_data` to generate predictions and then computes evaluation metrics
    (max error, mean absolute error, R-squared).

    Returns:
        dict: A response containing the prediction status and evaluation metrics.
            - `status`: Indicates the status of the prediction (e.g., "Predicted").
            - `metrics`: A tuple of evaluation metrics (max error, mean absolute error, R-squared).

    Raises:
        HTTPException:
            - 404 if the required training data (`x_test`) or the trained model (`sklearn_model`) is missing.
            - 500 if an error occurs during the prediction process (e.g., model prediction failure).
    """
    if verify_store_data("data", "x_test", "sklearn_model"):
        raise HTTPException(status_code=404, detail="Missing training or model data")
    try:
        x_test = pd.DataFrame(stored_data.x_test)
        y_test = stored_data.y_test

        y_pred = stored_data.sklearn_model.predict(x_test)
        results = metrics_regression(y_test, y_pred)
        return {"status": "Predicted", "metrics": results}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during the prediction. More about: {e}"
        )


def verify_store_data(key1: str, key2: str, key3: str):
    """
    Verifies that the required keys exist and are not `None` in the `stored_data` object.

    Args:
        key1 (str): The first key to check.
        key2 (str): The second key to check.
        key3 (str): The third key to check.

    Returns:
        bool: `True` if any of the keys in `stored_data` are missing (i.e., `None`), otherwise `False`.
    """
    value1 = getattr(stored_data, key1)
    value2 = getattr(stored_data, key2)
    value3 = getattr(stored_data, key3)

    return value1 is None or value2 is None or value3 is None


def metrics_regression(y_test, y_pred):
    """
    Calculates the evaluation metrics for regression tasks, including:
    - Max error: The largest absolute difference between true and predicted values.
    - Mean absolute error (MAE): The average of the absolute differences between true and predicted values.
    - R-squared (R²): A measure of how well the predictions approximate the true values.

    Args:
        y_test (list or array-like): The actual target values (real values).
        y_pred (list or array-like): The predicted target values.

    Returns:
        tuple: A tuple containing the following metrics:
            - max_error (float): The maximum error between the true and predicted values.
            - mae (float): The mean absolute error.
            - r2 (float): The R-squared value indicating the fit of the model.
    """
    error_max = max_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return error_max, mae, r2
