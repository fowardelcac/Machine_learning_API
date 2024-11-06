from fastapi.testclient import TestClient
from main import app, stored_data
import pandas as pd

client = TestClient(app)

url: str = "http://127.0.0.1:8000"


def test_root():
    """
    Tests the root endpoint (`/`) to ensure the server is running
    and returns the expected greeting message.

    Verifies that the response is a JSON containing:
    {"Hello": "World"}.
    """
    response = client.get("http://127.0.0.1:8000/")
    assert response.json() == {"Hello": "World"}


def test_not_initial_get_data():
    """
    Tests the `/data/data` endpoint when no data has been uploaded yet.

    Verifies that when no data is available, the server returns a 404 error
    with the detail message "Item is empty".
    """
    response = client.get(url=url + "/data/data")
    assert response.status_code == 404
    assert response.json() == {"detail": "Item is empty"}


def loader(file_path: str):
    """
    Helper function to load a file and return the appropriate payload
    for uploading the file.

    Args:
        file_path (str): The path to the file to be uploaded.

    Returns:
        dict: A dictionary containing the file in the proper format for upload.
    """
    return {"file": (open(file_path, "rb"))}


def upload_data(
    csv_path: str = "/home/fowardelcac/mlApp/API/Dataset/Diabetes.csv",
):
    """
    Uploads a CSV file to the API. Uses the `loader` function to prepare
    the file before sending it in a POST request.

    Args:
        csv_path (str): The path to the CSV file to upload (defaults to a specific diabetes dataset).

    Returns:
        response: The API response to the file upload request.
    """
    files = loader(csv_path)
    return client.post("/uploadfile/", files=files)


def complete_proccess():
    """
    Runs the complete processing flow, including uploading data,
    splitting the dataset into training and test sets, and training the model.

    First, it uploads the data, then performs the split, and finally trains the model.

    Returns:
        response: The API response after training the model.
    """
    upload_data()
    cols = ["age", "sex", "bmi"]
    body = {"columns": cols, "target": "Target", "test_size": 0.3}

    client.post("/split/", json=body)

    return client.post("/train/")


def dataset_basic():
    """
    Loads the basic dataset (CSV) and returns it as a DataFrame, along with the column names
    and the target variable.

    Returns:
        tuple: A tuple containing the DataFrame, column names list, and target variable values.
    """
    df = pd.read_csv(
        "/home/fowardelcac/mlApp/API/Dataset/Diabetes.csv",
    )
    cols = df.columns.tolist()
    target = df.Target
    return df, cols, target


def test_upload_file():
    """
    Tests the file upload endpoint with both invalid and valid file types.

    1. First, it tests with a non-CSV file (Excel file) to ensure it returns a 415 error.
    2. Then, it tests with a valid CSV file to check that the upload is successful.

    Verifies that the response status code and the returned message are correct in both cases.
    """
    excel_path = "/home/fowardelcac/mlApp/API/Dataset/diab_excel.xlsx"
    response = upload_data(excel_path)
    assert response.status_code == 415
    assert response.json() == {"detail": "Only CSV files are allowed."}
    print(response.json())

    # Test with a valid CSV file
    response = upload_data()
    assert response.status_code == 200
    assert response.json()["status"] == "Success."

    response = client.get(url=url + "/data/columns")
    assert response.status_code == 200

    df, cols, ignore = dataset_basic()
    assert response.json() == {"columns": cols}
    print(stored_data.columns)


def test_split():
    """
    Tests the `/split/` endpoint to ensure the dataset is correctly split into training and test sets.

    Verifies that:
    1. The split process returns a success status.
    2. If no data is uploaded (i.e., `stored_data.data` is `None`), the endpoint returns a 404 error.
    """
    cols = ["age", "sex", "bmi"]
    body = {"columns": cols, "target": "Target", "test_size": 0.3}

    response = client.post("/split/", json=body)
    assert response.status_code == 200
    assert response.json()["status"] == "Success"

    stored_data.data = None
    response = client.post("/split/", json=body)
    assert response.status_code == 404


def test_train():
    """
    Tests the `/train/` endpoint to ensure the model is trained correctly.

    Verifies that:
    1. The model is trained successfully.
    2. If no training data is available, a 404 error is returned with the message "Missing training data".
    """
    response = complete_proccess()
    assert response.status_code == 200
    assert response.json()["status"] == "Trained"
    assert stored_data.sklearn_model is not None

    # Clear data to trigger 404
    stored_data.data = None
    stored_data.x_train = None
    stored_data.y_train = None

    response = client.post("/train/")
    assert response.status_code == 404
    assert "Missing training data" in response.json()["detail"]


def test_train():
    """
    Tests the `/predict/` endpoint to ensure predictions are made correctly.

    Verifies that:
    1. Predictions are generated successfully after training.
    2. If no data or trained model is available, a 404 error is returned with the message "Missing training or model data".
    """
    response = complete_proccess()
    prediction_response = client.post("/predict/")
    assert prediction_response.status_code == 200
    assert prediction_response.json()["status"] == "Predicted"

    # Clear data to trigger 404
    stored_data.data = None
    stored_data.x_test = None
    stored_data.sklearn_model = None

    response = client.post("/predict/")
    assert response.status_code == 404
    assert "Missing training or model data" in response.json()["detail"]
