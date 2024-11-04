from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from io import StringIO
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, max_error, mean_absolute_error


def metrics(y_test, y_pred):
    error_max = max_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return error_max, mae, r2


class SplitRequest(BaseModel):
    columns: list
    target: str
    test_size: float = 0.3


class ParamsModelRequest(BaseModel):
    params: dict


app = FastAPI()


stored_data = {
    "data": None,
    "columns": None,
    "x_train": None,
    "x_test": None,
    "y_train": None,
    "y_test": None,
    "model_fitted": None,
}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/data/")
def get_data():
    if stored_data["data"] is None:
        return {"message": "No data available."}
    return {
        "data": stored_data["data"],
        "columns": stored_data["columns"],
        "x_train": stored_data["x_train"],
        "x_test": stored_data["x_test"],
        "y_train": stored_data["y_train"],
        "y_test": stored_data["y_test"],
        "model_fitted": stored_data["model_fitted"],
    }


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile | None = None):
    if not file:
        return {"message": "No upload file sent"}
    else:
        content = await file.read()
        decoded_content = content.decode("utf-8")

        # Use StringIO to read the CSV file into a DataFrame
        df: pd.DataFrame = pd.read_csv(StringIO(decoded_content))
        stored_data["data"] = df.to_dict(orient="records")
        stored_data["columns"] = df.columns.tolist()

        return {
            "status": "Success!",
            "filename": file.filename,
            "content_type": file.content_type,
            "data": stored_data["data"],
            "columns": stored_data["columns"],
        }


@app.post("/split/")
def train_split(request: SplitRequest):
    df = pd.DataFrame(stored_data["data"])
    X = df[request.columns]
    y = df[request.target]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=request.test_size
    )
    stored_data["x_train"] = x_train.to_dict(orient="records")
    stored_data["x_test"] = x_test.to_dict(orient="records")
    stored_data["y_train"] = y_train.tolist()
    stored_data["y_test"] = y_test.tolist()
    return {"status": "Splitted!"}


@app.post("/train/")
def train_model(params: ParamsModelRequest):
    x_train = pd.DataFrame(stored_data["x_train"])
    y_train = stored_data["y_train"]

    sklearn_model = RandomForestRegressor(**params.params).fit(x_train, y_train)
    stored_data["model_fitted"] = sklearn_model
    return {
        "status": "trained!",
    }


@app.post("/predict/")
def predict_model():
    if stored_data["model_fitted"] is None:
        return {"message": "Model has not been trained."}

    x_test = pd.DataFrame(stored_data["x_test"])
    y_test = stored_data["y_test"]

    y_pred = stored_data["model_fitted"].predict(x_test)
    error_max, mae, r2 = metrics(y_test, y_pred)
    return {
        "status": "Predicted!",
        "Metrics": {"Error max": error_max, "MAE": mae, "R2": r2},
    }
