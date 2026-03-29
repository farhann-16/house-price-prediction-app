import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


MODEL_FILE = "models/model.pkl"
PIPELINE_FILE = "models/pipeline.pkl"


def load_data():
    """Load dataset safely (local or fallback URL)"""

    local_path = "data/housing.csv"

    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    else:
        # fallback (public dataset)
        url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
        return pd.read_csv(url)


def train():
    """Train model and save artifacts"""

    print("Training started...")

    df = load_data()

    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create sample input for app
    os.makedirs("input", exist_ok=True)
    X_test.head(50).to_csv("input/sample_input.csv", index=False)

    # Column separation
    numerical_col = X_train.select_dtypes(include=["number"]).columns
    category_col = X_train.select_dtypes(include=["object"]).columns

    # Pipelines
    numerical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    category_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numerical_pipe, numerical_col),
            ("cat", category_pipe, category_col)
        ],
        verbose_feature_names_out=False
    )

    preprocess.set_output(transform="pandas")

    # Fit preprocessing
    X_train_processed = preprocess.fit_transform(X_train)

    # ✅ Lightweight model (cloud-friendly)
    model = LinearRegression()

    model.fit(X_train_processed, y_train)

    # Evaluation
    X_test_processed = preprocess.transform(X_test)
    predictions = model.predict(X_test_processed)

    print("MAE:", mean_absolute_error(y_test, predictions))
    print("R2:", r2_score(y_test, predictions))

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(preprocess, PIPELINE_FILE)

    print("Model trained and saved successfully.")


def predict(input_df):
    """Run predictions on new data"""

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    processed = pipeline.transform(input_df)
    predictions = model.predict(processed)

    input_df["prediction"] = predictions

    return input_df