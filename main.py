import argparse
import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score 

def main(alpha_value, solver_value):
    # Load the diabetes dataset (a regression dataset)
    X, y = datasets.load_diabetes(return_X_y=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create a Ridge Regression model with specified hyperparameters
    ridge = Ridge(alpha=alpha_value, solver=solver_value)

    # Train the model
    ridge.fit(X_train, y_train)

    # Predict on the test set
    y_pred = ridge.predict(X_test)

    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the results
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    # Set our tracking server URI (if needed)
    # mlflow.set_tracking_uri()

    # Create a new MLflow Experiment
    mlflow.set_experiment("MLflow Ridge Regression Quickstart")

    # Start an MLflow run
    with mlflow.start_run():
        # Log the model hyperparameters
        mlflow.log_params({"alpha": alpha_value, "solver": solver_value})

        # Log the metrics
        mlflow.log_metric("mean_squared_error", mse)
        mlflow.log_metric("r2_score", r2)

        # Set a tag for this run
        mlflow.set_tag("Model Type", "Ridge Regression")

        # Infer the model signature
        signature = infer_signature(X_train, ridge.predict(X_train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=ridge,
            artifact_path="ridge_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="ridge-regression-quickstart",
        )

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run Ridge Regression with specified hyperparameters.")
    parser.add_argument("alpha", type=float, help="Regularization strength")
    parser.add_argument("solver", type=str, help="Solver to use (e.g., 'auto', 'svd', 'cholesky', etc.)")

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.alpha, args.solver)
