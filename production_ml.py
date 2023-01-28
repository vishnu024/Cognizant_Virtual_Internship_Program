import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error



# Loading the data
def load_data(path: str = "/path/to/csv/"):
    """
    This function takes a path string to a CSV file and loads it into
    a Pandas DataFrame.
    :param      path (optional): str, relative path of the CSV file
    :return     df: pd.DataFrame
    """

    df = pd.read_csv(f"{path}")
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return df


# Create target variable and predictor variables
def create_target_and_predictors(
        data: pd.DataFrame = None,
        target: str = "estimated_stock_pct"
):
    """
    This function takes in a Pandas DataFrame and splits the columns
    into a target column and a set of predictor variables, i.e. X & y.
    These two splits of the data will be used to train a supervised
    machine learning model.
    :param      data: pd.DataFrame, dataframe containing data for the
                      model
    :param      target: str (optional), target variable that you want to predict
    :return     X: pd.DataFrame
                y: pd.Series
    """

    # Check to see if the target variable is present in the data
    if target not in data.columns:
        raise Exception(f"Target: {target} is not present in the data")

    X = data.drop(columns=[target])
    y = data[target]
    return X, y

# Split data
def split_data(X, y):
    """
    This function takes in target column and a set of predictor variables and
    splits them into training and validation dataset.
    Ideally, ML models use 80% of the overall data for training and 20% for validation.
    random_state controls the shuffling applied to the data before applying the split.
    Pass an int for reproducible output across multiple function calls.
    :X    set of predictor variables
    :y    target column
    :return     X_train: pd.DataFrame
                X_test: pd.DataFrame
                y_train: pd.Series
                y_test: pd.Series
    """

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Training the algorithm
def train_algorithm(
        X_train: pd.DataFrame = None,
        y_train: pd.Series = None,
        X_test: pd.DataFrame = None,
        y_test: pd.Series = None
):
    """
    This function takes the X_train, y_train, X_valid and y_valid and
    trains an extreme gradient boosting XGBoost model.
    :param      X_train: pd.DataFrame, training predictor variable
                X_valid: pd.DataFrame, testing predictor variable
                y_train: pd.Series, training target variable
                y_valid: pd.Series, testing target variable
    :return    mean absolute error
    """
    my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
    my_model.fit(X_train, y_train,
                 early_stopping_rounds=5,
                 eval_set=[(X_valid, y_valid)],
                 verbose=False)
    predictions = my_model.predict(X_valid)
    mae = mean_absolute_error(y_valid, predictions)
    return mae


def main():
    # Assuming that the data is already cleaned and processed data is stored in data.csv file
    df = load_data("/path/to/csv/")
    target = "estimated_stock_pct"
    X, y = create_target_and_predictors(df, target)
    X_train, X_test, y_train, y_test = split_data(X, y)
    mae = train_algorithm(X_train, X_test, y_train, y_test)
    print("Mean Absolute Error:", mae)


if __name__ == "__main__":
    main()