import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error


DATA = "data.csv"
X_COL_NAMES = ['Bedrooms', 'Bathrooms', 'Size', 'Location']
Y_COL_NAME = "Price"


def load_data(x_columns):
    data = pd.read_csv(DATA)

    y = data[Y_COL_NAME]
    x = data[x_columns]

    if 'Location' in x_columns:
        location = pd.get_dummies(data['Location'])
        x = pd.concat([x, location], axis=1)
        x.drop('Location', axis=1, inplace=True)

    return train_test_split(x, y, test_size=0.2, random_state=10)


def run_model(x_columns):
    x_train, x_test, y_train, y_test = load_data(x_columns)

    model = LinearRegression(normalize=True)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    test_error = np.sqrt(mean_squared_error(y_test, y_pred))

    y_train_pred = model.predict(x_train)
    train_error = np.sqrt(mean_squared_error(y_train, y_train_pred))
    return test_error, train_error


def plot_results(columns, test_error, train_error):

    merged_columns = ['\n'.join(col) for col in columns]

    test_error, train_error, merged_columns = zip(*reversed(sorted(zip(test_error, train_error, merged_columns))))

    ax = plt.gca()

    ind = np.arange(len(merged_columns))
    width = 0.4

    ax.bar(ind,
           test_error,
           width,
           align='center',
           color='r',
           tick_label=merged_columns,
           label='Test Error')

    ax.bar(ind + width,
           train_error,
           width,
           align='center',
           color='y',
           label='Train Error')

    ax.legend()

    plt.xlabel("Variables Used")
    plt.ylabel("Error")
    plt.title("House Price Prediction")

    plt.show()


def main():
    col_permutations = [list(itertools.combinations(X_COL_NAMES, i)) for i in range(1, len(X_COL_NAMES) + 1)]
    col_permutations = [list(item) for sublist in col_permutations for item in sublist]

    test_errors = []
    train_errors = []

    for perm in col_permutations:
        test_error, train_error = run_model(perm)
        test_errors.append(test_error)
        train_errors.append(train_error)

    plot_results(col_permutations, test_errors, train_errors)

if __name__ == "__main__":
    main()
