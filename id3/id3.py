import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from copy import copy
from node import Node
import matplotlib.pyplot as plt

HEIGTH_SEPARATOR = 50
WEIGHT_SEPARATOR = 40
API_SEPARATOR = 30
AGE_SEPARATOR = 3000
TRAIN_SIZE = 0.7
VALID_SIZE = 0.66


class Solver():
    """A solver. Parameters may be passed during initialization."""

    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        ...

    def fit(self, X: pd.DataFrame, Y: pd.Series, depth):
        """
        A method that fits the solver to the given data.
        X is the dataset without the class attribute.
        y contains the class attribute for each sample from X.
        It may return anything.
        """
        ...

        if (Y.iloc[0] == Y).all():
            node = Node(value=int(Y.iloc[0]))
            return node

        if len(X.columns) == 0 or depth == 0:
            more_frequent_value = Y.mode().astype(int)
            node = Node(more_frequent_value)
            return node

        d = max_inf_gain(X, Y)  # d - most significant attribute (column)
        
        node = Node(attr=d)
        for current_value in X[d].unique():
            X_Y = copy(X)
            X_Y[Y.name] = Y
            X_Y_current_val = X_Y.loc[X_Y[d] == current_value] 
            Y_current_val = X_Y_current_val[Y.name]
            X_current_val = X_Y_current_val.drop(Y.name, axis=1)
            node.add_child(current_value, self.fit(X_current_val, Y_current_val, depth - 1))
        return node

    def predict(self, X: pd.DataFrame, root: Node):
        """
        A method that returns predicted class for each row of X
        """
        predictions = X.apply(lambda row: self.row_prediction(row, root), axis=1)
        return predictions

    def row_prediction(self, test_row: pd.Series, root: Node):
        if root.is_leaf():
            return root.value
        current_row_attr = root.attr
        current_val = test_row[current_row_attr]
        for attr_val in root.children:
            if current_val == attr_val:
                return self.row_prediction(test_row, root.children.get(current_val))
        attribute_values = root.children.keys()
        best_value = min(attribute_values, key=lambda x: abs(current_val - x))
        return self.row_prediction(test_row, root.children.get(best_value))


def max_inf_gain(data_x: pd.DataFrame, data_y: pd.Series): 
    y_values_amount = len(data_y)
    sum = 0
    for unique_value in data_y.unique():
        count_occur = data_y[data_y == unique_value].count()
        cur_entropy = count_occur/y_values_amount*np.log2(count_occur/y_values_amount)
        sum -= cur_entropy
    entropy_y = sum
    d = None
    max_inf_gain = 0
    for attr in data_x.columns:
        inf_gain = entropy_y - column_entropy(data_x, data_y, attr) 
        if inf_gain >= max_inf_gain:
            max_inf_gain = inf_gain
            d = attr
    return d


def column_entropy(all_data_x: pd.DataFrame, all_data_y: pd.Series, column: str):
    x_data = copy(all_data_x)
    x_data[all_data_y.name] = all_data_y
    column_values = all_data_x[column].unique()
    ret = 0
    dataset_length = len(all_data_x)
    for val in column_values:
        x_data_single_val = x_data.loc[x_data[column] == val]
        value_probablity = len(x_data_single_val)/dataset_length
        temp_entropy = value_probablity * uninque_value_entropy(x_data_single_val, all_data_y)
        ret += temp_entropy
    return ret


def uninque_value_entropy(x_single_values: pd.DataFrame, data_y: pd.Series):
    ret = 0
    uniq_values = data_y.unique()    # cardio values {0, 1}
    for uniq_value in uniq_values:
        sum = 0
        uni_val_count = len(x_single_values)
        try:
            y_value_count = x_single_values[data_y.name].value_counts()[uniq_value]
        except KeyError:
            y_value_count = 0
        if y_value_count != 0:
            sum = y_value_count/uni_val_count * np.log2(y_value_count/uni_val_count)
        ret -= sum
    return ret


def make_dataframe(filename, separator):
    data_frame = pd.read_csv(filename, sep=separator)
    return data_frame


def prepare_dataframe(dataframe: pd.DataFrame):
    dataframe = dataframe.drop("id", axis=1)
    dataframe = dataframe.dropna(axis="index")
    # reduce unique values
    dataframe['age'] = dataframe['age'].apply(lambda x: x // AGE_SEPARATOR)
    dataframe['height'] = dataframe['height'].apply(lambda x: x // HEIGTH_SEPARATOR)
    dataframe['weight'] = dataframe['weight'].apply(lambda x: int(x // WEIGHT_SEPARATOR))
    dataframe['ap_hi'] = dataframe['ap_hi'].apply(lambda x: int(x // API_SEPARATOR))
    dataframe['ap_lo'] = dataframe['ap_lo'].apply(lambda x: int(x // API_SEPARATOR))
    x = copy(dataframe.drop(columns=['cardio']))
    y = dataframe['cardio']
    x_train, x_rem, y_train, y_rem = train_test_split(x, y, train_size=TRAIN_SIZE)
    x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, test_size=VALID_SIZE)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def find_best_depth(dataframe):
    x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_dataframe(dataframe)
    sol = Solver()
    # calculate train accuracy
    depths = list(range(1, 4))
    train_result = []
    for depthh in depths:
        root = sol.fit(x_train, y_train, depthh)
        result_of_tree = sol.predict(x_train, root)
        cleaned = result_of_tree.dropna(axis=1, how='any').squeeze()
        result = cleaned.compare(y_train, keep_shape=True)
        print(f"For Tree depth: {depthh}. Accuracy: { (result['self'].isna().sum() / len(result)) * 100} %")
        train_result.append((result['self'].isna().sum() / len(result)) * 100)
    # calculate valid accuarcy
    valid_result = []
    for depthh in depths:
        root = sol.fit(x_train, y_train, depthh)
        result_of_tree = sol.predict(x_valid, root)
        cleaned = result_of_tree.dropna(axis=1, how='any').squeeze()
        result = cleaned.compare(y_valid, keep_shape=True)
        print(f"For Tree depth: {depthh}. Accuracy: { (result['self'].isna().sum() / len(result)) * 100} %")
        valid_result.append((result['self'].isna().sum() / len(result)) * 100)
    make_line_plot(train_result, valid_result, depths)


def make_line_plot(t_r, v_r, ds):
    plt.plot(ds, t_r, label="Zbiór trenujący")
    plt.plot(ds, v_r, label="Zbiór walidacyjny")
    plt.xlabel("Parametr")
    plt.ylabel("Jakość")
    plt.legend()
    plt.title("")
    plt.savefig("proba1.png")


def rate_model(dataframe, max_depth):
    rates = []
    sol = Solver()
    for i in range(10):
        x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_dataframe(dataframe)
        root = sol.fit(x_train, y_train, max_depth)
        result_of_tree = sol.predict(x_test, root)
        cleaned = result_of_tree.dropna(axis=1, how='any').squeeze()
        result = cleaned.compare(y_test, keep_shape=True)
        rates.append((result['self'].isna().sum() / len(result)) * 100)
    print(f"maksymalna/minimalna ocena - {max(rates)} / {min(rates)}")
    print(f"średnia ocena - {sum(rates)/len(rates)}")


def test_model(dataframe, maxdepth):
    x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_dataframe(dataframe)
    sol = Solver()
    root = sol.fit(x_train, y_train, maxdepth)
    result_of_tree = sol.predict(x_test, root)
    cleaned = result_of_tree.dropna(axis=1, how='any').squeeze()
    result = cleaned.compare(y_test, keep_shape=True)
    print(f"Acuuracy for depth {maxdepth} is: {(result['self'].isna().sum() / len(result)) * 100}")


def main():
    dataframe = make_dataframe("cardio_train.csv", ";")
    find_best_depth(dataframe)
    # rate_model(dataframe, 6)
    # test_model(dataframe, 2)


if __name__ == "__main__":
    main()
