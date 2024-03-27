"""
This Python file provides some useful code for reading the training file
"clean_dataset.csv". You may adapt this code as you see fit. However,
keep in mind that the code provided does only basic feature transformations
to build a rudimentary kNN model in sklearn. Not all features are considered
in this code, and you should consider those features! Use this code
where appropriate, but don't stop here!
"""

import re
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import  numpy as np
file_name = "clean_dataset.csv"
random_state = 42

def to_numeric(s):
    """Converts string `s` to a float.

    Invalid strings and NaN values will be converted to float('nan').
    """

    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
    return float(s)

def get_number_list(s):
    """Get a list of integers contained in string `s`
        \d: match any digit from 0 to 9
        +: \d should match one or more times
        findall: returns a list of all non-overlapping matches of the pattern in
        the input string; Each match is represented as a string
        and return as a list
    """
    return [int(n) for n in re.findall("(\d+)", str(s))]

def get_number_list_clean(s):
    """Return a clean list of numbers contained in `s`.

    Additional cleaning includes removing numbers that are not of interest
    and standardizing return list size.

    Appends -1 to the n_list multiple times to standardize its size to 6 element

    If the original list n_list has fewer tha 6 elements,
    it adds enough -1 values to make its length 6
    """

    n_list = get_number_list(s)
    n_list += [-1]*(6-len(n_list))
    return n_list

def get_number(s):
    """Get the first number contained in string `s`.

    If `s` does not contain any numbers, return -1.
    """
    n_list = get_number_list(s)
    return n_list[0] if len(n_list) >= 1 else -1

def find_area_at_rank(l, i):
    """Return the area at a certain rank in list `l`.

    Areas are indexed starting at 1 as ordered in the survey.

    If area is not present in `l`, return -1.

    It returns the position of the element i in the list l, indicating its rank
    the area starting at 1 not 0
    """
    return l.index(i) + 1 if i in l else -1

def cat_in_s(s, cat):
    """Return if a category is present in string `s` as an binary integer.

    check if the category cat is present in the string s
    If cat is present in s, it becomes 1
    """
    return int(cat in s) if not pd.isna(s) else 0

if __name__ == "__main__":
    #  dataFrame: a two-dimensional labeled data structure in Pandas
    # key features of a DataFrame:
    # Indexing: Each row in a Df is associated with an index, which can be either integer based or labeled;
    # The index provides a way to uniquely identify each row

    # Column: in a DataFrame represent variable or attribute


    df = pd.read_csv(file_name)

    # Clean numerics

    # Access a column names Q6 from the DateFrame df
    df["Q7"] = df["Q7"].apply(to_numeric).fillna(0)
    #  apply in pandas: apply a functions along an axis of dataFrame or Series
    # to_numeric: is a function that converts its input that convert its input to a numeric type(float)
    # .fillna(0): in pandas it is used to fill missing(NAN) values in a DataFrame or Series with
    # a specific value. In this case, 0
    # Clean for number categories

    df["Q1"] = df["Q1"].apply(get_number)

    # Create area rank categories

    df["Q6"] = df["Q6"].apply(get_number_list_clean)

    # It is created to store the names of new columns
    # that will be added to the DataFrame df.

    temp_names = []
    for i in range(1,7):
        # The new column name
        col_name = f"rank_{i}"
        # embedded the i to the string rank_{}
        temp_names.append(col_name)
        # l represent a single element of Q6
        # The element of Q6:
        # Skyscrapers=>6,Sport=>4,Art and Music=>2,Carnival=>1,Cuisine=>3,Economic=>5
        df[col_name] = df["Q6"].apply(lambda l: find_area_at_rank(l, i))
        # lambda l: find_area_at_rank(l, i); A lambda function
    # remove the column named Q6 from DataFrame df
    del df["Q6"]

    # Create category indicators

    new_names = []
    # Create a new list contains ["Q1", "rank1", "rank2" ...]

    # After each iteration, the col inside the ["Q1"] + temp_names
    # will be deleted and some indicators which is specified by the col will be added to the df
    for col in ["Q1"] + temp_names:
        # create dummy variable; dummy variable=indicator variable
        #  prefix parameter is used to add a prefix to the column name of the dummy variables created
        indicators = pd.get_dummies(df[col], prefix=col)
        # Each row of the indicator is a one-hot vector
        # add the column name of the indicators DataFrame
        new_names.extend(indicators.columns)
        # Concatenates the DataFrame indicators to the existing
        # DataFrame df along the column axis, axis=1;
        df = pd.concat([df, indicators], axis=1)
        del df[col]

    # Create multi-category indicators

    for cat in ["Partner", "Friends", "Siblings", "Co-worker"]:
      cat_name = f"Q5{cat}"
      new_names.append(cat_name)
      df[cat_name] = df["Q5"].apply(lambda s: cat_in_s(s, cat))

    del df["Q5"]

    # Prepare data for training - use a simple train/test split for now

    # Reorders the columns of the DataFrame df
    # It selects the column specified in new_names(contain the names of the one-hot encoded columns created earlier)
    # The one-hot encoded column come first
    # followed  by Q7 and Label
    df = df[new_names + ["Q7", "Label"]]
    # shuffles the rows (specified by frac=1) of the DataFrame df using the
    # sample function
    # Setting the random_state to a specific value ensure that the random shuffling of rows is
    # reproducible
    df = df.sample(frac=1, random_state=random_state)


    # 1. drop() Make a copy of the df and delete the values of df col='Label', then return the new DF
    # 2. Assign the resulting DF without the Label column to the variable x
    # 3. Convert the DF into a NP array using the .values attribute

    # x: contain the input features for the ml model. It is obtained
    # by dropping the column labelled "Label"
    x = df.drop("Label", axis=1).values
    for a in x:
        print(x)
        break

    # contains the target labels for the ml model
    # It is obtained by applying one hot encoding to the
    # values in the Label column
    # Each unique label column of the DataFrame df using
    y = pd.get_dummies(df["Label"].values)

    # Specify the number of samples to be used for training the model
    n_train = 1200
    # Extracts the first n_train samples from the feature matrix x and
    # assigns them to x_train. Those
    x_train = x[:n_train]

    # select the first 1200 target
    y_train = y[:n_train]
    print(np.shape(y_train))
    # The rest of data and label
    x_test = x[n_train:]
    y_test = y[n_train:]
    print(np.shape(y_test))
    # Train and evaluate classifiers
    # Create KNN classifier with 3 neighbors
    clf = KNeighborsClassifier(n_neighbors=3)

    # train the classifier on the training data
    clf.fit(x_train, y_train)
    # Calculate the accuracy of the classifier on the
    # training data. The score method returns the mean accuracy on the
    # given data and labels
    train_acc = clf.score(x_train, y_train)
    test_acc = clf.score(x_test, y_test)
    print(f"{type(clf).__name__} train acc: {train_acc}")
    print(f"{type(clf).__name__} test acc: {test_acc}")

    # create DecisionTree
    clf1 = DecisionTreeClassifier(criterion="entropy", max_depth=100)
    clf1.fit(x_train, y_train)

    train_acc1 = clf1.score(x_train,y_train)
    test_acc1 = clf1.score(x_test, y_test)
    print(f"{type(clf1).__name__} train acc: {train_acc1}")
    print(f"{type(clf1).__name__} test acc: {test_acc1}")

    # create LinearRegression
    clf2 = LinearRegression(fit_intercept=True)
    # x_train: a 2D array like object containing the training input feature
    # y_train: a 1D array like object containing the target label
    clf2.fit(x_train, y_train)
    train_acc = clf2.score(x_train, y_train)
    test_acc = clf2.score(x_test, y_test)

    print(f"{type(clf2).__name__} train acc: {train_acc}")
    print(f"{type(clf2).__name__} test acc: {test_acc}")



