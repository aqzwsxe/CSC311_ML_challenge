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
from sklearn.model_selection import train_test_split
import  numpy as np
file_name = "clean_dataset.csv"
random_state = 42


# Q1: popularity   Q2 efficient  Q3 architectural uniqueness
# Q4: enthusiasm for spontaneous street parties
# Q5:If you were to travel to this city, who would be likely with you?
# Q6: Rank the following words from the least to most relatable to this city. Each area should have a different number assigned to it. (1 is the least relatable and 6 is the most relatable)
# Q7: In your opinion, what is the average temperature of this city over the month of January? (Specify your answer in Celsius)
# Q8: How many different languages might you overhear during a stroll through the city?
# Q9: How many different fashion styles might you spot within a 10-minute walk in the city?
# Q10: What quote comes to mind when you think of this city?
def to_numeric(s):
    """Converts string `s` to a float.

    Invalid strings and NaN values will be converted to float('nan').
    """

    if isinstance(s, str):
        # print("The s")
        # print(s)
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
    # print(n_list)
    return n_list[0] if len(n_list) >= 1 else -1

def find_area_at_rank(l, i):
    """Return the area at a certain rank in list `l`.

    Areas are indexed starting at 1 as ordered in the survey.

    If area is not present in `l`, return -1.

    It returns the position of the element i in the list l, indicating its rank
    the area starting at 1 not 0

    The rank of each element is index+1
    """
    a = l.index(i) + 1 if i in l else -1
    return a

def cat_in_s(s, cat):
    """Return if a category is present in string `s` as an binary integer.

    check if the category cat is present in the string s
    If cat is present in s, it becomes 1
    """
    return int(cat in s) if not pd.isna(s) else 0

#
# def tune_tree_hyper():
#     criterions = ["entropy", "gini"]
#     max_depths = [1, 5, 10, 15, 20, 25, 30, 50, 100]
#     min_sample_split = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
#     max_acc = -1
#     for c in criterions:
#         print(f"use the criterion: {c}")
#         for dep in max_depths:
#             for sp in min_sample_split:
#                 tree = DecisionTreeClassifier(criterion=c,max_depth=dep, min_samples_split=sp, x_train=X_train, )
#

def build_knn(X_train,t_train,X_valid,t_valid):
    out = {}
    for i in range(1, 151):
        print(i)
        out[i]={}
        k_model = KNeighborsClassifier(n_neighbors=i)
        k_model.fit(X_train, t_train)

        out[i]['train'] = k_model.score(X_valid, t_valid)
        out[i]['validation'] = k_model.score(X_train, t_train)
    return out

def build_all_models(max_depths,min_samples_split,criterion,X_train,t_train,X_valid,t_valid):
        """
        Parameters:
            `max_depths` - A list of values representing the max_depth values to be
                           try as hyperparameter values
            `min_samples_split` - An list of values representing the min_samples_split
                           values to try as hyperpareameter values
            `criterion` -  A string; either "entropy" or "gini"

        Returns a dictionary, `out`, whose keys are the the hyperparameter choices, and whose values are
        the training and validation accuracies (via the `score()` method).
        In other words, out[(max_depth, min_samples_split)]['val'] = validation score and
                        out[(max_depth, min_samples_split)]['train'] = training score
        For that combination of (max_depth, min_samples_split) hyperparameters.
        """
        out = {}

        for d in max_depths:
            for s in min_samples_split:
                out[(d, s)] = {}
                # Create a DecisionTreeClassifier based on the given hyperparameters and fit it to the data
                tree = DecisionTreeClassifier(max_depth=d, min_samples_split=s, criterion=criterion)
                # If there's no spcifies parameter, let the parameter=itself
                tree.fit(X_train, t_train)
                # TODO: store the validation and training scores in the `out` dictionary
                out[(d, s)]['val'] = tree.score(X_valid, t_valid)  # TODO
                out[(d, s)]['train'] = tree.score(X_train, t_train)  # TODO

        return out





if __name__ == "__main__":
    #  dataFrame: a two-dimensional labeled data structure in Pandas
    # key features of a DataFrame:
    # Indexing: Each row in a Df is associated with an index, which can be either integer based or labeled;
    # The index provides a way to uniquely identify each row

    # Column: in a DataFrame represent variable or attribute


    df = pd.read_csv(file_name)

    # print(type(df))
    # for row in df:
    #     print(row)
    #     print(type(row))
    # Clean numerics
    # print("the data of Q1")
    # print(type(["Q1"]))
    # print(type(["Q2"]))
    # print(type(["Q3"]))
    # print(type(["Q4"]))
    # print(type(["label"]))

    # for i in df["Q1"]:
    #     print(type(i))

    # print(df["Q7"])
    #  Q7: average temperature
    # Access a column names Q6 from the DateFrame df
    # for i in df["Q7"]:
    #     print(type(i))
    # String
    df["Q7"] = df["Q7"].apply(to_numeric).fillna(0)
    # Float
    #  apply in pandas: apply a functions along an axis of dataFrame or Series
    # to_numeric: is a function that converts its input that convert its input to a numeric type(float)
    # .fillna(0): in pandas it is used to fill missing(NAN) values in a DataFrame or Series with
    # a specific value. In this case, 0
    # Clean for number categories
    # print(df["Q1"])
    # before apply the get_number
    # 0 25
    # 1 20
    # 2 32
    # for i in df["Q1"]:
    #     print(i)
    # 4.0
    # 5.0
    # 4.0
    # 4.0
    # 4.0

    df["Q1"] = df["Q1"].apply(get_number)
    # for i in df["Q1"]:
    #     print(type(i))
    # 4
    # 4
    # 5
    # 5
    # 4
    # for i in df["Q1"]:
    #     print(i)
    # after apply the get_number
    # print("Print Q2")
    # print(type(df["Q2"]))
    # for i in df["Q1"]:
    #     print(type(i))
    # Create area rank categories
    # for i in df["Q6"]:
    #     print(i)
    df["Q6"] = df["Q6"].apply(get_number_list_clean)
    # df["Q6"] after all the get_number_list
    # [6, 4, 2, 1, 3, 5]
    # [6, 1, 2, 3, 4, 5]
    # [6, 2, 3, 3, 4, 5]

    # df["Q6"] after all the get_number_list_clean
    #[1, 2, 6, 5, 4, 3]
    # [1, 2, 6, 5, 4, 3]
    # [-1, -1, -1, -1, -1, -1]
    # [4, 6, 5, 4, 4, 4]

    # It is created to store the names of new columns
    # that will be added to the DataFrame df.
    # print(df["Q6"])
    # for i in df["Q6"]:
    #     print(i)
    temp_names = []
    for i in range(1,7):
        # The new column name
        col_name = f"rank_{i}"
        # embedded the i to the string rank_{}
        temp_names.append(col_name)
        # l represent a single element of Q6
        # The element of Q6:
        # [1, 2, 6, 5, 4, 3]
        # [1, 2, 6, 5, 4, 3]
        # [-1, -1, -1, -1, -1, -1]
        #  now df["Q6"] contains different list
        #  Apply the findArea function to each element of the df["Q6"]; Each element is a list
        df[col_name] = df["Q6"].apply(lambda l: find_area_at_rank(l, i))
        # print(type(df["Q6"].apply(lambda l: find_area_at_rank(l, i))))
        # lambda l: find_area_at_rank(l, i); A lambda function
    #  1: Iteration over each list in the "Q6" column: The loop iterates over each list (list1) contained
    #  within the "Q6" column of the DataFrame.

    # 2: Iteration over the range of numbers from 1 to 7 (exclusive): For each list1, the loop iterates
    # over numbers from 1 to 6 (inclusive).

    # 3: Finding the position of each number in the current list (list1): For each number A in the range
    # from 1 to 6, it finds the position (index) of A within the current list1.

    # 4: Collecting the positions for each number in each list: The positions of each number A are collected
    # across all rows to build a series for each rank N. If a rank is not present in the current list1, -1 is stored in that position.

    # rank_1: the position of rank1
    # rank_2: the position of rank2

    # remove the column named Q6 from DataFrame df
    del df["Q6"]

    # Create category indicators

    new_names = []
    # Create a new list contains ["Q1", "rank1", "rank2" ...]

    # After each iteration, the col inside the ["Q1"] + temp_names
    # will be deleted and some indicators which is specified by the col will be added to the df
    # temp_names: contain rank_N


    for col in ["Q1"] + temp_names:
        # create dummy variable; dummy variable=indicator variable
        #  prefix parameter is used to add a prefix to the column name of the dummy variables created
        # print(col)
        # Each of the indicator is a DataFrame. Each row of the dataFrame is a one hot vector
        indicators = pd.get_dummies(df[col], prefix=col)
        # The column is the Q1_-1   Q1_1   Q1_2   Q1_3   Q1_4   Q1_5
        # rank_1_-1  rank_1_1  rank_1_2  rank_1_3  rank_1_4  rank_1_5  rank_1_6

        # print(type(indicators))
        # Each row of the indicator is a one-hot vector
        # add the column name of the indicators DataFrame
        new_names.extend(indicators.columns)
        # Concatenates the DataFrame indicators to the existing
        # DataFrame df along the column axis, axis=1;
        df = pd.concat([df, indicators], axis=1)
        del df[col]

        # new_names
        # Q1_- 1
        # Q1_1
        # Q1_2
        # Q1_3
        # Q1_4

        # for i in new_names:
        #     print(i)
    # Create multi-category indicators

    # for i in df["Q5"]:
    #     print(i)
    for cat in ["Partner", "Friends", "Siblings", "Co-worker"]:
      cat_name = f"Q5{cat}"
      new_names.append(cat_name)
      df[cat_name] = df["Q5"].apply(lambda s: cat_in_s(s, cat))
    # Iterate through the list of categories: "Partner", "Friends", "Siblings", and "Co-worker".
    # For each category, create a new column in df with the name formatted as "Q5{category}". For example, if the category is "Partner", the new column name will be "Q5Partner".
    # For each row in the DataFrame df, apply a lambda function to the corresponding string in the "Q5" column.
    # The lambda function (lambda s: cat_in_s(s, cat)) checks if the given category string (like "Partner") is present in the input string s. If it is present, it returns 1; otherwise, it returns 0.
    # Populate the new column with the results obtained from applying the lambda function to each string in the "Q5" column.




    # for i in df["Q5"]:
    #     print(i)
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

    # shuffle the df to ensure that the order of the input data does not affect the prediction
    df = df.sample(frac=1, random_state=random_state)


    # 1. drop() Make a copy of the df and delete the values of df col='Label', then return the new DF
    # 2. Assign the resulting DF without the Label column to the variable x
    # 3. Convert the DF into a NP array using the .values attribute

    # x: contain the input features for the ml model. It is obtained
    # by dropping the column labelled "Label"
    x = df.drop("Label", axis=1).values
    # for a in x:
    #     print(x)


    # contains the target labels for the ml model
    # It is obtained by applying one hot encoding to the
    # values in the Label column
    # Each unique label column of the DataFrame df using
    # print(df["Label"].values)
    y = pd.get_dummies(df["Label"].values)
    #  Y is a one hot matrix
    # Specify the number of samples to be used for training the model
    n_train = 1200
    # Extracts the first n_train samples from the feature matrix x and
    # assigns them to x_train. Those
    x_train = x[:n_train]
    # for i in x:
    #     print(i)
    # select the first 1200 target
    y_train = y[:n_train]
    # print(np.shape(y_train))
    # The rest of data and label
    x_test = x[n_train:]
    y_test = y[n_train:]
    # print(np.shape(y_test))
    # Train and evaluate classifiers
    # Create KNN classifier with 3 neighbors
    X_tv, X_test, t_tv, t_test = train_test_split(x, y, test_size=1500 / 8000, random_state=1)
    # random_state:  make sure every time the code run, the split are the same
    X_train, X_valid, t_train, t_valid = train_test_split(X_tv, t_tv, test_size=1500 / 6500, random_state=1)
    clf = KNeighborsClassifier(n_neighbors=3)

    # train the classifier on the training data
    # clf.fit(x_train, y_train)
    # # Calculate the accuracy of the classifier on the
    # # training data. The score method returns the mean accuracy on the
    # # given data and labels
    # train_acc = clf.score(x_train, y_train)
    # test_acc = clf.score(x_test, y_test)
    # print(f"{type(clf).__name__} train acc: {train_acc}")
    # print(f"{type(clf).__name__} test acc: {test_acc}")
    clf.fit(X_train, t_train)
    train_acc = clf.score(X_train, t_train)
    test_acc = clf.score(X_test, t_test)
    print(f"{type(clf).__name__} train acc: {train_acc}")
    print(f"{type(clf).__name__} test acc: {test_acc}")
    # 1111111111111111111111111111111111111
    # create DecisionTree

    print("##########################")
    # Then, use `train_test_split` to split the training+validation data
    # into 5000 train and 1500 validation

    tree = DecisionTreeClassifier(max_depth=10, min_samples_split=32, criterion="entropy")
    tree.fit(X_train, t_train)

    train_acc1 = tree.score(X_train, t_train)
    # val_acc1 = tree.score(X_valid, t_valid)
    test_acc1 = tree.score(X_test, t_test)
    print("criterion=entropy")
    print(f"{type(tree).__name__} train acc: {train_acc1}")
    print(f"{type(tree).__name__} test acc: {test_acc1}")
    print("##########################")
    tree1 = DecisionTreeClassifier(max_depth=100, min_samples_split=2, criterion="gini")
    tree1.fit(X_train, t_train)
    train_acc2 = tree1.score(X_train,t_train)
    test_acc2 = tree1.score(X_test,t_test)
    print("criterion=gini")
    print(f"{type(tree1).__name__} train acc: {train_acc2}")
    print(f"{type(tree1).__name__} test acc: {test_acc2}")
    print("########################")
    # create LinearRegression
    clf2 = LinearRegression(fit_intercept=True)
    # x_train: a 2D array like object containing the training input feature
    # y_train: a 1D array like object containing the target label
    # clf2.fit(x_train, y_train)
    # train_acc = clf2.score(x_train, y_train)
    # test_acc = clf2.score(x_test, y_test)
    #
    # print(f"{type(clf2).__name__} train acc: {train_acc}")
    # print(f"{type(clf2).__name__} test acc: {test_acc}")

    clf2.fit(X_train, t_train)
    train_acc = clf2.score(X_train, t_train)
    test_acc = clf2.score(X_test, t_test)

    print(f"{type(clf2).__name__} train acc: {train_acc}")
    print(f"{type(clf2).__name__} test acc: {test_acc}")

    # Hyperparameters values to try in our grid search
    criterions = ["entropy", "gini"]
    max_depths = [1, 5, 10, 15, 20, 25, 30, 50, 100]
    min_samples_split = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    for criterion in criterions:
        print("\nUsing criterion {}".format(criterion))
        res = build_all_models(max_depths,min_samples_split,criterion,X_train,t_train,X_valid,t_valid)                    # TODO: call `build_all_models` for the given hyperparameters

        max_score = -1
        max_para = None
        for d, s in res:

            temp = res[(d, s)]['val']
            if temp > max_score:
                max_score = res[(d, s)]['val']
                max_para = (d, s)

        print("For criterion {}, the best parameters are {} with accuracy {}".format(criterion, max_para, max_score))

    out = build_knn(X_train, t_train, X_valid, t_valid)
    max_score1 = -1
    max_para1 = 0
    for i in range(1, 151):
        temp = out[i]['validation']
        if temp > max_para1:
            max_score1 = temp
            max_para1 = i
    print("The best parameter, the number of neighbours {}, with accuracy {}".format(max_para1, max_score1))

    k1 = KNeighborsClassifier(n_neighbors=1)
    k1.fit(X_train, t_train)
    train_acc = k1.score(X_train, t_train)
    test_acc = k1.score(X_test, t_test)
    print("when k=1")
    print(f"{type(k1).__name__} train acc: {train_acc}")
    print(f"{type(k1).__name__} test acc: {test_acc}")

    print("########################")

    k1 = KNeighborsClassifier(n_neighbors=5)
    k1.fit(X_train, t_train)
    train_acc = k1.score(X_train, t_train)
    test_acc = k1.score(X_test, t_test)
    print("When k=5")
    print(f"{type(k1).__name__} train acc: {train_acc}")
    print(f"{type(k1).__name__} test acc: {test_acc}")

    print("########################")

    k1 = KNeighborsClassifier(n_neighbors=10)
    k1.fit(X_train, t_train)
    train_acc = k1.score(X_train, t_train)
    test_acc = k1.score(X_test, t_test)
    print("When k=10")
    print(f"{type(k1).__name__} train acc: {train_acc}")
    print(f"{type(k1).__name__} test acc: {test_acc}")

    print("########################")

    k1 = KNeighborsClassifier(n_neighbors=15)
    k1.fit(X_train, t_train)
    train_acc = k1.score(X_train, t_train)
    test_acc = k1.score(X_test, t_test)
    print("When k=15")
    print(f"{type(k1).__name__} train acc: {train_acc}")
    print(f"{type(k1).__name__} test acc: {test_acc}")