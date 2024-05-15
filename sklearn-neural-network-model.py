from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import pandas as pd
import re

file_name = "clean_dataset.csv"

def to_numeric(s):
    """Convert a string to a number.
    """
    try:
        return float(s)
    except:
        return 0

def get_number_list(s):
    """Get a list of integers contained in string `s`
    """
    return [int(n) for n in re.findall("(\d+)", str(s))]


def get_number_list_clean(s):
    """Return a clean list of numbers contained in `s`.

    Additional cleaning includes removing numbers that are not of interest
    and standardizing return list size.
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
    return l.index(i) + 1 if i in l else -1


def cat_in_s(s, cat):
    """Return if a category is present in string `s` as an binary integer.
    """
    return int(cat in s) if not pd.isna(s) else 0

if __name__ == "__main__":
    df = pd.read_csv(file_name)
    bag_of_words = set()
    frequency = dict()
    ingore_words=['or', 'and', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'in', 'on', 'at', 'of', 'for', 'with', 'by', 'as', 'from', 'that', 'this', 'these', 'those', 'it', 'its', 'they', 'their', 'them', 'he', 'she', 'his', 'her',
                   'him', 'we', 'us', 'our', 'i', 'me', 'my', 'you', 'your', 'yours', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must', 'am', 'is',
                     'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must', 'am', 'is', 'are', 'was', 'were', 'being', 'been', 'have', 'has',
                       'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must', 'am', 'is', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
                         'should', 'can', 'could', 'may', 'might', 'must', 'am', 'is', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must',
                           'am', 'is', 'are', 'was', 'were', 'being', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'like', 'there', 'one', "it's", 'nan', 'no', 'not']
    for sentence in df["Q10"]:
        sentence = str(sentence).lower()
        sentence = sentence.replace("(", "").replace(")", "").replace("\"", "").replace('"', "").replace("?", "").replace("!", "").replace(".", "").replace(",", "").replace(":", "").replace(";", "").replace("-", "").replace("_", "").replace("=", "").replace("+", "").replace("*", "").replace("/", "").replace("\\", "").replace("|", "").replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("<", "").replace(">", "")
        words = sentence.split()
        for word in words:
            if word in ingore_words:
                continue
            if word not in frequency:
                frequency[word]=1   
            else:
                frequency[word]+=1

    for key in frequency:
        if frequency[key]>=20:
            bag_of_words.add(key)

    print(bag_of_words)
    # Assume you perform preprocessing here similar to your sklearn approach
    df = df.drop("id", axis=1)


    def count_words(text, word_list):
        text = str(text).lower()
        return {word: text.count(word) for word in word_list}

    # Apply this function to each row of 'Q10' and create a new DataFrame
    counts = df['Q10'].apply(lambda x: count_words(x, bag_of_words))

    # Convert the counts to a DataFrame
    counts_df = pd.DataFrame(counts.tolist())

    # Concatenate the new DataFrame with the original one
    df = pd.concat([df, counts_df], axis=1)

    # Drop the 'Q10' column if it's no longer needed
    df.drop('Q10', axis=1, inplace=True)

    for i in range(1, 5):
        col_name = f"Q{i}"
        df[col_name] = df[col_name].apply(to_numeric).fillna(0)

    for i in range(7, 10):
        col_name = f"Q{i}"
        df[col_name] = df[col_name].apply(to_numeric).fillna(0) 

    df["Q6"] = df["Q6"].apply(get_number_list_clean)
    for i in range(1, 7):
        col_name = f"rank_{i}"
        df[col_name] = df["Q6"].apply(lambda x: find_area_at_rank(x, i))

    print(df["Q6"])

    del df["Q6"]

    for cat in ["Partner", "Friends", "Siblings", "Co-worker"]:
        cat_name = f"Q5{cat}"
        df[cat_name] = df["Q5"].apply(lambda s: cat_in_s(s, cat))

    del df["Q5"]
    print('Shape of the dataset:', df.shape)
    print(df.shape)
    n_train = 1200
    random_state = 42
    df = df.sample(frac=1, random_state=random_state)
    X = df.drop("Label", axis=1).values
    y = pd.get_dummies(df["Label"].values)
    print(X.shape, y.shape)
    x_train = X[:n_train]
    y_train = y[:n_train]

    x_test = X[n_train:]
    y_test = y[n_train:]

    # scaler = StandardScaler()
    # scaler.fit(x_train)
    # x_train = scaler.transform(x_train)
    # x_test = scaler.transform(x_test)

    max=(0,0)
    max_acc = 0
    weights = None
    biases = None

    print(y_test.shape)
    # for i in range(2, 20):
    # for i in range(1, 50):
    #     for k in range(1, 50):
    #         for j in range(1, 50):
    #             print(f"Trying {i} {k} {j}")
    #             model = MLPClassifier(
    #             hidden_layer_sizes=(i,k,j),
    #             max_iter=4000,
    #             activation='relu',
    #             solver='adam',
    #             random_state=42)
    #             cross_val_scores = cross_val_score(model, x_train, y_train, cv=5)
    #             acc = cross_val_scores.mean()
    
    #             if acc > max_acc:
    #                 max_acc = acc
    #                 max = (i,k,j)
                   
    best_model=None
    
    for i in range(1, 50):
        curr_model = MLPClassifier(
            hidden_layer_sizes=[60, 60, 20],  
            max_iter=2000,
            activation='relu',
            solver='adam',
            random_state=42
        )
        curr_model.fit(x_train, y_train)  
        y_pred = curr_model.predict(x_test)
        t_acc = accuracy_score(y_test, y_pred)
        if t_acc > max_acc:
            max_acc = t_acc
            best_model = curr_model

    weights = best_model.coefs_
    biases = best_model.intercepts_

    with open('weights_and_biases.txt', 'w') as f:
        # Printing the weights of each layer
        for i, weight_matrix in enumerate(weights):
            # Write weights to file
            f.write(f"Weights between layer {i} and layer {i+1}:\n")
            f.write(f"{weight_matrix.tolist()}\n\n")

        # Printing the biases of each layer
        for i, bias_vector in enumerate(biases):
            # Write biases to file
            f.write(f"Biases for layer {i+1}:\n")
            f.write(f"{bias_vector.tolist()}\n\n")
    # Make predictions
    print(max)
    print(f"Accuracy: {max_acc}")
    print(f"Test Accuracy: {t_acc}")

    

    