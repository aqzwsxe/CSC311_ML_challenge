import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import re
import matplotlib.pyplot as plt


best_val_accuracy = -1  # Initialize best validation accuracy
best_model = None  # Initialize best model

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
    """Return the area at a certain rank in list `l`.

    Areas are indexed starting at 1 as ordered in the survey.

    If area is not present in `l`, return -1.
    """
    return l.index(i) + 1 if i in l else -1

def cat_in_s(s, cat):
    """Return if a category is present in string `s` as an binary integer.
    """
    return int(cat in s) if not pd.isna(s) else 0

def validate_model(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Deactivate autograd for evaluation to reduce memory usage and speed up computations
       for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            if labels.ndim > 1:  # Check if labels are one-hot encoded
                labels = torch.max(labels, 1)[1]  # Convert one-hot labels to class indices

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Validation Accuracy: {accuracy}')
    return accuracy

class MyNeuralNetwork(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super(MyNeuralNetwork, self).__init__()
        layers = []
        in_features = input_size
        for size in layer_sizes:
            layers.append(nn.Linear(in_features, size))
            layers.append(nn.ReLU())
            in_features = size
        layers.append(nn.Linear(in_features, 4))  # Assuming binary classification or adjust for multiclass
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

df = pd.read_csv("clean_dataset.csv")


bag_of_words = set()
frequency = dict()
for sentence in df["Q10"]:
    sentence = str(sentence).lower()
    sentence = sentence.replace("(", "").replace(")", "").replace("\"", "").replace('"', "").replace("?", "").replace("!", "").replace(".", "").replace(",", "").replace(":", "").replace(";", "").replace("-", "").replace("_", "").replace("=", "").replace("+", "").replace("*", "").replace("/", "").replace("\\", "").replace("|", "").replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("<", "").replace(">", "")
    words = sentence.split()
    for word in words:
        if word not in frequency:
            frequency[word]=1
        else:
            frequency[word]+=1

for key in frequency:
    if frequency[key]>=5:
        bag_of_words.add(key)

print(len(bag_of_words))
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

del df["Q6"]

for cat in ["Partner", "Friends", "Siblings", "Co-worker"]:
    cat_name = f"Q5{cat}"
    df[cat_name] = df["Q5"].apply(lambda s: cat_in_s(s, cat))

del df["Q5"]


# Convert features and labels to PyTorch tensors
X = torch.tensor(df.drop("Label", axis=1).values).float()
y = torch.tensor(pd.get_dummies(df["Label"].values).values).float()

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.66, random_state=42)

# Normalize the data
scaler = StandardScaler()
x_train = torch.tensor(scaler.fit_transform(x_train), dtype=torch.float)
x_test = torch.tensor(scaler.transform(x_test), dtype=torch.float)
x_val = torch.tensor(scaler.transform(x_val), dtype=torch.float)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
val_dataset = TensorDataset(x_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

def train_model(model, train_loader, lr=0.001, epochs=10):
    criterion = nn.BCEWithLogitsLoss()  # Adjust for your specific case
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

print(torch.cuda.is_available())
best_hidden=None
val_accuracies = []
hidden_layer_sizes = []
# Example hyperparameter grid (simplified)

hidden_layers = [300, 300, 70]
model = MyNeuralNetwork(input_size=X.shape[1], layer_sizes=hidden_layers).to('cuda')
train_model(model, train_loader, lr=0.001, epochs=20)
# Evaluate your model on validation set and keep track of the best one
val_accuracy = validate_model(model, val_loader, 'cuda')
val_accuracies.append(val_accuracy)
hidden_layer_sizes.append(hidden_layers)
if val_accuracy > best_val_accuracy:
    best_val_accuracy = val_accuracy
    best_model = model  # Store the best model
    best_hidden =hidden_layers

# Now, best_model contains the model with the highest validation accuracy
# Calculate test accuracy with the best model

best_model.eval()  # Set the model to evaluation mode
test_accuracy = 0.0
total_samples = 0

# Disable gradient computation for evaluation since it's not needed
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to('cuda'), target.to('cuda')
        # Make predictions
        outputs = best_model(data)
        # Get the prediction class assuming outputs are raw logits
        _, predicted = torch.max(outputs.data, 1)
        if target.ndim > 1:  # Check if labels are one-hot encoded
                target = torch.max(target, 1)[1]  # Convert one-hot labels to class indices
                print(target)
        total_samples += target.size(0)
        # Increment correct predictions
        test_accuracy += (predicted == target).sum().item()

# Compute the accuracy
test_accuracy = test_accuracy / total_samples
print(f'Test Accuracy: {test_accuracy}')

plt.figure(figsize=(10, 5))
plt.plot(range(len(val_accuracies)), val_accuracies, marker='o')
plt.title('Validation Accuracies for Different Hidden Layer Sizes')
plt.xlabel('Hyperparameter Setting Index')
plt.ylabel('Validation Accuracy')
plt.show()
# # extract the weights of the model
# model_state_dict = best_model.state_dict()

# # You can then access weights and biases for each specific layer
# for layer_name, weights in model_state_dict.items():
#     print("-----------------Start------------------")
#     print(f"Layer: {layer_name}")
#     print(f"Size: {weights.size()}")
#     print(weights)
    
#     # If you need the numpy array
#     weights_numpy = weights.cpu().numpy()
#     print("Weights as numpy array:")
#     print(weights_numpy)
#     print("-----------------End------------------")
for i in range(len(val_accuracies)):
    print(f"{i}th hidden layers {hidden_layer_sizes[i]}")
print(best_val_accuracy)
print(best_hidden)
# Save the best model
torch.save(best_model.state_dict(), 'best_model.pth')



