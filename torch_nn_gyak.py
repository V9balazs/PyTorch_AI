import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x


torch.manual_seed(41)
model = Model()

url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
my_df = pd.read_csv(url)

my_df["species"] = my_df["species"].replace("setosa", 0.0)
my_df["species"] = my_df["species"].replace("virginica", 2.0)
my_df["species"] = my_df["species"].replace("versicolor", 1.0)

X = my_df.drop("species", axis=1)
Y = my_df["species"]

X = X.values
Y = Y.values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
losses = []

for i in range(epochs):
    Y_pred = model.forward(X_train)
    loss = criterion(Y_pred, Y_train)
    losses.append(loss.detach().numpy())
    if i % 10 == 0:
        print(f"Epoch: {i} and loss: {loss}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    Y_eval = model.forward(X_test)
    loss = criterion(Y_eval, Y_test)

correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        Y_val = model.forward(data)
        print(f"{i+1}.) {str(Y_val)}")
        if Y_val.argmax().item() == Y_test[i]:
            correct +=  

print(f"We got {correct} out of {len(X_test)}")

new_iris = torch.tensor([4.7, 3.2, 1.3, 0.2])
with torch.no_grad():
    print(model(new_iris))

newer_iris = torch.tensor([5.1, 3.5, 1.4, 1.8])
with torch.no_grad():
    print(model(newer_iris))

torch.save(model.state_dict(), "iris_model.pt")

new_model = Model()
