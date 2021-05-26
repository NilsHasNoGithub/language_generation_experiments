import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import sys
from sklearn.model_selection import train_test_split



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


def train_model(model, data, optimizer, criterion, n_epochs):
    model.to(device)

    train_loss = []
    val_loss = []

    for i_epoch in range(n_epochs):
        try:
            t = time.time()
            cur_t_loss = 0
            cur_v_loss = 0

            for inputs, targets in data["train"]:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                cur_t_loss += loss.item()

            with torch.no_grad():
                for inputs, targets in data["valid"]:
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = model(inputs)

                    cur_v_loss += criterion(outputs, targets).item()

            train_loss.append(cur_t_loss / len(data["train"]))
            val_loss.append(cur_v_loss / len(data["valid"]))

            print(
                f"Epoch {i_epoch}:\ntrain loss: {train_loss[-1]}, val loss: {val_loss[-1]}")
            print(f"Took {time.time() - t} seconds")
            print("-------------")
        except KeyboardInterrupt:
            break

    return model, train_loss, val_loss

class NNISeqModel(nn.Module):

    def __init__(self):
        super().__init__()

        n_in = 1
        n_out = 1


        self.to_in = nn.Linear(1,2)
        self.lstm = nn.LSTM(2,2, batch_first=True)
        self.to_out = nn.Linear(2, 1)


    def forward(self, x):
        x = self.to_in(x)
        x, _ = self.lstm(x)
        x = self.to_out(x)
        return x

class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()

        n_in = 1
        n_out = 1


        self.to_in = nn.Linear(1,5)
        self.fc1 = nn.Linear(5,5)
        self.to_out = nn.Linear(5, 1)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.to_in(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.to_out(x)
        return x

def sequence_test_data(n_seq, seq_len):
    
    sequences = np.zeros([n_seq, seq_len, 1])
    for i in range(n_seq):
        interval = np.random.randint(5, 50)
        for j in range(seq_len):
            sequences[i, j, 0] = j * interval
    seq = sequences[0,:,:]
    inputs = sequences[:, :-1, :]
    targets = sequences[:, 1:, :]

    return inputs, targets


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, X, T):
        # [n_examples, n_samples, n_features]
        self.X = torch.from_numpy(X).type(torch.FloatTensor)
        self.T = torch.from_numpy(T).type(
            torch.FloatTensor)  # [n_examples, n_samples]

    def __getitem__(self, index):
        return self.X[index, :, :], self.T[index]

    def __len__(self):
        return self.X.size()[0]

    


a = torch.tensor([[1,2],[3,4]])
b = a.reshape([1, -1])

print(a)
print(b)

'''
model_path = "models/model2"

train_inputs, train_targets = sequence_test_data(10000, 1000)
val_inputs, val_targets = sequence_test_data(10, 100)

data = {
    "train": torch.utils.data.DataLoader(SequenceDataset(train_inputs, train_targets), batch_size=100),
    "valid": torch.utils.data.DataLoader(SequenceDataset(val_inputs, val_targets), batch_size=1)
}

model1 = NNISeqModel()
model2 = SimpleModel()
optimizer1 = optim.Adadelta(model1.parameters())
optimizer2 = optim.Adadelta(model2.parameters())
criterion = nn.MSELoss()
n_epochs = 100

model1, train_loss1, val_loss1 = train_model(
    model1, data, optimizer1, criterion, n_epochs)

model2, train_loss2, val_loss2 = train_model(
    model2, data, optimizer2, criterion, n_epochs)


plt.plot(range(len(train_loss1)), train_loss1, label="train loss m1")

plt.plot(range(len(val_loss1)), val_loss1, label="val loss m1")
plt.plot(range(len(train_loss2)), train_loss2, label="train loss m2")
plt.plot(range(len(val_loss2)), val_loss2, label="val loss m2")

plt.title("Losses")
plt.xlabel("nth epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig("out/tests2.png")
'''