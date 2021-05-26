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
import scipy
from sklearn.model_selection import train_test_split
from text_generator_model import TextModel, ConvolutionalTextModel
import shutil
from my_utils import print_progress_bar
import os


SEQ_LEN = 1000
ASCII_CHAR_AMOUNT = 128


def custom_loss(output, target):
    n_batches, seq_len, n_classes = output.size()
    losses = torch.zeros([output.size()[:2]]).to(device)
    for i_batch in range(n_batches):
        losses[i_batch] = F.cross_entropy(
            output[i_batch, :, :], target[i_batch, :])


def train_model(model, data, optimizer, criterion, n_epochs, validate=False, model_path=None):
    model.to(device)

    train_loss = []
    val_loss = []

    for i_epoch in range(n_epochs):
        try:
            t = time.time()
            cur_t_loss = 0
            cur_v_loss = 0

            len_ = len(data["train"])

            for i, (inputs, targets) in enumerate(data["train"]):
                print_progress_bar(i, len_)
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)

                loss = criterion(outputs.transpose(1, 2), targets)
                loss.backward()

                optimizer.step()

                cur_t_loss += loss.item()
                

            train_loss.append(cur_t_loss / len(data["train"]))

            if validate:
                with torch.no_grad():
                    for inputs, targets in data["valid"]:
                        inputs, targets = inputs.to(device), targets.to(device)

                        outputs = model(inputs)

                        cur_v_loss += criterion(outputs.transpose(1,
                                                                  2), targets).item()

                val_loss.append(cur_v_loss / len(data["valid"]))


            print(
                f"Epoch {i_epoch}:\ntrain loss: {train_loss[-1]}", end="")
            if validate:
                print(f", val loss: {val_loss[-1]}", end="")
            print(f"\nTook {time.time() - t} seconds")
            print("-------------")

            if model_path is not None:
                torch.save(model.state_dict(), model_path)
        except KeyboardInterrupt:
            break

    return model, train_loss, val_loss


def generate_text(model, start, amount):
    cpu = torch.device("cpu")
    model.to(cpu)
    text = [start]
    with torch.no_grad():
        x = to_one_hot_str(start).reshape([1, len(start), ASCII_CHAR_AMOUNT])
        x = torch.tensor(x, dtype=torch.float32)
        hss = None
        for i in range(amount):
            x, hss = model(x, hss=hss, return_hs=True)
            x_np = x[0, -1, :].numpy()
            chosen = np.random.choice(
                range(128), p=scipy.special.softmax(x_np))
            x[0, -1, :] = 0
            x[0, -1, chosen] = 1
            text.append(from_one_hot(x.cpu().numpy().reshape(
                (1, ASCII_CHAR_AMOUNT))))
    return "".join(text)



def to_one_hot_str(s: str):
    result = np.zeros((len(s), ASCII_CHAR_AMOUNT))
    for i, c in enumerate(s):
        result[i, ord(c)] = 1
    return result


def to_one_hot_ints(ints):
    result = np.zeros((len(ints), ASCII_CHAR_AMOUNT))
    for idx, int_ in enumerate(ints):
        result[idx, int_] = 1
    return result


def from_one_hot(encoded):
    chars = []
    for i in range(encoded.shape[0]):
        #print(encoded[i, :].sum())
        chars.append(chr(np.argmax(encoded[i, :])))
    return ''.join(chars)

# Shape will be [seq_amount, seq_len, 256]


class TextDataset(torch.utils.data.Dataset):

    def __init__(self, inputs, targets, lazy=False):
        # [n_seq, seq_len]
        self._lazy = lazy
        if lazy:
            self.inputs = torch.from_numpy(inputs).type(torch.LongTensor)
        else:
            n_seq, seq_len = inputs.shape
            self.inputs = torch.zeros([n_seq, seq_len, ASCII_CHAR_AMOUNT])
            for i in range(n_seq):
                self.inputs[i, :, :] = 0
                for j in range(seq_len):
                    self.inputs[i, j, int(inputs[i, j])] = 1

        self.targets = torch.from_numpy(targets).type(
            torch.LongTensor)  # [n_examples, n_samples]

    def __getitem__(self, index):
        if self._lazy:
            _, seq_len = self.inputs.shape
            one_hotted = torch.zeros([seq_len, ASCII_CHAR_AMOUNT])
            for i in range(seq_len):
                one_hotted[i, self.inputs[index, i]] = 1
            return one_hotted, self.targets[index, :]
        else:
            return self.inputs[index, :, :], self.targets[index, :]

    def __len__(self):
        return self.inputs.size()[0]


def text_data(file_path, n_sequences, seq_len):
    with open(file_path, "r") as f:
        contents = f.read()
    contents = " ".join(contents.split())

    sequences = np.zeros((n_sequences, seq_len))
    for i in range(n_sequences):
        idx = np.random.randint(len(contents) - seq_len)
        for j, c in enumerate(contents[idx:idx+seq_len]):
            sequences[i, j] = ord(c)

    inputs = sequences[:, :-1]
    targets = sequences[:, 1:]
    return inputs,  targets


if __name__ == "__main__":

    validate = "--validate" in sys.argv

    resume_train = "--resume-train" in sys.argv

    global device
    device = torch.device("cuda:0" if torch.cuda.is_available()
                          and "--cpu" not in sys.argv else "cpu")

    n_in = ASCII_CHAR_AMOUNT
    n_out = ASCII_CHAR_AMOUNT

    model_path = "models/model1"

    if "--train" in sys.argv:

        if os.path.exists(model_path):
            shutil.copyfile(model_path, model_path + '~')

        train_file = "data/shades.txt"

        train_inputs, train_targets = text_data(train_file, 4000, 1100)
        val_inputs, val_targets = text_data(train_file, 10, 100)

        data = {
            "train": torch.utils.data.DataLoader(TextDataset(train_inputs, train_targets), batch_size=3, num_workers=8),
            "valid": torch.utils.data.DataLoader(TextDataset(val_inputs, val_targets), batch_size=1)
        }

        model = ConvolutionalTextModel(n_in, n_out)
        if resume_train:
            model.load_state_dict(torch.load(model_path))
        optimizer = optim.Adadelta(model.parameters())
        criterion = nn.CrossEntropyLoss()
        n_epochs = 100000

        model, train_loss, val_loss = train_model(
            model, data, optimizer, criterion, n_epochs, validate=validate)
        torch.save(model.state_dict(), model_path)

        plt.title("Losses")
        plt.plot(range(len(train_loss)), train_loss, label="train loss")
        if validate:
            plt.plot(range(len(val_loss)), val_loss, label="val loss")
        plt.xlabel("nth epoch")
        plt.ylabel("loss")
        plt.savefig("out/losses.png")

    else:
        model = TextModel(n_in, n_out)
        model.load_state_dict(torch.load(model_path))

    print("Let's generate some text:")
    txt = generate_text(
        model, " ", 1000)
    print(f"'{txt}'")
