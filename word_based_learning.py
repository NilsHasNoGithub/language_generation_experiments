import json
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
from text_generator_model import *
import shutil
from my_utils import print_progress_bar, flatten
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Thread
from collections import deque


LOWER = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
         'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
UPPER = [c.upper() for c in LOWER]
ALPHABET = LOWER + UPPER

CAPITALIZE = "\\cap"


class Vocabulary:

    def __init__(self):
        self.words = []
        self.words_to_index = {}
        super().__init__()

    def from_list(self, word_list):
        self.words = word_list
        self.words_to_index = {}
        self.words_to_index = {word: idx for idx, word in enumerate(word_list)}

    def load(self, file_path):
        with open(file_path, "r") as f:
            contents = f.read()
        words = json.loads(contents)
        self.from_list(words)

    def save(self, file_path):
        js_o = json.dumps(self.words)
        with open(file_path, "w") as f:
            f.write(js_o)

    def add_txt_file(self, file_path):

        sequence = sequence_from_file(file_path)
        for item in sequence:
            if item not in self.words_to_index:
                self.words_to_index[item] = len(self.words)
                self.words.append(item)

    def add_txt_files(self, file_list):
        for f in file_list:
            self.add_txt_file(f)

    def __len__(self):
        return len(self.words_to_index)


def str_from_sequence(sequence):
    result = []
    i = 0
    while i < len(sequence):
        item = sequence[i]
        if item == CAPITALIZE and i + 1 < len(sequence):
            sequence[i+1] = sequence[i+1].capitalize()
        elif item in (',', "'", '"', '’', '.', ':', ';', '?', '!', ')', ']', '}', '*') and i > 0:
            result[-1] += item
        elif item in ('(', '[', '{') and i + 1 < len(sequence):
            sequence[i+1] = item + sequence[i+1]
        else:
            result.append(item)
        i += 1
    return " ".join(result)


def sequence_from_file(file_path):
    with open(file_path, "r") as f:
        contents = f.read()
    contents = contents.replace("\n", " ")
    new_contents = []
    for c in contents:
        if c in ALPHABET + [" "]:
            new_contents.append(c)
        else:
            new_contents.append(" " + c + " ")
    with open("tmp", "w") as f:
        f.write("".join(new_contents))
    contents = "".join(new_contents).split()
    contents = flatten([[CAPITALIZE, w.lower()] if w[0].isupper() else [
                       w.lower()] for w in contents])
    return contents


def generate_sequences_from_files(vocabulary: Vocabulary, files: list, n_seq: int, seq_len: int, context_size=1, n_workers=8):
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        file_seqs = [sequence_from_file(f) for f in files]
        weights = np.array([len(s) for s in file_seqs])
        p = weights / weights.sum()
        inputs = torch.zeros([n_seq, seq_len, context_size], dtype=int)
        targets = torch.zeros([n_seq, seq_len], dtype=int)

        futures = []
        for i_seq in range(n_seq):
            def task():
                file_seq = file_seqs[np.random.choice(
                    range(len(file_seqs)), p=p)].copy()
                seq_inputs = torch.zeros([seq_len, context_size], dtype=int)
                seq_targets = torch.zeros([seq_len], dtype=int)
                rn = np.random.randint(context_size - 1, len(file_seq) - seq_len - 1)
                for i_feat, j in enumerate(range(rn, rn+seq_len)):
                    for i_ctx in range(context_size):
                        seq_inputs[i_feat, i_ctx] = vocabulary.words_to_index[file_seq[j-i_ctx]]
                    seq_targets[i_feat] = vocabulary.words_to_index[file_seq[j + 1]]
                return seq_inputs, seq_targets

            futures.append(pool.submit(task))

        for i_seq, (seq_inputs, seq_targets) in enumerate(f.result() for f in futures):
            inputs[i_seq, :, :] = seq_inputs
            targets[i_seq, :] = seq_targets

    return inputs, targets


class TextDataset(torch.utils.data.Dataset):

    def __init__(self, file_list, embedding_dim, n_seq, seq_len, context_size=1, vocabulary=None, queue_size=1, n_workers=8, switch_after=5):
        assert queue_size >= 1

        self.context_size = context_size
        self.file_list = file_list
        self.queue_size = queue_size
        self.n_workers = n_workers
        self.n_seq = n_seq
        self.seq_len = seq_len
        self.switch_after = switch_after
        self.loop_count = 0

        if vocabulary is None:
            self.vocabulary = Vocabulary()
            self.vocabulary.add_txt_files(file_list)
        else:
            self.vocabulary = vocabulary

        self.vocabulary.save("tmp.json")

        self.embedding = nn.Embedding(len(self.vocabulary), embedding_dim)

        self.inputs, self.targets = None, None

        self._queue = deque([])
        self._fill_queue().join()

    def _fill_queue(self):

        def task():
            while len(self._queue) < self.queue_size:
                inp, oup = generate_sequences_from_files(
                    self.vocabulary,
                    self.file_list,
                    self.n_seq,
                    self.seq_len,
                    context_size=self.context_size,
                    n_workers=self.n_workers
                )

                self._queue.append((inp, oup))

        t = Thread(target=task)
        t.start()
        return t

    def __getitem__(self, index):

        if not self._queue and (index == 0 or self.inputs is None):
            self._fill_queue().join()

        if index == 0 or self.inputs is None:
            if self.loop_count < self.switch_after and self.inputs is not None:
                self.loop_count += 1
            else:
                if not self._queue:
                    self._fill_queue().join()
                self.loop_count = 0
                self.inputs, self.targets = self._queue.popleft()
                self._fill_queue()

        return self.inputs[index, :], self.targets[index, :]

    def __len__(self):
        return self.n_seq


def generate_text(model, vocabulary, context_size, start: list, amount):
    text = start
    assert len(start) == context_size
    with torch.no_grad():
        model.cpu()
        inp = torch.Tensor([[[vocabulary.words_to_index[s] for s in start]]]).type(
            torch.LongTensor)
        hss = None
        for i in range(amount):
            out, hss = model(inp, hss=hss, return_hs=True)
            out_np = out[0, -1, :].numpy()
            chosen = np.random.choice(
                range(len(vocabulary)), p=scipy.special.softmax(out_np))
            chosen = torch.Tensor([[[chosen]]]).type(torch.LongTensor)
            inp = np.concatenate([inp[:, :, 1:], chosen], axis=2)
            text.append(vocabulary.words[chosen])
    return str_from_sequence(text)


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
                f"Epoch {i_epoch}:\ntrain loss: {train_loss[-1]}")
            if i_epoch >= 1:
                print(f"Delta train loss: {train_loss[-1] - train_loss[-2]}")
            if validate:
                print(f", val loss: {val_loss[-1]}")
            print(f"\nTook {time.time() - t} seconds")
            print("-------------")

            if model_path is not None:
                if os.path.exists(model_path):
                    shutil.copyfile(model_path, model_path + '~')
                torch.save(model.state_dict(), model_path)
        except KeyboardInterrupt:
            break

    return model, train_loss, val_loss


if __name__ == '__main__':

    validate = "--validate" in sys.argv

    resume_train = "--resume-train" in sys.argv

    global device
    device = torch.device("cuda:0" if torch.cuda.is_available()
                          and "--cpu" not in sys.argv else "cpu")

    model_path = "models/model_word1"
    vocab_path = model_path + "_vocab"

    ctx_size = 5
    embed_dim = 100
    files = ["data/harry_potter.txt"]

    vocab = None

    n_seq = 1000
    seq_len = 250

    if resume_train:
        vocab = Vocabulary()
        vocab.load(vocab_path)

    data_train = TextDataset(files, embed_dim, n_seq,
                             seq_len, vocabulary=vocab, context_size=ctx_size, switch_after=10)
    vocab = data_train.vocabulary
    data_test = TextDataset(files, embed_dim, 10, 10)

    data = {
        "train": torch.utils.data.DataLoader(data_train, batch_size=10),
        "valid": torch.utils.data.DataLoader(data_test, batch_size=1)
    }


    model = ConvWordTextModel2(len(vocab), embed_dim, context_size=ctx_size)
    # model = SimpleWordTextModel(len(vocab), embed_dim)

    if resume_train:
        model.load_state_dict(torch.load(model_path))

    optimizer = optim.Adadelta(model.parameters())
    criterion = nn.CrossEntropyLoss()
    n_epochs = 100000

    model, train_loss, val_loss = train_model(
        model, data, optimizer, criterion, n_epochs)
    torch.save(model.state_dict(), model_path)
    vocab.save(vocab_path)
    print("lets generate some text:")
    txt = generate_text(model, vocab, ctx_size, ["this", "is", "great", "stuff", "hermoine"], 1000)
    print('"' + txt + '"')
