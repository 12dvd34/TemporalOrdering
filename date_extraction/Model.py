from date_extraction.DDLoss import DDLoss
import json
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Some parameters for the model and training
HIDDEN_SIZE = 4096
HIDDEN_2 = 512
HIDDEN_3 = 2048
NUM_EPOCHS = 200
TRAINING_DATA_SIZE = 4000
BATCH_SIZE = 20
# Change to "cpu" if no GPUs available
device = torch.device("cuda")


class Model(nn.Module):
    def __init__(self, input_size=768, output_size=43):
        super(Model, self).__init__()
        self.hidden = nn.Linear(input_size, HIDDEN_SIZE)
        self.hidden2 = nn.Linear(HIDDEN_SIZE, HIDDEN_2)
        self.hidden3 = nn.Linear(HIDDEN_2, HIDDEN_3)
        self.dropout = nn.Dropout(p=0.1)
        self.out = nn.Linear(HIDDEN_3, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.relu(x)
        x = self.dropout(x)
        output = self.out(x)
        return output


def date_to_tensor(date):
    # only day and month
    date_split = date.split(".")
    date_tensor = torch.zeros(43)
    date_tensor[int(date_split[0]) - 1] = 1
    date_tensor[int(date_split[1]) + 30] = 1
    return date_tensor


def infer(model, input):
    with torch.no_grad():
        output = model(input.to(device))
        tensor_day = torch.split(output, [31, 12])[0]
        tensor_month = torch.split(output, [31, 12])[1]
        num_day = torch.argmax(tensor_day)
        num_month = torch.argmax(tensor_month)
        return num_day.item() + 1, num_month.item() + 1


def test(model, data):
    days_off = 0
    for embedding, date in data:
        result_day, result_month = infer(model, embedding)
        day = int(date.split(".")[0])
        month = int(date.split(".")[1])
        try:
            date_target = datetime.date(2022, month, day)
        except ValueError:
            date_target = datetime.date(2022, month, 28)
        try:
            date_result = datetime.date(2022, result_month, result_day)
        except ValueError:
            date_result = datetime.date(2022, result_month, 28)
        days_off += abs((date_result - date_target).days)
    return round(days_off/len(data))


# Filepath might change depending on where the function is called from
def run(filepath="../synth_data_embs.jsonl"):
    # Load data
    eval_data = []
    train_data = []
    # Load the training data twice, but don't batch one of it, to evaluate performance on training data
    train_data_unbatched = []
    with open(filepath, "r", encoding="utf-8") as file:
        c = 0
        for line in file:
            if c >= TRAINING_DATA_SIZE:
                break
            text, date = json.loads(line)
            bert_embedding = torch.tensor(text)
            date_tensor = date_to_tensor(date)
            # Load roughly 10% of data for evaluation
            if c < int(TRAINING_DATA_SIZE / 10):
                eval_data.append((bert_embedding, date))
            else:
                train_data.append((bert_embedding, date_tensor))
                train_data_unbatched.append((bert_embedding, date))
            c += 1
    # Generate batches
    batched_inputs = []
    for d in train_data[::BATCH_SIZE]:
        t = torch.zeros(BATCH_SIZE, len(d[0]))
        for i in range(BATCH_SIZE):
            t[i] = d[0]
        batched_inputs.append(t)
    batched_targets = []
    for d in train_data[::BATCH_SIZE]:
        t = torch.zeros(BATCH_SIZE, len(d[1]))
        for i in range(BATCH_SIZE):
            t[i] = d[1]
        batched_targets.append(t)
    model = Model().to(device)
    dd_loss = DDLoss(device=device).to(device)
    optimizer = optim.Adam(model.parameters())
    # Training loop
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for i in range(len(batched_inputs)):
            optimizer.zero_grad()
            input = batched_inputs[i].to(device)
            target = batched_targets[i].to(device)
            output = model(input)
            loss = dd_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print("epoch " + str(epoch) + ": " + str(epoch_loss / len(train_data)))
            print("eval days off: " + str(test(model, eval_data)))
            print("train days off: " + str(test(model, train_data_unbatched)))

