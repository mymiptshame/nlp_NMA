import numpy as np
import pandas as pd


from time import time
from tqdm.notebook import tqdm
from IPython.display import clear_output
from collections import defaultdict

import torch

import matplotlib.pyplot as plt
import seaborn as sns


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_history(history, num_epochs, shuffle=False):
    plt.figure(figsize=(16, 7))

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(num_epochs) + 1, history['train_loss'], label='train loss')
    plt.plot(np.arange(num_epochs) + 1, history['val_loss'], label='validation loss')
    if shuffle:
        plt.plot(np.arange(num_epochs) + 1, history['val_loss_shuffled'], label='validation loss (shuffled)')
    plt.xlabel('num epochs')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(num_epochs) + 1, history['train_acc'], label='train accuracy')
    plt.plot(np.arange(num_epochs) + 1, history['val_acc'], label='validation accuracy')
    if shuffle:
        plt.plot(np.arange(num_epochs) + 1, history['val_acc_shuffled'], label='validation accuracy (shuffled)')
    plt.xlabel('num epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


def train(
        model,
        criterion,
        optimizer,
        train_data,
        val_data,
        num_epochs,
        model_path_to_save,
        history_path_to_save
):
    history = defaultdict(list)
    best_score = 0

    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        train_acc = 0
        val_acc = 0

        start_time = time()
        model.train(True)
        for X_train_batch, y_train_batch in tqdm(train_data):
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)

            predictions = model(X_train_batch)
            loss = criterion(predictions, y_train_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.cpu().item()

            target = y_train_batch.detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy().argmax(axis=1)

            train_acc += (target == predictions).mean()

        train_loss /= len(train_data)
        train_acc /= len(train_data)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        with torch.no_grad():
            model.train(False)
            for X_val_batch, y_val_batch in tqdm(val_data):
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                predictions = model(X_val_batch)
                loss = criterion(predictions, y_val_batch)

                val_loss += loss.cpu().item()

                target = y_val_batch.detach().cpu().numpy()
                predictions = predictions.detach().cpu().numpy().argmax(axis=1)

                val_acc += (target == predictions).mean()

            val_loss /= len(val_data)
            val_acc /= len(val_data)

            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if best_score < val_acc:
                torch.save(model.state_dict(), model_path_to_save)
                pd.DataFrame(history).to_csv(history_path_to_save, index=False)
                best_score = val_acc

        clear_output()
        plot_history(history, epoch + 1)
        print('epoch number: {}'.format(epoch + 1))
        print('time per epoch: {}s'.format(np.round(time() - start_time, 3)))
        print('validation acc: {}'.format(np.round(history['val_acc'][-1], 3) * 100))
        print('validation loss: {}'.format(np.round(history['val_loss'][-1], 3)))
    return model, history


def bert_training_loop(
        model,
        criterion,
        optimizer,
        train_data,
        val_data,
        num_epochs,
        model_path_to_save,
        history_path_to_save
):
    history = defaultdict(list)
    best_score = 0

    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        train_acc = 0
        val_acc = 0

        start_time = time()
        model.train(True)
        for train_batch in tqdm(train_data):
            train_batch = {key: value.to(device) for key, value in train_batch.items()}

            input_ids, attention_mask = train_batch['input_ids'], train_batch['attention_mask']
            target = train_batch['target']

            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.cpu().item()

            target = target.detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy().argmax(axis=1)

            train_acc += (target == predictions).mean()

        train_loss /= len(train_data)
        train_acc /= len(train_data)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        with torch.no_grad():
            model.train(False)
            for val_batch in tqdm(val_data):
                val_batch = {key: value.to(device) for key, value in val_batch.items()}

                input_ids, attention_mask = val_batch['input_ids'], val_batch['attention_mask']
                target = val_batch['target']

                predictions = model(input_ids, attention_mask)
                loss = criterion(predictions, target)

                val_loss += loss.cpu().item()

                target = target.detach().cpu().numpy()
                predictions = predictions.detach().cpu().numpy().argmax(axis=1)

                val_acc += (target == predictions).mean()

            val_loss /= len(val_data)
            val_acc /= len(val_data)

            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if best_score < val_acc:
                torch.save(model.state_dict(), model_path_to_save)
                pd.DataFrame(history).to_csv(history_path_to_save, index=False)
                best_score = val_acc

        clear_output()
        plot_history(history, epoch + 1)
        print('epoch number: {}'.format(epoch + 1))
        print('time per epoch: {}s'.format(np.round(time() - start_time, 3)))
        print('validation acc: {}'.format(np.round(history['val_acc'][-1], 3) * 100))
        print('validation loss: {}'.format(np.round(history['val_loss'][-1], 3)))
    return model, history
    

def bert_shuffled_training_loop(
        model,
        criterion,
        optimizer,
        train_data,
        val_data,
        val_data_shuffled,
        num_epochs,
        model_path_to_save,
        history_path_to_save,
        shuffled_model_path_to_save,
        shuffled_history_path_to_save
):
    history = defaultdict(list)
    best_score = 0
    best_score_shuffled = 0

    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        val_loss_shuffled = 0
        
        train_acc = 0
        val_acc = 0
        val_acc_shuffled = 0

        start_time = time()
        model.train(True)
        for train_batch in tqdm(train_data):
            train_batch = {key: value.to(device) for key, value in train_batch.items()}

            input_ids, attention_mask = train_batch['input_ids'], train_batch['attention_mask']
            target = train_batch['target']

            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.cpu().item()

            target = target.detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy().argmax(axis=1)

            train_acc += (target == predictions).mean()

        train_loss /= len(train_data)
        train_acc /= len(train_data)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        with torch.no_grad():
            model.train(False)
            for val_batch, val_batch_shuffled in tqdm(zip(val_data, val_data_shuffled)):
                val_batch = {key: value.to(device) for key, value in val_batch.items()}

                input_ids, attention_mask = val_batch['input_ids'], val_batch['attention_mask']
                target = val_batch['target']

                predictions = model(input_ids, attention_mask)
                loss = criterion(predictions, target)

                val_loss += loss.cpu().item()

                target = target.detach().cpu().numpy()
                predictions = predictions.detach().cpu().numpy().argmax(axis=1)

                val_acc += (target == predictions).mean()
                
                val_batch_shuffled = {key: value.to(device) for key, value in val_batch_shuffled.items()}

                input_ids, attention_mask = val_batch_shuffled['input_ids'], val_batch_shuffled['attention_mask']
                target = val_batch_shuffled['target']

                predictions = model(input_ids, attention_mask)
                loss = criterion(predictions, target)

                val_loss_shuffled += loss.cpu().item()

                target = target.detach().cpu().numpy()
                predictions = predictions.detach().cpu().numpy().argmax(axis=1)

                val_acc_shuffled += (target == predictions).mean()

            val_loss /= len(val_data)
            val_acc /= len(val_data)
            
            val_loss_shuffled /= len(val_data_shuffled)
            val_acc_shuffled /= len(val_data_shuffled)

            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_loss_shuffled'].append(val_loss_shuffled)
            history['val_acc_shuffled'].append(val_acc_shuffled)

            if best_score < val_acc:
                torch.save(model.state_dict(), model_path_to_save)
                pd.DataFrame(history).to_csv(history_path_to_save, index=False)
                best_score = val_acc
                
            if best_score_shuffled < val_acc_shuffled:
                torch.save(model.state_dict(), shuffled_model_path_to_save)
                pd.DataFrame(history).to_csv(shuffled_history_path_to_save, index=False)
                best_score_shuffled = val_acc_shuffled

        clear_output()
        plot_history(history, epoch + 1, True)
        print('epoch number: {}'.format(epoch + 1))
        print('time per epoch: {}s'.format(np.round(time() - start_time, 3)))
        print('validation acc: {}'.format(np.round(history['val_acc'][-1], 3) * 100))
        print('validation loss: {}'.format(np.round(history['val_loss'][-1], 3)))
    return model, history