import json
import time
import copy
from collections import deque

import numpy as np
from tqdm import tqdm
import torch
from statistics import mean
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, accuracy_score
from numpyencoder import NumpyEncoder


class ModelTrainer:
    """Class to perform model training

    Args:
        data_types (list): a list containing data set types, e.g. train, validate, test
        datasets (dict): a dict containing datasets for each type in data_types
        dataloaders (dict): a dict containing dataloaders
        model (Module): the model to be trained
    """

    def __init__(self, data_types: list, datasets: dict, dataloaders: dict, model: torch.nn.Module):
        assert isinstance(datasets, dict)
        for datatype in data_types:
            assert datatype in datasets and datatype in dataloaders, "Missing dataset or dataloader"
        self._l_datatypes = data_types
        self._datasets = datasets
        self._dataloaders = dataloaders
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Current device: {self.device}')

    def save_model(self, path):
        """Save the trained model

        :param path: path to saved model
        """
        print(f"saving model to {path}")
        torch.save(self.model.state_dict(), path)

    def train_model(self, criterion, optimizer, scheduler, num_epochs=2, report_len=500):
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        # Check device
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print(f'training device: {device}')

        # Find dataset sizes
        dataset_sizes = {x: len(self._datasets[x]) for x in self._l_datatypes}

        # Loop through epochs
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in self._l_datatypes:
                print(f'{phase} phase')
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Create a queue to monitor loss
                loss_q = deque(maxlen=report_len)
                acc_q = deque(maxlen=report_len)
                # Iterate over data.
                counter = 0
                for inputs in tqdm(self._dataloaders[phase]):
                    counter += 1
                    # move to gpu
                    inputs = {x: inputs[x].to(self.device) for x in inputs}
                    labels = inputs['label']

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        # print(f"output shape: {outputs.size()}; target shape: {labels.size()}")
                        # _, preds = torch.max(outputs, 1)
                        t_pred = outputs > 0.5
                        #print(f'pred: {t_pred.device}', f'labels: {labels.device}')
                        acc = (t_pred.squeeze() == labels).float().sum() / len(labels)
                        acc_q.append(acc.item())
                        loss = criterion(outputs, labels.unsqueeze(-1).float())
                        loss_q.append(loss.item())
                        if counter % report_len == 0:
                            print(f"Iter {counter}, loss: {mean(loss_q)}, accuracy:{mean(acc_q)}")
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs['image'].size(0)
                    running_corrects += torch.sum(t_pred.squeeze() == labels)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'validate' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)

    def generate_eval_report(self, json_path):
        """Generate evaluation report using trained model

        :param json_path: path to saved json file
        """
        assert 'test' in self._l_datatypes, 'test set not available!'
        self.model.eval()
        l_pred, l_labels = [], []
        with torch.no_grad():
            for batch in tqdm(self._dataloaders['test']):
                batch = {x: batch[x].to(self.device) for x in batch}
                bat_labels = batch['label']
                bat_out = self.model(batch)
                t_pred = bat_out > 0.5
                l_pred.append(t_pred.squeeze().cpu().numpy())
                l_labels.append(bat_labels.squeeze().cpu().numpy())
        v_pred = np.concatenate(l_pred, axis=0)
        v_labels = np.concatenate(l_labels, axis=0)
        metrics = {
            'cmat': confusion_matrix(v_labels, v_pred),
            'f1': f1_score(v_labels, v_pred),
            'mcc': matthews_corrcoef(v_labels, v_pred),
            'accuracy': accuracy_score(v_labels, v_pred)
        }
        with open(json_path, 'w') as j:
            json.dump(metrics, j, cls=NumpyEncoder)
