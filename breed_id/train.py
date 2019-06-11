import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tnrange
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.metrics import classification_report

# Package modules
from breed_id.breed_id_utils import SAVED_MODELS_DIR


class Trainer(object):
    def __init__(self, train_loader, validation_loader,
                 criterion=None, optimizer=None, scheduler=None,
                 save_dir=SAVED_MODELS_DIR):
        self._train_loader = train_loader
        self._validation_loader = validation_loader
        self._save_dir = save_dir
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._device = None

    def _fit(self, model, data_loader, train=False) -> tuple:
        """
        Helper function for completing forward and back propagation accordingly.

        Args:
            model (torch.nn.Nodule): Model to be trained.
            data_loader (Dataloader): Dataset to be trained.
            train (bool): Implements training workflow if True, else eval
                for validation step.
        """
        if train:
            model.train()
        else:
            model.eval()
        running_loss = 0.0
        running_acc = 0.0
        y_true = list()
        y_pred = list()
        for data, target in data_loader:
            y_true.extend(target.data.numpy().tolist())
            data, target = data.to(self._device), target.to(self._device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = self._criterion(output, target)
            if train:
                # clear the gradients of all optimized variables
                self._optimizer.zero_grad()
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self._optimizer.step()
            # update running training loss
            running_loss += float(loss.item()) * data.size(0)

            # Calculate accuracy
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            y_pred.extend(top_class.data.cpu().numpy().tolist())
            correct = top_class == target.view(*top_class.shape)
            running_acc += torch.mean(correct.type(torch.FloatTensor))

        loss = running_loss / len(data_loader.dataset)
        acc = 100. * running_acc / len(data_loader)

        return loss, acc, y_true, y_pred

    def train_model(self, arch: str, model, epochs,
                    device_int=0, unfreeze=False,
                    checkpoint=None, patience=20) -> dict:
        """ Trains and validates the data using the specified model and parameters.

        Args:
            arch (str): The base name for saving the model file.
            model (torch.nn.Module): Torch model to be trained.
            epochs (int): The number of iterations for training the model.
            device_int (in): To directly specify which cuda device to use
                if training on gpu. Default is 0.
            unfreeze (bool): Ensures feature extraction layers are unfrozen if
                you want to perform fine-tuning of feature layers for transfer
                learning.
            checkpoint (dict): Model checkpoint of a saved mode. Allows for
                continuous history tracking of metrics.
            patience (int): Allows for early stopping. How many epochs we want
                to wait after the last time the validation loss improved before
                breaking the training loop. If None or 0, no early stopping is
                implemented. Must be greater than 0. Default: 20

        Returns:
            checkpoint (dict): Last saved model checkpoint.
        """

        # Use GPU if available, attach device to model accordingly
        if torch.cuda.is_available():
            self._device = torch.device(type='cuda', index=device_int)
            torch.cuda.empty_cache()  # Free up unused pytorch cache
        else:
            self._device = torch.device(type='cpu')
        print(f'Training model on {self._device}')
        model.to(self._device)

        # Unfreeze all layers if requested
        if unfreeze:
            for param in model.parameters():
                param.requires_grad = True

        if checkpoint:
            loss_per_epoch_train = checkpoint['train_loss']
            loss_per_epoch_valid = checkpoint['valid_loss']
            acc_per_epoch_train = checkpoint['train_acc']
            acc_per_epoch_valid = checkpoint['valid_acc']
            mcc_per_epoch_train = checkpoint['train_mcc']
            mcc_per_epoch_valid = checkpoint['valid_mcc']
            epoch_beg = checkpoint['epoch'] + 1
            epoch_end = epoch_beg + epochs
            valid_loss_min = loss_per_epoch_valid[-1]  # Use last valid_min
        else:
            checkpoint = dict()
            loss_per_epoch_train = list()
            loss_per_epoch_valid = list()
            acc_per_epoch_train = list()
            acc_per_epoch_valid = list()
            mcc_per_epoch_train = list()
            mcc_per_epoch_valid = list()
            epoch_beg = 0
            epoch_end = epoch_beg + epochs
            valid_loss_min = np.Inf

        early_stop_counter = 0
        start_train_timer = time.time()

        for epoch in tnrange(epoch_beg, epoch_end):
            # Start timer
            start = time.time()

            # Pass forward through the model
            if self._scheduler is not None:
                self._scheduler.step()

            train_loss, train_acc, train_y_true, train_y_pred = self._fit(
                model, self._train_loader, train=True)
            valid_loss, valid_acc, valid_y_true, valid_y_pred = self._fit(
                model, self._validation_loader, train=False)

            # calculate average loss over an epoch
            elapsed_epoch = time.time() - start
            loss_per_epoch_train.append(train_loss)
            loss_per_epoch_valid.append(valid_loss)
            acc_per_epoch_train.append(train_acc)
            acc_per_epoch_valid.append(valid_acc)

            # Calculate and store mcc
            mcc_train = matthews_corrcoef(y_true=train_y_true,
                                          y_pred=train_y_pred)
            mcc_valid = matthews_corrcoef(y_true=valid_y_true,
                                          y_pred=valid_y_pred)
            mcc_per_epoch_train.append(mcc_train)
            mcc_per_epoch_valid.append(mcc_valid)

            # print training/validation statistics
            print('Epoch: {} - completed in: {:.0f}m {:.0f}s'.format(
                epoch + 1, elapsed_epoch // 60, elapsed_epoch % 60))
            print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                train_loss, valid_loss))
            print('\tTraining acc: {:.3f} \tValidation acc: {:.3f}'.format(
                train_acc, valid_acc))
            print('\tTraining mcc: {:.3f} \tValidation mcc: {:.3f}'.format(
                mcc_train, mcc_valid))

            # save model if validation loss has decreased and reset
            # early stopping counter if required
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min, valid_loss))

                checkpoint = {
                    'arch': arch,
                    'epoch': epoch,
                    'epochs': epochs,
                    'run_time': time.time() - start_train_timer,
                    'optimizer_state_dict': self._optimizer.state_dict(),
                    'state_dict': model.state_dict(),
                    'train_acc': acc_per_epoch_train,
                    'valid_acc': acc_per_epoch_valid,
                    'train_loss': loss_per_epoch_train,
                    'valid_loss': loss_per_epoch_valid,
                    'train_mcc': mcc_per_epoch_train,
                    'valid_mcc': mcc_per_epoch_valid,
                    'train_y_true': train_y_true,
                    'train_y_pred': train_y_pred,
                    'valid_y_true': valid_y_true,
                    'valid_y_pred': valid_y_pred
                }

                # Only save scheduler state if exists
                if self._scheduler:
                    checkpoint['sched_state_dict'] = self._scheduler.state_dict()
                else:
                    checkpoint['sched_state_dict'] = None

                model_name = f'{arch}.pt'
                print(f'\tModel saved as: {model_name}')
                save_path = os.path.join(self._save_dir, model_name)
                torch.save(checkpoint, save_path)
                valid_loss_min = valid_loss
                early_stop_counter = 0  # reset counter
            else:
                # If early stopping, increment counter
                if patience > 0:
                    early_stop_counter += 1
                    print(f'EarlyStopping counter: {early_stop_counter} out of {patience}')
                    if early_stop_counter >= patience:
                        print('\tSTOPPING TRAINING EARLY!')
                        break

        training_time = time.time() - start_train_timer
        hours = training_time // (60 * 60)
        training_time -= hours * 60 * 60
        print('Model training completed in: {:.0f}h {:.0f}m {:.0f}s'.format(
            hours, training_time // 60, training_time % 60))
        return checkpoint

    @staticmethod
    def display_classification_report(true_labels, pred_labels, classes):
        report = classification_report(y_true=true_labels,
                                       y_pred=pred_labels,
                                       labels=classes)
        print(report)

    @staticmethod
    def plot_confusion_matrix(true_labels, pred_labels, classes,
                              save_as='confusion_matrix.csv'):
        total_classes = len(classes)
        level_labels = [total_classes * [0], list(range(total_classes))]

        cm = confusion_matrix(y_true=true_labels, y_pred=pred_labels,
                              labels=classes)
        cm_df = pd.DataFrame(data=cm,
                             columns=pd.MultiIndex(levels=[['Predicted:'],
                                                           classes],
                                                   labels=level_labels),
                             index=pd.MultiIndex(levels=[['Actual:'], classes],
                                                 labels=level_labels))

        if save_as:
            cm_df.to_csv(os.path.join(SAVED_MODELS_DIR, save_as))
        print(cm_df)

    # Graph training loss vs validation loss and accuracy over epochs
    @staticmethod
    def plot_loss(arch: str, train_loss: list, valid_loss: list,
                  train_metric: list, valid_metric: list,
                  metric_title='Accuracy', train_metric_label='Train Accuracy',
                  valid_metric_label='Validation Accuracy'):

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        f.suptitle(f'{arch} Model Performance', fontsize=12)
        f.subplots_adjust(top=0.85, wspace=0.3)

        epochs = list(range(1, len(train_loss) + 1))
        ax1.plot(epochs, train_metric, label=train_metric_label)
        ax1.plot(epochs, valid_metric, label=valid_metric_label)
        if len(train_loss) > 20:
            ax1.set_xticks(range(1, len(train_loss) + 1, 10))
        else:
            ax1.set_xticks(epochs)
        ax1.set_ylabel(f'{metric_title} Value')
        ax1.set_xlabel('Epoch')
        ax1.set_title(metric_title)
        ax1.legend(loc="best")

        ax2.plot(epochs, train_loss, label='Train Loss')
        ax2.plot(epochs, valid_loss, label='Validation Loss')
        if len(train_loss) > 20:
            ax2.set_xticks(range(1, len(train_loss) + 1, 10))
        else:
            ax2.set_xticks(epochs)
        ax2.set_ylabel('Loss Value')
        ax2.set_xlabel('Epoch')
        ax2.set_title('Loss')
        ax2.legend(loc="best")
        plt.show()

        # x = list(range(1, len(train_loss) + 1))
        # fig, ax = plt.subplots()
        #
        # ax.plot(x, train_loss, label='Training Loss')
        # ax.plot(x, valid_loss, label="Validation Loss")
        # ax.legend(loc=2)
        # ax.set_xlabel = 'Epochs'
        # ax.set_ylabel = 'Loss'
        #
        # if train_metric and valid_metric:
        #     ax2 = ax.twinx()
        #     ax2.plot(x, valid_metric, label=valid_metric_label, color='red')
        #     ax2.plot(x, train_metric, label=train_metric_label, color='green')
        #     ax.set_ylabel = metric_title
        # plt.set_title = f'{arch} Model Loss Per Epoch'
        # plt.show()
