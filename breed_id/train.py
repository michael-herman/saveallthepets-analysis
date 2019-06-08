import torch
import time
import numpy as np


class Trainer(object):
    def __init__(self, train_loader, validation_loader,
                 criterion=None, optimizer=None, scheduler=None, lr=0.001):
        self._train_loader = train_loader
        self._validation_loader = validation_loader
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._lr = lr
        self._device = None

    def _fit(self, model, data_loader, train=False):
        if train:
            model.train()
        else:
            model.eval()
        running_loss = 0.0
        running_acc = 0.0
        for data, target in data_loader:
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
            running_loss += loss.item() * data.size(0)

            # Calculate accuracy
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            correct = top_class == target.view(*top_class.shape)
            running_acc += torch.mean(correct.type(torch.FloatTensor))

        loss = running_loss / len(data_loader.dataset)
        acc = 100. * running_acc / len(data_loader)

        return loss, acc

    def train_model(self, arch: str, model, epochs) -> dict:
        """ Trains and validates the data using the specified model and parameters.

        Returns:
            checkpoint (dict): Last saved model checkpoint.
        """

        # Use GPU if available, attach device to model accordingly
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        print(f'Training model on {self._device}')
        model.to(self._device)

        # Track losses and accuracy over epochs
        loss_per_epoch_train = list()
        loss_per_epoch_valid = list()
        acc_per_epoch_train = list()
        acc_per_epoch_valid = list()

        start_train_timer = time.time()
        valid_loss_min = np.Inf
        checkpoint = dict()
        for epoch in range(epochs):
            # Start timer
            start = time.time()

            # Pass forward through the model
            if self._scheduler is not None:
                self._scheduler.step()

            train_loss, train_acc = self._fit(model, self._train_loader,
                                              train=True)
            valid_loss, valid_acc = self._fit(model, self._validation_loader,
                                              train=False)

            # calculate average loss over an epoch
            elapsed_epoch = time.time() - start
            loss_per_epoch_train.append(train_loss)
            loss_per_epoch_valid.append(valid_loss)
            acc_per_epoch_train.append(train_acc)
            acc_per_epoch_valid.append(valid_acc)

            # print training/validation statistics
            print('Epoch: {} - completed in: {:.0f}m {:.0f}s'.format(
                epoch + 1, elapsed_epoch // 60, elapsed_epoch % 60))
            print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                train_loss, valid_loss))
            print('\tTraining acc: {:.3f} \tValidation acc: {:.3f}'.format(
                train_acc, valid_acc))

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min, valid_loss))

                checkpoint = {
                    'arch': arch,
                    'epoch': epoch,
                    'epochs': epochs,
                    'run_time': time.time() - start_train_timer,
                    'sched_state_dict': self._scheduler.state_dict(),
                    'optimizer_state_dict': self._optimizer.state_dict(),
                    'state_dict': model.state_dict(),
                    'train_acc': acc_per_epoch_train,
                    'valid_acc': acc_per_epoch_valid,
                    'train_loss': loss_per_epoch_train,
                    'valid_loss': loss_per_epoch_valid
                }

                model_name = f'{arch}_model.pt'
                print(f'\tModel saved as: {model_name}')
                torch.save(checkpoint, model_name)
                valid_loss_min = valid_loss

        training_time = time.time() - start_train_timer
        hours = training_time // (60 * 60)
        training_time -= hours * 60 * 60
        print('Model training completed in: {:.0f}h {:.0f}m {:.0f}s'.format(
            hours, training_time // 60, training_time % 60))
        return checkpoint
