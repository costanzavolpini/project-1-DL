"""
General class to define a model to train, test..etc.
"""
import torch
from torch import nn

from matplotlib.pylab import plt

##################### MODEL CLASS #####################
class Model(nn.Module):
    """
    Abstract class to implement model.
    """

    def __init__(self):
        """
        Initialize function for model and history.
        """
        super(Model, self).__init__()
        self.history = History()

    def fit(self, train_input, train_target, test_input = None, test_target = None, epochs = 25, batch_size=25, doPrint=True):
        """
        Fit the model on the training data.
        Inputs:
            train_input: train data
            train_target: train target data
            test_input: test data if any
            test_target: test target data if any
            epochs: number of epochs,
            batch_size: size of the batch,
            doPrint: bool (if true it prints the epochs with loss and accuracy)
        """

        # Iniziale Print to print accuracy and loss for each epoch
        if doPrint:
            p = Print(self)

        def get_loss_accuracy(input_, target):
            """
            Function to get the loss and accuracy. Given the input and target.
            """
            predicted = self(input_) # get predicted (forward method in each subclass model)
            loss = self.criterion(predicted, target) # apply loss (i.e. MSE)
            accuracy = self.compute_accuracy(predicted, target) # compute accuracy
            return loss, accuracy

        for e in range(1, epochs + 1):

            # shuffle the train set to select different batches at each epoch
            indices_shuffled = torch.randperm(train_input.shape[0]) # random permutation of integers from 0 to train_input.shape[0] - 1.
            train_input = train_input[indices_shuffled]

            # shuffle target set according to indices_shuffled
            if isinstance(train_target, tuple): # bool, digit1, digit2 -> take corresponding in each target
                train_target = tuple(t[indices_shuffled] for t in train_target)
            else:
                train_target = train_target[indices_shuffled] # no auxiliary loss then just bool

            # iterate over the batches
            train_loss = 0
            train_acc = 0
            n_batches = 0
            for batch_start in range(0, train_input.shape[0], batch_size):
                n_batches += 1
                # get the current batch from the trainset
                batch_end = batch_start + batch_size
                train_input_batch = train_input[batch_start:batch_end]
                if isinstance(train_target, tuple): # we are using an auxiliary loss -> target is a tuple
                    train_target_batch = tuple(t[batch_start:batch_end] for t in train_target)
                else: # no auxiliary loss -> target is not a tuple
                    train_target_batch = train_target[batch_start:batch_end]

                # set gradients to zero before train step
                self.optimizer.zero_grad()

                # call forward method of subclass inside get_loss_accuracy method to get predicted value
                train_loss_batch, train_acc_batch = get_loss_accuracy(train_input_batch, train_target_batch)
                train_loss += train_loss_batch.item()
                train_acc += train_acc_batch

                train_loss_batch.backward() # backward propagation of gradients
                self.optimizer.step() # update parameters of optimizer

            # do the average across batches
            train_loss /= n_batches
            train_acc /= n_batches

            # Train loss and accuracy have been computed before the train step, test loss and accuracy after it
            test_acc = None
            test_loss = None

            # If we have the test compute loss and accuracy and overwrite test_acc and test_loss
            if test_input is not None:
                test_loss, test_acc = get_loss_accuracy(test_input, test_target)
                test_loss = test_loss.item()

            # add epoch to history saving new values
            self.history.epoch(
                train_loss=train_loss, train_acc=train_acc,
                test_loss=test_loss, test_acc=test_acc
            )

            # print accuracy and loss for each epoch
            if doPrint:
                p()

        return self

    def get_accuracy_train(self):
        # just get the last accuracy of the train
        return self.history.get_accuracies_train()[-1]

    def get_accuracy_test(self):
        # just get the last accuracy of the test
        return self.history.get_accuracies_test()[-1]

    def plot_history(self):
        """
        Plots loss and accuracies.
        """
        self.history.plot_losses()
        plt.plot()
        self.history.plot_accuracies()
        plt.plot()
        plt.show()

    def compute_accuracy(self, y_pred, y_target):
        """
        Function to compute accuracy.
        """
        y_pred = y_pred.clone()
        y_pred[y_pred>0.5] = 1
        y_pred[y_pred<=0.5] = 0
        acc = 100 * ((y_pred == y_target).sum().type(torch.FloatTensor).item())
        n = y_pred.shape[0]
        return acc / n # normalize by divide by length (1000) -> same as mean

    def number_params(self, module=None):
        """
        Return the number of parameters of the model.
        """
        if module is None:
            module = self # parameters of model

        # p.numel() returns #entries (#parameters that define tensor p)
        # p.requires_grad = p is part of neural network
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

##################### HISTORY CLASS #####################

class History():
    """
    Handle history of loss during training.
    """
    def __init__(self):
        """
        Initialize the history (losses, epochs, accuracies of train and test)
        """
        self.epochs = 0
        self.train_losses = []
        self.train_acc = []
        self.test_losses = []
        self.test_acc = []

    def epoch(self, train_loss, train_acc, test_loss=None, test_acc=None):
        """
        Add a new epoch and update all values. This method is called at the end of each epoch from the model.
        """
        self.epochs += 1
        self.train_losses.append(train_loss)
        self.train_acc.append(train_acc)
        self.test_losses.append(test_loss)
        self.test_acc.append(test_acc)

    def plot_general(self, val1, val2, text1, text2, legend, isAccuracy=False):
        """
        General function to plot using matplotlib.
        """
        epochs_range = list(range(1, self.epochs + 1))

        ax = plt.figure(figsize=(7, 5)).gca() # gca = create current axes instance on the current figure

        ax.plot(epochs_range, val1, label=text1) # epochs
        ax.plot(epochs_range, val2, label=text2) # accuracy or loss depend on legend

        ax.grid()
        ax.set_xlabel("epochs", fontsize = 16)
        ax.set_ylabel(legend, fontsize = 16)

        if isAccuracy:
            ax.set_ylim((0, 100)) # accuracy is between 0 and 100

        ax.legend(fontsize = 13)


    def plot_losses(self):
        """
        Plot losses of train and test.
        """
        self.plot_general(self.test_losses, self.train_losses, 'Test loss', 'Train loss', 'loss', False)

    def plot_accuracies(self):
        """
        Plot accuracies of train and test.
        """
        self.plot_general(self.test_acc, self.train_acc, 'Test accuracy', 'Train accuracy', 'accuracy', True)

    # Getter function for accuracies
    def get_accuracies_train(self):
        return self.train_acc

    def get_accuracies_test(self):
        return self.test_acc

################################### PRINT CLASS ###################################
class Print():
    """
    Pring accuracy and loss during training for each epoch.
    """

    def __init__(self, model):
        self.model = model
        self.curr_epoch = self.model.history.epochs # epochs is updated time by time

        # set colors and style for print
        self.style = {
            'PURPLE': '\033[95m',
            'GREEN': '\033[92m',
            'RED': '\033[91m',
            'BOLD': '\033[1m',
            'END': '\033[0m',
        }

        # Print header
        print("{BOLD}{:^10}{END} | {BOLD}{:^25}{END} | {BOLD}{:^25}{END}".format('Epoch', 'Train loss - Accuracy', 'Dev. loss - Accuracy', **self.style))

    # Called every time we do p() -> then for each epoch
    def __call__(self):
        # get hisory
        h = self.model.history

        # compute #epoch
        epoch = str(h.epochs)

        # get last loss of train and test
        train_loss = h.train_losses[-1]
        test_loss =  h.test_losses[-1]

        # get last accuracies
        train_accuracy = h.train_acc[-1]
        test_accuracy = h.test_acc[-1]

        print(
            "{PURPLE}{:^10}{END} | {RED}{:^25}{END} | {GREEN}{:^25}{END}".format(
                epoch, # purple
                '{:5.4f} - {:3.2f}%'.format(train_loss, train_accuracy), # red
                'na - na' if test_loss is None else '{:5.4f} - {:3.2f}%'.format(test_loss, test_accuracy), # green
                **self.style
            )
        )
