"""
General class to define a model to train, test..etc.
"""
import torch
from torch import nn

from matplotlib.pylab import plt
from matplotlib.ticker import MaxNLocator

##################### MODEL CLASS #####################

class Model(nn.Module):
    """
    Abstract class to implement model.
    Subclass of this module should implement the following methods:
    - self.init_params: a map composed by
        - self.optim: pointer to optimizer passed in init.
        - self.criterion: pointer to loss fn passed in init.
    """
    def __init__(self):
        """
        Initialize function.
        """
        super(Model, self).__init__()
        self.history = History()

    def fit(self, train_input, train_target, test_input = None, test_target = None, epochs = 25, doPrint=True):
        """
        Fit the model on the training data.
        Inputs:
            train_input: train data
            train_target: train target data
            test_input: test data if any
            test_target: test target data if any
            epochs: int,
            doPrint: bool (if true it prints the epochs with loss and accuracy)
        """
        if doPrint:
            p = Print(self) #self is the model

        def get_loss_acc(input_, target):
            pred = self(input_)
            loss = self.criterion(pred, target) # apply loss (i.e. MSE)
            acc = self.compute_accuracy(pred, target) # compute accuracy
            return loss, acc

        for e in range(1, epochs + 1):

            self.optimizer.zero_grad()

            # return predicted value (forward method of subclass model)
            train_loss, train_acc = get_loss_acc(train_input, train_target)

            train_loss.backward() # backward propagation of grads
            self.optimizer.step() # update params

            # NOTE: train loss and accuracy have been computed before the train step, test loss and accuracy after it
            test_acc = None
            test_loss = None
            # do the same with test if it is passed and overwrite test_acc and test_loss
            if test_input is not None:
                test_loss, test_acc = get_loss_acc(test_input, test_target)

            # add epoch to history saving new values
            self.history.epoch(
                train_loss=train_loss.item(), train_acc=train_acc,
                test_loss=test_loss.item(), test_acc=test_acc
            )

            # ----- print all the information
            if doPrint:
                p()

        return self

    # def __hash__(self):
    #         return id(self)

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
        Just take first value in y_pred.
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
            module = self
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

    def epoch(self, train_loss, train_acc, test_loss = None, test_acc = None):
        """
        Add a new epoch and update all values.
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

        ax = plt.figure(figsize=(7, 5)).gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.plot(epochs_range, val1, label=text1)
        ax.plot(epochs_range, val2, label=text2)

        ax.grid()

        ax.set_xlabel("epochs", fontsize = 16)
        ax.set_ylabel(legend, fontsize = 16)

        if isAccuracy:
            ax.set_ylim((0, 100))
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

################################### PRINT CLASS ###################################
class Print():
    """ Pring log messages during training. """

    def __init__(self, model):
        self.model = model
        self.curr_epoch = self.model.history.epochs # epochs is updated time by time

        self.style = {
            'PURPLE': '\033[95m',
            'BLUE': '\033[94m',
            'GREEN': '\033[92m',
            'RED': '\033[91m',
            'BOLD': '\033[1m',
            'UNDERLINE': '\033[4m',
            'END': '\033[0m',
        }

        print("{BOLD}{:^10}{END} | {BOLD}{:^25}{END} | {BOLD}{:^25}{END}".format('Epoch', 'Train loss - Accuracy', 'Dev. loss - Accuracy', **self.style))

    def __call__(self):
        # compute how much time the epoch lasted
        h = self.model.history

        # compute epochs num
        curr_epoch = str(h.epochs)

        # get last losses
        tr_loss, test_loss = h.train_losses[-1], h.test_losses[-1]

        # get last accuracies
        tr_acc, test_acc = h.train_acc[-1], h.test_acc[-1]

        print(
            "{PURPLE}{:^10}{END} | {RED}{:^25}{END} | {GREEN}{:^25}{END}".format(
                curr_epoch,
                '{:5.4f} - {:3.2f}%'.format(tr_loss, tr_acc),
                'na - na' if test_loss is None else '{:5.4f} - {:3.2f}%'.format(test_loss, test_acc),
                **self.style
            )
        )
