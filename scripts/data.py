from scripts.dlc_practical_prologue import generate_pair_sets
import torch

class Data(object):
    def __init__(self, n=1000):
        """
        Initialize data to train and test.
        """
        super().__init__()
        self.train_input, self.train_target, self.train_classes, self.test_input, self.test_target, self.test_classes = generate_pair_sets(n)

    def flat_input(self):
        """
        Flat the input (append images after one other)
        """
        self.train_input_flatten = self.train_input.clone().reshape(self.train_input.size(0), -1)
        self.test_input_flatten = self.test_input.clone().reshape(self.test_input.size(0), -1)
        return self.train_input_flatten, self.test_input_flatten

    def get_data(self):
        """
        Return data as float
        """
        return self.train_input.clone().type(torch.FloatTensor), self.train_target.clone().type(torch.FloatTensor), self.train_classes.clone().type(torch.FloatTensor), self.test_input.clone().type(torch.FloatTensor), self.test_target.clone().type(torch.FloatTensor), self.test_classes.clone().type(torch.FloatTensor)

    def get_data_flatten(self):
        """
        Return data as float with inputs flatten
        """
        self.flat_input()
        return self.test_input_flatten.clone().type(torch.FloatTensor), self.train_target.clone().type(torch.FloatTensor), self.train_classes.clone().type(torch.FloatTensor), self.test_input_flatten.clone().type(torch.FloatTensor), self.test_target.clone().type(torch.FloatTensor), self.test_classes.clone().type(torch.FloatTensor)
