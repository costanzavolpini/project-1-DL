from dlc_practical_prologue import generate_pair_sets
import torch

"""
Class to handle the data of MNIST.
"""
class Data(object):

    def __init__(self, n=1000):
        """
        Initialize data to train and test. Call generate_pair_sets(n) of file dlc_practical_prologue.
        Inputs:
            - n = number of pairs for training and test sets
        Outputs:
            - train_input, test_input: images -> N x 2 x 14 x 14
            - train_target, test_target: class to predict in range[0,1] -> N
            - train_classes, test_classes: classes of the two digits in [0, 9] -> N x 2
        """
        super().__init__()
        self.train_input, self.train_target, self.train_classes, self.test_input, self.test_target, self.test_classes = generate_pair_sets(n)

    def get_data(self):
        """
        Return a copy of data as float.
        """
        return self.train_input.clone().type(torch.FloatTensor), self.train_target.clone().type(torch.FloatTensor), self.train_classes.clone().type(torch.FloatTensor), self.test_input.clone().type(torch.FloatTensor), self.test_target.clone().type(torch.FloatTensor), self.test_classes.clone().type(torch.FloatTensor)

    def get_data_3dCNN(self):
        """
        Return data as expected by a 3dCNN layer.
        Output:
            - train_input, test_input: images -> N x 1 x 2 x 14 x 14
            - train_target, test_target: class to predict in range[0,1] -> N x 1
        """
        train_input, train_target, train_classes, test_input, test_target, test_classes = self.get_data()
        train_input, train_target, test_input, test_target = train_input.unsqueeze(1), train_target.unsqueeze(1), test_input.unsqueeze(1), test_target.unsqueeze(1)

        # unsqueeze: add a dimension, example: (n, d) become (n, d, 1) with unsqueeze(2)
        return train_input, train_target, test_input, test_target

    def get_data_2dCNN(self):
        """
        Return data as expected by a 2dCNN layer.
        Output:
            - train_input, test_input: images -> N x 2 x 1 x 14 x 14
            - train_target, test_target: class to predict in range[0,1] -> N x 1
        """
        train_input, train_target, train_classes, test_input, test_target, test_classes = self.get_data()
        train_input, train_target, test_input, test_target = train_input.unsqueeze(2), train_target.unsqueeze(1), test_input.unsqueeze(2), test_target.unsqueeze(1)

        # unsqueeze: add a dimension
        return train_input, train_target, test_input, test_target

    def get_data_3dCNN2Loss(self):
        """
        Return data as expected by a 3dCNN layer with 2 losses.
        Output:
            - train_input, test_input: images -> N x 1 x 2 x 14 x 14
            - train_target, test_target: tuple of 3 composed by target_bool, class_img1, class_img2 -> (N x 1, N x 10, N x 10)
        """
        train_input, train_target, train_classes, test_input, test_target, test_classes = self.get_data()
        train_input, train_target, test_input, test_target = train_input.unsqueeze(1), train_target.unsqueeze(1), test_input.unsqueeze(1), test_target.unsqueeze(1)

        # Add one hot encoding
        train_classes_img1 = self.transform_one_hot_encoding(train_classes[:, 0])
        train_classes_img2 = self.transform_one_hot_encoding(train_classes[:, 1])
        test_classes_img1 = self.transform_one_hot_encoding(test_classes[:, 0])
        test_classes_img2 = self.transform_one_hot_encoding(test_classes[:, 1])

        # Create a tuple of 3 for the output (since we are using auxiliary losses)
        train_target = (train_target, train_classes_img1, train_classes_img2)
        test_target = (test_target, test_classes_img1, test_classes_img2)

        return train_input, train_target, test_input, test_target

    def get_data_NN2Loss(self):
        """
        Return data as expected by a NN layer with 2 losses.
        Output:
            - train_input, test_input: images -> N x 392
            - train_target, test_target: tuple of 3 composed by target_bool, class_img1, class_img -> (N x 1, N x 10, N x 10)
        """
        train_input, train_target, test_input, test_target = self.get_data_3dCNN2Loss()
        train_input = train_input.reshape(train_input.size(0), -1) # 392 = 14 x 14 x 2
        test_input = test_input.reshape(train_input.size(0), -1)
        return train_input, train_target, test_input, test_target

    def get_data_flatten(self):
        """
        Return data as float with inputs flatten.
        Output:
            - train_input, test_input: images -> N x 392
            - train_target, test_target: class to predict in range[0,1] -> N x 1
        """
        train_input, train_target, train_classes, test_input, test_target, test_classes = self.get_data()
        train_input = train_input.reshape(self.train_input.size(0), -1)
        test_input = test_input.reshape(self.test_input.size(0), -1)
        # unsqueeze: add a dimension
        train_target = train_target.unsqueeze(1)
        test_target = test_target.unsqueeze(1)
        return train_input, train_target, test_input, test_target

    def transform_one_hot_encoding(self, data):
        """
        Transform the target in one-hot-encoding to have 10 values for classes.
        Example: class 2: 0 1 0 0 0 0 0 0 0 0
        Output:
            - data_hot_encoding: 10 values for class. -> N x 10
        """
        data_hot_encoding = torch.zeros(data.shape[0], 10)

        # select cell where we want 1
        # range(data.shape[0]) -> iterate on all rows (1000)
        # and in each row at positions data.type(torch.LongTensor) (that is for example 2, 8, 3, 1 -> all positions for each row) put a 1
        data_hot_encoding[range(data.shape[0]), data.type(torch.LongTensor)] = 1

        return data_hot_encoding
