import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from code.model import Model

###################### 1. LINEAR MODEL #####################
# Give weight to each pixel
class LinearRegression(Model):
    """
    A linear layer accepts as input a vector of values for each sample, therefore,
    the input samples have been flattened since our raw input sample are overlaid images.
    Therefore, the linear layer assigns a weight to each pixel.
    Input: (N, 392)
    Output: (N, 1)
    """
    def __init__(self, features_in=392, features_out=1, optimizer=optim.Adam, criterion=nn.MSELoss):
        super(LinearRegression, self).__init__()

        # Applies a linear transformation to the incoming data: y = xw + b
        # x.shape = [N, 392] | w.shape = [392, 1] | b is a scalar
        self.classifier = nn.Sequential(
            nn.Linear(features_in, features_out)
        )

        # Adam uses a adaptive learning rates,
        # SGD has a single, fixed, learning rate for all the weights/parameters of the model.
        self.optimizer = optimizer(self.parameters()) #Adam
        self.criterion = criterion() #MSE

    def forward(self, x):
        # flatten the features for the linear layer in the classifier
        x = x.view(x.shape[0], -1)
        return self.classifier(x) # return predicted value

    @classmethod
    def reshape_data(cls, data):
        """
        Return data as float with inputs flatten.
        Output:
            - train_input, test_input: images -> N x 392
            - train_target, test_target: class to predict in range[0,1] -> N x 1
        """
        return data.get_data_flatten()


#################### 2. LOGISTIC MODEL #####################
class LogisticRegression(Model):
    """
    Input: (N, 392)
    Output: (N, 1)
    """
    def __init__(self, features_in=392, features_out=1, optimizer=optim.Adam, criterion=nn.MSELoss):
        super(LogisticRegression, self).__init__()

        # Applies a linear transformation to the incoming data: y = xw + b
        # x.shape = [N, 392] | w.shape = [392, 1] | b is a scalar
        self.classifier = nn.Sequential(
            nn.Linear(features_in, features_out),
            nn.Sigmoid() # return value in range [0, 1]
        )

        # Adam uses a adaptive learning rates,
        # SGD has a single, fixed, learning rate for all the weights/parameters of the model.
        self.optimizer = optimizer(self.parameters())
        self.criterion = criterion()

    def forward(self, x):
        # flatten the features for the linear layer in the classifier
        x = x.view(x.shape[0], -1)
        return self.classifier(x) # return predicted value

    @classmethod
    def reshape_data(cls, data):
        """
        Return data as float with inputs flatten.
        Output:
            - train_input, test_input: images -> N x 392
            - train_target, test_target: class to predict in range[0,1] -> N x 1
        """
        return data.get_data_flatten()

################## 3. NEURAL NET MODEL #####################
class NNModel1Loss(Model):
    """
    Input: (N, 392)
    Output: (N, 1)
    """
    def __init__(self,
        features_in=392, features_out=1, optimizer = optim.Adam, criterion = nn.MSELoss):

        super(NNModel1Loss, self).__init__()

        # Added activation functions: ReLU
        # 2 hidden layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(features_in, 256), # hidden layer
            nn.ReLU(),
            nn.Linear(256, 128), # hidden layer
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, features_out),
            nn.Sigmoid() # return value in range [0, 1]
        )

        # Adam uses a adaptive learning rates,
        # SGD has a single, fixed, learning rate for all the weights/parameters of the model.
        self.optimizer = optimizer(self.parameters())
        self.criterion = criterion()

    def forward(self, X):
        features = self.feature_extractor(X)

        # flatten the features for the linear layer in the classifier
        features = features.view(X.shape[0], -1)
        return self.classifier(features)

    @classmethod
    def reshape_data(cls, data):
        """
        Return data as float with inputs flatten.
        Output:
            - train_input, test_input: images -> N x 392
            - train_target, test_target: class to predict in range[0,1] -> N x 1
        """
        return data.get_data_flatten()

############## 4. NEURAL NET MODEL(2 LOSSES) ###############
class NNModel2Loss(Model):
    """
    Input: (N, 392)
    Output: (N, 1)
    """
    def __init__(self,
        features_in=392, features_out=1, optimizer=optim.Adam, criterion=nn.MSELoss):

        super(NNModel2Loss, self).__init__()

        # The feature extractor processes both images together (it does not consider the two images independently)
        # Added activation functions: ReLU
        # 2 hidden layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(features_in, 256), # hidden layer
            nn.ReLU(),
            nn.Linear(256, 128), # hidden layer
            nn.ReLU()
        )

        # classificatore for > (comparison between 2 images)
        self.classifier_bool = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # classificatore for digit
        # Since the feature extractor processes both images together, we decided to use a digit classifier
        # with 20 nodes as output (10 classes + 10 classes).
        self.classifier_digit = nn.Sequential(
            nn.Linear(128, 20),
            nn.Sigmoid()
        )

        # Adam uses a adaptive learning rates,
        # SGD has a single, fixed, learning rate for all the weights/parameters of the model.
        self.optimizer = optimizer(self.parameters())
        self.criterion = self.custom_criterion
        self.loss_criterion = criterion()

    # Since we are using an auxiliary loss we need to pass our predicted value as tuple (same for target).
    # Then we can use MSE on all the elements of tuple.
    # We give more weight to the loss of boolean classification since we are more interested in comparing 2 digits.
    def custom_criterion(self, train_pred, train_target):
            """
            Function to compute loss when we have an auxiliary loss.
            Input:
                - train_pred: tuple of 3 elements
                - train_target: tuple of 3 elements
            """
            bool_pred, digit_pred1, digit_pred2 = train_pred
            bool_target, digit_target1, digit_target2 = train_target

            bool_loss = self.loss_criterion(bool_pred, bool_target)
            digit_loss1 = self.loss_criterion(digit_pred1, digit_target1)
            digit_loss2 = self.loss_criterion(digit_pred2, digit_target2)

            return 10 * bool_loss + digit_loss1 + digit_loss2

    # predict how accurate predict img1 < img2
    def compute_accuracy(self, y_pred, y_target):
        """
            Function to compute accuracy when we have an auxiliary loss.
            We just compute the accuracy on the boolean prediction.
            Input:
                - y_pred: tuple of 3 elements
                - y_target: tuple of 3 elements
            """
        bool_pred, digit_pred1, digit_pred2 = y_pred
        bool_target, digit_target1, digit_target2 = y_target

        bool_pred = bool_pred.clone()
        bool_pred[bool_pred > 0.5] = 1
        bool_pred[bool_pred <= 0.5] = 0
        acc = 100 * ((bool_pred == bool_target).sum().type(torch.FloatTensor).item())
        n = bool_pred.shape[0]
        return acc / n # normalize by divide by length (1000) -> same as mean

    def forward(self, X):
        # X.shape = (1000, 392) -> (N, 2*14*14)
        # extract features
        features = self.feature_extractor(X)

        # classify >=
        bool_pred = self.classifier_bool(features)

        # classify digits
        digit_pred = self.classifier_digit(features)
        digit_pred_left = digit_pred[:, :10] # first 10 values refer to img1
        digit_pred_right = digit_pred[:, 10:] # last 10 values refer to img2

        return bool_pred, digit_pred_left, digit_pred_right

    @classmethod
    def reshape_data(cls, data):
        """
        Return data as expected by a NN layer with 2 losses.
        Output:
            - train_input, test_input: images -> N x 392
            - train_target, test_target: tuple of 3 composed by target_bool, class_img1, class_img -> (N x 1, N x 10, N x 10)
        """
        return data.get_data_NN2Loss()

############ 5. CONVOLUTIONAL NEURAL NETWORK (3D CONV) ###############
class CNNModel1Loss(Model):
    """
    Predicts whether the first image is <= than the second.
    Only one loss can be applied to the output of this model.
    Input: (N, 1, 2, 14, 14)
    Output: (N, 1)
    """
    def __init__(self,
        output_size=1, optimizer=optim.Adam, criterion=nn.MSELoss):

        super(CNNModel1Loss, self).__init__()

        self.feature_extractor = nn.Sequential(
            # Use batch normalization to normalize the input layer by adjusting and scaling the activations.
            # Allow each layer of a network to learn by itself a little bit more independently of other layers.
            nn.BatchNorm3d(1),

            # padding 2+2 on x-axis, 2+2 on y-axis (depth=0, height=2, width=2)
            # add padding to keep input dimensions
            nn.Conv3d(1, 64, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(64), #num_features=64 -> Learnable Parameters

            nn.Conv3d(64, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Dropout(0.3), # dropout = to make it more general and then have a more robust model
            nn.BatchNorm3d(32),

            # Stride controls how the filter convolves around the input volume. -> it is used to avoid to have a fraction in output volume instead of an integer
            # stride, 1 = filter moves on z-axis (1 pixel), shift by 2 on x-axis, shift by 2 on y-axis
            # padding 1+1 on x-axis, 1+1 on y-axis
            nn.Conv3d(32, 8, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2)),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm3d(8)
        )

        self.classifier = nn.Sequential(
            # 2 images
            # 7 x 7 instead of 14 x 14 as size of the image because we have applied a stride
            nn.Linear(8 * 2 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Sigmoid()
        )

        self.optimizer = optimizer(self.parameters())
        self.criterion = criterion()

    def forward(self, X):
        if len(X.shape) == 4:
            # Conv3d expects an input of shape (N, C_{in}, D, H, W)
            X = X.unsqueeze(1) # unsqueeze: add a dimension

        features = self.feature_extractor(X)

        # flatten the features for the linear layer in the classifier
        features = features.view(X.shape[0], -1)
        return self.classifier(features)

    @classmethod
    def reshape_data(cls, data):
        """
        Return data as expected by a 3dCNN layer.
        Output:
            - train_input, test_input: images -> N x 1 x 2 x 14 x 14
            - train_target, test_target: class to predict in range[0,1] -> N x 1
        """
        return data.get_data_3dCNN()

############ Just for report comparison: CONVOLUTIONAL NEURAL NETWORK 1 loss (2D CONV) ###############
class CNN2dModel1Loss(Model):
    """
    Predicts whether the first image is <= than the second. Only one loss can be applied to the output of this model.
    Input: (N, 2, 1, 14, 14)
    Output: (N, 1)
    """
    def __init__(self,
        output_size=1, optimizer=optim.Adam, criterion=nn.MSELoss):

        super(CNN2dModel1Loss, self).__init__()

        self.feature_extractor = nn.Sequential(
            # Use batch normalization to normalize the input layer by adjusting and scaling the activations.
            # Allow each layer of a network to learn by itself a little bit more independently of other layers.
            nn.BatchNorm2d(1),

            # padding 2+2 on x-axis, 2+2 on y-axis (height=2, width=2)
            # add padding to keep input dimensions
            nn.Conv2d(1, 64, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(64),  #num_features=64 -> Learnable Parameters

            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(0.3), # dropout = to make it more general and then have a more robust model
            nn.BatchNorm2d(32),

            # Stride controls how the filter convolves around the input volume. -> it is used to avoid to have a fraction in output volume instead of an integer
            # stride, filter shift by 2 on x-axis, shift by 2 on y-axis
            # padding 1+1 on x-axis, 1+1 on y-axis
            nn.Conv2d(32, 8, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm2d(8)
        )

        self.classifier = nn.Sequential(
            # 2 images
            # 7 x 7 instead of 14 x 14 as size of the image because we have applied a stride
            nn.Linear(8 * 2 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Sigmoid()
        )

        self.optimizer = optimizer(self.parameters())
        self.criterion = criterion()

    def forward(self, X):
        # Select features of img1 and img2 and flatten the features for the linear layer in the classifier
        features_l = self.feature_extractor(X[:, 0]).view(X.shape[0], -1)
        features_r = self.feature_extractor(X[:, 1]).view(X.shape[0], -1)

        # concat the 2 flatten features
        features = torch.cat([features_l, features_r], dim=1)
        return self.classifier(features)

    @classmethod
    def reshape_data(cls, data):
        """
        Return data as expected by a 2dCNN layer.
        Output:
            - train_input, test_input: images -> N x 2 x 1 x 14 x 14
            - train_target, test_target: class to predict in range[0,1] -> N x 1
        """
        return data.get_data_2dCNN()

############ 6. CONVOLUTIONAL NEURAL NETWORK (2 losses) ###############
class CNNModel2Loss(Model):
    """
    Two losses can be applied to the output of this model.
    Input: (N, 1, 2, 14, 14)
    Output: (N, 21) -> (10 possible values (0 to 9) for each image + 1 to check if image_1 <= image_2)
    """
    def __init__(self, features_in=392, output_size=21, optimizer=optim.Adam, criterion=nn.MSELoss):

        super(CNNModel2Loss, self).__init__()

        self.feature_extractor = nn.Sequential(
            # Use batch normalization to normalize the input layer by adjusting and scaling the activations.
            # Allow each layer of a network to learn by itself a little bit more independently of other layers.
            nn.BatchNorm3d(1),

            # padding 2+2 on x-axis, 2+2 on y-axis (depth=0, height=2, width=2)
            # add padding to keep input dimensions
            nn.Conv3d(1, 64, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(64),

            nn.Conv3d(64, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Dropout(0.3), # dropout = to make it more general and then have a more robust model
            nn.BatchNorm3d(32),

            # Stride controls how the filter convolves around the input volume. -> it is used to avoid to have a fraction in output volume instead of an integer
            # stride, 1 = filter moves on z-axis (1 pixel), shift by 2 on x-axis, shift by 2 on y-axis
            # padding 1+1 on x-axis, 1+1 on y-axis
            nn.Conv3d(32, 8, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2)),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm3d(8)
        )

        # classificatore for > (comparison between 2 images)
        self.classifier_bool = nn.Sequential(
            # 2 images
            # 7 x 7 instead of 14 x 14 as size of the image because we have applied a stride
            nn.Linear(8 * 2 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # classificatore for digit
        self.classifier_digit = nn.Sequential(
            # 8 x 7 x 7 = 392 since we have flatte
            nn.Linear(8 * 7 * 7, 10),
            nn.Sigmoid()
        )

        self.optimizer = optimizer(self.parameters())
        self.criterion = self.custom_criterion
        self.loss_criterion = criterion()

    # Since we are using an auxiliary loss we need to pass our predicted value as tuple (same for target).
    # Then we can use MSE on all the elements of tuple.
    # We give more weight to the loss of boolean classification since we are more interested in comparing 2 digits.
    def custom_criterion(self, train_pred, train_target):
            """
            Function to apply loss on all the elements of tuple.
            Input:
                - train_pred: tuple of 3 elements
                - train_target: tuple of 3 elements
            """
            bool_pred, digit_pred1, digit_pred2 = train_pred
            bool_target, digit_target1, digit_target2 = train_target

            bool_loss = self.loss_criterion(bool_pred, bool_target)
            digit_loss1 = self.loss_criterion(digit_pred1, digit_target1)

            digit_loss2 = self.loss_criterion(digit_pred2, digit_target2)

            return 10 * bool_loss + digit_loss1 + digit_loss2 # give more weight to bool_loss

    # predict how accurate predict img1 < img2
    def compute_accuracy(self, y_pred, y_target):
        """
        Function to compute accuracy when we have an auxiliary loss. Just compute accuracy of bool prediction.
        Input:
            - train_pred: tuple of 3 elements
            - train_target: tuple of 3 elements
        """
        bool_pred, digit_pred1, digit_pred2 = y_pred
        bool_target, digit_target1, digit_target2 = y_target

        bool_pred = bool_pred.clone()
        bool_pred[bool_pred>0.5] = 1
        bool_pred[bool_pred<=0.5] = 0
        acc = 100 * ((bool_pred == bool_target).sum().type(torch.FloatTensor).item())
        n = bool_pred.shape[0]
        return acc / n # normalize by divide by length (1000) -> same as mean

    def forward(self, x):
        if len(x.shape) == 4:
            # Conv3d expects an input of shape (N, C_{in}, D, H, W)
            x = x.unsqueeze(1) #add C_{in}

        features = self.feature_extractor(x)
        # divide features of both images and flat them
        features_first_image = (features[:, :, 0]).contiguous().view(x.shape[0], -1) # shape = (1000, 8 * 7 * 7)
        features_second_image = (features[:, :, 1]).contiguous().view(x.shape[0], -1)

        # flatten for the linear layer in the classifier
        features_flatten = features.view(x.shape[0], -1)
        return self.classifier_bool(features_flatten), self.classifier_digit(features_first_image), self.classifier_digit(features_second_image) # return predicted values

    @classmethod
    def reshape_data(cls, data):
        """
        Return data as expected by a 3dCNN layer with 2 losses.
        Output:
            - train_input, test_input: images -> N x 1 x 2 x 14 x 14
            - train_target, test_target: tuple of 3 composed by target_bool, class_img1, class_img2 -> (N x 1, N x 10, N x 10)
        """
        return data.get_data_3dCNN2Loss()