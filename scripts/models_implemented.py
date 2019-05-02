import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from scripts.model import Model

###################### 1. LINEAR MODEL #####################
# Give weight to each pixel (this weight is taken from gradient descent)
class LinearRegression(Model):

    def __init__(self, features_in=392, features_out=1, optimizer=optim.Adam, criterion=nn.MSELoss):
        super(LinearRegression, self).__init__()
        self.init_params = {
            'features_in': features_in,
            'features_out': features_out,
            'optimizer': optimizer,
            'criterion': criterion
        }
        self.classifier = nn.Sequential(
            nn.Linear(features_in, features_out)
        )
        self.optimizer = optimizer(self.parameters())
        self.criterion = criterion()

    def forward(self, x):
        # flatten the features for the linear layer in the classifier
        x = x.view(x.shape[0], -1)
        return self.classifier(x) # return predicted value

    @classmethod
    def reshape_data(cls, data):
        return data.get_data_flatten()


#################### 2. LOGISTIC MODEL #####################
class LogisticRegression(Model):

    def __init__(self, features_in=392, features_out=1, optimizer=optim.Adam, criterion=nn.MSELoss):
        super(LogisticRegression, self).__init__()
        self.init_params = {
            'features_in': features_in,
            'features_out': features_out,
            'optimizer': optimizer,
            'criterion': criterion
        }
        self.classifier = nn.Sequential(
            nn.Linear(features_in, features_out),
            nn.Sigmoid()
        )
        self.optimizer = optimizer(self.parameters())
        self.criterion = criterion()

    def forward(self, x):
        # flatten the features for the linear layer in the classifier
        x = x.view(x.shape[0], -1)
        return self.classifier(x) # return predicted value

    @classmethod
    def reshape_data(cls, data):
        return data.get_data_flatten()

################## 3. NEURAL NET MODEL #####################
class NNModel1Loss(Model):
    """
    Input: (N, 2, 14, 14)
    Output: (N, 1)
    """
    def __init__(self,
        features_in=392, features_out=1, optimizer = optim.Adam, criterion = nn.MSELoss):

        super(NNModel1Loss, self).__init__()

        self.init_params = {
            'optimizer': optimizer,
            'criterion': criterion
        }

        self.feature_extractor = nn.Sequential(
            nn.Linear(features_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, features_out),
            nn.Sigmoid()
        )

        self.optimizer = optimizer(self.parameters())
        self.criterion = criterion()

    def forward(self, X):
        features = self.feature_extractor(X)

        # flatten the features for the linear layer in the classifier
        features = features.view(X.shape[0], -1)
        return self.classifier(features)

    @classmethod
    def reshape_data(cls, data):
        return data.get_data_flatten()

############## 4. NEURAL NET MODEL(2 LOSSES) ###############
class NNModel2Loss(Model):
    def __init__(self,
        features_in=392, features_out=1, optimizer = optim.Adam, criterion = nn.MSELoss):

        super(NNModel2Loss, self).__init__()

        self.init_params = {
            'optimizer': optimizer,
            'criterion': criterion
        }

        self.feature_extractor = nn.Sequential(
            nn.Linear(features_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # classificatore for > (comparison between 2 images)
        self.classifier_bool = nn.Sequential(
            # 2 images
            # 7 x 7 instead of 14 x 14 as size of the image because we have applied a stride
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # classificatore for digit
        self.classifier_digit = nn.Sequential(
            nn.Linear(128, 20),
            nn.Sigmoid()
        )

        self.optimizer = optimizer(self.parameters())
        self.criterion = self.custom_criterion
        self.loss_criterion = criterion()

    def custom_criterion(self, train_pred, train_target):
            """
            Input:
                - train_pred: tuple of 3 elements
            """
            bool_pred, digit_pred1, digit_pred2 = train_pred
            bool_target, digit_target1, digit_target2 = train_target

            bool_loss = self.loss_criterion(bool_pred, bool_target)
            digit_loss1 = self.loss_criterion(digit_pred1, digit_target1)

            digit_loss2 = self.loss_criterion(digit_pred2, digit_target2)

            return 10 * bool_loss + digit_loss1 + digit_loss2 # give more weight to bool_loss (10*bool_loss + 1*digit_loss1 + 1*digit_loss2)

    # predict how accurate predict img1 < img2
    def compute_accuracy(self, y_pred, y_target):
        bool_pred, digit_pred1, digit_pred2 = y_pred
        bool_target, digit_target1, digit_target2 = y_target

        bool_pred = bool_pred.clone()
        bool_pred[bool_pred>0.5] = 1
        bool_pred[bool_pred<=0.5] = 0
        acc = 100 * ((bool_pred == bool_target).sum().type(torch.FloatTensor).item())
        n = bool_pred.shape[0]
        return acc / n # normalize by divide by length (1000) -> same as mean

    def forward(self, X):
        # X.shape = (N, 2*14*14)
        # extract features
        features = self.feature_extractor(X)

        # classify >=
        bool_pred = self.classifier_bool(features)

        # classify digits
        digit_pred = self.classifier_digit(features)
        digit_pred_left = digit_pred[:, :10]
        digit_pred_right = digit_pred[:, 10:]

        return bool_pred, digit_pred_left, digit_pred_right

    @classmethod
    def reshape_data(cls, data):
        return data.get_data_NN2Loss()

############ 5. CONVOLUTIONAL NEURAL NETWORK ###############
# add padding to keep input dimensions
# a = nn.Conv3d(1, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1))
class CNNModel1Loss(Model):
    """
    Predicts whether the first image is <= than the second. Only one loss can be applied to the output of this model.
    Input: (N, 2, 14, 14)
    Output: (N, 1)
    """
    def __init__(self,
        output_size=1, optimizer = optim.Adam, criterion = nn.MSELoss):

        super(CNNModel1Loss, self).__init__()

        self.init_params = {
            'optimizer': optimizer,
            'criterion': criterion
        }

        self.feature_extractor = nn.Sequential(
            nn.BatchNorm3d(1),

            # padding 2+2 on x-axis, 2+2 on y-axis
            nn.Conv3d(1, 64, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(64),

            nn.Conv3d(64, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Dropout(0.3), # dropout = to make it more general and then have a more robust model
            nn.BatchNorm3d(32),

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
            X = X.unsqueeze(1)

        features = self.feature_extractor(X)

        # flatten the features for the linear layer in the classifier
        features = features.view(X.shape[0], -1)
        return self.classifier(features)

    @classmethod
    def reshape_data(cls, data):
        return data.get_data_3dCNN()

############ 6. CONVOLUTIONAL NEURAL NETWORK (2 losses) ###############
# Output = 11 (10 possible values (0 to 9) + 1 to check if image_1 <= image_2)
class CNNModel2Loss(Model):

    def __init__(self, features_in=392, output_size=11, optimizer=optim.Adam, criterion=nn.MSELoss):

        super(CNNModel2Loss, self).__init__()

        self.init_params = {
            'optimizer': optimizer,
            'criterion': criterion
        }

        self.feature_extractor = nn.Sequential(
            nn.BatchNorm3d(1),

            # padding 2+2 on x-axis, 2+2 on y-axis
            nn.Conv3d(1, 64, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
            nn.ReLU(),
            nn.BatchNorm3d(64),

            nn.Conv3d(64, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Dropout(0.3), # dropout = to make it more general and then have a more robust model
            nn.BatchNorm3d(32),

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
            # 14 x 14 = 196 since we have flatted
            nn.Linear(14 * 14, 10),
            nn.Sigmoid()
        )

        self.optimizer = optimizer(self.parameters())
        self.criterion = self.custom_criterion
        self.loss_criterion = criterion()

    def custom_criterion(self, train_pred, train_target):
            """
            Input:
                - train_pred: tuple of 3 elements
            """
            bool_pred, digit_pred1, digit_pred2 = train_pred
            bool_target, digit_target1, digit_target2 = train_target

            bool_loss = self.loss_criterion(bool_pred, bool_target)
            digit_loss1 = self.loss_criterion(digit_pred1, digit_target1)

            digit_loss2 = self.loss_criterion(digit_pred2, digit_target2)

            return 10*bool_loss + digit_loss1 + digit_loss2 # give more weight to bool_loss (10*bool_loss + 1*digit_loss1 + 1*digit_loss2)

    # predict how accurate predict img1 < img2
    def compute_accuracy(self, y_pred, y_target):
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
            x = x.unsqueeze(1)
        features = self.feature_extractor(x)

        # flatten for the linear layer in the classifier
        features = features.view(x.shape[0], -1)
        first_image = (x[:, :, 0]).view(x.shape[0], -1) #shape = [1000, 14*14]
        second_image = (x[:, :, 1]).view(x.shape[0], -1)

        return self.classifier_bool(features), self.classifier_digit(first_image), self.classifier_digit(second_image) # return predicted values

    @classmethod
    def reshape_data(cls, data):
        return data.get_data_3dCNN2Loss()