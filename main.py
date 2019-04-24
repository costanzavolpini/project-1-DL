import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from scripts.model import Model
from scripts.data import Data

################### GENERATE DATASETS ###################

d = Data()
train_input, train_target, train_classes, test_input, test_target, test_classes = d.get_data_flatten()

#########################################################
model = 5

###################### 1. LINEAR MODEL #####################
if(model == 1):
    # Give weight to each pixel (this weight is taken from gradient descent)
    class LinearRegression(Model):

        def __init__(self, features_in=392, features_out=1, optimizer=optim.SGD, criterion=nn.MSELoss, learning_rate=1e-8):
            super(LinearRegression, self).__init__()
            self.init_params = {
                'features_in': features_in,
                'features_out': features_out,
                'optimizer': optimizer,
                'criterion': criterion,
                'learning_rate': learning_rate
            }
            self.classifier = nn.Sequential(
                nn.Linear(features_in, features_out)
            )
            self.optimizer = optimizer(self.parameters(), lr=learning_rate)
            self.criterion = criterion()

        def forward(self, x):

            # flatten the features for the linear layer in the classifier
            x = x.view(1000, -1)
            return self.classifier(x) # return predicted value

    # Train the model
    model_linear = LinearRegression()

    model_linear.fit(
        train_input, train_target,
        test_input, test_target,
        epochs=50,
        doPrint=True
    )

    model_linear.plot_history()

#################### 2. LOGISTIC MODEL #####################
elif(model == 2):
    class LogisticRegression(Model):

        def __init__(self, features_in=392, features_out=1, optimizer=optim.SGD, criterion=nn.MSELoss, learning_rate=1e-2):
            super(LogisticRegression, self).__init__()
            self.init_params = {
                'features_in': features_in,
                'features_out': features_out,
                'optimizer': optimizer,
                'criterion': criterion,
                'learning_rate': learning_rate
            }
            self.classifier = nn.Sequential(
                nn.Linear(features_in, features_out)
            )
            self.optimizer = optimizer(self.parameters(), lr=learning_rate)
            self.criterion = criterion()

        def forward(self, x):

            # flatten the features for the linear layer in the classifier
            x = x.view(1000, -1)
            return F.sigmoid(self.classifier(x)) # return predicted value

    # Train the model
    model_logistic = LogisticRegression()

    model_logistic.fit(
        train_input, train_target,
        test_input, test_target,
        epochs=50,
        doPrint=True
    )

    model_logistic.plot_history()

################## 3. NEURAL NET MODEL #####################
elif(model == 3):
    print("eh")

############## 4. NEURAL NET MODEL(2 LOSSES) ###############
elif(model == 4):
    print("eh")
############ 5. CONVOLUTIONAL NEURAL NETWORK ###############
elif(model == 5):
#     # add padding to keep input dimensions
#     a = nn.Conv3d(1, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1))

#     # simply add another dimension (as expected from Conv3d)
#     X = train_input.unsqueeze(1)
#     # a(X).shape --> 32 filters => in the second dimensions we have 32 "strati" ognuno calcolato da un filtro

    class CNNModel1Loss(Model):
        """
        Predicts whether the first image is <= than the second. Only one loss can be applied to the output of this model.
        Input: (N, 2, 14, 14)
        Output: (N, 1)
        """
        def __init__(self,
            # cambia output_size per decidere quante classi vuoi in output. Per l'altro modello con 2 losses
            # passare semplicemente output_size=21 invece che implementare un altra classe
            output_size=1, optimizer = torch.optim.Adam, criterion = torch.nn.MSELoss):

            super(CNNModel1Loss, self).__init__()

            self.init_params = {
                'optimizer': optimizer,
                'criterion': criterion
            }

            self.feature_extractor = nn.Sequential(
                nn.BatchNorm3d(1),

                nn.Conv3d(1, 32, kernel_size=(1, 5, 5), padding=(0, 2, 2)),
                nn.ReLU(),
                nn.BatchNorm3d(32),

                nn.Conv3d(32, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.ReLU(),
                nn.BatchNorm3d(16),

                nn.Conv3d(16, 8, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2)),
                nn.ReLU(),
                nn.BatchNorm3d(8),
            )

            self.classifier = nn.Sequential(
                nn.Linear(8 * 2 * 7 * 7, 256),
                nn.ReLU(),
                nn.Linear(256, output_size),
                nn.Tanh() # or sigmoid (TODO: attenzione al target che dai durante il train con la sigmoid)
            )

            self.optimizer = optimizer(self.parameters())
            self.criterion = criterion()

        def forward(self, X):
            if len(X.shape) == 4:
                # Conv3d expects an input of shape (N, C_{in}, D, H, W)
                X = X.unsqueeze(1)

            features = self.feature_extractor(X)

            # flatten the features for the linear layer in the classifier
            features = features.view(1000, -1)
            return self.classifier(features)

    model_cnn1 = CNNModel1Loss()

    model_cnn1.fit(
        train_input, train_target,
        test_input, test_target,
        epochs=50,
        doPrint=True
    )

    model_cnn1.plot_history()
############ 6. CONVOLUTIONAL NEURAL NETWORK (2 losses) ###############
else:
    print("best")