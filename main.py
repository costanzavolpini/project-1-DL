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

###################### LINEAR MODEL #####################
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