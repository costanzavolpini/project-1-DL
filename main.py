import torch
from scripts.data import Data
from scripts.models_implemented import *

model = 6

################### GENERATE DATASETS ###################

d = Data()

#TODO: for 1 to 10 of seed and to plotbox per far vedere outliers
#TODO: fare tabella con num_params per ogni modello

train_input = None
train_target = None
train_classes = None
test_input = None
test_target = None
test_classes = None

if model == 5 or model == 6:
    train_input, train_target, train_classes, test_input, test_target, test_classes = d.get_data_3dCNN()
else:
    train_input, train_target, train_classes, test_input, test_target, test_classes = d.get_data_flatten()

#########################################################

#TODO: cross validation to find best parameters
# Adam’s method considered as a method of Stochastic Optimization is a technique implementing adaptive learning rate. Whereas in normal SGD the learning rate has an equivalent type of effect for all the weights/parameters of the model.

###################### 1. LINEAR MODEL #####################
if(model == 1):
    torch.manual_seed(1)

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
    # Train the model
    torch.manual_seed(1)
    model_logistic = LogisticRegression()

    model_logistic.fit(
        train_input, train_target,
        test_input, test_target,
        epochs=1000,
        doPrint=True
    )

    model_logistic.plot_history()

################## 3. NEURAL NET MODEL #####################
elif(model == 3):
    print("eh")

############## 4. NEURAL NET MODEL(2 LOSSES) ###############
elif(model == 4):
    # class NN2Losses(Model):

    #     def __init__(...):
    #         super(NN2Losses, self).__init__()
    #         self.init_params = {
    #             ...
    #         }
    #         self.feature_extractor = nn.Sequential(...)
    #         # classificatore per >
    #         self.classifier_bool = nn.Sequential(
    #             nn.Linear(features_in, 1),
    #             nn.Sigmoid()
    #         )
    #         # classificatore per le cifre
    #         self.classifier_digit = nn.Sequential(
    #             nn.Linear(features_in, 10)
    #             nn.Sigmoid()
    #         )
    #         self.optimizer = optimizer(self.parameters(), lr=learning_rate)
    #         def custom_criterion(train_pred, test_target):
    #             bool_pred, left_digits_pred, right_digits_pred = train_pred
    #             bool_target, left_digits_target, right_digits_target = train_target

    #             bool_loss = criterion(bool_pred, bool_target)
    #             left_digit_loss = criterion ...
    #             right_digit_loss = criterion ...
    #             return bool_loss + left_digit_loss + right_digit_loss # volendo le puoi anche pesare dando più peso a bool_loss (10*bool_loss + 1*left_digit_loss c+ 1*right_digit_loss)
    #         self.criterion = custom_criterion

#            def compute_accuracy() -> quanto predice bene i digit e quanto il task (img1 < img2)

    #     def forward(self, x):
    #         features = self.feature_extractor(x)
    #         return self.classifier_bool(features), self.classifier_digit(x[prime immagini]), self.classifier_digit(x[seconde immagini]) # return predicted values

    # todo: preparare i target come tupla (target booleano di shape=(N, 1), target left digit di shape=(N, 10), target right digit di shape=(N, 10))
    print("eh")
############ 5. CONVOLUTIONAL NEURAL NETWORK ###############
elif(model == 5):
    # add another dimension (as expected from Conv3d)
    # X = train_input.unsqueeze(1)
    # a(X).shape --> 32 filters => in the second dimensions we have 32 "layers" and each one is calculated by a filter
    model_cnn1 = CNNModel1Loss()

    print("Number of parameters: {}".format(model_cnn1.number_params()))
    print("Number of parameters of feature_extractor: {}".format(model_cnn1.number_params(model_cnn1.feature_extractor)))
    print("Number of parameters of classifier: {}".format(model_cnn1.number_params(model_cnn1.classifier)))

    model_cnn1.fit(
        train_input, train_target,
        test_input, test_target,
        epochs=50,
        doPrint=True
    )

    model_cnn1.plot_history()

############ 6. CONVOLUTIONAL NEURAL NETWORK (2 losses) ###############
else:
    model_cnn2 = CNNModel2Loss()
    print("Number of parameters: {}".format(model_cnn2.number_params()))
    print("Number of parameters of feature_extractor: {}".format(model_cnn2.number_params(model_cnn2.feature_extractor)))
    print("Number of parameters of classifier: {}".format(model_cnn2.number_params(model_cnn2.classifier_bool)))

    model_cnn2.fit(
        train_input, (train_target, train_classes[:, 0], train_classes[:, 1]),
        test_input, (test_target, test_classes[:, 0], test_classes[:, 1]),
        epochs=50,
        doPrint=True
    )

    model_cnn2.plot_history()