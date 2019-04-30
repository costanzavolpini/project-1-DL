import torch
from scripts.data import Data
from scripts.models_implemented import *

model = 4

################### GENERATE DATASETS ###################

d = Data()

train_input = None
train_target = None
train_classes = None
test_input = None
test_target = None
test_classes = None

if model == 5 or model == 6: # get 3d images
    train_input, train_target, train_classes, test_input, test_target, test_classes = d.get_data_3dCNN()
elif model == 4: #get image in 2d
    train_input, train_target, train_classes, test_input, test_target, test_classes = d.get_data_2dCNN()
else: # append img 1 and 2
    train_input, train_target, train_classes, test_input, test_target, test_classes = d.get_data_flatten()

#########################################################

# Adam’s method considered as a method of Stochastic Optimization is a technique implementing adaptive learning rate.
# Whereas in normal SGD the learning rate has an equivalent type of effect for all the weights/parameters of the model.

###################### 1. LINEAR MODEL #####################
if(model == 1):
    torch.manual_seed(1)

    # Train the model
    model_linear = LinearRegression()

    model_linear.fit(
        train_input, train_target,
        test_input, test_target,
        epochs=50,
        batch_size=1000, #so fast that batch does not make sense to use batch
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
        batch_size=1000, #so fast that batch does not make sense to use batch
        doPrint=True
    )

    model_logistic.plot_history()

################## 3. NEURAL NET MODEL #####################
elif(model == 3):
    model_nn1 = NNModel1Loss()

    print("Number of parameters: {}".format(model_nn1.number_params()))
    print("Number of parameters of feature_extractor: {}".format(model_nn1.number_params(model_nn1.feature_extractor)))
    print("Number of parameters of classifier: {}".format(model_nn1.number_params(model_nn1.classifier)))

    model_nn1.fit(
        train_input, train_target,
        test_input, test_target,
        epochs=50,
        batch_size=1000, #so fast that batch does not make sense to use batch
        doPrint=True
    )

    model_nn1.plot_history()

############## 4. NEURAL NET MODEL(2 LOSSES) ###############
elif(model == 4):
    model_nn2 = NNModel2Loss()
    print("Number of parameters: {}".format(model_nn2.number_params()))
    print("Number of parameters of feature_extractor: {}".format(model_nn2.number_params(model_nn2.feature_extractor)))
    print("Number of parameters of classifier: {}".format(model_nn2.number_params(model_nn2.classifier_bool)))

    train_classes_img1 = d.transform_one_hot_encoding(train_classes[:, 0])
    train_classes_img2 = d.transform_one_hot_encoding(train_classes[:, 1])
    test_classes_img1 = d.transform_one_hot_encoding(test_classes[:, 0])
    test_classes_img2 = d.transform_one_hot_encoding(test_classes[:, 1])

    model_nn2.fit(
        train_input, (train_target, train_classes_img1, train_classes_img2),
        test_input, (test_target, test_classes_img1, test_classes_img2),
        epochs=50,
        batch_size=1000, #so fast that batch does not make sense to use batch
        doPrint=True
    )

    model_nn2.plot_history()

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

    train_classes_img1 = d.transform_one_hot_encoding(train_classes[:, 0])
    train_classes_img2 = d.transform_one_hot_encoding(train_classes[:, 1])
    test_classes_img1 = d.transform_one_hot_encoding(test_classes[:, 0])
    test_classes_img2 = d.transform_one_hot_encoding(test_classes[:, 1])

    model_cnn2.fit(
        train_input, (train_target, train_classes_img1, train_classes_img2),
        test_input, (test_target, test_classes_img1, test_classes_img2),
        epochs=50,
        doPrint=True
    )

    model_cnn2.plot_history()