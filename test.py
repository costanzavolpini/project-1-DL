import torch
from code.data import Data
from code.models_implemented import *

import timeit
import time

# train the best model
model = 6

################### GENERATE DATASETS ###################

d = Data()

###################### 1. LINEAR MODEL #####################
if(model == 1):

    # Train the model
    model_linear = LinearRegression()
    train_input, train_target, test_input, test_target = LinearRegression.reshape_data(d)

    print("Number of parameters: {}".format(model_linear.number_params()))
    print("Number of parameters of feature_extractor: /")
    print("Number of parameters of classifier: {}".format(model_linear.number_params(model_linear.classifier)))

    model_linear.fit(
        train_input, train_target,
        test_input, test_target,
        epochs=1000,
        batch_size=1000, #so fast that batch does not make sense to use batch
        doPrint=True
    )

    model_linear.plot_history()

#################### 2. LOGISTIC MODEL #####################
elif(model == 2):
    # Train the model
    model_logistic = LogisticRegression()
    train_input, train_target, test_input, test_target = LogisticRegression.reshape_data(d)

    print("Number of parameters: {}".format(model_logistic.number_params()))
    print("Number of parameters of feature_extractor: /")
    print("Number of parameters of classifier: {}".format(model_logistic.number_params(model_logistic.classifier)))

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
    train_input, train_target, test_input, test_target = NNModel1Loss.reshape_data(d)


    print("Number of parameters: {}".format(model_nn1.number_params()))
    print("Number of parameters of feature_extractor: {}".format(model_nn1.number_params(model_nn1.feature_extractor)))
    print("Number of parameters of classifier: {}".format(model_nn1.number_params(model_nn1.classifier)))

    model_nn1.fit(
        train_input, train_target,
        test_input, test_target,
        epochs=300,
        batch_size=1000, #so fast that batch does not make sense to use batch
        doPrint=True
    )

    model_nn1.plot_history()

############## 4. NEURAL NET MODEL(2 LOSSES) ###############
elif(model == 4):
    model_nn2 = NNModel2Loss()
    train_input, train_target, test_input, test_target = NNModel2Loss.reshape_data(d)

    print("Number of parameters: {}".format(model_nn2.number_params()))
    print("Number of parameters of feature_extractor: {}".format(model_nn2.number_params(model_nn2.feature_extractor)))
    print("Number of parameters of classifier: {}".format(model_nn2.number_params(model_nn2.classifier_bool)))

    model_nn2.fit(
        train_input, train_target,
        test_input, test_target,
        epochs=300,
        batch_size=1000, #so fast that batch does not make sense to use batch
        doPrint=True
    )

    model_nn2.plot_history()

############ 5. CONVOLUTIONAL NEURAL NETWORK ###############
elif(model == 5):
    model_cnn1 = CNNModel1Loss()
    train_input, train_target, test_input, test_target = CNNModel1Loss.reshape_data(d)
    print("Number of parameters: {}".format(model_cnn1.number_params()))
    print("Number of parameters of feature_extractor: {}".format(model_cnn1.number_params(model_cnn1.feature_extractor)))
    print("Number of parameters of classifier: {}".format(model_cnn1.number_params(model_cnn1.classifier)))

    model_cnn1.fit(
        train_input, train_target,
        test_input, test_target,
        epochs=25,
        batch_size=128,
        doPrint=True
    )

    model_cnn1.plot_history()

############ 6. CONVOLUTIONAL NEURAL NETWORK (2 losses) ###############
elif (model == 6):
    model_cnn2 = CNNModel2Loss()

    train_input, train_target, test_input, test_target = CNNModel2Loss.reshape_data(d)
    print("Number of parameters: {}".format(model_cnn2.number_params()))
    print("Number of parameters of feature_extractor: {}".format(model_cnn2.number_params(model_cnn2.feature_extractor)))
    print("Number of parameters of classifier: {}".format(model_cnn2.number_params(model_cnn2.classifier_bool)))

    model_cnn2.fit(
        train_input, train_target,
        test_input, test_target,
        epochs=25,
        batch_size=128,
        doPrint=True
    )

    model_cnn2.plot_history()

############ Just for comparison, CNN (1 loss and 2d conv) ###############
else:
    model_cnn2d_1 = CNN2dModel1Loss()

    train_input, train_target, test_input, test_target = CNN2dModel1Loss.reshape_data(d)
    print("Number of parameters: {}".format(model_cnn2d_1.number_params()))
    print("Number of parameters of feature_extractor: {}".format(model_cnn2d_1.number_params(model_cnn2d_1.feature_extractor)))

    # start = time.time()
    model_cnn2d_1.fit(
        train_input, train_target,
        test_input, test_target,
        epochs=25,
        batch_size=128,
        doPrint=True
    )
    # end = time.time()
    # print(end - start) #time: 164

    model_cnn2d_1.plot_history()