from sklearn.model_selection import KFold
from code.data import Data
from code.models_implemented import *
import matplotlib.pylab as plt

# get the data
d = Data()

# define the number of splits for the cross validation
n_splits=5

# define the parameters for the fit of the models
model_class_fit_params = {
    LinearRegression: {'epochs': 1000, 'batch_size': 1000},
    LogisticRegression: {'epochs': 1000, 'batch_size': 1000},
    NNModel1Loss: {'epochs': 500, 'batch_size': 1000},
    NNModel2Loss: {'epochs': 1000, 'batch_size': 1000},
    CNNModel1Loss: {'epochs': 70, 'batch_size': 50},
    CNNModel2Loss: {'epochs': 70, 'batch_size': 50},
}

# cross validate
kf = KFold(n_splits=n_splits)
accuracies = {}
for model_class, fit_params in model_class_fit_params.items():
    model_class_name = model_class.__name__
    print('Evaluating', model_class_name)
    train_input, train_target, test_input, test_target = model_class.reshape_data(d)

    accuracies[model_class_name] = {'cross_val_accuracies': [], 'train_accuracy': None, 'test_accuracy': None}
    # cross validdation estimate the score we will obtain with the test set
    for train_index, test_index in kf.split(train_input):
        X_train, X_test = train_input[train_index], train_input[test_index]
        if isinstance(train_target, tuple):
            y_train = tuple([y[train_index] for y in train_target])
            y_test = tuple([y[test_index] for y in train_target])
        else:
            y_train, y_test = train_target[train_index], train_target[test_index]

        model = model_class()
        model.fit(
            X_train, y_train,
            X_test, y_test,
            doPrint=False,
            **fit_params
        )

        accuracies[model_class_name]['cross_val_accuracies'].append(model.get_accuracy_test())

    # train the last time on all the trainset and compute the scores on test set
    model = model_class()
    model.fit(
        train_input, train_target,
        test_input, test_target,
        doPrint=False,
        **fit_params
    )
    # model.plot_history()
    # plt.savefig(model_class_name)

    accuracies[model_class_name]['train_accuracy'] = model.get_accuracy_train()
    accuracies[model_class_name]['test_accuracy'] = model.get_accuracy_test()

print(accuracies)
f = open("accuracies.txt","w")
f.write(str(accuracies))
f.close()