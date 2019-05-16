# CLASSIFICATION, WEIGHT SHARING, AUXILIARY LOSSES
## Project 1
**EPFL | Deep Learning | EE559** <br>
Project realized in context of the master course EE-559 Deep Learning at EPFL (Summer Semester 2018/2019).

Professor: François **Fleuret**

Students: Francis **Damachi**, Costanza **Volpini**

### DESCRIPTION:
The aim of this project was to show the impact of weight sharing and the use of an auxiliary loss. The task was to compare two digits represented as a two-channel image. We have started experimenting with a linear model and, then, we have implemented more complex non-linear and convolutional models. Our experiments show that, indeed, the use of weight sharing, i.e. convolutional layers, and the use of two losses leads to the best solution in his classification problem, achieving an error rate around 15%.

### RESULTS:
Convolutional Neural Network represents the best model with images recognition. Moreover, the CNN with 2 losses seems to perform slightly better since it learned to extract high-lever feature representing the two digits. We have seen that weight sharing (CNN) improves the accuracy and the robustness of the model, the use of an auxiliary loss in this context improves the obtained results. Our implementation of CNN takes around 190 seconds, the architecture that we have implemented requires __conv3d__ and __batchnorm3d__ that maybe are not optimized for CPU. In the __models_implemented.py__ we implemented the same architecture but using only 2D convolutions, this implementation requires around 160 seconds.

### CODE STRUCTURE:
- code/data.py: class to generate the dataset. Contains different methods (e.g. flat the input, get a dataset in 2D or in 3D, enable the hot-encoding).
- code/model.py: general class to define a model to train and test it (with corresponding plot and history).
- code/models_implemented.py: contains all the model classes (e.g. NNModel1Loss, CNNModel2Loss).
- main.py: contains examples of each model, in order to call and train it.
Files cross_validation_report.py and BoxPlot_generator.ipynb are made for report purposes.

### TO RUN THE CODE:
Ensure that you have in the root of the project the file __dlc_practical_prologue.py__.
From the root of the project: ``` python test.py ``` (by default we run the model CNN with 2 losses).

If you want to try different models, open __test.py__, at line 9 change the variable model with one of these int value:
- 1: Linear model
- 2: Logistic model
- 3: Neural Network model with 1 loss
- 4: Neural Network model with 2 losses
- 5: Convolutional Neural Network with 1 loss (3D conv)
- 6: Convolutional Neural Network with 2 losses (3D conv)
- 7 (or any other values that is not one of the listed above): Convolutional Neural Network with 1 loss (2D conv)
