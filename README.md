# Binary-and-Multiclass-Image-Classification
An implementation of the Perceptron and Logistic Regression Classification Models designed from scratch, on the MNIST Dataset

# Description
An analyis of the performance of Perceptron and Logistic Regression Classification models (built from the ground up, without using existing libraries) on Synthetic Data as well as on the MNIST Image Classification Dataset.

This implementation also contains comparative performances of Stochastic Gradient Descent vs Gradient Descent in Multiclass Image Classification

# Model Performance

    *** Synthetic data ***
    perceptron train acc: 0.994286, test acc: 1.000000
    logistic train acc: 0.994286, test acc: 1.000000

    *** Two Moon data ***
    perceptron train acc: 0.820000, test acc: 0.840000
    logistic train acc: 0.857143, test acc: 0.866667

    *** Binarized MNIST data ***
    perceptron train acc: 0.873400, test acc: 0.828000
    logistic train acc: 0.871000, test acc: 0.834000

    *** Toy Data multiclass: 3 class classification ***
    sgd training time: 0.042755 seconds
    train acc: 0.937143, test acc: 0.945000
    gd training time: 0.164852 seconds
    train acc: 0.950000, test acc: 0.950000

    *** MNIST: 10 class classification ***
    sgd training time: 0.061660 seconds
    train acc: 0.771800, test acc: 0.773000
    gd training time: 4.283981 seconds
    train acc: 0.945400, test acc: 0.896000
