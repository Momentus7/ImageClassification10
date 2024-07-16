def load_cifar10_data():
    from keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    return (X_train, y_train), (X_test, y_test)
