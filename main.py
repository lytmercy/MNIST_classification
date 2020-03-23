import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


class CNN:
    def __init__(self):
        self.batch_size = 128
        self.num_classes = 10
        self.epochs = 10

        # Create model for method :create_the_model
        self.model = Sequential()

    # Method for preprocess the data
    def precess_the_data(self):
        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape: ', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        return x_train, x_test, y_train, y_test, input_shape

    def create_the_model(self, input_shape):
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    def train_the_model(self, x_train, x_test, y_train, y_test):
        hist = self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs,
                              verbose=1, validation_data=(x_test, y_test))
        print("\n>>The model has successfully trained<<")

        self.model.save('mnist.h5')
        print("\n>>Saving the model as [mnist.h5]<<")

    def evaluate_the_model(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print('\nTest loss: ', score[0])
        print('Test accuracy: ', score[1])


def main():
    print('hello this Keras project<<')
    cnn_1 = CNN()

    # run method: precess_the_data
    x_train, x_test, y_train, y_test, input_shape = cnn_1.precess_the_data()

    # run method: create_the_model
    cnn_1.create_the_model(input_shape)

    # run method: train_the_model, and save model in file
    cnn_1.train_the_model(x_train, x_test, y_train, y_test)

    # run method: evaluate_the_model
    cnn_1.evaluate_the_model(x_test, y_test)


if __name__ == '__main__':
    main()
