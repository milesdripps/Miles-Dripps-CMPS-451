#Miles Dripps
#
# CMPS 451: BiLSTM on weather dataset: "psspredict.csv"
from keras import Input
from numpy import array
from keras.optimizers import Adam
from numpy import reshape
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
from sklearn.metrics import mean_absolute_error
import numpy as np
import time
import csv
import datetime
dateformat =  "%Y/%m/%d %H:%M"
from tensorflow.python.training.adam import AdamOptimizer


def get_sequence():
    with open('psspredict10.csv', mode='r') as csvfile:
        print("converting csv to array...")
        sequence = []
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            # See if the time provides help, convert to int date
            #dt = datetime.datetime.strptime(row[0], dateformat)
            #timestamp = float(dt.timestamp())
            sequence.append([
                float(row[3]), # Just temp
            ])
    return sequence

def create_sequences(sequence, n_steps):
    #Create an information array and an answer array
    # The information will be the past number of steps
    # The answer will be the next number
    x, y = [], []
    for i in range(len(sequence)):

        if (i + n_steps) > (len(sequence) - 1):
            break

        x_seq, y_seq = sequence[i:i + n_steps], sequence[i+n_steps][0]

        x.append(x_seq)
        y.append(y_seq)
    return np.array(x),np.array(y)


def main():
    input_features = input("What are the features? (rain, snow, etc): ")
    sequence = get_sequence()

    # Split the dataset into training and test sets
    # Train the model on 95% of the data( all but last 6 months )
    print("preparing data...")
    train_size = int(len(sequence) * 0.95)
    #test_size = len(sequence) - train_size
    train_sequence, test_sequence = sequence[:train_size], sequence[train_size:]

    # # # # # # CHANGE THESE PARAMS # # # # # # #

    n_steps = 30 # try 7 or 14?
    n_features = len(sequence[0])  # get amount of features from sequence row len
    n_epochs = 20
    neurons = 100
    learningrate = .001

    # # # # # # # # # # # # # # # # # # # # # # # #

    # Prepare training data
    x_train, y_train = create_sequences(train_sequence, n_steps)
    x_train = np.array(x_train)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], n_features)

    # Prepare test data
    x_test, y_test = create_sequences(test_sequence, n_steps)
    x_test = np.array(x_test)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], n_features)

    # Define model
    print("creating model...")
    model = Sequential()
    model.add(Input(shape=(n_steps,n_features)))
    model.add(Bidirectional(LSTM(units=neurons, activation='sigmoid')))
    model.add(Dense(1))
    optimizer = Adam(learning_rate = learningrate)
    model.compile(optimizer=optimizer, loss='mae')
    #losses
    #mae, mse, Huber(), mape, categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy
    #poisson, logcosh

    # Train model
    print("training model...")
    train_start = time.time()
    model.fit(x_train, y_train, epochs=n_epochs, verbose=1) # verbose 1 or 2 to see time
    train_end = time.time()
    train_time = round((train_end - train_start) / 60, 2)

    # Make predictions for the test set (last six months)
    print("predicting last six months...")
    y_pred = model.predict(x_test, verbose=1)

    # Measure accuracy using Mean Absolute Error (MAE) as a percentage
    mae = round((mean_absolute_error(y_test, y_pred)), 2)
    percentage_error = round(((mae / np.mean(y_test)) * 100), 2)

    print(f"Training time:{train_time}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Percentage Error: {100 - percentage_error}%")

    #Log results
    with open("log_trainings.csv", 'a', newline = '') as logfile:
        writer = csv.writer(logfile)
        writer.writerow([n_steps,f"{n_features}: {input_features}",neurons,n_epochs,learningrate,train_time, 100 - percentage_error])
        print("Results logged. Done")
main()



