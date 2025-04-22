from training import evaluate_model, load_data, DATASET_PATH, IMAGE_SIZE
import pickle


def main():
    (x_train, x_test, y_train, y_test), class_names = load_data(DATASET_PATH, IMAGE_SIZE)

    print("loading model")
    with open('cnn_multiclass_model.pkl', 'rb') as f:
        net = pickle.load(f)

    evaluate_model(net, x_test, y_test, class_names)

main()
