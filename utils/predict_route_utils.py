import pickle


def load_model():
    with open('final_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


