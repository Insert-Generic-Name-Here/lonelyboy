import pickle

def save_pickle(object, file)
    with open(file, "wb") as fp:   #Pickling
        pickle.dump(object, fp)
        print('Saved')

def load_pickle(file):
    print('Opening...')
    with open(file, "rb") as fp:
        return pickle.load(fp)
