import pickle
import hashlib

def save_pickle(object, filename):
    with open(filename, "wb") as fp:   #Pickling
        pickle.dump(object, fp)
        print('Saved')

def load_pickle(filename):
    print('Opening...')
    with open(filename, "rb") as fp:
        return pickle.load(fp)

def get_checksum_of_dataframe(df):
    return hashlib.sha256(df.to_json().encode()).hexdigest()
