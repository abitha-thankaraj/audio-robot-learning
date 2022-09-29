import pickle

def save_pkl_file(fname, obj):
    with open(fname, 'wb')  as f:
        pickle.dump(obj, f)
    print("Saved {}".format(fname))

def load_pkl_file(fname, print_msg=True):
    if print_msg:
        print("Loaded from {}".format(fname))
    with open(fname, 'rb')  as f:
        return pickle.load(f)
