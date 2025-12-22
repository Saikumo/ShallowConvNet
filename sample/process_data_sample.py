from preprocess_data import load_bnci2014_001_data_from_moabb


def preprocess_bnci2014_001_sample():
    X, y = load_bnci2014_001_data_from_moabb(subject_id=1, train=True)

    mean = X.mean(dim=(0, 2), keepdim=True)
    std = X.std(dim=(0, 2), keepdim=True)

    X = (X - mean) / (std + 1e-9)

    return X, y
