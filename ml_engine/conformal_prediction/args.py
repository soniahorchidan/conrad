import argparse


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--negative_sample_size", default=128, type=int)
    parser.add_argument("--hidden_dim", default=200, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--train_epochs", default=50, type=int)
    # calib size is calib_size * number of orginal training samples
    parser.add_argument("--calib_size", default=0.5, type=float)
    parser.add_argument("--log_epochs_freq", default=10, type=int)
    parser.add_argument("--val_epochs_freq", default=10, type=int)
    # the size of the training set is
    # calib_size * train_size * number of original training samples
    parser.add_argument("--train_size", default=0.1, type=float)
    # the size of the val set is
    # calib_size * val_size * number of original training samples
    parser.add_argument("--val_size", default=0.4, type=float)
    # the size of the real calibration set is
    # (1 - train_size - val_size) * calib_size * number of original training samples

    if args_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_known_args(args_list)[0]

    return args
