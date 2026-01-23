import argparse


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)

    # pretrained model
    parser.add_argument("--use_pretrained", action="store_true", default=False)
    parser.add_argument("--pretrained_model_url", default=None, type=str)

    parser.add_argument("--learning_rate", default=5.0e-4, type=float)
    parser.add_argument("--max_steps", default=300000, type=int)

    parser.add_argument("--valid_steps_freq", default=50000, type=int)
    parser.add_argument("--log_steps_freq", default=100, type=int)

    parser.add_argument("--train_min_hops", default=1, type=int)
    parser.add_argument("--train_max_hops", default=3, type=int)

    parser.add_argument("--val_min_hops", default=1, type=int)
    parser.add_argument("--val_max_hops", default=3, type=int)
    parser.add_argument("--early_stopping_threshold", default=3, type=int)

    parser.add_argument("--dropout_ratio", default=0.25, type=float)
    # GNNs trained on link prediction exhibit the multi-source propagation issue (see the paper)
    # We can partly alleviate it by thresholding intermediate scores
    # Specific to ULTRA implementation. Should not touch it if you don't know what you are doing
    # Note: For official UltraQuery checkpoints (ultraquery.pth), set this to 0.0
    # For vanilla ULTRA checkpoints (ultra_3g.pth, ultra_4g.pth, ultra_50g.pth), use 0.8 or higher
    parser.add_argument("--msp_threshold", default=0.0, type=float)
    parser.add_argument("--acceptance_threshold", default=0.7, type=float)

    # query sampler
    parser.add_argument("--max_num_ans", default=500, type=int)
    parser.add_argument("--gen_num", default="1.0,1.0", type=str)
    parser.add_argument("--gen_val_num", default=0.001, type=float)
    parser.add_argument("--query_generator_log_ratio", default=0.3, type=float)

    if args_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_known_args(args_list)[0]
    return args
