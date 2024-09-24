import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="gu_model")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="Beibei",
                        help="[Beibei, Taobao, Yelp, Tmall_1]")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )

    parser.add_argument('--epoch', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=512, help='batch size in evaluation phase')
    parser.add_argument('--dim', type=int, default=256, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight, 1e-5 for NGCF')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument("--mess_dropout", type=bool, default=False, help="consider mess dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.5, help="ratio of mess dropout")
    parser.add_argument("--edge_dropout", type=bool, default=False, help="consider edge dropout or not")
    parser.add_argument("--edge_dropout_rate", type=float, default=0.5, help="ratio of edge sampling")
    parser.add_argument("--batch_test_flag", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--ns", type=str, default='mixgcf', help="rns, mixgcf")
    parser.add_argument("--K", type=int, default=5, help="number of negative in K-pair loss")

    parser.add_argument("--n_negs", type=int, default=64, help="")
    parser.add_argument("--pool", type=str, default='mean', help="[concat, mean, sum, final]")

    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60]',
                        help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument("--context_hops", type=int, default=3, help="hop")

    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument(
        "--out_dir", type=str, default="./weights/yelp2018/", help="output directory for model"
    )
    parser.add_argument('--user_sim', nargs='?', default=10,
                        help='user共同购买item数量')
    parser.add_argument('--tua0', type=float, default=1.0,
                        help='infoNCE loss')
    parser.add_argument('--tua1', type=float, default=0.5,
                        help='infoNCE loss')
    parser.add_argument('--tua2', type=float, default=0.2,
                        help='infoNCE loss')
    parser.add_argument('--tua3', type=float, default=0.1,
                        help='infoNCE loss')
    parser.add_argument('--lamda', type=float, default=0.2,
                        help='two loss')
    parser.add_argument('--topK', type=int, default=10,
                        help='topK')
    parser.add_argument('--TopK', nargs='?', default=[1, 5, 10, 20, 40, 80],
                        help='topK')

    return parser.parse_args()
