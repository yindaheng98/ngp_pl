from argparse import ArgumentParser
from datasets import dataset_dict


def add_arguements(parser: ArgumentParser):
    parser.add_argument("--dataset", required=True, choices=list(dataset_dict.keys()))
    parser.add_argument("--root_dir", required=True, type=str)
    parser.add_argument("--downsample", default=1.0, type=float)
    return parser


def parse_args(parser):
    args = parser.parse_args()
    dataset = dataset_dict[args.dataset]
    root_dir = args.root_dir
    train_set = dataset(root_dir, split='train')
    test_set = dataset(root_dir, split='test')
    return train_set, test_set
