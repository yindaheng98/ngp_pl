from argparse import ArgumentParser
from datasets import dataset_dict


def add_arguements(parser: ArgumentParser):
    parser.add_argument("--dataset", required=True, choices=list(dataset_dict.keys()))
    parser.add_argument("--root_dir", required=True, type=str)
    parser.add_argument("--downsample", default=1.0, type=float)
    parser.add_argument('--ray_sampling_strategy', type=str, default='all_images',
                        choices=['all_images', 'same_image'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        for '--dataset train' only
                        ''')
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='''
                        anumber of rays in a batch
                        for '--dataset train' only
                        ''')
    return parser


def parse_args(parser, split='train'):
    args = parser.parse_args()
    dataset = dataset_dict[args.dataset]
    root_dir = args.root_dir
    dataset = dataset(root_dir, split=split)
    if split == 'train':
        dataset.ray_sampling_strategy = args.ray_sampling_strategy
        dataset.batch_size = args.batch_size
    return dataset


def dict_args(parser):
    args = parser.parse_args()
    return dict(
        dataset=args.dataset,
        root_dir=args.root_dir,
        downsample=args.downsample,
        ray_sampling_strategy=args.ray_sampling_strategy,
        batch_size=args.batch_size,
    )
