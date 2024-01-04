from argparse import ArgumentParser
from .networks import NGP


def add_arguements(parser: ArgumentParser):
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--use_exposure', action='store_true', default=False,
                        help='whether to train in HDR-NeRF setting')
    return parser


def parse_args(parser):
    args = parser.parse_args()
    rgb_act = 'None' if args.use_exposure else 'Sigmoid'
    return NGP(scale=args.scale, rgb_act=rgb_act)
