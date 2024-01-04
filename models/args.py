from argparse import ArgumentParser
import torch
from kornia.utils.grid import create_meshgrid3d
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
    model = NGP(scale=args.scale, rgb_act=rgb_act)
    G = model.grid_size
    model.register_buffer("density_grid", torch.zeros(model.cascades, G**3))
    model.register_buffer("grid_coords", create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))
    return model


def dict_args(parser):
    args = parser.parse_args()
    return dict(
        scale=args.scale,
        use_exposure=args.use_exposure
    )
