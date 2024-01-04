import argparse
import datasets.args as datasets
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser = datasets.add_arguements(parser)
train_set = datasets.parse_args(parser, split='train')
train_loader = DataLoader(
    train_set,
    num_workers=16,
    persistent_workers=True,
    batch_size=None,
    pin_memory=True)
print(len(train_loader))
