import argparse
import datasets.args as datasets
import models.args as models
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser = models.add_arguements(parser)
parser = datasets.add_arguements(parser)
model = models.parse_args(parser)
train_set = datasets.parse_args(parser, split='train')
train_loader = DataLoader(
    train_set,
    num_workers=16,
    persistent_workers=True,
    batch_size=None,
    pin_memory=True)

print(model)
print(len(train_loader))
