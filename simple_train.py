import argparse
import datasets.args as datasets
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser = datasets.add_arguements(parser)
test_set = datasets.parse_args(parser, split='test')
test_loader = DataLoader(
    test_set,
    num_workers=8,
    batch_size=None,
    pin_memory=True)
print(len(test_loader))
