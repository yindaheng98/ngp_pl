import argparse
import datasets.args as datasets

parser = argparse.ArgumentParser()
parser = datasets.add_arguements(parser)
train_set, test_set = datasets.parse_args(parser)
print(train_set)
print(test_set)
