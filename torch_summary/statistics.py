import torch
import torch.nn as nn


class ModelStat(object):
    def __init__(self, model, input_size, query_granularity=1):
        assert isinstance(model, nn.Module)
        assert isinstance(input_size, (tuple, list)) and len(input_size) == 3

    def _analyze_model(self):
        pass


def stat(model, input_size, query_granularity=1):
    pass

