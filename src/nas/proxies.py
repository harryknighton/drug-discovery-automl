from abc import ABC
from typing import List

from torch.nn import Module
from torch_geometric.data import Dataset


class Proxy(ABC):
    higher_is_better = False

    def calculate(self, model: Module, dataset: Dataset) -> float:
        pass

    def __call__(self, model: Module, dataset: Dataset) -> float:
        return self.calculate(model, dataset)


class MajorityVote(Proxy):
    def __init__(self, proxies: List[Proxy]) -> None:
        self.proxies = proxies

    def calculate(self, model: Module, dataset: Dataset) -> float:
        results = [proxy(model, dataset) for proxy in self.proxies]
        return sum(results) / len(results)


# ----------------------------------------------
# Data Independent Proxies


class NumParams(Proxy):
    def calculate(self, model: Module, dataset: Dataset) -> float:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SynapticFlow(Proxy):
    def calculate(self, model: Module, dataset: Dataset) -> float:
        pass

# ----------------------------------------------
# Data Dependent Proxies


class JacobianCovariance(Proxy):
    def calculate(self, model: Module, dataset: Dataset) -> float:
        pass


class ZiCo(Proxy):
    def calculate(self, model: Module, dataset: Dataset) -> float:
        pass


class GradientNorm(Proxy):
    def calculate(self, model: Module, dataset: Dataset) -> float:
        pass


class Snip(Proxy):
    def calculate(self, model: Module, dataset: Dataset) -> float:
        pass
