from abc import ABC, abstractmethod


class Constraint(ABC):
    @abstractmethod
    def delta(self):
        return 0.05

    @abstractmethod
    def constraint(self, *args, **kwargs):
        pass