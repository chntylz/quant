from abc import ABCMeta, abstractmethod

class BaseEventType(object):
    __metaclass = ABCMeta
    beta_path = '../data/dpzz500.csv'

    @abstractmethod
    def get_beta_path(self):
        pass
