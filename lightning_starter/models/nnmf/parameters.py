import torch

class NonNegativeParameter(torch.nn.Parameter):
    """
    A parameter that is constrained to be non-negative.
    """
    def __new__(cls, data, requires_grad=True):
        if torch.any(data < 0):
            raise ValueError("Negative values are not allowed in the parameter data.")
        return super(NonNegativeParameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def _check_negative_values(self, data):
        if torch.any(data < 0):
            raise ValueError("Negative values are not allowed in the parameter data.")

    def __setattr__(self, name, value):
        if name == 'data':
            self._check_negative_values(value)
            super(NonNegativeParameter, self).__setattr__(name, value)
        else:
            super(NonNegativeParameter, self).__setattr__(name, value)

    def __setitem__(self, key, value):
        if type(value) is torch.Tensor:
            self._check_negative_values(value)
        else:
            self._check_negative_values(torch.tensor(value))
        super(NonNegativeParameter, self).__setitem__(key, value)