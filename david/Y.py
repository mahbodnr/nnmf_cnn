import torch
from typing import Callable


class Y(torch.nn.Module):
    """
    A PyTorch module that splits the processing path of a input tensor
    and processes it through multiple torch.nn.Sequential segments,
    then combines the outputs using a specified methods.

    This module allows for creating split paths within a `torch.nn.Sequential`
    model, making it possible to implement architectures with skip connections
    or parallel paths without abandoning the sequential model structure.

    Attributes:
        segments (torch.nn.Sequential[torch.nn.Sequential]): A list of sequential modules to
            process the input tensor.
        combine_func (Callable | None): A function to combine the outputs
            from the segments.
        dim (int | None): The dimension along which to concatenate
            the outputs if `combine_func` is `torch.cat`.

    Args:
        segments (torch.nn.Sequential[torch.nn.Sequential]): A torch.nn.Sequential
            with a list of sequential modules to process the input tensor.
        combine (str, optional): The method to combine the outputs.
            "cat" for concatenation (default), or "func" to use a
            custom combine function.
        dim (int | None, optional): The dimension along which to
            concatenate the outputs if `combine` is "cat".
            Defaults to 1.
        combine_func (Callable | None, optional): A custom function
            to combine the outputs if `combine` is "func".
            Defaults to None.

    Example:
        A simple example for the `Y` module with two sub-torch.nn.Sequential:

                                 ----- segment_a -----
            main_Sequential ----|                     |---- main_Sequential
                                 ----- segment_b -----

        segments = [segment_a, segment_b]
        y_split = Y(segments)
        result = y_split(input_tensor)

    Methods:
        forward(input: torch.Tensor) -> torch.Tensor:
            Processes the input tensor through the segments and
            combines the results.
    """

    segments: torch.nn.Sequential
    combine_func: Callable
    dim: int | None

    def __init__(
        self,
        segments: torch.nn.Sequential,
        combine: str = "cat",  # "cat", "func"
        dim: int | None = 1,
        combine_func: Callable | None = None,
    ):
        super().__init__()
        self.segments = segments
        self.dim = dim

        if combine.upper() == "CAT":
            self.combine_func = torch.cat
        else:
            assert combine_func is not None
            self.combine_func = combine_func

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        results: list[torch.Tensor] = []
        for segment in self.segments:
            results.append(segment(input))

        if self.dim is None:
            return self.combine_func(results)
        else:
            return self.combine_func(results, dim=self.dim)
