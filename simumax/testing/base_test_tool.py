"""Some basic functions for tests."""

from typing import Callable, Union
import types


def compute_relative_error(result, golden, eps=1e-9):
    abs_error = abs(golden - result)
    relative_error = abs_error / (golden + eps)
    return relative_error


class Comparator:
    """
    The base class for the comparator
    """
    def __init__(self, rtol=1e-2, atol=1e-8):
        self._comparator: Union[Callable, types.LambdaType] = None
        self._atol = atol
        self._rtol = rtol

    def __call__(self, result: Union[int, float], golden: Union[int, float]):
        if self._comparator is None:
            raise NotImplementedError("Comparator is not implemented by a subclass")

        if not isinstance(self._comparator, (Callable, types.LambdaType)):
            raise TypeError("self._comparator must be a callable type")

        return self._comparator(result, golden)  # pylint: disable=not-callable

    def get_tolerance(self):
        return self._atol, self._rtol


class RelDiffComparator(Comparator):
    """The relative difference comparator."""

    def __init__(self, rtol=1e-2):
        """
        Use relative tolerance to compare the result and golden.
        """
        super().__init__(rtol=rtol)
        self._comparator = (
            lambda result, golden: compute_relative_error(result, golden) < self._rtol
        )


class ResultCheck(object):
    """
    The class to check the prediction result with the golden result.
    """
    def __init__(self, rtol=1e-1, atol=1e-8, comparator=None):
        self._num_comparator = (
            comparator if comparator is not None else RelDiffComparator(rtol=rtol)
        )
        self._rtol = rtol
        self._atol = atol

    def _compare_impl(self, result: dict, golden: dict):
        assert set(result.keys()) == set(
            golden.keys()
        ), f"result keys: {result.keys()} vs golden keys: {golden.keys()}"
        is_pass = True
        for k, _ in result.items():
            cur_r, cur_g = result[k], golden[k]
            if isinstance(cur_r, dict):
                is_pass = self._compare_impl(cur_r, cur_g) and is_pass
            elif isinstance(cur_r, str):
                is_pass = (cur_r == cur_g) and is_pass
            elif isinstance(cur_r, (int, float)):
                is_pass = self._num_comparator(cur_r, cur_g) and is_pass
            else:
                raise ValueError(f"Unsupported type {type(cur_r)} for {k}")
        return is_pass

    def __call__(self, result: dict, golden: dict):
        # compare_log_list = []
        is_pass = self._compare_impl(result, golden)
        return is_pass

    def get_tolerance(self):
        return self._rtol, self._atol
