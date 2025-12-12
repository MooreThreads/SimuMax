from typing import List, Tuple, Dict
from copy import deepcopy
from dataclasses import dataclass
import numpy as np

BPE = dict(
    bf16 = 2,
    fp16 = 2,
    fp32 = 4,
    fp8 = 1,
    int32 = 4,
    int64 = 8
)
class TensorSize:
    _id_counter = 0
    """record the shape of the tensor"""
    def __init__(self, shape: Tuple[int, ...], dtype: str = "bf16", grad_fn=None):
        # self.shape = shape
        self.shape = [int(i) for i in shape]
        self.dtype = dtype
        self.id = TensorSize._id_counter
        TensorSize._id_counter += 1
        self._prev = set()
        if grad_fn is not None and hasattr(grad_fn, 'inputs'):
            for i in grad_fn.inputs:
                self._prev.add(i)

    @property
    def ndim(self):
        return len(self.shape)
    
    @property
    def tensors(self):
        return [self]
    
    def size(self, index: int=None) -> int:
        """
        Get the size at the specified index in the tuple.
        """
        if index == None:
            return self.shape
        if index < 0:
            index = len(self.shape) + index

        if 0 <= index < len(self.shape) or index == -1:
            shape = self.shape[index]
        else:
            raise IndexError(
                f"Index {index} is out of range for size tuple {self.shape}"
            )
        return shape

    def numel(self) -> int:
        if len(self.shape) == 0:
            return 0
        res = 1
        for x in self.shape:
            res *= x
        return res
    
    def element_size(self) -> int:
        return BPE[self.dtype]
    
    @property
    def mem_size(self):
        return self.numel() * self.element_size()
    
    def get_memory_size(self):
        return self.numel() * self.element_size()

    def view(self, *args):
        # inplace the shape
        self.shape = list(args)
        return self

    def __getitem__(self, index: int) -> int:
        return self.shape[index]

    def new_with_dim(self, dim, new_size):
        new_shape = list(deepcopy(self.shape))
        new_shape[dim] = new_size
        return TensorSize(new_shape)
    
    def new(self):
        shape = deepcopy(self.shape)
        return TensorSize(shape)
    
    def unsqeeze(self, dim):
        # inplace
        self.shape.insert(dim, 1)
        return self
    
    @property
    def T(self):
        # inplace
        new_shape = deepcopy(self.shape[::-1])
        return TensorSize(shape=new_shape)
    
    def squeeze(self, dim): 
        # inplace
        self.shape = list(self.shape)
        dim_size = self.shape.pop(dim)
        if dim_size != 1:
            raise ValueError("squeeze dim size must be 1")
        return self
    
    def expand(self, *expand_sizes):
        # inplace
        assert len(expand_sizes) == len(self.shape)
        for i, size in enumerate(expand_sizes):
            if size != -1:
                self.shape[i] = size
        return self

    def transpose(self, dim0, dim1):
        # inplace
        new_shape = list(self.shape)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        return TensorSize(new_shape, dtype=self.dtype)
    
    def is_contiguous(self):
        # TODO(sherry): implement this
        return True
    
    def contiguous(self):  
        # TODO(sherry): implement this  
        return self
    
    def __add__(self, other):
        if isinstance(other, TensorSize):
            return TensorSize(deepcopy(self.shape))
        else:
            raise TypeError()

    def __str__(self):
        # return f"TensorSize(shape={self.shape}, dtype={self.dtype}, mem_size={self.mem_size/1024/1024:.4f} MB)"
        return f"TensorSize(shape={self.shape}, dtype={self.dtype})"

FakeTensor = TensorSize
class Float8Tensor(TensorSize):
    def __init__(self, shape):
        super().__init__(shape)
        self.dtype = "fp8"