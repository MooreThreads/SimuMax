from typing import List
from simumax.core.tensor import TensorSize
from simumax.core.base_struct import MetaModule, InputOutputInfo, PathDebugContext
# from simumax.core.transformer.dense_module import Qunatizer

class ConcatModule(MetaModule):
    def __init__(self, dim:int = -1, enable_recompute:bool = False, strategy=None, system=None):
        super().__init__(strategy, system)
        self.dim = dim
        self.enable_recompute = enable_recompute
        self.is_leaf_module = True
        
    @property
    def output_info(self):
        # return TensorSize or InputOutputInfo
        tensor_sizes = self.input_info.tensors
        if len(tensor_sizes) == 0:
            return InputOutputInfo([])
        concat_size = sum([t[self.dim] for t in tensor_sizes])
        return tensor_sizes[0].new(self.dim, concat_size)
    
    def extra_repr(self) -> str:
        repr_info = f"concat_dim={self.dim}, enable_recompute={self.enable_recompute}"
        return repr_info
    
    # TODO(sherry): concat的统计函数。。。

class Function:
    @staticmethod
    def apply(cls, *args, **kwargs):
        raise NotImplementedError
    
class ConcatFunction(Function):
    @staticmethod
    def apply(model:MetaModule, enable_recompute:bool, tensor_sizes: List[TensorSize], dim:int = -1, path_debug_context: PathDebugContext = None):
        # model.output_size = TensorSize.concat(tensor_sizes, dim)
        concat_module = ConcatModule(dim, enable_recompute, model.strategy, model.system)
        concat_module.parent_module = model  # Bind parent module 

        input_info = InputOutputInfo(tensor_sizes)
        out = concat_module(input_info, path_debug_context = path_debug_context) # Reuse the __call__ method of MetaModule, call related functions for statistics, and register the concat_module into parent_module

        return out

