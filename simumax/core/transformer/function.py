from typing import List
from simumax.core.tensor import TensorSize
from simumax.core.base_struct import MetaModule, InputOutputInfo, PathDebugContext
# from simumax.core.transformer.dense_module import Qunatizer

class Function:
    @staticmethod
    def apply(cls, *args, **kwargs):
        raise NotImplementedError
class ConcatModule(MetaModule):
    def __init__(self, dim:int = -1, enable_recompute:bool = False, strategy=None, system=None, name=None):
        super().__init__(strategy, system)
        self.dim = dim
        self.enable_recompute = enable_recompute
        self.is_leaf_module = True
        self.name = name if name else 'ConcatModule'

    def create_output_info(self):
        # return TensorSize or InputOutputInfo
        tensor_sizes = self.input_info.tensors
        if len(tensor_sizes) == 0:
            return InputOutputInfo([])
        concat_size = sum([t[self.dim] for t in tensor_sizes])
        output_info =  tensor_sizes[0].new_with_dim(self.dim, concat_size)
        return output_info
    
    def extra_repr(self) -> str:
        repr_info = f"concat_dim={self.dim}, enable_recompute={self.enable_recompute}"
        return repr_info

class SplitModule(MetaModule):
    def __init__(self, split_size_or_sections:List[int], split_dim, enable_recompute:bool = False, strategy=None, system=None, name=None):
        super().__init__(strategy, system)
        self.split_size_or_sections = split_size_or_sections
        self.split_dim = split_dim
        self.enable_recompute = enable_recompute
        self.name = name if name else 'SplitModule'

    def create_output_info(self):
        tensor_size = self.input_info.tensors[0] if isinstance(self.input_info, InputOutputInfo) else self.input_info
        split_dim = self.split_dim
        split_size_or_sections = self.split_size_or_sections
        if isinstance(split_size_or_sections, int):
            assert tensor_size[split_dim] % split_size_or_sections == 0, f"tensor_size[dim]={tensor_size[split_dim]} split_size_or_sections={split_size_or_sections}"
            print(f"split_size_or_sections is int, tensor_size[dim] is {tensor_size[split_dim]}, split_size_or_sections is {split_size_or_sections}")
            split_size_or_sections = [tensor_size[split_dim] // split_size_or_sections] * split_size_or_sections  

        assert tensor_size[split_dim] == sum(split_size_or_sections), f"tensor_size[dim]={tensor_size[split_dim]} sum(split_size_or_sections)={sum(split_size_or_sections)}, tensor_size={tensor_size.shape}, split_dim={split_dim}"
        output_info =  InputOutputInfo(tensors=[tensor_size.new_with_dim(split_dim, size) for size in split_size_or_sections])
        return output_info
    
    def extra_repr(self) -> str:
        repr_info = f"split_dim={self.split_dim}, enable_recompute={self.enable_recompute}"
        return repr_info

class AddModule(MetaModule):
    def __init__(self, enable_recompute:bool = False, strategy=None, system=None, name=None):
        super().__init__(strategy, system)
        self.enable_recompute = enable_recompute
        self.name = name

    def create_output_info(self):
        # recover the original input
        assert self.output_info_ is None
        output_info = InputOutputInfo(tensors=[self.input_info.tensors[0].new()])
        return output_info

    def extra_repr(self) -> str:
        repr_info = f"enable_recompute={self.enable_recompute}"
        return repr_info

class UnsqueezeModule(MetaModule):
    def __init__(self, unsqueeze_dim:int, enable_recompute:bool = False, strategy=None, system=None, name=None):
        super().__init__(strategy, system)
        self.unsqueeze_dim = unsqueeze_dim
        self.enable_recompute = enable_recompute
        self.name =  name if name else 'UnsqueezeModule'
    
    def create_output_info(self):
        inputs = self.input_info.tensors[0] if isinstance(self.input_info, InputOutputInfo) else self.input_info
        outputs = inputs.new()
        outputs.squeeze(self.unsqueeze_dim)
        return InputOutputInfo(tensors=outputs)

class ConcatFunction(Function):
    @staticmethod
    def apply(parent_model:MetaModule, enable_recompute:bool, tensor_sizes: List[TensorSize], dim:int = -1, path_debug_context: PathDebugContext = None, name = None):
        # model.output_size = TensorSize.concat(tensor_sizes, dim)
        concat_module = ConcatModule(dim, enable_recompute, parent_model.strategy, parent_model.system, name)
        concat_module.parent_module = parent_model  # Bind parent module 

        input_info = InputOutputInfo(tensor_sizes)
        out = concat_module(input_info, path_debug_context = path_debug_context) # Reuse the __call__ method of MetaModule, call related functions for statistics, and register the concat_module into parent_module
        return out

class SplitFunction(Function):
    @staticmethod
    def apply(parent_model:MetaModule, enable_recompute:bool, tensor_size:TensorSize, split_size_or_sections:int, split_dim:int = -1, path_debug_context: PathDebugContext = None, name = None):
        # model.output_size = TensorSize.split(tensor_size, split_size_or_sections, dim)    
        split_module = SplitModule(split_size_or_sections, split_dim,  enable_recompute, parent_model.strategy, parent_model.system, name)
        split_module.parent_module = parent_model  # Bind parent module 

        input_info = InputOutputInfo([tensor_size]) if isinstance(tensor_size, TensorSize) else tensor_size
        out = split_module(input_info, path_debug_context = path_debug_context) # Reuse the __call__ method of MetaModule, call related functions for statistics, and register the split_module into parent_module

        return out
    
class AddFunction(Function):
    @staticmethod
    def apply(parent_model:MetaModule, enable_recompute:bool, tensor_size1:TensorSize, tensor_size2:TensorSize, path_debug_context: PathDebugContext = None, name = None):
        # model.output_size = TensorSize.split(tensor_size, split_size_or_sections, dim)    
        add_module = AddModule(enable_recompute, parent_model.strategy, parent_model.system, name)
        add_module.parent_module = parent_model  # Bind parent module 

        if isinstance(tensor_size1, InputOutputInfo):
            tensor_size1 = tensor_size1.tensors[0]
        if isinstance(tensor_size2, InputOutputInfo):
            tensor_size2 = tensor_size2.tensors[0]
        input_info = InputOutputInfo([tensor_size1, tensor_size2])
        out = add_module(input_info, path_debug_context = path_debug_context) # Reuse the __call__ method of MetaModule, call related functions for statistics, and register the split_module into parent_module
        return out