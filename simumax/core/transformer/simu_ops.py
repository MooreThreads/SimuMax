from typing import List
from simumax.core.tensor import TensorSize
from simumax.core.base_struct import InputOutputInfo
#region ------------------ Single OP ------------------
def split(tensor_size: TensorSize, split_size_or_sections:List[int], dim = -1):
    if isinstance(split_size_or_sections, int):
        assert tensor_size[dim] % split_size_or_sections == 0, f"tensor_size[dim]={tensor_size[dim]} split_size_or_sections={split_size_or_sections}"
        print(f"split_size_or_sections is int, tensor_size[dim] is {tensor_size[dim]}, split_size_or_sections is {split_size_or_sections}")
        split_size_or_sections = [tensor_size[dim] // split_size_or_sections] * split_size_or_sections  

    assert tensor_size[dim] == sum(split_size_or_sections), f"tensor_size[dim]={tensor_size[dim]} sum(split_size_or_sections)={sum(split_size_or_sections)}"
    return [tensor_size.new_with_dim(dim, size) for size in split_size_or_sections]


def transpose(tensor_size: TensorSize, dim0: int, dim1: int):
    assert dim0 != dim1
    return TensorSize(dim1, tensor_size[dim0]), TensorSize(dim0, tensor_size[dim1])

def cat(tensor_sizes: List[TensorSize], dim:int = -1):
    if len(tensor_sizes) == 0:
        return tensor_sizes
    concat_size = sum([t[dim] for t in tensor_sizes])
    return tensor_sizes[0].new_with_dim(dim, concat_size)

def unsqueeze(tensor_size: TensorSize, dim:int):
    return tensor_size.unsqeeze(dim)

def squeeze(tensor_size: TensorSize, dim:int = None):
    return tensor_size.squeeze(dim) 

def apply_rotary_pos_emb():
    ...

def all_gather(input_info:InputOutputInfo, world_size):
    if isinstance(input_info, InputOutputInfo):
        assert len(input_info.tensors) == 1
        input_tensor = input_info.tensors[0]
    else:
        input_tensor = input_info
    gather_dim_size =  input_tensor.shape[-1] * world_size
    output_tensor = TensorSize((input_tensor.shape[:-1] + (gather_dim_size,)), dtype=input_tensor.dtype)
    # return all_gather_bwd(input_tensor, output_tensor, world_size, rank)
    return output_tensor

#endregion