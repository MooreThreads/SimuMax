from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
import ast
import json
import math
import graphviz
from graphviz import Digraph
from simumax.core.tensor import FakeTensor
from simumax.core.config import set_capture_graph_only

BPE = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int32" : 4}

@dataclass
class ValueInfo:
    """描述张量的信息（名称、形状、数据类型）"""
    name: str
    shape: List[int]  # 使用-1表示动态维度
    dtype: str = "float32"

@dataclass
class Node:
    """计算图中的节点（操作）"""
    name: str           # 节点名称
    op: callable
    op_type: str        # 操作类型（如 'Add', 'MatMul', 'Conv'）
    inputs: List[str]   # 输入张量名称列表
    outputs: List[str]  # 输出张量名称列表
    enable_recompute: bool = False  # 是否启用重计算
    attributes: Dict[str, Any] = None  # 操作属性
    cache_inputs: bool = False  # 是否缓存输入
    domain: str = ""    # 操作域（如 '' 表示AI, 'com.microsoft' 等）
    visited: bool = False

@dataclass
class Graph:
    """完整的计算图"""
    name: str = "model_graph"
    nodes: List[Node] = None           # 所有操作节点
    inputs: List[ValueInfo] = None     # 图输入
    outputs: List[ValueInfo] = None    # 图输出
    initializers: Dict[str, Any] = None  # 常量参数（权重、偏置等）
    value_info: Dict[str, ValueInfo] = None  # 中间张量信息
    name_counts = dict()  # 用于生成唯一名称的计数器
    forward_edges: Dict[FakeTensor, List[Node]] = field(default_factory=dict) # 前向边
    def __post_init__(self):
        if self.nodes is None:
            self.nodes = []
        if self.inputs is None:
            self.inputs = []
        if self.outputs is None:
            self.outputs = []
        if self.initializers is None:
            self.initializers = {}
        if self.value_info is None:
            self.value_info = {}
    
    def reset_noede(self):
        for node in self.nodes:
            node.visited = False

    def get_next_nodes_by_node(self, node: Node):
        nodes = []
        for output in node.outputs:
            if output in self.forward_edges:
                nodes.extend(self.forward_edges[output])
        return nodes
    
    def is_recompute_varaince_node(self, node: Node, next_nodes: List[Node]):
        if not node.enable_recompute:
            return False
        for o in next_nodes:
            if o.enable_recompute:
                return False
        return True

    def traverse_forward_from_tensor(self, tensor: FakeTensor, pre_node: Node = None, set_variance_node: bool = False):
        if isinstance(tensor, FakeTensor):
            input_name = tensor.onnx_name
        else:
            input_name = tensor
        if input_name not in self.forward_edges:
            return
        for node in self.forward_edges[input_name]:
            if node.visited:
                continue
            node.visited = True
            if set_variance_node and pre_node and pre_node.enable_recompute and \
                self.is_recompute_varaince_node(node, self.get_next_nodes_by_node(node)):
                node.op.set_variance_node(True)
            
            for output in node.outputs:
                self.traverse_forward_from_tensor(output, node, set_variance_node)

    def to_dict(self):
        """将图转换为字典格式，便于序列化"""
        return {
            "name": self.name,
            "nodes": [{
                "name": n.name,
                "op_type": n.op_type,
                "enable_recompute": n.enable_recompute,
                "is_variance_node": n.op.is_variance_node,
                "inputs": n.inputs,
                "outputs": n.outputs,
                "call_idx": n.op.call_idx,
                "attributes": n.attributes or {},
                "domain": n.domain
            } for n in self.nodes],
            "inputs": [{
                "name": i.name,
                "shape": i.shape,
                "dtype": i.dtype
            } for i in self.inputs],
            "outputs": [{
                "name": o.name,
                "shape": o.shape,
                "dtype": o.dtype
            } for o in self.outputs],
            "initializers": list(self.initializers.keys()),
            "value_info": {k: {"shape": v.shape, "dtype": v.dtype} 
                          for k, v in self.value_info.items()}
        }

    def export_json(self, filepath: str):
        """导出为JSON文件"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class SimuONNXGraphBuilder:
    """单例类，用于构建和存储计算图"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.graph = Graph()
            cls._instance.tensor_counter = 0
            cls._instance.node_counter = 0
        return cls._instance
    
    def capture(self, perf_model):
        print("Capture graph...")
        set_capture_graph_only(True)
        perf_model.run_estimate()
        set_capture_graph_only(False)
        print("Capture graph done.")
        return self.graph
    
    def reset(self):
        """重置图构建器"""
        self.graph = Graph()
        self.tensor_counter = 0
        self.node_counter = 0
    
    def get_unique_tensor_name(self, prefix: str = "tensor") -> str:
        """生成唯一的张量名称"""
        name = f"{prefix}_{self.tensor_counter}"
        self.tensor_counter += 1
        return name
    
    def get_unique_node_name(self, op_type: str) -> str:
        """生成唯一的节点名称"""
        name = f"{op_type}_{self.node_counter}"
        self.node_counter += 1
        # name_count = Graph.name_counts.get(op_type, 0)
        # Graph.name_counts[op_type] = name_count + 1
        # name = f"{op_type}_{name_count}"
        return name
    
    def add_node(self, op, op_type: str, inputs: List['FakeTensor'], 
                 outputs: List['FakeTensor'], attributes: Dict[str, Any] = None):
        """添加一个操作节点到图中"""
        node_name = self.get_unique_node_name(op.name)
        # print(f'Adding node: {op_type}, inputs={inputs}, outputs={outputs}')
        # 确保所有输入输出张量都有名称
        input_names = []
        for inp in inputs:
            if not hasattr(inp, 'onnx_name'):
                inp.onnx_name = self.get_unique_tensor_name("input")
            input_names.append(inp.onnx_name)
            
        output_names = []
        for out in outputs:
            if not hasattr(out, 'onnx_name'):
                out.onnx_name = self.get_unique_tensor_name("output")
            output_names.append(out.onnx_name)
        
        # 记录张量形状信息（简化版，实际需要更复杂的形状推断）
        for out in outputs:
            if hasattr(out, 'shape'):
                self.graph.value_info[out.onnx_name] = ValueInfo(
                    name=out.onnx_name, 
                    shape=out.shape if hasattr(out, 'shape') else [-1],
                    dtype=str(out.dtype) if hasattr(out, 'dtype') else "float32"
                )
        
        # 创建并添加节点
        node = Node(
            name=node_name,
            op = op,
            op_type=op_type,
            inputs=input_names,
            outputs=output_names,
            enable_recompute=op.enable_recompute,
            attributes=attributes or {},
            domain=""
        )
        self.graph.nodes.append(node)
        for inp in inputs:
            name = inp.onnx_name
            if name in self.graph.forward_edges:
                self.graph.forward_edges[name].append(node)
            else:
                self.graph.forward_edges[name] = [node]
        
        return node

    @staticmethod
    def export_json(filename):
        SimuONNXGraphBuilder._instance.graph.export_json(filename)

def export_onnx_style_graph(model, input_tensor: FakeTensor, 
                           output_path: str = "model_graph.json"):
    """
    导出类似ONNX的计算图
    
    Args:
        model: 要导出的模型
        input_tensor: 输入张量（用于推断形状）
        output_path: 输出JSON文件路径
    """
    # 重置图构建器
    graph_builder = SimuONNXGraphBuilder()
    graph_builder.reset()
    
    # 设置输入张量的ONNX名称和形状信息
    input_tensor.onnx_name = "input"
    if hasattr(input_tensor.data, 'shape'):
        graph_builder.graph.inputs.append(
            ValueInfo(name=input_tensor.onnx_name, 
                     shape=list(input_tensor.data.shape),
                     dtype=str(input_tensor.data.dtype))
        )
    
    # 执行前向传播（这会自动构建计算图）
    output_tensor = model(input_tensor)
    
    # 设置输出张量的信息
    output_tensor.onnx_name = "output"
    if hasattr(output_tensor.data, 'shape'):
        graph_builder.graph.outputs.append(
            ValueInfo(name=output_tensor.onnx_name,
                     shape=list(output_tensor.data.shape),
                     dtype=str(output_tensor.data.dtype))
        )
    
    # 添加模型参数到初始izer
    for name, param in model._parameters.items():
        if not hasattr(param, 'onnx_name'):
            param.onnx_name = name
        graph_builder.graph.initializers[param.onnx_name] = param.data
    
    # 导出为JSON
    graph_builder.graph.export_json(output_path)
    print(f"计算图已导出到: {output_path}")
    
    return graph_builder.graph

def visualize_with_graphviz(json_path, output_path="computational_graph"):
    def get_mem(size):
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.0f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / 1024 / 1024:.0f} MB"
        else:
            return f"{size / 1024 / 1024 / 1024:.0f} GB"
    """使用Graphviz进行可视化"""
    graph_data = json.load(open(json_path, 'r'))
    # 创建有向图
    dot = Digraph(comment=graph_data['name'])
    dot.attr(rankdir='TB')  # 从上到下布局
    dot.attr('node', shape='rectangle', style='filled')
    
    # 添加输入节点
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(color='blue', label='Inputs')
        for inp in graph_data['inputs']:
            c.node(inp['name'], f"Input: {inp['name']}\nShape: {inp['shape']}", 
                   fillcolor='lightgreen')
    
    # 添加输出节点
    with dot.subgraph(name='cluster_outputs') as c:
        c.attr(color='red', label='Outputs')
        for out in graph_data['outputs']:
            c.node(out['name'], f"Output: {out['name']}\nShape: {out['shape']}", 
                   fillcolor='lightcoral')
    
    # 添加参数节点
    with dot.subgraph(name='cluster_params') as c:
        c.attr(color='orange', label='Parameters')
        for param in graph_data['initializers']:
            c.node(param, f"Param: {param}", fillcolor='gold')
    
    # 添加操作节点
    for node in graph_data['nodes']:
        attrs_str = '\n'.join([f"{k}: {v}" for k, v in node['attributes'].items()])
        label = f"{node['op_type']}\n{node['name']}\ncall_idx:{node['call_idx']}"
        if attrs_str:
            label += f"\n---\n{attrs_str}"
        color = 'yellow' if node['enable_recompute'] else 'lightblue'
        color = 'green' if node['enable_recompute'] else 'lightblue'
        # color = 'yellow' if node['is_variance_node'] else color
        if node['is_variance_node']:
            style = 'filled,dashed'
            color = '#ffd700'
        else:
            style = 'filled,solid'
        dot.node(node['name'], label, fillcolor=color, style=style)
    
    # 添加中间张量节点
    for tensor_name, info in graph_data['value_info'].items():
        # dot.node(tensor_name, f"Tensor: {tensor_name}\nShape: {info['shape']}", 
        #         fillcolor='lightgray', shape='ellipse')
        # shape = ast.literal_eval(info['shape'])
        shape = info['shape']
        dtype = info['dtype']
        mem = get_mem(math.prod(shape)*BPE[dtype])
        dot.node(tensor_name, f"Tensor: {tensor_name}\nShape: {shape}, {dtype}\nMem:{mem}", 
                fillcolor='lightgray', shape='ellipse')
    
    # 添加边
    for node in graph_data['nodes']:
        for input_tensor in node['inputs']:
            dot.edge(input_tensor, node['name'])
        for output_tensor in node['outputs']:
            dot.edge(node['name'], output_tensor)
    
    # 保存和渲染
    dot.render(output_path, format='png', cleanup=True)
    print(f"Graphviz图已保存到: {output_path}.png")
    
    return dot

