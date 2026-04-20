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
    """Describes tensor info (name, shape, dtype)."""
    name: str
    shape: List[int]  # Use -1 for dynamic dimensions
    dtype: str = "float32"

@dataclass
class Node:
    """A node (operation) in the computation graph."""
    name: str           # Node name
    op: callable
    op_type: str        # Operation type (e.g. 'Add', 'MatMul', 'Conv')
    inputs: List[str]   # List of input tensor names
    outputs: List[str]  # List of output tensor names
    enable_recompute: bool = False  # Whether to enable recompute
    attributes: Dict[str, Any] = None  # Operation attributes
    cache_inputs: bool = False  # Whether to cache inputs
    domain: str = ""    # Operation domain (e.g. '' for AI, 'com.microsoft', etc.)
    visited: bool = False

@dataclass
class Graph:
    """Full computation graph."""
    name: str = "model_graph"
    nodes: List[Node] = None           # All operation nodes
    inputs: List[ValueInfo] = None     # Graph inputs
    outputs: List[ValueInfo] = None    # Graph outputs
    initializers: Dict[str, Any] = None  # Constant parameters (weights, biases, etc.)
    value_info: Dict[str, ValueInfo] = None  # Intermediate tensor info
    name_counts = dict()  # Counter used to generate unique names
    forward_edges: Dict[FakeTensor, List[Node]] = field(default_factory=dict) # Forward edges
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
        """Convert the graph to a dict for serialization."""
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
        """Export to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class SimuONNXGraphBuilder:
    """Singleton that builds and stores the computation graph."""
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
        """Reset the graph builder."""
        self.graph = Graph()
        self.tensor_counter = 0
        self.node_counter = 0
    
    def get_unique_tensor_name(self, prefix: str = "tensor") -> str:
        """Generate a unique tensor name."""
        name = f"{prefix}_{self.tensor_counter}"
        self.tensor_counter += 1
        return name
    
    def get_unique_node_name(self, op_type: str) -> str:
        """Generate a unique node name."""
        name = f"{op_type}_{self.node_counter}"
        self.node_counter += 1
        # name_count = Graph.name_counts.get(op_type, 0)
        # Graph.name_counts[op_type] = name_count + 1
        # name = f"{op_type}_{name_count}"
        return name
    
    def add_node(self, op, op_type: str, inputs: List['FakeTensor'], 
                 outputs: List['FakeTensor'], attributes: Dict[str, Any] = None):
        """Add an operation node to the graph."""
        node_name = self.get_unique_node_name(op.name)
        # print(f'Adding node: {op_type}, inputs={inputs}, outputs={outputs}')
        # Ensure all input/output tensors have names
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
        
        # Record tensor shape info (simplified; real shape inference is more complex)
        for out in outputs:
            if hasattr(out, 'shape'):
                self.graph.value_info[out.onnx_name] = ValueInfo(
                    name=out.onnx_name, 
                    shape=out.shape if hasattr(out, 'shape') else [-1],
                    dtype=str(out.dtype) if hasattr(out, 'dtype') else "float32"
                )
        
        # Create and add the node
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
    Export an ONNX-like computation graph.

    Args:
        model: The model to export
        input_tensor: Input tensor (used for shape inference)
        output_path: Output JSON file path
    """
    # Reset the graph builder
    graph_builder = SimuONNXGraphBuilder()
    graph_builder.reset()

    # Set the ONNX name and shape info for the input tensor
    input_tensor.onnx_name = "input"
    if hasattr(input_tensor.data, 'shape'):
        graph_builder.graph.inputs.append(
            ValueInfo(name=input_tensor.onnx_name, 
                     shape=list(input_tensor.data.shape),
                     dtype=str(input_tensor.data.dtype))
        )
    
    # Run the forward pass (this automatically builds the computation graph)
    output_tensor = model(input_tensor)

    # Set output tensor info
    output_tensor.onnx_name = "output"
    if hasattr(output_tensor.data, 'shape'):
        graph_builder.graph.outputs.append(
            ValueInfo(name=output_tensor.onnx_name,
                     shape=list(output_tensor.data.shape),
                     dtype=str(output_tensor.data.dtype))
        )
    
    # Add model parameters to initializers
    for name, param in model._parameters.items():
        if not hasattr(param, 'onnx_name'):
            param.onnx_name = name
        graph_builder.graph.initializers[param.onnx_name] = param.data
    
    # Export as JSON
    graph_builder.graph.export_json(output_path)
    print(f"Computation graph exported to: {output_path}")
    
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
    """Visualize with Graphviz."""
    graph_data = json.load(open(json_path, 'r'))
    # Create a directed graph
    dot = Digraph(comment=graph_data['name'])
    dot.attr(rankdir='TB')  # Top-to-bottom layout
    dot.attr('node', shape='rectangle', style='filled')

    # Add input nodes
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(color='blue', label='Inputs')
        for inp in graph_data['inputs']:
            c.node(inp['name'], f"Input: {inp['name']}\nShape: {inp['shape']}", 
                   fillcolor='lightgreen')
    
    # Add output nodes
    with dot.subgraph(name='cluster_outputs') as c:
        c.attr(color='red', label='Outputs')
        for out in graph_data['outputs']:
            c.node(out['name'], f"Output: {out['name']}\nShape: {out['shape']}", 
                   fillcolor='lightcoral')
    
    # Add parameter nodes
    with dot.subgraph(name='cluster_params') as c:
        c.attr(color='orange', label='Parameters')
        for param in graph_data['initializers']:
            c.node(param, f"Param: {param}", fillcolor='gold')
    
    # Add operation nodes
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
    
    # Add intermediate tensor nodes
    for tensor_name, info in graph_data['value_info'].items():
        # dot.node(tensor_name, f"Tensor: {tensor_name}\nShape: {info['shape']}", 
        #         fillcolor='lightgray', shape='ellipse')
        # shape = ast.literal_eval(info['shape'])
        shape = info['shape']
        dtype = info['dtype']
        mem = get_mem(math.prod(shape)*BPE[dtype])
        dot.node(tensor_name, f"Tensor: {tensor_name}\nShape: {shape}, {dtype}\nMem:{mem}", 
                fillcolor='lightgray', shape='ellipse')
    
    # Add edges
    for node in graph_data['nodes']:
        for input_tensor in node['inputs']:
            dot.edge(input_tensor, node['name'])
        for output_tensor in node['outputs']:
            dot.edge(node['name'], output_tensor)
    
    # Save and render
    dot.render(output_path, format='png', cleanup=True)
    print(f"Graphviz graph saved to: {output_path}.png")
    
    return dot

