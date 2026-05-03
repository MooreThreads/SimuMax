<p align="center">
  <a href="model.md">English</a>|
  <a href="model-zh.md">中文版本</a>
</p>

# Model Config

SimuMax models are defined with JSON files under [configs/models](../configs/models). The model config describes the static architecture that cost, memory, and simulator analysis build on top of.

SimuMax works with three input files together:

- **system**: machine capability and efficiency data
- **strategy**: parallelism and runtime policy
- **model**: architecture description

See also:

- [system.md](./system.md)
- [strategy.md](./strategy.md)

## Fastest way to start

Do not start from an empty file unless you have to.

Recommended path:

1. Copy the nearest existing JSON from [configs/models](../configs/models).
2. Keep the original file around as a reference.
3. Change only the structural fields that are different.
4. Run `perf` with a known-good `strategy` and `system` first.

Good starting points:

- dense model: [configs/models/llama3-8b.json](../configs/models/llama3-8b.json)
- MoE + MLA model: [configs/models/deepseekv2.json](../configs/models/deepseekv2.json)

## Minimal viable dense model config

```json
{
    "model_type": "dense",
    "model_name": "my_dense_model",
    "hidden_size": 4096,
    "head_num": 32,
    "kv_head_num": 8,
    "head_size": 128,
    "intermediate_size": 14336,
    "layer_num": 32,
    "vocab_size": 128256,
    "use_swiglu": true
}
```

For a first dense model adaptation, the usual shortest path is:

1. copy `llama3-8b.json`
2. update `model_name`
3. update `layer_num`, `hidden_size`, `head_num`, `kv_head_num`, `intermediate_size`, and `vocab_size`
4. update `attention_type` only if the target model is not standard MHA

## Which fields must match the real model

At minimum, keep these aligned with the target Megatron or real model:

- `layer_num`
- `hidden_size`
- `head_num`
- `kv_head_num`
- `intermediate_size`
- `vocab_size`
- `attention_type`
- `use_swiglu`

If these are wrong, both timing and memory can drift even when the strategy and system are correct.

## Core Fields

Common fields:

- `model_name`: display/debug name
- `layer_num`: number of transformer layers
- `hidden_size`
- `head_num`
- `kv_head_num`
- `intermediate_size`
- `vocab_size`
- `use_swiglu`
- `attention_type`

These fields drive the main dense attention/MLP math and the embedding / LM head shapes.

## MoE And MLA Fields

MoE-related fields:

- `expert_num`
- `topk`
- `moe_ffn_hidden_size`
- `moe_shared_expert_intermediate_size`
- `dense_layers`
- `capacity`
- `moe_pad_expert_input_to_capacity`
- `group_linear_mode`

Important note:

- `dense_layers` is the number of dense transformer layers that appear before the MoE layers in the current model layout. It matters for stage-level memory and timing, especially with pipeline parallelism.

MLA-related fields:

- `attention_type="mla"`
- `qk_head_dim`
- `qk_pos_emb_head_dim`
- `v_head_dim`
- `q_lora_rank`
- `kv_lora_rank`

MoE / MLA checklist:

- MoE users should double-check:
  - `expert_num`
  - `topk`
  - `moe_ffn_hidden_size`
  - `moe_shared_expert_intermediate_size`
  - `dense_layers`
- MLA users should double-check:
  - `attention_type="mla"`
  - `qk_head_dim`
  - `qk_pos_emb_head_dim`
  - `v_head_dim`
  - `q_lora_rank`
  - `kv_lora_rank`

## Vocabulary Padding

Megatron-style vocabulary padding is modeled with:

- `make_vocab_size_divisible_by`
- `padded_vocab_size`
- `orig_vocab_size`

This is relevant when aligning perf or simulator output with Megatron real runs.

## Relationship To Megatron

The model config should describe the same structural shape assumptions as the target Megatron run:

- number of layers
- dense vs MoE layout
- MLA vs MHA
- expert count and top-k
- LoRA ranks for MLA
- vocabulary size and padding behavior

If a real run and a model config disagree on these fields, both timing and memory can drift even when the strategy and system are correct.

## Typical Workflow

1. Start from an existing JSON in [configs/models](../configs/models).
2. Copy the nearest architecture.
3. Update only the structural fields that change.
4. Pair it with a strategy and system config.
5. Run `perf` first, then use `simulate()` if you need trace or memory lifecycle evidence.

## Example

```python
from simumax.core.config import ModelConfig

model = ModelConfig.init_from_config_file("configs/models/llama3-8b.json")
print(model.layer_num, model.hidden_size, model.model_type)
```
