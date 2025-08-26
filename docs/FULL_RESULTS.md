- [Benchmarks](#benchmarks)
  - [A100-Pcie](#a100-pcie)
# Benchmarks
Performance of some models on a single node. Llama3-70B was trimmed to 12 layers and DeepSeek-236B was trimmed to 4 layers.


## A100-Pcie

![alt text](../assets/A100-Pcie.png)


|    model   | mbc | parallelism | memory | Tflops |
|:----------:|:---:|:-----------:|--------|--------|
| llama3 70b |  4  |    tp1pp2   | -0.80% | -1.46% |
|  llama3 70b          |  4   |     tp2     | -0.32% | -3.09% |
|  llama3 70b          |  4   |     tp4     | -0.39% | -0.59% |
|   llama3 70b         |  4   |     tp8     | -0.50% | -1.86% |
|   llama3 70b         |  8  |    tp1pp2   | -2.85% | -0.10% |
|   llama3 70b         |  8   |     tp2     | -1.65% | -2.08% |
|    llama3 70b        |  8   |     tp4     | -1.44% | -1.28% |
|    llama3 70b        |  8   |     tp8     | -1.25% | -1.73% |
|   llama3 70b         |  32 |    tp1pp2   | -2.85% |  2.12% |
|   llama3 70b         |   32  |     tp2     | -1.64% |  0.03% |
|   llama3 70b         |  32   |     tp4     | -1.43% | -1.71% |
|    llama3 70b        |  32   |     tp8     | -1.26% | -1.47% |

|  model  | mbc | parallelism | memory | Tflops |
|:-------:|:---:|:-----------:|--------|--------|
| ds 236b |  4  |     ep8     | -1.45% | -3.62% |
| ds 236b        | 4    |    ep4pp2   | -4.82% | -0.66% |
| ds 236b        |  8  |     ep8     |  0.22% | -2.64% |
|  ds 236b       |  8   |    ep4pp2   | -4.70% | -1.47% |
|  ds 236b       |  32 |     ep8     | -1.45% | -1.82% |
|  ds 236b       |   32  |    ep4pp2   | -4.82% | -0.82% |

|   model   | mbc | parallelism | memory | tflops |
|:---------:|:---:|:-----------:|:------:|:------:|
| llama3 8b |  4  |    tp1pp2   | -1.10% |  0.04% |
|   llama3 8b        |4     |     tp2     | -0.63% | -2.35% |
|   llama3 8b        |  4   |     tp4     | -0.63% |  1.96% |
|    llama3 8b       | 4    |     tp8     | -0.63% | -0.50% |
|    llama3 8b       |  8  |    tp1pp2   | -1.10% |  1.53% |
|   llama3 8b        | 8    |     tp2     | -0.63% | -0.61% |
|   llama3 8b        | 8    |     tp4     | -0.63% |  1.35% |
|   llama3 8b        |  8   |     tp8     | -0.63% | -0.56% |
|   llama3 8b        |  8   |     tp8     | -0.50% | -1.86% |
|   llama3 8b        |  32 |    tp1pp2   | -1.10% |  3.76% |
|   llama3 8b        |  32   |     tp2     | -0.63% | -0.31% |
|   llama3 8b        |  32   |     tp4     | -0.63% |  2.08% |
|   llama3 8b        |  32   |     tp8     | -0.63% | -0.75% |


