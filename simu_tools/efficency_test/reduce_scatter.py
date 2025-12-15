import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import profile, record_function, ProfilerActivity

def run_reduce_scatter(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 每个rank准备 world_size 份输入，每份大小为 240MB / world_size
    total_bytes = 240 * 1024 * 1024
    chunk_bytes = total_bytes
    chunk_elements = chunk_bytes // 2  # float32 每个 4 字节

    input_list = [torch.randn(chunk_elements, dtype=torch.bfloat16).cuda(rank) for _ in range(world_size)]
    output_tensor = torch.zeros(chunk_elements, dtype=torch.bfloat16).cuda(rank)

    # Warmup
    for _ in range(10):
        dist.reduce_scatter(output_tensor, input_list, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # 正式测试 + profiling
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=5, active=50),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./log_rank{rank}"),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(50):
            with record_function("reduce_scatter_comm"):
                dist.reduce_scatter(output_tensor, input_list, op=dist.ReduceOp.SUM)
            prof.step()
    torch.cuda.synchronize()
    dist.destroy_process_group()

def main():
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())

    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("需要至少 2 张 GPU 才能进行 reduce-scatter 测试")
    mp.spawn(run_reduce_scatter, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
