import matplotlib.pyplot as plt
import matplotlib.patches as patches

def duration_dualpp(mbn, pp, F_cost, B_cost, W_cost, FandB_cost, opt_time, stage):
    bubble = (pp-2-stage)*FandB_cost - (pp/2-stage-1)*F_cost - (pp*3/2-3)*W_cost + stage*B_cost
    duration = mbn * (F_cost + B_cost) * 2 - (2*mbn-3/2*pp + stage + 1) * (F_cost + B_cost - FandB_cost) + bubble + opt_time
    return duration


def mfu_dualpp(mbn, pp, F_cost, B_cost, W_cost, FandB_cost, opt_time, stage, Flops_perbatch, default_flops=465):
    """
    opt_time: the cost of gradient reduce in 1F1B + weight update
    mbn: micro batch number in up/down stream
    Flops_perbatch: Flops each rank each batch 
    """
    opt_time = opt_time*2  #both opt in down and up need to be considered, 
                           #gradient size in each rank is 2 times of the original, weight update cost is the same 

    dur = duration_dualpp(mbn, pp, F_cost, B_cost, W_cost, FandB_cost, opt_time, stage)
    flops = Flops_perbatch * mbn * 2
    print(f'dur ={dur/1000}')
    mfu = flops / (dur / 1000) / default_flops / 1e12
    return mfu


def cal_FandB(attn_F, MLP_F, attn_B, attn_W, MLP_B, MLP_W, Dispatch, Combine):
    t_compute = {"attn_F":[], "MLP_F":[], "attn_B":[], "attn_W":[],"MLP_B":[], "MLP_W":[]}
    t_comm = {"Dispatch_F":[], "Combine_F":[], "Dispatch_B":[], "Combine_B":[]}
    
    t_compute['attn_F'] = [0,attn_F]
    
    t_comm['Dispatch_F'] = [t_compute['attn_F'][1], t_compute['attn_F'][1]+Dispatch]



    t_compute['MLP_B'] = [attn_F, attn_F+MLP_B]
    t_sync = max(t_compute['MLP_B'][1], t_comm['Dispatch_F'][1])
    t_comm['Dispatch_B'] = [t_sync, t_sync+Dispatch]

    t_compute['MLP_W'] = [t_compute['MLP_B'][1], t_compute['MLP_B'][1]+MLP_W]
    t_compute['MLP_F'] = [t_compute['MLP_W'][1], t_compute['MLP_W'][1]+MLP_F]

    t_sync =   max( t_compute['MLP_F'][1], t_comm['Dispatch_B'][1])
    t_comm['Combine_F'] = [t_sync, t_sync+Combine]
    t_compute['attn_B'] = [t_sync, t_sync+attn_B]
    
    t_sync = max( t_compute['attn_B'][1], t_comm['Combine_F'][1])
    t_comm['Combine_B'] = [t_sync, t_sync+Combine]
    t_compute['attn_W'] = [t_compute['attn_B'][1], t_compute['attn_B'][1]+attn_W]
    compute_dur = t_compute['attn_W'][1] - t_compute['MLP_B'][0]
    comm_dur = t_comm['Combine_B'][1] - t_comm['Dispatch_F'] [0]
    return compute_dur, comm_dur, t_compute, t_comm

def show_overlap_all2all(t_compute,t_comm,save_path):
    data1,data2 = t_compute,t_comm

    fig, ax = plt.subplots(figsize=(4*len(t_compute['attn_F']), 1))
    color_map = {
    'F': 'yellow',
    'W': 'lightblue',
    'B': 'green'
    }
    # 绘制data1
    for i, (key, intervals) in enumerate(data1.items()):
        for j in range(0, len(intervals), 2):
            start, end = intervals[j], intervals[j+1]
            facecolor = color_map.get(key.split('_')[-1], 'none')
            rect = patches.Rectangle((start, 1), end - start, 1, edgecolor='blue', facecolor=facecolor, linewidth=2)
            ax.add_patch(rect)
            ax.text((start + end) / 2, 1.2, key, color='blue', ha='center', va='center')

    # 绘制data2
    for i, (key, intervals) in enumerate(data2.items()):
        for j in range(0, len(intervals), 2):
            start, end = intervals[j], intervals[j+1]
            facecolor = color_map.get(key.split('_')[-1], 'none')
            rect = patches.Rectangle((start, 0), end - start, 1, edgecolor='red', facecolor=facecolor, linestyle='--', linewidth=2)
            ax.add_patch(rect)
            ax.text((start + end) / 2, 0.2, key, color='red', ha='center', va='center')

    ax.set_yticks([-0.5, 3])
    ax.set_xlabel('Time')
    ax.set_title('Intervals for Multiple Groups')
    ax.set_xlim( -1, max(data1['attn_W'][-1], data2['Combine_B'][-1]+1))
    plt.grid(True)
    plt.show()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)



def cal_FandB_multi_layers(attn_F, MLP_F, attn_B, attn_W, MLP_B, MLP_W, Dispatch, Combine, layers=1, if_show=True, save_path='./tmp/pp_overlap.jpg'):
    t_compute = {"attn_F":[], "MLP_F":[], "attn_B":[], "attn_W":[],"MLP_B":[], "MLP_W":[]}
    t_comm = {"Dispatch_F":[], "Combine_F":[], "Dispatch_B":[], "Combine_B":[]}
    
    t_compute['attn_F'] = [0,attn_F]
    t_comm['Dispatch_F'] = [t_compute['attn_F'][1], t_compute['attn_F'][1]+Dispatch]

    for i in range(layers):
        t_compute['MLP_B'] += [t_compute['attn_F'][-1], t_compute['attn_F'][-1]+MLP_B]
        t_sync = max(t_compute['MLP_B'][-1], t_comm['Dispatch_F'][-1])
        t_comm['Dispatch_B'] += [t_sync, t_sync+Dispatch]

        t_compute['MLP_W'] += [t_compute['MLP_B'][-1], t_compute['MLP_B'][-1]+MLP_W]
        t_compute['MLP_F'] += [t_compute['MLP_W'][-1], t_compute['MLP_W'][-1]+MLP_F]

        t_sync =   max( t_compute['MLP_F'][-1], t_comm['Dispatch_B'][-1])
        t_comm['Combine_F'] += [t_sync, t_sync+Combine]
        t_compute['attn_B'] += [t_sync, t_sync+attn_B]
        
        t_sync = max( t_compute['attn_B'][-1], t_comm['Combine_F'][-1])
        t_comm['Combine_B'] += [t_sync, t_sync+Combine]
        t_compute['attn_W'] += [t_compute['attn_B'][-1], t_compute['attn_B'][-1]+attn_W]
        
        if i<layers-1:
            t_compute['attn_F'] += [t_compute['attn_W'][-1],t_compute['attn_W'][-1]+attn_F]
            t_sync =   max( t_compute['attn_F'][-1], t_comm['Combine_B'][-1])
            t_comm['Dispatch_F'] += [t_sync, t_sync+Dispatch]

    compute_dur = t_compute['attn_W'][-1] - t_compute['attn_F'][0]
    comm_dur = t_comm['Combine_B'][-1] - t_comm['Dispatch_F'] [0]
    if if_show:
        show_overlap_all2all(t_compute, t_comm, save_path=save_path)
    return compute_dur, comm_dur, t_compute, t_comm

def cal_cost(attn_F, MLP_F, attn_B, attn_W, MLP_B, MLP_W, Dispatch, Combine, layers, save_path='./tmp/pp_overlap.jpg', FB_overlap=True):
    """
    *_B is the time cost of bwd for output 
    *_W is the time cost of bwd for weight
    
    return
    B_cost denotes the execution time of a full backward chunk
    W_cost denotes the execution time of a "backward for weights" chunk
    """
    F_cost = attn_F + Dispatch + MLP_F + Combine
    F_cost *= layers
    W_cost = (MLP_W + attn_W) 
    W_cost *= layers
    B_cost = Combine + MLP_B + Dispatch + attn_B
    B_cost *= layers 
    B_cost += W_cost

    # compute_dur, comm_dur, t_compute, t_comm = cal_FandB(attn_F, MLP_F, attn_B, attn_W, MLP_B, MLP_W, Dispatch, Combine)
    # FandB_cost = max(comm_dur, compute_dur)
    # FandB_cost *= layers
    if FB_overlap:
        compute_dur, comm_dur, t_compute, t_comm=cal_FandB_multi_layers(attn_F, MLP_F, attn_B, attn_W, MLP_B, MLP_W, Dispatch, Combine, layers=layers, if_show=True, save_path=save_path)
        FandB_cost = max(comm_dur, compute_dur)
    else:
        FandB_cost = F_cost+B_cost
    return F_cost, W_cost, B_cost, FandB_cost


def perf_dualpp(mbn, pp, attn_F, MLP_F, attn_B, attn_W, MLP_B, MLP_W, Dispatch, Combine,layers, opt_time, stage, Flops_perbatch, default_flops=465, save_path='./tmp/pp_overlap.jpg', FB_overlap=True):
    """
    opt_time: the cost of gradient reduce in 1F1B + weight update
    mbn: micro batch number in up/down stream
    Flops_perbatch
    """
    F_cost, W_cost, B_cost, FandB_cost=cal_cost(attn_F, MLP_F, attn_B, attn_W, MLP_B, MLP_W, Dispatch, Combine, layers, save_path=save_path, FB_overlap=FB_overlap)
    mfu = mfu_dualpp(mbn, pp, F_cost, B_cost, W_cost, FandB_cost, opt_time, stage, Flops_perbatch, default_flops=default_flops)
    return mfu


