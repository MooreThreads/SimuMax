from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
import numpy as np

DEVICE_BW = 900 # H100机内标称带宽
'''
https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf
    While the 4-
    GPU configuration includes point-to-point NVLink connections between GPUs and provides a 
    higher CPU-to-GPU ratio in the server, the 8-GPU configuration includes NVSwitch to provide 
    SHARP in-network reductions and full NVLink bandwidth of 900 GB/s between any pair of 
    GPUs. The H100 SXM5 GPU is also used in the powerful new DGX H100 servers and DGX 
    SuperPOD systems
'''

def get_bws(data, scale=1, ranks=8, title=""):
    x = []
    data = data.strip().split('\n')  # 移除头尾空行
    for d in data:
        t = d.split()
        n, cost, bw1, bw2 = t[0], t[5], t[6], t[7]
        x.append([n, cost, bw1, bw2])
    y = np.array(x, dtype=float)

    # X: 通信量（B）
    n = y[:, 0] * (ranks-1) / ranks * scale
    cost = y[:, 1]

    # 拟合
    a, b = np.polyfit(n[:11], cost[:11], 1)
    show = False
    if show:
        # -------- 第1张图：性能数据 + 拟合 --------
        plt.figure(figsize=(8, 4))
        plt.plot(n, cost, marker='o', label='Data')
        plt.plot(n, n*a + b, '--', label='Linear Fit')
        plt.xscale('log', base=2)
        plt.xlabel("Bytes")
        plt.ylabel("Time (us)")
        plt.title(f"Cost vs Bytes- {title}")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)

        # -------- 第2张图：相对误差 --------
        plt.figure(figsize=(8, 4))
        rel_error = np.abs(n*a + b - cost) / cost
        plt.plot(n, rel_error, marker='x', color='red')
        plt.xscale('log', base=2)
        plt.xlabel("Bytes")
        plt.ylabel("Relative Error")
        plt.title(f"Relative Fit Error- {title}")
        plt.grid(True, which='both', linestyle='--', alpha=0.5)

    # -------- 返回估算带宽 --------
    bw = 1 / a / 1024**3 * 1000**2
    eff = round(bw / DEVICE_BW , 4) 
    
    latency_per_p2p = b / ((ranks-1) * scale)
    print(f"{title} Estimated Bandwidth: {bw:.2f} GB/s, Efficiency: {eff:.4f}, Latency per P2P: {latency_per_p2p:.2f} μs")
    return bw, eff, latency_per_p2p  # 带宽 (GB/s), 起始开销（μs）

get_bws(data='''     1048576        524288  bfloat16     sum      -1    46.24   22.68   39.69      0    45.89   22.85   39.99      0
     2097152       1048576  bfloat16     sum      -1    45.71   45.88   80.29      0    45.82   45.77   80.10      0
     4194304       2097152  bfloat16     sum      -1    57.93   72.41  126.71      0    58.22   72.04  126.06      0
     8388608       4194304  bfloat16     sum      -1    86.55   96.92  169.61      0    85.32   98.32  172.05      0
    16777216       8388608  bfloat16     sum      -1    125.9  133.24  233.17      0    126.7  132.46  231.81      0
    33554432      16777216  bfloat16     sum      -1    203.3  165.09  288.90      0    203.3  165.03  288.80      0
    67108864      33554432  bfloat16     sum      -1    328.4  204.37  357.65      0    327.8  204.74  358.30      0
   134217728      67108864  bfloat16     sum      -1    579.2  231.72  405.51      0    579.9  231.45  405.04      0
   268435456     134217728  bfloat16     sum      -1   1090.5  246.16  430.79      0   1089.8  246.31  431.04      0
   536870912     268435456  bfloat16     sum      -1   2106.2  254.90  446.07      0   2107.3  254.76  445.84      0
  1073741824     536870912  bfloat16     sum      -1   4022.4  266.94  467.15      0   4017.6  267.26  467.70      0
  2147483648    1073741824  bfloat16     sum      -1   7924.8  270.98  474.22      0   7921.2  271.10  474.43      0
  4294967296    2147483648  bfloat16     sum      -1    15736  272.95  477.66      0    15755  272.62  477.08      0
  8589934592    4294967296  bfloat16     sum      -1    31450  273.13  477.98      0    31436  273.26  478.20      0''', scale = 2, ranks = 8, title = 'all_reduce_nccl_test')
get_bws(data='''     1048576         65536  bfloat16    none      -1    47.14   22.24   19.46      0    45.43   23.08   20.20      0
     2097152        131072  bfloat16    none      -1    45.69   45.90   40.16      0    45.76   45.83   40.10      0
     4194304        262144  bfloat16    none      -1    47.23   88.80   77.70      0    50.69   82.74   72.40      0
     8388608        524288  bfloat16    none      -1    48.87  171.67  150.21      0    49.08  170.90  149.54      0
    16777216       1048576  bfloat16    none      -1    73.03  229.73  201.02      0    70.13  239.21  209.31      0
    33554432       2097152  bfloat16    none      -1    128.2  261.71  228.99      0    125.2  267.99  234.50      0
    67108864       4194304  bfloat16    none      -1    211.3  317.58  277.88      0    208.6  321.64  281.43      0
   134217728       8388608  bfloat16    none      -1    386.4  347.34  303.92      0    384.5  349.08  305.45      0
   268435456      16777216  bfloat16    none      -1    730.2  367.64  321.68      0    728.5  368.49  322.43      0
   536870912      33554432  bfloat16    none      -1   1394.3  385.04  336.91      0   1387.1  387.04  338.66      0
  1073741824      67108864  bfloat16    none      -1   2734.1  392.72  343.63      0   2707.8  396.54  346.97      0
  2147483648     134217728  bfloat16    none      -1   5379.0  399.23  349.33      0   5323.3  403.42  352.99      0
  4294967296     268435456  bfloat16    none      -1    10615  404.62  354.05      0    10460  410.60  359.28      0
  8589934592     536870912  bfloat16    none      -1    21025  408.57  357.49      0    20741  414.15  362.38      0''', scale = 1, ranks = 8, title = 'all_gather_nccl_test')
get_bws(data='''1048576         65536  bfloat16     sum      -1    43.62   24.04   21.03      0    43.24   24.25   21.22      0
     2097152        131072  bfloat16     sum      -1    44.87   46.74   40.90      0    44.02   47.64   41.69      0
     4194304        262144  bfloat16     sum      -1    59.15   70.91   62.05      0    46.64   89.92   78.68      0
     8388608        524288  bfloat16     sum      -1    54.41  154.17  134.90      0    53.33  157.30  137.64      0
    16777216       1048576  bfloat16     sum      -1    75.44  222.41  194.60      0    192.4   87.19   76.29      0
    33554432       2097152  bfloat16     sum      -1    117.8  284.86  249.25      0    116.5  288.03  252.02      0
    67108864       4194304  bfloat16     sum      -1    205.3  326.96  286.09      0    202.1  332.07  290.56      0
   134217728       8388608  bfloat16     sum      -1    385.5  348.20  304.68      0    380.8  352.51  308.44      0
   268435456      16777216  bfloat16     sum      -1    726.0  369.73  323.52      0    724.0  370.75  324.41      0
   536870912      33554432  bfloat16     sum      -1   1410.0  380.76  333.16      0   1404.3  382.29  334.51      0
  1073741824      67108864  bfloat16     sum      -1   2758.3  389.27  340.61      0   2750.0  390.46  341.65      0
  2147483648     134217728  bfloat16     sum      -1   5419.6  396.25  346.72      0   5403.0  397.46  347.78      0
  4294967296     268435456  bfloat16     sum      -1    10584  405.82  355.09      0    10599  405.22  354.56      0
  8589934592     536870912  bfloat16     sum      -1    20940  410.21  358.94      0    20909  410.83  359.47      0''', scale = 1, ranks = 8, title = 'reduce_scatter_nccl_test')
get_bws(data='''     1048576         65536  bfloat16    none      -1    75.76   13.84   12.11      0    74.98   13.99   12.24    N/A
     2097152        131072  bfloat16    none      -1    75.06   27.94   24.45      0    74.05   28.32   24.78    N/A
     4194304        262144  bfloat16    none      -1    76.64   54.73   47.89      0    76.01   55.18   48.28    N/A
     8388608        524288  bfloat16    none      -1    79.10  106.05   92.80      0    79.59  105.40   92.23    N/A
    16777216       1048576  bfloat16    none      -1    82.85  202.51  177.19      0    82.91  202.36  177.07    N/A
    33554432       2097152  bfloat16    none      -1    129.9  258.33  226.04      0    128.7  260.75  228.15    N/A
    67108864       4194304  bfloat16    none      -1    220.1  304.96  266.84      0    214.5  312.84  273.74    N/A
   134217728       8388608  bfloat16    none      -1    398.7  336.60  294.52      0    398.5  336.84  294.74    N/A
   268435456      16777216  bfloat16    none      -1    761.2  352.66  308.58      0    754.9  355.61  311.16    N/A
   536870912      33554432  bfloat16    none      -1   1461.0  367.46  321.52      0   1461.0  367.46  321.53    N/A
  1073741824      67108864  bfloat16    none      -1   2822.3  380.44  332.89      0   2821.4  380.57  333.00    N/A
  2147483648     134217728  bfloat16    none      -1   5520.5  389.00  340.38      0   5515.5  389.36  340.69    N/A
  4294967296     268435456  bfloat16    none      -1    10937  392.69  343.60      0    10998  390.52  341.71    N/A
  8589934592     536870912  bfloat16    none      -1    21730  395.31  345.89      0    21834  393.43  344.25    N/A''', scale = 1, ranks = 8, title = 'alltoall_nccl_test')
get_bws(data='''     1048576        524288  bfloat16     sum      -1    15.49   67.71   67.71      0     1.62  646.87  646.87    N/A
     2097152       1048576  bfloat16     sum      -1    15.01  139.70  139.70      0     1.51  1387.01  1387.01    N/A
     4194304       2097152  bfloat16     sum      -1    15.69  267.32  267.32      0     1.48  2832.08  2832.08    N/A
     8388608       4194304  bfloat16     sum      -1    16.79  499.62  499.62      0     1.43  5886.74  5886.74    N/A
    16777216       8388608  bfloat16     sum      -1    23.09  726.70  726.70      0     1.44  11691.44  11691.44    N/A
    33554432      16777216  bfloat16     sum      -1    37.97  883.69  883.69      0     1.43  23448.24  23448.24    N/A
    67108864      33554432  bfloat16     sum      -1    61.50  1091.22  1091.22      0     1.70  39522.30  39522.30    N/A
   134217728      67108864  bfloat16     sum      -1    110.2  1217.81  1217.81      0     1.44  93466.38  93466.38    N/A
   268435456     134217728  bfloat16     sum      -1    203.9  1316.39  1316.39      0     1.39  193816.21  193816.21    N/A
   536870912     268435456  bfloat16     sum      -1    389.7  1377.78  1377.78      0     1.43  374386.97  374386.97    N/A
  1073741824     536870912  bfloat16     sum      -1    763.8  1405.86  1405.86      0     1.40  769156.03  769156.03    N/A
  2147483648    1073741824  bfloat16     sum      -1   1508.7  1423.37  1423.37      0     1.46  1472896.88  1472896.88    N/A
  4294967296    2147483648  bfloat16     sum      -1   2999.7  1431.81  1431.81      0     1.63  2639807.80  2639807.80    N/A
  8589934592    4294967296  bfloat16     sum      -1   5968.9  1439.11  1439.11      0     2.50  3435973.84  3435973.84    N/A''', scale = 1, ranks = 2, title = 'sendrecv_nccl_test')