end=8G
echo "run all_reduce_perf"
./build/all_reduce_perf -n 10 -b 1M -e $end -f 2 -g 8 -w 2 -d bfloat16 > perf_all_reduce.txt
echo "run all_gather_perf"
./build/all_gather_perf -n 10 -b 1M -e $end -f 2 -g 8 -w 2 -d bfloat16 > perf_all_gather.txt
echo "run reduce_scatter_perf"
./build/reduce_scatter_perf -n 10 -b 1M -e $end -f 2 -g 8 -w 2 -d bfloat16 > perf_reduce_scatter.txt
echo "run alltoall_perf"
./build/alltoall_perf -n 10 -b 1M -e $end -f 2 -g 8 -w 2 -d bfloat16 > perf_alltoall_perf.txt
echo "run sendrecv_perf"
./build/sendrecv_perf -n 1 -b 1M -e $end -f 2 -g 1 -t 1  -d bfloat16 > send_recv.txt