Run experiments:
```bash
~/pho-coordinator$ nohup python3 ./main.py -w "/tmp/pho" -b "/home/ubuntu/multibenches" -p "/home/ubuntu/gcc_dyn_list/plugin.so" -e 100 1> stdout.log 2> stderr.log &
```

Collect experiments results:
```bash
~/pho-coordinator$ nohup python3 ./collect_experiments_results.py -w "/home/ubuntu/pho-coordinator-working-dir" -b "/home/ubuntu/multibenches" -p "/home/ubuntu/gcc_dyn_list/plugin.so" -e 100 1> stdout.log 2> stderr.log &
```

Evaluate optimization cache:
```bash
~/pho-coordinator$ nohup python3 ./evaluate_optimization_cache.py -w "/home/ubuntu/pho-coordinator-opt-cache-dir" -b "/home/ubuntu/debug-bench" -p "/home/ubuntu/gcc_dyn_list/plugin.so" 1> stdout.log 2> stderr.log &
```