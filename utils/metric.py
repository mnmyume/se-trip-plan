import os, json, psutil, time, threading
from datetime import datetime

def metric_benchmark(run_fn, set_type, node_mode, output_dir="../../outputs/metrics"):
    os.makedirs(output_dir, exist_ok=True)
    p = psutil.Process(os.getpid())

    cpu_samples, freq_samples, mem_samples = [], [], []
    stop = threading.Event()

    # backend thread sampling
    def sampler():
        while not stop.is_set():
            # cpu usage
            cpu_samples.append(p.cpu_percent(interval=10))
            # # cpu frequency (no permission yet)
            # f = psutil.cpu_freq(percpu=False)
            # if f:
            #     freq_samples.append(f.current)
            # memory (MB)
            mem_info = p.memory_info().rss / 1024**2  # RSS
            mem_samples.append(mem_info)

    # time start
    start = time.time()
    start_cpu = p.cpu_times()
    t = threading.Thread(target=sampler, daemon=True)
    t.start()

    # run
    run_fn()

    # time end
    stop.set()
    t.join()
    end = time.time()
    end_cpu = p.cpu_times()

    # calculate
    wall_time = end - start
    cpu_time = (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)
    cpu_ratio = cpu_time / wall_time if wall_time > 0 else 0.0

    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    max_cpu = max(cpu_samples) if cpu_samples else 0.0

    avg_freq = sum(freq_samples) / len(freq_samples) if freq_samples else 0.0
    max_freq = max(freq_samples) if freq_samples else 0.0
    min_freq = min(freq_samples) if freq_samples else 0.0

    avg_mem = sum(mem_samples) / len(mem_samples) if mem_samples else 0.0
    max_mem = max(mem_samples) if mem_samples else 0.0
    min_mem = min(mem_samples) if mem_samples else 0.0


    result = {
        "node_mode": node_mode,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "wall_time_sec": round(wall_time, 3),
        "cpu_time_sec": round(cpu_time, 3),
        "cpu_ratio": round(cpu_ratio, 3),
        "avg_cpu_percent": round(avg_cpu, 2),
        "max_cpu_percent": round(max_cpu, 2),
        "avg_cpu_freq_MHz": round(avg_freq, 2),
        "min_cpu_freq_MHz": round(min_freq, 2),
        "max_cpu_freq_MHz": round(max_freq, 2),
        "avg_memory_MB": round(avg_mem, 2),
        "min_memory_MB": round(min_mem, 2),
        "max_memory_MB": round(max_mem, 2),
        "samples_count": len(cpu_samples)
    }

    filename = os.path.join(output_dir, set_type, node_mode, "metrics_results.json")

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(result)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Metrics appended to: {os.path.abspath(filename)}")
    return result
