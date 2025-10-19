import os, json, psutil, time, threading
from datetime import datetime

def metric_benchmark(run_fn, label, output_dir="../../utils/results"):
    os.makedirs(output_dir, exist_ok=True)
    p = psutil.Process(os.getpid())

    cpu_samples, freq_samples, stop = [], [], threading.Event()


    # backend thread sampling
    def sampler():
        while not stop.is_set():
            cpu_samples.append(psutil.cpu_percent(interval=0.5))
            f = psutil.cpu_freq(percpu=False)
            freq_samples.append(f.current)


    start = time.time()
    start_cpu = p.cpu_times()
    t = threading.Thread(target=sampler, daemon=True)
    t.start()

    run_fn()

    stop.set(); t.join()
    end = time.time()
    end_cpu = p.cpu_times()


    wall_time = end - start
    cpu_time = (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)
    cpu_ratio = cpu_time / wall_time if wall_time > 0 else 0.0

    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    max_cpu = max(cpu_samples) if cpu_samples else 0.0
    avg_freq = sum(freq_samples) / len(freq_samples) if freq_samples else 0.0
    max_freq = max(freq_samples) if freq_samples else 0.0
    min_freq = min(freq_samples) if freq_samples else 0.0

    result = {
        "label": label,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "wall_time_sec": round(wall_time, 3),
        "cpu_time_sec": round(cpu_time, 3),
        "cpu_ratio": round(cpu_ratio, 3),
        "avg_cpu_percent": round(avg_cpu, 2),
        "max_cpu_percent": round(max_cpu, 2),
        "avg_cpu_freq_MHz": round(avg_freq, 2),
        "min_cpu_freq_MHz": round(min_freq, 2),
        "max_cpu_freq_MHz": round(max_freq, 2),
        "samples_count": len(cpu_samples)
    }

    filename = os.path.join(output_dir, f"{label.replace(' ', '_')}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("Metrics saved.")
