import psutil

cpu_times = psutil.cpu_times()

cpu_percent = psutil.cpu_percent(interval=1)

cpu_percent_percpu = psutil.cpu_percent(interval=1, percpu=True)

print(cpu_times)

