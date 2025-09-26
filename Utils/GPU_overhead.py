# this code needs to be run in parallel with the main training script
# it will log GPU metrics (power, memory, utilization, PCIe throughput, etc.)

import pynvml
import time

def log_gpu_metrics(output_file="gpu_metrics_log.csv", duration=10, interval=1.0):
    """
    Logs GPU metrics (power, memory, utilization, PCIe throughput, etc.)
    for all GPUs into a CSV file.

    Parameters:
        output_file (str): Path to save the log.
        duration (float): Total logging time in seconds.
        interval (float): Sampling interval in seconds.
    """
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    with open(output_file, "w") as f:
        # CSV header
        header = [
            "timestamp", "gpu_index", "name",
            "power_W", "power_limit_W",
            "gpu_util_%", "mem_util_%",
            "mem_used_GiB", "mem_total_GiB",
            "temp_C", "fan_%", "clock_sm_MHz", "clock_mem_MHz",
            "pcie_rx_MBps", "pcie_tx_MBps"
        ]
        f.write(",".join(header) + "\n")

        start = time.time()
        while (time.time() - start) < duration:
            timestamp = time.time()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")

                # Power
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
                power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0

                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                mem_util = util.memory

                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = mem_info.used / (1024**3)
                mem_total = mem_info.total / (1024**3)

                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                # Fan
                try:
                    fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                except:
                    fan_speed = -1  # Some GPUs don't support fans

                # Clocks
                clock_sm = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                clock_mem = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)

                # PCIe throughput
                pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1024.0
                pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1024.0

                # Write row
                row = [
                    f"{timestamp:.3f}", str(i), name,
                    f"{power:.2f}", f"{power_limit:.2f}",
                    str(gpu_util), str(mem_util),
                    f"{mem_used:.3f}", f"{mem_total:.3f}",
                    str(temp), str(fan_speed), str(clock_sm), str(clock_mem),
                    f"{pcie_rx:.2f}", f"{pcie_tx:.2f}"
                ]
                f.write(",".join(row) + "\n")

            time.sleep(interval)

    pynvml.nvmlShutdown()
    print(f"Saved GPU metrics log to {output_file}")


# Example usage
if __name__ == "__main__":
    log_gpu_metrics(output_file="metrics_gpu_fl_name.csv", duration=3600, interval=1)
