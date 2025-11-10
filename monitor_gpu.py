import torch
import time
import psutil
import GPUtil
from datetime import datetime

def monitor_gpu():
    print("\nGPU Monitoring Started...")
    print("Press Ctrl+C to stop monitoring")
    print("\nTimestamp | GPU Util % | GPU Mem % | CPU % | RAM %")
    print("-" * 50)
    
    try:
        while True:
            # Get GPU stats
            gpu = GPUtil.getGPUs()[0]  # Get the first GPU
            gpu_util = gpu.load * 100
            gpu_mem = gpu.memoryUtil * 100
            
            # Get CPU and RAM stats
            cpu_percent = psutil.cpu_percent()
            ram_percent = psutil.virtual_memory().percent
            
            # Get current timestamp
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # Print stats
            print(f"{timestamp} | {gpu_util:6.1f}% | {gpu_mem:6.1f}% | {cpu_percent:4.1f}% | {ram_percent:4.1f}%")
            
            # Wait before next update
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    monitor_gpu()