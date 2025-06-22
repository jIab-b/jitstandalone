import os
import psutil

def print_memory_usage():
    # Get system-wide memory usage
    virtual_mem = psutil.virtual_memory()
    total_gb = virtual_mem.total / (1024**3)
    available_gb = virtual_mem.available / (1024**3)
    used_gb = virtual_mem.used / (1024**3)
    percent_used = virtual_mem.percent

    # Get memory usage of the current python process
    process = psutil.Process(os.getpid())
    process_mem_mb = process.memory_info().rss / (1024**2)  # RSS in MB

    print("--- System Memory ---")
    print(f"Total: {total_gb:.2f} GB")
    print(f"Available: {available_gb:.2f} GB")
    print(f"Used: {used_gb:.2f} GB ({percent_used}%)")
    print("--- Python Process Memory ---")
    print(f"Current Process Usage: {process_mem_mb:.2f} MB")
    print("-----------------------")


