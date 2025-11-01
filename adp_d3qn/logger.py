import csv, os, threading

class CsvLogger:
    """线程安全的实验日志记录器"""
    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self._lock = threading.Lock()
        self.reward_file = os.path.join(out_dir, "reward_log.csv")
        self.latency_file = os.path.join(out_dir, "latency_log.csv")
        with open(self.reward_file, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "reward"])
        with open(self.latency_file, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "latency_ms"])

    def write(self, epoch: int, reward: float, latency: float):
        with self._lock:
            with open(self.reward_file, "a", newline="") as f:
                csv.writer(f).writerow([epoch, round(reward, 3)])
            with open(self.latency_file, "a", newline="") as f:
                csv.writer(f).writerow([epoch, round(latency, 3)])
