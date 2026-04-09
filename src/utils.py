import time
from collections import defaultdict

class SemanticTimer:
    """A clean context manager to profile specific blocks of code."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Clears the timings for the next game."""
        self.timings = defaultdict(float)
        self.counts = defaultdict(int)

    class _TimerContext:
        def __init__(self, parent, name):
            self.parent = parent
            self.name = name
            self.start = None

        def __enter__(self):
            self.start = time.perf_counter()

        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.perf_counter() - self.start
            self.parent.timings[self.name] += elapsed
            self.parent.counts[self.name] += 1

    def time(self, name: str):
        """Wrap this around a block of code using 'with'."""
        return self._TimerContext(self, name)

    def log_to_tensorboard(self, writer, table_id, step: int):
        """Logs the aggregated timings to TensorBoard."""
        for name, total_time in self.timings.items():
            calls = self.counts[name]
            avg_time_ms = (total_time / calls) * 1000 if calls > 0 else 0
            if isinstance(table_id, int):
            # Groups logs beautifully in TensorBoard by Table and Metric Type
                writer.add_scalar(f"Table_{table_id}_TotalTime_s/{name}", total_time, step)
                writer.add_scalar(f"Table_{table_id}_AvgTime_ms/{name}", avg_time_ms, step)
            else:
                writer.add_scalar(f"{table_id}_TotalTime_s/{name}", total_time, step)
                writer.add_scalar(f"{table_id}_AvgTime_ms/{name}", avg_time_ms, step)
            # Optional: writer.add_scalar(f"Table_{table_id}_Calls/{name}", calls, step)