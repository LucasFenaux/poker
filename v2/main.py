from global_settings import NUM_CPUS, NUM_GPUS

import ray
import torch
import os
from datetime import datetime
import uuid

from casino_manager import CasinoManager


def get_save_folder(base_path="results"):
    """
    Generates a unique folder name based on the current time and a random ID.
    Example: results/run_20231027_143005_a1b2c3d4
    """
    # 1. Get current time (YearMonthDay_HourMinuteSecond)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2. Generate a short unique ID (first 8 characters of a UUID)
    unique_id = uuid.uuid4().hex[:8]

    # 3. Combine them
    folder_name = f"run_{timestamp}_{unique_id}"
    full_path = os.path.join(base_path, folder_name)

    # 4. Create the folder safely
    os.makedirs(full_path, exist_ok=True)

    return full_path


if __name__ == '__main__':
    ray.init(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS, include_dashboard=False)
    device = torch.device("cpu")
    save_folder = get_save_folder()
    manager: CasinoManager = CasinoManager(device)
    manager.start()