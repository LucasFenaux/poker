import ray
import torch
import os
from datetime import datetime
import uuid

from src.casino_manager import CasinoManager


def get_save_folder(base_path="results"):
    """
    Generates a unique folder name based on the current time and a random ID.
    Example: results/run_20231027_143005_a1b2c3d4
    """
    base_path = os.path.abspath(base_path)
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
    try:
        ray.init("auto", namespace="casino")
        device = torch.device("cpu")
        save_folder = get_save_folder()
        manager: CasinoManager = CasinoManager(device, save_folder=save_folder)
        manager.start()
        ray.shutdown()
    except Exception as e:
        ray.shutdown()