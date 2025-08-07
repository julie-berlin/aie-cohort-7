import os
from datetime import datetime

def save_to_file(content: str, filename: str) -> None:
    """Save content to a text file with the specified filename and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{filename}_{timestamp}.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)
