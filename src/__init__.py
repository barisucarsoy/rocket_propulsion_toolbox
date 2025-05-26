# Make the loaded configuration data available directly when the package is imported.
from .config_loader import config_data

# Optional: You might want to add an __all__ if you want to control
# what 'from src import *' imports, though explicit imports are often preferred.
__all__ = ['config_data']

# print(f"Package initialized. Configuration loaded into 'config_data'.")
