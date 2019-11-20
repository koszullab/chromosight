from chromosight.utils.io import load_kernel_config
import pathlib
import sys
from os.path import basename

# Here, each pattern config file detected in the kernels directory is loaded and
# made available as a variable in the API

# Get parent module (chromosight.kernels)
current_module = sys.modules[__name__]
# Find all json files in kernel dir
kernel_dir = pathlib.Path(__file__).parents[0]
for kernel_file in kernel_dir.glob("*.json"):
    # Get pattern names based on config file name
    pattern_name = str(kernel_file.with_suffix("").name)
    # Declare pattern configs as module level (chromosight.kernels) variables
    setattr(
        current_module,
        pattern_name,
        load_kernel_config(pattern_name, custom=False),
    )
