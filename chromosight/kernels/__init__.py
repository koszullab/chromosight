"""
Chromosight's kernel submodule contains each default kernel in the form of a
dictionary whith the kernel name. The items of the dictionaries are the
key-value pairs from the kernel's json file, with the kernel matrices
pre-loaded under the "kernels" key. Here is the kernel submodule can be used to
extract the first borders kernel:

.. code-block:: python

    import chromosight.kernels as ck
    kernel = ck.borders['kernels'][0]

A list of all available kernel names can also be accessed directly:

.. code-block:: python

    import chromosight.kernels as ck
    names = ck.kernel_names

"""
from chromosight.utils.io import load_kernel_config
import pathlib
import sys

# Here, each pattern config file detected in the kernels directory is loaded
# and made available as a variable in the API

# Get parent module (chromosight.kernels)
current_module = sys.modules[__name__]
# Find all json files in kernel dir
kernel_dir = pathlib.Path(__file__).parents[0]
kernel_names = []
for kernel_file in kernel_dir.glob("*.json"):
    # Get pattern names based on config file name
    pattern_name = str(kernel_file.with_suffix("").name)
    # Declare pattern configs as module level (chromosight.kernels) variables
    setattr(
        current_module,
        pattern_name,
        load_kernel_config(pattern_name, custom=False),
    )
    kernel_names.append(pattern_name)

setattr(current_module, 'kernel_names', kernel_names)
