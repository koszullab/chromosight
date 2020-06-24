chromosight.kernels package
===========================

Chromosight's kernel submodule contains each default kernel in the form of a
dictionary whith the kernel name. The items of the dictionaries are the
key-value pairs from the kernel's json file, with the kernel matrices
pre-loaded under the "kernels" key. Here is the kernel submodule can be used to
extract the first borders kernel:

.. code-block:: python

    import chromosight.kernels as ck
    kernel = ck.borders['kernels'][0]

Module contents
---------------

.. automodule:: chromosight.kernels
    :members:
    :undoc-members:
    :show-inheritance:
