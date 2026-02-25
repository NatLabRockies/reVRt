Installing reVRt
================

.. inclusion-install

Installing from PyPI
--------------------

reVRt can be installed via pip from
`PyPI <https://pypi.org/project/NLR-reVRt>`__.

.. code-block:: shell

    pip install nlr-revrt

.. note::

    You must have ``pip>=19.3`` to install from PyPI.

.. note::

    It is recommended to install and run reVRt from a virtual environment, for example,
    using `Miniconda <https://www.anaconda.com/docs/getting-started/miniconda/main>`__.


Handling ImportErrors
---------------------

If you encounter an ``ImportError``, it usually means that Python couldn't find reVRt in the list of available
libraries. Python internally has a list of directories it searches through, to find packages. You can
obtain these directories with.

.. code-block:: python

    import sys
    sys.path

One way you could be encountering this error is if you have multiple Python installations on your system
and you don't have reVRt installed in the Python installation you're currently using.
In Linux/Mac you can run ``which python`` on your terminal and it will tell you which Python installation you're
using. If it's something like "/usr/bin/python", you're using the Python from the system, which is not recommended.
Instead, it is highly recommended to use an isolated environment, such as one created using
`Miniconda <https://www.anaconda.com/docs/getting-started/miniconda/main>`__, for package and dependency updates.


Installing from Source
----------------------

If you would like to install and run reVRt from source, we recommend using
`pixi <https://pixi.sh/latest/>`_. This tool allows developers to install libraries and
applications in a reproducible way across multiple platforms. This means bugs are easier
to reproduce, and it's easier to move your development environment to a new piece of hardware.

We keep a version-controlled ``pixi.lock`` in the repository to allow locking with
the full requirements tree so we can reproduce behaviors and bugs and easily compare
results.

Start by cloning the ``reVRt`` repository:

- Using ssh: :code:`git clone git@github.com:NatLabRockies/reVRt.git`
- Using https: :code:`git clone https://github.com/NatLabRockies/reVRt.git`

If you don't already have ``pixi`` installed on your system, follow the (simple) `installation
instructions <https://pixi.sh/latest/#installation>`_.

Once you have both ``pixi`` and the ``reVRt`` source code, simply run:

.. code-block:: shell

    pixi shell

from the source code repository. You can now start using ``reVRt``!

If you are planning to contribute to ``reVRt``, you can use the ``dev`` feature in ``pixi`` to
get all necessary Python and Rust development tools:

.. code-block:: shell

    pixi shell -e pdev

You are welcome to use a different environment manager (e.g. ``conda``, ``mamba``, etc),
but we make no promises to provide support on environment-related issues/bugs in this case.
