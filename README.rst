*******************************
Welcome to reV Routing (reVRt)!
*******************************

|License| |Zenodo| |PythonV| |PyPi| |Ruff| |Pixi| |SWR| |Codecov|


.. |PythonV| image:: https://badge.fury.io/py/nlr-revrt.svg
    :target: https://pypi.org/project/nlr-revrt/

.. |PyPi| image:: https://img.shields.io/pypi/pyversions/nlr-revrt.svg
    :target: https://pypi.org/project/nlr-revrt/

.. |Ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff

.. |License| image:: https://img.shields.io/badge/License-BSD_3--Clause-orange.svg
    :target: https://opensource.org/licenses/BSD-3-Clause

.. |Pixi| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json
    :target: https://pixi.sh

.. |SWR| image:: https://img.shields.io/badge/SWR--25--112_-blue?label=NLR
    :alt: Static Badge

.. |Zenodo| image:: https://zenodo.org/badge/944738283.svg
    :target: https://doi.org/10.5281/zenodo.17173574

.. |Codecov| image:: https://codecov.io/github/NatLabRockies/reVRt/graph/badge.svg?token=W072E3ZEMR
    :target: https://codecov.io/github/NatLabRockies/reVRt

.. inclusion-intro

The reV Routing tool is a computational framework for modeling and optimizing
transmission infrastructure requirements for electrical grid connections. By
employing a spatially-aware least-cost-path methodology, it allows users to
incorporate a wide range of factors including siting constraints, regional
component costs, land composition costs, point-of-interconnection costs, and
network upgrade costs. Additionally, the tool enables advanced follow-on
analyses, such as land characterization for potential transmission line routes,
to support informed decision-making. Although it's designed to integrate
seamlessly with the reV model, the reV Routing tool is versatile and can also
be utilized independently for standalone analyses in transmission planning and
resource assessment scenarios.


Installing reVRt
================
The quickest way to install reVRt for users and analysts is from PyPi:

.. code-block:: bash

    pip install nlr-revrt

If you would like to install and run reVRt from source, we recommend using `pixi <https://pixi.sh/latest/>`_:

.. code-block:: bash

    git clone git@github.com:NatLabRockies/reVRt.git; cd reVRt
    pixi run reVRt


For detailed instructions, see the `installation documentation <https://natlabrockies.github.io/reVRt/misc/installation.html>`_.


Development
===========
Please see the `Development Guidelines <https://natlabrockies.github.io/reVRt/dev/index.html>`_
if you wish to contribute code to this repository.
