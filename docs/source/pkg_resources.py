"""Compatibility shim for extensions that still import pkg_resources.

This docs build only needs ``declare_namespace`` for ``sphinx_tabs``.
"""


def declare_namespace(_name):
    """No-op namespace declaration for legacy setuptools consumers."""
