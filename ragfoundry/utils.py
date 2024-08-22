def check_package_installed(package_name: str, optional_msg: str = ""):
    """
    Check if a package is installed.
    """

    import importlib.util

    if importlib.util.find_spec(package_name) is None:
        raise ImportError(f"{package_name} package is not installed; {optional_msg}")
