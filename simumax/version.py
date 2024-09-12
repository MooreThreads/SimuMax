"""Version information of SimuMax"""


def _as_tuple(version: str):
    return tuple(int(i) for i in version.split(".") if i.isdigit())


__version__ = "0.1.dev0"
__version_info__ = _as_tuple(__version__)
