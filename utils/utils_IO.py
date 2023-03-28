import os
from typing import TextIO

def safe_open_w(path: str) -> TextIO:
    """
    Open "path" for writing, creating any parent directories as needed.

    Args
        path: path to file, in string format

    Return
        file: TextIO object
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')