import hashlib
import json
from typing import Any, Dict


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """
    Hash dictionary using MD5. Used in step caching; steps are cached based on the signature.
    """
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def is_jsonable(x):
    """
    Test if input is JSON-serializable.
    """
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False
