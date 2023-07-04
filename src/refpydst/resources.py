import json
import pkgutil
from typing import Any


def _read_resource(relative_path: str) -> str:
    """
    Reads in a file resource as a string, for some file distributed with this package. Useful for reading in JSON data
    that is released in this package without requiring users to manage paths and data locations (e.g. for schema.json)

    :param relative_path: relative path within the package of the file, e.g. schema.json is in "db/multiwoz/schema.json"
    :return: string file content (not parsed in JSON case)
    """
    return pkgutil.get_data(__name__, relative_path).decode("utf-8")


def read_json_resource(relative_path: str) -> Any:
    """
    Reads in a file resource as a string, then parses it to JSON.
    :param relative_path: relative path within the package of the file, e.g. schema.json is in "db/multiwoz/schema.json"
    :return:
    """
    return json.loads(_read_resource(relative_path))