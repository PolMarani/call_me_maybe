from typing import cast
import json


def load_json(path_file: str) -> list:
    """Load and parse a JSON file into a Python list.

    Parameters
    ----------
    path_file : str
        Path to the JSON file to load.

    Returns
    -------
    list
        The parsed JSON content as a list. Returns an empty list if
        the file does not exist or contains invalid JSON.
    """
    try:
        with open(path_file, 'r') as f:
            json_opened = json.load(f)
        return cast(list, json_opened)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def write_json(output_data: list[dict], output_filepath: str) -> None:
    """Write a list of dictionaries to a file as formatted JSON.

    Parameters
    ----------
    output_data : list[dict]
        The data to serialize and write to disk.
    output_filepath : str
        Path to the output JSON file.

    Returns
    -------
    None
        This function does not return a value. If writing fails
        (missing directory, permission error, or non-serializable
        data), the error is printed instead of raised.
    """
    try:
        with open(output_filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
    except (FileNotFoundError, PermissionError, TypeError) as e:
        print("ERRROOOOOOORRRRR:", e)
