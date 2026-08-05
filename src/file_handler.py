import json


def load_json(path_file: str) -> list:

    try:
        with open(path_file, 'r') as f:
            json_opened = json.load(f)
        return json_opened
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def write_json(output_data: list[dict], output_filepath: str) -> None:
    try:
        with open(output_filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
    except (FileNotFoundError, PermissionError, TypeError) as e:
        print("ERRROOOOOOORRRRR:", e)
