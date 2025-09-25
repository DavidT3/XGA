# -*- coding: utf-8 -*-
"""
This script parses the XSELECT mission database into a Python dictionary.

It was made largely using Google's Gemini tool.

The file's hierarchical structure (e.g., Mission:Submission:Detector:Datamode:keyword)
is converted into a nested dictionary, with the top-level keys being the telescope
mission names. The script accounts for the initial comment block and begins
parsing data from line 79 onwards, as specified.
"""
#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 25/09/2025, 17:22. Copyright (c) The Contributors

import json
import os
import sys


def _convert_to_proper_type(value):
    """
    Attempts to convert a string value to its correct type (int, float, bool, None).
    If conversion fails, the original string is returned.
    """
    if not isinstance(value, str):
        return value

    # Try converting to boolean
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False

    # Try converting "NONE" to Python's None
    if value.lower() == 'none':
        return None

    # Try converting to integer
    try:
        return int(value)
    except (ValueError, TypeError):
        pass

    # Try converting to float
    try:
        return float(value)
    except (ValueError, TypeError):
        pass

    # Return the original string if no conversion is possible
    return value


def parse_xselect_file(filepath):
    """
    Parses a file with colon-separated key-value pairs into a nested dictionary.

    Args:
        filepath (str): The path to the input file.

    Returns:
        tuple: A tuple containing:
            - A dictionary representing the parsed data.
            - A set of all unique telescope names found.
            - A list of error messages for any lines that could not be parsed.
    """
    telescope_data = {}
    telescope_names = set()
    errors = []
    data_start_line = 79

    try:
        with open(filepath, 'r') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()

                # Skip comment lines and empty lines, as well as lines before the data section
                if line_number < data_start_line or not line or line.startswith('!'):
                    continue

                try:
                    # Check for lines that contain a colon, which indicates a hierarchical structure.
                    if ':' in line:
                        # Split the line by the first space to separate the hierarchical key path from the values.
                        # This handles cases where the key path itself has spaces (though unlikely)
                        parts = line.split(' ', 1)
                        key_path_str = parts[0].strip()

                        # Split the key path by colon to get the hierarchy
                        key_path_parts = [part.strip() for part in key_path_str.split(':')]

                        # Add the top-level telescope name to our set
                        if key_path_parts:
                            telescope_names.add(key_path_parts[0])

                        # The actual key is the last part of the key path. The nested path is everything before it.
                        final_key = key_path_parts[-1]
                        nested_path = key_path_parts[:-1]

                        # Get the remaining string as values and split by space
                        value_str = parts[1].strip() if len(parts) > 1 else ""
                        values = [_convert_to_proper_type(v.strip()) for v in value_str.split() if v.strip()]

                        # Build the nested dictionary structure
                        current_level = telescope_data
                        for part in nested_path:
                            current_level = current_level.setdefault(part, {})

                        # Assign the final key with the list of values
                        if len(values) == 1:
                            current_level[final_key] = values[0]
                        else:
                            current_level[final_key] = values
                    else:
                        # Case for lines with no colon, just space-separated words
                        line_parts = line.split()
                        if len(line_parts) > 1:
                            key = line_parts[0].strip()
                            value = [_convert_to_proper_type(v.strip()) for v in line_parts[1:] if v.strip()]
                            if len(value) == 1:
                                telescope_data[key] = value[0]
                            else:
                                telescope_data[key] = value

                except ValueError:
                    errors.append(f"Line {line_number}: Invalid format, missing separator in '{line}'")
                except Exception as e:
                    errors.append(f"Line {line_number}: An unexpected error occurred: {e}")

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.", file=sys.stderr)
        return {}, set(), [f"File not found: {filepath}"]
    except Exception as e:
        print(f"Error: An unexpected error occurred while reading the file: {e}", file=sys.stderr)
        return {}, set(), [f"File reading error: {e}"]

    return telescope_data, telescope_names, errors


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_to_parse = sys.argv[1]
    else:
        headas = os.environ['HEADAS']
        file_to_parse = os.path.join("/".join(headas.split('/')[:-1]), "ftools/xselect/xselect.mdb")

    parsed_dict, telescopes, parsing_errors = parse_xselect_file(file_to_parse)

    if parsed_dict:
        print("Successfully parsed data into a dictionary:")
        print(json.dumps(parsed_dict, indent=2))
        with open("xselect_mission_database.json", 'w') as f:
            json.dump(parsed_dict, f, indent=2)

    if telescopes:
        print("\n-----------------------------------------------------")
        print("List of all unique telescopes found in the file:")
        print(sorted(list(telescopes)))
        print("-----------------------------------------------------")

    if parsing_errors:
        print("\n-----------------------------------------------------")
        print("Encountered the following errors during parsing:")
        for error in parsing_errors:
            print(f"- {error}")
        print("-----------------------------------------------------")
