
import numpy as np

def display_fields(name, fields, optional_fields, data_entry):
    if len(fields) > 0:
        print(f"{name}: {type(data_entry[fields[0]])}")
        if isinstance(data_entry[fields[0]], np.ndarray):
            print(f"fields[0] {fields[0]} shape: {data_entry[fields[0]].shape}")
        if isinstance(data_entry[fields[0]], list):
            print(f"data_entry[fields[0]][0] {fields[0]} shape: {data_entry[fields[0]][0].shape} data_entry[fields[0]] len: {len(data_entry[fields[0]])}")
        else:
            print(f"data_entry[fields[0]] {fields[0]} type: {type(data_entry[fields[0]])}")
    else:
        print(f"{name}: {fields} + {optional_fields}")