# model_configs.py — Advanced Model Parameters

## Overview
`model_configs.py` defines structured metadata for advanced inference parameters. It provides a typed schema that both the web and desktop UIs serialise into dynamic configuration controls.

## Role in the Project
- Supplies option descriptors that the Flask templates embed in JSON for client-side rendering.
- Enables validation and coercion of user-provided values before they are applied to model inference.
- Keeps model-specific tuning values in one place, reducing duplication across interfaces.

## Key Components
- **`ParameterOption`**: Simple dataclass representing a selectable value with an optional human-readable label.
- **`ParameterSchema`**: Describes the expected type, default, bounds, and presentation metadata for a parameter. Provides `coerce` and `to_frontend` helpers.
- **`MODEL_CONFIGS`**: Nested dictionary keyed by model name and parameter key that enumerates supported tweaks (input size, matte strength, device selection, etc.).
- **Utility functions**: `parse_model_options` and `serialise_model_configs` convert request payloads into validated dictionaries and frontend-friendly JSON structures.

## Implementation Notes
- `ParameterSchema.coerce` handles blank values by falling back to defaults and performs bounds checking for numeric types.
- When adding a new model parameter, update both the schema entry and any frontend forms consuming the generated JSON.
- The Flask and Tkinter UIs rely on consistent keys—renaming parameters is a breaking change.

## Invocation Example
```python
from model_configs import parse_model_options, MODEL_CONFIGS

payload = {"matte_strength": "1.4"}
validated = parse_model_options("briaai/RMBG-2.0", payload)
print(validated)
# {'matte_strength': 1.4, 'input_size': 1024, 'output_format': 'rgba', 'device': 'auto'}
```

> **Shared Module:** Consumed by Flask web UI and Tkinter GUI for dynamic controls.
