# Pydantic Schema Code Generation Examples

This folder contains command-line tools and examples demonstrating bidirectional conversion between Pydantic models and JSON schemas.

## CLI Tools

### `schema_to_pydantic.py`
Converts JSON schema files to Pydantic Python model source code.

**Usage:**
```bash
# Generate and print to stdout
python schema_to_pydantic.py sample_schema.json

# Save to results.py
python schema_to_pydantic.py sample_schema.json -o results.py

# With additional options
python schema_to_pydantic.py sample_schema.json -o results.py --test-dynamic --log-level DEBUG
```

**Options:**
- `-o, --output`: Output file (default: stdout)
- `--no-dynamic`: Skip dynamic class creation (faster)
- `--test-dynamic`: Test dynamic classes after generation
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)

### `pydantic_to_schema.py`
Parses Pydantic Python model files and generates JSON schema dictionaries.

**Usage:**
```bash
# Parse and print to stdout
python pydantic_to_schema.py sample_models.py

# Save to results.json
python pydantic_to_schema.py sample_models.py -o results.json

# With additional options
python pydantic_to_schema.py sample_models.py -o results.json --normalize --test-roundtrip --log-level DEBUG
```

**Options:**
- `-o, --output`: Output file (default: stdout)
- `--normalize`: Normalize parsed schema
- `--test-roundtrip`: Test roundtrip conversion after parsing
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)

## Example Files

### Input Files
- `sample_schema.json` - Example JSON schema with User and Product models
- `sample_models.py` - Example hand-written Pydantic models (BlogPost and Comment)

### Output Files
- `results.py` - Generated Pydantic models (from schema_to_pydantic.py)
- `results.json` - Generated JSON schema (from pydantic_to_schema.py)

## Usage Examples

### Forward Conversion (Schema → Pydantic)
```bash
# Convert JSON schema to Python models
python schema_to_pydantic.py sample_schema.json -o results.py

# Test the generated models
python -c "from results import User, Product; print(User(user_id=1, username='test', email='test@example.com'))"
```

### Reverse Conversion (Pydantic → Schema)  
```bash
# Convert Python models to JSON schema
python pydantic_to_schema.py sample_models.py -o results.json

# View the generated schema
cat results.json | jq '.'
```

### Roundtrip Testing
```bash
# Test complete roundtrip conversion
python schema_to_pydantic.py sample_schema.json -o temp_models.py
python pydantic_to_schema.py temp_models.py -o temp_schema.json --test-roundtrip
```

## Testing Workflow

1. **Generate models from schema:**
   ```bash
   python schema_to_pydantic.py sample_schema.json -o results.py --test-dynamic
   ```

2. **Parse models back to schema:**
   ```bash
   python pydantic_to_schema.py results.py -o results.json --test-roundtrip
   ```

3. **Verify consistency:**
   ```bash
   # Compare original and roundtrip schemas
   diff sample_schema.json results.json
   ```