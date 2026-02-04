# TensorSuggestLite Examples

This directory contains functional, tested examples in all supported formats.

## Files

- **example.json** (Recommended) - JSON format with sentiment analysis data
- **example.yaml** - YAML format (same data)
- **example.toml** - TOML format (same data)

## Format Recommendation

**Use JSON** for production. It offers:
- Universal tool support
- Clear error messages
- Better validation
- Widest compatibility

YAML and TOML are fully supported alternatives.

## Schema

All files use the English schema:

```json
{
  "CATEGORY_NAME": {
    "responses": [
      {"response": "training text example"}
    ],
    "synonyms": ["related", "terms"]
  }
}
```

## Usage

### Quick Test

```bash
# From repository root
python -c "
from json_interpreter import trainer
result = trainer.train_from_json('examples/example.json', epochs=2)
print(f'Trained on {result[\"items_serialized\"]} samples')
"
```

### GUI

1. Launch: `python tensorsuggestlite.py`
2. Click **Import JSON**
3. Select `examples/example.json`
4. Click **Train Model**

## Data Content

The examples contain sentiment analysis data:
- **POSITIVE**: 5 responses + 5 synonyms
- **NEGATIVE**: 5 responses + 5 synonyms
- **NEUTRAL**: 5 responses + 5 synonyms

Total: 45 training samples across 3 categories.

## Extending

To create your own training data:

1. Copy `example.json`
2. Replace categories with your labels
3. Add your training examples
4. Keep the same schema structure

### Best Practices

- **Balance**: Similar sample counts per category
- **Variety**: Diverse phrasing
- **Quality**: Representative real-world examples
- **Minimum**: 20-50 samples per category
- **Recommended**: 100-500 samples per category

## Validation

All examples have been validated:
- ✅ Valid JSON/YAML/TOML syntax
- ✅ Correct schema structure
- ✅ Successfully train models
- ✅ Generate all artifacts
- ✅ Convert to TFLite

Last validated: February 3, 2026

