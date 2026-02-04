# TensorSuggestLite Developer Guide

**Version**: 2.0  
**Last Updated**: February 3, 2026  
**Maintainer**: CoreRed Project

---

## Overview

TensorSuggestLite is a configuration-driven text classification trainer. This guide explains the architecture, code structure, and how to extend or modify the codebase safely.

## Architecture

### Core Concept

TensorSuggestLite treats configuration files (JSON, YAML, TOML) as training datasets. The application:

1. Parses structured configuration into training samples
2. Trains a simple neural network using TensorFlow/Keras
3. Exports model artifacts (tokenizer, model, label encoder)
4. Converts to TFLite for embedded deployment

### Components

```
┌─────────────────────────────────────────┐
│         tensorsuggestlite.py           │
│        (GUI + Orchestration)            │
└──────────────┬──────────────────────────┘
               │
      ┌────────┴────────┐
      │                 │
      ▼                 ▼
┌──────────┐      ┌──────────┐
│ Trainer  │      │  PyQt6   │
│ Modules  │      │   UI     │
└──────────┘      └──────────┘
      │
┌─────┴──────┬──────────┬─────────┐
│            │          │         │
▼            ▼          ▼         ▼
json/      yaml/     toml/    TensorFlow
trainer    trainer  trainer   /Keras

Generated Artifacts:
- tokenizer.json
- text_classifier.keras
- label_encoder.json
- text_classifier.tflite
```

## Module Structure

### 1. Main Application (`tensorsuggestlite.py`)

**Responsibilities**:
- PyQt6 GUI window and event handling
- Thread management for training/conversion
- File system watching for artifact detection
- Theme management (dark/light)

**Key Classes**:

- `TensorSuggestLiteUI(QWidget)` - Main window
  - `training_progress` (signal) - Progress updates (0-100)
  - `log_signal` (signal) - Log message passing
  - `_on_train_clicked()` - Initiates training in background thread
  - `_on_convert_clicked()` - Initiates TFLite conversion
  - `_append_log()` - Thread-safe console logging

**Threading Model**:
- GUI runs on main Qt thread
- Training/conversion run in daemon threads
- Communication via signals and `QTimer.singleShot()`

### 2. Interpreter Modules

Each format has its own package: `json_interpreter/`, `yaml_interpreter/`, `toml_interpreter/`

**Structure**:
```python
format_interpreter/
├── __init__.py      # Exports trainer module
└── trainer.py       # Training and conversion logic
```

**Key Functions**:

```python
def train_from_{format}(
    file_path: str,
    progress_cb: Optional[Callable[[int], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    epochs: int = 12
) -> Dict[str, Any]:
    """
    Train model from configuration file.
    
    Returns:
        {
            'items_serialized': int,
            'tokenizer_path': str,
            'model_path': str,
            'label_encoder_path': str
        }
    """
```

```python
def convert_to_tflite(
    model_dir: Optional[str] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None
) -> str:
    """
    Convert Keras model to TFLite.
    
    Returns:
        Path to generated .tflite file
    """
```

## Data Flow

### Training Flow

1. **File Selection** (GUI)
   - User clicks "Import JSON/YAML/TOML"
   - `_import_file_dialog()` → `_on_file_selected()`
   - Sets `selected_path` and `selected_kind`

2. **Training Initiation**
   - User clicks "Train Model"
   - `_on_train_clicked()` spawns `_train_worker()` thread
   - Status: "training model"

3. **Trainer Execution**
   - `train_from_{format}()` called with callbacks
   - Progress: 0% → 100% reported via `progress_cb`
   - Logs: Sent via `log_cb`

4. **Artifact Generation**
   - Files saved to `generated/{format}/`:
     - `tokenizer.json` - Vocabulary mapping
     - `text_classifier.keras` - Full model
     - `label_encoder.json` - Category → index mapping

5. **UI Update**
   - File watcher detects artifacts
   - "Generate TFLite" button enabled
   - Status: "model trained"

### Conversion Flow

1. **TFLite Conversion**
   - User clicks "Generate TFLite"
   - `_on_convert_clicked()` spawns `_convert_worker()` thread

2. **Conversion Process**
   - Loads Keras model from `generated/{format}/`
   - Applies `tf.lite.Optimize.DEFAULT` quantization
   - Saves to `text_classifier.tflite`

3. **Export**
   - "Export TFLite" button appears
   - User saves file to desired location

## Code Conventions

### Naming

- **Classes**: PascalCase (`TensorSuggestLiteUI`)
- **Functions**: snake_case (`train_from_json`)
- **Private methods**: Leading underscore (`_append_log`)
- **Constants**: UPPER_SNAKE_CASE (rare, inline literals preferred)

### Error Handling

**DO**:
```python
try:
    result = risky_operation()
except SpecificError as e:
    log_error(f"Operation failed: {e}")
    raise RuntimeError("Context-specific message") from e
```

**DON'T**:
```python
try:
    result = risky_operation()
except:  # Too broad
    pass  # Silent failure
```

### Type Hints

All public functions have type hints:

```python
def train_from_json(
    json_path: str,
    progress_cb: Optional[Callable[[int], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    epochs: int = 12
) -> Dict[str, Any]:
    ...
```

### Docstrings

Google style format:

```python
def function_name(arg1: str, arg2: int) -> bool:
    """Brief description.

    Longer description if needed.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When input is invalid.
    """
```

## Extending the Codebase

### Adding a New Format

To add support for a new configuration format (e.g., XML):

1. **Create Package**:
   ```bash
   mkdir xml_interpreter
   touch xml_interpreter/__init__.py
   touch xml_interpreter/trainer.py
   ```

2. **Implement Trainer** (`xml_interpreter/trainer.py`):
   ```python
   def train_from_xml(xml_path, progress_cb=None, log_cb=None, epochs=12):
       # Parse XML into preguntas (texts) and categorias (labels)
       # Follow json_interpreter/trainer.py as template
       ...
       
       # Save to generated/xml/
       gen_dir = os.path.join(_project_root(), 'generated', 'xml')
       ...
   
   def convert_to_tflite(model_dir=None, progress_cb=None, log_cb=None):
       # Reuse pattern from json_interpreter
       ...
   ```

3. **Update GUI** (`tensorsuggestlite.py`):
   ```python
   # Add button
   self.btn_xml = QPushButton("Import XML")
   
   # Add to _train_worker
   elif self.selected_kind == 'xml':
       from xml_interpreter import trainer as xml_trainer
       trainer = xml_trainer
       train_fn = getattr(trainer, 'train_from_xml')
   ```

4. **Test**:
   ```bash
   python -c "from xml_interpreter import trainer; print('OK')"
   ```

### Modifying the Model

Model is defined in each `trainer.py` around line 120:

```python
vocab_size = len(tokenizer.word_index) + 1
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, mask_zero=True),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(unique_labels), activation='softmax')
])
```

**To modify**:
1. Change architecture (add layers, change units)
2. Update all three trainers identically (json, yaml, toml)
3. Test on sample data
4. Update model version in artifacts if breaking change

### Adding Progress Checkpoints

Trainers report progress via callbacks:

```python
_safe_progress(0)   # Start
_safe_progress(10)  # Tokenizer ready
_safe_progress(15)  # Preprocessing done
_safe_progress(20-85)  # Training (via epoch callback)
_safe_progress(95)  # Artifacts saved
_safe_progress(100) # Complete
```

**To add checkpoint**:
```python
_safe_progress(17)
_safe_log('Custom step completed')
```

## Testing

### Manual Testing

```bash
# Test import
python -c "from json_interpreter import trainer"

# Test training (requires TensorFlow)
python -c "
from json_interpreter import trainer
result = trainer.train_from_json('test/question_data.json', epochs=1)
print(result)
"

# Test GUI
python tensorsuggestlite.py
```

### Automated Testing (Future)

When implementing tests:

```bash
mkdir tests
touch tests/__init__.py
touch tests/test_json_trainer.py
```

Example test:
```python
import pytest
from json_interpreter import trainer

def test_train_basic():
    result = trainer.train_from_json('test/question_data.json', epochs=1)
    assert 'model_path' in result
    assert os.path.exists(result['model_path'])
```

Run: `make test` or `pytest tests/`

## Debugging

### Enable Verbose Logging

Trainers log via callbacks. To see all messages:

```python
def verbose_log(msg):
    print(f"[DEBUG] {msg}")

trainer.train_from_json(
    'data.json',
    log_cb=verbose_log
)
```

### Common Issues

**Issue**: GUI freezes during training  
**Cause**: Thread safety violation or blocking operation  
**Fix**: Ensure all UI updates use `QTimer.singleShot()` or signals

**Issue**: TFLite conversion fails  
**Cause**: Keras model not found or corrupted  
**Fix**: Check `generated/{format}/text_classifier.keras` exists and is valid

**Issue**: Import errors  
**Cause**: TensorFlow not installed  
**Fix**: `pip install tensorflow` or use `requirements-pinned.txt`

## Performance Considerations

### Training Speed

- **Small datasets** (<1000 samples): ~30 seconds
- **Medium datasets** (1000-10000 samples): 1-5 minutes
- **Large datasets** (>10000 samples): 5-30 minutes

Bottleneck: TensorFlow model.fit() - single-threaded CPU

### Memory Usage

- **Typical**: 200-500 MB
- **With TensorFlow loaded**: 1-2 GB
- **During training**: +500 MB per 10k samples

### Optimizations

1. **Reduce epochs**: Default is 12, can use 5-8 for faster prototyping
2. **Smaller vocabulary**: Limit tokenizer word count (not currently exposed)
3. **Batch size**: Default 32, increase to 64/128 for speed (edit trainer.py)

## Security Considerations

### Input Validation

Currently MINIMAL. Configuration files are trusted.

**Risks**:
- Malformed files can crash the application
- Large files can consume excessive memory
- No size limits enforced

**Mitigations**:
- Only load files from trusted sources
- Run in isolated environment if processing untrusted data

### Model Artifacts

Generated models contain embedded training data patterns.

**Do NOT**:
- Train on sensitive/proprietary data and distribute models
- Share models without understanding data leakage risks

### Dependencies

TensorFlow has known vulnerabilities. See `SECURITY.md`.

## Contributing

See `CONTRIBUTING.md` for:
- Code style guidelines
- Pull request process
- Review criteria

## Support

- **Issues**: GitHub Issues
- **Email**: houjou.sxnnyside@sxnnysideproject.com
- **Security**: security.sxnnyside@sxnnysideproject.com

---

**Last Review**: February 3, 2026  
**Next Review**: Upon major architecture change

