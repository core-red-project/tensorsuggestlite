# TensorSuggestLite User Guide

**Version**: 2.0  
**Last Updated**: February 3, 2026

---

## What is TensorSuggestLite?

TensorSuggestLite is a desktop application that trains text classification models from simple configuration files. It's designed for rapid prototyping and experimentation with machine learning without writing code.

## Installation

### Requirements

- **Python**: 3.8 or later
- **Operating System**: Linux, macOS, or Windows
- **RAM**: 2 GB minimum, 4 GB recommended
- **Disk**: 500 MB for application + 1 GB for TensorFlow

### Quick Install

```bash
# Clone repository
git clone https://github.com/Sxnnyside-Project/TensorSuggestLite.git
cd TensorSuggestLite

# Run setup
chmod +x SETUP.sh
./SETUP.sh
```

Or using Make:

```bash
make setup
```

### Manual Install

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements-pinned.txt
```

## Basic Usage

### Launch Application

```bash
# Using Make
make run

# Or directly
python tensorsuggestlite.py
```

### Training a Model

1. **Prepare Data File**

   Create a JSON, YAML, or TOML file with this structure:

   ```json
   {
     "CATEGORY_NAME": {
       "respuestas": [
         {"respuesta": "training text example 1"},
         {"respuesta": "training text example 2"}
       ],
       "sinonimos": ["synonym1", "synonym2"]
     }
   }
   ```

   See `test/question_data.json` for a complete example.

2. **Import File**

   Click one of:
   - **Import JSON**
   - **Import YAML**
   - **Import TOML**

   Select your configuration file.

3. **Train**

   Click **Train Model**. Progress will display in the status bar and console.

   Training typically takes 1-5 minutes depending on data size.

4. **Generate TFLite**

   After training completes, click **Generate TFLite** to create a mobile-optimized model.

5. **Export**

   Click **Export TFLite** and choose where to save the `.tflite` file.

## Configuration File Format

### Structure

All formats (JSON, YAML, TOML) follow the same structure:

```yaml
CATEGORY_NAME:
  respuestas:
    - respuesta: "text sample 1"
    - respuesta: "text sample 2"
  sinonimos:
    - "alternative term 1"
    - "alternative term 2"
```

### Fields

- **CATEGORY_NAME**: The classification label (e.g., "SPORTS", "POLITICS")
- **respuestas**: List of training text samples for this category
- **sinonimos**: Additional related terms (optional)

### Example: Sentiment Analysis

```json
{
  "POSITIVE": {
    "respuestas": [
      {"respuesta": "This is great!"},
      {"respuesta": "I love this product"},
      {"respuesta": "Excellent service"}
    ],
    "sinonimos": ["happy", "satisfied", "pleased"]
  },
  "NEGATIVE": {
    "respuestas": [
      {"respuesta": "This is terrible"},
      {"respuesta": "Very disappointing"},
      {"respuesta": "Waste of money"}
    ],
    "sinonimos": ["unhappy", "dissatisfied", "angry"]
  }
}
```

### Best Practices

1. **Balanced Data**: Include similar numbers of samples per category
2. **Variety**: Use diverse phrasing within each category
3. **Quality**: More representative samples > more samples
4. **Minimum**: At least 20-50 samples per category for basic results
5. **Recommended**: 100-500 samples per category for good results

## Generated Artifacts

After training, files are created in `generated/{format}/`:

| File | Purpose |
|------|---------|
| `tokenizer.json` | Vocabulary and text preprocessing configuration |
| `text_classifier.keras` | Full Keras model (can be loaded for inference) |
| `label_encoder.json` | Maps category names to numeric IDs |
| `text_classifier.tflite` | Optimized model for mobile/embedded deployment |

### Using the Model

**Python (Keras)**:
```python
import tensorflow as tf

model = tf.keras.models.load_model('generated/json/text_classifier.keras')
# Use model for predictions
```

**Mobile (TFLite)**:
Copy `text_classifier.tflite` to your Android/iOS app and use TFLite runtime.

## GUI Interface

### Main Window

```
┌──────────────────────────────────────────┐
│ TensorSuggestLite                    🌙  │
├──────────────────────────────────────────┤
│ [Import JSON] [Import TOML] [Import YAML]│
│ ┌────────────────────────────────────┐  │
│ │ /path/to/file.json                 │  │
│ └────────────────────────────────────┘  │
│ ┌────────────────────────────────────┐  │
│ │ Console output...                  │  │
│ │                                    │  │
│ └────────────────────────────────────┘  │
├──────────────────────────────────────────┤
│ [Train Model]                            │
│ [Generate TFLite]                        │
│ [Clear Generated Folders]                │
│ [Clear Screen]                           │
│                                          │
│ Status: model not trained                │
│ ▓▓▓▓▓▓░░░░░░░░░░░░░░░░ 30%            │
└──────────────────────────────────────────┘
```

### Buttons

- **Import JSON/YAML/TOML**: Select training data file
- **Train Model**: Start training process
- **Generate TFLite**: Convert trained model to TFLite format
- **Clear Generated Folders**: Delete all artifacts for current format
- **Clear Screen**: Clear console output
- **Export TFLite**: Save TFLite model to chosen location
- **🌙/☀️**: Toggle between dark and light themes

### Status Indicators

| Status | Meaning |
|--------|---------|
| model not trained | No model exists yet |
| training model | Training in progress |
| model trained | Training complete, ready for conversion |
| generating tflite | TFLite conversion in progress |
| tflite generated successfully | Conversion complete, ready for export |
| process error | Something went wrong (check console) |

## Common Workflows

### Quick Prototype

```bash
# 1. Setup (first time only)
make setup

# 2. Create small test file
cat > quick_test.json << 'EOF'
{
  "YES": {
    "respuestas": [
      {"respuesta": "yes"}, {"respuesta": "yeah"}, {"respuesta": "sure"}
    ],
    "sinonimos": ["affirmative", "okay"]
  },
  "NO": {
    "respuestas": [
      {"respuesta": "no"}, {"respuesta": "nope"}, {"respuesta": "nah"}
    ],
    "sinonimos": ["negative", "deny"]
  }
}
EOF

# 3. Run app, import file, train
make run
```

### Production Model

1. Collect 500+ samples per category
2. Format as JSON/YAML/TOML
3. Train with default settings (12 epochs)
4. Generate TFLite
5. Test model with sample inputs
6. Deploy to target platform

### Iterative Refinement

1. Train initial model
2. Note misclassifications
3. Add more samples for problematic cases
4. Click "Clear Generated Folders"
5. Train again with updated data
6. Repeat until satisfied

## Troubleshooting

### Application Won't Start

**Symptom**: `python tensorsuggestlite.py` fails  
**Solutions**:
- Check Python version: `python3 --version` (must be 3.8+)
- Verify virtual environment: `source .venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements-pinned.txt`

### Training Fails

**Symptom**: Error during training, status shows "training error"  
**Solutions**:
- Check console output for specific error
- Verify input file format (valid JSON/YAML/TOML)
- Ensure at least 2 categories with 5+ samples each
- Check disk space (need ~100 MB free)

### GUI Freezes

**Symptom**: Window becomes unresponsive  
**Cause**: Training is CPU-intensive  
**Solution**: Wait for training to complete (1-5 minutes typical)

### "Generate TFLite" Button Disabled

**Symptom**: Button stays grayed out after training  
**Solutions**:
- Verify training completed successfully (check console)
- Confirm artifacts exist: `ls generated/{format}/`
- Restart application and reimport file

### TFLite Conversion Fails

**Symptom**: Error during conversion, status shows "tflite generation error"  
**Solutions**:
- Check that Keras model exists: `ls generated/{format}/text_classifier.keras`
- Verify TensorFlow installation: `python -c "import tensorflow; print(tensorflow.__version__)"`
- Try training again (might be corrupted model)

## Performance Tips

### Faster Training

- Use fewer epochs (5-8 instead of 12) for testing
- Smaller datasets train faster
- Close other applications to free RAM

### Better Results

- More diverse training samples per category
- Balance categories (similar sample counts)
- Remove duplicate or very similar samples
- Use representative real-world examples

## Programmatic API

For automation or scripting, use the Python API directly:

```python
from json_interpreter import trainer

# Train model
def progress(pct):
    print(f"Progress: {pct}%")

def log(msg):
    print(f"[LOG] {msg}")

result = trainer.train_from_json(
    'data.json',
    progress_cb=progress,
    log_cb=log,
    epochs=12
)

print(f"Model: {result['model_path']}")

# Convert to TFLite
tflite_path = trainer.convert_to_tflite(
    model_dir='generated/json',
    progress_cb=progress,
    log_cb=log
)

print(f"TFLite: {tflite_path}")
```

## Data Privacy

- All processing happens locally
- No data sent to external servers
- Models contain patterns from training data
- Do not share models trained on sensitive data

## Limitations

- No input validation (files must be well-formed)
- No hyperparameter tuning interface
- Single-threaded training (CPU only)
- No real-time evaluation metrics
- Basic model architecture (not state-of-the-art)

See `REPOSITORY_STATUS.md` for complete limitations list.

## Support

- **Documentation**: `docs/` directory
- **Examples**: `test/` directory
- **Issues**: [GitHub Issues](https://github.com/Sxnnyside-Project/TensorSuggestLite/issues)
- **Email**: support.sxnnyside@sxnnysideproject.com

## License

MIT License - See `LICENSE` file for details.

---

**Need More Help?**

1. Check `docs/DEVELOPER_GUIDE.md` for technical details
2. Review functional examples in `examples/` directory
3. Open a GitHub Issue with:
   - Python version
   - Operating system
   - Console error output
   - Steps to reproduce

---

Built by CoreRed Project with discipline.

