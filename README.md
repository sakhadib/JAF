# JAF Story Generation Tool

A multi-threaded story generation tool using OpenRouter API with support for thinking models.

## Features

✨ **Key Features**
- 🧠 **Thinking Model Support**: Automatically detects and leverages reasoning-enabled models
- ⚡ **Multi-threaded Processing**: Concurrent execution with configurable thread count (default: 5)
- 📊 **Smart Deduplication**: Prevents duplicate story generation by tracking processed prompts
- 🎨 **Colorful Console Output**: Minimal, clean progress reporting without progress bars
- 💾 **Token Tracking**: Stores thinking tokens and reasoning metrics when available
- 🔄 **Resume-friendly**: Stop and restart without regenerating already-completed stories

## Requirements

- Python 3.8+
- `python-dotenv` (for API key management)
- `requests` (for API calls)
- OpenRouter API key

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
source .venv/bin/activate      # macOS/Linux

# Install packages
pip install python-dotenv requests
```

### 2. Configure API Key

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_api_key_here
```

Get your API key from [openrouter.ai](https://openrouter.ai)

### 3. Prepare Prompts

Place your prompts in `prompts.json` with this structure:

```json
[
  {
    "prompt_id": "P001",
    "culture": "Bengali",
    "story_type": "Origin Story",
    "prompt": "Your prompt text here...",
    "other_fields": "will be preserved in output"
  },
  ...
]
```

## Usage

### Basic Usage

```bash
python run.py --model deepseek/deepseek-r1
```

### With Custom Thread Count

```bash
python run.py --model openai/gpt-5.2 --threads 10
```

### With Custom Prompts File

```bash
python run.py --model deepseek/deepseek-r1 --prompts custom_prompts.json
```

### Full Example

```bash
python run.py --model deepseek/deepseek-r1 --threads 5 --prompts prompts.json
```

## Thinking Models

The script automatically detects models that support thinking/reasoning:

**Supported Models (examples)**
- `deepseek/deepseek-r1` ✓ Thinking support
- `openai/o1` ✓ Thinking support  
- `openai/o3` ✓ Thinking support
- `qwen/qwq-32b` ✓ Thinking support
- `anthropic/claude-opus` ✓ Thinking support
- `google/gemini-2.5-flash` ✓ Thinking support

For other models, check [OpenRouter models list](https://openrouter.ai/models)

## Output Structure

Results are saved in `results/<model_name>/` as JSON files:

```json
{
  "prompt_id": "P001",
  "culture": "Bengali",
  "prompt": "...",
  "story": "Generated story text here...",
  "thought": "Internal reasoning chain (if model supports thinking)",
  "thinking_tokens": 1500,
  "prompt_tokens": 2100,
  "completion_tokens": 850,
  ...
}
```

### Result Fields

- `prompt_id`: Original prompt ID
- `prompt`: The original prompt text
- `story`: Generated story content
- `thought`: Internal reasoning traces (thinking models only)
- `thinking_tokens`: Tokens used for reasoning (if available)
- `prompt_tokens`: Input tokens
- `completion_tokens`: Output tokens
- All original prompt fields are preserved

## Console Output Example

```
============================================================
Loading Prompts from prompts.json
============================================================

✓ Loaded 10 prompts

============================================================
Story Generation
============================================================

ℹ Model: deepseek/deepseek-r1
ℹ Thinking support: Yes
ℹ Processing: 8 prompts
ℹ Threads: 5
ℹ Output: results/deepseek_deepseek-r1

✓ [T1] P001 → P001.json
✓ [T2] P002 → P002.json
✓ [T1] P003 → P003.json
⚠ [T3] Failed to generate: P004
...

============================================================
Summary
============================================================

✓ Completed: 7
⚠ Failed: 1
ℹ Time elapsed: 45.3s (6.5s per prompt)
ℹ Results saved to: results/deepseek_deepseek-r1
```

## Features Explained

### Thinking Models Support

When you use a thinking model (detected by keywords like "thinking", "r1", "o1", "o3", "opus", "qwq"), the script:

1. Automatically enables reasoning in the API request
2. Captures internal reasoning chains
3. Stores thinking tokens separately
4. Preserves complete reasoning for analysis

### Deduplication

The script checks existing results before processing:
- Reads all `.json` files in the results directory
- Extracts prompt IDs from filenames
- Skips already-processed prompts
- Allows safe stop/restart without duplicate generation

### Multi-threading

- Default: 5 worker threads
- Each thread independently generates stories
- Thread ID shown in progress (e.g., `[T1]`, `[T2]`)
- Graceful shutdown on completion
- No duplicate story generation even with multiple threads

## Troubleshooting

### "OPENROUTER_API_KEY not found"
- Create `.env` file in project root
- Add your API key: `OPENROUTER_API_KEY=sk-or-...`
- Ensure `.env` is in the same directory as `run.py`

### "No choices in response"
- API rate limit may have been hit
- Try again with reduced threads: `--threads 2`
- Check API key validity

### "Timeout generating story"
- Model might be slow or API is overwhelmed
- Try with fewer threads or different model
- Default timeout is 120 seconds

### "All prompts already processed"
- Delete unwanted results from `results/` directory
- Or use a different model ID
- Results are stored separately per model

## Model Selection Tips

**For fast results:**
```bash
python run.py --model gpt-4o-mini
```

**For deep reasoning:**
```bash
python run.py --model deepseek/deepseek-r1 --threads 2
```

**For balanced performance:**
```bash
python run.py --model gpt-5.2 --threads 5
```

**For free models:**
```bash
python run.py --model deepseek/deepseek-r1-0528:free
```

## API Costs

Thinking models typically cost more due to internal reasoning. Check OpenRouter pricing:
- Prompt tokens: varies by model
- Completion tokens: usually 5-10x prompt cost
- Thinking/reasoning tokens: charged in completion

## Advanced Usage

### Resume Failed Generation

If generation fails, you can safely run again:

```bash
python run.py --model deepseek/deepseek-r1 --threads 5
```

Only unprocessed prompts will be generated.

### Switch Models

Generate same prompts with different models:

```bash
python run.py --model openai/o1
# Results go to: results/openai_o1/

python run.py --model deepseek/deepseek-r1
# Results go to: results/deepseek_deepseek-r1/
```

## License

MIT - Use freely for research and development

## Support

For issues with:
- **OpenRouter API**: See [OpenRouter docs](https://openrouter.ai/docs)
- **Thinking models**: Check model-specific documentation
- **This script**: Review the code comments in `run.py`
