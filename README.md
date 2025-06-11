PocketFlow-Tutorial-Codebase-Knowledge (Modified version for Claude 4)
===

This is a forked and modified repo from the original repo, with the ability to truncate token to work with Claude sonnet 4 model when we input some large repos.

# The original Repo:
https://github.com/The-Pocket/PocketFlow-Tutorial-Codebase-Knowledge


# How to use

1. Clone this repository
   ```bash
   git clone https://github.com/johnnyliao-nzxt/PocketFlow-Tutorial-Codebase-Knowledge-NX.git
   ```

2. Make sure your computer has already installed Python 3
  ```bash
  python --version
  ```

  It should show something like `Python 3.xx.x` or higher.

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Generate your own API keys from [Anthropic Console](https://console.anthropic.com/settings/keys), then copy & paste into .env

5. Copy `.env.sameple` to `.env`, and update `.env` file with your API keys:
```bash
# in .env:
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

6. The LLM in [`utils/call_llm.py`](./utils/call_llm.py) has already modified to use Claude sonnet 4 in this repo. You can change it to other models if you want as the original repo README.md shows.


   You can verify that it is correctly set up by running:
   ```bash
   python utils/call_llm.py
   ```

7. Generate a complete codebase tutorial by running the main script:
    ```bash
    # Analyze a GitHub repository
    python main.py --repo https://github.com/username/repo --include "*.py" "*.js" --exclude "tests/*" --max-size 50000

    # Or, analyze a local directory
    python main.py --dir /path/to/your/codebase --include "*.js" --exclude "*test*"

    # Or, generate a tutorial in Chinese
    python main.py --repo https://github.com/username/repo --language "Chinese"
    ```

    - `--repo` or `--dir` - Specify either a GitHub repo URL or a local directory path (required, mutually exclusive)
    - `-n, --name` - Project name (optional, derived from URL/directory if omitted)
    - `-t, --token` - GitHub token (or set GITHUB_TOKEN environment variable)
    - `-o, --output` - Output directory (default: ./output)
    - `-i, --include` - Files to include (e.g., "`*.py`" "`*.js`")
    - `-e, --exclude` - Files to exclude (e.g., "`tests/*`" "`docs/*`")
    - `-s, --max-size` - Maximum file size in bytes (default: 100KB)
    - `--language` - Language for the generated tutorial (default: "english")
    - `--max-abstractions` - Maximum number of abstractions to identify (default: 10)
    - `--no-cache` - Disable LLM response caching (default: caching enabled)

The application will crawl / read the repository, analyze the codebase structure, generate tutorial content in the specified language, and save the output in the specified directory (default: ./output).