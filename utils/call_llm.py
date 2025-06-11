#from google import genai
import os
import logging
import json
from datetime import datetime
import tiktoken

# Configure logging
log_directory = os.getenv("LOG_DIR", "logs")
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(
    log_directory, f"llm_calls_{datetime.now().strftime('%Y%m%d')}.log"
)

# Set up logger
logger = logging.getLogger("llm_logger")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent propagation to root logger
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# Simple cache configuration
cache_file = "llm_cache.json"

# Token counting and truncation utilities
def estimate_tokens(text: str) -> int:
    """Estimate token count for Claude models using tiktoken"""
    try:
        # Use cl100k_base encoding as a reasonable approximation for Claude
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except:
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4

def truncate_context_intelligently(context: str, max_tokens: int = 180000) -> str:
    """
    Intelligently truncate context to fit within token limits.
    Prioritizes keeping file headers and important content.
    """
    current_tokens = estimate_tokens(context)

    if current_tokens <= max_tokens:
        return context

    print(f"Context too large ({current_tokens} tokens > {max_tokens}). Truncating intelligently...")

    # Split into file sections
    file_sections = context.split("--- File Index ")
    if len(file_sections) <= 1:
        # No file structure, just truncate from the end
        target_chars = int(len(context) * (max_tokens / current_tokens))
        truncated = context[:target_chars]
        truncated += "\n\n[... Content truncated due to size limits ...]"
        return truncated

    # Keep the first part (before any files)
    result = file_sections[0]
    remaining_tokens = max_tokens - estimate_tokens(result)

    # Process file sections, prioritizing smaller files and important file types
    file_data = []
    for i, section in enumerate(file_sections[1:], 1):
        if not section.strip():
            continue

        # Extract file info
        lines = section.split('\n', 2)
        if len(lines) < 2:
            continue

        header = "--- File Index " + lines[0]
        content = '\n'.join(lines[1:]) if len(lines) > 1 else ""

        # Calculate priority (smaller files and important extensions get higher priority)
        file_size = len(content)
        file_path = lines[0].split(': ', 1)[1] if ': ' in lines[0] else ""

        # Priority scoring
        priority = 1000000 - file_size  # Smaller files first

        # Boost priority for important file types
        important_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.h', '.md', '.yml', '.yaml', '.json', '.rs', '.c', '.cpp']
        if any(file_path.lower().endswith(ext) for ext in important_extensions):
            priority += 500000

        # Boost priority for config/main files
        important_names = ['main', 'index', 'config', 'setup', '__init__', 'app']
        if any(name in file_path.lower() for name in important_names):
            priority += 300000

        file_data.append((priority, header, content, estimate_tokens(header + content)))

    # Sort by priority (highest first)
    file_data.sort(key=lambda x: x[0], reverse=True)

    # Add files until we hit the token limit
    included_files = 0
    total_files = len(file_data)

    for priority, header, content, tokens in file_data:
        if remaining_tokens - tokens > 1000:  # Keep some buffer
            result += header + content
            remaining_tokens -= tokens
            included_files += 1
        else:
            # Try to include just the header and a truncated version
            header_tokens = estimate_tokens(header)
            if remaining_tokens - header_tokens > 500:
                available_for_content = remaining_tokens - header_tokens - 100
                if available_for_content > 0:
                    # Truncate content to fit
                    content_chars = int(len(content) * (available_for_content / estimate_tokens(content)))
                    truncated_content = content[:content_chars] + "\n[... File content truncated ...]"
                    result += header + truncated_content
                    included_files += 1
            break

    if included_files < total_files:
        result += f"\n\n[... {total_files - included_files} additional files truncated due to size limits ...]"

    return result


# By default, we Google Gemini 2.5 pro, as it shows great performance for code understanding
""" def call_llm(prompt: str, use_cache: bool = True) -> str:
    # Log the prompt
    logger.info(f"PROMPT: {prompt}")

    # Check cache if enabled
    if use_cache:
        # Load cache from disk
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except:
                logger.warning(f"Failed to load cache, starting with empty cache")

        # Return from cache if exists
        if prompt in cache:
            logger.info(f"RESPONSE: {cache[prompt]}")
            return cache[prompt]

    # # Call the LLM if not in cache or cache disabled
    # client = genai.Client(
    #     vertexai=True,
    #     # TODO: change to your own project id and location
    #     project=os.getenv("GEMINI_PROJECT_ID", "your-project-id"),
    #     location=os.getenv("GEMINI_LOCATION", "us-central1")
    # )

    # You can comment the previous line and use the AI Studio key instead:
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY", ""),
    )
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro-exp-03-25")
    # model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-04-17")
    
    response = client.models.generate_content(model=model, contents=[prompt])
    response_text = response.text

    # Log the response
    logger.info(f"RESPONSE: {response_text}")

    # Update cache if enabled
    if use_cache:
        # Load cache again to avoid overwrites
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except:
                pass

        # Add to cache and save
        cache[prompt] = response_text
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    return response_text """


# # Use Azure OpenAI
# def call_llm(prompt, use_cache: bool = True):
#     from openai import AzureOpenAI

#     endpoint = "https://<azure openai name>.openai.azure.com/"
#     deployment = "<deployment name>"

#     subscription_key = "<azure openai key>"
#     api_version = "<api version>"

#     client = AzureOpenAI(
#         api_version=api_version,
#         azure_endpoint=endpoint,
#         api_key=subscription_key,
#     )

#     r = client.chat.completions.create(
#         model=deployment,
#         messages=[{"role": "user", "content": prompt}],
#         response_format={
#             "type": "text"
#         },
#         max_completion_tokens=40000,
#         reasoning_effort="medium",
#         store=False
#     )
#     return r.choices[0].message.content

# Use Anthropic Claude Sonnet 4 Extended Thinking
def call_llm(prompt: str, use_cache: bool = True) -> str:
    from anthropic import Anthropic

    prompt_tokens = estimate_tokens(prompt)
    max_context_tokens = 150000  # Much more conservative limit to account for additional prompt content

    if prompt_tokens > max_context_tokens:
        print(f"Prompt too large ({prompt_tokens} tokens), truncating to {max_context_tokens} tokens...")
        prompt = truncate_context_intelligently(prompt, max_context_tokens)
        prompt_tokens = estimate_tokens(prompt)
        print(f"Truncated prompt size: {prompt_tokens} tokens")

    # Log the prompt (truncated version for very long prompts)
    log_prompt = prompt if len(prompt) < 10000 else prompt[:5000] + "\n[... PROMPT TRUNCATED FOR LOG ...]" + prompt[-2000:]
    logger.info(f"PROMPT: {log_prompt}")

    # Check cache if enabled
    if use_cache:
        # Load cache from disk
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except:
                logger.warning(f"Failed to load cache, starting with empty cache")

        # Return from cache if exists
        if prompt in cache:
            logger.info(f"RESPONSE: {cache[prompt]}")
            return cache[prompt]

    # Call Claude API with error handling for token limits
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",  # Using the latest stable Claude model
            max_tokens=8192,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        response_text = response.content[0].text
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        if "prompt is too long" in str(e):
            # If still too long, try more aggressive truncation
            print("Prompt still too long, applying more aggressive truncation...")
            aggressive_limit = max_context_tokens // 3  # Much more aggressive truncation
            print(f"Attempting more aggressive truncation to {aggressive_limit} tokens...")
            prompt = truncate_context_intelligently(prompt, aggressive_limit)
            truncated_tokens = estimate_tokens(prompt)
            print(f"Aggressively truncated prompt size: {truncated_tokens} tokens")
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=8192,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                response_text = response.content[0].text
            except Exception as e2:
                # If it still fails, raise a custom error to break the retry loop
                raise RuntimeError(f"Failed to truncate prompt sufficiently. Original error: {e}. Second attempt error: {e2}")
        else:
            raise e

    # Log the response
    logger.info(f"RESPONSE: {response_text}")

    # Update cache if enabled
    if use_cache:
        # Load cache again to avoid overwrites
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except:
                pass

        # Add to cache and save
        cache[prompt] = response_text
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    return response_text

# # Use OpenAI o1
# def call_llm(prompt, use_cache: bool = True):
#     from openai import OpenAI
#     client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-api-key"))
#     r = client.chat.completions.create(
#         model="o1",
#         messages=[{"role": "user", "content": prompt}],
#         response_format={
#             "type": "text"
#         },
#         reasoning_effort="medium",
#         store=False
#     )
#     return r.choices[0].message.content

# Use OpenRouter API
# def call_llm(prompt: str, use_cache: bool = True) -> str:
#     import requests
#     # Log the prompt
#     logger.info(f"PROMPT: {prompt}")

#     # Check cache if enabled
#     if use_cache:
#         # Load cache from disk
#         cache = {}
#         if os.path.exists(cache_file):
#             try:
#                 with open(cache_file, "r", encoding="utf-8") as f:
#                     cache = json.load(f)
#             except:
#                 logger.warning(f"Failed to load cache, starting with empty cache")

#         # Return from cache if exists
#         if prompt in cache:
#             logger.info(f"RESPONSE: {cache[prompt]}")
#             return cache[prompt]

#     # OpenRouter API configuration
#     api_key = os.getenv("OPENROUTER_API_KEY", "")
#     model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
    
#     headers = {
#         "Authorization": f"Bearer {api_key}",
#     }

#     data = {
#         "model": model,
#         "messages": [{"role": "user", "content": prompt}]
#     }

#     response = requests.post(
#         "https://openrouter.ai/api/v1/chat/completions",
#         headers=headers,
#         json=data
#     )

#     if response.status_code != 200:
#         error_msg = f"OpenRouter API call failed with status {response.status_code}: {response.text}"
#         logger.error(error_msg)
#         raise Exception(error_msg)
#     try:
#         response_text = response.json()["choices"][0]["message"]["content"]
#     except Exception as e:
#         error_msg = f"Failed to parse OpenRouter response: {e}; Response: {response.text}"
#         logger.error(error_msg)        
#         raise Exception(error_msg)
    

#     # Log the response
#     logger.info(f"RESPONSE: {response_text}")

#     # Update cache if enabled
#     if use_cache:
#         # Load cache again to avoid overwrites
#         cache = {}
#         if os.path.exists(cache_file):
#             try:
#                 with open(cache_file, "r", encoding="utf-8") as f:
#                     cache = json.load(f)
#             except:
#                 pass

#         # Add to cache and save
#         cache[prompt] = response_text
#         try:
#             with open(cache_file, "w", encoding="utf-8") as f:
#                 json.dump(cache, f)
#         except Exception as e:
#             logger.error(f"Failed to save cache: {e}")

#     return response_text

if __name__ == "__main__":
    test_prompt = "Hello, how are you?"

    # First call - should hit the API
    print("Making call...")
    response1 = call_llm(test_prompt, use_cache=False)
    print(f"Response: {response1}")
