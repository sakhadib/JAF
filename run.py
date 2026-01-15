#!/usr/bin/env python3
"""
Story generation script using OpenRouter API with thinking model support.
Supports multi-threaded prompt processing with progress tracking and deduplication.
"""

import json
import os
import sys
import argparse
import time
import threading
from pathlib import Path
from queue import Queue
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import requests

# Color codes for console output
class Color:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print a colored header."""
    print(f"\n{Color.CYAN}{Color.BOLD}{'='*60}{Color.RESET}")
    print(f"{Color.CYAN}{Color.BOLD}{text}{Color.RESET}")
    print(f"{Color.CYAN}{Color.BOLD}{'='*60}{Color.RESET}\n")

def print_success(text: str):
    """Print success message."""
    print(f"{Color.GREEN}✓ {text}{Color.RESET}")

def print_info(text: str):
    """Print info message."""
    print(f"{Color.CYAN}ℹ {text}{Color.RESET}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Color.YELLOW}⚠ {text}{Color.RESET}")

def print_error(text: str):
    """Print error message."""
    print(f"{Color.RED}✗ {text}{Color.RESET}")

def load_env():
    """Load environment variables from .env file."""
    load_dotenv()
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print_error("OPENROUTER_API_KEY not found in .env file")
        sys.exit(1)
    return api_key

def load_prompts(prompts_file: str) -> list:
    """Load prompts from JSON file."""
    with open(prompts_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_existing_files(results_dir: str) -> set:
    """Get set of existing prompt IDs that have been processed."""
    existing = set()
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                prompt_id = filename.replace('.json', '')
                existing.add(prompt_id)
    return existing

def check_thinking_support(model_id: str, api_key: str) -> bool:
    """Check if a model supports thinking/reasoning features."""
    # Common thinking model patterns
    thinking_keywords = [
        'thinking', 'deepseek-r1', 'qwq', 'claude-opus', 'o1', 'o3', 
        'gpt-5', 'reasoning', 'reflect'
    ]
    model_lower = model_id.lower()
    return any(keyword in model_lower for keyword in thinking_keywords)

def generate_story(
    prompt_obj: Dict[str, Any],
    model_id: str,
    api_key: str,
    thinking_enabled: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Generate a story using OpenRouter API.
    
    Args:
        prompt_obj: The prompt object containing prompt_id and prompt text
        model_id: The model ID to use
        api_key: OpenRouter API key
        thinking_enabled: Whether to enable thinking mode
    
    Returns:
        Response object with story and optional thinking tokens
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/research/jaf",
        "X-Title": "JAF Story Generation",
        "Content-Type": "application/json"
    }
    
    # Prepare the message
    messages = [
        {
            "role": "user",
            "content": prompt_obj['prompt']
        }
    ]
    
    # Prepare request body
    body = {
        "model": model_id,
        "messages": messages,
        "max_tokens": 4000,
        "temperature": 0.7,
    }
    
    # Add thinking/reasoning if supported
    if thinking_enabled and check_thinking_support(model_id, api_key):
        body["reasoning"] = {"enabled": True}
        body["include_reasoning"] = True
    
    try:
        response = requests.post(url, json=body, headers=headers, timeout=120)
        
        if response.status_code != 200:
            print_error(f"API error {response.status_code}: {response.text[:100]}")
            return None
        
        api_response = response.json()
        
        # Extract the response
        if 'choices' not in api_response or len(api_response['choices']) == 0:
            print_error("No choices in response")
            return None
        
        choice = api_response['choices'][0]
        
        # Build result object - copy the original prompt
        result = prompt_obj.copy()
        
        # Extract story from message
        message = choice.get('message', {})
        
        # Check for thinking/reasoning content
        thinking_content = None
        story_content = None
        
        if 'content' in message:
            story_content = message['content']
        
        # Some models return thinking in reasoning_content
        if 'reasoning_content' in message:
            thinking_content = message['reasoning_content']
        
        # Add story and thinking to result
        result['story'] = story_content or ""
        if thinking_content:
            result['thought'] = thinking_content
        
        # Try to extract thinking tokens if available
        if 'usage' in api_response:
            usage = api_response['usage']
            if 'reasoning_tokens' in usage:
                result['thinking_tokens'] = usage['reasoning_tokens']
            if 'prompt_tokens' in usage:
                result['prompt_tokens'] = usage['prompt_tokens']
            if 'completion_tokens' in usage:
                result['completion_tokens'] = usage['completion_tokens']
        
        return result
        
    except requests.exceptions.Timeout:
        print_error(f"Timeout generating story for {prompt_obj['prompt_id']}")
        return None
    except Exception as e:
        print_error(f"Error generating story: {str(e)[:100]}")
        return None

def save_result(result: Dict[str, Any], results_dir: str):
    """Save a single result to a JSON file."""
    os.makedirs(results_dir, exist_ok=True)
    
    prompt_id = result['prompt_id']
    filepath = os.path.join(results_dir, f"{prompt_id}.json")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return filepath

def worker_thread(
    queue: Queue,
    model_id: str,
    api_key: str,
    results_dir: str,
    thread_id: int,
    stats: Dict
):
    """Worker thread function."""
    while True:
        item = queue.get()
        
        if item is None:  # Poison pill to stop thread
            break
        
        prompt_obj = item
        prompt_id = prompt_obj['prompt_id']
        
        try:
            # Generate story
            result = generate_story(
                prompt_obj, 
                model_id, 
                api_key,
                thinking_enabled=True
            )
            
            if result:
                # Save result
                filepath = save_result(result, results_dir)
                stats['completed'] += 1
                
                # Build message with token counts
                msg = f"[T{thread_id}] {prompt_id} → {os.path.basename(filepath)}"
                if 'prompt_tokens' in result and 'completion_tokens' in result:
                    total_tokens = result['prompt_tokens'] + result['completion_tokens']
                    msg += f" ({result['prompt_tokens']}in + {result['completion_tokens']}out = {total_tokens})"
                
                print_success(msg)
            else:
                stats['failed'] += 1
                print_error(f"[T{thread_id}] Failed to generate: {prompt_id}")
        
        except Exception as e:
            stats['failed'] += 1
            print_error(f"[T{thread_id}] Exception: {str(e)[:80]}")
        
        finally:
            queue.task_done()

def main():
    parser = argparse.ArgumentParser(
        description='Generate stories using OpenRouter models with thinking support'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Model ID (e.g., deepseek/deepseek-r1, openai/gpt-5.2)'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=5,
        help='Number of concurrent threads (default: 5)'
    )
    parser.add_argument(
        '--prompts',
        default='prompts.json',
        help='Path to prompts JSON file'
    )
    
    args = parser.parse_args()
    
    # Load API key
    api_key = load_env()
    
    # Load prompts
    print_header(f"Loading Prompts from {args.prompts}")
    if not os.path.exists(args.prompts):
        print_error(f"Prompts file not found: {args.prompts}")
        sys.exit(1)
    
    prompts = load_prompts(args.prompts)
    print_success(f"Loaded {len(prompts)} prompts")
    
    # Setup results directory
    results_dir = f"results/{args.model.replace('/', '_')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Check for existing files
    existing = get_existing_files(results_dir)
    pending_prompts = [p for p in prompts if p['prompt_id'] not in existing]
    
    if existing:
        print_info(f"Found {len(existing)} existing results, skipping...")
    
    if not pending_prompts:
        print_warning("All prompts already processed!")
        print_info(f"Results are in: {results_dir}")
        return
    
    # Check thinking support
    thinking_support = check_thinking_support(args.model, api_key)
    print_header(f"Story Generation")
    print_info(f"Model: {Color.BOLD}{args.model}{Color.RESET}")
    print_info(f"Thinking support: {Color.BOLD}{'Yes' if thinking_support else 'No'}{Color.RESET}")
    print_info(f"Processing: {Color.BOLD}{len(pending_prompts)}{Color.RESET} prompts")
    print_info(f"Threads: {Color.BOLD}{args.threads}{Color.RESET}")
    print_info(f"Output: {Color.BOLD}{results_dir}{Color.RESET}\n")
    
    # Thread-safe statistics
    stats = {'completed': 0, 'failed': 0}
    stats_lock = threading.Lock()
    
    # Create work queue
    queue = Queue()
    for prompt in pending_prompts:
        queue.put(prompt)
    
    # Start worker threads
    threads = []
    start_time = time.time()
    
    for i in range(args.threads):
        t = threading.Thread(
            target=worker_thread,
            args=(queue, args.model, api_key, results_dir, i+1, stats),
            daemon=False
        )
        t.start()
        threads.append(t)
    
    # Wait for queue to be processed with Ctrl+C handling
    try:
        queue.join()
    except KeyboardInterrupt:
        print_warning("\nCtrl+C detected, stopping all threads...")
        # Clear the queue
        while not queue.empty():
            try:
                queue.get_nowait()
            except:
                break
        # Send poison pills to stop threads
        for _ in range(args.threads):
            queue.put(None)
    
    # Send poison pills to stop threads (if not already sent)
    for _ in range(args.threads):
        queue.put(None)
    
    # Wait for all threads to finish
    for t in threads:
        t.join()
    
    elapsed = time.time() - start_time
    
    # Print summary
    print_header("Summary")
    print_success(f"Completed: {stats['completed']}")
    print_warning(f"Failed: {stats['failed']}") if stats['failed'] > 0 else None
    if stats['completed'] > 0:
        avg_time = elapsed / stats['completed']
        print_info(f"Time elapsed: {elapsed:.1f}s ({avg_time:.1f}s per prompt)")
    print_info(f"Results saved to: {Color.BOLD}{results_dir}{Color.RESET}")

if __name__ == '__main__':
    main()
