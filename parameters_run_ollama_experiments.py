import ollama
import os
import requests
from pathlib import Path
import json
from datetime import datetime
from github import Github

# --- Configuration ---
GITHUB_OWNER = "MahitaBontu"
GITHUB_REPO = "ollama-phishing-study"
LOCAL_PROMPT_DIR = "prompts"
OUTPUT_DIR = "ollama_responses"

# --- Helper Functions ---
def download_prompts(owner, repo, local_dir):
    """Downloads prompt files from a GitHub repository using PyGithub."""
    # If you have a GitHub token, use it for higher rate limits.  Otherwise, it will work without, but may be rate limited.
    if 'GITHUB_TOKEN' in globals() and GITHUB_TOKEN:
        g = Github(GITHUB_TOKEN)
    else:
        g = Github()
    
    try:
        repo = g.get_repo(f"{owner}/{repo}")
    except Exception as e:
        print(f"Error accessing repository: {e}")
        return

    Path(local_dir).mkdir(parents=True, exist_ok=True)
    folders = ["prompts/normal_phishing_prompts", "prompts/jailbreak_phishing_prompts"]

    for folder in folders:
        local_folder = Path(local_dir) / folder
        local_folder.mkdir(parents=True, exist_ok=True)
        try:
            # Get the contents of the folder
            contents = repo.get_contents(folder)
            for file_content in contents:
                if file_content.type == "file" and file_content.name.endswith(".txt"):
                    try:
                        file_data = requests.get(file_content.download_url).text
                        with open(local_folder / file_content.name, "w", encoding='utf-8') as f:
                            f.write(file_data)
                        print(f"Downloaded: {file_content.name} to {local_folder}")
                    except Exception as e:
                        print(f"Error downloading {file_content.name}: {e}")
            print(f"Downloaded prompts from '{folder}' to '{local_folder}'")
        except Exception as e:
            print(f"Error accessing folder {folder}: {e}")
            print(f"Please check if the folder '{folder}' exists in the repository root.")

def run_prompt(model_name, prompt):
    """Runs a prompt on a specified Ollama model."""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ]
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error with model '{model_name}': {e}")
        return None

def save_response(model_name, prompt_type, prompt_file, response_text, output_dir):
    """Saves the model's response to a JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{output_dir}/{model_name.replace(':', '_')}_{prompt_type}_{Path(prompt_file).stem}_{timestamp}.json"
    data = {
        "model": model_name,
        "prompt_type": prompt_type,
        "prompt_file": prompt_file,
        "prompt": Path(prompt_file).read_text(),
        "response": response_text
    }
    with open(output_filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved response to '{output_filename}'")

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Download Prompts
    download_prompts(GITHUB_OWNER, GITHUB_REPO, LOCAL_PROMPT_DIR)

    # 2. Define Smallest Models
    # Using smaller models due to memory constraints
    models = [
        "gemma:2b",  # Smallest model first
        "falcon:1.3b",
        "neural-chat",  # Alternative smaller models
        "mistral:latest",
        "llama2:chat"  # Using base versions instead of 7b
    ]

    # 3. Run Experiments
    for model in models:
        print(f"\n--- Running experiments for model: {model} ---")
        prompt_folders = {
            "normal": Path(LOCAL_PROMPT_DIR) / "normal_phishing_prompts",
            "jailbreak": Path(LOCAL_PROMPT_DIR) / "jailbreak_phishing_prompts"
        }

    for model in ollama:
        top p = 50
        top k = 0.8
        temperature = 0.5
        while prompt is running:
            top p += 0.1
            
        
        
        for prompt_type, folder in prompt_folders.items():
            print(f"\n--- {prompt_type.capitalize()} Prompts ---")
            for prompt_file in sorted(os.listdir(folder)):
                if prompt_file.endswith(".txt"):
                    prompt_path = folder / prompt_file
                    with open(prompt_path, "r") as f:
                        prompt_text = f.read().strip()
                    print(f"\nRunning prompt from: {prompt_path}")
                    response = run_prompt(model, prompt_text)
                    if response:
                        save_response(model, prompt_type, str(prompt_path), response, OUTPUT_DIR)
                    else:
                        print(f"No response received for prompt: {prompt_path}")

    print("\n--- Experiment completed. Responses saved in the 'ollama_responses' directory. ---")
    
    
    print("/n--- Experiment model run time error. Run again or exit the interface before running again. ---")