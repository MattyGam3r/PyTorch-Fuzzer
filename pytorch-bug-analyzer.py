import os
import requests
import re
import random
import time
import json
import subprocess
import tempfile
import sys
import csv
from datetime import datetime
from openai import OpenAI
import torch

# Set up API keys
github_token = os.getenv("github_token")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Global variables for statistics
total_programs = 0
valid_programs = 0
buggy_programs = 0
api_coverage = set()
program_recalls = 0
start_time = None

def update_statistics(runtime):
    file_exists = os.path.isfile("statistics.csv")
    
    with open("statistics.csv", "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Total programs", "Valid programs", "Buggy programs", "API coverage", "Program recalls", "Runtime (seconds)"])
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_programs,
            valid_programs,
            buggy_programs,
            f"{len(api_coverage)}/{len(dir(torch))}",
            program_recalls,
            round(runtime.total_seconds(), 2)
        ])


def save_buggy_program(code, index):
    if not os.path.exists("buggy-programs"):
        os.makedirs("buggy-programs")
    filename = f"buggy_program_{index}.py"
    clean_code = clean_code_snippet(code)
    with open(os.path.join("buggy-programs", filename), "w") as f:
        f.write(clean_code)
    print(f"Buggy program saved as: {filename}")


def save_buggy_program(code, index):
    if not os.path.exists("buggy-programs"):
        os.makedirs("buggy-programs")
    filename = f"buggy_program_{index}.py"
    clean_code = clean_code_snippet(code)
    with open(os.path.join("buggy-programs", filename), "w") as f:
        f.write(clean_code)
    print(f"Buggy program saved as: {filename}")



def check_github_token():
    if not github_token:
        print("Error: GitHub token not found. Please set the github_token environment variable.")
        print("You can create a token at https://github.com/settings/tokens")
        print("Ensure the token has the 'public_repo' scope to read public repositories.")

def fetch_pytorch_issues(page=1, per_page=100):
    url = f"https://api.github.com/repos/pytorch/pytorch/issues"
    headers = {
        "User-Agent": "MattyGam3r",
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    params = {
        "state": "all",
        "page": page,
        "per_page": per_page,
        "labels": "bug"  # Only fetch issues labeled as bugs
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print(f"Error fetching issues: {response.status_code}")
        print(f"Response content: {response.text}")
        return None
    return response.json()

def extract_code_snippets(issue_body):
    if not isinstance(issue_body, str):
        print(f"Unexpected issue_body type: {type(issue_body)}")
        return []
    code_blocks = re.findall(r'```python(.*?)```', issue_body, re.DOTALL)
    return [block.strip() for block in code_blocks]

def clean_code_snippet(code):
    # Remove ```python from the start and ``` from the end
    code = re.sub(r'^```python\s*', '', code)
    code = re.sub(r'\s*```$', '', code)
    return code.strip()

def analyze_pytorch_bug(code):
    clean_code = clean_code_snippet(code)
    
    if not clean_code:
        print("Error: No valid code snippet provided for analysis.")
        return None

    prompt = f"""
Analyze the following PyTorch code snippet from a GitHub issue for bugs, particularly focusing on API misuse:

{clean_code}

Provide a concise explanation of the bug and the PyTorch API or concept it relates to.
"""
    try:
        response = client.chat.completions.create(
            model="nousresearch/hermes-3-llama-3.1-405b:free",
            messages=[
                {"role": "system", "content": "You are an expert PyTorch developer tasked with identifying bugs in PyTorch code."},
                {"role": "user", "content": prompt}
            ]
        )
        
        if response and response.choices and len(response.choices) > 0 and response.choices[0].message:
            return response.choices[0].message.content
        else:
            print("Error: Received an unexpected response structure from the API.")
            print(f"Response: {response}")
            return None
    except Exception as e:
        print(f"Error occurred while analyzing the PyTorch bug: {str(e)}")
        return None

def generate_new_buggy_program(bug_analysis):
    prompt = f"""
Based on this bug analysis in a PyTorch program:

{bug_analysis}

Create a new, complete, and runnable PyTorch program that is likely to produce different results on CPU vs GPU. Follow these guidelines:

1. The program should be complete and runnable (import necessary libraries, create sample data if needed).
2. Focus on operations that might have precision differences between CPU and GPU, such as:
   - Using float32 vs float64 datatypes
   - Operations involving very large or very small numbers
   - Reductions (sum, mean) over large tensors
   - Complex mathematical operations (exp, log, trigonometric functions)
   - Operations that might be affected by different threading models on CPU vs GPU
3. Try to use PyTorch functions that have known differences in implementation between CPU and GPU.
4. The bug should be subtle and not immediately obvious.
5. The program MUST NOT have any syntax errors and should run without crashing.
6. Include comments explaining the intended behavior and where the potential CPU-GPU difference might occur.
7. Keep the program under 30 lines of code.
8. Ensure the program runs without crashing on both CPU and GPU, but may produce different results.
9. Add print statements to show the output, which will be compared between CPU and GPU runs.

Provide only the complete code, without any additional explanations.
"""
    try:
        response = client.chat.completions.create(
            model="nousresearch/hermes-3-llama-3.1-405b:free",
            messages=[
                {"role": "system", "content": "You are a PyTorch developer tasked with creating example programs that might produce different results on CPU vs GPU."},
                {"role": "user", "content": prompt}
            ]
        )
        
        if response and response.choices and len(response.choices) > 0 and response.choices[0].message:
            generated_code = response.choices[0].message.content
            return clean_code_snippet(generated_code)
        else:
            print("Error: Received an unexpected response structure from the API.")
            print(f"Response: {response}")
            return None
    except Exception as e:
        print(f"Error occurred while generating new buggy program: {str(e)}")
        return None


def get_random_buggy_snippet():
    all_snippets = []
    page = 1
    while len(all_snippets) < 50:  # Fetch until we have at least 50 snippets or no more issues
        issues = fetch_pytorch_issues(page)
        if issues is None:
            break
        if not isinstance(issues, list):
            print(f"Unexpected issues type: {type(issues)}")
            print(f"Issues content: {json.dumps(issues, indent=2)}")
            break
        for issue in issues:
            if isinstance(issue, dict) and 'body' in issue:
                snippets = extract_code_snippets(issue['body'])
                all_snippets.extend(snippets)
            else:
                print(f"Unexpected issue structure: {issue}")
        if not issues:  # No more issues to fetch
            break
        page += 1
        time.sleep(1)  # Be nice to GitHub API
    return random.choice(all_snippets) if all_snippets else None



def execute_python_code(code):
    global total_programs, valid_programs, buggy_programs, api_coverage

    clean_code = clean_code_snippet(code)

    # Add seed setting to the beginning of the code
    seed = random.randint(0, 2**32 - 1)
    seeded_code = f"""
import random
import numpy as np
import torch

random.seed({seed})
np.random.seed({seed})
torch.manual_seed({seed})
if torch.cuda.is_available():
    torch.cuda.manual_seed_all({seed})

{clean_code}
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(seeded_code)
        temp_file_path = temp_file.name

    try:
        python_executable = sys.executable
        
        # Run on CPU
        result_cpu = subprocess.run([python_executable, temp_file_path], capture_output=True, text=True, timeout=10, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))
        output_cpu = result_cpu.stdout
        error_cpu = result_cpu.stderr

        # Run on GPU (if available)
        if torch.cuda.is_available():
            result_gpu = subprocess.run([python_executable, temp_file_path], capture_output=True, text=True, timeout=10, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
            output_gpu = result_gpu.stdout
            error_gpu = result_gpu.stderr
        else:
            output_gpu = output_cpu
            error_gpu = error_cpu

        # Check for syntax errors
        if "SyntaxError" in error_cpu or "SyntaxError" in error_gpu:
            return None, "Syntax error detected", None, None, "Program has syntax errors."

        # Check for TypeErrors
        if "TypeError" in error_cpu or "TypeError" in error_gpu:
            return None, "TypeError detected", None, None, "Program has a TypeError and needs to be regenerated."

        # Check for different results between CPU and GPU
        if output_cpu != output_gpu or error_cpu != error_gpu:
            buggy_programs += 1
            save_buggy_program(seeded_code, buggy_programs)
            explanation = "Program produced different results on CPU vs GPU."
            explanation += f"\nCPU Output: {output_cpu}\nCPU Error: {error_cpu}\nGPU Output: {output_gpu}\nGPU Error: {error_gpu}"
            return output_cpu, error_cpu, output_gpu, error_gpu, explanation

        # If we've reached here, the program is valid (same output on CPU and GPU)
        valid_programs += 1

        # Update API coverage
        for line in code.split('\n'):
            if 'torch.' in line:
                api_call = line.split('torch.')[1].split('(')[0]
                api_coverage.add(api_call)

        return output_cpu, error_cpu, output_gpu, error_gpu, "Program executed successfully without bugs."

    except subprocess.TimeoutExpired:
        explanation = "Execution timed out after 10 seconds"
        return "Execution timed out after 10 seconds", "", "", "", explanation
    finally:
        os.unlink(temp_file_path)


def main():
    global program_recalls, total_programs, valid_programs, buggy_programs, start_time

    print("Welcome to the Automated PyTorch Bug Recreator!")
    start_time = datetime.now()

    check_github_token()

    print("\nFetching a random buggy snippet from PyTorch's GitHub Issues...")
    buggy_snippet = get_random_buggy_snippet()

    if not buggy_snippet:
        print("Unable to fetch a buggy snippet from GitHub Issues.")
        print("Please check your internet connection and GitHub token permissions.")
        return

    print("\nAnalyzing the buggy snippet...")
    bug_analysis = analyze_pytorch_bug(buggy_snippet)
    
    if bug_analysis is None:
        print("Failed to analyze the buggy snippet. Exiting the program.")
        return

    print(f"\nBug analysis:\n{bug_analysis}")

    print("\nNow, generating new programs that recreate similar bugs with different API calls:")
    while buggy_programs < 10:
        total_programs += 1
        print(f"\nProgram {total_programs}:")
        attempts = 0
        max_attempts = 3
        while attempts < max_attempts:
            new_program = generate_new_buggy_program(bug_analysis)
            if new_program is None:
                print("Failed to generate a new program. Retrying...")
                attempts += 1
                program_recalls += 1
                continue
            
            print("Code:")
            print(new_program)
            
            print("\nExecuting the program:")
            output_cpu, error_cpu, output_gpu, error_gpu, explanation = execute_python_code(new_program)
            
            if explanation == "Program has syntax errors." or explanation == "Program has a TypeError and needs to be regenerated.":
                print(explanation + " Regenerating...")
                attempts += 1
                program_recalls += 1
            elif explanation:
                print("Program executed successfully and contains bugs:")
                print(explanation)
                break
            else:
                print("Program executed successfully without bugs.")
                break

        if attempts == max_attempts:
            print(f"Failed to generate a valid program after {max_attempts} attempts. Moving to the next program.")

    end_time = datetime.now()
    runtime = end_time - start_time
    update_statistics(runtime)

    print("\nFinal Statistics for this run:")
    with open("statistics.csv", "r") as f:
        lines = f.readlines()
        if len(lines) > 1:
            print(lines[0].strip())  # Print header
            print(lines[-1].strip())  # Print last line (current run)

    print(f"\nTotal runtime: {runtime}")

if __name__ == "__main__":
    main()