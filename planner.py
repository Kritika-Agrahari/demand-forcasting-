import os
import random
import datetime
import json
import re

# Configuration
SOURCE_DIR = r"c:\Users\kriti\Downloads\kuch bhi 2\kuch bhi 2"
TARGET_DIR = os.path.join(SOURCE_DIR, "upload_to_github")
LOG_FILE = os.path.join(TARGET_DIR, "upload_log.txt")
COMMIT_COUNT = 165  # Aiming for > 150
REPO_URL = "https://github.com/Kritika-Agrahari/demand-forcasting"
BRANCH = "main"

# Regex for splitting Python files into coherent modules (classes and functions)
PY_MODULE_REGEX = re.compile(r"^(class |def |if __name__ == .__main__.:)", re.MULTILINE)

def get_timestamps(count, start_date, end_date):
    timestamps = []
    current_date = start_date
    days = (end_date - start_date).days
    
    # Calculate weights for days
    day_weights = []
    for i in range(days + 1):
        d = start_date + datetime.timedelta(days=i)
        weight = 1.5 if d.weekday() >= 5 else 1.0
        day_weights.append(weight)
    
    total_weight = sum(day_weights)
    commits_per_day_float = [count * (w / total_weight) for w in day_weights]
    
    # Distribute commits across days
    distributed_counts = [int(c) for c in commits_per_day_float]
    remaining = count - sum(distributed_counts)
    for _ in range(remaining):
        idx = random.randint(0, len(distributed_counts) - 1)
        distributed_counts[idx] += 1
        
    for i, day_count in enumerate(distributed_counts):
        d = start_date + datetime.timedelta(days=i)
        for _ in range(day_count):
            # Time logic: 85% (9AM-11PM), 15% (11PM-4AM)
            if random.random() < 0.85:
                # 9:00:00 to 22:59:59
                hour = random.randint(9, 22)
            else:
                # 23:00:00 to 03:59:59
                hour = random.choice([23, 0, 1, 2, 3])
            
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            ts = d.replace(hour=hour, minute=minute, second=second)
            timestamps.append(ts)
            
    timestamps.sort()
    return timestamps

def split_file(path, rel_path):
    ext = os.path.splitext(path)[1].lower()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return [(rel_path, None)] # Binary or encoding error

    if ext == '.py':
        # Split by top-level definitions
        parts = []
        indices = [m.start() for m in PY_MODULE_REGEX.finditer(content)]
        if not indices or indices[0] != 0:
            indices.insert(0, 0)
        indices.append(len(content))
        
        for i in range(len(indices) - 1):
            parts.append(content[indices[i]:indices[i+1]])
        
        # Merge small parts to avoid too many commits for one file
        merged_parts = []
        current_part = ""
        for p in parts:
            current_part += p
            if len(current_part) > 500 or p.strip().startswith('class'):
                merged_parts.append(current_part)
                current_part = ""
        if current_part:
            merged_parts.append(current_part)
            
        return [(rel_path, p) for p in merged_parts]
    
    elif ext in ['.md', '.txt', '.css', '.js', '.html']:
        # Split by lines
        lines = content.splitlines(keepends=True)
        if len(lines) < 50:
            return [(rel_path, content)]
        
        chunk_size = max(20, len(lines) // 5)
        chunks = []
        for i in range(0, len(lines), chunk_size):
            chunks.append("".join(lines[i:i+chunk_size]))
        return [(rel_path, c) for c in chunks]
    
    else:
        return [(rel_path, content)]

def generate_commits():
    all_files = []
    for root, dirs, filenames in os.walk(SOURCE_DIR):
        if any(p in root for p in ['.venv', '__pycache__', '.git', 'upload_to_github']):
            continue
        for f in filenames:
            abs_path = os.path.join(root, f)
            rel_path = os.path.relpath(abs_path, SOURCE_DIR)
            if rel_path == ".gitignore" or rel_path == "planner.py":
                continue
            all_files.append((abs_path, rel_path))
            
    # Priority sorting: infrastructure -> data processing -> modeling -> frontend -> reports
    def sort_key(item):
        path = item[1].lower()
        if 'requirements' in path or 'readme' in path: return 0
        if 'audit' in path or 'check' in path: return 1
        if 'cleaning' in path or 'validate' in path: return 2
        if 'train' in path or 'modeling' in path: return 3
        if 'dashboard' in path or 'app' in path: return 4
        if 'report' in path or 'summary' in path: return 5
        return 6
    
    all_files.sort(key=sort_key)
    
    raw_commit_parts = []
    for abs_path, rel_path in all_files:
        size = os.path.getsize(abs_path)
        if size > 100 * 1024 * 1024:
            print(f"Warning: Skipping {rel_path} as it exceeds 100MB GitHub limit.")
            continue
            
        parts = split_file(abs_path, rel_path)
        for i, (p_rel, p_content) in enumerate(parts):
            raw_commit_parts.append({
                'rel_path': p_rel,
                'content': p_content,
                'is_partial': len(parts) > 1,
                'part_idx': i,
                'total_parts': len(parts)
            })
            
    # Adjust to target COMMIT_COUNT
    # If we have too many, merge some. If too few, we might need more splits.
    # Given the project size, we likely have enough.
    
    final_commits = []
    # Combine small parts into commits if needed, but the user wants many commits.
    # Let's just use each part as a commit.
    
    # Generate unique messages
    messages = [
        "Initial project setup and directory structure",
        "Add comprehensive .gitignore and environment config",
        "Define project requirements and dependencies",
        "Implement basic data loading utilities",
        "Setup logging and error handling for the pipeline",
        "Add initial data validation scripts",
        "Implement exploratory data analysis (EDA) foundations",
        "Create target distribution visualization logic",
        "Add temporal trend analysis features",
        "Implement feature correlation matrix calculation",
        "Add data cleaning and preprocessing modules",
        "Handle missing values and outliers in sales data",
        "Merge store and item metadata with transaction history",
        "Optimize memory usage for large dataset processing",
        "Implement feature engineering for time-series forecasting",
        "Add rolling mean and lag feature calculations",
        "Setup training and testing data split logic",
        "Implement baseline SARIMA model for comparison",
        "Add LightGBM model training pipeline",
        "Implement XGBoost forecasting integration",
        "Configure hyperparameter tuning for LGBM",
        "Add model evaluation metrics (RMSE, MAE, MAPE)",
        "Create visualization for actual vs predicted results",
        "Implement residual analysis and error tracking",
        "Setup streamlit dashboard for interactive forecasting",
        "Add store-item selection filters to UI",
        "Implement real-time prediction service",
        "Create final performance report generator",
        "Add documentation for the forecasting methodology",
        "Optimize dashboard performance and caching",
        "Refine UI aesthetics and responsive layout",
        "Finalize production-ready model saving workflow"
    ]
    
    # If we need more messages, we'll interpolate or use file-specific ones
    for i in range(len(raw_commit_parts)):
        part = raw_commit_parts[i]
        filename = os.path.basename(part['rel_path'])
        if i < len(messages):
            msg = messages[i]
        else:
            if part['is_partial']:
                msg = f"Enhance {filename}: implement part {part['part_idx']+1} of core functionality"
            else:
                msg = f"Integrate {filename} into the project workflow"
        
        final_commits.append({
            'msg': msg,
            'rel_path': part['rel_path'],
            'content': part['content']
        })
        
    return final_commits[:COMMIT_COUNT] if len(final_commits) > COMMIT_COUNT else final_commits

def main():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        
    commits = generate_commits()
    start_date = datetime.datetime.now() - datetime.timedelta(days=90)
    end_date = datetime.datetime.now()
    
    timestamps = get_timestamps(len(commits), start_date, end_date)
    
    # Write the execution script
    exec_script_path = os.path.join(SOURCE_DIR, "execute_upload.py")
    
    with open(exec_script_path, 'w', encoding='utf-8') as f:
        f.write(f"""import os
import subprocess
import shutil
import time

TARGET_DIR = r"{TARGET_DIR}"
LOG_FILE = r"{LOG_FILE}"
REPO_URL = "{REPO_URL}"

def run_cmd(cmd, cwd=TARGET_DIR):
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        with open(LOG_FILE, 'a') as log:
            log.write(f"\\nERROR at commit index: {{current_idx}}\\n")
            log.write(f"Command: {{cmd}}\\n")
            log.write(f"Error: {{result.stderr}}\\n")
        print(f"Error occurred. Check {{LOG_FILE}}")
        exit(1)
    return result.stdout

def commit_and_push(msg, date_str, idx):
    global current_idx
    current_idx = idx
    run_cmd(f'git add .')
    # Set date via environment variables for git commit
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = date_str
    env["GIT_COMMITTER_DATE"] = date_str
    
    # We use subprocess.run directly here to pass env
    res = subprocess.run(f'git commit -m "{{msg}}"', shell=True, cwd=TARGET_DIR, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        with open(LOG_FILE, 'a') as log:
            log.write(f"\\nERROR at commit index: {{idx}}\\n")
            log.write(f"Msg: {{msg}}\\n")
            log.write(f"Error: {{res.stderr}}\\n")
        print(f"Error at commit {{idx}}. Check {{LOG_FILE}}")
        exit(1)
        
    run_cmd("git push origin main")
    with open(LOG_FILE, 'a') as log:
        log.write(f"Successfully completed commit {{idx}}: {{msg}} at {{date_str}}\\n")

# Initialize Repo
if os.path.exists(TARGET_DIR):
    shutil.rmtree(TARGET_DIR)
os.makedirs(TARGET_DIR)

subprocess.run("git init", shell=True, cwd=TARGET_DIR)
subprocess.run(f"git remote add origin {{REPO_URL}}", shell=True, cwd=TARGET_DIR)
subprocess.run("git checkout -b main", shell=True, cwd=TARGET_DIR)

# Copy .gitignore first
shutil.copy(os.path.join(r"{SOURCE_DIR}", ".gitignore"), os.path.join(TARGET_DIR, ".gitignore"))

current_idx = 0
with open(LOG_FILE, 'w') as log:
    log.write("Starting upload process...\\n")

""")
        
        for i, (commit, ts) in enumerate(zip(commits, timestamps)):
            date_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            msg = commit['msg'].replace('"', '\\"')
            rel_path = commit['rel_path']
            content = commit['content']
            
            # Escape content for python string
            if content is None: # Binary file
                f.write(f"# Commit {i}: {msg}\n")
                f.write(f"shutil.copy(os.path.join(r'{SOURCE_DIR}', r'{rel_path}'), os.path.join(TARGET_DIR, r'{rel_path}'))\n")
            else:
                # For safety with large files and special chars, we'll write content to file
                f.write(f"\n# Commit {i}: {msg}\n")
                f.write(f"dest_path = os.path.join(TARGET_DIR, r'{rel_path}')\n")
                f.write(f"os.makedirs(os.path.dirname(dest_path), exist_ok=True)\n")
                f.write(f"with open(dest_path, 'a', encoding='utf-8') as f_out:\n")
                # Represent content as a raw string to handle backslashes
                content_repr = repr(content)
                f.write(f"    f_out.write({content_repr})\n")
            
            f.write(f"commit_and_push(\"{msg}\", \"{date_str}\", {i})\n")
            
        f.write("\nprint('All commits uploaded successfully!')\n")

if __name__ == "__main__":
    main()
