import re
import ast
import subprocess
from pathlib import Path
import pandas as pd
import os
import sys

def parse_bash_value(value_str):
    """Parse a bash value into a Python value."""
    # Remove any surrounding quotes
    value_str = value_str.strip().strip('"\'')
    
    # Handle Python expressions
    if value_str.startswith('$(') and value_str.endswith(')'):
        # Extract the Python code
        python_code = value_str[2:-1]
        if python_code.startswith('python3 -c "print('):
            # Extract the actual Python expression
            expr = python_code[len('python3 -c "print('):-2]
            try:
                return ast.literal_eval(expr)
            except:
                return expr
        return value_str
    
    # Handle arrays
    if value_str.startswith('(') and value_str.endswith(')'):
        # Convert bash array to Python list
        items = value_str[1:-1].split()
        return [ast.literal_eval(item) if item.isdigit() else item.strip('"\'') for item in items]
    
    # Handle numbers
    if value_str.isdigit():
        return int(value_str)
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Handle booleans
    if value_str.lower() in ('true', 'false'):
        return value_str.lower() == 'true'
    
    return value_str

def load_bash_config(bash_script_path):
    """Load variables from a bash script into a Python dictionary."""
    config = {}
    
    # Read the bash script
    with open(bash_script_path, 'r') as f:
        content = f.read()
    
    # Find all variable assignments
    # This regex matches: VAR=value or VAR="value" or VAR='value' or VAR=(value1 value2)
    pattern = r'^([A-Za-z0-9_]+)=(.*?)(?=\n[A-Za-z0-9_]+=|$)'
    matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
    
    for match in matches:
        var_name = match.group(1)
        var_value = match.group(2).strip()
        
        # Skip commented lines
        if var_value.startswith('#'):
            continue
            
        # Parse the value
        try:
            config[var_name] = parse_bash_value(var_value)
        except Exception as e:
            print(f"Warning: Could not parse {var_name}={var_value}: {e}")
            config[var_name] = var_value
    
    return config


def load_data_normalized(input_file,config):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File does not exist: {input_file}")
    columns = ["date", "time", "generation"] 
    columns += [f"o{i+1}" for i,_ in enumerate(config['OBJECTIVES'])]
    df = pd.read_csv(input_file,sep=" ",header=None)
    df.columns = columns
    return df

def load_data_merged_hybrids(input_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File does not exist: {input_file}")
    df = pd.read_csv(input_file,sep=",")
    return df

def load_data_hybrids(input_file,config,seq = 77):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File does not exist: {input_file}")
    columns = ["date", "time", "pf","generation"] 
    columns += [f"o{i+1}" for i,_ in enumerate(config['OBJECTIVES'])]
    pairs = [f"tt{i}" for i in range(seq)]
    columns += pairs    
    df = pd.read_csv(input_file,sep=" ",header=None)
    df.columns = columns
    return df

if __name__ == "__main__":
    # Get the path to script_constants.sh
    script_path = Path("script_constants.sh")
    
    # Load the configuration
    config = load_bash_config(script_path)
    
    # Print the configuration in a Python-friendly format
    print("Configuration loaded. You can access it in your notebook like this:")
    print("\nfrom load_config import load_bash_config")
    print("config = load_bash_config('script_constants.sh')")
    print("\nAvailable variables:")
    for key in sorted(config.keys()):
        print(f"config['{key}'] = {config[key]}") 