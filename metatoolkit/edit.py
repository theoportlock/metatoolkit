#!/usr/bin/env python3
import subprocess
import sys
import os

def open_vim_in_new_window(target_file):
    # 1. Get absolute path of the file
    abs_path = os.path.abspath(target_file)

    # 2. Path to your env script (using absolute path for reliability)
    env_script = "/home/tpor598/recovery/env.sh"

    # 3. Create a bash command that:
    #    - Sources the environment (which handles cd, PATH, and venv)
    #    - Launches vim
    bash_logic = f"source {env_script} && vim '{abs_path}'"

    # 4. Use cmd.exe to start a new WSL window executing that logic
    win_cmd = f'cmd.exe /c start wsl bash -c "{bash_logic}"'

    try:
        subprocess.Popen(win_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✔ Sent {os.path.basename(target_file)} to a new window with environment loaded.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./edit.py <filename>")
    else:
        open_vim_in_new_window(sys.argv[1])
