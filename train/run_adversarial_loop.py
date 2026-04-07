import argparse
import subprocess
import sys
import os

def run_command(cmd):
    print(f"[Loop] Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Error] Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run Adversarial Learning Loop")
    parser.add_argument("--rounds", type=int, default=3, help="Total number of rounds to run")
    parser.add_argument("--config", type=str, default="configs/adversarial.yaml", help="Path to configuration file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    python_exec = sys.executable
    script_path = "adversarial-learning.py"
    
    if not os.path.exists(script_path):
        print(f"[Error] Script {script_path} not found in current directory. Please run this script from the 'train' folder.")
        sys.exit(1)

    print(f"Starting Adversarial Learning Loop for {args.rounds} rounds...")
    
    for r in range(1, args.rounds + 1):
        print(f"\n{'='*20} Round {r} {'='*20}")
        
        # 1. Train Defender
        print(f"\n--- Training Defender (Round {r}) ---")
        cmd_def = f"{python_exec} {script_path} --config {args.config} --device {args.device} --seed {args.seed} --round {r} --side Def"
        run_command(cmd_def)
        
        # 2. Train Attacker
        print(f"\n--- Training Attacker (Round {r}) ---")
        cmd_att = f"{python_exec} {script_path} --config {args.config} --device {args.device} --seed {args.seed} --round {r} --side Att"
        run_command(cmd_att)
        
    print("\n[Success] All rounds completed!")

if __name__ == "__main__":
    main()
