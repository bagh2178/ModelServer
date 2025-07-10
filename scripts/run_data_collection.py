#!/usr/bin/env python3
import os
import time
import subprocess
import signal
import glob
import shutil

def main():
    episode_index = 0 # Default starting index
    base_data_dir = "/home/tl/yh/data/pnp1*5" # Define base data directory
    dataset_id = os.path.basename(base_data_dir)

    # Prompt user for starting episode index
    start_index_str = input("Enter the starting episode index (default 0): ")
    if start_index_str.isdigit():
        try:
            episode_index = int(start_index_str)
            if episode_index < 0:
                print("Invalid index. Starting from 0.")
                episode_index = 0
        except ValueError:
             print("Invalid input. Starting from 0.")
             episode_index = 0 # Keep default if conversion fails
    elif start_index_str: # If input is not empty but not digits
        print("Invalid input. Starting from 0.")
        # episode_index remains 0
    # If input is empty, episode_index remains 0

    print(f"Starting initial data collection for Episode {episode_index} in 3 seconds...") # Update initial message
    time.sleep(3) # Add initial 3-second pause
    
    while True:
        episode_data_dir = os.path.join(base_data_dir, f"episode_{episode_index}")
        print(f"\n--- Starting data collection for Episode {episode_index} ---")
        print(f"Target directory: {episode_data_dir}")
        print("Press Enter to stop the current collection.")
        
        os.makedirs(base_data_dir, exist_ok=True)

        if os.name == 'posix':  # Linux/macOS
            process = subprocess.Popen(
                ["python", "iDP3_data_collect.py", "--dataset_id", dataset_id, "--episode_index", str(episode_index)],
                preexec_fn=os.setsid
            )
        else:  # Windows
            process = subprocess.Popen(
                ["python", "iDP3_data_collect.py", "--dataset_id", dataset_id, "--episode_index", str(episode_index)],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        
        input()
        
        if os.name == 'posix':  # Linux/macOS
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:  # Windows
            process.send_signal(signal.CTRL_BREAK_EVENT)
            process.terminate()
        
        print(f"Data collection for Episode {episode_index} stopped.")
        
        while True:
            choice = input("Press 'y' to start next collection, 'n' to delete current data, or Enter to exit: ").lower()
            
            if choice == 'n':
                if os.path.exists(episode_data_dir):
                    try:
                        shutil.rmtree(episode_data_dir)
                        print(f"Deleted directory: {episode_data_dir}")
                    except Exception as e:
                        print(f"Error deleting directory {episode_data_dir}: {e}")
                else:
                    print(f"Directory not found, nothing to delete: {episode_data_dir}")
                print(f"Attempting to recollect data for Episode {episode_index}.")
                print("Pausing for 3 seconds before retry...")
                time.sleep(3)
                break
            elif choice == 'y':
                print(f"Starting next collection in 3 seconds...")
                time.sleep(3)
                episode_index += 1
                break
            elif choice == '':
                print("Exiting data collection.")
                return
            else:
                print("Invalid input. Please try again.")

if __name__ == "__main__":
    main() 