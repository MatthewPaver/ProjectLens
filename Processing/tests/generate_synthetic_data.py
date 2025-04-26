import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging # Added for potential debug logging

# --- Setup logging (optional, can be useful for debugging generation)
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Helper functions
def random_date(start, end):
    """Generate a random datetime between `start` and `end`"""
    # Ensure start is not after end
    if start > end:
        start = end - timedelta(days=1) # Adjust if start is somehow after end
    # Ensure the range has at least one day
    day_diff = int((end - start).days)
    if day_diff <= 0:
        return start
    return start + timedelta(days=random.randint(0, day_diff))

def messy_percent_complete():
    """Generate percent complete with messy formatting"""
    choices = [
        str(random.randint(0, 100)) + '%',
        random.random(),
        random.randint(0, 100),
        "Complete" if random.random() < 0.1 else None,
        None
    ]
    return random.choice(choices)

def messy_boolean(true_probability=0.5):
    """Generate boolean with messy formatting, controlling True probability."""
    # Determine outcome based on probability
    is_true = random.random() < true_probability
    
    # Select a representation
    if is_true:
        return random.choice(["Yes", "TRUE", 1, True])
    else:
        return random.choice(["No", "FALSE", 0, False, None]) # Include None in False cases

def messy_status():
    """Generate status values"""
    # Bias towards In Progress or Complete in later stages if needed
    return random.choice(["Not Started", "In Progress", "Complete", "IP", None])

def random_risk_flag():
    return random.choice(["Low", "Medium", "High", "H", "L", None])

def generate_task_update_data(task_info, update_num, total_updates):
    """
    Generates data for a *single* task for a specific update phase,
    using existing baseline data and simulating progress.
    """
    task_id = task_info['task_id']
    baseline_start = task_info['baseline_start']
    baseline_end = task_info['baseline_end']

    # --- Simulate Progress ---
    # Actual Start: Likely set early on, keep consistent or slightly adjust
    if 'actual_start' not in task_info or task_info['actual_start'] is None:
        # Set initial actual_start somewhere around baseline_start
        task_info['actual_start'] = baseline_start + timedelta(days=random.randint(-3, 3)) # Reduced variance
    actual_start = task_info['actual_start']

    # --- MODIFIED: Actual Finish Logic for Dynamic Changes ---
    previous_actual_finish = task_info.get('actual_finish', None)
    current_actual_finish = previous_actual_finish # Start with the previous value

    # Decide if a re-evaluation/change might occur this update
    re_evaluate_chance = 0.25 # 25% chance to potentially change the finish date per update
    significant_slip_chance = 0.05 # 5% chance per update for a *large* slip
    correction_chance = 0.03 # 3% chance to correct *slightly* earlier if already slipped

    # Always re-evaluate if not finished yet
    if previous_actual_finish is None or random.random() < re_evaluate_chance:
        
        # 1. Check for significant slip first (higher impact)
        slip_magnitude_days = 0
        if random.random() < significant_slip_chance and update_num > 1:
            slip_magnitude_days = random.randint(10, 30 + update_num * 2) # Adjust magnitude
            # Calculate new finish based on baseline OR previous finish, whichever is later
            base_for_slip = max(baseline_end, previous_actual_finish) if previous_actual_finish else baseline_end
            current_actual_finish = base_for_slip + timedelta(days=slip_magnitude_days)
            # logging.debug(f"Task {task_id} Update {update_num}: Applied significant slip. New Est: {current_actual_finish.strftime('%Y-%m-%d')}")

        # 2. If no significant slip, apply normal progress/minor variation or initial finish
        elif slip_magnitude_days == 0:
            # If never finished, calculate initial finish probability
            if previous_actual_finish is None:
                finish_probability = (update_num / total_updates) * 0.7 # Base chance to finish
                if random.random() < finish_probability:
                    potential_finish = baseline_end + timedelta(days=random.randint(-5, 15)) # Initial variance
                    current_actual_finish = potential_finish
                    # logging.debug(f"Task {task_id} Update {update_num}: Initial finish set. Est: {current_actual_finish.strftime('%Y-%m-%d')}")
            
            # If already finished, add minor random variation (small slips/corrections)
            else:
                # Chance for minor correction
                if random.random() < correction_chance:
                     current_actual_finish = previous_actual_finish - timedelta(days=random.randint(1, 5))
                     # logging.debug(f"Task {task_id} Update {update_num}: Applied minor correction. Est: {current_actual_finish.strftime('%Y-%m-%d')}")
                # More likely to add minor slips
                else:
                    minor_slip_days = random.randint(0, 7) # Small additional slip
                    current_actual_finish = previous_actual_finish + timedelta(days=minor_slip_days)
                    # logging.debug(f"Task {task_id} Update {update_num}: Applied minor variation/slip. Est: {current_actual_finish.strftime('%Y-%m-%d')}")

        # 3. Ensure finish date is always after start date
        if current_actual_finish is not None and current_actual_finish <= actual_start:
            current_actual_finish = actual_start + timedelta(days=random.randint(1, 10)) # Ensure minimum duration if correction/slip goes wrong
            # logging.debug(f"Task {task_id} Update {update_num}: Adjusted finish date to be after start date. Est: {current_actual_finish.strftime('%Y-%m-%d')}")

        # Store the updated finish date back into task_info for next iteration's persistence
        task_info['actual_finish'] = current_actual_finish
    
    # --- END MODIFIED Actual Finish Logic ---

    # --- Format Dates --- (Using current_actual_finish)
    actual_start_str = actual_start.strftime("%Y-%m-%d") if actual_start else None
    actual_finish_str = current_actual_finish.strftime("%Y-%m-%d") if current_actual_finish else None # Use the final calculated date for this update
    baseline_start_str = baseline_start.strftime("%Y-%m-%d") if baseline_start else None
    baseline_end_str = baseline_end.strftime("%Y-%m-%d") if baseline_end else None

    # --- Generate other messy data (can reference update_num for realism) ---
    percent_comp = messy_percent_complete()
    status = messy_status()
    if actual_finish_str:
         percent_comp = random.choice(["100%", 1.0, "Complete"]) # Likely complete if finished
         status = "Complete"
    elif actual_start_str and update_num > 1:
        # If started and not first update, likely In Progress
        status = random.choice(["In Progress", "IP"])
        if isinstance(percent_comp, (int, float)) and percent_comp < 1.0:
             percent_comp = max(percent_comp, (update_num / total_updates) * random.uniform(0.8, 1.0)) # Ensure %comp increases somewhat
        elif isinstance(percent_comp, str) and '%' in percent_comp:
             try:
                  current_pct = int(percent_comp.replace('%',''))
                  percent_comp = f"{min(100, max(current_pct, int((update_num / total_updates) * random.uniform(70, 100))))}%"
             except ValueError:
                  pass # Keep original messy value


    task_update = {
        "task_id": task_id,
        "task_name": f"Task {task_id.split('-')[1]}", # Generate name from ID
        "actual_start": actual_start_str,
        "actual_finish": actual_finish_str, # Use consistent format string
        "baseline_start": baseline_start_str,
        "baseline_end": baseline_end_str,
        "percent_complete": percent_comp,
        "duration": random.choice([random.randint(5, 50), f"{random.randint(1,10)} days", f"{random.randint(1,5)} wks", None]),
        "status": status,
        "is_milestone": task_info['is_milestone'], # Keep consistent
        "is_critical": task_info['is_critical'], # Keep consistent
        "risk_flag": random_risk_flag(), # Can vary
        "severity_score": random.choice([round(random.uniform(0.0, 1.0), 2), None]), # Can vary
        "update_phase": f"Update_{update_num:03d}"
    }
    return task_update

# Main generator
def create_project_structure(base_path="Data/input", projects=("Slippages", "Updates"), updates=10, num_tasks_per_project=50):
    """Creates project folders and generates update files with consistent task history."""
    os.makedirs(base_path, exist_ok=True)
    today = datetime.today()
    start_window = today - timedelta(days=180) # Shorter window for more overlap
    end_window = today + timedelta(days=180)

    for project in projects:
        project_path = os.path.join(base_path, project)
        os.makedirs(project_path, exist_ok=True)

        # 1. Generate base task info ONCE per project
        project_tasks = {}
        for i in range(1, num_tasks_per_project + 1):
            task_id = f"TSK-{i:03d}"
            baseline_start = random_date(start_window, end_window)
            # Ensure baseline_end is definitely after baseline_start
            baseline_end = baseline_start + timedelta(days=random.randint(10, 60)) # Min duration 10 days
            
            # --- CHANGE: Increase Milestone Probability ---
            # Set milestone status with a higher probability (e.g., 25%)
            is_milestone = messy_boolean(true_probability=0.25)
            is_critical = messy_boolean(true_probability=0.3) # Slightly increase critical chance too
            
            project_tasks[task_id] = {
                'task_id': task_id,
                'baseline_start': baseline_start,
                'baseline_end': baseline_end,
                'is_milestone': is_milestone, # Use controlled boolean
                'is_critical': is_critical,   # Use controlled boolean
                # Store actual dates here as they evolve
                'actual_start': None,
                'actual_finish': None
            }

        # 2. Create update files, generating data for the *same* tasks each time
        for update_num in range(1, updates + 1):
            update_data = []
            # Iterate through the *master* project_tasks dictionary
            for task_id, task_info in project_tasks.items():
                 # Pass the *current state* of task_info (which includes potentially updated actual_finish)
                 task_update_row = generate_task_update_data(task_info, update_num, updates)
                 update_data.append(task_update_row)
                 # Update the master task_info with the latest actual_start/finish if they were set/changed
                 # This ensures the *next* call to generate_task_update_data uses the persistent state
                 if task_update_row['actual_start'] and task_info['actual_start'] is None:
                      try:
                          task_info['actual_start'] = datetime.strptime(task_update_row['actual_start'], "%Y-%m-%d")
                      except (ValueError, TypeError):
                          pass # Keep original if format is wrong
                 # ACTUAL_FINISH is already updated inside generate_task_update_data and stored back into task_info

            df = pd.DataFrame(update_data)

            # Biased towards CSV, but keep some XLSX
            file_format = "csv" if random.random() < 0.9 else "xlsx"
            filename = f"{project}_Update_{update_num:03d}.{file_format}"
            file_path = os.path.join(project_path, filename)

            if file_format == "csv":
                df.to_csv(file_path, index=False)
            else:
                # Ensure Excel writer handles NaT/None correctly if pandas version requires it
                try:
                    df.to_excel(file_path, index=False)
                except Exception as e:
                     print(f"⚠️ Warning: Failed to write Excel file {file_path}. Error: {e}. Skipping file.")
                     continue # Skip this file if Excel writing fails


            print(f"✅ Created: {file_path}")

if __name__ == "__main__":
    # Use the new parameter name num_tasks_per_project
    create_project_structure(
        projects=("Alpha", "Beta", "Gamma", "Delta", "Slippages", "Updates"),
        updates=10, # Generate 10 updates
        num_tasks_per_project=50 # Renamed parameter
    )