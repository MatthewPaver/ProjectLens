#!/usr/bin/env python3
"""Generates Synthetic Project Data for Testing.

This script creates mock project data files (CSV or Excel) suitable for testing
the ProjectLens processing pipeline. It simulates multiple projects, each with 
several update files containing task progress information.

The generated data includes:
- Consistent task information across updates.
- Simulated progress (start/finish dates, percentage complete) with variability.
- Introduction of delays and potential corrections.
- 'Messy' data formats for fields like percentage complete and booleans to test cleaning.
- Calculated metrics like delay days, delay percentage, and severity score.

This script is intended to be run manually or as part of a test setup process
to populate the `Data/input` directory with test cases.
"""
import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging # Added for potential debug logging

# --- Setup logging (optional, can be useful for debugging generation)
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Add at the top of the file after imports
"""
Severity Score Calculation Guide
------------------------------
The severity score is calculated on a 0-10 scale based on the following factors:

1. Delay Impact (0-5 points):
   - No delay: 0 points
   - 1-10 days: 1 point
   - 11-20 days: 2 points
   - 21-30 days: 3 points
   - 31-40 days: 4 points
   - >40 days: 5 points

2. Risk Level (0-2 points):
   - Low/L: 0 points
   - Medium/M: 1 point
   - High/H: 2 points

3. Critical Path (0-2 points):
   - Not on critical path: 0 points
   - On critical path: 2 points

4. Milestone Status (0-1 point):
   - Not a milestone: 0 points
   - Is a milestone: 1 point

Maximum possible score: 10 points
"""

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

def calculate_severity_score(task_info, update_num, total_updates):
    """
    Calculate severity score (0-10) based on multiple factors.
    Returns a tuple of (score, list of contributing factors).
    
    Scoring Breakdown:
    - Delay: 0-5 points (based on days delayed)
    - Risk: 0-2 points (based on risk flag)
    - Critical Path: 0-2 points
    - Milestone: 0-1 point
    """
    score = 0
    factors = []
    
    # 1. Delay Impact (0-5 points)
    if task_info.get('actual_finish') and task_info.get('baseline_end'):
        delay_days = (task_info['actual_finish'] - task_info['baseline_end']).days
        if delay_days > 0:
            # Calculate delay score on a scale
            if delay_days <= 10:
                delay_score = 1
            elif delay_days <= 20:
                delay_score = 2
            elif delay_days <= 30:
                delay_score = 3
            elif delay_days <= 40:
                delay_score = 4
            else:
                delay_score = 5
            
            score += delay_score
            factors.append(f"Delayed by {delay_days} days (+{delay_score} points)")
    
    # 2. Risk Impact (0-2 points)
    risk_flag = task_info.get('risk_flag', 'Low')
    risk_scores = {'Low': 0, 'L': 0, 'Medium': 1, 'M': 1, 'High': 2, 'H': 2}
    if risk_flag in risk_scores:
        risk_score = risk_scores[risk_flag]
        score += risk_score
        if risk_score > 0:
            factors.append(f"Risk Level: {risk_flag} (+{risk_score} points)")
    
    # 3. Critical Path Impact (0-2 points)
    if task_info.get('is_critical'):
        score += 2
        factors.append("On Critical Path (+2 points)")
    
    # 4. Milestone Impact (0-1 point)
    if task_info.get('is_milestone'):
        score += 1
        factors.append("Milestone Task (+1 point)")

    # Ensure score doesn't exceed 10
    final_score = min(10, round(score, 1))
    
    # Add total to factors if it was capped
    if score > 10:
        factors.append(f"Score capped at 10 (raw score: {score})")
    
    return final_score, factors

def is_task_active(actual_start, actual_finish, status):
    """Determine if a task is currently active based on dates and status"""
    if status in ["Complete", "Completed"]:
        return False
    if actual_start and not actual_finish:
        return True
    if status in ["In Progress", "IP"]:
        return True
    return False

def calculate_delay_metrics(baseline_end, actual_finish, baseline_duration):
    """
    Calculate delay metrics for a task
    Returns: (is_delayed, delay_days, delay_percentage)
    """
    if not baseline_end or not actual_finish:
        return False, 0, 0
    
    delay_days = (actual_finish - baseline_end).days
    is_delayed = delay_days > 0
    
    # Calculate delay as a percentage of original duration
    if baseline_duration and baseline_duration > 0:
        delay_percentage = (delay_days / baseline_duration) * 100
    else:
        delay_percentage = 0
        
    return is_delayed, delay_days, round(delay_percentage, 1)

def generate_task_update_data(task_info, update_num, total_updates):
    """
    Generates data for a single task for a specific update phase,
    using existing baseline data and simulating progress.
    """
    task_id = task_info['task_id']
    baseline_start = task_info['baseline_start']
    baseline_end = task_info['baseline_end']
    
    # Calculate baseline duration
    baseline_duration = (baseline_end - baseline_start).days if baseline_start and baseline_end else None

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
    re_evaluate_chance = 0.60 # Increased from 0.40 (60% chance to potentially change the finish date per update)
    significant_slip_chance = 0.25 # Increased from 0.15 (25% chance per update for a *large* slip)
    correction_chance = 0.05 # Increased slightly from 0.03 (5% chance to correct *slightly* earlier if already slipped)
    abrupt_change_chance = 0.05 # New: 5% chance of a drastic, sudden change

    # Always re-evaluate if not finished yet
    if previous_actual_finish is None or random.random() < re_evaluate_chance:

        # 0. Check for abrupt changes first
        abrupt_magnitude_days = 0
        if random.random() < abrupt_change_chance and update_num > 2: # Only after a couple of updates
            direction = random.choice([-1, 1]) # Pull in or push out
            abrupt_magnitude_days = random.randint(40, 80) * direction
            base_for_abrupt = max(baseline_end, previous_actual_finish) if previous_actual_finish else baseline_end
            current_actual_finish = base_for_abrupt + timedelta(days=abrupt_magnitude_days)
            # logging.debug(f"Task {task_id} Update {update_num}: Applied abrupt change. New Est: {current_actual_finish.strftime('%Y-%m-%d')}")

        # 1. If no abrupt change, check for significant slip
        slip_magnitude_days = 0
        if abrupt_magnitude_days == 0 and random.random() < significant_slip_chance and update_num > 1:
            slip_magnitude_days = random.randint(20, 50 + update_num * 3) # Increased magnitude range slightly
            # Calculate new finish based on baseline OR previous finish, whichever is later
            base_for_slip = max(baseline_end, previous_actual_finish) if previous_actual_finish else baseline_end
            current_actual_finish = base_for_slip + timedelta(days=slip_magnitude_days)
            # logging.debug(f"Task {task_id} Update {update_num}: Applied significant slip. New Est: {current_actual_finish.strftime('%Y-%m-%d')}")

        # 2. If no significant slip or abrupt change, apply normal progress/minor variation or initial finish
        elif slip_magnitude_days == 0 and abrupt_magnitude_days == 0:
            # If never finished, calculate initial finish probability
            if previous_actual_finish is None:
                # Increased chance to finish earlier in the updates
                finish_probability = (update_num / total_updates) * 0.85 # Increased from 0.7
                if random.random() < finish_probability:
                    # Wider initial variance, more chance for initial slip
                    potential_finish = baseline_end + timedelta(days=random.randint(-5, 25)) # Increased range
                    current_actual_finish = potential_finish
                    # logging.debug(f"Task {task_id} Update {update_num}: Initial finish set. Est: {current_actual_finish.strftime('%Y-%m-%d')}")

            # If already finished, add minor random variation (small slips/corrections)
            else:
                # Chance for minor correction
                if random.random() < correction_chance:
                     current_actual_finish = previous_actual_finish - timedelta(days=random.randint(1, 7)) # Increased correction range
                     # logging.debug(f"Task {task_id} Update {update_num}: Applied minor correction. Est: {current_actual_finish.strftime('%Y-%m-%d')}")
                # More likely to add minor slips
                else:
                    minor_slip_days = random.randint(0, 10) # Increased minor slip range
                    current_actual_finish = previous_actual_finish + timedelta(days=minor_slip_days)
                    # logging.debug(f"Task {task_id} Update {update_num}: Applied minor variation/slip. Est: {current_actual_finish.strftime('%Y-%m-%d')}")

        # 3. Ensure finish date is always after start date
        if current_actual_finish is not None and current_actual_finish <= actual_start:
            current_actual_finish = actual_start + timedelta(days=random.randint(5, 15)) # Ensure minimum duration if correction/slip goes wrong
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
        status = random.choice(["In Progress", "IP", "Started"]) # Added "Started"
        # Make percent complete progression slightly more realistic
        if isinstance(percent_comp, (int, float)):
             current_pct_float = percent_comp if isinstance(percent_comp, float) else percent_comp / 100.0
             # Ensure progress increases, but add some randomness
             estimated_progress = (update_num / total_updates) * random.uniform(0.7, 1.1) # Target progress
             new_pct_float = min(1.0, max(current_pct_float, estimated_progress))
             percent_comp = new_pct_float # Store as float 0.0-1.0 internally for messy formatting later
        elif isinstance(percent_comp, str) and '%' in percent_comp:
             try:
                  current_pct = int(percent_comp.replace('%',''))
                  estimated_progress_pct = int((update_num / total_updates) * random.uniform(70, 110))
                  percent_comp = f"{min(100, max(current_pct, estimated_progress_pct))}%"
             except ValueError:
                  # Retain existing value when string parsing fails to preserve data format
                  # This maintains irregularities that might exist in real project data
                  ...
        else: # If None or other messy value, maybe assign a starting point
             if random.random() < 0.5: # 50% chance to give it a % value if started
                 percent_comp = f"{min(95, int((update_num / total_updates) * random.uniform(50, 90)))}%"

    # --- Reformat percent_comp before output if it's a float ---
    if isinstance(percent_comp, float):
        if random.random() < 0.7: # 70% chance to format as percentage string
            percent_comp = f"{int(percent_comp * 100)}%"
        # else keep as float or int(percent_comp*100) based on messy_percent_complete logic if needed

    # Calculate delay metrics
    is_delayed, delay_days, delay_percentage = calculate_delay_metrics(
        baseline_end,
        current_actual_finish,
        baseline_duration
    )

    # Calculate severity score with factors
    severity_score, severity_factors = calculate_severity_score(task_info, update_num, total_updates)
    
    # Determine if task is active
    is_active = is_task_active(actual_start, current_actual_finish, status)

    # --- MODIFIED: Use consistent is_critical from initial task info ---
    is_critical = task_info['is_critical']
    # --- END MODIFICATION ---

    task_update = {
        "project_name": task_info['project_name'],  # Add project name
        "task_id": task_id,
        "task_name": task_info['task_name'],
        "actual_start": actual_start_str,
        "actual_finish": actual_finish_str,
        "baseline_start": baseline_start_str,
        "baseline_end": baseline_end_str,
        "baseline_duration": baseline_duration,
        "is_delayed": is_delayed,
        "delay_days": delay_days,
        "delay_percentage": delay_percentage,
        "percent_complete": percent_comp,
        "status": status,
        "is_milestone": task_info['is_milestone'],
        "is_critical": is_critical,
        "is_active": is_active,
        "risk_flag": task_info.get('risk_flag', random_risk_flag()),
        "severity_score": severity_score,
        "severity_factors": "; ".join(severity_factors),
        "update_phase": f"Update_{update_num:03d}"
    }
    return task_update

# Main generator
def create_project_structure(base_path="Data/input", projects=("Slippages", "Updates"), updates=10, num_tasks_per_project=50):
    """Creates project folders and generates update files with consistent task history."""
    os.makedirs(base_path, exist_ok=True)
    today = datetime.today()
    start_window = today - timedelta(days=180)
    end_window = today + timedelta(days=180)

    for project in projects:
        project_path = os.path.join(base_path, project)
        os.makedirs(project_path, exist_ok=True)

        # Generate base task info ONCE per project with unique task IDs
        project_tasks = {}
        used_task_names = set()
        
        for i in range(1, num_tasks_per_project + 1):
            while True:
                task_name = f"Task {i:03d}"
                if task_name not in used_task_names:
                    used_task_names.add(task_name)
                    break
                i += 1
            
            task_id = f"TSK-{i:03d}"
            baseline_start = random_date(start_window, end_window)
            baseline_end = baseline_start + timedelta(days=random.randint(10, 60))
            
            is_milestone = random.random() < 0.25  # 25% chance
            is_critical = random.random() < 0.30   # 30% chance
            risk_flag = random_risk_flag()
            
            # Initialise with None for actual dates - they'll be updated progressively
            project_tasks[task_id] = {
                'project_name': project,
                'task_id': task_id,
                'task_name': task_name,
                'baseline_start': baseline_start,
                'baseline_end': baseline_end,
                'is_milestone': is_milestone,
                'is_critical': is_critical,
                'risk_flag': risk_flag,
                'actual_start': None,
                'actual_finish': None,
                'last_update': None  # Track the last update number
            }

        # Create update files with progressive changes
        for update_num in range(1, updates + 1):
            update_data = []
            
            # Update each task progressively
            for task_id, task_info in project_tasks.items():
                # Only generate new data if task hasn't been completed or needs updating
                if (task_info['actual_finish'] is None or 
                    random.random() < 0.1):  # 10% chance to update even completed tasks
                    
                    task_update_row = generate_task_update_data(task_info, update_num, updates)
                    
                    # Update the master task_info with any changes
                    if task_update_row['actual_start']:
                        try:
                            task_info['actual_start'] = datetime.strptime(task_update_row['actual_start'], "%Y-%m-%d")
                        except (ValueError, TypeError):
                            # Ignore parse error, keep previous value
                            ...
                    
                    if task_update_row['actual_finish']:
                        try:
                            task_info['actual_finish'] = datetime.strptime(task_update_row['actual_finish'], "%Y-%m-%d")
                        except (ValueError, TypeError):
                            # Ignore parse error, keep previous value
                            ...
                    
                    task_info['last_update'] = update_num
                    
                else:
                    # Use the last known state for completed tasks
                    task_update_row = {
                        "project_name": task_info['project_name'],
                        "task_id": task_id,
                        "task_name": task_info['task_name'],
                        "actual_start": task_info['actual_start'].strftime("%Y-%m-%d") if task_info['actual_start'] else None,
                        "actual_finish": task_info['actual_finish'].strftime("%Y-%m-%d") if task_info['actual_finish'] else None,
                        "baseline_start": task_info['baseline_start'].strftime("%Y-%m-%d"),
                        "baseline_end": task_info['baseline_end'].strftime("%Y-%m-%d"),
                        "baseline_duration": (task_info['baseline_end'] - task_info['baseline_start']).days,
                        "is_delayed": task_info.get('is_delayed', False),
                        "delay_days": task_info.get('delay_days', 0),
                        "delay_percentage": task_info.get('delay_percentage', 0),
                        "percent_complete": "100%" if task_info['actual_finish'] else task_info.get('percent_complete', "0%"),
                        "status": "Complete" if task_info['actual_finish'] else task_info.get('status', "In Progress"),
                        "is_milestone": task_info['is_milestone'],
                        "is_critical": task_info['is_critical'],
                        "is_active": False if task_info['actual_finish'] else True,
                        "risk_flag": task_info['risk_flag'],
                        "severity_score": task_info.get('severity_score', 0),
                        "severity_factors": task_info.get('severity_factors', ""),
                        "update_phase": f"Update_{update_num:03d}"
                    }
                
                update_data.append(task_update_row)

            # Create the update file
            df = pd.DataFrame(update_data)
            
            # Ensure consistent data types
            if not df.empty:
                # Convert date columns to datetime
                date_columns = ['actual_start', 'actual_finish', 'baseline_start', 'baseline_end']
                for col in date_columns:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                
                # Ensure numeric columns are float
                numeric_columns = ['delay_days', 'delay_percentage', 'severity_score']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

            # Save with consistent format
            file_format = "csv" if random.random() < 0.9 else "xlsx"
            filename = f"{project}_Update_{update_num:03d}.{file_format}"
            file_path = os.path.join(project_path, filename)

            try:
                if file_format == "csv":
                    df.to_csv(file_path, index=False)
                else:
                    df.to_excel(file_path, index=False)
                print(f"✅ Created: {file_path}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to write file {file_path}. Error: {e}")
                continue

if __name__ == "__main__":
    # Use the new parameter name num_tasks_per_project
    create_project_structure(
        projects=("Alpha", "Beta", "Gamma", "Delta", "Slippages", "Updates"),
        updates=15, # Increased from 10
        num_tasks_per_project=75 # Increased from 50
    )
