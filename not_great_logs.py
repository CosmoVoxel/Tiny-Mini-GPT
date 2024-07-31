import os
import shutil
from tensorboard.backend.event_processing import event_accumulator

def get_step_count_from_loss_folder(loss_dir):
    ea = event_accumulator.EventAccumulator(loss_dir)
    try:
        ea.Reload()
        scalar_tags = ea.Tags().get('scalars', [])
        if scalar_tags:
            # Get the total number of steps from all scalar tags
            step_count = max(len(ea.Scalars(tag)) for tag in scalar_tags)
        else:
            step_count = 0
    except Exception as e:
        print(f"Could not read events from {loss_dir}: {e}")
        step_count = 0
    return step_count

def get_minimum_loss_from_loss_folder(loss_dir):
    ea = event_accumulator.EventAccumulator(loss_dir)
    try:
        ea.Reload()
        scalar_tags = ea.Tags().get('scalars', [])
        if scalar_tags:
            # Get the minimum loss value from all scalar tags
            min_loss = min(min(scalar.value for scalar in ea.Scalars(tag)) for tag in scalar_tags)
        else:
            min_loss = float('inf')
    except Exception as e:
        print(f"Could not read events from {loss_dir}: {e}")
        min_loss = float('inf')
    return min_loss

def delete_runs(logdir, min_steps, max_loss):
    for root, dirs, files in os.walk(logdir):
        for d in dirs:
            if d.startswith('version'):
                run_dir = os.path.join(root, d)
                loss_dir = os.path.join(run_dir, 'Loss_Train Loss')
                
                if not os.path.exists(loss_dir):
                    print(f"Deleting {run_dir} because 'Loss' folder does not exist")
                    shutil.rmtree(run_dir)
                else:
                    step_count = get_step_count_from_loss_folder(loss_dir)
                    min_loss = get_minimum_loss_from_loss_folder(loss_dir)
                    if step_count < min_steps:
                        print(f"Deleting {run_dir} with {step_count} steps")
                    if min_loss > max_loss:
                        print(f"Deleting {run_dir} with minimum loss of {min_loss}")

                       
                        shutil.rmtree(run_dir)

# Set your log directory, minimum step count threshold, and maximum acceptable loss
log_directory = "tb_logs/GPT/"
min_steps_threshold = 5000
max_loss_threshold = 2.5

delete_runs(log_directory, min_steps_threshold, max_loss_threshold)
