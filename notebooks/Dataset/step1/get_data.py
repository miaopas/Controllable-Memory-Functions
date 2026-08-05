import pickle
from tqdm import tqdm  # tqdm for Jupyter notebooks

import pandas as pd
import wandb
from omegaconf import OmegaConf
import pandas as pd
import json  # if the dict is in string format
from pandas import json_normalize

entity = "Long_Memory_Arena"
api = wandb.Api()
projects = [p.name for p in api.projects(entity=entity)]

def retrive_data(project):
    print(f'Retriving {project}')
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(f"Long_Memory_Arena/{project}")

    summary_list, config_list, name_list, tag_list = [], [], [], []
    loss_list = []

    for run in tqdm(runs):
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

        tag_list.append(run.tags)

        try:
            history = run.scan_history(keys=["train/rLoss1_epoch"])
            losses = [row["train/rLoss1_epoch"] for row in history if "train/rLoss1_epoch" in row]
        except Exception as e:
            print(f"Failed to get loss for run {run.name}: {e}")
            losses = []

        loss_list.append(losses)


    df = pd.DataFrame(
        {"summary": summary_list, "config": config_list, "name": name_list, "tag": tag_list, "loss_history": loss_list}
    )

    with open(f"{project}_data.pkl", "wb") as f:
        pickle.dump(df, f)
    print(f'Saved {project}')

def clean_data(p):
    with open(f"{p}_data.pkl", "rb") as f:
        df = pickle.load(f)

    # Example: Assuming `df` is your DataFrame, and the column is called 'config'
    # If it's a string, convert to actual dict:
    df['config_dict'] = df['config'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df['summary_dict'] = df['summary'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    # Now flatten the nested dict into separate columns
    expanded_df = json_normalize(df['config_dict'])
    expanded_df1 = json_normalize(df['summary_dict'])

    # Merge back with original df if needed
    df = df.drop(columns=['config'])  # optionally drop the original column
    df = df.drop(columns=['summary'])
    df = pd.concat([df, expanded_df,expanded_df1], axis=1)

    df.to_pickle(f"{p}_data_clean.pkl")

# projects = sorted([p.name for p in api.projects(entity=entity) if "step3" in p.name])

projects = sorted([p.name for p in api.projects(entity=entity)])

print(projects)
import time


for p in tqdm(projects):

    
    retrive_data(p)

    clean_data(p)
    print(f'Cleaned {p}')
    
    time.sleep(30)

