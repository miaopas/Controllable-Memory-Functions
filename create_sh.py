import os

commands = [
 "src/train_multiple.py trainer=gpu experiment=Dataset/Mamba/exp/step2/coarse +skip_confirm=true",
"src/train_multiple.py trainer=gpu experiment=Dataset/Mamba/polynomial/step2/coarse +skip_confirm=true",
"src/train_multiple.py trainer=gpu experiment=Dataset/Mamba/impulse/step2/coarse +skip_confirm=true",
"src/train_multiple.py trainer=gpu experiment=Dataset/Mamba/airy/step2/coarse +skip_confirm=true"
]

for idx, command in enumerate(commands):
    # Use the experiment path as the filename, replacing slashes with underscores
    parts = command.split("experiment=")[-1].split()[0]
    filename = f"run{idx}.sh" 

    with open(filename, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {os.getcwd()}\n\n")
        f.write(f"/app/.venv/bin/python {command}\n")

    print(f"run{idx}.sh")
