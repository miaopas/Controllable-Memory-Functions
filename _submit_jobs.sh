#!/bin/bash

# Wrapper script to submit PBS array jobs with multiple script files
# Usage: ./submit_jobs.sh [--seed-offset N] script1.sh script2.sh script3.sh

# Parse optional seed offset
SEED_OFFSET=""
SCRIPT_FILES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --seed-offset)
            if [ -z "$2" ] || [[ "$2" == -* ]]; then
                echo "Error: --seed-offset requires a numeric value"
                exit 1
            fi
            SEED_OFFSET="$2"
            shift 2
            ;;
        *)
            SCRIPT_FILES="$SCRIPT_FILES $1"
            shift
            ;;
    esac
done

# Trim leading/trailing spaces
SCRIPT_FILES=$(echo "$SCRIPT_FILES" | xargs)

if [ -z "$SCRIPT_FILES" ]; then
    echo "Usage: $0 [--seed-offset N] <script_file1> [script_file2] [script_file3] ..."
    echo ""
    echo "Options:"
    echo "  --seed-offset N    Add '+run_specs.seed_offset=N' to all commands"
    echo ""
    echo "Examples:"
    echo "  $0 scripts/Dataset/LSTM.sh scripts/Dataset/Transformer.sh"
    echo "  $0 --seed-offset 1000 scripts/Mixed_Models/LSTM_LSTM.sh"
    echo "  $0 --seed-offset 100 scripts/Dataset/*.sh"
    exit 1
fi

echo "Submitting PBS job with script files:"
if [ -n "$SEED_OFFSET" ]; then
    echo "Seed offset: $SEED_OFFSET (will append '+run_specs.seed_offset=$SEED_OFFSET' to all commands)"
fi
TOTAL_COMMANDS=0

# Create temporary combined file to count commands
CMD_DIR="$PWD/logs/pbs/cmds"
mkdir -p "$CMD_DIR"
TEMP_FILE="$CMD_DIR/pbs_commands_wrapper_$$_$(date +%s).txt"
> $TEMP_FILE

for file in $SCRIPT_FILES; do
    if [ -f "$file" ]; then
        # Extract non-comment, non-empty lines
        VALID_COMMANDS=$(grep "^\s*[^#].*\S" "$file")
        COMMANDS_IN_FILE=$(echo "$VALID_COMMANDS" | grep -c ".")

        echo "  - $file ($COMMANDS_IN_FILE commands)"

        # Add each valid command to temp file (one per line)
        # Append seed_offset if specified
        if [ -n "$SEED_OFFSET" ]; then
            while IFS= read -r cmd; do
                if [ -n "$cmd" ]; then
                    echo "$cmd +run_specs.seed_offset=$SEED_OFFSET" >> $TEMP_FILE
                fi
            done <<< "$VALID_COMMANDS"
        else
            echo "$VALID_COMMANDS" >> $TEMP_FILE
        fi

        TOTAL_COMMANDS=$((TOTAL_COMMANDS + COMMANDS_IN_FILE))
    else
        echo "  - $file (FILE NOT FOUND!)"
        echo "Error: Script file '$file' does not exist."
        rm -f $TEMP_FILE
        exit 1
    fi
done

echo ""
echo "=== Final command file contents ==="
cat -n "$TEMP_FILE"
echo "=== End of command file ==="

echo ""
NUM_SCRIPT_FILES=$(echo "$SCRIPT_FILES" | wc -w)
echo "Total script files: $NUM_SCRIPT_FILES"
echo "Total commands: $TOTAL_COMMANDS"

if [ $TOTAL_COMMANDS -eq 0 ]; then
    echo "Error: No valid commands found in script files!"
    rm -f $TEMP_FILE
    exit 1
fi

# Submit the PBS job with array range and temp file (PBS arrays start from 0)
ARRAY_END=$((TOTAL_COMMANDS - 1))
echo "Submitting array job with range 0-$ARRAY_END"
qsub -J 0-$ARRAY_END -v TEMP_FILE="$TEMP_FILE",WANDB_API_KEY _run_array_jobs.pbs

echo "Job submitted successfully!"
echo "Temporary command file: $TEMP_FILE"