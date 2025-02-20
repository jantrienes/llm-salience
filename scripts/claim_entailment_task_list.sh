#!/bin/bash

DATASET=$1
FACTS_PATH=$2

filename=$(basename "$FACTS_PATH")

if [ "$filename" == "facts.json" ]; then
    out_dir="nli"
elif [ "$filename" == "discord_facts.json" ]; then
    out_dir="discord-qa-nli"
else
    echo "Unknown facts type."
    exit 1
fi

# Generate a list of input files
SUMMARIES_FILES=(output/$DATASET/*/summaries/{temperature*-*,output*}.json)
# SUMMARIES_FILES=(output/$DATASET/*/summaries/{temperature0.3-,temperature0.7-,output}*.json)

task_list="entailment_tasks_$DATASET.txt"
> "$task_list"  # Clear the file list

for file in "${SUMMARIES_FILES[@]}"; do
    # Skip summary files for which we already have an output
    # Below subsitutes "summaries" in the file path with $out_dir
    output_file="${file/summaries/$out_dir}"
    if [[ -e "$output_file" ]]; then
        continue
    fi
    echo "$file" >> "$task_list"
done

# Count the number of valid input files
FILE_COUNT=$(wc -l < "$task_list")

# Submit the job array, where each job processes a different file
echo "Tasks: $FILE_COUNT"
echo "Task file: $task_list"
echo "Use the following command to start the job array:"
echo sbatch --array=0-$(($FILE_COUNT - 1))%20 scripts/claim_entailment_array.sh $FACTS_PATH $task_list
