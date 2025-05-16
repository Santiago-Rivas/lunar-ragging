#!/bin/bash

# Check if config file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_file.json>"
    exit 1
fi

CONFIG_FILE=$1

# Extract parameters from JSON using jq
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed. Please install jq first."
    exit 1
fi

# Extract configuration arrays
CHUNK_SIZES=($(jq -r '.configs.chunk_size[]' $CONFIG_FILE))
CHUNK_OVERLAPS=($(jq -r '.configs.chunk_overlap[]' $CONFIG_FILE))
EVENT_ID=$(jq -r '.configs.event_id' $CONFIG_FILE)
K_VALUES=($(jq -r '.configs.k[]' $CONFIG_FILE))
MMR_LAMBDAS=($(jq -r '.configs.mmr_lambda[]' $CONFIG_FILE))
TEST_USERS=($(jq -r '.test_users[]' $CONFIG_FILE))

echo "Running with the following configurations:"
echo "Chunk sizes: ${CHUNK_SIZES[@]}"
echo "Chunk overlaps: ${CHUNK_OVERLAPS[@]}"
echo "Event ID: $EVENT_ID"
echo "K values: ${K_VALUES[@]}"
echo "MMR lambda values: ${MMR_LAMBDAS[@]}"
echo "Test users: ${TEST_USERS[@]}"

# First, run indexing for all chunk size and overlap combinations
for CHUNK_SIZE in "${CHUNK_SIZES[@]}"; do
    for CHUNK_OVERLAP in "${CHUNK_OVERLAPS[@]}"; do
        echo "Indexing with chunk_size=$CHUNK_SIZE, chunk_overlap=$CHUNK_OVERLAP"
        python src/rag_refactored_no_parallel.py index \
            --dossier_dir data/space_data/dossiers \
            --html_dir data/space_data/html \
            --event_id "$EVENT_ID" \
            --reset_db \
            --chunk_size "$CHUNK_SIZE" \
            --chunk_overlap "$CHUNK_OVERLAP"
        
        # Then for each chunk size and overlap combination, run suggest with each k value and mmr_lambda
        for K_VALUE in "${K_VALUES[@]}"; do
            for MMR_LAMBDA in "${MMR_LAMBDAS[@]}"; do
                echo "Running suggestions with chunk_size=$CHUNK_SIZE, chunk_overlap=$CHUNK_OVERLAP, k=$K_VALUE, mmr_lambda=$MMR_LAMBDA"
                python src/rag_refactored_no_parallel.py suggest \
                    --user_id "${TEST_USERS[@]}" \
                    --event_id "$EVENT_ID" \
                    --evaluate \
                    --chunk_size "$CHUNK_SIZE" \
                    --chunk_overlap "$CHUNK_OVERLAP" \
                    --k "$K_VALUE" \
                    --mmr_lambda "$MMR_LAMBDA"
            done
        done
    done
done

echo "All tests completed."

