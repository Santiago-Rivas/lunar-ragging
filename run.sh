#!/bin/bash


# python src/rag_refactored_no_parallel.py index --dossier_dir data/space_data/dossiers --html_dir data/space_data/html --event_id 1 --reset_db --chunk_size 100 --chunk_overlap 10

python src/rag_refactored_no_parallel.py suggest --user_id 1 2 3 4 --event_id 1 --evaluate --chunk_size 100 --chunk_overlap 10 --visualize

