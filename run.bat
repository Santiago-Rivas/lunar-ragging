@echo off

python src\rag_refactored_no_parallel.py index --dossier_dir data\space_data\dossiers --html_dir data\space_data\html --event_id 1 --reset_db --chunk_size 2048 --chunk_overlap 128

python src\rag_refactored_no_parallel.py suggest --user_id celebrity-0001 celebrity-0002 celebrity-0003 celebrity-0004 --event_id 1 --evaluate --chunk_size 2048 --chunk_overlap 128

pause 