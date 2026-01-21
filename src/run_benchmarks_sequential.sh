#!/bin/bash

# Run three benchmark commands sequentially
# Each command will complete before the next one starts

echo "Starting ThreeHopPipeline benchmark at $(date)"
python -u run_crc_benchmark_auto.py --confidence-levels 0.5 0.6 0.7 0.8 0.9 --max-eval-queries 1000 --args_file ../config/orb_config.json --model_to_infer ThreeHopPipeline --disable-raps --calib_batch_size 2 --dataset nell-955
echo "ThreeHopPipeline completed at $(date)"

rm -rf ../artifacts/snapshots/ultra_nell955/vector_crc_cache/*

echo "Starting TwoUnionPipeline benchmark at $(date)"
python -u run_crc_benchmark_auto.py --confidence-levels 0.5 0.6 0.7 0.8 0.9 --max-eval-queries 1000 --args_file ../config/orb_config.json --model_to_infer TwoUnionPipeline --disable-raps --calib_batch_size 2 --dataset nell-955
echo "TwoUnionPipeline completed at $(date)"

rm -rf ../artifacts/snapshots/ultra_nell955/vector_crc_cache/*

echo "Starting TwoIntersectProjectPipeline benchmark at $(date)"
python -u run_crc_benchmark_auto.py --confidence-levels 0.5 0.6 0.7 0.8 0.9 --max-eval-queries 5000 --args_file ../config/orb_config.json --model_to_infer TwoIntersectProjectPipeline --disable-raps
echo "TwoIntersectProjectPipeline completed at $(date)"

echo "All benchmarks completed at $(date)"

