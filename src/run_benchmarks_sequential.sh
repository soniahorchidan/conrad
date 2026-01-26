#!/bin/bash

# Run three benchmark commands sequentially
# Each command will complete before the next one starts

echo "Starting ThreeHopPipeline benchmark at $(date)"
python -u run_crc_benchmark_auto.py --confidence-levels 0.5 0.6 0.7 0.8 0.9 --max-eval-queries 1000 --model_to_infer ThreeHopPipeline --calib_batch_size 8 --dataset fb15k-237 --incompleteness 20 --use-ultraquery
echo "ThreeHopPipeline completed at $(date)"

rm -rf ../artifacts/snapshots/ultraquery/vector_crc_cache/*

echo "Starting TwoUnionPipeline benchmark at $(date)"
python -u run_crc_benchmark_auto.py --confidence-levels 0.5 0.6 0.7 0.8 0.9 --max-eval-queries 1000 --model_to_infer TwoUnionPipeline --calib_batch_size 2 --dataset nell-955 --incompleteness 20
echo "TwoUnionPipeline completed at $(date)"

rm -rf ../artifacts/snapshots/ultra_nell955/vector_crc_cache/*

echo "Starting TwoIntersectProjectPipeline benchmark at $(date)"
python -u run_crc_benchmark_auto.py --confidence-levels 0.5 0.6 0.7 0.8 0.9 --max-eval-queries 1000 --model_to_infer TwoIntersectProjectPipeline --calib_batch_size 4 --dataset nell-955 --incompleteness 20
echo "TwoIntersectProjectPipeline completed at $(date)"

echo "All benchmarks completed at $(date)"

