# Conrad - Inference-Only Benchmarking Project

This is a self-contained extraction from the orb/ project, focused on inference benchmarking with Neo4j, Ultra, DBExec, and the three pipeline models.

## Supported Models

- **ULTRA** - Neural link prediction model
- **DBExecModel** - Neo4j symbolic execution model
- **MultiHopPredictor** - Combines Ultra + DBExec
- **ThreeHopPipeline** - 3-hop query pipeline
- **TwoUnionPipeline** - 2-union query pipeline
- **TwoIntersectProjectPipeline** - 2-intersect-project query pipeline

## Directory Structure

```
conrad/
├── build.sh                    # Build script (neo4j, install_deps only)
├── scripts/                    # Build scripts
│   ├── constants.env
│   ├── install_neo4j.sh
│   └── change_neo4j_db_path.sh
├── config/
│   └── config.json        # Configuration file
├── benchmark/
│   └── scripts/
│       ├── generate_calibration.sh
│       └── delete_random_edges.py
├── src/
│   ├── inference/             # Model loading and factory
│   ├── utils/                 # Utilities
│   ├── graph_handler/         # Neo4j backend only
│   ├── models/                # Ultra, DBExec, MultiHop, Topology
│   ├── conformal_prediction/  # CRC implementation
│   ├── run_crc_benchmark_auto.py
│   ├── run_crc_benchmark_sweep.sh
│   ├── run_benchmarks_sequential.sh
│   ├── run_baseline_benchmarks.py
│   └── validate_crc_composition.py
├── requirements-cpu.txt
└── requirements-gpu.txt
```
