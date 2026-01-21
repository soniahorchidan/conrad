# Conrad - Inference-Only Benchmarking Project

This is a self-contained extraction from the orb/ project, focused on inference benchmarking with Neo4j, Ultra, DBExec, and the three pipeline models.

## Supported Models

- **ULTRA** - Neural link prediction model
- **DBExecModel** - Neo4j symbolic execution model
- **MultiHopPredictor** - Combines Ultra + DBExec
- **ThreeHopPipeline** - 3-hop query pipeline
- **TwoUnionPipeline** - 2-union query pipeline
- **TwoIntersectProjectPipeline** - 2-intersect-project query pipeline

## Quick Start

### 1. Install Dependencies

```bash
./build.sh install_deps [cpu|cuda]
```

### 2. Setup Neo4j

```bash
./build.sh neo4j prepare
```

### 3. Start Neo4j

```bash
./build.sh neo4j start --without-docker
```

### 4. Generate Calibration Data

```bash
benchmark/scripts/generate_calibration.sh --delete-edges-perc 30 --dataset fb15k-237
```

### 5. Run Benchmarks

```bash
cd ml_engine
./run_benchmarks_sequential.sh
```

Or run individual benchmarks:

```bash
python run_crc_benchmark_auto.py --confidence-levels 0.5 0.6 0.7 --args_file ../config/orb_config.json --model_to_infer ThreeHopPipeline --dataset fb15k-237
```

## Directory Structure

```
conrad/
├── build.sh                    # Build script (neo4j, install_deps only)
├── scripts/                    # Build scripts
│   ├── constants.env
│   ├── install_neo4j.sh
│   └── change_neo4j_db_path.sh
├── config/
│   └── orb_config.json        # Configuration file
├── benchmark/
│   └── scripts/
│       ├── generate_calibration.sh
│       └── delete_random_edges.py
├── ml_engine/
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

## What Was Removed

- Training code (`training/`, `main_train.py`)
- Kuzu backend (only Neo4j supported)
- BigQuery backend
- VectorDB code
- Unused models (Query2box, TransR, NodePiece, RelationPrediction)
- FastAPI inference server
- Graph manager/sampler (training only)

## Notes

- This project is inference-only - no training functionality
- Only Neo4j backend is supported (no Kuzu or BigQuery)
- All paths are relative to the conrad/ directory
- The project maintains modularity and code reuse from the original orb/ project
