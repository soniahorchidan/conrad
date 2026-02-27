#!/bin/bash

set -e

# Get script directory for resolving paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set defaults if variables aren't set
DATA_PATH="${DATA_PATH:-./artifacts/data}"
DATABASES_PATH="${DATABASES_PATH:-./artifacts/databases}"
NEO4J_CONTAINER="${NEO4J_CONTAINER:-neo4j_orb}"
LOCAL_NEO4J_PATH="${LOCAL_NEO4J_PATH:-./external/}"
MAKE_JOBS="${MAKE_JOBS:-4}"
NEO4J_USERNAME="${NEO4J_USERNAME:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password123}"

get_orb_config() {
    # read priority: input parameter -> ./orb_config.json.local -> ./config/config.json
    local config_path="${1:-./orb_config.json.local}"
    if [ ! -f "$config_path" ]; then
        config_path="config/config.json"
    fi
    if [ ! -f "$config_path" ]; then
        echo "Error: Config file not found at $config_path"
        exit 1
    fi
    echo "$config_path"
}

################### Auxiliary Functions ##################

create_directory() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "Directory created: $dir"
    else
        echo "Directory already exists: $dir"
    fi
}

require_command() {
    local cmd="$1"
    if ! command -v "$cmd" &> /dev/null
    then
        echo "$cmd needs to be installed."
        return 1
    else
        return 0

    fi
}

install_deps() {
    # Check for user-specified mode
    if [[ "$1" == "cuda" ]]; then
        USE_CUDA=true
    elif [[ "$1" == "cpu" ]]; then
        USE_CUDA=false
    else
        # Auto-detect CUDA
        if command -v nvidia-smi &> /dev/null; then
            USE_CUDA=true
        else
            USE_CUDA=false
        fi
    fi

    # Install dependencies based on CUDA availability
    if $USE_CUDA; then
        echo "Installing CUDA dependencies..."
        pip install --no-cache-dir -r requirements-gpu.txt
    else
        echo "Installing CPU dependencies..."
        pip install -r requirements-cpu.txt
    fi
}

check_deps() {
    # check if the required python packages are installed
    # by checking if the import torch command works
    if ! python3 -c "import torch" &> /dev/null; then
        echo "Python dependencies not installed. Please run: $0 install_deps [cpu|cuda]"
        exit 1
    fi
}

################ Start Neo4j ###############
# Neo4j only supports one database (non-enterprise).
# So we can only ever load a single dataset into the default database "neo4j".

prepare_neo4j() {
    bash ./scripts/install_neo4j.sh
}


# This function can ONLY be used in combination with the orb_dependency NEO4J instance.
# This is especially used in the the CI.
import_neo4j() {
    local dataset="$1"
    
    if [ -z "$dataset" ]; then
        echo "Error: Dataset name is required"
        echo "Usage: $0 neo4j import <dataset> [--without-docker]"
        exit 1
    fi

    # Check if Neo4j is running
    if neo4j status | grep -q "running"; then
        echo "Error: Neo4j is currently running. Please stop the service before importing the dataset."
        exit 1
    fi

    # Validate that required data files exist
    local node_header_file="$DATA_PATH/$dataset/neo4j_train_ind_ent_header.csv"
    local node_file="$DATA_PATH/$dataset/neo4j_train_ind_ent.csv"
    local rel_header_file="$DATA_PATH/$dataset/neo4j_train_ind_rels_header.csv"
    local rel_file="$DATA_PATH/$dataset/neo4j_train_ind_rels.csv"

    if [ ! -f "$node_header_file" ]; then
        echo "Error: Required file not found: $node_header_file"
        echo "Please ensure the dataset files are available at $DATA_PATH/$dataset/"
        if [ "$dataset" == "fb15k-237" ] || [ "$dataset" == "nell-955" ] || [ "$dataset" == "yago310" ]; then
            echo "For $dataset, you can run: python3 scripts/process_dataset.py $dataset"
        fi
        exit 1
    fi
    if [ ! -f "$node_file" ]; then
        echo "Error: Required file not found: $node_file"
        echo "Please ensure the dataset files are available at $DATA_PATH/$dataset/"
        if [ "$dataset" == "fb15k-237" ] || [ "$dataset" == "nell-955" ] || [ "$dataset" == "yago310" ]; then
            echo "For $dataset, you can run: python3 scripts/process_dataset.py $dataset"
        fi
        exit 1
    fi
    if [ ! -f "$rel_header_file" ]; then
        echo "Error: Required file not found: $rel_header_file"
        echo "Please ensure the dataset files are available at $DATA_PATH/$dataset/"
        if [ "$dataset" == "fb15k-237" ] || [ "$dataset" == "nell-955" ] || [ "$dataset" == "yago310" ]; then
            echo "For $dataset, you can run: python3 scripts/process_dataset.py $dataset"
        fi
        exit 1
    fi
    if [ ! -f "$rel_file" ]; then
        echo "Error: Required file not found: $rel_file"
        echo "Please ensure the dataset files are available at $DATA_PATH/$dataset/"
        if [ "$dataset" == "fb15k-237" ] || [ "$dataset" == "nell-955" ] || [ "$dataset" == "yago310" ]; then
            echo "For $dataset, you can run: python3 scripts/process_dataset.py $dataset"
        fi
        exit 1
    fi

    rm -rf "$DATABASES_PATH/neo4j"
    rm -rf "$DATABASES_PATH/data/transactions/neo4j"

    echo "Importing dataset $dataset into Neo4j..."
    create_directory "$DATABASES_PATH/neo4j"
    # This is necessary to create an isolated environment for the CI
    # Check issues: https://github.com/orbdb/orb/pull/623 and https://github.com/orbdb/orb/issues/587.
    change_neo4j_database_path
    neo4j-admin database import full --nodes "$(pwd)/$node_header_file,$(pwd)/$node_file" \
                                     --relationships "$(pwd)/$rel_header_file,$(pwd)/$rel_file" \
                                     --overwrite-destination

    echo "========== Import $dataset into Neo4j...done."
    echo

    rm -rf ./import.report
}

start_neo4j() {
    echo "Starting Neo4j..."
    create_directory "$DATABASES_PATH/neo4j"

    neo4j start

    echo
    echo "Connect to Neo4j at http://localhost:7474"
    echo "Username: neo4j; Password: password123"
}

start_neo4j_docker() {
    echo "Starting Neo4j..."
    create_directory "$DATABASES_PATH/neo4j"

    docker run -d \
        --publish=7474:7474 --publish=7687:7687 \
        --env NEO4J_AUTH=none \
        --volume="$(pwd)/$DATABASES_PATH/neo4j:/data" \
        --name="$NEO4J_CONTAINER" \
        neo4j:5.16.0

    echo
    echo "Connect to Neo4j at http://localhost:7474"
    echo "Username: neo4j; Password: password123"
}

stop_neo4j() {
    neo4j stop
    echo "Stop Neo4j...done"
}

stop_neo4j_docker() {
    docker stop "$NEO4J_CONTAINER"
    docker rm "$NEO4J_CONTAINER"
    echo "Stop Neo4j...done"
}

# This is necessary to create an isolated environment for the CI
# Check issues: https://github.com/orbdb/orb/pull/623 and https://github.com/orbdb/orb/issues/587.
change_neo4j_database_path() {
    ABS_NEO4J_DATABASES_PATH=$(realpath "$DATABASES_PATH/neo4j")
    bash ./scripts/change_neo4j_db_path.sh "$ABS_NEO4J_DATABASES_PATH"
}

case $1 in
    neo4j)
        WITHOUT_DOCKER=false
        # determine by if user specified `--without-docker`
        if [[ "$*" == *"--without-docker"* ]]; then
            WITHOUT_DOCKER=true
        fi
        if [[ "$2" == "start" ]]; then
            if [ "$WITHOUT_DOCKER" = true ]; then
                start_neo4j
            else
                start_neo4j_docker
            fi
        elif [[ "$2" == "stop" ]]; then
            if [ "$WITHOUT_DOCKER" = true ]; then
                stop_neo4j
            else
                stop_neo4j_docker
            fi
        elif [[ "$2" == "import" ]]; then
            if [ "$WITHOUT_DOCKER" = true ]; then
                # Only used in combination with orb_dependency neo4j instance.
                # Get dataset name from command line arguments (first non-flag argument after "import")
                dataset=""
                # Find first non-flag argument (dataset name) - check $3 and $4
                if [ -n "$3" ] && [[ "$3" != "--without-docker" ]] && [[ ! "$3" =~ ^-- ]]; then
                    dataset="$3"
                elif [ -n "$4" ] && [[ "$4" != "--without-docker" ]] && [[ ! "$4" =~ ^-- ]]; then
                    dataset="$4"
                fi
                if [ -z "$dataset" ]; then
                    echo "Error: Dataset name is required"
                    echo "Usage: $0 neo4j import <dataset> [--without-docker]"
                    exit 1
                fi
                import_neo4j "$dataset"
            else
                echo "Error: Docker import not supported in conrad. Use --without-docker"
                exit 1
            fi
        elif [[ "$2" == "prepare" ]]; then
            prepare_neo4j
        else
            echo "Command unknown, use: $0 neo4j <start|stop|import|prepare>"
        fi
        ;;
    install_deps)
        install_deps "$2"  # Pass the second argument for CUDA/CPU selection
        ;;
    *)
        echo "Usage:
---------- Constants -----------
Local constants are read from <constants.env> at project root.
Otherwise default constants from <scripts/constants.env> are used.

----------- Tooling ------------

Setup Neo4j if not using Docker:
> $0 neo4j prepare

Install Python dependencies:
> $0 install_deps cpu|cuda

Imports dataset (Note: The dataset name is passed as an argument, overwrites existing data):
(Specify --without-docker to use Neo4j orb_dependency instance)
> $0 neo4j import [--without-docker]

Start or stop Neo4j (Specify --without-docker to use Neo4j orb_dependency instance manually installed):
> $0 neo4j start|stop [--without-docker]"
        ;;
esac
