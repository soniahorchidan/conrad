#!/bin/bash

# import constants
. ./scripts/constants.env

create_directory() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "Directory created: $dir"
    else
        echo "Directory already exists: $dir"
    fi
}

reactivate_conda_env_after_sourcing_rc_file() {
    # Remember the current Conda environment name
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        local current_env=$CONDA_DEFAULT_ENV
    fi

    determine_rc_file
    . $RC_FILE

    # Reactivate the Conda environment if it was previously set
    if [[ -n "$current_env" ]]; then
        conda activate $current_env
    fi
}

# Check if Neo4j is already installed
check_neo4j_installed() {
    if command -v neo4j >/dev/null 2>&1; then
        echo "Neo4j is already installed."
        return 0
    else
        return 1
    fi
}

# Check if Java is already installed
check_java_installed() {
    if command -v java >/dev/null 2>&1; then
        echo "Java is already installed."
        return 0
    else
        return 1
    fi
}

# Function to get installed Java version
get_java_version() {
    if command -v java >/dev/null 2>&1; then
        java -version 2>&1 | awk -F '"' '/version/ {print $2}'
    else
        echo "Java is not installed."
    fi
}

# Function to check if the correct Java version is installed for Neo4j
check_java_compatibility() {
    local neo4j_version="$1"
    local java_version
    java_version=$(get_java_version)

    case "$neo4j_version" in
        3.*)
            if ! version_gte "$java_version" "1.8" || version_gte "$java_version" "1.9"; then
                echo "Java 8 is required for Neo4j 3.x"
                return 1
            fi
            ;;
        4.*)
            if ! version_gte "$java_version" "11.0" || version_gte "$java_version" "12.0"; then
                echo "Java 11 is required for Neo4j 4.x"
                return 1
            fi
            ;;
        5.14)
            if ! version_gte "$java_version" "17.0" && ! version_gte "$java_version" "21.0"; then
                echo "Java 17 or 21 is required for Neo4j 5.14"
                return 1
            fi
            ;;
        5.*)
            if ! version_gte "$java_version" "17.0" || version_gte "$java_version" "18.0"; then
                echo "Java 17 is required for Neo4j 5.x"
                return 1
            fi
            ;;
        *)
            echo "Unsupported Neo4j version: $neo4j_version"
            return 1
            ;;
    esac
    
    return 0
}

# Function to compare versions
version_gte() {
    [ "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1" ]
}

# Function to add lines to rc file if they don't already exist
add_to_rc_file() {
    local line=$1
    if ! grep -Fxq "$line" "$RC_FILE"; then
        echo "$line" >> "$RC_FILE"
        echo "$line added to $RC_FILE"
    else
        echo "$line is already set in $RC_FILE"
    fi
}

# Function to determine the appropriate rc file based on the current shell
determine_rc_file() {
    local current_shell=$(basename $SHELL)
    case $current_shell in
        bash)
            RC_FILE="$HOME/.bashrc"
            ;;
        zsh)
            RC_FILE="$HOME/.zshrc"
            ;;
        ksh)
            RC_FILE="$HOME/.kshrc"
            ;;
        fish)
            RC_FILE="$HOME/.config/fish/config.fish"
            ;;
        *)
            echo "Unsupported shell: $current_shell"
            exit 1
            ;;
    esac
}

install_neo4j() {
    echo "Installing Neo4j..."
    PREV=$(pwd)
    cd $LOCAL_NEO4J_PATH
    wget https://dist.neo4j.org/neo4j-community-5.20.0-unix.tar.gz
    tar -xzf neo4j-community-5.20.0-unix.tar.gz
    rm neo4j-community-5.20.0-unix.tar.gz
    
    # Path to the Neo4j configuration file
    NEO4J_CONF_PATH="./neo4j-community-5.20.0/conf/neo4j.conf"

    # Check if the configuration file exists
    if [[ -f "$NEO4J_CONF_PATH" ]]; then
        # Use sed to find and replace the commented line to disable authentication
        sed -i.bak 's/^#dbms.security.auth_enabled=false/dbms.security.auth_enabled=false/' "$NEO4J_CONF_PATH"
        
        # Verify if the change was successful
        if grep -q "^dbms.security.auth_enabled=false" "$NEO4J_CONF_PATH"; then
            echo "Authentication disabled successfully in $NEO4J_CONF_PATH."
        else
            echo "Failed to disable authentication. Please check the configuration file."
        fi
    else
        echo "Configuration file not found at $NEO4J_CONF_PATH."
        exit 1
    fi

    cd $PREV
    # Add Neo4j environment variables to rc file
    add_to_rc_file "export NEO4J_HOME=$LOCAL_NEO4J_PATH/neo4j-community-5.20.0"
    add_to_rc_file "export PATH=\$NEO4J_HOME/bin:\$PATH"
}

# Determine the appropriate rc file
determine_rc_file

# Create the installation directory
create_directory "$LOCAL_NEO4J_PATH"

LOCAL_NEO4J_PATH=$(realpath $LOCAL_NEO4J_PATH)

# Install Neo4j if not already installed
if ! check_neo4j_installed; then
    # Install JDK 21 if not already installed
    if ! check_java_installed; then
        echo "Java is not installed. Please install JDK 21."
        exit 1
    else
        # Check Java compatibility for Neo4j
        if ! check_java_compatibility "5.14"; then
            echo "Incompatible Java version for Neo4j. Please install JDK 21."
            exit 1
        else
            echo "Java version compatible with Neo4j."
            install_neo4j
        fi
    fi
else
    echo "Skipping Neo4j installation."
fi

# Source the rc file to apply changes
echo "Done! Please run 'source $RC_FILE' or open a new terminal session to apply the changes."