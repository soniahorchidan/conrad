# !/bin/bash

# Define the new data path
NEW_DATA_PATH=$1

# Define the path to the neo4j.conf file
NEO4J_CONF="$NEO4J_HOME/conf/neo4j.conf"

# Check if the neo4j.conf file exists
if [ ! -f "$NEO4J_CONF" ]; then
    echo "neo4j.conf file not found at $NEO4J_CONF. Please check the path and try again."
    exit 1
fi

# Update the server.directories.data setting
if grep -q "^#*server.directories.data=" "$NEO4J_CONF"; then
    # Uncomment and modify the existing line
    perl -pi -e "s|^#*server.directories.data=.*|server.directories.data=$NEW_DATA_PATH|" "$NEO4J_CONF"
else
    # Add the line if it doesn't exist
    echo "server.directories.data=$NEW_DATA_PATH" >> "$NEO4J_CONF"
fi

echo "Neo4j database path updated to: $NEW_DATA_PATH"
