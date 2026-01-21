#!/usr/bin/env python3
"""
Simple test script to verify that the saved calibration data can be loaded correctly.
This version avoids complex imports and just tests the pickle files directly.
"""

import os
import pickle
import torch


def test_calibration_data():
    """Test loading the saved calibration data files directly."""
    
    calibration_path = "./calibration_data/three_hop_pipeline"
    
    if not os.path.exists(calibration_path):
        print(f"❌ ERROR: Calibration data not found at {calibration_path}")
        print("Please run simple_calibration_sampler.py first to generate the data.")
        return False
    
    print(f" Found calibration data at: {calibration_path}")
    
    try:
        # Test loading metadata
        print("\n📊 Loading metadata...")
        metadata_path = os.path.join(calibration_path, "metadata.pkl")
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        
        print(f"  - Number of queries: {metadata['num_queries']}")
        print(f"  - Number of answers: {metadata['num_answers']}")
        print(f"  - Graph nodes: {metadata['graph_num_nodes']}")
        print(f"  - Graph edges: {metadata['graph_num_edges']}")
        print(f"  - Min hops: {metadata['min_hops']}")
        print(f"  - Max hops: {metadata['max_hops']}")
        
        # Test loading queries
        print("\n📋 Loading queries...")
        queries_path = os.path.join(calibration_path, "queries.pkl")
        with open(queries_path, "rb") as f:
            queries = pickle.load(f)
        
        print(f"  Loaded {len(queries)} queries")
        if queries:
            sample_query = queries[0]
            print(f"  Sample query: {sample_query}")
            print(f"  Query type: {type(sample_query)}")
            print(f"  Query length: {len(sample_query)}")
        
        # Test loading answers
        print("\n🎯 Loading answers...")
        answers_path = os.path.join(calibration_path, "answers.pkl")
        with open(answers_path, "rb") as f:
            answers = pickle.load(f)
        
        print(f"  Loaded {len(answers)} answer mappings")
        if queries:
            sample_query = queries[0]
            sample_answers = list(answers[sample_query])
            print(f"  Sample query: {sample_query}")
            print(f"  Sample answers: {sample_answers[:10]}..." if len(sample_answers) > 10 else f"  Sample answers: {sample_answers}")
            print(f"  Number of answers: {len(sample_answers)}")
        
        # Test loading graph data
        print("\n🕸️ Loading graph data...")
        graph_path = os.path.join(calibration_path, "graph_data.pkl")
        with open(graph_path, "rb") as f:
            graph_data = pickle.load(f)
        
        print(f"  Graph type: {type(graph_data)}")
        print(f"  Number of nodes: {graph_data.num_nodes}")
        print(f"  Number of edges: {graph_data.num_edges}")
        print(f"  Edge index shape: {graph_data.edge_index.shape}")
        print(f"  Edge type shape: {graph_data.edge_type.shape}")
        
        # Test query formatting
        print("\n🔧 Testing query formatting...")
        if queries:
            sample_query = queries[0]
            # Format query as expected by the model: [src, rel1, rel2, rel3] -> [[src, rel1, rel2, rel3]]
            formatted_query = [[sample_query[0]] + list(sample_query[1])]
            print(f"  Original query: {sample_query}")
            print(f"  Formatted query: {formatted_query}")
            
            # Convert to tensor
            import numpy as np
            query_tensor = torch.from_numpy(np.array(formatted_query))
            print(f"  Query tensor shape: {query_tensor.shape}")
            print(f"  Query tensor: {query_tensor}")
        
        # Test batch creation
        print("\n📦 Testing batch creation...")
        if len(queries) >= 4:
            batch_size = 4
            batch_queries = queries[:batch_size]
            batch_answers = [list(answers[q]) for q in batch_queries]
            
            # Format queries for batch
            formatted_batch = [[[q[0]] + list(q[1])] for q in batch_queries]
            import numpy as np
            batch_tensor = torch.concat([torch.from_numpy(np.array(q)) for q in formatted_batch], dim=0)
            
            print(f"  Batch size: {batch_size}")
            print(f"  Batch tensor shape: {batch_tensor.shape}")
            print(f"  Batch answers length: {len(batch_answers)}")
            print(f"  First batch query: {batch_tensor[0].tolist()}")
            print(f"  First batch answers: {batch_answers[0][:5]}..." if len(batch_answers[0]) > 5 else f"  First batch answers: {batch_answers[0]}")
        
        print(f"\n🎉 All tests passed! The calibration data is working correctly.")
        print(f"\n💡 Usage summary:")
        print(f"   - {metadata['num_queries']} calibration queries ready")
        print(f"   - Graph with {metadata['graph_num_nodes']} nodes and {metadata['graph_num_edges']} edges")
        print(f"   - All data properly formatted for model input")
        print(f"   - Data saved at: {calibration_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run the test."""
    
    success = test_calibration_data()
    
    if success:
        print(f"\n SUCCESS: Calibration data is ready to use!")
        exit(0)
    else:
        print(f"\n❌ FAILURE: There were issues with the calibration data.")
        exit(1)


if __name__ == "__main__":
    main()
