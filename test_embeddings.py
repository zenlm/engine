#!/usr/bin/env python3
"""Test script for the Hanzo Engine embeddings endpoint."""

import requests
import json
import sys

def test_embeddings():
    """Test the embeddings endpoint with various inputs."""
    
    # Test cases
    test_cases = [
        {
            "name": "Single string",
            "payload": {
                "input": "Hello, world!",
                "model": "snowflake-arctic-embed-l"
            }
        },
        {
            "name": "Array of strings",
            "payload": {
                "input": ["Hello", "World", "Testing embeddings"],
                "model": "snowflake-arctic-embed-l"
            }
        },
        {
            "name": "With dimensions",
            "payload": {
                "input": "Test with dimension reduction",
                "dimensions": 512
            }
        }
    ]
    
    base_url = "http://localhost:36900"
    endpoint = f"{base_url}/v1/embeddings"
    
    print(f"Testing embeddings endpoint at {endpoint}\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        print(f"Payload: {json.dumps(test['payload'], indent=2)}")
        
        try:
            response = requests.post(
                endpoint,
                json=test['payload'],
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                assert "object" in data, "Missing 'object' field"
                assert data["object"] == "list", f"Expected object='list', got '{data['object']}'"
                assert "data" in data, "Missing 'data' field"
                assert isinstance(data["data"], list), "Data should be a list"
                
                if len(data["data"]) > 0:
                    embedding_item = data["data"][0]
                    assert "embedding" in embedding_item, "Missing 'embedding' field"
                    assert isinstance(embedding_item["embedding"], list), "Embedding should be a list"
                    
                    # Check if it's not a placeholder
                    embedding = embedding_item["embedding"]
                    if len(embedding) > 0:
                        # Check if values are not all the same (placeholder)
                        unique_values = set(embedding[:min(10, len(embedding))])
                        if len(unique_values) == 1 and 0.09 < list(unique_values)[0] < 0.11:
                            print("  ⚠️  WARNING: Appears to be placeholder embeddings (all values ~0.1)")
                        else:
                            print(f"  ✓ SUCCESS: Got real embeddings with {len(embedding)} dimensions")
                            print(f"    First 5 values: {embedding[:5]}")
                            print(f"    Unique values in first 10: {len(set(embedding[:10]))}")
                    
                    # Check dimensions if specified
                    if "dimensions" in test["payload"]:
                        expected_dim = test["payload"]["dimensions"]
                        actual_dim = len(embedding)
                        if actual_dim == expected_dim:
                            print(f"  ✓ Dimensions correct: {actual_dim}")
                        else:
                            print(f"  ✗ Dimension mismatch: expected {expected_dim}, got {actual_dim}")
                
                print(f"  Model: {data.get('model', 'unknown')}")
                if "usage" in data:
                    print(f"  Usage: {data['usage']}")
                    
            else:
                print(f"  ✗ ERROR: Status code {response.status_code}")
                print(f"  Response: {response.text}")
                
        except requests.ConnectionError:
            print(f"  ✗ ERROR: Could not connect to {endpoint}")
            print("  Make sure the server is running: cargo run --package hanzo-engine -- serve")
            return False
        except requests.Timeout:
            print(f"  ✗ ERROR: Request timed out")
            return False
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            return False
            
        print()
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Hanzo Engine Embeddings Test")
    print("=" * 60)
    
    success = test_embeddings()
    
    print("=" * 60)
    if success:
        print("✓ All tests completed")
    else:
        print("✗ Tests failed")
        sys.exit(1)