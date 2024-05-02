# This file will be executed when a user wants to query your project.
import argparse
from os.path import join
import json

# TODO Implement the inference logic here
def handle_user_query(query, query_id, output_path):
    result = {
        "generated_queries": [ "sports", "soccer", "Munich vs Dortmund" ],
        "detected_language": "de",
    }
    
    with open(join(output_path, f"{query_id}.json"), "w") as f:
        json.dump(result, f)
    

# This is a sample argparse-setup, you probably want to use in your project:
parser = argparse.ArgumentParser(description='Run the inference.')
parser.add_argument('--query', type=str, help='The user query.', required=True, action="append")
parser.add_argument('--query_id', type=str, help='The IDs for the queries, in the same order as the queries.', required=True, action="append")
parser.add_argument('--output', type=str, help='Path to the output directory.', required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    queries = args.query
    query_ids = args.query_id
    output = args.output
    
    assert len(queries) == len(query_ids), "The number of queries and query IDs must be the same."
    
    for query, query_id in zip(queries, query_ids):
        handle_user_query(query, query_id, output)
    