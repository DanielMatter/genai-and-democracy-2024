# This file will be executed when a user wants to query your project.
import argparse
from os.path import join
import json
import requests as r

# TODO Implement the inference logic here
def handle_user_query(query, query_id, output_path):
    call_data = {
        "model": "llama3",
        "prompt": "User Query: " + query,
        "system": "You are a helpful assistant working at the EU. It is your job to give users unbiased article recommendations. You will be prompted with a user generated search query. Try to improve it by adding more context, such that a recommendation system has an easier time to find relevant articles. Response with the improved query only. The query should be unbiased and in English. The query is given as the next message. Only answer with the improved query. After the query, append LANGUAGE: [lang_code] where you use the default two-letter language code in a new line.",
        "stream": False
    }
    response = r.post("http://localhost:11434/api/generate", json=call_data)
    response = response.json()
    response_text = response["response"]
    
    new_query, language_code = response_text.split("LANGUAGE: ")
    new_query = new_query.strip()
    language_code = language_code.strip()
    
    result = {
        "generated_queries": [ new_query ],
        "detected_language": language_code,
    }
    
    print(response_text)
    
    with open(join(output_path, f"{query_id}.json"), "w") as f:
        json.dump(result, f)

# TODO OPTIONAL
# This function is optional for you
# You can use it to interfer with the default ranking of your system.
#
# If you do embeddings, this function will simply compute the cosine-similarity
# and return the ordering and scores
def rank_articles(generated_queries, article_representations):
    """
    This function takes as arguments the generated / augmented user query, as well as the
    transformed article representations.
    
    It needs to return a list of shape (M, 2), where M <= #article_representations.
    Each tuple contains [index, score], where index is the index in the article_repr array.
    The list need already be ordered by score. Higher is better, between 0 and 1.
    
    An empty return list indicates no matches.
    """
    
    article_string = [ f"Article {index}: {json.dumps(article)}" for index, article in enumerate(article_representations) ]
    call_data = {
        "model": "llama3",
        "prompt": "User Query: " + json.dumps(generated_queries) + "\n\n" + "Articles: " + "\n".join(article_string),
        "system": "You are a helpful assistant working at the EU. It is your job to give users unbiased article recommendations. You will be given a user query, and a list of articles. Your task is to rank the articles by relevance to the query. Response with a list of article indices, separated by commas. The first index should be the most relevant article, the last the least relevant. The indices should be 0-based. If there are multiple queries, it is only variations of one original one. Only return the rankings as valid JSON, for example [ 3, 5, 9 ]. Do not give me anything else, nor any explanation.",
        "stream": False
    }
    response = r.post("http://localhost:11434/api/generate", json=call_data)
    response = response.json()
    response_text = response["response"]
    
    
    print(response_text)    
    
    return []


if True:
    # handle_user_query("What are the benefits of LLMs in programming?", "1", "output")
    rank_articles(["What are the benefits of LLMs in programming?"], [[ "llms", "ai", "programming" ], [ "war in ukraine", "russia", "ukraine"]])
    exit(0)
    
exit(0)



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
    