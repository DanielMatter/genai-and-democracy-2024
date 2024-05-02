import argparse
import subprocess
import sys
from random import sample
from os import mkdir, rmdir
from os.path import join, isfile, isdir, split as split_path
import shutil
import json

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def nanoid(n=10):
    return ''.join(sample('abcdefghijklmnopqrstuvwxyz', n))

def tranform_output(output):
    lines = output.split('\n')
    lines = map(lambda x: x.strip(), lines)
    lines = filter(lambda x: x != '', lines)
    return "◦◦◦ " + "\n◦◦◦ ".join(lines)

def test_setup():
    result = subprocess.run([sys.executable, "user_setup.py"], check=False, capture_output=True)
    if result.returncode != 0:
        print(bcolors.FAIL + "> The setup script did not run successfully.")
        print(bcolors.FAIL + tranform_output(result.stderr.decode()))
    
    else:
        print(bcolors.OKGREEN + "> The setup script ran successfully.")
        print(bcolors.OKBLUE + tranform_output(result.stdout.decode()))
    

def test_preprocess():
    output_dir = nanoid()
    mkdir(output_dir)
    
    args = [ "--output", output_dir ]
    articles = ["sample_data/article_1.json"]
    
    for article in articles:
        args.append("--input")
        args.append(article)

    
    result = subprocess.run([sys.executable, "user_preprocess.py", *args], check=False, capture_output=True)
    if result.returncode != 0:
        print(bcolors.FAIL + "> The preprocess script did not run successfully.")
        print(bcolors.FAIL + tranform_output(result.stderr.decode()))
    
    else:
        try:
            for article in articles:
                assert isfile(join(output_dir, split_path(article)[-1])), f"The file {split_path(article)[-1]} was not created."
                with open(join(output_dir, split_path(article)[-1]), "r") as f:
                    data = json.load(f)
                    assert "transformed_representation" in data, f"The key 'transformed_representation' was not found in the file {split_path(article)[-1]}."
        
            print(bcolors.OKGREEN + "> The preprocess script ran successfully.")
            print(bcolors.OKBLUE + tranform_output(result.stdout.decode()))
        except Exception as e:
            print(bcolors.FAIL + "> The preprocess script did not create the expected output.")
            print(bcolors.FAIL + tranform_output(str(e)))

    shutil.rmtree(output_dir, ignore_errors=True)


def test_inference():
    out_dir = nanoid()
    mkdir(out_dir)
    
    args = [ "--output", out_dir ]
    queries = ["sports", "soccer", "Munich vs Dortmund"]
    for query in queries:
        args.append("--query")
        args.append(query)
    
    query_ids = [nanoid() for _ in queries]
    for query_id in query_ids:
        args.append("--query_id")
        args.append(query_id)
    
    result = subprocess.run([sys.executable, "user_inference.py", *args], check=False, capture_output=True)
    if result.returncode != 0:
        print(bcolors.FAIL + "> The inference script did not run successfully.")
        print(bcolors.FAIL + tranform_output(result.stderr.decode()))
    
    else:
        try: 
            for query_id in query_ids:
                assert isfile(join(out_dir, f"{query_id}.json")), f"The file {query_id}.json was not created."
                with open(join(out_dir, f"{query_id}.json"), "r") as f:
                    filedata = json.load(f)
                    assert "detected_language" in filedata, f"The key 'detected_language' was not found in the file {query_id}.json."
                    assert "generated_query" in filedata, f"The key 'generated_query' was not found in the file {query_id}.json."
                    
            print(bcolors.OKGREEN + "> The inference script ran successfully.")
            print(bcolors.OKBLUE + tranform_output(result.stdout.decode()))
        
        except Exception as e:
            print(bcolors.FAIL + "> The inference script did not create the expected output.")
            print(bcolors.FAIL + tranform_output(str(e)))
        
    shutil.rmtree(out_dir, ignore_errors=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess the data.')
    parser.add_argument('--part', type=str, help='Which part of the pipeline to test.', required=True, choices=['preprocess', 'setup', 'inference'])
    
    args = parser.parse_args()
    if args.part == 'preprocess':
        print(bcolors.OKCYAN + "> This is the preprocess part.")
        test_preprocess()
        
    elif args.part == 'setup':
        print(bcolors.OKCYAN + "> This is the setup part.")
        test_setup()
        
    elif args.part == 'inference':
        print(bcolors.OKCYAN + "> This is the inference part.")
        test_inference()