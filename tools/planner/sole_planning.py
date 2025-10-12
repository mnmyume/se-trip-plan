import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from agents.prompts import planner_agent_prompt, transportation_agent_prompt, attraction_agent_prompt, accommodation_agent_prompt, restaurant_agent_prompt, combination_agent_prompt
import json
from tqdm import tqdm
from tools.planner.apis import Planner
import argparse
from datasets import load_dataset


def load_line_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.read().strip().split('\n'):
            unit = json.loads(line)
            data.append(unit)
    return data

def extract_numbers_from_filenames(directory):
    # Define the pattern to match files
    pattern = r'annotation_(\d+).json'

    # List all files in the directory
    files = os.listdir(directory)

    # Extract numbers from filenames that match the pattern
    numbers = [int(re.search(pattern, file).group(1)) for file in files if re.match(pattern, file)]

    return numbers


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--set_type", type=str, default="test")
    parser.add_argument("--model_name", type=str, default="gemma-3-27b-it")
    parser.add_argument("--output_dir", type=str, default="../../outputs/output")
    parser.add_argument("--strategy", type=str, default="direct")
    args = parser.parse_args()
    directory = f'{args.output_dir}/{args.set_type}'
    if args.set_type == 'train':
        query_data_list  = load_dataset('osunlp/TravelPlanner','train')['train']
    elif args.set_type == 'validation':
        query_data_list  = load_dataset('osunlp/TravelPlanner','validation')['validation']
    elif args.set_type == 'test':
        query_data_list  = load_dataset('osunlp/TravelPlanner','test')['test']
    numbers = [i for i in range(1,len(query_data_list)+1)]

    if args.strategy == 'direct':
        planner = Planner(model_name=args.model_name,
                          planner_prompt=planner_agent_prompt,
                          transportation_prompt=transportation_agent_prompt,
                          attraction_prompt=attraction_agent_prompt,
                          accommodation_prompt=accommodation_agent_prompt,
                          restaurant_prompt=restaurant_agent_prompt,
                          combination_prompt=combination_agent_prompt)


    for number in tqdm(numbers[:]):

        query_data = query_data_list[number-1]
        reference_information = query_data['reference_information']
        while True:
                if args.strategy in ['react','reflexion']:
                    planner_results, scratchpad  = planner.run(reference_information, query_data['query'])
                else:
                    planner_results  = planner.run(reference_information, query_data['query'])
                if planner_results != None:
                    break
        print(planner_results.replace("\n", " ")[:100])
        # check if the directory exists
        if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}')):
            os.makedirs(os.path.join(f'{args.output_dir}/{args.set_type}'))
        if not os.path.exists(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json')):
            result =  [{}]
        else:
            result = json.load(open(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json')))
        if args.strategy in ['react','reflexion']:
            result[-1][f'{args.model_name}_{args.strategy}_sole-planning_results_logs'] = scratchpad
        result[-1][f'{args.model_name}_{args.strategy}_sole-planning_results'] = planner_results
        # write to json file
        with open(os.path.join(f'{args.output_dir}/{args.set_type}/generated_plan_{number}.json'), 'w') as f:
            json.dump(result, f, indent=4)
