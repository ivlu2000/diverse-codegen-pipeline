import csv
import re
import sys
import argparse

def rank_all_results(input_file_path):
    """
    Parses, ranks, and displays all runs from an evaluation CSV, highlighting the best one.

    Args:
        input_file_path (str): The path to the input CSV file.
    """
    # Regex for base tests only
    humaneval_re = re.compile(r"humaneval \(base tests\) pass@1:\s*([\d\.]+)")
    humaneval_plus_re = re.compile(r"humaneval\+ \(base \+ extra tests\) pass@1:\s*([\d\.]+)")
    mbpp_re = re.compile(r"mbpp \(base tests\) pass@1:\s*([\d\.]+)")
    mbpp_plus_re = re.compile(r"mbpp\+ \(base \+ extra tests\) pass@1:\s*([\d\.]+)")

    best_mean_score = -1.0
    # Use the run directory as a unique identifier for the best run
    best_run_identifier = None
    
    all_runs_data = []

    try:
        with open(input_file_path, 'r', newline='') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                scores = []
                # Find humaneval score
                he_match = humaneval_re.search(row.get('eval_humaneval', ''))
                if he_match:
                    try:
                        scores.append(float(he_match.group(1)))
                    except (ValueError, IndexError):
                        pass

                # Find humaneval+ score
                he_plus_match = humaneval_plus_re.search(row.get('eval_humaneval', ''))
                if he_plus_match:
                    try:
                        scores.append(float(he_plus_match.group(1)))
                    except (ValueError, IndexError):
                        pass

                # Find mbpp score
                mbpp_match = mbpp_re.search(row.get('eval_mbpp', ''))
                if mbpp_match:
                    try:
                        scores.append(float(mbpp_match.group(1)))
                    except (ValueError, IndexError):
                        pass

                # Find mbpp+ score
                mbpp_plus_match = mbpp_plus_re.search(row.get('eval_mbpp', ''))
                if mbpp_plus_match:
                    try:
                        scores.append(float(mbpp_plus_match.group(1)))
                    except (ValueError, IndexError):
                        pass

                # Calculate mean and check if this is the new best run
                current_mean = -1.0
                if scores:
                    current_mean = sum(scores) / len(scores)
                    if current_mean > best_mean_score:
                        best_mean_score = current_mean
                        best_run_identifier = row.get('run_dir')
                
                # Store the parsed data for every run
                all_runs_data.append({
                    'r': row.get('r'),
                    'alpha': row.get('alpha'),
                    'target_modules': row.get('target_modules'),
                    'run_dir': row.get('run_dir'),
                    'humaneval_pass@1': he_match.group(1) if he_match else 'N/A',
                    'humaneval_plus_pass@1': he_plus_match.group(1) if he_plus_match else 'N/A',
                    'mbpp_pass@1': mbpp_match.group(1) if mbpp_match else 'N/A',
                    'mbpp_plus_pass@1': mbpp_plus_match.group(1) if mbpp_plus_match else 'N/A',
                    'mean_score': f"{current_mean:.4f}" if current_mean != -1.0 else 'N/A'
                })

    except FileNotFoundError:
        print(f"Error: The file '{input_file_path}' was not found.", file=sys.stderr)
        return
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        return

    # Print the results table if we have data
    if all_runs_data:
        print("--- All Processed Runs ---")
        # Define table headers
        header = f"{'Highlight':<13} | {'r':<4} | {'alpha':<5} | {'target_modules':<11} | {'save_head':<11} | {'HumanEval':<11} | {'HumanEval+':<11} | {'MBPP':<11} | {'MBPP+':<11} | {'Mean Score':<12}"
        print(header)
        print("-" * len(header))

        best_run_details = None
        for run in all_runs_data:
            highlight = "🏆 BEST -> " if run['run_dir'] == best_run_identifier else " " * 13
            if run['run_dir'] == best_run_identifier:
                best_run_details = run # Save the full details of the best run

            # Print formatted row
            print(f"{highlight} | {run['r']:<4} | {run['alpha']:<5} | {run['target_modules']:<11} | {run['humaneval_pass@1']:<11} | {run['humaneval_plus_pass@1']:<11} | {run['mbpp_pass@1']:<11} | {run['mbpp_plus_pass@1']:<11} | {run['mean_score']:<12}")

        print("\n" + "="*40)
        print("🏆 Detailed Best Run Summary 🏆")
        print("="*40)
        if best_run_details:
             print(f"Mean Pass@1 Score: {best_run_details['mean_score']}")
             print(f"  - HumanEval Score: {best_run_details['humaneval_pass@1']}")
             print(f"  - HumanEval+ Score: {best_run_details['humaneval_plus_pass@1']}")
             print(f"  - MBPP Score:      {best_run_details['mbpp_pass@1']}")
             print(f"  - MBPP+ Score:      {best_run_details['mbpp_plus_pass@1']}")
             print("\nConfiguration:")
             print(f"  - r: {best_run_details['r']}")
             print(f"  - alpha: {best_run_details['alpha']}")
             print(f"  - target_modules: {best_run_details['target_modules']}")
             print(f"  - run_dir: {best_run_details['run_dir']}")
        else:
            print("No best run could be determined.")

    else:
        print("Could not find any valid results in the file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rank and display all humaneval/mbpp results, highlighting the best."
    )
    parser.add_argument(
        "input_file", 
        nargs='?', 
        default="sweep_results.csv",
        help="Path to the input CSV file (default: sweep_results.csv)"
    )
    args = parser.parse_args()
    
    rank_all_results(args.input_file)