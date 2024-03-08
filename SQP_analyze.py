from SQP import classical_sqp, quantum_sqp
import pandas as pd

def read_test_problems(file_path):
    problems = []
    with open(file_path, 'r') as file:
        for line in file:
            name, classification = line.strip().split(' ', 1)  # Split on first space
            problems.append((name, classification))
    return problems

def run_one_experiment(problem_name, gx_tolerance, box, sample_number, api_key):
    classical_result = sqp(problem_name, gx_tolerance, box=box, sample_number=sample_number)
    quantum_result = sqp(problem_name, api_key, gx_tolerance, box=box, sample_number=sample_number)
    
    runtime_dif = classical_result['total_time'] - quantum_result['total_time']
    runtime_percent_dif = (runtime_dif / classical_result['total_time']) * 100
    minimum_dif = classical_result['minimum'] - quantum_result['minimum']
    
    return {
        'problem': problem_name,
        'dimension': classical_result['dimension'],
        'classical_Rtime': classical_result['total_time'],
        'quantum_Rtime': quantum_result['total_time'],
        'runtime_percent_dif': runtime_percent_dif,
        'classical_iteration': classical_result["iteration_num"],
        'quantum_iteration': quantum_result["iteration_num"],
        'classical_min': classical_result['minimum'],
        'quantum_min': quantum_result['minimum'],
        'minimum_dif': minimum_dif
    }
    
def store_results(results, file_name):
    df = pd.DataFrame(results)
    format_sci = lambda x: f"{x:.3e}"
    df[['classical_Rtime', 'quantum_Rtime', 'classical_min', 'quantum_min', 'minimum_dif']] = df[['classical_Rtime', 'quantum_Rtime', 'classical_min', 'quantum_min', 'minimum_dif']].map(format_sci)
    df[['runtime_percent_dif']] = df[['runtime_percent_dif']].round(3)
    df.to_csv(file_name, index=False)
    
def run_multiple_experiment(api_key, input_file, file, gx_tolerance = 1e-4, start_idx = 0, end_idx = 1000, box_size = 5, shots = 10, repeat = 1):
    problems = read_test_problems(input_file)
    end_idx = min(end_idx, len(problems) - 1)
    problems = problems[start_idx:end_idx + 1]
    results = []
    
    for name, classification in problems:
        for _ in range(repeat):
            result = run_one_experiment(name, gx_tolerance=gx_tolerance, box = box_size, sample_number = shots, api_key = api_key)
            result['classification'] = classification  
            results.append(result)
    
    store_results(results, file)

def main():
    ak = "DEV-7aa39ca5c55f857048c112d91c8b819ce75b525f"
    input_file = "Problems/i1.txt"
    file = "SQP_Results/test1.csv"
    sidx = 6
    eidx = 6
    box = 3
    guess_num = 10
    gx_tolerance = 1e-4
    run_multiple_experiment(ak, input_file, file, gx_tolerance=gx_tolerance, start_idx=sidx, end_idx=eidx, box_size=box, shots=guess_num, repeat=2)
    print("done")
    
if __name__ == '__main__':
    main()