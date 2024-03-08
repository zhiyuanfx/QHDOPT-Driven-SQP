from Problems.Julia_Interface.CUTEst_Extractor import extract_model
import numpy as np 
from qhdopt import QHD


def check_stop(gx, gx_tolerance, cur_min, pre_min):
    return np.linalg.norm(np.array(gx), ord=2) <= gx_tolerance or cur_min > pre_min

def process_original_bounds(nlp):
    """
    Returns the bounds (as list of tuples) of the given optimization problem.
    Python Infinity for "None" Bounds
    """
    bounds = []
    lb, ub = nlp["meta"]["lvar"], nlp["meta"]["uvar"]
    for lower, upper in zip(lb, ub):
        
        if lower is None:
            lower_bound = -np.inf
        else:
            lower_bound = lower
        
        if upper is None:
            upper_bound = np.inf
        else:
            upper_bound = upper
        
        bounds.append((lower_bound, upper_bound))
    
    return bounds

def get_x0(x0, bounds):
    feasible_x0 = []
    for x, (lower, upper) in zip(x0, bounds):
        if x < lower:
            feasible_x = lower
        elif x > upper:
            feasible_x = upper
        else:
            feasible_x = x
        feasible_x0.append(feasible_x)
    return feasible_x0

def get_new_bound(x, old_bound, box_size = 5):
    """
    Returns the intersection between overall box of original problem and the 
    smaller box around current solution x
    """
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    
    for i, val in enumerate(x):
        if val < old_bound[i][0] or val > old_bound[i][1]:
            raise ValueError("x is not feasible")
    
    new_lower_bound = x - box_size
    new_upper_bound = x + box_size
    
    adjusted_lower_bounds = []
    adjusted_upper_bounds = []
    
    for i, (new_lb, new_ub) in enumerate(zip(new_lower_bound, new_upper_bound)):
        # Overall bounds for the current dimension
        overall_lb, overall_ub = old_bound[i]
        
        # Calculate the intersection of the new bounds with the overall bounds
        adj_lb = max(new_lb, overall_lb)
        adj_ub = min(new_ub, overall_ub)
        
        adjusted_lower_bounds.append(adj_lb)
        adjusted_upper_bounds.append(adj_ub)
    
    adjusted_bounds = list(zip(adjusted_lower_bounds, adjusted_upper_bounds))
    return adjusted_bounds

def get_d_bound(x, x_bound):
    """
    Returns the bound for the search direction d to matain x + d feasible
    """
    return [(x_bound[i][0] - xi, x_bound[i][1] - xi) for i, xi in enumerate(x)]

def update_x(x, d, x_bound):
    x_next = np.array(x) + np.array(d)
    x_next = np.clip(x_next, a_min = [b[0] for b in x_bound], a_max = [b[1] for b in x_bound]) 
    return x_next.tolist()

def get_total_time(res):
    if res.info["backend_time"] == 0:
        return res.info['fine_tuning_time']
    else:
        total_runtime = res.info["time_on_machine"]
        total_runtime += res.info['fine_tuning_time'] if res.info["fine_tune_status"] else 0
        return total_runtime

def sqp(problem_name, solver, api_key = None, gx_tolerance = 1e-4, max_box = 5, sample_number = 10):
    """
    Returns the result of sqp using classical or qhdopt solver as a map with keys:
    dimension, solution, minimum, total_time, and iteration_num
    """
    m = extract_model(problem_name)
    nlp = m["nlp"]
    old_bound = process_original_bounds(nlp)
    x = get_x0(nlp["meta"]["x0"], old_bound)
    m = extract_model(problem_name, np.array(x))
    Hx, min, gx = (m[key] for key in ("Hx", "fx", "gx"))
    
    box = 3
    total_time = 0
    iteration_num = 0
    result = dict()
    # visited = set()
    # visited.add(tuple(np.round(x, 4)))
    
    print(problem_name, "cls")
    while True:
        iteration_num += 1
        print(iteration_num, x, gx)
        x_bound = get_new_bound(x, old_bound, box_size = box)
        d_bound = get_d_bound(x, x_bound)
        model = QHD.QP(Hx, gx, bounds = d_bound)
        if solver == "classical":
            response = model.classically_optimize(initial_guess = sample_number)
        elif solver == "qhdopt":
            if not api_key:
                raise ValueError("API key is required for qhdopt solver.")
            model.dwave_setup(resolution = 8, api_key = api_key, shots = sample_number) 
            response = model.optimize()
        else:
            raise ValueError(f"Unsupported solver '{solver}'. Use 'classical' or 'qhdopt'.")
        total_time += get_total_time(response)
        d = response.minimizer
        xtemp = update_x(x, d, x_bound)
        m = extract_model(problem_name, np.array(xtemp))
        Hxtemp, mintemp, gxtemp = (m[key] for key in ("Hx", "fx", "gx"))
        rho = (min - mintemp) / (-d.T @ gx - 0.5 * (d.T @ Hx @ d))
        if rho < 0.25:
            box *= 0.25
        elif rho > 0.75 and abs(np.linalg.norm(np.array(d), ord=2) - box) < gx_tolerance:
            box = min(2 * box, max_box)
        if rho > 0:
            x = xtemp
            Hx = Hxtemp
            temp = min
            min = mintemp
            gx = gxtemp
            if abs(temp - mintemp) < gx_tolerance:
                break
        # t = tuple(np.round(xtemp, 4))
        # if t in visited:
        #     break
        # visited.add(t)
        if np.linalg.norm(np.array(gx), ord=2) <= gx_tolerance:
            break
            
    result["dimension"] = len(nlp["meta"]["x0"])
    result["solution"] = x
    result["minimum"] = min
    result["total_time"] = total_time
    result["iteration_num"] = iteration_num
    return result