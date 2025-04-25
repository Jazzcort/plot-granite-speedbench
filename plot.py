import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import csv
import math
from enum import Enum
from typing import List

class DataType(Enum):
    TTFT = "time_to_first_token"
    TPS = "tokens_per_sec"

home_dir = os.path.expanduser("~")
path = f"{home_dir}/.granite-speedbench/output"

def quadratic_model(x, a, b, c):
    return a * np.square(x) + b * x + c

def linear_model(x, a, b):
    return a * x + b

def plot_result(x_data, y_data, data_type: DataType, fit_model, out_path):
    popt, _ = curve_fit(fit_model, x_data, y_data)
    x_limit, y_limit = np.max(x_data), np.max(y_data)
    x_limit, y_limit = x_limit * 1.05, y_limit * 1.05
    unit = "seconds" if data_type == DataType.TTFT else "seconds/token"
    data_label = "ttft" if data_type == DataType.TTFT else "1/tps"

    x_fit = np.linspace(1, int(x_limit), 100)
    y_fit = fit_model(x_fit, *popt)

    # Clean the plot
    plt.clf()

    plt.plot(x_data, y_data, 'o', label="data")
    plt.plot(x_fit, y_fit, '-', label="fit")
    plt.xlabel("num_tokens (tokens)")
    plt.ylabel(f"{data_label} ({unit})")
    plt.title(f"num_tokens vs {data_label}")
    plt.xlim(0, x_limit)
    plt.ylim(0, y_limit)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{out_path}")

def read_csv(path: str):
    res = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            res.append(row)
    return res

def extract_data(sample_lst, data_type: DataType):
    num_tokens = np.array([])
    data = np.array([])
    try:
        for row in sample_lst:
            num_tokens = np.append(num_tokens, row["num_tokens"])
            data = np.append(data, row[data_type.value])
    except KeyError:
        return None

    num_tokens, data = num_tokens.astype(float), data.astype(float)

    if data_type == DataType.TPS:
        data = np.reciprocal(data)

    return num_tokens, data

def extract_model_name(input_filename: str):
    name_parts = input_filename.split(".")
    name_parts.pop()
    return ".".join(name_parts)

def generate_output_filename(model_name: str, data_type: DataType):
    return model_name + "-" + data_type.value + ".png"

def calculate_mean(data: List[float]):
    return sum(data) / len(data) 

def calculate_standard_error(data: List[float], precalculated_mean=None):
    if not data or len(data) < 2:
        raise ValueError("At least two data points are required to calculate standard error.")

    mean = calculate_mean(data) if not precalculated_mean else precalculated_mean
    squared_diffs = [ (x - mean) ** 2 for x in data ]
    variance = sum(squared_diffs) / (len(data) - 1)

    standard_deviation = math.sqrt(variance)

    return standard_deviation / math.sqrt(len(data))

def calculate_and_print_result(ttft_data, tps_data, cur_id):
    if cur_id:
        ttft_mean = calculate_mean(ttft_data)
        ttft_std_err = calculate_standard_error(ttft_data, ttft_mean)
        tps_mean = calculate_mean(tps_data)
        tps_std_err = calculate_standard_error(tps_data, tps_mean)
        print(f"{cur_id}")
        print(f"Time to first token -> mean: {ttft_mean:.3f} seconds, standard error: {ttft_std_err:.3f}")
        print(f"Tokens per second -> mean: {tps_mean:.3f} seconds, standard error: {tps_std_err:.3f}")

def generate_sample_mean_and_standard_error(sample_lst):
    cur_id = ""
    ttft = []
    tps = []
    try:
        for row in sample_lst:
            if row["ID"] == cur_id:
                ttft.append(float(row[DataType.TTFT.value]))
                tps.append(float(row[DataType.TPS.value]))
            else:
                calculate_and_print_result(ttft, tps, cur_id)
                cur_id = row["ID"]
                ttft = [ float(row[DataType.TTFT.value]) ]
                tps = [ float(row[DataType.TPS.value]) ]

        calculate_and_print_result(ttft, tps, cur_id)
    except ValueError:
        pass
    except KeyError:
        pass

def main():

    if not os.path.isdir(path):
        print(f"Can't find output directory: {path}")
        return 
    
    for root, _, files in os.walk(path):
        if root != path:
            continue

        for file in files:
            if file.endswith(".csv"):
                is_invalid = False
                sample_lst = read_csv(f"{root}/{file}")
                model_name = extract_model_name(file)

                # Calculate mean and standard error
                print(f"{model_name}")
                generate_sample_mean_and_standard_error(sample_lst)
                print("************************************************************************")

                # Plot num_tokens vs time_to_first_token
                res = extract_data(sample_lst, DataType.TTFT)
                if res:
                    x_data, y_data = res
                    out_file = generate_output_filename(model_name, DataType.TTFT)
                    plot_result(x_data, y_data, DataType.TTFT, quadratic_model, f"{root}/{out_file}")
                else:
                    is_invalid = True

                # Plot num_tokens vs tokens_per_sec
                res = extract_data(sample_lst, DataType.TPS)
                if res:
                    x_data, y_data = res
                    out_file = generate_output_filename(model_name, DataType.TPS)
                    plot_result(x_data, y_data, DataType.TPS, linear_model, f"{root}/{out_file}")
                else:
                    is_invalid = True

                if is_invalid:
                    print(f"Invalid csv format: {file}")

    print(f"Output Directiry: {path}")

if __name__ == "__main__":
    main()


