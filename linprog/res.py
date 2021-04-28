
import numpy as np
import json
from scipy.optimize import linprog

def json_read(path):
    with open(path, "r") as read_file:
        data = json.load(read_file)
    return [data["a_l"], data["b_l"], np.dot(data["c"],-1)]


a_l, b_l, c_new = json_read("data_file.json")
res_lin = linprog(c_new, a_l, b_l)
print(res_lin)
