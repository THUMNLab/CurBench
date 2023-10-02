import os
import re

# 定义文件夹路径
runs_folder = "runs"

mapping = {"bert": "3", "gpt": "3", "lstm": "10", "vit": "200", "resnet18": "200", "lenet": "200", "gcn": "200", "gat": "200", "sage": "200"}

# 初始化存储结果的字典
results = {}

# 定义正则表达式模式，匹配文件夹名
pattern = re.compile(r"^([\w]+)-([\w]+)-(noise-0\.4-|imbalance-50-)?([\w]+)-([\d]+)-([\d]+)$")
# print(len(os.listdir(runs_folder)))
# 遍历文件夹
for folder_name in os.listdir(runs_folder):
    # print(folder_name)
    match = pattern.match(folder_name)
    # 检查文件夹名是否符合规则
    if not match:
        print("Not Match")
        continue
    folder_path = os.path.join(runs_folder, folder_name)
    # 检查是否是文件夹
    if not os.path.isdir(folder_path):
        print("Not folder")
        continue
    
    parts = folder_name.split("-")
    method = parts[0]
    dataset = parts[1]
    setting = "standard"
    if "noise" in folder_name:
        setting = "noise"
    elif "imbalance" in folder_name:
        setting = "imbalance"
    backbone = parts[-3]
    epoch = parts[-2]
    seed = parts[-1]
    
    log_file_path = os.path.join(folder_path, "train.log")
    # 检查log文件是否存在
    if not os.path.exists(log_file_path):
        # print("No log file")
        continue
    
    # 排除epoch异常的测试实验
    if mapping[backbone] != epoch:
        print("Wrong epoch setting: %s" % (folder_name))
        continue
    
    # mnli有两个验证集，暂时不看
    if dataset == "mnli":
        # print("Mnli")
        continue

    # print(log_file_path)
    # 读取log文件中的指标
    log_pattern = re.compile(r".*?Final.*=\s*([\d.]+)")
    with open(log_file_path, "r") as file:
        for line in file:
            pattern_match = log_pattern.search(line)
            if pattern_match:
                # print(line.strip())
                metric = float(pattern_match.group(1))
                # config = (setting, dataset, backbone, method, seed)
                # results[config] = metric
                config = (setting, dataset, backbone)
                if config not in results:
                    results[config] = {}
                if method not in results[config]:
                    results[config][method] = {}
                results[config][method][seed] = metric
                break  # 找到第一个匹配的行后，停止读取文件

# 打印结果
# for config, metric in results.items():
#     print(f"Configuration: {config}, Metric: {metric}")

# for setting in ["noise"]:
#     for dataset in ["cifar10", "cifar100", "tinyimagenet"]:
#         for backbone in ["lenet", "resnet18", "vit"]:
#             config = (setting, dataset, backbone)
#             for method in ["base", "self_paced", "transfer_teacher", "minimax", "screener_net", "meta_reweight", "meta_weight_net", "data_parameters", "local_to_global", "dds", "dihcl", "superloss", "cbs", "coarse_to_fine", "adaptive"]:

for setting in ["standard"]:
    for dataset in ["rte", "mrpc", "stsb", "cola", "sst2", "qnli", "qqp"]: # "mnli", "wnli"
        for backbone in ["lstm", "bert", "gpt"]:
            config = (setting, dataset, backbone)
            for method in ["base", "self_paced", "transfer_teacher", "minimax", "screener_net", "meta_reweight", "meta_weight_net", "data_parameters", "dds", "dihcl", "superloss", "adaptive"]:

# for setting in ["noise"]:
#     for dataset in ["mutag", "ptc_mr", "nci1", "proteins", "dd"]:
#         for backbone in ["gcn", "gat", "sage"]:
#             config = (setting, dataset, backbone)
#             for method in ["base", "self_paced", "transfer_teacher", "minimax", "screener_net", "meta_reweight", "meta_weight_net", "data_parameters", "dds", "dihcl", "superloss", "adaptive"]:

                if config in results.keys() and method in results[config]:
                    for seed in ["42", "666", "777", "888", "999"]:
                        if seed in results[config][method]:
                            print(results[config][method][seed], end="")
                        if seed != "999":
                            print(",", end="")
                    print()
                else:
                    print(",,,,")
