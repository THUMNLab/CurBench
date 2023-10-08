import os
import re

# 定义文件夹路径
runs_folder = "runs"

mapping = {"bert": "3", "gpt": "3", "lstm": "10", "vit": "200", "resnet18": "200", "lenet": "200", "gcn": "200", "gat": "200", "sage": "200"}
backbone2type = {"lenet": "image", "resnet18": "image", "vit": "image", "lstm": "text", "bert": "text", "gpt": "text", "gcn": "graph", "gat": "graph", "sage": "graph"}
target_datasets = {"image": ["cifar10", "cifar100", "tinyimagenet"], "text": ["rte", "mrpc", "stsb", "cola", "sst2", "qnli", "qqp", "mnli-matched", "mnli-mismatched"], "graph": ["mutag", "ptc_mr", "nci1", "proteins", "dd"]}
target_backbones = {"image": ["lenet", "resnet18", "vit"], "text": ["lstm", "bert", "gpt"], "graph": ["gcn", "gat", "sage"]}
target_methods = {"image": ["base", "self_paced", "transfer_teacher", "minimax", "screener_net", "meta_reweight", "meta_weight_net", "data_parameters", "local_to_global", "dds", "dihcl", "superloss", "cbs", "coarse_to_fine", "adaptive"],
                "text":  ["base", "self_paced", "transfer_teacher", "minimax", "screener_net", "meta_reweight", "meta_weight_net", "data_parameters", "dds", "dihcl", "superloss", "adaptive"],
                "graph": ["base", "self_paced", "transfer_teacher", "minimax", "screener_net", "meta_reweight", "meta_weight_net", "data_parameters", "dds", "dihcl", "superloss", "adaptive"]}

# 初始化存储结果的字典
results = {}

# 定义正则表达式模式，匹配文件夹名
pattern = re.compile(r"^([\w]+)-([\w]+)-(noise-0\.4-|imbalance-50-)?([\w]+)-([\d]+)-([\d]+)$")

target_type = "image"
target_setting = "noise"

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
    cur_type = backbone2type[backbone]

    # 仅保留当前表格需要的文件进行读取
    if cur_type != target_type or setting != target_setting:
        print("Exclude " + folder_name)
        continue
    
    print("Include " + folder_name)

    log_file_path = os.path.join(folder_path, "train.log")
    # 检查log文件是否存在
    if not os.path.exists(log_file_path):
        # print("No log file")
        continue
    
    # 排除epoch异常的测试实验
    if mapping[backbone] != epoch:
        print("Wrong epoch setting: %s" % (folder_name))
        continue
    
    # mnli有两个验证集，单独处理
    if dataset == "mnli":
        # 读取log文件中的指标
        log_pattern = re.compile(r".*?Final.*=\s*([\d.]+)")
        with open(log_file_path, "r") as file:
            count = 0
            for line in file:
                pattern_match = log_pattern.search(line)
                if pattern_match:
                    count += 1
                    if count == 1:
                        config = (setting, "mnli-matched", backbone)
                    elif count == 2:
                        config = (setting, "mnli-mismatched", backbone)
                    if config not in results:
                        results[config] = {}
                    if method not in results[config]:
                        results[config][method] = {}
                    results[config][method][seed] = float(pattern_match.group(1))
                    if count >= 2:
                        break  # 找到两个匹配的行后，停止读取文件
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

for dataset in target_datasets[target_type]:
    for backbone in target_backbones[target_type]:
        config = (target_setting, dataset, backbone)
        for method in target_methods[target_type]:
            if config in results.keys() and method in results[config]:
                for seed in ["42", "666", "777", "888", "999"]:
                    if seed in results[config][method]:
                        print(results[config][method][seed], end="")
                    if seed != "999":
                        print(",", end="")
                print()
            else:
                print(",,,,")
