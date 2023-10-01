import os
import re


# 删除模型checkpoint
# for root in ['runs', 'runs_']:
#     dirs = os.listdir(root)
#     for dir in dirs:
#         path = os.path.join(root, dir, 'net.pkl')
#         if os.path.exists(path):
#             os.system('rm %s' % path)

# 删除空的train.log文件
# for root in ['runs', 'runs_']:
#     dirs = os.listdir(root)
#     for dir in dirs:
#         path = os.path.join(root, dir, 'train.log')
#         if os.path.exists(path):
#             if os.path.getsize(path) == 0:
#                 os.remove(path)

# 删除空文件夹
# for root in ['runs', 'runs_']:
#     dirs = os.listdir(root)
#     for dir in dirs:
#         path = os.path.join(root, dir)
#         if not os.listdir(path):
#             os.rmdir(path)

# 删除没有结果的train.log文件
# pattern = re.compile(r".*?Final.*=\s*([\d.]+)")
# for root in ['runs', 'runs_']:
#     dirs = os.listdir(root)
#     for dir in dirs:
#         path = os.path.join(root, dir, 'train.log')
#         exists = False
#         if os.path.exists(path):
#             if os.path.getsize(path) == 0:
#                 os.remove(path)
#             else:
#                 with open(path, "r") as file:
#                     for line in file:
#                         match = pattern.search(line)
#                         if match:
#                             exists = True
#                             break
#                 if not exists:
#                     os.remove(path)

# 删除没有train.log的文件夹
# for root in ['runs', 'runs_']:
#     dirs = os.listdir(root)
#     for dir in dirs:
#         path = os.path.join(root, dir)
#         if 'train.log' not in os.listdir(path):
#             os.rmdir(path)

# 统计文件夹数量
for root in ['runs', 'runs_']:
    dirs = os.listdir(root)
    print(len(dirs))