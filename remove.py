import os


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


# 统计文件夹数量
for root in ['runs', 'runs_']:
    dirs = os.listdir(root)
    print(len(dirs))