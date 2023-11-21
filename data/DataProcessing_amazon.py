# 重点去搞明白输入进去的都是什么，输出出来的数字是什么含义,彻底搞清楚.txt文件中的数字含义
import gzip
from collections import defaultdict
from datetime import datetime
import os
import copy
import json

# current_directory = os.getcwd()
# print(current_directory)


true = True
false =False
def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        # print(l)
        yield eval(l)


countU = defaultdict(lambda: 0)
countP = defaultdict(lambda: 0)
line = 0

# DATASET = 'Luxury_Beauty'
DATASET = 'All_Beauty'
# dataname = '/home/zfan/BDSC/projects/datasets/reviews_' + DATASET + '_5.json.gz'
# dataname = r'D:/Downloads/STOSA/data/All_Beauty/' + DATASET + '_5.json.gz'
dataname = r'D:/Downloads/STOSA/data/All_Beauty/' + DATASET + '.json.gz'

#文件路径尽量弄到配置里，或者设置好参数传入,像这个代码一样
if not os.path.isdir('./'+DATASET):
    os.mkdir('./'+DATASET)
# print('./'+DATASET)#输出 ./All_Beauty / './'当前目录   实际是在D:\Downloads\STOSA\data\寻找
train_file = './'+DATASET+'/train.txt'
valid_file = './'+DATASET+'/valid.txt'
test_file = './'+DATASET+'/test.txt'
imap_file = './'+DATASET+'/imap.json'
umap_file = './'+DATASET+'/umap.json'

data_file = './'+DATASET+'.txt'



for one_interaction in parse(dataname):
    rev = one_interaction['reviewerID']#"reviewerID": 这是提交评价的用户的唯一标识符。每个用户都有一个与其账户关联的唯一ID，用于区分不同的用户
    # print(rev)
    asin = one_interaction['asin']# ASIN代表“Amazon Standard Identification Number”，这是亚马逊为每个产品在其目录中分配的唯一标识符。通过这个ID，可以准确地找到特定的产品。
    # print(asin)
    #这个属性表示评价提交的时间，但它的格式是一个UNIX时间戳。UNIX时间戳（也称为Epoch时间或POSIX时间）是表示从1970年1月1日00:00:00 UTC（协调世界时）到某一特定时间点之间经过的秒数。
    #它是一个通用的、跨平台的方式来表示时间，经常用于编程和系统设计中。举例，时间戳1252800000表示的日期是2009年9月13日。你可以使用多种编程语言或在线工具将UNIX时间戳转换为可读的日期格式
    time = float(one_interaction['unixReviewTime'])
    # print(time)
    countU[rev] += 1
    countP[asin] += 1

usermap = dict()
usernum = 1
itemmap = dict()
itemnum = 1
User = dict()

#这个for循环形成
for one_interaction in parse(dataname):
    rev = one_interaction['reviewerID']
    asin = one_interaction['asin']
    time = float(one_interaction['unixReviewTime'])
    if countU[rev] < 5 or countP[asin] < 5:
        continue 

    if rev in usermap:
        userid = usermap[rev]
    else:
        userid = usernum
        usermap[rev] = userid
        User[userid] = []
        usernum += 1
        # print()
        # print("finished")
    if asin in itemmap:
        itemid = itemmap[asin]
    else:
        itemid = itemnum
        itemmap[asin] = itemid
        itemnum += 1
    # print(itemmap)
    User[userid].append([itemid, time])
    # User[userid].append([itemid])
    # print(User.items())
# sort reviews in User according to time


with open(imap_file, 'w') as f:
    json.dump(itemmap, f)

with open(umap_file, 'w') as f:
    json.dump(usermap, f)

for userid in User.keys():
    User[userid].sort(key=lambda x: x[1])


user_train = {}
user_valid = {}
user_test = {}
for user in User:
    nfeedback = len(User[user])
    if nfeedback < 3:
        user_train[user] = User[user]
        user_valid[user] = []
        user_test[user] = []
    else:
        #倒数第二个做为验证集，倒数第一做为测试集
        user_train[user] = User[user][:-2]
        user_valid[user] = []
        user_valid[user].append(User[user][-2])
        user_test[user] = []
        user_test[user].append(User[user][-1])

print(User[1305])
# 您要查找的 itemid
target_itemid = [1213, 899, 451, 665,]

for asin, itemid in itemmap.items():
    if itemid in target_itemid:
        print(f"Found asin '{asin}' for itemid {itemid}")
        # 如果找到了，可以从列表中移除该 itemid，以避免重复打印
        target_itemid.remove(itemid)
    else:
        continue

# 检查是否有未找到的 itemid
for itemid in target_itemid:
    print(f"asin for itemid {itemid} not found")

# def writetofile(data, dfile):
#     with open(dfile, 'w') as f:
#         for u, ilist in sorted(data.items()):
#             for i, t in ilist:
#                 f.write(str(u) + '\t'+ str(i) + '\t' + str(t) + "\n")
#
# def writetofile_v2(data, dfile):
#     with open(dfile, 'w') as f:
#         for u, ilist in sorted(data.items()):
#             f.write(str(u))
#             for i, t in ilist:
#                 f.write(' '+ str(i))
#             f.write("\n")
#
# #writetofile(user_train, train_file)
# #writetofile(user_valid, valid_file)
# #writetofile(user_test, test_file)
#
# writetofile_v2(User, data_file)
#
#
#
# num_instances = sum([len(ilist) for _, ilist in User.items()])
# print('total user: ', len(User))
# print('total instances: ', num_instances)
# print('avg length: ', num_instances / len(User))
# print('total items: ', itemnum)
# print('density: ', num_instances / (len(User) * itemnum))
# print('valid #users: ', len(user_valid))
# numvalid_instances = sum([len(ilist) for _, ilist in user_valid.items()])
# print('valid instances: ', numvalid_instances)
# numtest_instances = sum([len(ilist) for _, ilist in user_test.items()])
# print('test #users: ', len(user_test))
# print('test instances: ', numtest_instances)
