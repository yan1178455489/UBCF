from numpy import *
import numpy
import random
import time

def load_rating_data(file_path='MovieLens/u.data'):
    ratings=[]
    for line in open(file_path,'r'):
        data=line.split('\t')
        uid=int(data[0])
        mid=int(data[1])
        rat=int(data[2])
        ratings.append([uid,mid,rat])
    data=array(ratings)
    return data

def split_rating_data(data,ratio=0.2):
    train_data=[]
    test_data=[]
    for line in data:
        num=random.random()
        if num < 0.2:
            test_data.append(line)
        else:
            train_data.append(line)
    train_data=array(train_data)
    test_data=array(test_data)
    return train_data,test_data

def getNearestNeighbor(userId, cgsimmat):
    neighbors_dist = []
    for i in range(1,u_num):
        if cgsimmat[userId][i]!=0:
            neighbors_dist.append([cgsimmat[userId][i],i])
    neighbors_dist.sort(reverse=True)
    return neighbors_dist

def getTopN(groupID, ratings):
    topn_ratings = []
    for i in range(1,i_num):
        if ratings[groupID][i]!=0:
            topn_ratings.append([ratings[groupID][i],i])
    topn_ratings.sort(reverse=True)
    return topn_ratings

def integrate_strategy():
    if strategy == "AVG":
        for i in range(len(groups)):
            for k in range(len(group_cand[i])):
                total_rating = 0
                for j in range(group_len):
                    total_rating += predict_mat[groups[i][j]][group_cand[i][k]]
                cand_ratings[i][group_cand[i][k]] = total_rating / group_len

    elif strategy == "MP":
        for i in range(len(groups)):
            for j in range(len(group_cand[i])):
                max = -1
                for k in groups[i]:
                    if predict_mat[k][group_cand[i][j]]>max:
                        max = predict_mat[k][group_cand[i][j]]
                cand_ratings[i][group_cand[i][j]] = max

    elif strategy == "LM":
        for i in range(len(groups)):
            for j in range(len(group_cand[i])):
                min = 100
                for k in groups[i]:
                    if predict_mat[k][group_cand[i][j]]<min and predict_mat[k][group_cand[i][j]]!=0:
                        min = predict_mat[k][group_cand[i][j]]
                cand_ratings[i][group_cand[i][j]] = min

if __name__ == '__main__':#只有用到训练集和测试集的编号才需要-1
    #beta=0.6
    para = {}
    i=0
    with open('UBCF.txt') as f:  # 需要重新打开文本进行读取
        for line in f:
            content = line.rstrip()  # 删除字符串末尾的空白
            if(len(content)>1):
                kv = content.split('=')
                para[kv[0]] = kv[1]
    f.close()
    group_len = para['gSize']
    neighbor_num = para['neighborNum']
    top_n = para['topN']
    strategy = para['gStrategy']
    data=load_rating_data()
    file_path = 'MovieLens/u1.base'
    train_data = []
    with open(file_path) as f:
        for line in f:
            tdata = line.split('\t')
            uid = int(tdata[0])
            mid = int(tdata[1])
            rat = int(tdata[2])
            train_data.append([uid, mid, rat])
    f.close()
    train_data = array(train_data)

    file_path = 'MovieLens/u1.test'
    test_data = []
    with open(file_path) as f:
        for line in open(file_path, 'r'):
            tdata = line.split('\t')
            uid = int(tdata[0])
            mid = int(tdata[1])
            rat = int(tdata[2])
            test_data.append([uid, mid, rat])
    f.close()
    test_data = array(test_data)

    u_num = max(data[:,0])+1
    i_num = max(data[:,1])+1
    rating_mat = zeros((u_num,i_num),dtype=int16)#评分矩阵

    participant_list = []
    for i in range(i_num):
        participant_list.append([])

    for ratingtuple in data:
        (i, j, rat) = ratingtuple
        participant_list[j].append(i)

    # sum = 0
    # for i in range(1, i_num):
    #     sum += len(participant_list[i])
    # sum = sum / (i_num - 1)

    group_len = int(group_len)
    modified = 0
    start = time.time()

    file_path = 'MovieLens/'+str(group_len) + '_group.data'
    groups = []
    for line in open(file_path, 'r'):
        mbs = line.split('\t')
        group = []
        for i in range(0, group_len):
            group.append(int(mbs[i]))
        groups.append(group)

    for i in range(len(train_data)):
        rating_mat[train_data[i,0],train_data[i,1]]=train_data[i,2]

    count = 0
    user_avg = zeros(u_num,dtype=int)
    for i in range(1,u_num):
        total=0
        for j in range(1,i_num):
            total+=rating_mat[i][j]
            if rating_mat[i][j]!=0:
                count+=1
        user_avg[i]=total/count

    cgsim_mat=zeros((u_num,u_num),dtype=float)
    avgsim=0
    rating_array = numpy.array(rating_mat)
    for i in range(1,u_num,1):
        for j in range(i+1,u_num,1):
            fenzi = numpy.dot(rating_array[i],rating_array[j])
            fenmu1 = numpy.sqrt(rating_array[i].dot(rating_array[i]))
            fenmu2 = numpy.sqrt(rating_array[j].dot(rating_array[j]))
            if fenmu1 * fenmu2!=0:
                cgsim_mat[i][j] = cgsim_mat[j][i] = fenzi / (fenmu1 * fenmu2)
            #avgsim+=cgsim_mat[i][j]
    #avgsim=avgsim*2/(u_num*(u_num-1))

    user_cand = []
    for i in range(u_num):
        user_cand.append([])

    for i in range(1,u_num):
        for j in range(1,i_num):
            if rating_mat[i][j]==0:
                user_cand[i].append(j)

    neighbor_num=int(neighbor_num)
    predict_mat=zeros((u_num,i_num),dtype=float)
    for i in range(1,u_num):
        fenzi=0
        fenmu=0
        neighbor_list=getNearestNeighbor(i,cgsim_mat)[:neighbor_num]
        for k in user_cand[i]:
            for j in neighbor_list:
                fenzi+=cgsim_mat[i][j[1]]*(rating_mat[j[1]][k]-user_avg[j[1]])
                fenmu+=abs(cgsim_mat[i][j[1]])
            predict_mat[i][k]=user_avg[i] + fenzi/fenmu


    user_training = []
    for i in range(u_num):
        user_training.append([])

    for ratingtuple in train_data:
        (i, j, rat) = ratingtuple
        user_training[i].append(j)

    group_all = numpy.zeros((len(groups), i_num), dtype=int)

    for i in range(len(groups)):
        for j in range(len(groups[i])):
            for k in range(len(user_training[groups[i][j]])):
                group_all[i][user_training[groups[i][j]][k]] = 1

    group_cand = []
    for i in range(len(groups)):
        group_cand.append([])

    for j in range(len(groups)):
        for i in range(1, i_num, 1):
            if group_all[j][i] == 0:
                group_cand[j].append(i)

    cand_ratings = numpy.zeros((len(groups), i_num), dtype=float)
    integrate_strategy()

    if (modified == 1):
        for i in range(len(groups)):
            for j in range(1, i_num):
                if len(participant_list[j]) > 300:
                    predict_mat[i][j] = predict_mat[i][j] / math.log(len(participant_list[j]), 350)
    top_n = int(top_n)
    recommend_list = []
    for i in range(len(groups)):
        tmp = getTopN(i,cand_ratings)[:top_n]
        sub_list = []
        for j in tmp:
            sub_list.append(j[1])
        recommend_list.append(sub_list)

    end = time.time()
    runtime = end - start
    test_mat = numpy.zeros((u_num, i_num), dtype=int)
    for i in range(len(test_data)):
        test_mat[test_data[i][0]][test_data[i][1]] = 1

    user_testset = []
    for i in range(u_num):
        test_set = []
        for j in range(1, i_num, 1):
            if test_mat[i][j] == 1:
                test_set.append(j)
        user_testset.append(test_set)

    group_testset = []
    for i in range(len(groups)):
        tmp_intersaction = user_testset[groups[i][0]]
        for j in range(len(groups[i]) - 1):
            tmp_intersaction = [k for k in tmp_intersaction if k in user_testset[groups[i][j + 1]]]
        group_testset.append(tmp_intersaction)

    Precision = 0
    Recall = 0
    Novelty = 0
    hits = 0
    for i in range(len(groups)):
        for j in range(len(recommend_list[i])):
            if recommend_list[i][j] in group_testset[i]:
                hits += 1
            Novelty += len(participant_list[recommend_list[i][j]])
        Precision += top_n
        if len(group_testset[i]) != 0:
            Recall += len(group_testset[i])
    Novelty = Novelty / (top_n * u_num * len(groups))
    Precision = hits/Precision
    Recall = hits/Recall

    f = open('result.txt','w')
    f.write('precisions='+str(Precision)+'\n')
    f.write('recall=' + str(Recall) + '\n')
    f.write('novelty=' + str(Novelty))
    f.close()
    # print('Precision=', Precision)
    # print('Recall=', Recall)
    # print('Novelty=', Novelty)
    print('time =',runtime)