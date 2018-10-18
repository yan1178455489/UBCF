from numpy import *
import numpy
import random

def dfs(mstart, pstart, gr, sizes):
    for i in range(pstart + 1, len(movie_participant[mstart])):
        if cgsim_mat[movie_participant[mstart][pstart]][movie_participant[mstart][i]]>=thresholds:
            gr.append(movie_participant[mstart][i])
            if len(gr) == sizes:
                group = []
                for i in range(0,sizes):
                    group.append(gr[i])
                groups.append(group)
                gr.pop()
                return
            dfs(mstart,i,gr,sizes)
            gr.pop()

    #return groups

if __name__ == '__main__':
    sizes = input('pls input group size:')
    sizes = int(sizes)
    file_path = 'u1.test'
    test_data = []
    for line in open(file_path, 'r'):
        tdata = line.split('\t')
        uid = int(tdata[0])
        mid = int(tdata[1])
        rat = int(tdata[2])
        test_data.append([uid, mid, rat])
    test_data = array(test_data)

    file_path = 'u.data'
    data = []
    for line in open(file_path, 'r'):
        tdata = line.split('\t')
        uid = int(tdata[0])
        mid = int(tdata[1])
        rat = int(tdata[2])
        data.append([uid, mid, rat])
    data = array(data)

    u_num = max(data[:, 0]) + 1 #用户数+1
    i_num = max(data[:, 1]) + 1
    rating_mat = zeros((u_num, i_num), dtype=int16)  # 评分矩阵
    for i in range(len(data)):
        rating_mat[data[i,0],data[i,1]]=data[i,2]

    cgsim_mat = zeros((u_num, u_num), dtype=float)
    avgsim = 0
    rating_array = numpy.array(rating_mat)
    for i in range(1, u_num, 1):
        for j in range(i + 1, u_num, 1):
            fenzi = numpy.dot(rating_array[i], rating_array[j])
            fenmu1 = numpy.sqrt(rating_array[i].dot(rating_array[i]))
            fenmu2 = numpy.sqrt(rating_array[j].dot(rating_array[j]))
            if fenmu1 * fenmu2 != 0:
                cgsim_mat[i][j] = cgsim_mat[j][i] = fenzi / (fenmu1 * fenmu2)
            avgsim += cgsim_mat[i][j]
    avgsim=avgsim*2/(u_num*(u_num-1))
    print('平均相似度：',avgsim)
    thresholds = input('输入相似度阈值：')
    thresholds = float(thresholds)
    movie_participant = []
    for i in range(i_num):
        movie_participant.append([])
    for i in range(len(test_data)):
        movie_participant[test_data[i, 1]].append(test_data[i, 0])

    groups = []
    for i in range(1, len(movie_participant), 1):
        for j in range(len(movie_participant[i])):
            group = []
            group.append(movie_participant[i][j])
            dfs(i, j, group, sizes)
    file_path = str(sizes)+'_group.data'
    file = open(file_path, 'w')
    for gr in groups:
        for mb in gr:
            file.writelines(str(mb)+'\t')
        file.writelines('\n')
    file.close()

