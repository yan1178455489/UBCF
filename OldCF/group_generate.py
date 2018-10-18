
class Generate_group:

    def __init__(self):
        self.movie_participant = []
        self.groups = []

    def dfs(self,mstart, pstart, gr, sizes):
        for i in range(pstart + 1, len(self.movie_participant[mstart])):
            gr.append(self.movie_participant[mstart][i])
            if len(gr) == sizes:
                gr=list(set(gr))
                if len(gr) == sizes:
                    self.groups.append(gr)
                #print(gr)
                return
            if len(gr) > sizes:
                return
            self.dfs(mstart,i,gr,sizes)

    def generate_group(self,sizes,test_tuple,num_movie):
        rating_tuple = test_tuple

        for i in range(num_movie + 1):
            self.movie_participant.append([])

        for i in range(len(rating_tuple)):
            self.movie_participant[rating_tuple[i, 1]].append(rating_tuple[i, 0])
        # print(movie_participant)

        for i in range(1, len(self.movie_participant), 1):
            for j in range(len(self.movie_participant[i])):
                group = []
                group.append(self.movie_participant[i][j])
                self.dfs(i, j, group, sizes)
                if(len(self.groups)>500):
                    return self.groups
        return self.groups