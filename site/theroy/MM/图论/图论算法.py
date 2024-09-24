import numpy as np
from scipy.sparse.csgraph import dijkstra

# 创建邻接矩阵，0表示无连接，其他值表示边权重
matrix = np.array([[0, 1, 4, 0],
                   [1, 0, 2, 5],
                   [4, 2, 0, 1],
                   [0, 5, 1, 0]])

# 使用Dijkstra算法计算从所有节点出发的最短路径
dist_matrix_dijkstra, predecessors_dijkstra = dijkstra(matrix, return_predecessors=True)

# 该函数有两个返回值

# dist_matrix_dijkstra返回的时最短路矩阵,其中的元素代表任意两个节点之间最短路的长度

# predecessors_dijkstra返回的是父顶点信息p[i,j]代表的是从i到j的最短路中,j的前驱结点的信息
# 根据图最短路的性质,i到j的前驱结点的最短路是i到j的最短路的子集,所以可以继续搜索前驱结点,然后一步步还原路径信息

print("Shortest path matrix using Dijkstra algorithm:")
print(dist_matrix_dijkstra)
print(predecessors_dijkstra)


