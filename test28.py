# 螺旋矩阵 link:https://leetcode-cn.com/problems/spiral-matrix/
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        # 取首行，去除首行后，对矩阵翻转来创建新的矩阵，
        # 再递归直到新矩阵为[],退出并将取到的数据返回
        ret = []
        if matrix == []:
            return ret
        ret.extend(matrix[0]) # 上侧
        new = [reversed(i) for i in matrix[1:]]
        if new == []:
            return ret
        r = self.spiralOrder([i for i in zip(*new)])
        ret.extend(r)
        return ret