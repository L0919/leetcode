# Pow(x,n) link:https://leetcode-cn.com/problems/powx-n/
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1
        if n < 0:
            return 1 / self.myPow(x, -n)
        #计算奇数
        if n & 1:
            return x * self.myPow(x, n - 1)
        return self.myPow(x*x, n // 2)