# 括号生成 link:https://leetcode-cn.com/problems/generate-parentheses/
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        ans = []
        def f(l, r, s):
            l == r == n and ans.append(s)
            l < n and f(l + 1, r, s + '(')
            r < l and f(l, r + 1, s + ')')
        f(0, 0, '')
        return ans