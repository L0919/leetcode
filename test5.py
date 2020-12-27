# 字符串转换整数(atoi) link:https://leetcode-cn.com/problems/string-to-integer-atoi/
class Solution:
    def myAtoi(self, s: str) -> int:
        return max(min(int(*re.findall('^[\+\-]?\d+', s.lstrip())), 2**31 - 1), -2**31)