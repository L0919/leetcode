# 实现strStr() link:https://leetcode-cn.com/problems/implement-strstr/
class Solution(object):
    def strStr(self, haystack, needle):
        i = 0
        if needle == "":
            return 0
        while i <= len(haystack) - len(needle):
            if haystack[i: i + len(needle)] == needle:
                return i
            i += 1
        return -1