# 最长公共前缀 link:https://leetcode-cn.com/problems/longest-common-prefix/
class Solusion(object):
    def longestCommonPrefix(self, strs):
        if not strs: return ""
        ss = list(map(set, zip(*strs)))
        res = ""
        for i, x in enumerate(ss):
            x = list(x)
            if len(x) > 1:
                break
            res = res + x[0]
        return res