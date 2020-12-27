# 字母异位词分组 link:https://leetcode-cn.com/problems/group-anagrams/
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        res = []
        dic = {}
        for s in strs:
            keys = "".join(sorted(s))
            if keys not in dic:
                dic[keys] = [s]
            else:
                dic[keys].append(s)
        return list(dic.values())