# 四数之和 link:https://leetcode-cn.com/problems/4sum/
class Solution:
    def fourSum(self, nums, target):
        d={}
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                d.setdefault(nums[i]+nums[j],[]).append((i,j))
        result=set()
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                for a,b in d.get(target-nums[i]-nums[j],[]):
                    temp={i,j,a,b}
                    if len(temp)==4:
                        result.add(tuple(sorted(nums[t] for t in temp)))
        return result