# 移除元素 link:https://leetcode-cn.com/problems/remove-element/
class Solution:
    def removeElement(self, nums, val):
        while val in nums:
            nums.remove(val)
        return len(nums)