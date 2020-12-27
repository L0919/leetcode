# 删除排序数组中的重复项 link:https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/
class Solution:
    def removeDuplicates(self, nums):
        p = 0
        while p < len(nums) - 1:
            if nums[p + 1] == nums[p]:
                nums.pop(p + 1)
            else:
                p += 1

        return len(nums)