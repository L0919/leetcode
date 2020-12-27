# 搜索旋转排序数组 link:https://leetcode-cn.com/problems/search-in-rotated-sorted-array/
class Solution:
    def search(self, nums, target):
        if len(nums) == 0:
            return -1

        l = 0
        r = len(nums) - 1

        while l < r:
            mid = l + (r - l) // 2
            if nums[mid] < nums[r]:  # [mid, r]有序
                if nums[mid] < target <= nums[r]:
                    l = mid + 1
                else:
                    r = mid
            else:  # [l, mid]有序
                if nums[l] <= target <= nums[mid]:
                    r = mid
                else:
                    l = mid + 1
        return -1 if nums[l] != target else l