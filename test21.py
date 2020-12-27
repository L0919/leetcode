# 在排序数组中查找元素的第一个和最后一个位置 link:https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/
class Solution:
    def searchRange(self, nums, target):
        return [-1, -1] if target not in nums else [bisect.bisect_left(nums, target), bisect.bisect_right(nums, target) - 1]