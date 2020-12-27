# 合并两个有序链表 link:https://leetcode-cn.com/problems/merge-two-sorted-lists/
class Solution:
    def mergeTwoLists(self, l1, l2):
        res = ListNode(None)
        node = res
        while l1 and l2:
            if l1.val<l2.val:
                node.next,l1 = l1,l1.next
            else:
                node.next,l2 = l2,l2.next
            node = node.next
        if l1:
            node.next = l1
        else:
            node.next = l2
        return res.next