# 删除链表的倒数第N个节点 link:https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/
G = {'c': -1}


class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:

        if not head:
            if G['c'] == -1:
                G['c'] = 0
            return

        head.next = self.removeNthFromEnd(head.next, n)

        if G['c'] > -1:
            G['c'] += 1

        if n == G['c']:
            G['c'] = -1  # 为新的测试用例重置为-1
            return head.next

        return head