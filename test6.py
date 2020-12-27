# 回文数 link:https://leetcode-cn.com/problems/palindrome-number/
class Solution:
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0:
            return False
        else:
            y = str(x)[::-1]
            if y == str(x):
                return True
            else:
                return False