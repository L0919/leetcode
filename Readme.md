###  [主界面](https://leetcode-cn.com/problemset/all/)

#### 1. [两数之和](https://leetcode-cn.com/problems/two-sum/)

```python
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        hashmap = {}
        for index, num in enumerate(nums):
            another_num = target - num
            if another_num in hashmap:
                return [hashmap[another_num], index]
            hashmap[num] = index
        return None

nums = [2,7,11,15]
target = 9
print(Solution.twoSum(Solution,nums,target))
```

#### 2.[两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

```python
class Solution:
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        re = ListNode(0)
        r=re
        carry=0
        while(l1 or l2):
            x= l1.val if l1 else 0
            y= l2.val if l2 else 0
            s=carry+x+y
            carry=s//10
            r.next=ListNode(s%10)
            r=r.next
            if(l1!=None):l1=l1.next
            if(l2!=None):l2=l2.next
        if(carry>0):
            r.next=ListNode(1)
        return re.next
```

#### 3.[无重复字符的最长字串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

```python
class Solution:
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        st = {}
        i, ans = 0, 0
        for j in range(len(s)):
            if s[j] in st:
                i = max(st[s[j]], i)
            ans = max(ans, j - i + 1)
            st[s[j]] = j + 1
        return ans;
```

#### 4.[整数反转](https://leetcode-cn.com/problems/reverse-integer/)

```python
class Solution:
    def reverse(self,x):
        a = str(x) if x > 0 else str(-x)+'-'
        a = int(a[::-1])
        return a if a <= 2**31 - 1 and a >= -2**31 - 1 else 0
```

#### 5.[字符串转换整数(atoi)](https://leetcode-cn.com/problems/string-to-integer-atoi/)

```python
class Solution:
    def myAtoi(self, s: str) -> int:
        return max(min(int(*re.findall('^[\+\-]?\d+', s.lstrip())), 2**31 - 1), -2**31)
```

#### 6.[回文数](https://leetcode-cn.com/problems/palindrome-number/)

```python
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
```

#### 7.[整数转罗马数字](https://leetcode-cn.com/problems/integer-to-roman/)

```python
class Solution:
    def intToRoman(self, num: int) -> str:
        list1=[1000,900,500,400,100,90,50,40,10,9,5,4,1]
        list2=['M','CM','D','CD','C','XC','L','XL','X','IX','V','IV','I']
        result=""
        for i in range(len(list1)):
            while num>=list1[i]:
                result+=list2[i]
                num-=list1[i]
        return result
```

#### 8.[罗马数字转整数](https://leetcode-cn.com/problems/roman-to-integer/)

```python
class Solution:
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        a = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}        
        ans=0        
        for i in range(len(s)):            
            if i<len(s)-1 and a[s[i]]<a[s[i+1]]:                
                ans-=a[s[i]]
            else:
                ans+=a[s[i]]
        return ans
```

#### 9.[最长公共前缀](https://leetcode-cn.com/problems/longest-common-prefix/)

```python
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
```

#### 10.[三数之和](https://leetcode-cn.com/problems/3sum/)

```python
class Solution(object):
    def threeSum(self, nums):
        # 将nums分成三组：zeros，positives，negatives
        zeros, positives, negatives = 0, {}, {}
        for num in nums:
            if num == 0:
                zeros += 1
            elif num > 0:
                positives.setdefault(num, 0)
                positives[num] += 1
            else:
                negatives.setdefault(num, 0)
                negatives[num] += 1

        # 相加为0的三元组可能的组成形式：3个0，两个负数一个正数，两个正数一个负数，一正一负加一零
        results = []
        if zeros >= 3:
            results.append([0] * 3)
            
        if len(positives) != 0 and len(negatives) != 0:
            for pi in positives:
                count = positives[pi]
                if count >= 2 and (-2 * pi) in negatives:
                    results.append([pi] * 2 + [-2 * pi])
                if -pi in negatives and zeros > 0:
                    results.append([pi, -pi, 0])
                for pj in positives:
                    if pj <= pi:
                        continue
                    if -1 * (pi + pj) in negatives:
                        results.append([-1 * (pi + pj), pi, pj])
    
            for ni in negatives:
                count = negatives[ni]
                if count >= 2 and (-2 * ni) in positives:
                    results.append([ni] * 2 + [-2 * ni])
                for nj in negatives:
                    if nj <= ni:
                        continue
                    if -1 * (ni + nj) in positives:
                        results.append([-1 * (ni + nj), ni, nj])

        return results
```

#### 11.[电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

```python
class Solution:
    def letterCombinations(self, digits: str):
        KEY = {'2': ['a', 'b', 'c'],
               '3': ['d', 'e', 'f'],
               '4': ['g', 'h', 'i'],
               '5': ['j', 'k', 'l'],
               '6': ['m', 'n', 'o'],
               '7': ['p', 'q', 'r', 's'],
               '8': ['t', 'u', 'v'],
               '9': ['w', 'x', 'y', 'z']}
        if digits == '':
            return []
        ans = ['']
        for num in digits:
            ans = [pre+suf for pre in ans for suf in KEY[num]]
        return ans

```

#### 12.[四数之和](https://leetcode-cn.com/problems/4sum/)

```python
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
```

#### 13.[删除链表的倒数第N个节点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

```python
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
            G['c'] = -1 # 为新的测试用例重置为-1
            return head.next

        return head
```

#### 14.[有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

```python
class Solution:
    def isValid(self, s):
        while '{}' in s or '()' in s or '[]' in s:
            s = s.replace('{}', '')
            s = s.replace('[]', '')
            s = s.replace('()', '')
        return s == ''
```

#### 15.[合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

```python
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
```

#### 16.[括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        ans = []
        def f(l, r, s):
            l == r == n and ans.append(s)
            l < n and f(l + 1, r, s + '(')
            r < l and f(l, r + 1, s + ')')
        f(0, 0, '')
        return ans
```

#### 17.[删除排序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

```python
class Solution:
    def removeDuplicates(self, nums):
        p = 0
        while p < len(nums) - 1:
            if nums[p + 1] == nums[p]:
                nums.pop(p + 1)
            else:
                p += 1

        return len(nums)
```

#### 18.[移除元素](https://leetcode-cn.com/problems/remove-element/)

```python
class Solution:
    def removeElement(self, nums, val):
        while val in nums:
            nums.remove(val)
        return len(nums)
```

#### 19.[实现strStr()](https://leetcode-cn.com/problems/implement-strstr/)

```python
class Solution(object):
    def strStr(self, haystack, needle):
        i = 0
        if needle == "":
            return 0
        while i <= len(haystack) - len(needle):
            if haystack[i: i + len(needle)] == needle:
                return i
            i += 1
        return -1
```

#### 20.[搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

```python
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
```

#### 21.[在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

```python
class Solution:
    def searchRange(self, nums, target):
        return [-1, -1] if target not in nums else [bisect.bisect_left(nums, target), bisect.bisect_right(nums, target) - 1]
```

#### 22.[搜索插入位置](https://leetcode-cn.com/problems/search-insert-position/)

```python
class Solution:
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if target in nums:
            return nums.index(target)
        else:
            nums.append(target)
            nums.sort()
            return nums.index(target)
```

#### 23.[全排列](https://leetcode-cn.com/problems/permutations/)

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        def backtrack(nums, tmp):
            if not nums:
                res.append(tmp)
                return 
            for i in range(len(nums)):
                backtrack(nums[:i] + nums[i+1:], tmp + [nums[i]])
        backtrack(nums, [])
        return res
```

#### 24.[旋转图像](https://leetcode-cn.com/problems/rotate-image/)

```python
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        matrix[:] = zip(*matrix[::-1])
        return matrix
```

#### 25.[字母异位词分组](https://leetcode-cn.com/problems/group-anagrams/)

```python
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
```

#### 26.[Pow(x,n)](https://leetcode-cn.com/problems/powx-n/)

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1
        if n < 0:
            return 1 / self.myPow(x, -n)
        #计算奇数
        if n & 1:
            return x * self.myPow(x, n - 1)
        return self.myPow(x*x, n // 2)
```

#### 27.[最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

```python
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
         """
        for i in range(1, len(nums)):
            nums[i]= nums[i] + max(nums[i-1], 0)
        return max(nums)
```

#### 28.[螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/)

```python
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        # 取首行，去除首行后，对矩阵翻转来创建新的矩阵，
        # 再递归直到新矩阵为[],退出并将取到的数据返回
        ret = []
        if matrix == []:
            return ret
        ret.extend(matrix[0]) # 上侧
        new = [reversed(i) for i in matrix[1:]]
        if new == []:
            return ret
        r = self.spiralOrder([i for i in zip(*new)])
        ret.extend(r)
        return ret
```

#### 29.[合并区间](https://leetcode-cn.com/problems/merge-intervals/)

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        temp=sorted(intervals,key=lambda x:x[0])
        n=len(temp)
        if n<2:
            return intervals
        cur=temp[0]
        res=[]
        for elem in temp[1:]:
            if elem[0]>cur[1]:
                res.append(cur)
                cur=elem
            elif elem[1]>cur[1]:
                cur[1]=elem[1]
        res.append(cur)
        return res
```

#### 30.[最后一个单词的长度](https://leetcode-cn.com/problems/length-of-last-word/)

```python
class Solution:
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        cnt, tail = 0, len(s) - 1
        while tail >= 0 and s[tail] == ' ':
            tail -= 1
        while tail >= 0 and s[tail] != ' ':
            cnt += 1
            tail -= 1
        return cnt
```



