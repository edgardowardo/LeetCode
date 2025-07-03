import Foundation


/**
HELPERS
 */
extension Array<Int> {
    
    /// Assumes array of integer self is ordered
    func firstIndex(of range: ClosedRange<Int>) -> Int {
        var low = 0
        var high = self.count - 1
        while low <= high {
            let mid = low + (high - low) / 2
            if range.contains(self[mid]) {
                return mid
            } else if range.upperBound < self[mid] {
                high = mid - 1
            } else {
                low = mid + 1
            }
        }
        guard low < self.count && range.contains(self[low]) else {
            return -1
        }
        return low
    }
    
    /// Assumes array of integer self is ordered
    func firstIndex(of range: Range<Int>) -> Int {
        var low = 0
        var high = self.count - 1
        while low <= high {
            let mid = low + (high - low) / 2
            if range.contains(self[mid]) {
                return mid
            } else if range.upperBound < self[mid] {
                high = mid - 1
            } else {
                low = mid + 1
            }
        }
        guard low < self.count && range.contains(self[low]) else {
            return -1
        }
        return low
    }
    
    /// Assumes array of integer self is ordered
    func firstIndexBST(of target: Int) -> Int {
        guard !isEmpty else {
            return -1
        }
        var mid = 0
        var low = 0
        var high = self.count - 1
        while low <= high {
            mid = low + (high - low) / 2
            if self[mid] == target {
                return mid
            } else if target < self[mid] {
                high = mid - 1
            } else {
                low = mid + 1
            }
        }
        guard low < self.count && self[low] == target else {
            return -1
        }
        return low
    }
 
}

/*
assert([0,1,2,4,5].firstIndex(of: 1..<3) == 2)
assert([0,2,4,5].firstIndex(of: 1..<3) == 1)
assert([0,3,4,5].firstIndex(of: 1..<3) == -1)
assert([100].firstIndex(of: 1..<3) == -1)
assert([Int]().firstIndex(of: 1..<3) == -1)

assert([0,1,2,4,5].firstIndex(of: 1...2) == 2)
assert([0,2,4,5].firstIndex(of: 1...2) == 1)
assert([0,3,4,5].firstIndex(of: 1...2) == -1)
assert([100].firstIndex(of: 1...2) == -1)
assert([Int]().firstIndex(of: 1...2) == -1)

assert([0,1,2,4,5].firstIndexBST(of: 1) == 1)
assert([0,2,4,5].firstIndexBST(of: 2) == 1)
assert([0,3,4,5].firstIndexBST(of: 6) == -1)
assert([0,3,4,5].firstIndexBST(of: -2) == -1)
assert([100].firstIndexBST(of: -1) == -1)
assert([Int]().firstIndexBST(of: 1) == -1)
 */
 


///---------------------------------------------------------------------------------------
/// Leetcode 121
///https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/
func maxProfit1(_ prices: [Int]) -> Int {
    
    guard let first = prices.first, prices.count > 1 else { return 0 }
    
    var buyPrice = first
    var maxPrice = 0
    
    for i in 1..<prices.count {
        let currentPrice = prices[i]
        let currentProfit = currentPrice - buyPrice
        
        if currentProfit > 0 {
            maxPrice = max(maxPrice, currentProfit)
        } else {
            buyPrice = currentPrice
        }
    }
    
    return maxPrice
}

//assert(maxProfit1([7,6,4,3,1]) == 0)
//assert(maxProfit1([1,2,3,4,5]) == 4)
//assert(maxProfit1([7,1,5,3,6,4]) == 5)


func maxProfit0(_ prices: [Int]) -> Int {
    
    guard let first = prices.first else { return 0 }
    
    var buyPrice = first
    var maxProfit = 0

    for i in 1..<prices.count {
        let currentPrice = prices[i]
        
        let profit = currentPrice - buyPrice
        
        if profit > 0 {
            maxProfit = max(profit, maxProfit)
        } else {
            buyPrice = currentPrice
        }
    }
    
    return maxProfit
}

//assert(maxProfit0([7,6,4,3,1]) == 0)
//assert(maxProfit0([1,2,3,4,5]) == 4)
//assert(maxProfit0([7,1,5,3,6,4]) == 5)

///---------------------------------------------------------------------------------------
/// Leetcode 122
///https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/description/
func maxProfit(_ prices: [Int]) -> Int {

    guard let first = prices.first, prices.count > 1 else { return 0 }

    var i = 0
    var lo = first
    var hi = first
    var profit = 0
    let n = prices.count

    while i < n - 1 {
        // buy = going down
        while i < n - 1 && prices[i] >= prices[i + 1] {
            i += 1
        }
        lo = prices[i]

        // sell
        while i < n - 1 && prices[i] <= prices[i + 1] {
            i += 1
        }
        hi = prices[i]
        profit += hi - lo
    }

    return profit
}

//maxProfit([7,6,4,3,1])
//maxProfit([1,2,3,4,5])
//maxProfit([7,1,5,3,6,4])



///---------------------------------------------------------------------------------------
/// Leetcode 189
///https://leetcode.com/problems/rotate-array/description/
class Leet0189 {
    func rotate(_ nums: inout [Int], _ k: Int) {
        let k = k % nums.count
        let left = nums[0..<nums.count - k]
        let right = nums[nums.count - k..<nums.count]
        var numsOut = [Int]()
        numsOut.append(contentsOf: right)
        numsOut.append(contentsOf: left)
        nums = numsOut
    }
    
    static func test() {
        let sut = Leet0189()
        var nums = [1,2,3,4,5,6,7]
        sut.rotate(&nums, 3)
        assert(nums == [5,6,7,1,2,3,4])
        
        nums =  [-1,-100,3,99]
        sut.rotate(&nums, 2)
        assert(nums == [3,99,-1,-100])
    }
}
//Leet0189.test()




///---------------------------------------------------------------------------------------
/// Leetcode 217
/// https://leetcode.com/problems/contains-duplicate/
func containsDuplicate(_ nums: [Int]) -> Bool {
    Set(nums).count != nums.count
}

//assert(containsDuplicate([1,2,1]) == true)


///---------------------------------------------------------------------------------------
/// Leetcode 136
///https://leetcode.com/problems/single-number/description/
func singleNumber(_ nums: [Int]) -> Int {
    var ans = 0
    for num in nums {
        ans ^= num
    }
    return ans

}

//assert(singleNumber([2,2,1]) == 1)
//assert(singleNumber([4,1,2,1,2]) == 4)
//assert(singleNumber([1]) == 1)


func xxx_singleNumber(_ nums: [Int]) -> Int {
    var hash = [Int: Int]()
    for n in nums {
        if let _ = hash[n] {
            hash[n] = nil
        } else {
            hash[n] = 1
        }
    }

    return hash.first!.key
}


///---------------------------------------------------------------------------------------
/// Leetcode 350
/// https://leetcode.com/problems/intersection-of-two-arrays-ii/description/
func intersect(_ nums1: [Int], _ nums2: [Int]) -> [Int] {

    var out = [Int]()
    var hash = [Int: Int]()

    for m in nums1 {
        if let _ = hash[m] {
            hash[m]! += 1
        } else {
            hash[m] = 1
        }
    }


    for l in nums2 {

        if let count = hash[l] {
            out.append(l)

            hash[l]! -= 1

            if count == 1 {
                hash[l] = nil
            }
        }
    }

    return out
}

//assert(intersect([1,2], [1,1]) == [1])
//assert(intersect([1,2,2,1], [2,2]) == [2,2])
//assert(intersect([4,9,5], [9,4,8,4]) == [9,4])



///---------------------------------------------------------------------------------------
/// Leetcode 66
/// https://leetcode.com/problems/plus-one/description/
func plusOne(_ digits: [Int]) -> [Int] {

    var i = digits.count - 1
    var out = [Int]()
    var added = false

    while i > -1 {

        if digits[i] == 9 && !added {

            out.insert(0, at: 0)

            if i == 0 {
                out.insert(1, at: 0)
            }

        } else {
            if added {
                out.insert(digits[i], at: 0)
            } else {
                out.insert(digits[i] + 1, at: 0)
                added = true
            }
        }
        i -= 1
    }

    return out
}

//assert(plusOne([1,9,9]) == [2,0,0])
//assert(plusOne([5,1,6]) == [5,1,7])
//assert(plusOne([1,2,3]) == [1,2,4])
//assert(plusOne([4,3,2,1]) == [4,3,2,2])
//assert(plusOne([9]) == [1,0])
//assert(plusOne([9,8,7,6,5,4,3,2,1,0]) == [9, 8, 7, 6, 5, 4, 3, 2, 1, 1])


///---------------------------------------------------------------------------------------
/// Leetcode 283
///https://leetcode.com/problems/move-zeroes/description/

class Leet0283 {
    var example1 = [0,1,0,3,12]
    var example2 = [0]
    
    func moveZeroes(_ nums: inout [Int]) {
        var indexNonZero = 0
        
        for index in 0 ..< nums.count {
            if nums[index] != 0 {
                nums[indexNonZero] = nums[index]
                indexNonZero += 1
            }
        }
        
        var indexZero = indexNonZero
        
        while indexZero < nums.count {
            nums[indexZero] = 0
            indexZero += 1
        }
    }
}

//let sut0283 = Leet0283()
//
//sut0283.moveZeroes(&sut0283.example1)
//assert(sut0283.example1 == [1,3,12,0,0])
//
//sut0283.moveZeroes(&sut0283.example2)
//assert(sut0283.example2 == [0])


///---------------------------------------------------------------------------------------
/// Leetcode 1
///https://leetcode.com/problems/two-sum/description/
class Leet0001 {
    
    func twoSum(_ nums: [Int], _ target: Int) -> [Int] {
        
        // hash stores the number and it's position in the array
        var hash = [Int: Int]()
        
        // loop through all integers with index
        for i in 0 ..< nums.count {
            let num = nums[i]
            let diff = target - num
            
            // check if number exists in the hash
            if let _ = hash[diff] {
                let out = [hash[diff]!, i]
                return out
                // number does not exist in hash so store it
            } else {
                hash[num] = i
            }
        }
        
        return []
    }
    
    static func test() {
        let sut = Leet0001()
        assert(sut.twoSum([2,7,11,15], 9) == [0,1])
        assert(sut.twoSum([3,2,4], 6) == [1,2])
        assert(sut.twoSum([3,3], 6) == [0,1])
    }
}
//Leet0001.test()



///---------------------------------------------------------------------------------------
/// Leetcode 167
/// https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
class Leet0167 {
    func twoSum(_ numbers: [Int], _ target: Int) -> [Int] {
        var left = 0
        var right = numbers.count - 1
 
        while left < right {
            let sum = numbers[left] + numbers[right]
            
            if sum == target {
                return [left+1, right+1]
            } else if sum < target {
                left += 1
            } else {
                right -= 1
            }
        }

        return [left+1, right+1]
    }
    
    static func test() {
        let sut0167 = Leet0167()
        assert(sut0167.twoSum([2, 7, 11, 15], 9) == [1, 2])
        assert(sut0167.twoSum([2, 3, 4], 6) == [1, 3])
        assert(sut0167.twoSum([-1, 0], -1) == [1, 2])
        assert(sut0167.twoSum([0, 0, 3, 4], 0) == [1, 2])
        assert(sut0167.twoSum([-10, -8, -2, 1, 2, 5, 6], 0) == [3, 5])
        assert(sut0167.twoSum([3, 6, 21, 23, 25], 27) == [2, 3])
    }
}
//Leet0167.test()






///---------------------------------------------------------------------------------------
/// Leetcode 48
///https://leetcode.com/problems/rotate-image/description/
func rotate(_ matrix: inout [[Int]]) {

    let n = matrix.count-1

    // transpose
    for i in 0..<matrix.count {
        for j in i..<matrix.count {
            let temp = matrix[i][j]
            matrix[i][j] = matrix[j][i]
            matrix[j][i] = temp
        }
    }

    // horizontal reflection
    for i in 0..<matrix.count {
        for j in 0..<(matrix.count/2) {
            let temp = matrix[i][j]
            matrix[i][j] = matrix[i][n-j]
            matrix[i][n-j] = temp
        }
    }
}

//var matrix = [[1,2,3],[4,5,6],[7,8,9]]
//rotate(&matrix)
//assert(matrix == [[7,4,1],[8,5,2],[9,6,3]])
//
//matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
//rotate(&matrix)
//assert(matrix == [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]])




///---------------------------------------------------------------------------------------
/// Leetcode 344
///https://leetcode.com/problems/reverse-string/
class Leet0344 {
    func reverseString(_ s: inout [Character]) {
        var (left, right) = (0, s.count - 1)
        while left < right {
            s.swapAt(left, right)
            left += 1
            right -= 1
        }
    }
    static func test() {
        let sut = Leet0344()
        var s: [Character] = ["h","e","l","l","o"]
        sut.reverseString(&s)
        assert(s == ["o","l","l","e","h"])
        
        s = ["H","a","n","n","a","h"]
        sut.reverseString(&s)
        assert(s == ["h","a","n","n","a","H"])
    }
}
//Leet0344.test()

func reverseString(_ s: inout [Character]) {
    s.reverse()
}
//var string: [Character] = ["h","e","l","l","o"]
//reverseString(&string)
//assert(string == ["o","l","l","e","h"])











///---------------------------------------------------------------------------------------
/// Leetcode 7
///https://leetcode.com/problems/reverse-integer/
func reverse(_ x: Int) -> Int {
    var array = Array(String(abs(x)))
    array.reverse()
    var number = ""
    for char in array {
        number = "\(number)\(char)"
    }
    guard let out = Int(number), Int32.max >= out else { return 0 }

    if x < 0 {
        return out * -1
    }
    return out
}

//reverse(123)
//reverse(-123)
//reverse(120)
//reverse(1534236469)


//var array = Array(String(abs(-123)))
//
//array.reverse()
//var number = ""
//for char in array {
//    number = "\(number)\(char)"
//}
//
//let out = Int(number)
//
//print(out)
//print(Int32.max)
//1534236469
//2147483647 > 9646324351




///---------------------------------------------------------------------------------------
/// Leetcode 387
/// https://leetcode.com/problems/first-unique-character-in-a-string/description/
func firstUniqChar(_ s: String) -> Int {

    var hash = [Character: Int]()
    let chars = Array(s)
    var i = 0

    for char in chars {
        if let count = hash[char] {
            hash[char] = count + 1
        } else {
            hash[char] = 1
        }
    }

    for i in 0 ..< chars.count {
        let char = chars[i]
        if let count = hash[char], count == 1 {
            return i
        }
    }
    return -1
}

//assert(firstUniqChar("leetcode") == 0)
//assert(firstUniqChar("loveleetcode") == 2)
//assert(firstUniqChar("aabb") == -1)

 








///---------------------------------------------------------------------------------------
/// Leetcode 242
///https://leetcode.com/problems/valid-anagram/description/
func isAnagram(_ s: String, _ t: String) -> Bool {

    var sHash = hash(s)
    var tHash = hash(t)

    guard sHash.count == tHash.count else {
        return false
    }

    for s in sHash {
        if let tCount = tHash[s.key] {
            if tCount != s.value {
                return false
            }
        } else {
            return false
        }
    }

    return true
}

func hash(_ s: String) -> [Character: Int] {

    // Collect hash counts in s
    var sHash = [Character: Int]()
    let sChars = Array(s)

    for char in sChars {
        if let count = sHash[char] {
            sHash[char] = count + 1
        } else {
            sHash[char] = 1
        }
    }

    return sHash
}

//assert(isAnagram("anagram", "nagaram"))
//assert(!isAnagram("rat", "car"))




///---------------------------------------------------------------------------------------
/// Leetcode 125
///https://leetcode.com/problems/valid-palindrome/description/
///O(n) + O(n) Time and Space complexity
func isPalindrome(_ s: String) -> Bool {
    
    let s2 = Array(s.lowercased().filter{ $0.isLetter || $0.isNumber })
    
    if s2.count == 0 {
        return true
    }
    
    for i in 0 ... s2.count / 2 {
        guard s2[i] == s2[s2.count - 1 - i] else {
            return false
        }
    }
    return true
}

//assert(isPalindrome(" A man, a plan, a canal: Panama.") == true)
//assert(isPalindrome("racecar") == true)
//assert(isPalindrome("race a car") == false)
//assert(isPalindrome("`l;`` 1o1 ??;l`") == true)
//assert(isPalindrome(" ") == true)
//assert(isPalindrome("`l;`` 1o1 ??;l`") == true)
//assert(isPalindrome("a") == true)
//assert(isPalindrome("a.") == true)
//assert(isPalindrome("0P") == false)


extension Character {
    var isLetterOrNumber: Bool { isLetter || isNumber }
}


///---------------------------------------------------------------------------------------
/// Leetcode 125
///O(n) + O(1) Time and Space complexity
class Leet0125 {

    func isPalindrome(_ s: String) -> Bool {
        
        var left = s.startIndex
        var right = s.index(before: s.endIndex)
        
        
        while left < right {
            
            guard s[left].isLetterOrNumber else  {
                left = s.index(after: left)
                continue
            }
            
            let leftChar = s[left]

            guard s[right].isLetterOrNumber else {
                right = s.index(before: right)
                continue
            }
            
            let rightChar = s[right]
            
            if leftChar.lowercased() != rightChar.lowercased() {
                return false
            }
            
            left = s.index(after: left)
            right = s.index(before: right)
        }
        
        return true

    }

}


//let sut0125 = Leet0125()
//assert(sut0125.isPalindrome(" A man, a plan, a canal: Panama.") == true)
//assert(sut0125.isPalindrome("racecar") == true)
//assert(sut0125.isPalindrome("race a car") == false)
//assert(sut0125.isPalindrome("`l;`` 1o1 ??;l`") == true)
//assert(sut0125.isPalindrome(" ") == true)
//assert(sut0125.isPalindrome("`l;`` 1o1 ??;l`") == true)
//assert(sut0125.isPalindrome("a") == true)
//assert(sut0125.isPalindrome("a.") == true)
//assert(sut0125.isPalindrome("0P") == false)


///---------------------------------------------------------------------------------------
/// Leetcode 8
///https://leetcode.com/problems/string-to-integer-atoi/description/
func myAtoi(_ s: String) -> Int {

    var string = s.trimmingCharacters(in: .whitespacesAndNewlines)

    var negativeMultiplier = 1
    if let first = string.first, first == "-" {
        string.removeFirst()
        negativeMultiplier = -1
    }

    if let first = string.first, first == "+" {
        string.removeFirst()

        if negativeMultiplier < 0 {
            return 0
        }
    }

    var number = ""
    var isZeroLeading = true
    for char in Array(string) {
        if char == "0" && isZeroLeading {
           continue
        } else if char.isNumber {
            number = "\(number)\(char)"
            isZeroLeading = false
        } else {
            break
        }
    }


    if negativeMultiplier > 0 {
        guard number.count <= String(Int32.max).count else { return Int(Int32.max) }
    } else {
        guard number.count <= String(Int32.min).count else { return Int(Int32.min) }
    }

    guard let out = Int(number) else { return 0 }

    if negativeMultiplier > 0 {
        guard out <= Int32.max else { return Int(Int32.max) }
    }

    if negativeMultiplier < 0 {
        guard out < Int(Int32.min) * -1 else { return Int(Int32.min) }
    }

    return out * negativeMultiplier
}

//myAtoi("42")
//myAtoi("-042")
//myAtoi("1337c0d3")
//myAtoi("0-1")
//myAtoi("words and 987")
//myAtoi("-91283472332")
//myAtoi("20000000000000000000")
//myAtoi("  0000000000012345678")
//myAtoi("21474836460")
//myAtoi("-9223372036854775809")




///---------------------------------------------------------------------------------------
/// Leetcode 28
///https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/description/
func strStr(_ haystack: String, _ needle: String) -> Int {

    if needle.isEmpty {
        return 0
    }

    if let range = haystack.range(of: needle) {

        return haystack.distance(from: haystack.startIndex, to: range.lowerBound)
    }

    return -1
}

//assert(strStr("sadbutsad", "sad") == 0)
//assert(strStr("leetcode", "leeto") == -1)


///---------------------------------------------------------------------------------------
/// Leetcode 14
/// https://leetcode.com/problems/longest-common-prefix/description/
func longestCommonPrefix(_ strs: [String]) -> String {

    guard var prefix = strs.first else { return "" }

    for str in strs {

        while !str.hasPrefix(prefix) {
            prefix.removeLast()
            if prefix.isEmpty {
                return ""
            }
        }
    }

    return prefix
}

//assert(longestCommonPrefix(["flower","flow","flight"]) == "fl")
//assert(longestCommonPrefix(["dog","racecar","car"]) == "")



public class ListNode {
    public var val: Int
    public var next: ListNode?
    public init(_ val: Int) {
        self.val = val
        self.next = nil
    }
    public init(_ val: Int, _ next: ListNode?) {
        self.val = val
        self.next = next
    }
}

extension Array<Int> {
    func makeListNode() -> ListNode? {
        ListNode.makeList(self)
    }
}

extension ListNode {

    static func makeList(_ arr: [Int]) -> ListNode? {
        arr.reversed().reduce(nil) { (node, val) in
            ListNode(val, node)
        }
    }

    static func toArray(_ head: ListNode?) -> [Int] {
        var arr: [Int] = []
        var node = head
        
        while node != nil {
            arr.append(node!.val)
            node = node?.next
        }
        return arr
    }
    
    func toArray() -> [Int] {
        Self.toArray(self)
    }
}


///---------------------------------------------------------------------------------------
/// Leetcode 237
/// https://leetcode.com/problems/delete-node-in-a-linked-list/description/
func deleteNode(_ node: ListNode?) {
    node?.val = node?.next?.val ?? 0
    node?.next = node?.next?.next
}



///---------------------------------------------------------------------------------------
/// Leetcode 19
///https://leetcode.com/problems/remove-nth-node-from-end-of-list/description/
func removeNthFromEnd(_ head: ListNode?, _ n: Int) -> ListNode? {

    var dummy: ListNode? = ListNode(-1)
    dummy?.next = head

    var p1 = dummy
    var p2 = dummy

    for _ in 0 ... n {
        p2 = p2?.next
    }

    while p2 != nil {
        p1 = p1?.next
        p2 = p2?.next
    }

    p1?.next = p1?.next?.next

    return dummy?.next
}

//assert(removeNthFromEnd(ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5))))), 2)?.toArray() == [1, 2, 3, 5])
//assert(removeNthFromEnd(ListNode(1, ListNode(2)), 1)?.toArray() == [1])
//assert(removeNthFromEnd(ListNode(1), 1)?.toArray() == nil)


///---------------------------------------------------------------------------------------
/// Leetcode 206
/// https://leetcode.com/problems/reverse-linked-list/description/
func reverseList(_ head: ListNode?) -> ListNode? {
    var prev: ListNode? = nil
    var curr = head
    while curr != nil {
        let next = curr?.next
        curr?.next = prev
        prev = curr
        curr = next
    }
    return prev
}

//assert(reverseList(ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5))))))?.toArray() == [5, 4, 3, 2, 1])
//assert(reverseList(ListNode(1, ListNode(2)))?.toArray() == [2, 1])
//assert(reverseList(ListNode(1))?.toArray() == [1])
//assert(reverseList(nil)?.toArray() == nil)




/*
func getTwoSum(from numbers: [Int], target: Int) -> [Int] {

    guard numbers.count > 1 else { return [] }

    var hash = [Int: Int]()

    for num in numbers.enumerated() {
        print("\(num.offset), \(num.element)")

        let diff = target - num.element
        print(diff)

        if hash[diff] != nil {
            let out = [hash[diff]!, num.offset]
            print("\(diff) exists!, returning \(out)")
            return out
        } else {
            print("adding \(num.element), \(num.offset)")
            hash[num.element] = num.offset
        }
    }

    return []
}


assert(getTwoSum(from: [2,7,11,15], target: 9) == [0,1])

assert(getTwoSum(from: [1,2,3,4], target: 7) == [2,3])

assert(getTwoSum(from: [3,2,4], target: 6) == [1,2])

assert(getTwoSum(from: [3,3], target: 6) == [0,1])

*/



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/merge-two-sorted-lists/
class Leet0021 {

    func mergeTwoLists(_ list1: ListNode?, _ list2: ListNode?) -> ListNode? {
        var prehead = ListNode(0), prev: ListNode? = prehead, l1 = list1, l2 = list2
        while l1 != nil && l2 != nil {
            if l1!.val < l2!.val {
                prev?.next = l1
                l1 = l1?.next
            } else {
                prev?.next = l2
                l2 = l2?.next
            }
            prev = prev?.next
        }
        prev?.next = l1 != nil ? l1 : l2
        return prehead.next
    }
}


///---------------------------------------------------------------------------------------
/// Leetcode 206
///https://leetcode.com/problems/reverse-linked-list/description/
func reverse(_ head: ListNode?) -> ListNode? {

    var prev: ListNode? = nil
    var curr = head

    while curr != nil {
        let next = curr?.next
        curr?.next = prev
        prev = curr
        curr = next
    }

    return prev
}

//reverse(ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5))))))

/*
func isPalindrome(_ head: ListNode?) -> Bool {

    var slow = head
    var fast = head?.next

    while fast != nil && fast?.next != nil {
        slow = slow?.next
        fast = fast?.next?.next
    }

    var rev = reverse(slow?.next)
    var head = head

    while rev != nil {
        if let h = head?.val, let r = rev?.val, r != h {
            return false
        }
        head = head?.next
        rev = rev?.next
    }

    return true
}
*/

///---------------------------------------------------------------------------------------
/// Leetcode 234
///https://leetcode.com/problems/palindrome-linked-list/description/
func isPalindrome(_ head: ListNode?) -> Bool {

    var p = [Int]()
    var c = head

    while c != nil {
        if let v = c?.val {
            p.append(v)
        }
        c = c?.next
    }

    var l = 0
    var r = p.count - 1

    while l <= r {
        if p[l] != p[r] {
            return false
        }
        l += 1
        r -= 1
    }

    return true
}

//assert(isPalindrome(ListNode(1, ListNode(2, ListNode(2, ListNode(1))))))
//assert(!isPalindrome(ListNode(1, ListNode(2))))
//assert(isPalindrome(ListNode(1, ListNode(2, ListNode(3, ListNode(2, ListNode(1)))))))
//assert(!isPalindrome(ListNode(1, ListNode(1, ListNode(2, ListNode(1))))))


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/linked-list-cycle/description/
class Leet0141 {
    func hasCycle(_ head: ListNode?) -> Bool {
        var slow: ListNode? = head, fast: ListNode? = head
        while fast != nil && fast?.next != nil {
            guard slow !== fast?.next else { return true }
            slow = slow?.next
            fast = fast?.next?.next
        }
        return false
    }
}


///---------------------------------------------------------------------------------------
/// Leetcode 141
///https://leetcode.com/problems/linked-list-cycle/description/
func hasCycle(_ head: ListNode?) -> Bool {

    var curr = head
    var ids = Set<ObjectIdentifier>()

    while curr != nil {

        let id = ObjectIdentifier(curr!)
//            print("\(id), \(curr!.val)")

        if ids.contains(id) {
            return true
        } else {
            ids.insert(id)
        }

        curr = curr?.next
    }

    return false
}



//hasCycle(ListNode(1, ListNode(1, ListNode(2, ListNode(1)))))



public class TreeNode {
    public var val: Int
    public var left: TreeNode?
    public var right: TreeNode?
    public init() {
        self.val = 0
        self.left = nil
        self.right = nil
    }
    public init(_ val: Int) {
        self.val = val
        self.left = nil
        self.right = nil
    }
    public init(_ val: Int, _ left: TreeNode?, _ right: TreeNode?) {
        self.val = val
        self.left = left
        self.right = right
    }
}


extension TreeNode {
    func buildArray() -> [Int?] {
        var result: [Int?] = []
        var deque: Deque<TreeNode?> = [self]
        while !deque.isEmpty {
            for _ in deque {
                let node = deque.removeFirst()
                guard let node else {
                    result.append(nil)
                    continue
                }
                result.append(node.val)
                deque.append(node.left)
                deque.append(node.right)
            }
        }
        while !result.isEmpty, let last = result.last, last == nil {
            result.removeLast()
        }
        return result
    }
}


extension TreeNode: CustomStringConvertible {
    public var description: String {
        buildArray().map { $0 == nil ? "nil" : "\($0!)" } .joined(separator: ", ")
    }
}

extension Array<Int?> {
    func buildTree() -> TreeNode? {
        guard !self.isEmpty else { return nil }
        guard let first = self.first, let val = first else { return nil }
        let root = TreeNode(val)
        var deque = Deque<TreeNode?>([root]), index = 1
        while !deque.isEmpty, index < self.count {
            for _ in deque {
                let node = deque.removeFirst()
                guard let node else { continue }
                guard index < self.count else { break }
                if let val = self[index] {
                    node.left = TreeNode(val)
                }
                index += 1
                deque.append(node.left)
                guard index < self.count else { break }
                if let val = self[index] {
                    node.right = TreeNode(val)
                }
                index += 1
                deque.append(node.right)
            }
        }
        return root
    }
}


public class Node {
    public var val: Int
    public var next: Node?
    public var random: Node?
    public init(_ val: Int) {
        self.val = val
        self.next = nil
        self.random = nil
    }
}


///---------------------------------------------------------------------------------------
/// Leetcode 104
///https://leetcode.com/problems/maximum-depth-of-binary-tree/description/
func maxDepth(_ root: TreeNode?) -> Int {
    guard let root else { return 0 }
    return 1 + max(maxDepth(root.left), maxDepth(root.right))
}


//assert(maxDepth(TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))) == 3)


/*
func isValidBST(_ root: TreeNode?) -> Bool {

    guard let root else { return false }

    guard let left = root.left else { return true }

    var eval = left.val < root.val && isValidBST(left)

    guard let right = root.right else { return true }

    return eval && right.val > root.val && isValidBST(right)
}
*/

func isValid(root: TreeNode?, min: Int, max: Int) -> Bool {

    guard let root else {
        return true
    }

    if root.val <= min || root.val >= max {
        return false
    }

    return isValid(root: root.left, min: min, max: root.val)
    && isValid(root: root.right, min: root.val, max: max)
}

///---------------------------------------------------------------------------------------
/// Leetcode 98
///https://leetcode.com/problems/validate-binary-search-tree/description/
func isValidBST(_ root: TreeNode?) -> Bool {
    isValid(root: root, min: .min, max: .max)
}


//assert(!isValidBST(TreeNode(5, TreeNode(4), TreeNode(6, TreeNode(3), TreeNode(7)))))
//assert(!isValidBST(TreeNode(5, TreeNode(1), TreeNode(4, TreeNode(3), TreeNode(6)))))
//assert(isValidBST(TreeNode(4, TreeNode(3), TreeNode(6))))
//assert(isValidBST(TreeNode(2, TreeNode(1), TreeNode(3))))
//assert(!isValidBST(TreeNode(1, TreeNode(1), nil)))



/*
enum Side {
    case left, right, mid

    var opposite: Side {
        switch self {
        case .left: .right
        case .right: .left
        case .mid: .mid
        }
    }
}

typealias Item = (val: Int, side: Side)

func flatten(_ root: TreeNode?, side: Side, list: inout [Item]) {

    guard let root else {
        return
    }

    if let left = root.left {
        flatten(left, side: .left, list: &list)
    }

    list.append((root.val, side))

    if let right = root.right {
        flatten(right, side: .right, list: &list)
    }
}


func isSymmetric(_ root: TreeNode?) -> Bool {

    var list = [Item]()
    flatten(root, side: .mid, list: &list)

//    print(list)

    for i in 0..<list.count/2 {
        guard list[i].val == list[list.count-1-i].val, list[i].side == list[list.count-1-i].side.opposite
        else { return false }
    }

    return true

}

 */


func isMirror(_ left: TreeNode?, _ right: TreeNode?) -> Bool {

    if left == nil, right == nil {
        return true
    }

    guard left?.val == right?.val else { return false }

    return isMirror(left?.left, right?.right) && isMirror(left?.right, right?.left)
}

///---------------------------------------------------------------------------------------
/// Leetcode 101
///https://leetcode.com/problems/symmetric-tree/description/
func isSymmetric(_ root: TreeNode?) -> Bool {
    isMirror(root?.left, root?.right)
}

//assert(isSymmetric(TreeNode(1, TreeNode(2, TreeNode(3), TreeNode(4)), TreeNode(2, TreeNode(4), TreeNode(3)))))
//assert(isSymmetric(TreeNode(1, TreeNode(2), TreeNode(2))))
//assert(!isSymmetric(TreeNode(1, TreeNode(2, nil, TreeNode(3)), TreeNode(2, nil, TreeNode(3)))))
//assert(isSymmetric(TreeNode(1, TreeNode(2, TreeNode(2), nil), TreeNode(2, nil, TreeNode(2)))))
//assert(!isSymmetric(TreeNode(1, TreeNode(2, TreeNode(2), nil), TreeNode(2, TreeNode(2), nil))))


/*
extension String {
    var anagramKey: String {

        var hash = [Character: Int]()

        self.forEach {
            if let count = hash[$0] {
                hash[$0] = count + 1
            } else {
                hash[$0] = 1
            }
        }

        var key = ""
        hash.keys.sorted().forEach {
            key.append("\($0)\(hash[$0] ?? -1)")
        }

        return key
    }
}


func groupAnagrams(_ strs: [String]) -> [[String]] {

    let sortedStrings = strs.sorted { $0.count < $1.count }
    var hash = [String: [String]]()

    for str in sortedStrings {
        let anagramKey = str.anagramKey
        if let group = hash[anagramKey] {
            var strings = group
            strings.append(str)
            hash[anagramKey] = strings
        } else {
            hash[anagramKey] = [str]
        }
    }

    var groups = [[String]]()
    hash.keys.forEach { groups.append(hash[$0]!) }

    return groups
}
*/

#warning("resubmit correct solution!")
func groupAnagrams(_ strs: [String]) -> [[String]] {

    var hash = [String: [String]]()

    for s in strs {
        let key = String(s.sorted())
        hash[key, default: []].append(s)
    }

    return hash.values.map { $0 }
}



//assert(groupAnagrams(["eat","tea","tan","ate","nat","bat"]) ==  [["tan","nat"],["bat"],["eat","tea","ate"]])
//assert(groupAnagrams([""]) == [])
//assert(groupAnagrams(["a"]) == [["a"]])

//print(groupAnagrams(["a"]))

///---------------------------------------------------------------------------------------
///  Hacker Rank Test SPORTY GROUP
func getGroupedAnagrams(words: [String]) -> Int {
    
    var set = Set<String>()

    for s in words {
        let key = String(s.sorted())
        set.insert(key)
    }
    
    return set.count
}


//assert(getGroupedAnagrams(words: ["inch", "cat", "chin", "kit", "act"]) == 3)
//assert(getGroupedAnagrams(words: ["cat", "listen", "silent", "kitten", "salient"]) == 4)


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/group-anagrams/description/
class Leet0049 {
    func groupAnagrams(_ strs: [String]) -> [[String]] {
        strs.reduce(into: [String: [String]]()) { r, s in r[s.sortedCharacters, default: []].append(s) }.values.compactMap { $0 }
    }
}
extension String {
    var sortedCharacters: String { String(self.sorted()) }
}





/*

func fizzBuzz(n: Int) -> Void {

    for n in 1...n {

        if n % 3 == 0 &&  n % 5 == 0 {
            print("FizzBuzz")
        } else if n % 3 == 0 {
            print("Fizz")
        } else if n % 5 == 0 {
            print("Buzz")
        } else {
            print(n)
        }
    }


}


fizzBuzz(n: 15)

*/

/*
let k = 3
"abcd".map { Character(UnicodeScalar($0.asciiValue! -  UInt8(3))) }

//"abcd".forEach { print("\($0.asciiValue!) \($0.asciiValue)! + UInt8(3) )" ) }


//"ABCDEFGHIJKLMNOPQRSTUVWXYZ".forEach { print("\($0) \($0.asciiValue!)") }



 func simpleCipher(encrypted: String, k: Int) -> String {

     String(encrypted.map {
         var value = $0.asciiValue! -  UInt8(k)

         let aAscii = Character("A").asciiValue!
         let zAscii = Character("Z").asciiValue!

         if value < aAscii {
             let diff = aAscii - value
             value = zAscii - diff + 1
         }

         return Character(UnicodeScalar( value )) }
     )

 }

//print(simpleCipher(encrypted: "ABCD", k: 3))

simpleCipher(encrypted: "G", k: 12)

*/

/*
 



extension String {
    /*
    var date: Date {

        let calendar = Calendar.current
        var components = DateComponents()
        components.year = Int(self.suffix(4))

        let start = str.index(str.startIndex, offsetBy: 3)
        let end = str.index(str.endIndex, offsetBy: -5)
        let range = start..<end

        components.month = mmm(String(self[range]))
        components.day = Int(self.prefix(2))
        components.hour = 0
        components.minute = 0

        return  calendar.date(from: components)!
    }
     */

    var date: Date {
        let format = "dd LLL yyyy"
        let formatter = DateFormatter()
        formatter.dateFormat = format

        return formatter.date(from: self)!
    }
}


extension Date {
    var string: String {
        let outputFormat = DateFormatter()
        outputFormat.dateFormat = "dd MMM yyyy"
        let newDateString = outputFormat.string(from: self)
        return newDateString
    }
}

Date().string

func sortDates(dates: [String]) -> [String] {
    dates.map { $0.date }.sorted().map { $0.string }
}



let format = "dd LLL yyyy"
let formatter = DateFormatter()
formatter.dateFormat = format

formatter.date(from: "01 Mar 2017")


print(sortDates(dates: ["01 Mar 2017", "03 Feb 2017", "15 Jan 1998"]))

*/


func caesarCipher(s: String, k: Int) -> String {

    let AAscii = Character("A").asciiValue!
    let ZAscii = Character("Z").asciiValue!
    let aAscii = Character("a").asciiValue!
    let zAscii = Character("z").asciiValue!

    return String(s.map {

        guard let ascii = $0.asciiValue,
                ascii >= AAscii && ascii <= ZAscii
                || ascii >= aAscii && ascii <= zAscii else {
            return $0
        }

        var value = $0.asciiValue! + UInt8(k % 26 )

        if $0.isUppercase && value > ZAscii {
            let diff = value - ZAscii
            value = AAscii + diff - 1
        } else if $0.isLowercase && value > zAscii {
            let diff = value - zAscii
            value = aAscii + diff - 1
        }

        return Character(UnicodeScalar( value )) }
    )
}

//assert(caesarCipher(s: "abcdefghijklmnopqrstuvwxyz", k: 3) == "defghijklmnopqrstuvwxyzabc")
//assert(caesarCipher(s: "T", k: 3) == "W")
//assert(caesarCipher(s: "www.abc.xy", k: 87) == "fff.jkl.gh")
//assert(caesarCipher(s: "middle-Outz", k: 2) == "okffng-Qwvb")




extension String {
    var date: Date {
        let format = "dd LLL yyyy"
        let formatter = DateFormatter()
        formatter.dateFormat = format

        return formatter.date(from: self)!
    }
}

extension Date {
    var string: String {
        let outputFormat = DateFormatter()
        outputFormat.dateFormat = "dd MMM yyyy"
        let newDateString = outputFormat.string(from: self)
        return newDateString
    }
}

func sortDates(dates: [String]) -> [String] {
    dates.map { $0.date }.sorted().map { $0.string }
}

//assert(sortDates(dates: ["01 Mar 2017", "03 Feb 2017", "15 Jan 1998"]) == ["15 Jan 1998", "03 Feb 2017", "01 Mar 2017"])


/*
extension String {

    var romanInt: Int {
        switch self {
        case "I" : return 1
        case "V" : return 5
        case "X" : return 10
        case "L" : return 50
        case "C" : return 100
        case "D" : return 500
        case "M" : return 1000
        default:
            return 0
        }
    }
}

import RegexBuilder


//func romanToInt(_ s: String) -> Int {
//    //^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$
//
//    let search = /(?<thousand>.+?) and I'm (?<age>\d+) years old./
//
//    if let result = try? search.wholeMatch(in: s) {
//        print("Name: \(result.thousand)")
//        print("Age: \(result.age)")
//    }
//
//    return 0
//}

let romanTensRegex = Regex {

}

let refThousands = Reference(Substring.self)
let refHundreds = Reference(Substring.self)
let refTens = Reference(Substring.self)
let refOnes = Reference(Substring.self)

let romanRegex = Regex {
    Anchor.startOfLine
    Capture(as: refThousands) {
        Repeat("M", 0...3) // Thousands (M{0,3})
    }
    Capture(as: refHundreds) {
        ChoiceOf {
            "CM" // 900
            "CD" // 400
            Regex {
                Optionally("D") // Optional 500
                Repeat("C", 0...3) // 100-300
            }
        }
    }
    Capture(as: refTens) {
        ChoiceOf {
            "XC" // 90
            "XL" // 40
            Regex {
                Optionally("L") // Optional 50
                Repeat("X", 0...3) // 10-30
            }
        }
    }
    Capture(as: refOnes) {
        ChoiceOf {
            "IX" // 9
            "IV" // 4
            Regex {
                Optionally("V") // Optional 5
                Repeat("I", 0...3) // 1-3
            }
        }
    }
    Anchor.endOfLine
}

func romanToInt(_ s: String) -> Int {

    var i = 0

    if let result = try? romanRegex.firstMatch(in: s) {

        // thousands
        i += result[refThousands].count * 1000

        // hundreds
        if result[refHundreds] == "CM" {
            i += 900
        } else if result[refHundreds] == "CD" {
            i += 400
        } else {
            var hundreds = result[refHundreds]
            if hundreds.hasPrefix("D") {
                hundreds.removeFirst()
                i += 500
            }
            i += hundreds.count * 100
        }

        // tens
        if result[refTens] == "XC" {
            i += 90
        } else if result[refTens] == "XL" {
            i += 40
        } else {
            var tens = result[refTens]
            if tens.hasPrefix("L") {
                tens.removeFirst()
                i += 50
            }
            i += tens.count * 10
        }

        // ones
        if result[refOnes] == "IX" {
            i += 9
        } else if result[refOnes] == "IV" {
            i += 4
        } else {
            var tens = result[refOnes]
            if tens.hasPrefix("V") {
                tens.removeFirst()
                i += 5
            }
            i += tens.count
        }


        //    print("\(result[refThousands])")
        //    print("\(result[refHundreds])")
        //    print("\(result[refTens])")
        //    print("\(result[refOnes])")

    }
    return i
}
*/


enum RomanNumeral: String {
    case I, V, X, L, C, D, M
}

extension RomanNumeral {
    var numeric: Int {
        switch self {
        case .I: 1
        case .V: 5
        case .X: 10
        case .L: 50
        case .C: 100
        case .D: 500
        case .M: 1000
        }
    }

    var reducable: RomanNumeral? {
        switch self {
        case .I: nil
        case .V: .I
        case .X: .I
        case .L: .X
        case .C: .X
        case .D: .C
        case .M: .C
        }
    }
}


///---------------------------------------------------------------------------------------
/// Leetcode 13
///https://leetcode.com/problems/roman-to-integer/description/
func romanToInt(_ s: String) -> Int {
    var int = 0
    var string = s

    while !string.isEmpty {
        let last = string.removeLast()

        guard let roman = RomanNumeral(rawValue: String(last)) else { break }

        int += roman.numeric
        if let next = string.last,
            let reducable = RomanNumeral(rawValue: String(next)),
            roman.reducable == reducable
        {
            int -= reducable.numeric
            string.removeLast()
        }
    }
    return int
}

//assert(romanToInt("LVIII") == 58)
//assert(romanToInt("MCMXCIV") == 1994)
//assert(romanToInt("MMMDCCXLIX") == 3749)



func marsExploration(s: String) -> Int {
    guard s.count % 3 == 0 else { return 0 }

    let sos = "SOS"
    var count = 0
    var string = s

    while !string.isEmpty {
        var prefix = string.prefix(3)
        var mySos = sos

        if !string.hasPrefix(sos) {
            while !mySos.isEmpty {
                guard let mySosLast = mySos.removeLast().asciiValue else {
                    break
                }
                guard let prefixLast = prefix.removeLast().asciiValue else {
                    break
                }

                //                print("\(mySosLast) \(prefixLast) \(mySosLast ^ prefixLast)")

                count += Int(mySosLast ^ prefixLast) > 0 ? 1 : 0
            }
        }
        string.removeFirst(sos.count)
    }

    return count
}

/*
marsExploration(s: "SOSSPSSQSSOR")
marsExploration(s: "SOSSOT")
marsExploration(s: "SOSSOSSOS")
marsExploration(s: "SOSOOSOSOSOSOSSOSOSOSOSOSOS")
*/

/*
 typealias My = RomanNumeral
func intToRoman(_ num: Int) -> String {
    var roman = ""
    var int = num

    for i in 1...(int/My.M.numeric) {
        roman.append(My.M.rawValue)
        int -= My.M.numeric
    }

    let reduced = My.M.reduced
    if int / (reduced) == 1 {
        roman.append(My.M.reducable!.rawValue)
        roman.append(My.M.rawValue)
        int -= reduced
    }

    return roman
}
*/

/**

 900 = CM
 800 = DCCC
 700 = DCC
 600 = DC
 500 = D
 400 = CD
 300 = CCC
 200 = CC
 100 = C

 */


extension RomanNumeral {
    var reduced: Int {
        switch self {
        case .I: 0
        case .V: self.numeric - Self.I.numeric
        case .X: self.numeric - Self.I.numeric
        case .L: self.numeric - Self.X.numeric
        case .C: self.numeric - Self.X.numeric
        case .D: self.numeric - Self.C.numeric
        case .M: self.numeric - Self.C.numeric
        }
    }
}

extension RomanNumeral: CaseIterable {}
typealias My = RomanNumeral
func intToRoman(_ num: Int) -> String {
    var roman = ""
    var int = num

    for my in My.allCases.reversed() {
        let upper = int / my.numeric
        if upper > 0 {
            for i in 1 ... (int / my.numeric) {
                roman.append(my.rawValue)
                int -= my.numeric
            }
        }

        let reduced = my.reduced
        if reduced != 0, int / reduced == 1 {
            roman.append(my.reducable!.rawValue)
            roman.append(my.rawValue)
            int -= reduced
        }
    }

    return roman
}



///---------------------------------------------------------------------------------------
/// Leetcode 12
///https://leetcode.com/problems/integer-to-roman/description/
//func intToRoman(_ num: Int) -> String {
//    let values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
//    let letters = [
//        "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I",
//    ]
//    var roman = ""
//    var num = num
//
//    for i in 0..<values.count {
//        while num >= values[i] {
//            num -= values[i]
//            roman.append(letters[i])
//        }
//    }
//
//    return roman
//}

//assert(intToRoman(58) == "LVIII")
//assert(intToRoman(1994) == "MCMXCIV")
//assert(intToRoman(3749) == "MMMDCCXLIX")


//print(Int.max )
//print(Int32.max)

// 1_000_000_000_000_000_000
// 9_999_999_999_999_999
// 9 223 372 036 854 775 807
//             2 147 483 647

// 9 223 372 036 854 775 807
//
//    Q   T   B    M HT  H

extension Int {

    var words: String {
        guard self > -1 else { return "" }

        switch self {
        case 0: return "zero"
        case 1: return "one"
        case 2: return "two"
        case 3: return "three"
        case 4: return "four"
        case 5: return "five"
        case 6: return "six"
        case 7: return "seven"
        case 8: return "eight"
        case 9: return "nine"
        case 10: return "ten"
        case 11: return "eleven"
        case 12: return "twelve"
        case 13: return "thirteen"
        case 15: return "fifteen"
        case 18: return "eighteen"
        case 14, 16, 17, 19: return "\((self % 10).words)teen"
        case 20: return "twenty"
        case 30: return "thirty"
        case 40: return "forty"
        case 50: return "fifty"
        case 80: return "eighty"
        case 60, 70, 90: return "\((self / 10).words)ty"
        case 21...99: return "\((self / 10 * 10).words) \((self % 10).words)"
        case 100...999:
            return
                "\((self / 100).words) hundred\( self % 100 > 0 ? " \((self % 100).words)" : "" )"
        case 1_000...999_999:
            return
                "\((self / 1000).words) thousand\( self % 1000 > 0 ? " \((self % 1000).words)" : "" )"
        case 1_000_000...999_999_999:
            return
                "\((self / 1_000_000).words) million\( self % 1_000_000 > 0 ? " \((self % 1_000_000).words)" : "" )"
        case 1_000_000_000...9_999_999_999:
            return
                "\((self / 1_000_000_000).words) billion\( self % 1_000_000_000 > 0 ? " \((self % 1_000_000_000).words)" : "" )"

        default: return "error"
        }
    }

}

///---------------------------------------------------------------------------------------
/// Leetcode 273
/// https://leetcode.com/problems/integer-to-english-words/
func numberToWords(_ num: Int) -> String {
    num.words.capitalized
}


//Int32.max
//assert(numberToWords(400) == "Four Hundred")
//assert(numberToWords(Int(Int32.max)) == "Two Billion One Hundred Forty Seven Million Four Hundred Eighty Three Thousand Six Hundred Forty Seven")
//for i in Int(Int32.max-100)...(Int(Int32.max)) {
//    print(numberToWords(i))
//}




func pangrams(s: String) -> String {
    var result = "pangram"
    var count = 0
    let alphaRange = Character("a").asciiValue!...Character("z").asciiValue!

    var hash = [UInt8: Bool]()  // asciiValue: isFound
    for c in alphaRange {
        hash[c] = false
    }

    for c in s {
        guard let ascii = Character(c.lowercased()).asciiValue,
            alphaRange.contains(ascii)
        else { continue }

        if let isFound = hash[ascii], isFound == false {
            hash[ascii] = true
            count += 1
        }
        if count == hash.count {
            return result
        }
    }

    return "not \(result)"

}

/*
pangrams(s: "We promptly judged antique ivory buckles for the next prize")
pangrams(s: "We promptly judged antique ivory buckles for the prize")
pangrams(s: "The quick brown fox jumps over the lazy dog")

*/

///---------------------------------------------------------------------------------------
/// Leetcode 1832
///https://leetcode.com/problems/check-if-the-sentence-is-pangram/description/
func checkIfPangram(_ sentence: String) -> Bool {
    let alphaRange = Character("a").asciiValue!...Character("z").asciiValue!
    return Set(sentence.trimmingCharacters(in: .whitespaces)).count != alphaRange.count
}

/*
checkIfPangram("We promptly judged antique ivory buckles for the next prize")
checkIfPangram("We promptly judged antique ivory buckles for the prize")
checkIfPangram("The quick brown fox jumps over the lazy dog")
*/

///---------------------------------------------------------------------------------------
/// Leetcode 481
///https://leetcode.com/problems/magical-string/description/
func magicalString(_ n: Int) -> Int {

    var turn = 1
    var count1 = 1
    var string = [1, 2, 2]
    var i = 2

    while string.count < n {

        let occurance = string[i]
        i += 1

        for j in 1...occurance {
            string.append(turn)

            if turn == 1 {
                count1 += 1
            }
            guard string.count < n else { break }
        }

        if turn == 1 {
            turn = 2
        } else {
            turn = 1
        }
    }

    //    print(string)

    return count1
}

/*
magicalString(6)
magicalString(4)
magicalString(7)

magicalString(20)
 */

/**

 1221121221221121122


 1 22 11 2 1 22 1 22 11 2 11 22
 1 2    2  1 1 2   1 2   2   1 2  2

 */



///---------------------------------------------------------------------------------------
/// Leetcode 226
///https://leetcode.com/problems/invert-binary-tree/description/
func invertTree(_ root: TreeNode?) -> TreeNode? {

    guard root != nil else { return nil }

    let right = root?.right
    root?.right = root?.left
    root?.left = right

    invertTree(root?.left)
    invertTree(root?.right)

    return root
}

//invertTree(TreeNode(2, TreeNode(1), nil))
//invertTree(TreeNode(1, TreeNode(2, TreeNode(3), nil), TreeNode(4, nil, TreeNode(5))))
//invertTree(TreeNode(4, TreeNode(2, TreeNode(1), TreeNode(3)), TreeNode(7, TreeNode(6), TreeNode(9))))
//invertTree(TreeNode(4, TreeNode(2, TreeNode(1), TreeNode(3)), nil))

///---------------------------------------------------------------------------------------
/// Leetcode 102
///https://leetcode.com/problems/binary-tree-level-order-traversal/description/
func add(_ root: TreeNode?, _ level: Int, array: inout [[Int]]) {

    guard let root else { return }

    if level == array.count {
        array.append([root.val])
    } else {
        array[level].append(root.val)
    }

    add(root.left, level + 1, array: &array)
    add(root.right, level + 1, array: &array)
}

func levelOrder(_ root: TreeNode?) -> [[Int]] {
    var array = [[Int]]()
    add(root, 0, array: &array)
    return array
}

//assert(levelOrder(TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))) == [[3], [9, 20], [15, 7]])
//assert(levelOrder(TreeNode(1)) == [[1]])
//assert(levelOrder(nil) == [])

/*
func constructBST(_ arr: [Int], _ start: Int, _ end: Int, _ root: TreeNode?) -> TreeNode? {

    guard start <= end else { return nil }

    let mid = (start + end) / 2

    var tree: TreeNode?
    if root == nil {
        tree = TreeNode(arr[mid])
    } else {
        tree = root
    }

    tree?.left = constructBST(arr, start, mid-1, tree?.left)
    tree?.right = constructBST(arr, mid+1, end, tree?.right)

    return tree
}

func sortedArrayToBST(_ nums: [Int]) -> TreeNode? {
    constructBST(nums, 0, nums.count - 1, nil)
}
 */

/*
func buildBST(_ arr: [Int]) -> TreeNode? {
    guard arr.count > 0 else { return nil }
    let mid = arr.count / 2
    let tree = TreeNode(arr[mid])
    tree.left = buildBST(Array(arr[0..<mid]))
    tree.right = buildBST(Array(arr[(mid+1)..<arr.count]))
    return tree
}
 */

/*
func sortedArrayToBST(_ nums: [Int]) -> TreeNode? {
    guard nums.count > 0 else { return nil }
    let mid = nums.count / 2
    let node = TreeNode(nums[mid])
    node.left = sortedArrayToBST(Array(nums[0..<mid]))
    node.right = sortedArrayToBST(Array(nums[(mid+1)..<nums.count]))
    return node
}
 */

///---------------------------------------------------------------------------------------
/// Leetcode 108
///https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/description/
class Leet0108 {
    
    func sortedArrayToBST(_ nums: [Int]) -> TreeNode? {
        guard nums.count > 0 else { return nil }
        let mid = nums.count / 2
        let node = TreeNode(nums[mid])
        node.left = sortedArrayToBST(Array(nums[0..<mid]))
        node.right = sortedArrayToBST(Array(nums[(mid + 1)..<nums.count]))
        return node
    }
    static func test() {
        let sut = Leet0108()
        assert(sut.sortedArrayToBST([-10, -3, 0, 5, 9]) != nil)
        assert(sut.sortedArrayToBST([1, 3]) != nil)
    }
    
}
//Leet0108.test()


///---------------------------------------------------------------------------------------
/// Leetcode 1207
/// https://leetcode.com/problems/unique-number-of-occurrences/description/
func uniqueOccurrences(_ arr: [Int]) -> Bool {

    var hash = [Int: Int]()  // num: count

    for i in arr {
        if let count = hash[i] {
            hash[i] = count + 1
        } else {
            hash[i] = 1
        }
    }

    return Set(hash.values).count == hash.count
}

//uniqueOccurrences([1,2,2,1,1,3])
//
//uniqueOccurrences([1,2])
//
//uniqueOccurrences([-3,0,1,-3,1,1,1,-3,10,0])

/*
func merge(_ nums1: inout [Int], _ m: Int, _ nums2: [Int], _ n: Int) {
    var result = [Int]()
    var index1 = nums1.count - 1
    var index2 = 0
    var num1 = 0

    while index1 >= 0 {
        if nums1[index1] == 0 {
            nums1.removeLast()
        } else {
            break
        }
        index1 -= 1
    }
//    print(nums1)

    for n1 in nums1 {

        while index2 < nums2.count {
            let num2 = nums2[index2]

            if n1 >= num2 {
                result.append(num2)
            } else {
//                print(n1)
                break
            }
            index2 += 1
        }

        num1 = n1
        result.append(n1)
    }

    while index2 < nums2.count {
        let num2 = nums2[index2]
        if num1 < num2 {
            result.append(num2)
        }

        index2 += 1
    }

//    print(result)

    nums1 = result
}

*/

/*

func merge(_ nums1: inout [Int], _ m: Int, _ nums2: [Int], _ n: Int) {

    var result = [Int]()
    var index1 = 0
    var index2 = 0
    var num1 = Int.min

    for i in 0 ..< m {

        let n1 = nums1[i]

        while index2 < nums2.count {
            let num2 = nums2[index2]

            if n1 >= num2 {
                result.append(num2)
            } else {
                break
            }
            index2 += 1
        }

        num1 = n1
        result.append(n1)
    }

    while index2 < n {
        let num2 = nums2[index2]
        if num1 < num2 {
            result.append(num2)
        }

        index2 += 1
    }

//    print(result)

    nums1 = result

}

 */


///---------------------------------------------------------------------------------------
/// Leetcode 88
/// https://leetcode.com/problems/merge-sorted-array/description/
func merge(_ nums1: inout [Int], _ m: Int, _ nums2: [Int], _ n: Int) {
    var end = m + n - 1
    var m = m - 1
    var n = n - 1

    while m >= 0 && n >= 0 {
        if nums1[m] > nums2[n] {
            nums1[end] = nums1[m]
            m -= 1
        } else {
            nums1[end] = nums2[n]
            n -= 1
        }
        end -= 1
    }

    while n >= 0 {
        nums1[end] = nums2[n]
        n -= 1
        end -= 1
    }
}

//var c1 = [-1,-1,0,0,0,0]
//merge(&c1, 4, [-1, 0], 2)

//var a1 = [1,2,3,0,0,0]
//merge(&a1, 3, [2,5,6], 3)

//var b1 = [2,4,5,6,0,0]
//merge(&b1, 3, [1,3,5,7,8,9], 3)

//var b1 = [1]
//merge(&b1, 1, [], 0)

//var b1 = [-1,0,0,3,3,3,0,0,0]
//merge(&b1, 6, [1,2,2], 3)

//var c1 = [0, 0, 0]
//merge(&c1, 0, [-50,-50,-48,-47,-44,-44,-37,-35,-35,-32,-32,-31,-29,-29,-28,-26,-24,-23,-23,-21,-20,-19,-17,-15,-14,-12,-12,-11,-10,-9,-8,-5,-2,-2,1,1,3,4,4,7,7,7,9,10,11,12,14,16,17,18,21,21,24,31,33,34,35,36,41,41,46,48,48], 63)

//          1      2      3      4     5
//let bads = [false, false, false, true, true]

//          1      2      3      4*    5     6     7     8     9     10
//let bads = [false, false, false, true, true, true, true, true, true, true]



///---------------------------------------------------------------------------------------
/// Leetcode 278
///https://leetcode.com/problems/first-bad-version/description/
class Leet0278 {
    
    var bads = Array(repeating: true, count: 100)
    init () {
        for i in 0...52 {
            bads[i] = false
        }
    }

    func isBadVersion(_ n: Int) -> Bool {
        bads[n - 1]
    }
    
    func firstBadVersion(_ n: Int) -> Int {
        var left = 1
        var right = n
        while left < right {
            let mid = left + (right - left) / 2
            if isBadVersion(mid) {
                right = mid
            } else {
                left = mid + 1
            }
        }
        return left
    }
}

/*
func firstBadVersion(_ n: Int) -> Int {

    print(n)

    guard n > 0 else {
        return n
    }

    if isBadVersion(n) {
        return firstBadVersion(n/2)
    } else {
        if isBadVersion(n+1) {
            return n + 1
        } else {
            return firstBadVersion(n + n/2)
        }
    }
}
 */

//firstBadVersion(100)


///---------------------------------------------------------------------------------------
/// Leetcode 412
///https://leetcode.com/problems/fizz-buzz/description/
func fizzBuzz(_ n: Int) -> [String] {

    var result = [String]()
    for i in 1...n {
        if i % 5 == 0 && i % 3 == 0 {
            result.append("FizzBuzz")
        } else if i % 3 == 0 {
            result.append("Fizz")
        } else if i % 5 == 0 {
            result.append("Buzz")
        } else {
            result.append("\(i)")
        }
    }

    return result
}

//fizzBuzz(15)

/*
 // BRUTEFORCE!!!
func erathosthenesSieve(_ n: Int) -> [Int] {

    guard n > 1 else { return [] }

    var sieve = Array(repeating: true, count: n)
    var primes = [Int]()

    for i in 2 ..< n {
        if sieve[i] {
            primes.append(i)

            var j = i + i
            while j < n {
                sieve[j] = false
                j += i
            }
        }
    }

    return primes
}
*/


func erathosthenesPrimes(_ n: Int) -> [Int] {
    guard n > 1 else {
        return []
    }

    var nonPrimes = Set<Int>()
    var primes = [Int]()

    let upperbound = Int(sqrt(Double(n))) + 1

    for i in 2 ..< upperbound {
        if !nonPrimes.contains(i) {
            primes.append(i)
            var j = i * i
            while j < n {
                nonPrimes.insert(j)
                j += i
            }
        }
    }

    for k in upperbound ..< n {
        if !nonPrimes.contains(k) {
            primes.append(k)
        }
    }
//    print(primes)
    return primes
}

func getPrimes(_ limit: Int) -> [Int] {
    var isPrime = Array(repeating: true, count: limit + 1)
    var primes: [Int] = []
    for number in 2...limit where isPrime[number] {
        primes.append(number)
        for multiple in stride(from: number * number, to: isPrime.count, by: number) {
            isPrime[multiple] = false
        }
    }
    return primes
}

/*
func countPrimes(_ n: Int) -> Int {
    erathosthenesPrimes(n).count
}
*/


///---------------------------------------------------------------------------------------
/// Leetcode 204
///https://leetcode.com/problems/count-primes/description/
func countPrimes(_ n: Int) -> Int {
    if n < 3 {
        return 0
    }

    var primes = Array(repeating: true, count: n)
    primes[0] = false
    primes[1] = false
    let upperLimit = max(2, Int(Double(n).squareRoot()))

    for i in 2...upperLimit {
        if primes[i] {
            for j in stride(from: i * i, to: n, by: i) {
                primes[j] = false
            }
        }
    }

//    print(primes.enumerated().compactMap { $0.element ? $0.offset : nil })

    return primes.reduce(0) { partialResult, isPrime in
        partialResult + (isPrime ? 1 : 0)
    }
}

//countPrimes(121)
//countPrimes(5 * 1000000)
//countPrimes(10)
//countPrimes(0)
//countPrimes(1)

//for i in 2 ... 10 {
//    print("primes of \(i) is \(getPrimes(i))\n")
//}
//
//for i in 2 ... 10 {
//    print("primes of \(i) is \(erathosthenesPrimes(i))\n")
//}


//[1,2,3].reduce(0, +)

//"magazine".reduce(into: [:]) { counts, letter in counts[letter, default: 0] += 1 }

func canWrite(ransomNote: String, from magazine: String) -> Bool {

    guard ransomNote.count <= magazine.count else {
        return false
    }

    var magazineCounts = magazine.reduce(into: [:]) { counts, letter in
        counts[letter, default: 0] += 1
    }

    var ransomCounts = ransomNote.reduce(into: [:]) { counts, letter in
        counts[letter, default: 0] += 1
    }

    for ransomCount in ransomCounts {

        guard let magazineCount = magazineCounts[ransomCount.key] else {
            return false
        }

        if ransomCount.value > magazineCount {
            return false
        }
    }

    //    for char in ransomNote {
    //        if let count = magazineCounts[char], count > 0 {
    //            magazineCounts[char]! -= 1
    //        } else {
    //            return false
    //        }
    //    }

    return true
}

//assert(canWrite(ransomNote: "a", from: "b") == false)
//assert(canWrite(ransomNote: "aa", from: "a") == false)
//assert(canWrite(ransomNote: "aa", from: "ab") == false)
//assert(canWrite(ransomNote: "aa", from: "aa") == true)
//assert(canWrite(ransomNote: "aa", from: "aaa") == true)
//assert(canWrite(ransomNote: "aa", from: "aab") == true)


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/ransom-note/
class Leet0383 {
    func canConstruct(_ ransomNote: String, _ magazine: String) -> Bool {
        let m = magazine
            .reduce(into: [Character:Int]()) { m, c in m[c, default: 0] += 1 }
        return ransomNote
            .reduce(into: [Character:Int]()) { r, c in r[c, default: 0] += 1 }
            .allSatisfy { m[$0.key] ?? 0 >= $0.value }
    }
}

func canConstruct(_ ransomNote: String, _ magazine: String) -> Bool {
    guard ransomNote.count <= magazine.count else {
        return false
    }
    
    var magazineCounts = magazine.reduce(into: [:]) { counts, letter in counts[letter, default: 0] += 1 }
    

    for c in ransomNote {
        if magazineCounts[c] == nil {
            return false
        }
        magazineCounts[c]! -= 1
        if magazineCounts[c]! < 0 {
            return false
        }
    }

    return true
}


//assert(canConstruct("a", "b") == false)
//assert(canConstruct("aa", "a") == false)
//assert(canConstruct("aa", "ab") == false)
//assert(canConstruct("aa", "aa") == true)
//assert(canConstruct("aa", "aaa") == true)
//assert(canConstruct("aa", "aab") == true)


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/power-of-two/
class Leet0231 {
    func isPowerOfTwo(_ n: Int) -> Bool {
        n > 0 && (n & (n-1)) == 0
    }
}


///---------------------------------------------------------------------------------------
/// Leetcode 326
/// https://leetcode.com/problems/power-of-three/description/
func isPowerOfThree(_ n: Int) -> Bool {
    return n > 0 && 1_162_261_467 % n == 0
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/power-of-four/
class Leet0342 {
    func isPowerOfFour(_ n: Int) -> Bool {
        n > 0 && (n & (n-1)) == 0 && (n & 0x55555555) != 0
    }
}

//for i in -1 ... 27 {
//    print("\(i)   \(isPowerOfThree(i))")
//}

///---------------------------------------------------------------------------------------
/// Leetcode 384
///https://leetcode.com/problems/shuffle-an-array/description/
class Leet0384 {

    private let og: [Int]
    private var nums: [Int]

    init(_ nums: [Int]) {
        self.nums = nums
        self.og = nums
    }

    func reset() -> [Int] {
        og
    }

    func shuffle() -> [Int] {
        nums.shuffle()
        return nums
    }
}

//var array = Leet0384([1,2,3, 4, 5, 6])
//print(array.shuffle())
//print(array.shuffle())
//print(array.shuffle())
//print(array.reset())
//print(array.shuffle())

/*
func addTwoNumbers(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {

    var carry = 0
    var sum = 0
    var head1 = l1
    var head2 = l2
    let start1 = l1
    let start2 = l2

    while true {

        if let h1 = head1, let h2 = head2 {
            sum = h1.val + h2.val + carry
            if sum > 9 {
                carry = 1
                sum -= 10
            } else {
                carry = 0
            }
            h1.val = sum
            h2.val = h1.val

            head1 = h1.next
            head2 = h2.next

            if head1 == nil && head2 == nil {
                if carry == 1 {
                    h1.next = ListNode(carry)
                }
                return start1
            }

        } else if let h1 = head1, head2 == nil {
            sum = h1.val + carry
            if sum > 9 {
                carry = 1
                sum -= 10
            } else {
                carry = 0
            }
            h1.val = sum

            head1 = h1.next

            if head1 == nil {
                if carry == 1 {
                    h1.next = ListNode(carry)
                }
                return start1
            }
        } else if let h2 = head2, head1 == nil {
            sum = h2.val + carry
            if sum > 9 {
                carry = 1
                sum -= 10
            } else {
                carry = 0
            }
            h2.val = sum

            head2 = h2.next

            if head2 == nil {
                if carry == 1 {
                    h2.next = ListNode(carry)
                }
                return start2
            }
        } else if head1 == nil && head2 == nil {
            return head1
        }
    }
}
 */

///---------------------------------------------------------------------------------------
/// Leetcode 2
/// https://leetcode.com/problems/add-two-numbers/description/
func addTwoNumbers(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
    let dummy = ListNode(0)  // Dummy node to simplify list operations
    var current = dummy
    var carry = 0

    var p = l1
    var q = l2

    while p != nil || q != nil || carry > 0 {
        let x = p?.val ?? 0
        let y = q?.val ?? 0
        let sum = x + y + carry
        carry = sum / 10

        current.next = ListNode(sum % 10)
        current = current.next!

        p = p?.next
        q = q?.next
    }

    return dummy.next
}

/*
// test for  l1 of 2, 4, 3 and l2 of 5, 6, 4

let l1 = ListNode(2)
l1.next = ListNode(4)
l1.next?.next = ListNode(3)

let l2 = ListNode(5)
l2.next = ListNode(6)
l2.next?.next = ListNode(4)

let result = addTwoNumbers(l1, l2)
print(result?.val)
print(result?.next?.val)
print(result?.next?.next?.val)
print()

// test for l1: 0, l2: 0

let l1_0 = ListNode(0)
let l2_0 = ListNode(0)

let result_0 = addTwoNumbers(l1_0, l2_0)
print(result_0?.val)
print()


// test for l1: 9,9,9,9,9,9,9 and l2: 9,9,9,9

let l1_9 = ListNode(9)
l1_9.next = ListNode(9)
l1_9.next?.next = ListNode(9)
l1_9.next?.next?.next = ListNode(9)
l1_9.next?.next?.next?.next = ListNode(9)
l1_9.next?.next?.next?.next?.next = ListNode(9)
l1_9.next?.next?.next?.next?.next?.next = ListNode(9)

let l2_9 = ListNode(9)
l2_9.next = ListNode(9)
l2_9.next?.next = ListNode(9)
l2_9.next?.next?.next = ListNode(9)

let result_9 = addTwoNumbers(l1_9, l2_9)
print(result_9?.val)
print(result_9?.next?.val)
print(result_9?.next?.next?.val)
print(result_9?.next?.next?.next?.val)
print(result_9?.next?.next?.next?.next?.val)
print(result_9?.next?.next?.next?.next?.next?.val)
print(result_9?.next?.next?.next?.next?.next?.next?.val)
print(result_9?.next?.next?.next?.next?.next?.next?.next?.val)
print()

// test for l1: 9, 9, 9 and l2: 1

let l1_1 = ListNode(9)
l1_1.next = ListNode(9)
l1_1.next?.next = ListNode(9)

let l2_1 = ListNode(1)

let result_1 = addTwoNumbers(l1_1, l2_1)
print(result_1?.val)
print(result_1?.next?.val)
print(result_1?.next?.next?.val)
print(result_1?.next?.next?.next?.val)
print(result_1?.next?.next?.next?.next?.val)
print()

// test for l1: 5 and l2: 5
let l1_5 = ListNode(5)
let l2_5 = ListNode(5)

let result_5 = addTwoNumbers(l1_5, l2_5)
print(result_5?.val)
print(result_5?.next?.val)
print()

*/


func add(_ n1: [Int], _ n2: [Int]) -> [Int] {
    var sumNum = [Int]()
    var carry = 0
    var i1 = 0
    var i2 = 0

    while i1 < n1.count || i2 < n2.count || carry > 0 {
        let num1 = i1 < n1.count ? n1[i1] : 0
        let num2 = i2 < n2.count ? n2[i2] : 0
        let sum = num1 + num2 + carry
        carry = sum / 10

        let digit = sum % 10
        sumNum.append(digit)

        i1 += 1
        i2 += 1
    }
    return sumNum
}

//assert(add([8, 3, 7], [0, 8, 3, 7]) == [8, 1, 1, 8])
//assert(add([1, 2, 3], [4, 5, 6]) == [5, 7, 9])
//assert(add([1], [2]) == [3])
//assert(add([], []) == [])
//assert(add([1], []) == [1])
//assert(add([], [1]) == [1])
//assert(add([9, 9, 9], [1]) == [0, 0, 0, 1])
//assert(add([1], [9, 9, 9]) == [0, 0, 0, 1])

///---------------------------------------------------------------------------------------
/// Leetcode 43
///https://leetcode.com/problems/multiply-strings/description/
func multiply(_ num1: String, _ num2: String) -> String {

    guard num1 != "0" && num2 != "0" else {
        return "0"
    }

    let num1Array = Array(num1).reversed()
    let num2Array = Array(num2).reversed().map { String($0) }
    var resultArray = [Int]()

    for i in 0..<num2Array.count {
        var tempResultArray = [Int]()
        var carry = 0
        let num2Digit = num2Array[i]

        guard let num2 = Int(num2Digit) else { return "error" }
        var temp: Int = 0

        for num1Digit in num1Array {
            guard let num1 = Int(String(num1Digit)) else { return "error" }
            let product = num1 * num2 + carry
            temp = product % 10
            carry = product / 10
            tempResultArray.append(temp)
        }

        if carry > 0 {
            tempResultArray.append(carry)
        }

        for _ in 0..<i {
            tempResultArray.insert(0, at: 0)
        }

        resultArray = add(resultArray, tempResultArray)
    }
    return resultArray.reversed().map { String($0) }.joined()
}

//assert(multiply("123", "456") == "56088")
//assert(multiply("456", "123") == "56088")
//assert(multiply("123", "66") == "8118")
//assert(multiply("66", "123") == "8118")
//assert(multiply("2", "3") == "6")
//assert(multiply("123456789", "987654321") == "121932631112635269")
//assert(multiply("987654321", "123456789") == "121932631112635269")
//assert(multiply("0", "0") == "0")
//assert(multiply("1", "0") == "0")
//assert(multiply("0", "1") == "0")
//assert(multiply("1", "1") == "1")
//assert(multiply("1", "9") == "9")
//assert(multiply("9", "1") == "9")
//assert(multiply("1", "99") == "99")
//assert(multiply("99", "1") == "99")
//assert(multiply("9", "999") == "8991")
//assert(multiply("999", "9") == "8991")
//assert(multiply("999", "999") == "998001")
//assert(multiply("0", "100") == "0")
//assert(multiply("1110", "0") == "0")


/*
 
 // JavaScript
 var climbStairs = function (n) {
     return climb_Stairs(0, n);
 };
 var climb_Stairs = function (i, n) {
     if (i > n) {
         return 0;
     }
     if (i == n) {
         return 1;
     }
     return climb_Stairs(i + 1, n) + climb_Stairs(i + 2, n);
 };
 
 */


func climbStairs(_ i: Int, _ n: Int) -> Int {
    
    if i > n {
        return 0
    }
    
    if i == n {
        return 1
    }
    
    return climbStairs(i + 1, n) + climbStairs(i + 2, n)
}

func climbStairsBruteForce(_ n: Int) -> Int {
    climbStairs(0, n)
}

///---------------------------------------------------------------------------------------
/// Leetcode 70
/// https://leetcode.com/problems/climbing-stairs/description/
func climbStairs(_ n: Int) -> Int {
    
    if 1...2 ~= n  {
        return n
    }
    var first = 1
    var second  = 2
    
    for i in 3...n {
        let third = first + second
        first = second
        second = third
    }
    return second
}

//for i in 1...45 {
//    print("\(i): \(climbStairs(i))")
//}

/*
// JavaScript
var climbStairs = function (n) {
    var sqrt5 = Math.sqrt(5);
    var phi = (1 + sqrt5) / 2;
    var psi = (1 - sqrt5) / 2;
    return Math.floor(
 (Math.pow(phi, n + 1) - Math.pow(psi, n + 1)) / sqrt5
 );
};
 */

func fibonacciFormula(_ n: Int) -> Int {
    let sqrt5 = 5.squareRoot()
    let phi = (1 + sqrt5) / 2
    let psi = (1 - sqrt5) / 2
    return Int((pow(phi, Double(n + 1)) - pow(psi, Double(n + 1))) / sqrt5 )
}

//for i in 1...45 {
//    print("\(i): \(fibonacciFormula(i))")
//}


/*
 1: 1
 2: 2
 3: 3
 4: 5
 5: 8
 6: 13
 7: 21
 8: 34
 9: 55
 10: 89
 11: 144
 12: 233
 13: 377
 14: 610
 15: 987
 16: 1597
 17: 2584
 18: 4181
 19: 6765
 20: 10946
 21: 17711
 22: 28657
 23: 46368
 24: 75025
 25: 121393
 26: 196418
 27: 317811
 28: 514229
 29: 832040
 30: 1346269
 31: 2178309
 32: 3524578
 33: 5702887
 34: 9227465
 35: 14930352
 */

func fibonacci(_ n : Int) -> Int {
    if (1...2) ~= n {
        return n
    }
    var first = 1
    var second = 2
    
    for i in 3...n {
        let third = first + second
        first = second
        second = third
    }
    
    return second
}

///---------------------------------------------------------------------------------------
/// Leetcode 509
///https://leetcode.com/problems/fibonacci-number/description/
func fib(_ n : Int) -> Int {
    if (0...1) ~= n {
        return n
    }
    return fibonacci(n-1)
}

//for i in 0...30 {
//    print("\(i): \(fib(i))")
//}





//func maxSumOfSubArray(in nums: [Int]) -> Int {
//    if nums.count == 1 {
//        return nums[0]
//    }
//    
//    let maxSum = nums.reduce(into: [Int]()) { maxSum, num in
//        if maxSum.isEmpty {
//            print("maxSum==\(maxSum), num== \(num)")
//            maxSum.append(num)
//        } else {
//            let prevMaxSum = maxSum[maxSum.count-1]
//            maxSum.append(max(prevMaxSum+num, num))
//            print("maxSum==\(maxSum), max between prevMaxSum==\(prevMaxSum) + num== \(num) [\(prevMaxSum + num)] and num== \(num)")
//        }
//    }
//    print(maxSum)
//    return maxSum.max() ?? 0
//}

//assert(maxSumOfSubArray(in: [5,4,-1,7,8]) == 23)
//assert(maxSumOfSubArray(in: [-2,1,-3,4,-1,2,1,-5,4]) == 6)
//assert(maxSumOfSubArray(in: [1]) == 1)
//assert(maxSumOfSubArray(in: [1,2]) == 3)
//assert(maxSumOfSubArray(in: [-2,-1,-3,-4,-1,-2,-1,-5,-4]) == -1)
//assert(maxSumOfSubArray(in: [-1]) == -1)
//assert(maxSumOfSubArray(in: []) == 0)
//assert(maxSumOfSubArray(in: [-1,-2]) == -1)

///---------------------------------------------------------------------------------------
/// Leetcode 53
/// https://leetcode.com/problems/maximum-subarray/description/
func maxSubArray(_ nums: [Int]) -> Int {
    if nums.count == 1 {
        return nums[0]
    }
    
    guard let first = nums.first else { return 0 }
    
    var currentSum = first
    var maxSum = first
    
    for i in 1..<nums.count {
        let current = nums[i]
        currentSum = max(currentSum+current, current)
        maxSum = max(maxSum, currentSum)
    }
    return maxSum
}

//assert(maxSubArray([-2,1000000000,-3,4,-1,2,1,-5,4]) == 1000000003)
//assert(maxSubArray([5,4,-1,7,8]) == 23)
//assert(maxSubArray([-2,1,-3,4,-1,2,1,-5,4]) == 6)
//assert(maxSubArray([1]) == 1)
//assert(maxSubArray([1,2]) == 3)
//assert(maxSubArray([-2,-1,-3,-4,-1,-2,-1,-5,-4]) == -1)
//assert(maxSubArray([-1]) == -1)
//assert(maxSubArray([]) == 0)
//assert(maxSubArray([-1,-2]) == -1)


func robDp(_ nums: [Int]) -> Int {
    guard !nums.isEmpty else { return 0 }
    guard nums.count > 1 else { return nums[0] }
        
    var dp: [Int] = Array(repeating: 0, count: nums.count + 1)
    dp[1] = nums[0]
    
    for i in 2..<nums.count + 1 {
        dp[i] = max(dp[i-1], dp[i-2] + nums[i-1])
    }
    
    return dp[nums.count]
}

//assert(robDp([100, 1, 1, 100]) == 200)
//assert(robDp([1,2,3,1]) == 4)
//assert(robDp([2,7,9,3,1]) == 12)
//assert(robDp([]) == 0)
//assert(robDp([1]) == 1)
//assert(robDp([2,1,1,2]) == 4)

///---------------------------------------------------------------------------------------
/// Leetcode 198
/// https://leetcode.com/problems/house-robber/description/
func rob(_ nums: [Int]) -> Int {
    guard !nums.isEmpty else { return 0 }
    guard nums.count > 1 else { return nums[0] }
        
    var prev = nums[0]
    var current = max(prev, nums[1])
    
    for i in 2..<nums.count {
        let temp = current
        current = max(prev + nums[i], current)
        prev = temp
    }
    
    return current
}

//assert(rob([100, 1, 1, 100]) == 200)
//assert(rob([1,2,3,1]) == 4)
//assert(rob([2,7,9,3,1]) == 12)
//assert(rob([]) == 0)
//assert(rob([1]) == 1)
//assert(rob([2,1,1,2]) == 4)


///---------------------------------------------------------------------------------------
/// Leetcode 155
/// https://leetcode.com/problems/min-stack/description/
class Leet0155 {
    
    class MinStack {
        
        var stack: [Int] = []
        var minStack: [Int] = []
        
        init() {
        }
        
        func push(_ x: Int) {
            stack.append(x)
            
            if minStack.isEmpty || minStack.last! >= x {
                minStack.append(x)
            }
        }
        
        func pop() {
            if stack.last! == minStack.last! {
                minStack.removeLast()
            }
            stack.removeLast()
        }
        
        func top() -> Int {
            stack.last!
        }
        
        func getMin() -> Int {
            minStack.last!
        }
    }
    
    static func test() {
        let minStack = MinStack()
        minStack.push(-2)
        minStack.push(0)
        minStack.push(-3)
        minStack.getMin()
        minStack.pop()
        minStack.top()
        minStack.getMin()
    }
}




/*
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop();
minStack.top();    // return 0
minStack.getMin(); // return -2
*/

///---------------------------------------------------------------------------------------
/// Leetcode 191
/// https://leetcode.com/problems/number-of-1-bits/description/
func hammingWeight(_ n: Int) -> Int {
    n.nonzeroBitCount
}

//assert(hammingWeight(11) == 3)
//assert(hammingWeight(128) == 1)
//assert(hammingWeight(2147483645) == 30)

///---------------------------------------------------------------------------------------
/// Leetcode 2220
/// https://leetcode.com/problems/minimum-bit-flips-to-convert-number/description/
///---------------------------------------------------------------------------------------
/// Leetcode 461
/// https://leetcode.com/problems/hamming-distance/description/
func hammingDistance(_ x: Int, _ y: Int) -> Int {
    (x ^ y).nonzeroBitCount
}

//hammingDistance(1, 2)
//hammingDistance(1, 3)
//hammingDistance(1, 4)
//hammingDistance(1, 10)
//hammingDistance(10, 7)
//
//0 ^ 0
//0 ^ 1
//1 ^ 0
//1 ^ 1
//
//7^10
//13.nonzeroBitCount

//11111111111111111111111111111101.bitPattern


//"00000010100101000001111010011100"
//"000000" + String(43261596, radix: 2)
//UInt16.max
//
//"00111001011110000010100101000000"
//"00" + String(964176192, radix: 2)
//
//"11111111111111111111111111111101"
//String(4294967293, radix: 2)
//
//"10111111111111111111111111111111"
//String(3221225471, radix: 2)
//
//
//"10111111111111111111111111111111".count
//
//Int(Int16.max) * 2
//UInt32.max
//1 << 16
//>> 16



//for i in 0 ... 32 {
//    print("\(i): \(pow(2,i))")
//}

/*
 
 // you need treat n as an unsigned value
 public int reverseBits(int n) {
     int ret = 0, power = 31;
     while (n != 0) {
         ret += (n & 1) << power;
         n = n >>> 1;
         power -= 1;
     }
     return ret;
 }
 
 */

func reverseBits2(_ n: Int) -> Int {
    var n = n
    var result = 0
    var power = 31
    
    while n != 0 {
        result += (n & 1) << power
        n >>= 1
        power -= 1
    }
    return result
}

extension String {
    func padded(to length: Int, with padding: Character = "0") -> String {
        guard self.count < length else { return self }
        return String(repeatElement(padding, count: length - self.count)) + self
    }
}

//"00000010100101000001111010011100"
//String(43261596, radix: 2).padded(to: 32)

//"00111001011110000010100101000000"
//String(964176192, radix: 2).padded(to: 32)


extension Decimal {
    var int: Int {
        return NSDecimalNumber(decimal: self).intValue
    }
}

//var n = 6
//String(n, radix: 2)
//n = n >> 1
//String(n, radix: 2)
//n = n >> 1
//String(n, radix: 2)
//n = n >> 1
//String(n, radix: 2)


/*
func reverseBits(_ n: Int) -> Int {
    //we know integer has 32 bits, so shift one side to left (right half will be filled with zeros) and other side to right (left half side will be filled with zeros) take a logical OR of them and you have swapped right and left side
    n = ( n >>> 16 | n << 16);
    //create mask 8bits of 1's then 8bits of 0's then 8bits of 1's and then 8 bits of 0's take logical AND between previous result to get location of 1's, then right shift it by 8 bits. At the same time create opposite mask (starting with 0's) and take logical AND of that and previous result. Combine both results by taking logical OR - you just swapped bits every 8 bits. Continue same pattern by dividing number by 2 previous number used for alternations and shifting (so next mask is alternating bits every 4 and then shift by 4)
    n = ((n & 0b11111111000000001111111100000000) >>> 8) | ((n & 0b00000000111111110000000011111111) << 8);
    n = ((n & 0b11110000111100001111000011110000) >>> 4) | ((n & 0b00001111000011110000111100001111) << 4);
    n = ((n & 0b11001100110011001100110011001100) >>> 2) | ((n & 0b00110011001100110011001100110011) << 2);
    n = ((n & 0b10101010101010101010101010101010) >>> 1) | ((n & 0b01010101010101010101010101010101) << 1);
    return n;
}
*/

///---------------------------------------------------------------------------------------
/// Leetcode 190
/// https://leetcode.com/problems/reverse-bits/description/
func reverseBits(_ n: Int) -> Int {
    var n = n >> 16 | n << 16
    n = ((n & 0b11111111000000001111111100000000) >> 8) | ((n & 0b00000000111111110000000011111111) << 8)
    n = ((n & 0b11110000111100001111000011110000) >> 4) | ((n & 0b00001111000011110000111100001111) << 4)
    n = ((n & 0b11001100110011001100110011001100) >> 2) | ((n & 0b00110011001100110011001100110011) << 2)
    n = ((n & 0b10101010101010101010101010101010) >> 1) | ((n & 0b01010101010101010101010101010101) << 1)
    return n
}

//reverseBits(43261596) // 964176192
//reverseBits(4294967293) // 3221225471
//assert(reverseBits(43261596) == 964176192)
//assert(reverseBits(4294967293) == 3221225471)
//assert(reverseBits(0) == 0)
//assert(reverseBits(1) == pow(2,31).int)
//assert(reverseBits(2) == pow(2,30).int)
//assert(reverseBits(3) == pow(2,31).int + pow(2,30).int)

///---------------------------------------------------------------------------------------
/// Leetcode 118
/// https://leetcode.com/problems/pascals-triangle/description/
func generatePascalsTriangle(_ numRows: Int) -> [[Int]] {
    
    if numRows <= 0 { return [] }
    if numRows == 1 { return [[1]] }
    if numRows == 2 { return [[1], [1, 1]] }
    
    var triangle: [[Int]] = [[1], [1, 1]]
    
    for row in 2..<numRows {
        triangle.append([])
        for col in 0..<row+1 {
            if col == 0 || col == row {
                triangle[row].append(1)
            } else {
                triangle[row].append(triangle[row-1][col-1] + triangle[row-1][col])
            }
        }
    }

//    print(triangle)
    return triangle
}

//generatePascalsTriangle(5)


///---------------------------------------------------------------------------------------
/// Leetcode 20
/// https://leetcode.com/problems/valid-parentheses/description/
class ParenthesisValidator {
    
    enum ParenthesisType: Character {
        case openParenthesis = "("
        case closeParenthesis = ")"
        case openCurlyBracket = "{"
        case closeCurlyBracket = "}"
        case openSquareBracket = "["
        case closeSquareBracket = "]"
        
        var closingType: ParenthesisType? {
            switch self {
            case .openParenthesis: return .closeParenthesis
            case .openCurlyBracket: return .closeCurlyBracket
            case .openSquareBracket: return .closeSquareBracket
            default: return nil
            }
        }
        
        var isOpen: Bool {
            switch self {
            case .openParenthesis, .openCurlyBracket, .openSquareBracket: return true
            default: return false
            }
        }
    }
    
    func isValid(_ s: String) -> Bool {
        var stack = Array<ParenthesisType>()
        let parenthesisTypes: [ParenthesisType] = [.openParenthesis, .openCurlyBracket, .openSquareBracket]
        let closingParenthesisTypes: [ParenthesisType] = [.closeParenthesis, .closeCurlyBracket, .closeSquareBracket]
        
        for char in s {
            guard let parenthesis = ParenthesisType(rawValue: char) else { return false }
            if parenthesisTypes.contains(parenthesis) {
                stack.append(parenthesis)
            } else if closingParenthesisTypes.contains(parenthesis) {
                if let open = stack.popLast() {
                    if open.closingType != parenthesis {
                        return false
                    }
                } else {
                    return false
                }
            }
        }
        
        return stack.isEmpty
    }
}


///---------------------------------------------------------------------------------------
///  Hacker Rank Test SPORTY GROUP
func matchingBraces(braces: [String]) -> [String] {
    // Write your code here

    let validator = ParenthesisValidator()
    return braces.map { validator.isValid($0) ? "YES" : "NO" }
    
}


//let parenthesisValidator = ParenthesisValidator()
//assert(parenthesisValidator.isValid("()"))
//assert(parenthesisValidator.isValid("()[]{}"))
//assert(parenthesisValidator.isValid("{[]}"))
//assert(parenthesisValidator.isValid("{[()]}"))
//assert(!parenthesisValidator.isValid("{"))
//assert(!parenthesisValidator.isValid("{[]()"))
//assert(!parenthesisValidator.isValid("{[}]"))
//assert(!parenthesisValidator.isValid("]"))
//assert(!parenthesisValidator.isValid("]]"))
//assert(!parenthesisValidator.isValid("}])"))


///---------------------------------------------------------------------------------------
/// Leetcode 298
/// https://leetcode.com/problems/missing-number/description/
func missingNumber(_ nums: [Int]) -> Int {
    
    var total = 0
    var numsTotal = 0
    
    for i in 0..<nums.count {
        total += i
        numsTotal += nums[i]
    }
    total += nums.count
    
    return total - numsTotal
}


//assert(missingNumber([3,0,1]) == 2) // n is 3
//assert(missingNumber([0,1]) == 2) // n is 2
//assert(missingNumber([1]) == 0) // n is 1
//assert(missingNumber([1,2,3]) == 0)
//assert(missingNumber([9,6,4,2,3,5,7,0,1]) == 8)


///---------------------------------------------------------------------------------------
/// Leetcode 1480
/// https://leetcode.com/problems/running-sum-of-1d-array/description/
func runningSum(_ nums: [Int]) -> [Int] {
    
    var runningSum = [Int]()
    
    for i in 0..<nums.count {
        if i == 0 {
            runningSum.append(nums[i])
        } else {
            runningSum.append(runningSum[i-1] + nums[i])
        }
    }
    
    return runningSum
}


//assert(runningSum([1,2,3]) == [1,3,6])
//assert(runningSum([1]) == [1])
//assert(runningSum([]) == [])
//assert(runningSum([1,2,3,4]) == [1,3,6,10])
//assert(runningSum([1,1,1,1,1]) == [1,2,3,4,5])
//assert(runningSum([3,1,2,10,1]) == [3,4,6,16,17])
//assert(runningSum([-1,-2,-3,-4,-5,-6]) == [-1,-3,-6,-10,-15,-21])
//assert(runningSum([1,2,3,4,5,6]) == [1,3,6,10,15,21])

///---------------------------------------------------------------------------------------
/// Leetcode 1672
/// https://leetcode.com/problems/richest-customer-wealth/description/
func maximumWealth(_ accounts: [[Int]]) -> Int {
    accounts.map { $0.reduce(0, +) }.max() ?? 0
}

//assert(maximumWealth([[1,2,3],[4,5,6],[7,8,9]]) == 24)
//assert(maximumWealth([[1,2,3],[4,5,6],[]]) == 15)
//assert(maximumWealth([[]]) == 0)
//assert(maximumWealth([[1,2,3],[3,2,1]]) == 6)
//assert(maximumWealth([[1,5],[7,3],[3,5]]) == 10)
//assert(maximumWealth([[2,8,7],[7,1,3],[1,9,5]]) == 17)




///---------------------------------------------------------------------------------------
/// Leetcode 1342
/// https://leetcode.com/problems/number-of-steps-to-reduce-a-number-to-zero/description/
func numberOfSteps(_ num: Int) -> Int {
    
    if num == 0 {
        return 0
    }
    
    var steps = 0
    var num = num
    
    while num != 0 {
        if num.isMultiple(of: 2) {
            num /= 2
        } else {
            num -= 1
        }
        steps += 1
    }
    
    return steps
}

//assert(numberOfSteps(0) == 0)
//assert(numberOfSteps(1) == 1)
//assert(numberOfSteps(14) == 6)
//assert(numberOfSteps(8) == 4)
//assert(numberOfSteps(123) == 12)

///---------------------------------------------------------------------------------------
/// Leetcode 876
///https://leetcode.com/problems/middle-of-the-linked-list/description/
func middleNode(_ head: ListNode?) -> ListNode? {
    var slow = head
    var fast = head
    
    while fast != nil && fast?.next != nil {
        slow = slow?.next
        fast = fast?.next?.next
    }
    
    return slow
}


//assert(middleNode(ListNode(1))?.val == 1)
//assert(middleNode(ListNode(1, ListNode(2)))?.val == 2)
//assert(middleNode(ListNode(1, ListNode(2, ListNode(3))))?.val == 2)
//assert(middleNode(ListNode(1, ListNode(2, ListNode(3, ListNode(4)))))?.val == 3)
//assert(middleNode(ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5))))))?.val == 3)
//assert(middleNode(ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5, ListNode(6)))))))?.val == 4)




///---------------------------------------------------------------------------------------
/// Leetcode 2095
/// https://leetcode.com/problems/delete-the-middle-node-of-a-linked-list/description/
func deleteMiddle(_ head: ListNode?) -> ListNode? {
    guard head?.next != nil else { return nil }
    var prev: ListNode? = nil
    var slow = head
    var fast = head
    
    while fast != nil && fast?.next != nil {
        prev = slow
        slow = slow?.next
        fast = fast?.next?.next
    }
    prev?.next = slow?.next
    slow?.next = nil
    return head
}


//assert(ListNode.toArray(deleteMiddle(ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5))))))) == [1,2,4,5])
//assert(ListNode.toArray(deleteMiddle(ListNode(1, ListNode(2, ListNode(3))))) == [1,3])
//assert(ListNode.toArray(deleteMiddle(ListNode(1, ListNode(3, ListNode(4, ListNode(7, ListNode(1, ListNode(2, ListNode(6))))))))) == [1,3,4,1,2,6] )
//assert(ListNode.toArray(deleteMiddle(ListNode(1, ListNode(2, ListNode(3, ListNode(4)))))) == [1,2,4])
//assert(ListNode.toArray(deleteMiddle(ListNode(1))) == [])
//assert(ListNode.toArray(deleteMiddle(ListNode(1, ListNode(2)))) == [1])
//assert(ListNode.toArray(deleteMiddle(ListNode(2, ListNode(1)))) == [2])


//func xxx_isPalindrome(_ x: Int) -> Bool {
//    guard x >= 0 else { return false }
//    
//    let reversed = String(String(x).reversed())
//    
//    guard reversed <= String(Int32.max) else {
//        print("reversed \(reversed) > \(Int32.max)")
//        return false
//    }
//    print("\(x) \(reversed)")
//    return String(x) == reversed
//}

// fails         assert(!isPalindrome(1000021))
//func xxx_isPalindrome(_ x: Int) -> Bool {
//    guard x >= 0 else { return false }
//    
//    guard x <= Int32.max else { return false }
//    
//    var num = x
//    var mod = 10
//    
//    while num > 9 {
//        let divisor = pow(10, String(num).count).int
//        let left = num / divisor
//        let right = num % mod
//
//        if left != right {
//            return false
//        }
//
//        mod *= 10
//        num -= left * divisor
//    }
//    
//    return true
//}






func xxx_isPalindrome(_ x: Int) -> Bool {
    guard x >= 0 else { return false }

    let reversed = String(String(x).reversed())

    return String(x) == reversed
}

/*
 
 function isPalindrome(x: number): boolean {
     // Special cases:
     // As discussed above, when x < 0, x is not a palindrome.
     // Also if the last digit of the number is 0, in order to be a palindrome,
     // the first digit of the number also needs to be 0.
     // Only 0 satisfy this property.
     if (x < 0 || (x % 10 == 0 && x != 0)) {
         return false;
     }

     let revertedNumber = 0;
     while (x > revertedNumber) {
         revertedNumber = revertedNumber * 10 + (x % 10);
         x = Math.floor(x / 10);
     }

     // When the length is an odd number, we can get rid of the middle digit by revertedNumber/10
     // For example when the input is 12321, at the end of the while loop we get x = 12, revertedNumber = 123,
     // since the middle digit doesn't matter in palidrome(it will always equal to itself), we can simply get rid of it.
     return x == revertedNumber || x == Math.floor(revertedNumber / 10);
 }
 
 */


///---------------------------------------------------------------------------------------
/// Leetcode 9
/// https://leetcode.com/problems/palindrome-number/description/
func isPalindrome(_ x: Int) -> Bool {
    if x < 0 || ( x % 10 == 0 && x != 0 ) {
        return false
    }
    
    var x = x
    var revertedNumber = 0
    
    while x > revertedNumber {
        revertedNumber = revertedNumber * 10 + ( x % 10 )
        x = x / 10
    }
    
    return x == revertedNumber || x == revertedNumber / 10
}

//assert(!isPalindrome(1000021))
//assert(isPalindrome(12321))
//assert(!isPalindrome(1232))
//assert(!isPalindrome(-1232))
//assert(!isPalindrome(123))
//assert(!isPalindrome(12))
//assert(isPalindrome(1))
//assert(isPalindrome(0))
//assert(!isPalindrome(-1))
//assert(!isPalindrome(2147483647))
//assert(isPalindrome(2147447412))
//for i in 1...9 {
//    assert(isPalindrome(i))
//}
//
//
//
//59 / 10
//
//pow(10, 0)
//
//
//pow(10, 3).int
//
//pow(2, 31) - 1
//Int32.max

//String(String(2147447412).reversed()) < String(Int32.max)
//String(String(Int32.max).reversed())
//String(2147483647) == String(Int32.max)
//String(Int32.max) < String(7463847412)


///---------------------------------------------------------------------------------------
/// Leetcode 289
/// https://leetcode.com/problems/game-of-life/description/
func gameOfLife(_ board: inout [[Int]]) {
    
    let rowCount = board.count
    guard let colCount = board.first?.count else { return }

    for row in 0..<rowCount {
        for col in 0..<colCount {
            var liveNeighbours = 0
            
            let top = max(0, row-1)
            let bottom = min(row+1, rowCount-1)
            let left = max(0, col-1)
            let right = min(col+1, colCount-1)
            
            for i in top..<bottom+1 {
                for j in left..<right+1 {
                    if !(i == row && j == col) && abs(board[i][j]) == 1 {
                        liveNeighbours += 1
                    }
                }
            }
            
            if board[row][col] == 1 {
                if liveNeighbours < 2 || liveNeighbours > 3 {
                    board[row][col] = -1
                }
            } else {
                if liveNeighbours == 3 {
                    board[row][col] = 2
                }
            }
        }
    }
    
    for row in 0..<rowCount {
        for col in 0..<colCount {
            board[row][col] = board[row][col] > 0 ? 1 : 0
        }
    }
}

//var gol = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
//gameOfLife(&gol)
//assert(gol == [[0,0,0],[1,0,1],[0,1,1],[0,1,0]])
//
//gol = [[1,1],[1,0]]
//gameOfLife(&gol)
//assert(gol == [[1,1],[1,1]])
//
//gol = [[1,1,0],[0,0,1],[0,0,0]]
//gameOfLife(&gol)
//assert(gol == [[0,1,0],[0,1,0],[0,0,0]])




//
//import SwiftUI
//import PlaygroundSupport
//
//// render and simulate game of life using swift ui
//
//struct ContentView: View {
//    @State var board: [[Int]] = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
//    
//    var body: some View {
//        VStack {
//            Text("Game of Life")
//                .font(.title)
//            Button("Next Generation") {
//                gameOfLife(&board)
//            }
//        }
//    }
//}
//
//struct GameOfLifeSwiftUiApp: App {
//    var body: some Scene {
//        WindowGroup {
//            ContentView()
//        }
//    }
//}
//
//PlaygroundPage.current.setLiveView(ContentView())


///---------------------------------------------------------------------------------------
/// Leetcode 1089
/// https://leetcode.com/problems/duplicate-zeros/description/
func duplicateZeros(_ arr: inout [Int]) {
    
    let count = arr.count
    var index = 0
    
    while index < count {
        if arr[index] == 0 {
            if index + 1 < count {
                arr.insert(0, at: index + 1)
            } else {
                arr.append(0)
            }
            arr.removeLast()
            index += 1
        }
        index += 1
    }
}

//var unduplicated = [1,0,2,3,0,4,5,0]
//duplicateZeros(&unduplicated)
//assert(unduplicated == [1,0,0,2,3,0,0,4])
//
//unduplicated = [1,2,3]
//duplicateZeros(&unduplicated)
//assert(unduplicated == [1,2,3])
//
//unduplicated = [0,0,0,0,0,0,0]
//duplicateZeros(&unduplicated)
//assert(unduplicated == [0,0,0,0,0,0,0])
//
//unduplicated = [8,4,5,0,0,0,0,7]
//duplicateZeros(&unduplicated)
//assert(unduplicated == [8,4,5,0,0,0,0,0])


///---------------------------------------------------------------------------------------
/// Leetcode 27
/// https://leetcode.com/problems/remove-element/description/
func removeElement(_ nums: inout [Int], _ val: Int) -> Int {
    var i = 0
    for j in 0..<nums.count {
        if nums[j] != val {
            nums[i] = nums[j]
            i += 1
        }
    }
//    print(nums)
    return i
}

//var removeElements = [3,2,2,3]
//
//removeElement(&removeElements, 3)
//assert(removeElements == [2,2,2,3])
//
//removeElements = [0,1,2,2,3,0,4,2]
//removeElement(&removeElements, 2)
//assert(removeElements == [0,1,3,0,4,0,4,2])


///---------------------------------------------------------------------------------------
/// Leetcode 35
/// https://leetcode.com/problems/search-insert-position/description/
func searchInsertIndex(_ nums: [Int], _ target: Int) -> Int {

    var mid = 0
    var low = 0
    var high = nums.count - 1
    
    while low <= high {
        mid = low + (high - low) / 2
        if nums[mid] == target {
            return mid
        } else if target < nums[mid] {
            high = mid - 1
        } else {
            low = mid + 1
        }
    }
    
    return low
}

//assert(searchInsertIndex([2,3,5,6], 1) == 0)
//assert(searchInsertIndex([1,3,5,6], 5) == 2)
//assert(searchInsertIndex([1,3,5,6], 2) == 1)
//assert(searchInsertIndex([1,3,5,6], 7) == 4)




///---------------------------------------------------------------------------------------
/// Leetcode 485
/// https://leetcode.com/problems/max-consecutive-ones/description/
func findMaxConsecutiveOnes(_ nums: [Int]) -> Int {
    var maxCount = 0
    var currentCount = 0
   
    for num in nums {
        if num == 1 {
            currentCount += 1
        } else {
            maxCount = max(maxCount, currentCount)
            currentCount = 0
        }
    }
    
    return max(maxCount, currentCount)
}

//assert(findMaxConsecutiveOnes([1,1,0,1,1,1]) == 3)
//assert(findMaxConsecutiveOnes([1,0,1,1,0,1]) == 2)
//assert(findMaxConsecutiveOnes([1,1,1,1]) == 4)
//assert(findMaxConsecutiveOnes([]) == 0)
//assert(findMaxConsecutiveOnes([1]) == 1)
//assert(findMaxConsecutiveOnes([0]) == 0)
//assert(findMaxConsecutiveOnes([1,0,1]) == 1)
//assert(findMaxConsecutiveOnes([0,1,0,1]) == 1)
//assert(findMaxConsecutiveOnes([0,1,1,0,1,1,1,0]) == 3)

//345/10
//12/10
//1/10

///---------------------------------------------------------------------------------------
/// Leetcode 1295
/// https://leetcode.com/problems/find-numbers-with-even-number-of-digits/description/
func countEvenLengthNumbers0(_ nums: [Int]) -> Int {
    var evenCount = 0
    for n in nums {
        var count = 0
        var num = n
        while num > 0 {
            count += 1
            num = num / 10
        }
        evenCount += (count % 2 == 0) ? 1 : 0
    }
    return evenCount
}


func countEvenLengthNumbers(_ nums: [Int]) -> Int {
    nums.reduce(0) {  String($1).count % 2 == 0 ? $0 + 1 : $0 }
}

//assert(countEvenLengthNumbers([12,345,2,6,7896]) == 2)
//assert(countEvenLengthNumbers([555,901,482,1771]) == 1)


extension Int {
    var count: Int {
        var n = self, count = 0
        while n != 0 {
            n /= 10
            count += 1
        }
        return count
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-numbers-with-even-number-of-digits/
class Leet1295 {
    func findNumbers(_ nums: [Int]) -> Int {
        nums.count { $0.count.isMultiple(of: 2) }
    }
}

///---------------------------------------------------------------------------------------
/// Leetcode 977
/// https://leetcode.com/problems/squares-of-a-sorted-array/
func sortedSquares(_ nums: [Int]) -> [Int] {
    var result = Array(repeating: 0, count: nums.count)
    var left = 0
    var right = nums.count - 1
    var i = result.count - 1
    
    while left <= right {
        let leftSquare = nums[left] * nums[left]
        let rightSquare = nums[right] * nums[right]
        
        if leftSquare > rightSquare {
            result[i] = leftSquare
            left += 1
        } else {
            result[i] = rightSquare
            right -= 1
        }
        i -= 1
    }
    
    return result
}


//assert(sortedSquares([-4,-1,0,3,10]) == [0,1,9,16,100])
//assert(sortedSquares([-7,-3,2,3,11]) == [4,9,9,49,121])
//
//sortedSquares([-5815])
//sortedSquares([-1868,5061])
//sortedSquares([-10,-9,-8,-7,-6,-5,-4,-3,-2,-1])
//sortedSquares([-9764,-9613,-8918,-8778,-8747,-8611,-8469,-8434,-7935,-7907,-7713,-7500,-7008,-6398,-6315,-5773,-5479,-5287,-5021,-5018,-4937,-4775,-4533,-4355,-4314,-4290,-4163,-3696,-3692,-3681,-3666,-3638,-3097,-3065,-3039,-2817,-2665,-2655,-2557,-2425,-2284,-2043,-822,-721,-623,-507,-396,-341,-224,-49])
//sortedSquares([309,381,681,1066,1127,1254,1436,1709,2305,2387,2392,2500,2518,2573,2665,2688,2835,2900,3182,3386,3441,3468,3987,4113,4256,4257,4277,4639,5040,5086,5614,5803,5856,6029,6226,6288,6447,6481,6726,6959,7303,7384,7669,7701,8198,9129,9548,9652,9794,9978])
//sortedSquares([-9662,-9489,-9264,-9225,-8439,-8177,-7675,-7398,-7379,-6374,-6295,-6199,-5457,-3899,-3762,-3696,-1638,-1316,-341,-76,503,1442,1707,2230,2729,2747,3633,3658,3763,3885,4552,4562,4688,4711,4972,5169,5355,5734,6482,6880,6938,7213,7467,7575,7940,8096,8603,8873,8979,9442])
//sortedSquares([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
//sortedSquares([-4280,-4280,-4280,-4280,-4280,-4280,-4280,-4280,-4280,-4280,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,3659,3659,3659,3659,3659,3659,3659,3659])
//sortedSquares([-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-19,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-18,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-17,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-16,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-15,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-14,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-13,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-11,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-6,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-3,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9])


///---------------------------------------------------------------------------------------
/// Leetcode 88
///https://leetcode.com/problems/merge-sorted-array/
class Leet0088 {
    
    var example1 = [1,2,3,0,0,0]
    var example2 = [1]
    var example3 = [0]

    func merge(_ nums1: inout [Int], _ m: Int, _ nums2: [Int], _ n: Int) {
        var m = m - 1
        var index = nums1.count - 1
        var n = n - 1
        
        while index >= 0 {
            let num1: Int
            let num2: Int
            num1 = (m >= 0) ? nums1[m] : Int.min
            num2 = (n >= 0) ? nums2[n] : Int.min
            
            if num1 < num2 {
                nums1[index] = num2
                n -= 1
            } else {
                nums1[index] = num1
                m -= 1
            }
            index -= 1
        }
    }
}

//let sut0088 = Leet0088()
//sut0088.merge(&sut0088.example1, 3, [2,5,6], 3)
//assert(sut0088.example1 == [1,2,2,3,5,6])
//
//sut0088.merge(&sut0088.example2, 1, [], 0)
//assert(sut0088.example2 == [1])
//
//sut0088.merge(&sut0088.example3, 0, [1], 1)
//assert(sut0088.example3 == [1])


///---------------------------------------------------------------------------------------
/// Leetcode 26
///https://leetcode.com/problems/remove-duplicates-from-sorted-array/description/
class Leet0026 {
 
    var example1 = [1,1,2]
    var example2 = [0,0,1,1,1,2,2,3,3,4]
    
    func removeDuplicates(_ nums: inout [Int]) -> Int {
        var currentIndex = 0
        for index in 0..<nums.count {
            if index == 0 || nums[index] != nums[index-1] {
                nums[currentIndex] = nums[index]
                currentIndex += 1
            }
        }
        return currentIndex
    }
}

//let sut0026 = Leet0026()
//assert(sut0026.removeDuplicates(&sut0026.example1) == 2)
//assert(sut0026.removeDuplicates(&sut0026.example2) == 5)


///---------------------------------------------------------------------------------------
/// Leetcode 2460
/// https://leetcode.com/problems/apply-operations-to-an-array/
class Leet2460 {

    var example1 = [1,0,2,0,0,1]
    var example2 = [1,2,2,1,1,0]
    var example3 = [0,1]
    
    func applyOperations(_ nums: [Int]) -> [Int] {
        var nums = nums
        
        for i in 0..<nums.count {
            guard i + 1 < nums.count else { break }
            if nums[i] == nums[i+1] {
                nums[i] *= 2
                nums[i+1] = 0
            }
        }
        moveZeroes(&nums)
        
        return nums
    }
    
    func moveZeroes(_ nums: inout [Int]) {
        var indexNonZero = 0
        
        for index in 0 ..< nums.count {
            if nums[index] != 0 {
                nums[indexNonZero] = nums[index]
                indexNonZero += 1
            }
        }
        
        var indexZero = indexNonZero
        
        while indexZero < nums.count {
            nums[indexZero] = 0
            indexZero += 1
        }
    }
}

//let sut2460 = Leet2460()
//assert(sut2460.applyOperations(sut2460.example1) == [1,2,1,0,0,0])
//assert(sut2460.applyOperations(sut2460.example2) == [1,4,2,0,0,0])
//assert(sut2460.applyOperations(sut2460.example3) == [1,0])



///---------------------------------------------------------------------------------------
/// Leetcode 1346
///https://leetcode.com/problems/check-if-n-and-its-double-exist/description/
class Leet1346 {
    
    var example1 = [10,2,5,3]
    var example2 = [7,1,14,11]
    var example3 = [3,1,7,11]
    var example4 = [0,0]
    var example5 = [-2,0,10,-19,4,6,-8]
    var example6 = [-16,-13,8]
    var example7 = [10,2,7,3,0,0,-13]
    var example8 = [7,15,3,4,30]
    var example9 = [0,2,-7,11,4,18]
    var example10 = [357,-53,277,-706,980,826,93,-352,-669,989,-193,920,209,-574,-389,221,383,352,665,873,759,-480,-64,-103,-721,-623,-642,-680,20,-168,528,-336,-656,264,581,-714,-458,721,815,106,328,476,351,325,-954,890,-174,635,95,-443,338,907,-648,113,-278,498,532,-778,95,-487,-909,-642,774,296,417,-132,-951,857,-867,321,-960,180,108,-984,-54,103,703,-118,-252,235,577,-703,842,-638,-888,-981,-246,484,202,328,661,447,-831,946,-888,-749,-702]
    var example11 = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997]
    
    func checkIfExist(_ arr: [Int]) -> Bool {
        var set: Set<Int> = []
        
        for num in arr {
            if set.contains(num * 2) || set.contains(num / 2) && num % 2 == 0 {
                print("\(num) ... \(set)")
                return true
            }
            set.insert(num)
        }
        return false
    }
}

//let sut1346 = Leet1346()
//assert(sut1346.checkIfExist(sut1346.example1))
//assert(sut1346.checkIfExist(sut1346.example2))
//assert(sut1346.checkIfExist(sut1346.example3) == false)
//assert(sut1346.checkIfExist(sut1346.example4))
//
//assert(sut1346.checkIfExist(sut1346.example5) == false)
//sut1346.checkIfExist(sut1346.example6)
//sut1346.checkIfExist(sut1346.example7)
//sut1346.checkIfExist(sut1346.example8)
//sut1346.checkIfExist(sut1346.example9)
//sut1346.checkIfExist(sut1346.example10)
//sut1346.checkIfExist(sut1346.example11)


///---------------------------------------------------------------------------------------
/// Leetcode 941
///https://leetcode.com/problems/valid-mountain-array/description/
class Leet0941 {

    var example1 = [2,1]
    var example2 = [3,5,5]
    var example3 = [0,3,2,1]
    var example4 = [3,6,5,6,7,6,5,3,0]
    
    func validMountainArray(_ arr: [Int]) -> Bool {
        var index = 0

        while index + 1 < arr.count && arr[index] < arr[index + 1] {
            index += 1
        }
        
        guard index > 0 && index + 1 < arr.count else {
            return false 
        }
        
        while index + 1 < arr.count && arr[index] > arr[index + 1] {
            index += 1
        }
        
        return index == arr.count - 1
    }
}

//let sut0941 = Leet0941()
//assert(!sut0941.validMountainArray(sut0941.example1))
//assert(!sut0941.validMountainArray(sut0941.example2))
//assert(sut0941.validMountainArray(sut0941.example3))
//assert(sut0941.validMountainArray(sut0941.example4))


///---------------------------------------------------------------------------------------
/// Leetcode 1299
/// https://leetcode.com/problems/replace-elements-with-greatest-element-on-right-side/description/
class Leet1299 {
    
    var case1: [Int] = [17,18,5,4,6,1]
    var case2: [Int] = [400]
    
    func replaceElements(_ arr: [Int]) -> [Int] {
        
        guard arr.count > 1 else {
            return [-1]
        }
        
        var arr = arr
        var index = arr.count - 1
        var right = arr[index]
        arr[index] = -1
        index -= 1
        
        while index >= 0 {
            let temp = arr[index]
            arr[index] = right
            right = max(right, temp)
            index -= 1
        }
        
        return arr
    }
}

//let sut1299 = Leet1299()
//assert(sut1299.replaceElements(sut1299.case1) == [18,6,6,6,1,-1])
//assert(sut1299.replaceElements(sut1299.case2) == [-1])


///---------------------------------------------------------------------------------------
/// Leetcode 905
///https://leetcode.com/problems/sort-array-by-parity/description/
class Leet0905 {
    
    var case1 = [3,1,2,4]
    var case2 = [0]
    var case3 = [-1,1]
    
    func sortArrayByParity(_ nums: [Int]) -> [Int] {
        var sorted = Array(nums.filter { Int($0) % 2 == 0})
        sorted.append(contentsOf: nums.filter { Int($0) % 2 != 0})
        return sorted
    }
}


//let sut0905 = Leet0905()
//assert(sut0905.sortArrayByParity(sut0905.case1) == [2,4,3,1])
//assert(sut0905.sortArrayByParity(sut0905.case2) == [0])
//assert(sut0905.sortArrayByParity(sut0905.case3) == [-1,1])


///---------------------------------------------------------------------------------------
/// Leetcode 1051
///https://leetcode.com/problems/height-checker/
class Leet1051 {
    
    var case1 = [1,1,4,2,1,3]
    var case2 = [5,1,2,3,4]
    var case3 = [1,2,3,4,5]
    
    func heightChecker(_ heights: [Int]) -> Int {
    
        var sorted = heights
        sorted.sort()
        
        var count = 0
        for (i, v) in heights.enumerated() {
            if v != sorted[i] {
                count += 1
            }
        }
        
        return count
    }
}

//let sut1051 = Leet1051()
//assert(sut1051.heightChecker(sut1051.case1) == 3)
//assert(sut1051.heightChecker(sut1051.case2) == 5)
//assert(sut1051.heightChecker(sut1051.case3) == 0)


///---------------------------------------------------------------------------------------
///  Hacker Rank Test SPORTY GROUP
class FrequencyDecoder {
    
    var case0 = """
    1226#24#
    """

    var case1 = """
    2110#(2)
    """

    var case2 = """
    23#(2)24#25#26#23#(3)
    """
    
    var case3 = """
    1(2)23(3)
    """
    
    enum Alpha: String, CaseIterable {
        case j = "10#"
        case k = "11#"
        case l = "12#"
        case m = "13#"
        case n = "14#"
        case o = "15#"
        case p = "16#"
        case q = "17#"
        case r = "18#"
        case s = "19#"
        case t = "20#"
        case u = "21#"
        case v = "22#"
        case w = "23#"
        case x = "24#"
        case y = "25#"
        case z = "26#"
        case a = "1"
        case b = "2"
        case c = "3"
        case d = "4"
        case e = "5"
        case f = "6"
        case g = "7"
        case h = "8"
        case i = "9"
        
        var index: Int {
            let result: Int
            if self.rawValue.count == 1 {
                result = Int(self.rawValue)! - 1
            } else {
                result = Int(self.rawValue.dropLast())! - 1
            }
            return result
        }
    }
    
    func frequency(s: String) -> [Int] {
        var results: [Int] = Alpha.allCases.map { _ in 0 }
        var s = s
        
        while !s.isEmpty {
            var count = 1
            if s.last == ")" {
                _ = s.popLast()
                if let index = s.lastIndex(of: "(") {
                    var countString = String(s[index..<s.endIndex])
                    countString.removeFirst()
                    count = Int(countString) ?? -1
                    s.removeSubrange(index..<s.endIndex)
                }
            }
            
            for alpha in Alpha.allCases {
                if s.hasSuffix(alpha.rawValue) {
                    s = String(s.dropLast(alpha.rawValue.count))
                    
                    results[alpha.index] += count
                    break
                }
            }
        }
        return results
    }
    
}


func frequency(s: String) -> [Int] {
    FrequencyDecoder().frequency(s: s)
}


//let decoder = FrequencyDecoder()
//assert(frequency(s: decoder.case0) == [1 ,1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1])
//assert(frequency(s: decoder.case1) == [1, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
//assert(frequency(s: decoder.case2) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1, 1, 1])
//assert(frequency(s: decoder.case3) == [2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])



///---------------------------------------------------------------------------------------
/// Leetcode 1309
///https://leetcode.com/problems/decrypt-string-from-alphabet-to-integer-mapping/
class Leet1309 {
    
    enum Alpha: String, CaseIterable {
        case j = "10#"
        case k = "11#"
        case l = "12#"
        case m = "13#"
        case n = "14#"
        case o = "15#"
        case p = "16#"
        case q = "17#"
        case r = "18#"
        case s = "19#"
        case t = "20#"
        case u = "21#"
        case v = "22#"
        case w = "23#"
        case x = "24#"
        case y = "25#"
        case z = "26#"
        case a = "1"
        case b = "2"
        case c = "3"
        case d = "4"
        case e = "5"
        case f = "6"
        case g = "7"
        case h = "8"
        case i = "9"
        
        var char: Character {
            switch self {
                case .a: return "a"
                case .b: return "b"
                case .c: return "c"
                case .d: return "d"
                case .e: return "e"
                case .f: return "f"
                case .g: return "g"
                case .h: return "h"
                case .i: return "i"
                case .j: return "j"
                case .k: return "k"
                case .l: return "l"
                case .m: return "m"
                case .n: return "n"
                case .o: return "o"
                case .p: return "p"
                case .q: return "q"
                case .r: return "r"
                case .s: return "s"
                case .t: return "t"
                case .u: return "u"
                case .v: return "v"
                case .w: return "w"
                case .x: return "x"
                case .y: return "y"
                case .z: return "z"
            }
        }
    }
    
    var case1 = "10#11#12"
    var case2 = "1326#"
    
    func freqAlphabets(_ s: String) -> String {
        var result = [String]()
        var s = s
        
        while !s.isEmpty {
            for alpha in Alpha.allCases {
                if s.hasSuffix(alpha.rawValue) {
                    s = String(s.dropLast(alpha.rawValue.count))
                    result.append(String(alpha.char))
                }
            }
        }
        
        return result.reversed().joined()
    }
}

//let leet1309 = Leet1309()
//assert(leet1309.freqAlphabets(leet1309.case1) == "jkab")
//assert(leet1309.freqAlphabets(leet1309.case2) == "acz")



///---------------------------------------------------------------------------------------
/// Leetcode 487
///https://leetcode.com/problems/max-consecutive-ones-ii/
/// Note in this sliding window we do not examine the elements equal to one. We only look at elements equal to 0 which makes the window invalid!
class Leet0487 {
    func findMaxConsecutiveOnes(_ nums: [Int]) -> Int {
        var numZeros = 0
        var left = 0
        var right = 0
        var longestLength = 0
        
        while right < nums.count {
            
            // increment numZeros when found
            if nums[right] == 0 {
                numZeros += 1
            }
            
            // Window is invalid. ie numZeros > 1 == 2.
            // Therefore contract the left side of the window by incrementing left and decreasing numZeros when value is zero
            while numZeros > 1 {
                if nums[left] == 0 {
                    numZeros -= 1
                }
                left += 1
            }
            
            // update longest length value
            longestLength = max(longestLength, right - left + 1)
            // move our right index
            right += 1
        }
               
        return longestLength
    }
}

//let sut0487 = Leet0487()
//assert(sut0487.findMaxConsecutiveOnes([0]) == 1)
//assert(sut0487.findMaxConsecutiveOnes([0, 0, 0, 0]) == 1)
//assert(sut0487.findMaxConsecutiveOnes([0, 0, 0, 1, 0, 1, 1, 0]) == 4)
//assert(sut0487.findMaxConsecutiveOnes([1, 0, 1, 1, 0]) == 4)
//assert(sut0487.findMaxConsecutiveOnes([1, 0, 1, 1, 0, 1]) == 4)
//assert(sut0487.findMaxConsecutiveOnes([1, 0, 1, 1, 0, 1, 1]) == 5)


///---------------------------------------------------------------------------------------
/// Leetcode 1004
///https://leetcode.com/problems/max-consecutive-ones-iii/
class Leet1004 {
    let case1 = ([1,1,1,0,0,0,1,1,1,1,0], 2)
    let case2 = ([0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1], 3)
    let case3 = ([1,0,1,0,1], 1)
    let case4 = ([1,1,1], 0)
    let case5 = ([1,0,1], 1)

    func longestOnes(_ nums: [Int], _ k: Int) -> Int {
        var numZeros = 0
        var left = 0
        var right = 0
        var longestLength = 0
        
        while right < nums.count {
            
            
            if nums[right] == 0 {
                numZeros += 1
            }
            
            while numZeros > k {
                if nums[left] == 0 {
                    numZeros -= 1
                }
                left += 1
            }
            
            longestLength = max(longestLength, right - left + 1)
            right += 1
        }
        
        return longestLength
    }
}

//let sut1004 = Leet1004()
//assert(sut1004.longestOnes(sut1004.case1.0, sut1004.case1.1) == 6)
//assert(sut1004.longestOnes(sut1004.case2.0, sut1004.case2.1) == 10)
//assert(sut1004.longestOnes(sut1004.case3.0, sut1004.case3.1) == 3)
//assert(sut1004.longestOnes(sut1004.case4.0, sut1004.case4.1) == 3)
//assert(sut1004.longestOnes(sut1004.case5.0, sut1004.case5.1) == 3)


///---------------------------------------------------------------------------------------
/// Leetcode 414
///https://leetcode.com/problems/third-maximum-number/description/
class Leet0414 {
    let case1 = ([3,2,1], 1)
    let case2 = ([1,2], 2)
    let case3 = ([2,2,3,1], 1)
    let case4 = ([1,1,1,1], 1)
    let case5 = ([1,2,2,5,3,5], 2)
    
    func thirdMax(_ nums: [Int]) -> Int {
        var threeMax: Set<Int> = []
        for num in nums {
            threeMax.insert(num)
            
            if threeMax.count > 3, let minValue = threeMax.min() {
                threeMax.remove(minValue)
            }
        }
        
        if threeMax.count == 3, let minValue = threeMax.min() {
            return minValue
        } else {
            return nums.max() ?? 0
        }
    }
}


//let sut0414 = Leet0414()
//assert(sut0414.thirdMax(sut0414.case1.0) == sut0414.case1.1)
//assert(sut0414.thirdMax(sut0414.case2.0) == sut0414.case2.1)
//assert(sut0414.thirdMax(sut0414.case3.0) == sut0414.case3.1)
//assert(sut0414.thirdMax(sut0414.case4.0) == sut0414.case4.1)
//assert(sut0414.thirdMax(sut0414.case5.0) == sut0414.case5.1)



///---------------------------------------------------------------------------------------
/// Leetcode 448
///https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/description/
class Leet0448 {
    let case1 = [4,3,2,7,8,2,3,1]
    let case2 = [1,1]
    func findDisappearedNumbers(_ nums: [Int]) -> [Int] {
        
        var nums = nums
        for num in nums {
            
            let index = num - 1
            
            if nums[index] > 0 {
                nums[index] *= -1
            }
        }
        
        var result: [Int] = []
        var index = 1
        for num in nums {
            if num > 0 {
                result.append(index)
            }
            index += 1
        }
        
        return result
   
    }
}

//let sut0448 = Leet0448()
//assert(sut0448.findDisappearedNumbers(sut0448.case1) == [5,6])
//assert(sut0448.findDisappearedNumbers(sut0448.case2) == [2])



///---------------------------------------------------------------------------------------
/// Leetcode 392
/// https://leetcode.com/problems/is-subsequence/
class Leet0392 {
    func isSubsequence(_ s: String, _ t: String) -> Bool {
        var i = s.startIndex
        var j = t.startIndex
        while i < s.endIndex && j < t.endIndex {
            if s[i] == t[j] {
                i = s.index(after: i)
            }
            j = t.index(after: j)
        }
        return i == s.endIndex
    }
}

//let sut0932 = Leet0392()
//assert(sut0932.isSubsequence("", "") == true)
//assert(sut0932.isSubsequence("a", "b") == false)
//assert(sut0932.isSubsequence("abc", "ahbgdc") == true)
//assert(sut0932.isSubsequence("axc", "ahbgdc") == false)
//assert(sut0932.isSubsequence("acb", "ahbgdc") == false)
//assert(sut0932.isSubsequence("leetcode", "yyyyylyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyeyyyyyyyyyyyyyyyyyyyeyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyytyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyycyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyoyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyydyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyeyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy") == true)



///---------------------------------------------------------------------------------------
/// Leetcode 541
///https://leetcode.com/problems/reverse-string-ii/description/
class Leet0541 {
    func reverseStr(_ s: String, _ k: Int) -> String {
        
        var result: [Character] = []
        var index = s.startIndex
        var isReversing = true
        
        while index < s.endIndex {
            let endIndex = s.index(index, offsetBy: k, limitedBy: s.endIndex) ?? s.endIndex
            var substring = s[index..<endIndex]
            if isReversing {
                substring = Substring(substring.reversed())
            }
            result.append(contentsOf: substring)
            
            isReversing.toggle()
            index = endIndex
        }
        return String(result)
    }
}

//let sut0541 = Leet0541()
//assert(sut0541.reverseStr("abcdefg", 2) == "bacdfeg")
//assert(sut0541.reverseStr("abcd", 2) == "bacd")
//assert(sut0541.reverseStr("abcdefg", 7) == "gfedcba")
//assert(sut0541.reverseStr("abcdefg", 1) == "abcdefg")
//assert(sut0541.reverseStr("hyzqyljrnigxvdtneasepfahmtyhlohwxmkqcdfehybknvdmfrfvtbsovjbdhevlfxpdaovjgunjqlimjkfnqcqnajmebeddqsgl", 39) == "fdcqkmxwholhytmhafpesaentdvxginrjlyqzyhehybknvdmfrfvtbsovjbdhevlfxpdaovjgunjqllgsqddebemjanqcqnfkjmi")

extension Character {
    var isVowel: Bool {
        switch self {
        case "a", "e", "i", "o", "u", "A", "E", "I", "O", "U":
            return true
        default:
            return false
        }
    }
}

///---------------------------------------------------------------------------------------
/// Leetcode 345
///https://leetcode.com/problems/reverse-vowels-of-a-string/description/?envType=problem-list-v2&envId=two-pointers
class Leet0345 {
    func reverseVowels(_ s: String) -> String {
        var left = 0
        var right = s.count - 1
        var stringAsArray: [Character] = Array(s)
        
        while left < right {
            guard stringAsArray[left].isVowel else {
                left += 1
                continue
            }
            guard stringAsArray[right].isVowel else {
                right -= 1
                continue
            }

            stringAsArray.swapAt(left, right)
            
            left += 1
            right -= 1
        }
        return String(stringAsArray)
    }
}

//let sut0345 = Leet0345()
//assert(sut0345.reverseVowels("IceCreAm") == "AceCreIm")
//assert(sut0345.reverseVowels("leetcode") == "leotcede")
//assert(sut0345.reverseVowels("aA") == "Aa")
//assert(sut0345.reverseVowels("aeiou") == "uoiea")
//assert(sut0345.reverseVowels("hello") == "holle")
//assert(sut0345.reverseVowels("racecar") == "racecar")
//assert(sut0345.reverseVowels("hanna") == "hanna")
//assert(sut0345.reverseVowels("race car") == "race car")
//assert(sut0345.reverseVowels("A man, a plan, a canal: Panama!") == "a man, a plan, a canal: PanamA!")
//assert(sut0345.reverseVowels("Yo! Bottoms up, U.S. Motto, boy!") == "Yo! Bottoms Up, u.S. Motto, boy!")


///---------------------------------------------------------------------------------------
/// Leetcode 202
///https://leetcode.com/problems/happy-number/description/
class Leet0202 {
    func isHappy(_ n: Int) -> Bool {
        if [1,7].contains(n) {
            return true
        }
        guard n > 9 else {
            return false
        }
        var n = n
        var sum = 0
        
        while n > 0 {
            let digit = n % 10
            sum += digit * digit
            n /= 10
        }
        return isHappy(sum)
    }
}

//let sut0202 = Leet0202()
//assert(sut0202.isHappy(19) == true)
//assert(sut0202.isHappy(2) == false)
//assert(sut0202.isHappy(1) == true)
//assert(sut0202.isHappy(7) == true)
//assert(sut0202.isHappy(10) == true)
//assert(sut0202.isHappy(1111111) == true)
//assert(sut0202.isHappy(99) == false )
//assert(sut0202.isHappy(999) == false )
//assert(sut0202.isHappy(9999999999999) == false )


extension String {

    func lastNumber() -> Int? {
        guard !isEmpty else { return nil }
        var end = self.index(before: self.endIndex)
        return lastNumber(from: &end)
    }
    
    func lastNumber(from index: inout String.Index) -> Int? {
        var end = index
        var number: Int? = nil
        // look for first number from right
        while self.startIndex < index, self[index].isNumber == false {
            end = self.index(before: end)
            index = end
        }
        // look for the start and end number from right
        while self.startIndex <= index, let num = Int(String(self[index...end])) {
            number = num
            guard startIndex < index else {
                break
            }
            index = self.index(before: index)
        }
        return number
    }
}
//var sutLastNumber = "123abc456def789"
//assert(sutLastNumber.lastNumber() == 789)
//sutLastNumber.removeLast(1)
//assert(sutLastNumber.lastNumber() == 78)
//sutLastNumber.removeLast(2)
//assert(sutLastNumber.lastNumber() == 456)
//sutLastNumber.removeLast(6)
//assert(sutLastNumber.lastNumber() == 123)
//assert("".lastNumber() == nil)

extension String {

    func firstNumber() -> Int? {
        var start = self.startIndex
        return firstNumber(from: &start)
    }
    
    func firstNumber(from index: inout String.Index) -> Int? {
        var start = index
        var number: Int? = nil
        // look for first number from left
        while index < self.endIndex, self[index].isNumber == false {
            start = self.index(after: start)
            index = start
        }
        // look for the start and end number from left
        while index < self.endIndex, let num = Int(String(self[start...index])) {
            number = num
            index = self.index(after: index)
        }
        return number
    }
}
//var sutFirstNumber = "123abc456def789"
//assert(sutFirstNumber.firstNumber() == 123)
//sutFirstNumber.removeFirst(1)
//assert(sutFirstNumber.firstNumber() == 23)
//sutFirstNumber.removeFirst(2)
//assert(sutFirstNumber.firstNumber() == 456)
//sutFirstNumber.removeFirst(6)
//assert(sutFirstNumber.firstNumber() == 789)
//assert("".firstNumber() == nil)

///---------------------------------------------------------------------------------------
/// Leetcode 408
///https://leetcode.com/problems/valid-word-abbreviation/description/?envType=problem-list-v2&envId=two-pointers
class Leet0408 {
    
    func validWordAbbreviation(_ word: String, _ abbr: String) -> Bool {
        
        var wordIndex = word.startIndex
        var abbrIndex = abbr.startIndex
        
        while wordIndex < word.endIndex && abbrIndex < abbr.endIndex {
            let wordChar = word[wordIndex]
            let abbrChar = abbr[abbrIndex]
            if wordChar == abbrChar {
                wordIndex = word.index(after: wordIndex)
                abbrIndex = abbr.index(after: abbrIndex)
            } else {
                if abbr[abbrIndex] == "0" {
                    return false
                }
                // extract number
                if let count = abbr.firstNumber(from: &abbrIndex) {
                    if let index = word.index(wordIndex, offsetBy: count, limitedBy: word.endIndex) {
                        wordIndex = index
                    } else {
                        return false
                    }
                }
                if wordIndex < word.endIndex && abbrIndex < abbr.endIndex {
                    let wordChar = word[wordIndex]
                    let abbrChar = abbr[abbrIndex]
                    if wordChar != abbrChar  {
                        return false
                    }
                }
            }
        }
        return  wordIndex == word.endIndex && abbrIndex == abbr.endIndex
    }
}
//let sut0408 = Leet0408()
//assert(sut0408.validWordAbbreviation("internationalization", "i12iz4n"))
//assert(sut0408.validWordAbbreviation("apple", "a2e") == false)
//assert(sut0408.validWordAbbreviation("apple", "a2e") == false)
//assert(sut0408.validWordAbbreviation("ppee", "2e") == false)
//assert(sut0408.validWordAbbreviation("substitution", "s10n"))
//assert(sut0408.validWordAbbreviation("substitution", "sub4u4"))
//assert(sut0408.validWordAbbreviation("substitution", "12"))
//assert(sut0408.validWordAbbreviation("substitution", "su3i1u2on"))
//assert(sut0408.validWordAbbreviation("substitution", "substitution"))
//assert(sut0408.validWordAbbreviation("substitution", "s55n") == false)
//assert(sut0408.validWordAbbreviation("substitution", "s010n") == false)
//assert(sut0408.validWordAbbreviation("substitution", "s0ubstitution") == false)
//assert(sut0408.validWordAbbreviation("hi", "02") == false)


///---------------------------------------------------------------------------------------
/// Leetcode 349
/// https://leetcode.com/problems/intersection-of-two-arrays/description/?envType=problem-list-v2&envId=two-pointers
class Leet0349 {
    
    let case1 = ([1, 2, 2, 1], [2, 2])
    let case2 = ([4, 9, 5], [9, 4, 9, 8, 4])
    func intersection(_ nums1: [Int], _ nums2: [Int]) -> [Int] {
        Array(Set(nums1).intersection(Set(nums2)))
    }
}
//let sut0349 = Leet0349()
//assert(sut0349.intersection(sut0349.case1.0, sut0349.case1.1) == [2])
//assert([[9, 4], [4, 9]].contains(sut0349.intersection(sut0349.case2.0, sut0349.case2.1)))



///---------------------------------------------------------------------------------------
/// Leetcode 0557
///https://leetcode.com/problems/reverse-words-in-a-string-iii/description/
class Leet0557 {
    func reverseWords(_ s: String) -> String {
        var result = ""
        for word in s.split(separator: " ") {
            result.append(" \(String(word.reversed()))")
        }
        result.trimPrefix(" ")
        return result
    }
}
//let sut0557 = Leet0557()
//assert(sut0557.reverseWords("Let's take LeetCode contest") == "s'teL ekat edoCteeL tsetnoc")
//assert(sut0557.reverseWords("Mr Ding") == "rM gniD")


///---------------------------------------------------------------------------------------
/// Leetcode 1768
///https://leetcode.com/problems/merge-strings-alternately/?envType=problem-list-v2&envId=two-pointers
class Leet1768 {
    func mergeAlternately0(_ word1: String, _ word2: String) -> String {
        var result = ""
        var index1 = word1.startIndex
        var index2 = word2.startIndex
        while index1 < word1.endIndex && index2 < word2.endIndex {
            result.append(word1[index1])
            result.append(word2[index2])
            index1 = word1.index(after: index1)
            index2 = word2.index(after: index2)
        }
        if index1 < word1.endIndex {
            result.append(contentsOf: word1[index1...])
        }
        if index2 < word2.endIndex {
            result.append(contentsOf: word2[index2...])
        }
        return result
    }
    
    func mergeAlternately(_ word1: String, _ word2: String) -> String {
        var result = ""
        let word1 = Array(word1)
        let word2 = Array(word2)
        
        for i in 0..<max(word1.count, word2.count) {
            if i < word1.count {
                result.append(word1[i])
            }
            if i < word2.count {
                result.append(word2[i])
            }
        }
        return result
    }
}
//let sut1768 = Leet1768()
//assert(sut1768.mergeAlternately("abc", "pqr") == "apbqcr")
//assert(sut1768.mergeAlternately("ab", "pqrs") == "apbqrs")
//assert(sut1768.mergeAlternately("abcd", "pq") == "apbqcd")







///---------------------------------------------------------------------------------------
/// Leetcode 170
/// https://leetcode.com/problems/two-sum-iii-data-structure-design/?envType=problem-list-v2&envId=two-pointers
class Leet0170 {
    var nums : [Int:Int]
    init() {
        nums = [:]
    }
    func add(_ number: Int) {
        nums[number, default: 0] += 1
    }
    func find(_ value: Int) -> Bool {
        for (num, count) in nums {
            let complement = value - num
            if complement == num {
                if nums[complement]! > 1 {
                    return true
                }
            } else {
                if nums[complement] != nil {
                    return true
                }
            }
        }
        return false
    }
    
    static func test() {
        let sut0170 = Leet0170()
        sut0170.add(1)
        sut0170.add(3)
        sut0170.add(5)
        assert(sut0170.find(4) == true)
        assert(sut0170.find(7) == false)
    }
}
//Leet0170.test()



///---------------------------------------------------------------------------------------
/// Leetcode 246
/// https://leetcode.com/problems/strobogrammatic-number
class Leet0246 {
    
    /***
     6 => 9
     9 => 6
     8 => 8
     1 => 1
     0 => 0
     */
    
    func isStrobogrammatic(_ num: String) -> Bool {
        var left = num.startIndex
        var right = num.index(before: num.endIndex)
        let map: [Character: Character] = ["6": "9", "9": "6", "8": "8", "1": "1", "0": "0"]
        while left <= right {
            guard let rightChar = map[num[right]], num[left] == rightChar else {
                return false
            }
            left = num.index(after: left)
            guard num.startIndex < right else { break }
            right = num.index(before: right)
        }
        return true
    }
    static func test() {
        let sut0246 = Leet0246()
        assert(sut0246.isStrobogrammatic("69"))
        assert(sut0246.isStrobogrammatic("88"))
        assert(sut0246.isStrobogrammatic("00"))
        assert(sut0246.isStrobogrammatic("11"))
        assert(sut0246.isStrobogrammatic("9960966"))
        assert(sut0246.isStrobogrammatic("962") == false)
        assert(sut0246.isStrobogrammatic("9") == false)
        assert(sut0246.isStrobogrammatic("6") == false)
        assert(sut0246.isStrobogrammatic("8"))
        assert(sut0246.isStrobogrammatic("0"))
        assert(sut0246.isStrobogrammatic("1"))
    }
}
//Leet0246.test()



///---------------------------------------------------------------------------------------
/// Leetcode 455
/// https://leetcode.com/problems/assign-cookies
class Leet0455 {
    func findContentChildren(_ g: [Int], _ s: [Int]) -> Int {
        var g = g.sorted()
        var s = s.sorted()
        var gIndex = 0
        var sIndex = 0
        while gIndex < g.count && sIndex < s.count {
            if s[sIndex] >= g[gIndex] {
                gIndex += 1
            }
            sIndex += 1
        }
        return gIndex
    }
    static func test() {
        let sut0455 = Leet0455()
        assert(sut0455.findContentChildren([4,7,9], [8,2,5,8]) == 2)
        assert(sut0455.findContentChildren([1,1,1], [10]) == 1)
        assert(sut0455.findContentChildren([1,2,3], [1,1]) == 1)
        assert(sut0455.findContentChildren([1,2], [1,2,3]) == 2)
    }
}
//Leet0455.test()




///---------------------------------------------------------------------------------------
/// Leetcode 2410
/// https://leetcode.com/problems/maximum-matching-of-players-with-trainers/
class Leet2410 {
    func matchPlayersAndTrainers(_ players: [Int], _ trainers: [Int]) -> Int {
        var players = players.sorted()
        var trainers = trainers.sorted()
        var indexPlayer = 0
        var indexTrainer = 0
        while indexPlayer < players.count && indexTrainer < trainers.count {
            if trainers[indexTrainer] >= players[indexPlayer] {
                indexPlayer += 1
            }
            indexTrainer += 1
        }
        return indexPlayer
    }
}
//let sut2410 = Leet2410()
//assert(sut2410.matchPlayersAndTrainers([4,7,9], [8,2,5,8]) == 2)
//assert(sut2410.matchPlayersAndTrainers([1,1,1], [10]) == 1)
//assert(sut2410.matchPlayersAndTrainers([1,2,3], [1,1]) == 1)
//assert(sut2410.matchPlayersAndTrainers([1,2], [1,2,3]) == 2)





///---------------------------------------------------------------------------------------
/// Leetcode 925
/// https://leetcode.com/problems/long-pressed-name/
class Leet0925 {
    func isLongPressedName(_ name: String, _ typed: String) -> Bool {
        var nameIndex = name.startIndex
        var typedIndex = typed.startIndex
        
        while nameIndex < name.endIndex && typedIndex < typed.endIndex {
            if name[nameIndex] == typed[typedIndex] {
                nameIndex = name.index(after: nameIndex)
                typedIndex = typed.index(after: typedIndex)
            } else if typed.startIndex < typedIndex && typed[typedIndex] == typed[typed.index(before: typedIndex)] {
                typedIndex = typed.index(after: typedIndex)
            } else {
                return false
            }
        }
        guard nameIndex == name.endIndex else { return false }
        
        while typedIndex < typed.endIndex && typed[typedIndex] == typed[typed.index(before: typedIndex)] {
            typedIndex = typed.index(after: typedIndex)
        }
        return typedIndex == typed.endIndex
    }
}
//let sut0925 = Leet0925()
//assert(sut0925.isLongPressedName("alex", "aaleex"))
//assert(sut0925.isLongPressedName("alex", "aaleexa") == false)
//assert(sut0925.isLongPressedName("saeed", "ssaaedd") == false)
//assert(sut0925.isLongPressedName("leetcode", "leetcode"))
//assert(sut0925.isLongPressedName("vtkgn", "vttkgnn"))
//assert(sut0925.isLongPressedName("leetcode", "leetcodeeee"))
//assert(sut0925.isLongPressedName("alex", "alexs") == false)
//assert(sut0925.isLongPressedName("a", "b") == false)


///---------------------------------------------------------------------------------------
/// Leetcode 680
///O(n) + O(1) Time and Space complexity
class Leet0680 {
    private func isPalindrome(_ s: String, left: String.Index, right: String.Index) -> Bool {
        var left = left
        var right = right
        while left < right {
            guard s[left] == s[right] else { return false }
            left = s.index(after: left)
            right = s.index(before: right)
        }
        return true
    }
    
    func validPalindrome(_ s: String) -> Bool {
        var left = s.startIndex
        var right = s.index(before: s.endIndex)
        
        while left < right {
            if s[left] != s[right] {
                return isPalindrome(s, left: s.index(after: left), right: right) || isPalindrome(s, left: left, right: s.index(before: right))
            }
            left = s.index(after: left)
            right = s.index(before: right)
        }
        return true
    }
    
    static func test() {
        let sut0680 = Leet0680()
        assert(sut0680.validPalindrome("deddde") == true)
        assert(sut0680.validPalindrome("aba") == true)
        assert(sut0680.validPalindrome("abca") == true)
        assert(sut0680.validPalindrome("abc") == false)
        assert(sut0680.validPalindrome("abcd") == false)
    }
}





///---------------------------------------------------------------------------------------
/// Leetcode
///https://leetcode.com/problems/count-binary-substrings/
class Leet0696 {
    
    func countBinarySubstrings(_ s: String) -> Int {
        
        guard s.count > 1 else { return 0 }
        
        var index = s.index(after: s.startIndex)
        var countPrevious = 0
        var countCurrent = 1
        var total = 0
        
        while index < s.endIndex {
            let currentChar = s[index]
            let previousChar = s[s.index(before: index)]
            
            if previousChar != currentChar {
                // then calculate the minimum counts and accumulate to total
                total += min(countPrevious, countCurrent)
                                               
                // then reset the new group
                countPrevious = countCurrent
                countCurrent = 1
            } else {
                countCurrent += 1
            }
            index = s.index(after: index)
        }
        total += min(countPrevious, countCurrent)
        
        return total
    }
    
    static func run() {
        let sut = Leet0696()
        assert(sut.countBinarySubstrings("00110011") == 6)
        assert(sut.countBinarySubstrings("10101") == 4)
        assert(sut.countBinarySubstrings("0") == 0)
        assert(sut.countBinarySubstrings("1") == 0)
        assert(sut.countBinarySubstrings("01") == 1)
    }
}
//Leet0696.run()


///---------------------------------------------------------------------------------------
/// Leetcode
///https://leetcode.com/problems/shortest-distance-to-a-character/
class Leet0821 {
    
    func shortestToChar(_ s: String, _ c: Character) -> [Int] {
        var result = Array(repeating: Int.max, count: s.count)
        var prev = Int.min / 2
        var index = 0
        for ch in s {
            if ch == c {
                prev = index
            }
            result[index] = abs(index - prev)
            index += 1
        }
        
        prev = Int.min / 2
        index = s.count - 1
        var stringIndex = s.index(before: s.endIndex)
        while s.startIndex < stringIndex {
            let ch = s[stringIndex]
            if ch == c {
                prev = index
            }
            result[index] = min(result[index], abs(prev - index))
            index -= 1
            stringIndex = s.index(before: stringIndex)
        }
        result[index] = min(result[index], abs(prev - index))
        return result
    }
    
    static func test() {
        let sut = Leet0821()
        assert(sut.shortestToChar("loveleetcode", "e") == [3,2,1,0,1,0,0,1,2,2,1,0])
        assert(sut.shortestToChar("aaab", "b") == [3,2,1,0])
    }
}
//Leet0821.test()









///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/reverse-only-letters
class Leet0917 {
    func reverseOnlyLetters(_ s: String) -> String {
        var result = Array(s)
        var left = 0
        var right = s.count - 1
        
        while left < right {
            let leftChar = result[left]
            let rightChar = result[right]
            
            guard leftChar.isLetter else {
                left += 1
                continue
            }
            
            guard rightChar.isLetter else {
                right -= 1
                continue
            }

            result.swapAt(left, right)
            
            left += 1
            right -= 1
        }
        return String(result)
        
    }
    static func test() {
        let sut = Leet0917()
        assert(sut.reverseOnlyLetters("ab-cd") == "dc-ba")
        assert(sut.reverseOnlyLetters("a-bC-dEf-ghIj") == "j-Ih-gfE-dCba")
        assert(sut.reverseOnlyLetters("Test1ng-Leet=code-Q!") == "Qedo1ct-eeLg=ntse-T!")
    }
}
//Leet0917.test()
//for i in 33...122 {
//    print(Character(UnicodeScalar(i)!))
//}




///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/two-sum-iv-input-is-a-bst
class Leet0653 {
    func find(in root: TreeNode?, and hash: inout Set<Int>, _ target: Int) -> Bool {
        guard let root = root else {
            return false
        }
        if hash.contains(target - root.val) {
            return true
        }
        hash.insert(root.val)
        return find(in: root.left, and: &hash, target) || find(in: root.right, and: &hash, target)
    }
    
    func findTarget(_ root: TreeNode?, _ k: Int) -> Bool {
        var hashSet: Set<Int> = []
        return find(in: root, and: &hashSet, k)
    }
    
    static func test() {
        let sut = Leet0653()
        assert(sut.findTarget(TreeNode(1, TreeNode(2), TreeNode(3)), 5))
        assert(sut.findTarget(TreeNode(5, TreeNode(3, TreeNode(2), TreeNode(4)), TreeNode(6, nil, TreeNode(7))), 9))
        assert(sut.findTarget(TreeNode(5, TreeNode(3, TreeNode(2), TreeNode(4)), TreeNode(6, nil, TreeNode(7))), 28) == false)
    }
    
}
//Leet0653.test()


///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/intersection-of-two-linked-lists
class Leet0160 {
    func getIntersectionNode(_ headA: ListNode?, _ headB: ListNode?) -> ListNode? {
        var a: ListNode? = headA
        var b: ListNode? = headB
        
        while a !== b {
            a = a == nil ? headB : a?.next
            b = b == nil ? headA : b?.next
        }
        return a
    }
    static func test() {
        let sut = Leet0160()
        
        let node3 = ListNode(8, ListNode(4, ListNode(5)))
        assert(sut.getIntersectionNode(ListNode(4, ListNode(1, node3)), ListNode(5, ListNode(6, ListNode(1, node3)))) === node3)
                
        let node2 = ListNode(2, ListNode(4))
        assert(sut.getIntersectionNode(ListNode(1, ListNode(9, ListNode(1, node2))), ListNode(3, node2)) === node2)
        
        assert(sut.getIntersectionNode(ListNode(2, ListNode(6, ListNode(4))), ListNode(1, ListNode(5))) == nil)
    }
}
//Leet0160.test()


///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/flipping-an-image/
class Leet0832 {
    
    func flipAndInvertImage(_ image: [[Int]]) -> [[Int]] {
        var image = image
        for i in 0..<image.count {
            image[i].reverse()
            for j in 0..<image[i].count {
                image[i][j] = image[i][j] == 0 ? 1 : 0
            }
        }
        return image
    }
    
    static func test() {
        let sut = Leet0832()
        assert(sut.flipAndInvertImage([[1,1,0],[1,0,1],[0,0,0]]) == [[1,0,0],[0,1,0],[1,1,1]])
        assert(sut.flipAndInvertImage([[1,1,0,0],[1,0,0,1],[0,1,1,1],[1,0,1,0]]) == [[1,1,0,0],[0,1,1,0],[0,0,0,1],[1,0,1,0]])
    }
}
//Leet0832.test()


///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/backspace-string-compare
class Leet0844 {
    func backspaceCompare(_ s: String, _ t: String) -> Bool {
        let s = Array(s)
        let t = Array(t)
        
        guard s.count > 0 || t.count > 0 else {
            return true
        }
        
        let backspace = Character("#")
        var i = s.count - 1
        var j = t.count - 1
        var skipS = 0
        var skipT = 0
        
        while i >= 0 || j >= 0 {

            while i >= 0 {
                if s[i] == backspace {
                    skipS += 1
                    i -= 1
                } else if skipS > 0 {
                    skipS -= 1
                    i -= 1
                } else {
                    break
                }
            }
            
            while j >= 0 {
                if t[j] == backspace {
                    skipT += 1
                    j -= 1
                } else if skipT > 0 {
                    skipT -= 1
                    j -= 1
                } else {
                    break
                }
            }
            
            if i >= 0 && j >= 0 && s[i] != t[j] {
                return false
            }
            
            if ((i >= 0) != (j >= 0)) {
                return false
            }

            i -= 1
            j -= 1
        }
        return true
    }
    
    static func test() {
        let sut = Leet0844()
        assert(sut.backspaceCompare("ab#c", "ad#c"))
        assert(sut.backspaceCompare("ab##", "c#d#"))
        assert(sut.backspaceCompare("ab###", "c#d#"))
        assert(sut.backspaceCompare("a#c", "b") == false)
        assert(sut.backspaceCompare("a#c", "#") == false)
    }
}
//Leet0844.test()






///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sort-array-by-parity-ii/
class Leet0922 {
    func sortArrayByParityII(_ nums: [Int]) -> [Int] {
        var nums = nums
        var j = 1

        // examine all even elements
        for i in stride(from: 0, to: nums.count, by: 2) {
            // an odd number is in an even element.
            if nums[i] % 2 != 0 {
                // look for the next even number in an odd index to swap
                while nums[j] % 2 != 0 {
                    j += 2
                }
                nums.swapAt(i, j)
            }
        }
        return nums
    }
    
    static func test() {
        let sut = Leet0922()
        assert(sut.sortArrayByParityII([4,2,5,7]) == [4,5,2,7])
        assert(sut.sortArrayByParityII([2,3]) == [2,3])
        assert(sut.sortArrayByParityII([6,4,2,5,7,9]) == [6,7,2,5,4,9])
        assert(sut.sortArrayByParityII([3,4]) == [4,3])
    }
}
//Leet0922.test()




///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/di-string-match
class Leet0942 {
    func diStringMatch(_ s: String) -> [Int] {
        var increase = 0
        var decrease = s.count
        var result: [Int] = []
        for c in s {
            if c == "I" {
                result.append(increase)
                increase += 1
            } else {
                result.append(decrease)
                decrease -= 1
            }
        }
        result.append(increase)
        return result
    }
    static func test() {
        let sut = Leet0942()
        assert(sut.diStringMatch("IDID") == [0,4,1,3,2])
        assert(sut.diStringMatch("III") == [0,1,2,3])
        assert(sut.diStringMatch("DDI") == [3,2,0,1])
    }
}
//Leet0942.test()




///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-permutation/
class Leet0484 {
    func findPermutation(_ s: String) -> [Int] {
        var stack = [Int]()
        var result = [Int]()
        var i = 1
        for c in s {
            if c == "I" {
                stack.append(i)
                while !stack.isEmpty {
                    result.append(stack.removeLast())
                }
            } else {
                stack.append(i)
            }
            i += 1
        }
        stack.append(i)
        while !stack.isEmpty {
            result.append(stack.removeLast())
        }
        return result
    }
    
    static func test() {
        let sut = Leet0484()
        assert(sut.findPermutation("I") == [1,2])
        assert(sut.findPermutation("II") == [1,2,3])
        assert(sut.findPermutation("DI") == [2,1,3])
        assert(sut.findPermutation("DDIIIID") == [3,2,1,4,5,6,8,7])
        assert(sut.findPermutation("IIDDIIID") == [1,2,5,4,3,6,7,9,8])
        assert(sut.findPermutation("IIIDIDDD") == [1,2,3,5,4,9,8,7,6])
        assert(sut.findPermutation("DDD") == [4,3,2,1])
    }
}
//Leet0484.test()



///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/construct-smallest-number-from-di-string/
class Leet2375 {
    func smallestNumber(_ pattern: String) -> String {
        var stack = [Int]()
        var result = ""
        var i = 1
        for c in pattern {
            if c == "I" {
                stack.append(i)
                while !stack.isEmpty {
                    result.append(String(stack.removeLast()))
                }
            } else {
                stack.append(i)
            }
            i += 1
        }
        stack.append(i)
        while !stack.isEmpty {
            result.append(String(stack.removeLast()))
        }
        return result
    }
    static func test() {
        let sut = Leet2375()
        assert(sut.smallestNumber("I") == "12")
        assert(sut.smallestNumber("II") == "123")
        assert(sut.smallestNumber("DI") == "213")
        assert(sut.smallestNumber("DDIIIID") == "32145687")
        assert(sut.smallestNumber("IIDDIIID") == "125436798")
        assert(sut.smallestNumber("IIIDIDDD") == "123549876")
        assert(sut.smallestNumber("DDD") == "4321")
    }
}
//Leet2375.test()

///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/remove-palindromic-subsequences
class Leet1332 {
    private func isPalindrome(_ s: String, left: String.Index, right: String.Index) -> Bool {
        var left = left
        var right = right
        while left < right {
            guard s[left] == s[right] else { return false }
            left = s.index(after: left)
            right = s.index(before: right)
        }
        return true
    }
    
    func removePalindromeSub(_ s: String) -> Int {
        if s.isEmpty {
            return 0
        } else if isPalindrome(s, left: s.startIndex, right: s.index(before: s.endIndex)) {
            return 1
        }
        return 2
    }
    static func test() {
        let sut = Leet1332()
        assert(sut.removePalindromeSub("ababa") == 1)
        assert(sut.removePalindromeSub("abb") == 2)
        assert(sut.removePalindromeSub("baabb") == 2)
        assert(sut.removePalindromeSub("aa") == 1)
        assert(sut.removePalindromeSub("") == 0)
    }
}
//Leet1332.test()

///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/two-sum-less-than-k/
class Leet1099 {
    func twoSumLessThanK(_ nums: [Int], _ k: Int) -> Int {
        var nums = nums.sorted()
        var left = 0
        var right = nums.count - 1
        var maxSum: Int = -1
        while left < right {
            let sum = nums[left] + nums[right]
            if sum < k {
                maxSum = max(maxSum, sum)
                left += 1
            } else {
                right -= 1
            }
        }
        return maxSum
    }
    static func test() {
        let sut = Leet1099()
        assert(sut.twoSumLessThanK([34,23,1,24,75,33,54,8], 60) == 58)
        assert(sut.twoSumLessThanK([10,20,30], 15) == -1)
        assert(sut.twoSumLessThanK([254,914,110,900,147,441,209,122,571,942,136,350,160,127,178,839,201,386,462,45,735,467,153,415,875,282,204,534,639,994,284,320,
                                    865,468,1,838,275,370,295,574,309,268,415,385,786,62,359,78,854,944], 200) == 198)
    }
}
//Leet1099.test()

///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/check-if-a-word-occurs-as-a-prefix-of-any-word-in-a-sentence
class Leet1455 {
    func isPrefixOfWord(_ sentence: String, _ searchWord: String) -> Int {
        (sentence.split(separator: " ").enumerated().first { $0.element.starts(with: searchWord) }?.offset ?? -2) + 1
    }
    static func test() {
        let sut = Leet1455()
        assert(sut.isPrefixOfWord("i love eating burger", "burg") == 4)
        assert(sut.isPrefixOfWord("this problem is an easy problem", "pro") == 2)
        assert(sut.isPrefixOfWord("i am tired", "you") == -1)
        assert(sut.isPrefixOfWord("hello world", "hello") == 1)
        assert(sut.isPrefixOfWord("love i love eating burger burger", "i") == 2)
    }
}
//Leet1455.test()


///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/counting-words-with-a-given-prefix/
class Leet2185 {
    func prefixCount(_ words: [String], _ pref: String) -> Int {
        words.filter { $0.hasPrefix(pref) }.count
    }
    static func test() {
        let sut = Leet2185()
        assert(sut.prefixCount(["apple", "banana", "orange"], "app") == 1)
        assert(sut.prefixCount(["apple", "banana", "orange"], "pine") == 0)
        assert(sut.prefixCount(["pay","attention","practice","attend"], "at") == 2)
        assert(sut.prefixCount(["leetcode","win","loops","success"], "code") == 0)
    }
}
//Leet2185.test()



///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-prefixes-of-a-given-string/
class Leet2255 {
    func countPrefixes(_ words: [String], _ s: String) -> Int {
        words.filter { s.hasPrefix($0) }.count
    }
    static func test() {
        let sut = Leet2255()
        assert(sut.countPrefixes(["a","b","c","ab","bc","abc"], "abc") == 3)
        assert(sut.countPrefixes(["a","b","c","ab","bc","abc"], "ab") == 2)
        assert(sut.countPrefixes(["a","a"], "aa") == 2)
    }
}
//Leet2255.test()


///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/check-if-string-is-a-prefix-of-array/
class Leet1961 {
    func isPrefixString(_ s: String, _ words: [String]) -> Bool {
//        (words.reduce("", +)).hasPrefix(s) && s.count > words.first?.count ?? 0
        
        var string = ""
        for word in words {
            string += word
            if s == string {
                return true
            }
        }
        return false
    }
    static func test() {
        let sut = Leet1961()
        assert(sut.isPrefixString("iloveleetcode", ["i","love","leetcode","apples"]))
        assert(!sut.isPrefixString("iloveleetcode", ["apples","i","love","leetcode"]))
        assert(!sut.isPrefixString("ccccccccc" , ["c","cc"]))
        assert(!sut.isPrefixString("a" , ["aa","aaaa","banana"]))
        assert(!sut.isPrefixString("aaa" , ["aa","aaa","fjaklfj"]))
        assert(!sut.isPrefixString("fajsldfsa" , ["faj","s","ldfs","afdfs","jfkdlsj","f"]))
    }
}
//Leet1961.test()





///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-the-distance-value-between-two-arrays
class Leet1385 {
    func findTheDistanceValue(_ arr1: [Int], _ arr2: [Int], _ d: Int) -> Int {
        let arr2 = arr2.sorted()
        return arr1.filter { arr2.firstIndex(of: $0-d...($0+d)) == -1 }.count
    }
    static func test() {
        let sut = Leet1385()
        assert(sut.findTheDistanceValue([-3,10,2,8,0,10], [-9,-1,-4,-9,-8], 9) == 2)
        assert(sut.findTheDistanceValue([4,5,8], [10,9,1,8], 2) == 2)
        assert(sut.findTheDistanceValue([1,4,2,3], [-4,-3,6,10,20,30], 3) == 2)
        assert(sut.findTheDistanceValue([2,1,100,3], [-5,-2,10,-3,7], 6) == 1)
        
    }
}
//Leet1385.test()




///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/faulty-sensor
class Leet1826 {
    func badSensor(_ sensor1: [Int], _ sensor2: [Int]) -> Int {
        var start = 0
        while start < sensor1.count && sensor1[start] == sensor2[start]  {
            start += 1
        }
        
        var bad = 0
        
        // test sensor 1 is defective
        var index2 = start + 1
        while index2 < sensor2.count && sensor1[index2-1] == sensor2[index2]  {
            index2 += 1
        }
        if index2 == sensor2.count {
            bad += 1
        }
        
        // test sensor 2 is defective
        var index1 = start + 1
        while index1 < sensor1.count && sensor2[index1-1] == sensor1[index1]  {
            index1 += 1
        }
        if index1 == sensor1.count {
            bad += 2
        }
        
        return [1,2].contains(bad) ? bad : -1
    }
    static func test() {
        let sut = Leet1826()
        assert(sut.badSensor([2,3,4,5], [2,1,3,4]) == 1)
        assert(sut.badSensor([2,2,2,2,2], [2,2,2,2,5]) == -1)
        assert(sut.badSensor([2,3,2,2,3,2], [2,3,2,3,2,7]) == 2)
        assert(sut.badSensor([8,2,2,6,3,8,7,2,5,3], [2,8,2,2,6,3,8,7,2,5]) == 1)
        assert(sut.badSensor([4,9,10,4,5,5,1,7,7,2], [4,9,10,4,5,5,7,1,7,7]) == 1)
        assert(sut.badSensor([1,2,3,2,3,2], [1,2,3,3,2,3]) == -1)
        assert(sut.badSensor([7,1,3,5,9,7,6,1,10,1], [7,1,5,9,7,6,1,10,1,1]) == 2)
        assert(sut.badSensor([1,2,3,1], [2,3,1,2]) == 2)
        assert(sut.badSensor([1], [1]) == -1)
    }
}
//Leet1826.test()





///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/reverse-prefix-of-word/
class Leet2000 {
    func reversePrefix(_ word: String, _ ch: Character) -> String {
        var wordList = Array(word)
        var index = 0
        while index < wordList.count && wordList[index] != ch {
            index += 1
        }
        guard index < wordList.count else {
            return word
        }
        wordList[...index].reverse()
        return String(wordList)
    }
    static func test() {
        let sut = Leet2000()
        assert(sut.reversePrefix("abcdefd", "d") == "dcbaefd")
        assert(sut.reversePrefix("xyxzxe", "z") == "zxyxxe")
        assert(sut.reversePrefix("abcd", "z") == "abcd")
        
    }
}
//Leet2000.test()



///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-first-palindromic-string-in-the-array
class Leet2108 {
    
    func isPalindrome(_ s: String) -> Bool {
        var left = s.startIndex
        var right = s.index(before: s.endIndex)
        while left < right {
            guard s[left].isLetterOrNumber else  {
                left = s.index(after: left)
                continue
            }
            let leftChar = s[left]
            
            guard s[right].isLetterOrNumber else {
                right = s.index(before: right)
                continue
            }
            let rightChar = s[right]
            
            if leftChar.lowercased() != rightChar.lowercased() {
                return false
            }
            left = s.index(after: left)
            right = s.index(before: right)
        }
        return true
    }
    
    func firstPalindrome(_ words: [String]) -> String {
        guard let firstPalindrome = words.first(where: isPalindrome) else {
            return ""
        }
        return firstPalindrome
    }
    
    static func test() {
        let sut = Leet2108()
        assert(sut.firstPalindrome(["abc","car","ada","racecar","cool"]) == "ada")
        assert(sut.firstPalindrome(["notapalindrome","racecar"]) == "racecar")
        assert(sut.firstPalindrome(["a"]) == "a")
        assert(sut.firstPalindrome(["def","ghi"]) == "")
    }
}
//Leet2108.test()






///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-all-k-distant-indices-in-an-array
class Leet2200 {
   func findKDistantIndices(_ nums: [Int], _ key: Int, _ k: Int) -> [Int] {
       // when key is not defined yet, we can't evaluate abs(i-j) the distance. hence the optional distances
       var distances = [Int?]()
       // look for the key to get j
       var j : Int?
       
       // pass nums increasing i
       for (i, num) in nums.enumerated() {
           if num == key {
               j = i
           }
           if let j = j {
               distances.append(abs(i-j))
           } else {
               distances.append(nil)
           }
       }
       // reset j
       j = nil
               
       // pass nums decreasing i
       // compare the minimum distance and use that
       for (i, num) in nums.enumerated().reversed() {
           if num == key {
               j = i
           }
           if let j = j {
               if let entry = distances[i] {
                   distances[i] = min(entry, abs(i-j))
               } else {
                   distances[i] = abs(i-j)
               }
           }
       }
       return distances.enumerated().filter { $0.element! <= k }.map { $0.offset }
   }
   
   static func test() {
       let s = Leet2200()
       assert(s.findKDistantIndices([3,4,9,1,3,9,5], 9, 1) == [1,2,3,4,5,6])
       assert(s.findKDistantIndices([2,2,2,2,2], 2, 2) == [0,1,2,3,4])
   }
}
//Leet2200.test()












///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/shortest-word-distance/
class Leet0243 {
    func shortestDistance(_ wordsDict: [String], _ word1: String, _ word2: String) -> Int {
        var index1: Int?
        var index2: Int?
        var distance: Int = Int.max
        for (i, word) in wordsDict.enumerated() {
            if word == word1 {
                index1 = i
            } else if word == word2 {
                index2 = i
            }
            if let index1 = index1, let index2 = index2 {
                distance = min(distance, abs(index1 - index2))
            }
        }
        return distance
    }
    static func test() {
        let sut = Leet0243()
        assert(sut.shortestDistance(["practice", "makes", "perfect", "coding", "makes"], "coding", "practice") == 3)
        assert(sut.shortestDistance(["practice", "makes", "perfect", "coding", "makes"], "makes", "coding") == 1)
    }
}
//Leet0243.test()




///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/shortest-word-distance-ii/
/// NOTE: a better solution is to store the indeces on init and calculate them. The shortest function will be faster!
/// I copied my code on problem 243 and saved
class WordDistance {
    let words: EnumeratedSequence<[String]>
    var distances = [String : Int]()
    init(_ wordsDict: [String]) {
        words = wordsDict.enumerated()
    }
    
    func shortest(_ word1: String, _ word2: String) -> Int {
        
        let key1 = word1 + word2
        let key2 = word2 + word1
        
        if let d1 = distances[key1] {
            return d1
        }
        
        if let d2 = distances[key2] {
            return d2
        }
        
        var index1: Int?
        var index2: Int?
        var distance: Int = Int.max
        for (i, word) in words {
            if word == word1 {
                index1 = i
            } else if word == word2 {
                index2 = i
            }
            if let index1 = index1, let index2 = index2 {
                distance = min(distance, abs(index1 - index2))
            }
        }
        
        distances[key1] = distance
        distances[key2] = distance
        
        return distance
    }
}

class Leet0244 {
    static func test() {
        let sut = WordDistance(["practice", "makes", "perfect", "coding", "makes"])
        assert(sut.shortest("coding", "practice") == 3)
        assert(sut.shortest("makes", "coding") == 1)
    }
}
//Leet0244.test()





///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/shortest-word-distance-iii/
class Leet0245 {
    func shortestWordDistance(_ wordsDict: [String], _ word1: String, _ word2: String) -> Int {
        var index1: Int?
        var index2: Int?
        var distance: Int = Int.max
        var word1Count = 0
    
        for (i, word) in wordsDict.enumerated() {
            
            if word1 != word2 {
                if word == word1 {
                    index1 = i
                } else if word == word2 {
                    index2 = i
                }
            } else {
                if word == word1 {
                    word1Count += 1
                    if word1Count == 1 {
                        index1 = i
                    } else if word1Count == 2 {
                        index2 = i
                    } else {
                        index1 = index2
                        index2 = i
                    }
                }
            }
            
            if let index1 = index1, let index2 = index2 {
                distance = min(distance, abs(index1 - index2))
            }
        }
        return distance
    }
    static func test() {
        let sut = Leet0245()
        assert(sut.shortestWordDistance(["practice", "makes", "perfect", "coding", "makes"], "makes", "coding") == 1)
        assert(sut.shortestWordDistance(["practice", "makes", "perfect", "coding", "makes"], "makes", "makes") == 3)
        assert(sut.shortestWordDistance(["practice", "makes", "perfect", "coding", "makes"], "coding", "practice") == 3)
        assert(sut.shortestWordDistance(["a", "b"], "a", "b") == 1)
        assert(sut.shortestWordDistance(["a", "c", "a", "a"], "a", "a") == 1)
        assert(sut.shortestWordDistance(["a", "c", "c", "a", "c", "a", "a"], "a", "a") == 1)
    }
}
//Leet0245.test()




///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/number-of-arithmetic-triplets
class Leet2367 {
    func arithmeticTriplets(_ nums: [Int], _ diff: Int) -> Int {
        nums.filter { nums.contains($0) && nums.contains($0+diff) && nums.contains($0-diff) }.count
    }
    static func test() {
        let sut = Leet2367()
        assert(sut.arithmeticTriplets([0,1,4,6,7,10], 3) == 2)
        assert(sut.arithmeticTriplets([4,5,6,7,8,9], 2) == 2)
    }
}
//Leet2367.test()




///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/largest-positive-integer-that-exists-with-its-negative
class Leet2441 {
    func findMaxK(_ nums: [Int]) -> Int {
        let nums = nums.sorted()
        var i = 0
        var j = nums.count-1
        guard nums[i] < 0 && nums[j] > 0 else { return -1 }

        while i < j {
            if nums[i] < 0 && nums[j] > 0 {
                if abs(nums[i]) < abs(nums[j]) {
                    j -= 1
                    continue
                } else if abs(nums[i]) > abs(nums[j]) {
                    i += 1
                    continue
                } else {
                    return abs(nums[i])
                }
            }
            i += 1
            j -= 1
        }
        return -1
    }
    static func test() {
        let sut = Leet2441()
        assert(sut.findMaxK([-1,2,-3,3]) == 3)
        assert(sut.findMaxK([-1,10,6,7,-7,1]) == 7)
        assert(sut.findMaxK([-10,8,6,7,-2,-3]) == -1)
        assert(sut.findMaxK([-1,-2,-3,-4]) == -1)
        assert(sut.findMaxK([1,2,3,4]) == -1)
    }
}
//Leet2441.test()




///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/number-of-distinct-averages/
class Leet2465 {
    func distinctAverages(_ nums: [Int]) -> Int {
        var set = Set<Double>()
        let nums = nums.sorted()
        var i = 0
        var j = nums.count-1
        while i < j {
            let sum = Double(nums[i] + nums[j])
            set.insert(sum/2)
            i += 1
            j -= 1
        }
        return set.count
    }
    static func test() {
        let sut = Leet2465()
        assert(sut.distinctAverages([4,1,4,0,3,5]) == 2)
        assert(sut.distinctAverages([1,100]) == 1)
        assert(sut.distinctAverages([9,5,7,8,7,9,8,2,0,7]) == 5)
        
    }
}
//Leet2465.test()




///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-enemy-forts-that-can-be-captured
class Leet2511 {
    
    func captureForts(_ forts: [Int]) -> Int {
        var maxCount = 0
        var currentCount = 0
        var start = 0 // Must be either 1 or -1
        
        for fort in forts {
            if fort == 0 {
                if start != 0 { // either 1 or -1 is found, we count
                    currentCount += 1
                }
            } else {
                if start == -fort { // we found the start and end between 1 and -1 so get maxCount
                    maxCount = max(maxCount, currentCount)
                }
                // reset
                currentCount = 0
                start = fort
            }
        }
        return maxCount
    }
    static func test() {
        let sut = Leet2511()
        assert(sut.captureForts([1,0,0,-1,0,0,0,0,1]) == 4)
        assert(sut.captureForts([0,0,1,-1]) == 0)
        assert(sut.captureForts([0,-1,-1,0,-1]) == 0)
        assert(sut.captureForts([1,0,-1,1,0,-1]) == 1)
        assert(sut.captureForts([-1,0,-1,0,1,1,1,-1,-1,-1]) == 1)
    }
}
//Leet2511.test()




///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-common-value
class Leet2540 {
    func getCommon(_ nums1: [Int], _ nums2: [Int]) -> Int {
        var index1 = 0
        var index2 = 0
        
        while index1 < nums1.count && index2 < nums2.count {
            let num1 = nums1[index1]
            let num2 = nums2[index2]
            
            if num1 == num2 {
                return num1
            } else if num1 < num2 {
                index1 += 1
            } else {
                index2 += 1
            }
        }
        return -1
    }
    static func test() {
        let sut = Leet2540()
        assert(sut.getCommon([1,2,3], [2,4]) == 2)
        assert(sut.getCommon([1,2,3], [1,4]) == 1)
        assert(sut.getCommon([1,2,3,6], [2,3,4,5]) == 2)
        
    }
}
//Leet2540.test()



///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-the-array-concatenation-value/
class Leet2562 {
    func findTheArrayConcVal(_ nums: [Int]) -> Int {
        var left = 0
        var right = nums.count - 1
        var result = 0
        
        if nums.count == 1 {
            return nums[0]
        }
        
        while left <= right {
            
            guard left < right else {
                result += nums[left]
                break
            }
            
            // concat left and right
            var rightValue = nums[right]
            var current = rightValue
            var multiplier = 1
            while rightValue > 0 {
                multiplier *= 10
                rightValue /= 10
            }
            let leftValue = nums[left] * multiplier
            current += leftValue
            
            result += current
            
            left += 1
            right -= 1
        }
        
        return result
    }
    
    static func test() {
        let sut = Leet2562()
        assert(sut.findTheArrayConcVal([7,52,2,4]) == 596)
        assert(sut.findTheArrayConcVal([5,14,13,8,12]) == 673)
    }
}
//Leet2562.test()




///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/merge-two-2d-arrays-by-summing-values
class Leet2570 {
    func mergeArrays(_ nums1: [[Int]], _ nums2: [[Int]]) -> [[Int]] {
        var result = [[Int]]()
        var id = 1
        var index1 = 0
        var index2 = 0
        while index1 < nums1.count && index2 < nums2.count {
            var row1 = nums1[index1]
            var row2 = nums2[index2]
            if let id1 = row1.first, let val1 = row1.last, let id2 = row2.first, let val2 = row2.last {
                var mergedVal = 0
                if id == id1 {
                    mergedVal += val1
                    index1 += 1
                }
                if id == id2 {
                    mergedVal += val2
                    index2 += 1
                }
                if mergedVal > 0 {
                    result.append([id, mergedVal])
                }
                id += 1
            }
        }
        
        while index1 < nums1.count {
            var row1 = nums1[index1]
            if let id1 = row1.first, let val1 = row1.last {
                var mergedVal = 0
                if id == id1 {
                    mergedVal += val1
                    index1 += 1
                }
                if mergedVal > 0 {
                    result.append([id, mergedVal])
                }
                id += 1
            }
        }
        
        while index2 < nums2.count {
            var row2 = nums2[index2]
            if let id2 = row2.first, let val2 = row2.last {
                var mergedVal = 0
                if id == id2 {
                    mergedVal += val2
                    index2 += 1
                }
                if mergedVal > 0 {
                    result.append([id, mergedVal])
                }
                id += 1
            }
        }
        
        return result
    }
    static func test() {
        let s = Leet2570()
        assert(s.mergeArrays([[1,2],[2,3],[4,5]], [[1,4],[3,2],[4,1]]) ==  [[1,6],[2,3],[3,2],[4,6]])
        assert(s.mergeArrays([[2,4],[3,6],[5,5]], [[1,3],[4,3]]) ==  [[1,3],[2,4],[3,6],[4,3],[5,5]])
        assert(s.mergeArrays([[148,597],[165,623],[306,359],[349,566],[403,646],[420,381],[566,543],[730,209],[757,875],[788,208],[932,695]], [[74,669],[87,399],[89,165],[99,749],[122,401],[138,16],[144,714],[148,206],[177,948],[211,653],[285,775],[309,289],[349,396],[386,831],[403,318],[405,119],[420,153],[468,433],[504,101],[566,128],[603,688],[618,628],[622,586],[641,46],[653,922],[672,772],[691,823],[693,900],[756,878],[757,952],[770,795],[806,118],[813,88],[919,501],[935,253],[982,385]]) ==  [[74,669],[87,399],[89,165],[99,749],[122,401],[138,16],[144,714],[148,803],[165,623],[177,948],[211,653],[285,775],[306,359],[309,289],[349,962],[386,831],[403,964],[405,119],[420,534],[468,433],[504,101],[566,671],[603,688],[618,628],[622,586],[641,46],[653,922],[672,772],[691,823],[693,900],[730,209],[756,878],[757,1827],[770,795],[788,208],[806,118],[813,88],[919,501],[932,695],[935,253],[982,385]])
    }
}
//Leet2570.test()


///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/merge-similar-items/
class Leet2363 {
    func mergeSimilarItems(_ items1: [[Int]], _ items2: [[Int]]) -> [[Int]] {
        let items1 = items1.sorted { $0.first! < $1.first! }
        let items2 = items2.sorted { $0.first! < $1.first! }
        var result = [[Int]]()
        var id = 1
        var index1 = 0
        var index2 = 0
        while index1 < items1.count && index2 < items2.count {
            var row1 = items1[index1]
            var row2 = items2[index2]
            if let id1 = row1.first, let val1 = row1.last, let id2 = row2.first, let val2 = row2.last {
                var mergedVal = 0
                if id == id1 {
                    mergedVal += val1
                    index1 += 1
                }
                if id == id2 {
                    mergedVal += val2
                    index2 += 1
                }
                if mergedVal > 0 {
                    result.append([id, mergedVal])
                }
                id += 1
            }
        }
        
        while index1 < items1.count {
            var row1 = items1[index1]
            if let id1 = row1.first, let val1 = row1.last {
                var mergedVal = 0
                if id == id1 {
                    mergedVal += val1
                    index1 += 1
                }
                if mergedVal > 0 {
                    result.append([id, mergedVal])
                }
                id += 1
            }
        }
        
        while index2 < items2.count {
            var row2 = items2[index2]
            if let id2 = row2.first, let val2 = row2.last {
                var mergedVal = 0
                if id == id2 {
                    mergedVal += val2
                    index2 += 1
                }
                if mergedVal > 0 {
                    result.append([id, mergedVal])
                }
                id += 1
            }
        }
        
        return result
    }
    static func test() {
        let sut = Leet2363()
        assert(sut.mergeSimilarItems([[1,1],[4,5],[3,8]], [[3,1],[1,5]]) == [[1,6],[3,9],[4,5]])
        assert(sut.mergeSimilarItems([[1,1],[3,2],[2,3]], [[2,1],[3,2],[1,3]]) == [[1,4],[2,4],[3,4]])
        assert(sut.mergeSimilarItems([[1,3],[2,2]], [[7,1],[2,2],[1,4]]) == [[1,7],[2,4],[7,1]])
    }
}
//Leet2363.test()



///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/lexicographically-smallest-palindrome/
class Leet2697 {
    func makeSmallestPalindrome(_ s: String) -> String {
        var list = Array(s)
        var left = 0
        var right = list.count - 1
        
        while left < right {
            if let leftAsciiValue = list[left].asciiValue, let rightAsciiValue = list[right].asciiValue {
                if leftAsciiValue < rightAsciiValue {
                    list[right] = list[left]
                } else if leftAsciiValue > rightAsciiValue {
                    list[left] = list[right]
                }
            }
            left += 1
            right -= 1
        }
        return String(list)
    }
    static func test() {
        let sut = Leet2697()
        assert(sut.makeSmallestPalindrome("egcfe") == "efcfe")
        assert(sut.makeSmallestPalindrome("abcd") == "abba")
        assert(sut.makeSmallestPalindrome("seven") == "neven")
        assert(sut.makeSmallestPalindrome("atie") == "aiia")
        
    }
}
//Leet2697.test()




///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-pairs-whose-sum-is-less-than-target
class Leet2864 {
    func countPairs(_ nums: [Int], _ target: Int) -> Int {
        let nums = nums.sorted()
        var count = 0
        var left = 0
        var right = nums.count - 1
        
        while left < right {
            let sum = nums[left] + nums[right]
            if sum < target {
#warning("how is this counting the pairs?")
                count += (right - left)
                left += 1
            } else {
                right -= 1
            }
        }
        return count
    }
    static func test() {
        let sut = Leet2864()
        assert(sut.countPairs([-1, 1, 2, 3, 1], 2) == 3)
        assert(sut.countPairs([-6, 2, 5, -2, -7, -1, 3], -2) == 10)
    }
}
//Leet2864.test()



///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-indices-with-index-and-value-difference-i
class Leet2903 {
    func findIndices(_ nums: [Int], _ indexDifference: Int, _ valueDifference: Int) -> [Int] {
        var i = 0
        var j = 0
        while j < nums.count {
            if abs(i-j) >= indexDifference && abs(nums[i]-nums[j]) >= valueDifference {
                return [i, j]
            }
            j += 1
            if j == nums.count && i < nums.count - 1 {
                i += 1
                j = i
            }
        }
        return [-1, -1]
    }
    static func test() {
        let sut = Leet2903()
        assert(sut.findIndices([5,1,4,1], 2, 4) == [0, 3])
        assert(sut.findIndices([2,1], 0, 0) == [0, 0])
        assert(sut.findIndices([1,2,3], 2, 4) == [-1, -1])
    }
}
//Leet2903.test()


///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-indices-with-index-and-value-difference-ii
#warning("Sliding Windows really is the key here.")
class Leet2905 {
    func findIndices(_ nums: [Int], _ indexDifference: Int, _ valueDifference: Int) -> [Int] {
        var minJ = 0
        var maxJ = 0
        var i = indexDifference
        var j = 0
        while i < nums.count {
            minJ = nums[minJ] < nums[j] ? minJ : j
            maxJ = nums[maxJ] > nums[j] ? maxJ : j
            if abs(nums[i] - nums[minJ]) >= valueDifference {
                return [minJ, i]
            }
            if abs(nums[i] - nums[maxJ]) >= valueDifference {
                return [maxJ, i]
            }
            i += 1
            j += 1
        }
        return [-1, -1]
    }
    static func test() {
        let sut = Leet2905()
        assert(sut.findIndices([5,1,4,1], 2, 4) == [0, 3])
        assert(sut.findIndices([2,1], 0, 0) == [0, 0])
        assert(sut.findIndices([1,2,3], 2, 4) == [-1, -1])
    }
}
//Leet2905.test()



///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-average-of-smallest-and-largest-elements/
class Leet3194 {
    func minimumAverage(_ nums: [Int]) -> Double {
        let nums = nums.sorted()
        var left = 0
        var right = nums.count - 1
        var smallestAve = Double(Int.max)
        while left < right {
            let ave = Double(nums[left] + nums[right]) / 2
            smallestAve = min(ave, smallestAve)
            left += 1
            right -= 1
        }
        return smallestAve
    }
    static func test() {
        let sut = Leet3194()
        assert(sut.minimumAverage([7,8,3,4,15,13,4,1]) == 5.5)
        assert(sut.minimumAverage([1,9,8,3,10,5]) == 5.5)
        assert(sut.minimumAverage([1,2,3,7,8,9]) == 5.0)
    }
}
//Leet3194.test()




///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/subarray-product-less-than-k/
class Leet0713 {
    func numSubarrayProductLessThanK(_ nums: [Int], _ k: Int) -> Int {
        guard k > 1 else {
            return 0
        }
        var count = 0
        var product = 1
        var left = 0
        for right in 0..<nums.count {
            product *= nums[right]
            while product >= k {
                product /= nums[left]
                left += 1
            }
            count += right - left + 1
        }
        return count
    }
    static func test() {
        let sut = Leet0713()
        assert(sut.numSubarrayProductLessThanK([10, 5, 2, 6], 100) == 8)
        assert(sut.numSubarrayProductLessThanK([1, 2, 3], 0) == 0)
    }
}
//Leet0713.test()





///The length of the subarray gives you the possible combinations that will score less than `k` in that subarray. Similar to the problem 713.
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-subarrays-with-score-less-than-k/
class Leet2310 {
    func countSubarrays(_ nums: [Int], _ k: Int) -> Int {
        var sum = 0
        var left = 0
        var count = 0
        var score = 0
        for right in 0..<nums.count {
            sum += nums[right]
            score = sum * (right - left + 1)
            while score >= k {
                sum -= nums[left]
                left += 1
                score = sum * (right - left + 1)
            }
            count += right - left + 1
        }
        return count
    }
    static func test() {
        let sut = Leet2310()
        assert(sut.countSubarrays([2,1,4,3,5], 10) == 6)
        assert(sut.countSubarrays([1,1,1], 5) == 5)
    }
}
//Leet2310.test()




///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/contains-duplicate-ii/
/// Sliding Window but using a set!
class Leet0219 {
    func containsNearbyDuplicate(_ nums: [Int], _ k: Int) -> Bool {
        var seen: Set<Int> = []
        for i in 0..<nums.count {
            if seen.contains(nums[i]) {
                return true
            }
            seen.insert(nums[i])
            if seen.count > k {
                seen.remove(nums[i - k])
            }
        }
        return false
    }
    static func test() {
        let sut = Leet0219()
        assert(sut.containsNearbyDuplicate([1,2,3,1], 3))
        assert(sut.containsNearbyDuplicate([1,0,1,1], 1))
        assert(sut.containsNearbyDuplicate([1,2,3,1,2,3], 2) == false)
        assert(sut.containsNearbyDuplicate([0,1,2,3,2,5], 3))
        assert(sut.containsNearbyDuplicate([1,2,1,4,5], 4))
        assert(sut.containsNearbyDuplicate([0,1,2,3,2,5], 3))
        assert(sut.containsNearbyDuplicate([1,1,3,4,5,6,7,1], 3))
        assert(sut.containsNearbyDuplicate([0,1,2,3,4,0,0,7,8,9,10,11,12,0], 1))
        assert(sut.containsNearbyDuplicate([1,2,3,4,6,2,8,2], 2))
    }
}
//Leet0219.test()






///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/longest-harmonious-subsequence/
///NOTE: dictionary is a more intuitive approach?
class Leet0594 {
    func findLHS(_ nums: [Int]) -> Int {
        let nums = nums.sorted()
        // use sliding window
        var maxLength = 0
        var left = 0
        for right in 1..<nums.count {
            while (nums[right] - nums[left]) > 1 {
                left += 1
            }
            if nums[right] - nums[left] == 1 {
                maxLength = max(maxLength, right - left + 1)
            }
        }
        return maxLength
    }
    static func test() {
        let sut = Leet0594()
        assert(sut.findLHS([1,3,2,2,5,2,3,7]) == 5)
        assert(sut.findLHS([1,2,3,4]) == 2)
        assert(sut.findLHS([1,1,1,1]) == 0)
        assert(sut.findLHS([1,3,2,2,5,2,3,7,3,5,3,2,2,3]) == 10)
    }
}
//Leet0594.test()




///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-average-subarray-i
class Leet0643 {
    func findMaxAverage(_ nums: [Int], _ k: Int) -> Double {
        var sum = nums[0..<k].reduce(0, +), maxAverage = Double(sum)/Double(k)
        for i in k..<nums.count {
            sum += nums[i] - nums[i-k]
            maxAverage = max(maxAverage, Double(sum)/Double(k))
        }
        return maxAverage
    }
    static func test() {
        let sut = Leet0643()
        assert(sut.findMaxAverage([1,12,-5,-6,50,3], 4) == 12.75)
        assert(sut.findMaxAverage([5], 1) == 5.0)
    }
}
//Leet0643.test()






///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/diet-plan-performance/
class Leet1176 {
    func dietPlanPerformance(_ calories: [Int], _ k: Int, _ lower: Int, _ upper: Int) -> Int {
        var points = 0
        var windowSum = 0
        for i in 0..<k {
            windowSum += calories[i]
        }
        points += windowSum < lower ? -1 : windowSum > upper ? 1 : 0
        for i in k..<calories.count {
            windowSum += calories[i] - calories[i - k]
            points += windowSum < lower ? -1 : windowSum > upper ? 1 : 0
        }
        return points
    }
    static func test() {
        let sut = Leet1176()
        assert(sut.dietPlanPerformance([1,2,3,4,5], 1, 3, 3) == 0)
        assert(sut.dietPlanPerformance([3,2], 2, 0, 1) == 1)
        assert(sut.dietPlanPerformance([6,5,0,0], 2, 1, 5) == 0)
        assert(sut.dietPlanPerformance([6,13,8,7,10,1,12,11], 6, 5, 37) == 3)
    }
}
//Leet1176.test()



///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/defuse-the-bomb/
class Leet1652 {
    func decrypt(_ code: [Int], _ k: Int) -> [Int] {
        let isReversed = k < 0
        let code: [Int] = isReversed ? code.reversed() : code
        let k = abs(k)
        let count = code.count
        var result: [Int] = []
        var windowSum = 0
        for i in 0..<k {
            windowSum += code[i % count]
        }
        for i in k..<count+k {
            windowSum += code[i % count] - code[(i - k) % count]
            result.append(windowSum)
        }
        return isReversed ? result.reversed() : result
    }
    static func test() {
        let sut = Leet1652()
        assert(sut.decrypt([5,7,1,4], 2) == [8,5,9,12])
        assert(sut.decrypt([5,7,1,4], -2) == [5,9,12,8])
        assert(sut.decrypt([5,7,1,4], 3) == [12,10,16,13])
        assert(sut.decrypt([1,2,3,4], 0) == [0,0,0,0])
        assert(sut.decrypt([2,4,9,3], -2) == [12,5,6,13])
        assert(sut.decrypt([4,10,87,71,36,14,33,91,12,97,41,90,12,77,20,3,15,12,46,40,23,88,21], 7) == [342, 344, 354, 324, 378, 376, 420, 349, 340, 258, 229, 185, 213, 159, 227, 245, 234, 232, 273, 304, 317, 243, 255])
                    
    }
}
//Leet1652.test()



//private extension Set<Character> {
//    
//    func isCharacterBad(_ char: Character) -> Bool {
//        char.isLowercase && !contains(char.uppercased().first!) ||
//        char.isUppercase && !contains(char.lowercased().first!)
//    }
//    
//    var badCharacters: [Character] {
//        Array(self).filter { isCharacterBad($0)}
//    }
//    
//    var isNice: Bool {
//        return isEmpty || badCharacters.isEmpty
//    }
//}
//
//private extension Character {
//    var oppositeCase: Character {
//        isLowercase ? uppercased().first! : lowercased().first!
//    }
//}
//
//private extension Dictionary<Character, Int> {
//    func isCharacterGood(_ char: Character) -> Bool {
//        char.isLowercase && self[char.uppercased().first!] != nil ||
//        char.isUppercase && self[char.lowercased().first!] != nil
//    }
//    func isCharacterBad(_ char: Character) -> Bool { isCharacterGood(char) == false }
//    
//    var badCharacters: [Character] { Array(self.keys).filter { isCharacterBad($0)} }
//    
//    var isNice: Bool { isEmpty || badCharacters.isEmpty }
//}


// BRUTE FORCE
//private extension Array<Character> {
//    var isNice: Bool {
//        let set = Set(self)
//        for char in set {
//            if char.isLowercase && !contains(char.uppercased().first!) ||
//                char.isUppercase && !contains(char.lowercased().first!) {
//                return false
//            }
//        }
//        return true
//    }
//}


private extension Character {
    var oppositeCase: Character { isLowercase ? uppercased().first! : lowercased().first! }
}

private extension Dictionary<Character, Int> {
    var isNice: Bool { self.keys.reduce(into: true) { (result, char) in result = self[char.oppositeCase] == nil ? false : result } }
    func isGood(_ char: Character) -> Bool { self[char.oppositeCase] != nil }
}


///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/longest-nice-substring
class Leet1763 {

    func longestNiceSubstring(_ s: [Character], _ left: Int, _ right: Int) -> [Character] {
        // base case
        guard s.count > 1 else {
            return []
        }

        var countMap: [Character: Int] = [:]
        for char in s[left..<right] {
            countMap[char, default: 0] += 1
        }
        guard countMap.isNice == false else {
            return Array(s[left..<right])
        }
        
        // divide and look for mid
        for mid in left..<right {
            let charMid = s[mid]
            guard !countMap.isGood(charMid) else {
                continue // continue when good
            }
            
            // found bad character. look for more bad chars.
            var midNext = mid + 1
            while midNext < right && !countMap.isGood(s[midNext]) {
                midNext += 1
            }
            
            let leftSubstring = longestNiceSubstring(s, left, mid)
            let rightSubstring = longestNiceSubstring(s, midNext, right)
            if leftSubstring.count < rightSubstring.count {
                return rightSubstring
            } else {
                return leftSubstring
            }
        }
        return []
    }
    
    func longestNiceSubstring(_ s: String) -> String {
        String(longestNiceSubstring(Array(s), 0, s.count))
    }
    
//    func xxx_Brute_Force_longestNiceSubstring(_ s: String) -> String {
//        var maxCharacters: [Character] = []
//        let characters = Array(s)
//        for left in 0..<characters.count-1 {
//            for right in left + 1..<characters.count {
//                let windowCharacters = Array(characters[left...right])
//                if windowCharacters.isNice {
//                    maxCharacters = maxCharacters.count < windowCharacters.count ? windowCharacters : maxCharacters
//                }
//            }
//        }
//        return String(maxCharacters)
//    }
    
    static func test() {
        let sut = Leet1763()
        assert(sut.longestNiceSubstring("cXlLwEZhtDPSiToVeWssVzzRMCrxmqSsZeIkzcWvzePMxsrpDMO") == "lL") // Ss
        assert(sut.longestNiceSubstring("qlERNCNVvWLOrrkAaDcXnlaDQxNEneRXQMKnrNN") == "NEne") // Vv
        assert(sut.longestNiceSubstring("xLeElzxgHzcWslEdgMGwEOZCXwwDMwcEhgJHLL") == "LeEl")
        assert(sut.longestNiceSubstring("YZ") == "")
        assert(sut.longestNiceSubstring("YazzzaAay") == "aAa")
        assert(sut.longestNiceSubstring("Bb") == "Bb")
        assert(sut.longestNiceSubstring("c") == "")
        assert(sut.longestNiceSubstring("jcJ") == "")
        assert(sut.longestNiceSubstring("BebjJE") == "BebjJE")
        assert(sut.longestNiceSubstring("aAa") == "aAa")
        assert(sut.longestNiceSubstring("AaA") == "AaA")
        assert(sut.longestNiceSubstring("aAbBcC") == "aAbBcC")
        assert(sut.longestNiceSubstring("aAay") == "aAa")
        assert(sut.longestNiceSubstring("HkhBubUYy") == "BubUYy")
        assert(sut.longestNiceSubstring("BubUYyHkh") == "BubUYy")
        assert(sut.longestNiceSubstring("HkhBubUYyhkh") == "BubUYy")
        assert(sut.longestNiceSubstring("HkhBubUYyHkh") == "hBubUYyH")
    }
}
//Leet1763.test()







///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/
///https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/editorial/comments/2550069
///
class Leet0395 {
    private func longestSubstring(_ s: [Character], _ start: Int, _ end: Int, _ k: Int) -> Int {
        guard end >= k else {
            return 0
        }
        var countMap: [Character: Int] = [:]
        for i in start..<end {
            countMap[s[i], default: 0] += 1
        }
        for mid in start..<end {
            let charMid = s[mid]
            guard let countCharMid = countMap[charMid], countCharMid < k else {
                continue
            }
            // found invalid where count < k
            var midNext = mid + 1
            while midNext < end && countMap[s[midNext]]! < k {
                midNext += 1
            }
            return max(longestSubstring(s, start, mid, k),
                       longestSubstring(s, midNext, end, k))
        }
        return end - start
    }
    func longestSubstring(_ s: String, _ k: Int) -> Int {
        longestSubstring(Array(s), 0, s.count, k)
    }
    static func test() {
        let sut = Leet0395()
        assert(sut.longestSubstring("aaabb", 3) == 3)
        assert(sut.longestSubstring("ababbc", 2) == 5)
        assert(sut.longestSubstring("bbaaacbd", 3) == 3)
        assert(sut.longestSubstring("ababcabaaadc", 2) == 4)
    }
}
//Leet0395.test()




///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sort-an-array/
class Leet0912 {
    func sortArray(_ nums: [Int]) -> [Int] {
        guard nums.count > 1 else { return nums }
        let mid = nums.count / 2
        let left = Array(nums[..<mid])
        let right = Array(nums[mid...])
        let leftSorted = sortArray(left)
        let rightSorted = sortArray(right)
        var merged: [Int] = []
        var i = 0, j = 0
        while i < leftSorted.count && j < rightSorted.count {
            if leftSorted[i] < rightSorted[j] {
                merged.append(leftSorted[i])
                i += 1
            } else {
                merged.append(rightSorted[j])
                j += 1
            }
        }
        if i < leftSorted.count {
            merged.append(contentsOf: leftSorted[i...])
        }
        if j < rightSorted.count {
            merged.append(contentsOf: rightSorted[j...])
        }
        return merged
    }
    
    static func test() {
        let s = Leet0912()
        assert(s.sortArray([5,2,3,1]) == [1,2,3,5])
        assert(s.sortArray([-1,5,3,4,2]) == [-1,2,3,4,5])
    }
}
//Leet0912.test()









///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/search-a-2d-matrix-ii/
class Leet0240 {
      
//
// DIVIDE & CONQUER ALGORITHM
//
//    var target = 0
//    var matrix: [[Int]] = []
//    
//    func search(_ left: Int, _ top: Int, _ right: Int, _ bottom: Int) -> Bool {
//        // sub matrix is empty
//        guard left <= right, top <= bottom else { return false }
//        
//        // in between corners
//        guard target >= matrix[top][left], target <= matrix[bottom][right] else { return false }
//        
//        let mid = (left + right) / 2 // mid is the column
//        var row = top
//        
//        // locate `row` such that matrix[row-1][mid] < target < matrix[row][mid]
//        while row <= bottom && matrix[row][mid] <= target {
//            if matrix[row][mid] == target {
//                return true
//            }
//            row += 1
//        }
//        return search(left, row, mid - 1, bottom) || search(mid + 1, top, right, row - 1)
//    }
//    
//    func searchMatrix(_ matrix: [[Int]], _ target: Int) -> Bool {
//
//        self.matrix = matrix
//        self.target = target
//        
//        guard matrix.count > 0 else { return false }
//        
//        return search(0, 0, matrix[0].count - 1, matrix.count - 1)
//
//    }
    
    
    
    //
    // SPACE REDUCTIOn
    //
    func searchMatrix(_ matrix: [[Int]], _ target: Int) -> Bool {
        var row = matrix.count - 1
        var col = 0
        while row >= 0 && col < matrix[0].count {
            if matrix[row][col] > target {
                row -= 1
            } else if matrix[row][col] < target {
                col += 1
            } else {
                return true
            }
        }
        return false
    }
    
    static func test() {
        let sut = Leet0240()
        assert(sut.searchMatrix([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], 5))
        assert(sut.searchMatrix([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], 20) == false)
    }
}
//Leet0240.test()







///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/search-a-2d-matrix/
class Leet0074 {
    
    // Binary Search
    func searchMatrix(_ matrix: [[Int]], _ target: Int) -> Bool {
        let m = matrix.count
        guard m > 0 else { return false }
        let n = matrix[0].count
        
        var low = 0
        var high = m * n - 1
        
        while low <= high {
            let mid = (low + high) / 2
            let pivot = matrix[mid / n][mid % n]
            if pivot < target {
                low = mid + 1
            } else if pivot > target {
                high = mid - 1
            } else {
                return true
            }
        }
        return false
    }
    static func test() {
        let sut = Leet0074()
        assert(sut.searchMatrix([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], 5))
        assert(sut.searchMatrix([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], 20) == false)
        assert(sut.searchMatrix([[]], 1) == false)
        assert(sut.searchMatrix([[1,3,5,7],[10,11,16,20],[23,30,34,60]], 3))
        assert(sut.searchMatrix([[1,3,5,7],[10,11,16,20],[23,30,34,60]], 13) == false)
    }
}
//Leet0074.test()








///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/substrings-of-size-three-with-distinct-characters
class Leet1876 {
    private func countGoodSubstrings(_ s: [Character]) -> Int {
        guard s.count >= 3 else { return 0 }
        var result = 0
        for i in 2..<s.count {
            let three = Set([s[i-2], s[i-1], s[i]])
            result += three.count == 3 ? 1 : 0
        }
        return result
    }
    func countGoodSubstrings(_ s: String) -> Int {
        countGoodSubstrings(Array(s))
    }
    static func test() {
        let sut = Leet1876()
        assert(sut.countGoodSubstrings("") == 0)
        assert(sut.countGoodSubstrings("xyzzaz") == 1)
        assert(sut.countGoodSubstrings("aababcabc") == 4)
        assert(sut.countGoodSubstrings("owuxoelszb") == 8)
    }
}
//Leet1876.test()





///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-difference-between-highest-and-lowest-of-k-scores
class Leet1984 {
    func minimumDifference(_ nums: [Int], _ k: Int) -> Int {
        let nums = nums.sorted()
        var left = 0
        var right = k-1
        var minDiff = Int.max
        while right < nums.count {
            minDiff = min(minDiff, abs(nums[right] - nums[left]))
            left += 1
            right += 1
        }
        return minDiff
    }
    static func test() {
        let sut = Leet1984()
        assert(sut.minimumDifference([90], 1) == 0)
        assert(sut.minimumDifference([1,2,4], 2) == 1)
        assert(sut.minimumDifference([9,4,1,7], 2) == 2)
        assert(sut.minimumDifference([87063, 61094, 44530, 21297, 95857, 93551, 9918], 6) == 74560)
        assert(sut.minimumDifference([1,4,9,11,12,13,15], 3) == 2)
    }
}
//Leet1984.test()








///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-the-k-beauty-of-a-number
class Leet2269 {
    func divisorSubstrings(_ num: Int, _ k: Int) -> Int {
        let list = Array(String(num))
        guard list.count >= k else { return 0 }
        var left = 0
        var right = k
        var count = 0
        while right <= list.count {
            defer {
                left += 1
                right += 1
            }
            guard let rightNum = Int(String(list[left..<right])), num.isMultiple(of: rightNum) else {
                continue
            }
            count += 1
        }
        return count
    }
    static func test() {
        let sut = Leet2269()
        assert(sut.divisorSubstrings(430043, 2) == 2)
        assert(sut.divisorSubstrings(240, 2) == 2)
    }
}
//Leet2269.test()








///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-recolors-to-get-k-consecutive-black-blocks
class Leet2379 {
    func minimumRecolors(_ blocks: String, _ k: Int) -> Int {
        let list = Array(blocks)
        var windowCount = list[0..<k].count(where: { $0 == "W" })
        var minCount = windowCount
        for right in k..<list.count {
            windowCount += (list[right] == "W" ? 1 : 0) + (list[right-k] == "W" ? -1 : 0)
            minCount = min(minCount, windowCount)
        }
        return minCount
    }
    static func test() {
        let sut = Leet2379()
        assert(sut.minimumRecolors("WBBWWBBWBW", 7) == 3)
        assert(sut.minimumRecolors("WBWBBBW", 2) == 0)
    }
}
//Leet2379.test()







///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/longest-even-odd-subarray-with-threshold/
class Leet2760 {
    func longestAlternatingSubarray(_ nums: [Int], _ threshold: Int) -> Int {
        var maxLength = 0
        var left = 0
        var right = left
        var nextModulusResult = 1
 
        // exhaust all windows and find the max
        while right < nums.count {
            
            // look for the next left window
            left = right
            while left < nums.count {
                if nums[left] % 2 == 0 && nums[left] <= threshold {
                    break
                }
                left += 1
            }
            guard left < nums.count, nums[left] % 2 == 0 && nums[left] <= threshold else {
                return maxLength
            }
            
            // look for the next right window, ie the end
            right = left + 1
            nextModulusResult = 1
            while right < nums.count, nums[right] % 2 == nextModulusResult && nums[right] <= threshold {
                nextModulusResult = (nextModulusResult == 1) ? 0 : 1
                right += 1
            }
            maxLength = max(maxLength, right - left)
        }

        return max(maxLength, right - left)
    }
    static func test() {
        let sut = Leet2760()
        
        assert(sut.longestAlternatingSubarray([8,4], 6) == 1)
        assert(sut.longestAlternatingSubarray([1,3], 16) == 0)
        assert(sut.longestAlternatingSubarray([3,2,5,4], 5) == 3)
        assert(sut.longestAlternatingSubarray([3,2,5,4,2,5,4,3], 5) == 4)
        assert(sut.longestAlternatingSubarray([1,2], 2) == 1)
        assert(sut.longestAlternatingSubarray([2,3,4,5], 4) == 3)
        assert(sut.longestAlternatingSubarray([4], 4) == 1)
        assert(sut.longestAlternatingSubarray([2,3,3,10], 10) == 2)
        assert(sut.longestAlternatingSubarray([5,6,7,8], 4) == 0)
        assert(sut.longestAlternatingSubarray([4,10,3], 4) == 1)
        assert(sut.longestAlternatingSubarray([1,2,8], 2) == 1)
        assert(sut.longestAlternatingSubarray([2,8], 2) == 1)
        assert(sut.longestAlternatingSubarray([4,10,3,8,4,5,4,1], 16) == 4)

    }
}
//Leet2760.test()







///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-strong-pair-xor-i/
class Leet2932 {
    func maximumStrongPairXor(_ nums: [Int]) -> Int {
        var maximum = 0
        for x in nums {
            for y in nums {
                if abs(x-y) <= min(x,y) {
                    maximum = max(maximum, x^y)
                }
            }
        }
        return maximum
    }
    static func test() {
        let solution = Leet2932()
        assert(solution.maximumStrongPairXor([1,2,3,4,5]) == 7)
        assert(solution.maximumStrongPairXor([10,100]) == 0)
        assert(solution.maximumStrongPairXor([5,6,25,30]) == 7)
    }
}
//Leet2932.test()












///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/shortest-subarray-with-or-at-least-k-i
class Leet3095 {
    func minimumSubarrayLength(_ nums: [Int], _ k: Int) -> Int {
        var smallest = Int.max
        for i in 0..<nums.count {
            for j in i..<nums.count {
                let sum = nums[i...j].reduce(0, |)
                if sum >= k {
                    smallest = min(smallest, j - i + 1)
                    break
                }
            }
        }
        return smallest == Int.max ? -1 : smallest
    }
    static func test() {
        let solution = Leet3095()
        assert(solution.minimumSubarrayLength([1,2,3], 2) == 1)
        assert(solution.minimumSubarrayLength([2,1,8], 10) == 3)
        assert(solution.minimumSubarrayLength([1,2], 0) == 1)
    }
}
//Leet3095.test()



///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/alternating-groups-i/
class Leet3206 {
    func numberOfAlternatingGroups(_ colors: [Int]) -> Int {
        var result = 0
        for i in 0..<(colors.count) {
            if colors[i] != colors[(i+1) % colors.count] && colors[(i+1) % colors.count] != colors[(i+2) % colors.count] {
                result += 1
            }
        }
        return result
    }
    static func test() {
        let solution = Leet3206()
        assert(solution.numberOfAlternatingGroups([1,1,1]) == 0)
        assert(solution.numberOfAlternatingGroups([0,1,0,0,1]) == 3)
    }
}
//Leet3206.test()











///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-substrings-that-satisfy-k-constraint-i/
class Leet3258 {
//    func countKConstraintSubstrings(_ s: String, _ k: Int) -> Int {
//        let list = Array(s)
//        var (result, count1s, count0s) = (0, 0, 0)
//        for i in 0..<list.count {
//            (count0s, count1s) = (0, 0)
//            for j in i..<list.count {
//                if list[j] == "1" {
//                    count1s += 1
//                } else {
//                    count0s += 1
//                }
//                
//                if count0s <= k || count1s <= k {
//                    result += 1
//                }
//            }
//        }
//        return result
//    }

    func countKConstraintSubstrings(_ s: String, _ k: Int) -> Int {
        var (leftIndex, rightIndex) = (s.startIndex, s.startIndex)
        var result = 0
        while leftIndex < s.endIndex {
            defer { leftIndex = s.index(after: leftIndex) }
            var (count0s, count1s) = (0, 0)
            rightIndex = leftIndex
            while rightIndex < s.endIndex {
                defer { rightIndex = s.index(after: rightIndex) }
                if s[rightIndex] == "1" {
                    count1s += 1
                } else {
                    count0s += 1
                }
                if count0s <= k || count1s <= k {
                    result += 1
                } else { break }
               
            }
        }
        return result
    }
    
    static func test() {
        let sut = Leet3258()
        assert(sut.countKConstraintSubstrings("10101", 1) == 12)
        assert(sut.countKConstraintSubstrings("1010101", 2) == 25)
        assert(sut.countKConstraintSubstrings("11111", 1) == 15)
    }
}
//Leet3258.test()




/*
///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/shortest-subarray-with-or-at-least-k-ii/
class Leet3097 {
    func minimumSubarrayLength(_ nums: [Int], _ k: Int) -> Int {
        0
    }
    static func test() {
        let sut = Leet3097()
        
    }
}
*/





///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-x-sum-of-all-k-long-subarrays-i
class Leet3318 {
    func findXSum(_ nums: [Int], _ k: Int, _ x: Int) -> [Int] {
        var frequency = nums[0...k-1].reduce(into: [:]) { $0[$1, default: 0] += 1 }
        var result: [Int] = []
        var sum = 0
        for right in k-1..<nums.count {
            // slide the window
            if right > k-1 {
                frequency[nums[right-k]]! -= 1 // remove left
                frequency[nums[right], default: 0] += 1 // add right
            }
            // get the x sum of the most highest frequency values in the dictionary, when equal use higher key
            let xSum = frequency.sorted { (a,b) -> Bool in
                let ((a, aFreq), (b, bFreq)) = (a, b)
                return aFreq == bFreq ? a > b : aFreq > bFreq
            }.prefix(x).map(*).reduce(0,+)
            result.append(xSum)
        }
        return result
    }
    static func test() {
        let sut = Leet3318()
        assert(sut.findXSum([1,1,2,2,3,4,2,3], 6, 2) ==  [6,10,12])
        assert(sut.findXSum([3,8,7,8,7,5], 2, 2) ==  [11,15,15,15,12])
    }
}
//Leet3318.test()



///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-positive-sum-subarray/
class Leet3364 {
    func minimumSumSubarray(_ nums: [Int], _ l: Int, _ r: Int) -> Int {
        var result = Int.max
        for i in 0..<nums.count {
            for j in i..<nums.count {
                let diff = j-i+1
                if (l...r).contains(diff) {
                    let currentSum = nums[i...j].reduce(0,+)
                    if currentSum > 0 {
                        result = min(result, currentSum)
                    }
                }
            }
        }
        return result == Int.max ? -1 : result
    }
    static func test() {
        let sut = Leet3364()
        assert(sut.minimumSumSubarray([3, -2, 1, 4], 2, 3) == 1)
        assert(sut.minimumSumSubarray([-2, 2, -3, 1], 2, 3) == -1)
        assert(sut.minimumSumSubarray([1, 2, 3, 4], 2, 4) == 3)
    }
}
//Leet3364.test()





private func gcd(_ m: Int, _ n: Int) -> Int {
    var b = max(m, n)
    var r = min(m, n)
    
    while r != 0 {
        let temp = b
        b = r
        r = temp % b
    }
    return b
}


private func lcm(_ m: Int, _ n: Int) -> Int {
    m*n / gcd(m, n)
}

//gcd(9,45)
//lcm(9,45)



class Leet1979 {
    func findGCD(_ nums: [Int]) -> Int {
        guard let min = nums.min(), let max = nums.max() else {
            return 1
        }
        return gcd(min, max)
    }
    static func test() {
        let sut = Leet1979()
        assert(sut.findGCD([2,5,6,9,10]) == 2)
        assert(sut.findGCD([7,5,6,8,3]) == 1)
        assert(sut.findGCD([3,3]) == 3)
    }
}
//Leet1979.test()



extension Int {
    func gcd(_ other: Int) -> Int {
        var a = abs(self)
        var b = abs(other)
        
        while b != 0 {
            let temp = a
            a = b
            b = temp % b
        }
        return a
    }
    
    func lcm(_ other: Int) -> Int {
        self * other / self.gcd(other)
    }
}

///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-subarray-with-equal-products/
class Leet3411 {
    func maxLength(_ nums: [Int]) -> Int {
        var result = 0
        for i in 0..<nums.count {
            for j in i..<nums.count {
                let subArray = nums[i...j]
                let gcd = subArray.reduce(0) { $0.gcd($1) }
                let lcm = subArray.reduce(1) { $0.lcm($1) }
                let product = subArray.map { Double($0) }.reduce(1.0, *)
                
                if product == Double(gcd * lcm) {
                    result = max(result, j - i + 1)
                }
            }
        }
        return result
    }
    static func test() {
        let sut = Leet3411()
        assert(sut.maxLength([1, 2, 1, 2, 1, 1, 1]) == 5)
        assert(sut.maxLength([2, 3, 4, 5, 6]) == 3)
        assert(sut.maxLength([1, 2, 3, 1, 4, 5, 1]) == 5)
        assert(sut.maxLength([1,2,8,3,5,5,9,3,5,6,7,5,5,6,6,7,7,3,5,1,2,2,1,7,6,8,3,4,7,3,1,5,5,4,4,6,8,8,7,6,3,8,7,3,6,10,8,7,4,8]) == 5)
    }
}
//Leet3411.test()




///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/n-queens/
class Leet0051 {
    
    private var size = 0
    private var solutions: [[String]] = []
    
    private func createBoardString(_ state: [[Character]]) -> [String] {
        state.map { String($0) }
    }
    
    func solveNQueens(_ n: Int) -> [[String]] {
        self.size = n
        self.solutions.removeAll()
        var state: [[Character]] = Array(repeating: Array(repeating: ".", count: n), count: n)
        var (colSet, diagonalSet, antiDiagonalSet) : (Set<Int>,  Set<Int>,  Set<Int>) = ([], [], [])
        backtrack(0, &colSet, &diagonalSet, &antiDiagonalSet, &state)
        return solutions
    }
    
    private func backtrack(_ row: Int, _ colSet: inout Set<Int>, _ diagonalSet: inout Set<Int>, _ antiDiagonalSet: inout Set<Int>, _  state: inout [[Character]]) {
        
        guard row < size else {
            return solutions.append(createBoardString(state))
        }
        
        for col in 0..<size {
            let diagonalIndex = row - col
            let antiDiagonalIndex = row + col
            
            // If the queen is not placeable
            if colSet.contains(col) || diagonalSet.contains(diagonalIndex) || antiDiagonalSet.contains(antiDiagonalIndex) {
                continue
            }
            
            // "Add" the queen to the board
            colSet.insert(col)
            diagonalSet.insert(diagonalIndex)
            antiDiagonalSet.insert(antiDiagonalIndex)
            state[row][col] = "Q"
                        
            // Move on to the next row with the updated board state
            backtrack(row + 1, &colSet, &diagonalSet, &antiDiagonalSet, &state)

            // "Remove" the queen from the board since we have already
            // explored all valid paths using the above function call
            colSet.remove(col)
            diagonalSet.remove(diagonalIndex)
            antiDiagonalSet.remove(antiDiagonalIndex)
            state[row][col] = "."
        }
    }
    
    static func test() {
        let sut = Leet0051()
        assert(sut.solveNQueens(4) == [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]])
        assert(sut.solveNQueens(1) == [["Q"]])
        assert(sut.solveNQueens(8) == [["Q.......","....Q...",".......Q",".....Q..","..Q.....","......Q.",".Q......","...Q...."],["Q.......",".....Q..",".......Q","..Q.....","......Q.","...Q....",".Q......","....Q..."],["Q.......","......Q.","...Q....",".....Q..",".......Q",".Q......","....Q...","..Q....."],["Q.......","......Q.","....Q...",".......Q",".Q......","...Q....",".....Q..","..Q....."],[".Q......","...Q....",".....Q..",".......Q","..Q.....","Q.......","......Q.","....Q..."],[".Q......","....Q...","......Q.","Q.......","..Q.....",".......Q",".....Q..","...Q...."],[".Q......","....Q...","......Q.","...Q....","Q.......",".......Q",".....Q..","..Q....."],[".Q......",".....Q..","Q.......","......Q.","...Q....",".......Q","..Q.....","....Q..."],[".Q......",".....Q..",".......Q","..Q.....","Q.......","...Q....","......Q.","....Q..."],[".Q......","......Q.","..Q.....",".....Q..",".......Q","....Q...","Q.......","...Q...."],[".Q......","......Q.","....Q...",".......Q","Q.......","...Q....",".....Q..","..Q....."],[".Q......",".......Q",".....Q..","Q.......","..Q.....","....Q...","......Q.","...Q...."],["..Q.....","Q.......","......Q.","....Q...",".......Q",".Q......","...Q....",".....Q.."],["..Q.....","....Q...",".Q......",".......Q","Q.......","......Q.","...Q....",".....Q.."],["..Q.....","....Q...",".Q......",".......Q",".....Q..","...Q....","......Q.","Q......."],["..Q.....","....Q...","......Q.","Q.......","...Q....",".Q......",".......Q",".....Q.."],["..Q.....","....Q...",".......Q","...Q....","Q.......","......Q.",".Q......",".....Q.."],["..Q.....",".....Q..",".Q......","....Q...",".......Q","Q.......","......Q.","...Q...."],["..Q.....",".....Q..",".Q......","......Q.","Q.......","...Q....",".......Q","....Q..."],["..Q.....",".....Q..",".Q......","......Q.","....Q...","Q.......",".......Q","...Q...."],["..Q.....",".....Q..","...Q....","Q.......",".......Q","....Q...","......Q.",".Q......"],["..Q.....",".....Q..","...Q....",".Q......",".......Q","....Q...","......Q.","Q......."],["..Q.....",".....Q..",".......Q","Q.......","...Q....","......Q.","....Q...",".Q......"],["..Q.....",".....Q..",".......Q","Q.......","....Q...","......Q.",".Q......","...Q...."],["..Q.....",".....Q..",".......Q",".Q......","...Q....","Q.......","......Q.","....Q..."],["..Q.....","......Q.",".Q......",".......Q","....Q...","Q.......","...Q....",".....Q.."],["..Q.....","......Q.",".Q......",".......Q",".....Q..","...Q....","Q.......","....Q..."],["..Q.....",".......Q","...Q....","......Q.","Q.......",".....Q..",".Q......","....Q..."],["...Q....","Q.......","....Q...",".......Q",".Q......","......Q.","..Q.....",".....Q.."],["...Q....","Q.......","....Q...",".......Q",".....Q..","..Q.....","......Q.",".Q......"],["...Q....",".Q......","....Q...",".......Q",".....Q..","Q.......","..Q.....","......Q."],["...Q....",".Q......","......Q.","..Q.....",".....Q..",".......Q","Q.......","....Q..."],["...Q....",".Q......","......Q.","..Q.....",".....Q..",".......Q","....Q...","Q......."],["...Q....",".Q......","......Q.","....Q...","Q.......",".......Q",".....Q..","..Q....."],["...Q....",".Q......",".......Q","....Q...","......Q.","Q.......","..Q.....",".....Q.."],["...Q....",".Q......",".......Q",".....Q..","Q.......","..Q.....","....Q...","......Q."],["...Q....",".....Q..","Q.......","....Q...",".Q......",".......Q","..Q.....","......Q."],["...Q....",".....Q..",".......Q",".Q......","......Q.","Q.......","..Q.....","....Q..."],["...Q....",".....Q..",".......Q","..Q.....","Q.......","......Q.","....Q...",".Q......"],["...Q....","......Q.","Q.......",".......Q","....Q...",".Q......",".....Q..","..Q....."],["...Q....","......Q.","..Q.....",".......Q",".Q......","....Q...","Q.......",".....Q.."],["...Q....","......Q.","....Q...",".Q......",".....Q..","Q.......","..Q.....",".......Q"],["...Q....","......Q.","....Q...","..Q.....","Q.......",".....Q..",".......Q",".Q......"],["...Q....",".......Q","Q.......","..Q.....",".....Q..",".Q......","......Q.","....Q..."],["...Q....",".......Q","Q.......","....Q...","......Q.",".Q......",".....Q..","..Q....."],["...Q....",".......Q","....Q...","..Q.....","Q.......","......Q.",".Q......",".....Q.."],["....Q...","Q.......","...Q....",".....Q..",".......Q",".Q......","......Q.","..Q....."],["....Q...","Q.......",".......Q","...Q....",".Q......","......Q.","..Q.....",".....Q.."],["....Q...","Q.......",".......Q",".....Q..","..Q.....","......Q.",".Q......","...Q...."],["....Q...",".Q......","...Q....",".....Q..",".......Q","..Q.....","Q.......","......Q."],["....Q...",".Q......","...Q....","......Q.","..Q.....",".......Q",".....Q..","Q......."],["....Q...",".Q......",".....Q..","Q.......","......Q.","...Q....",".......Q","..Q....."],["....Q...",".Q......",".......Q","Q.......","...Q....","......Q.","..Q.....",".....Q.."],["....Q...","..Q.....","Q.......",".....Q..",".......Q",".Q......","...Q....","......Q."],["....Q...","..Q.....","Q.......","......Q.",".Q......",".......Q",".....Q..","...Q...."],["....Q...","..Q.....",".......Q","...Q....","......Q.","Q.......",".....Q..",".Q......"],["....Q...","......Q.","Q.......","..Q.....",".......Q",".....Q..","...Q....",".Q......"],["....Q...","......Q.","Q.......","...Q....",".Q......",".......Q",".....Q..","..Q....."],["....Q...","......Q.",".Q......","...Q....",".......Q","Q.......","..Q.....",".....Q.."],["....Q...","......Q.",".Q......",".....Q..","..Q.....","Q.......","...Q....",".......Q"],["....Q...","......Q.",".Q......",".....Q..","..Q.....","Q.......",".......Q","...Q...."],["....Q...","......Q.","...Q....","Q.......","..Q.....",".......Q",".....Q..",".Q......"],["....Q...",".......Q","...Q....","Q.......","..Q.....",".....Q..",".Q......","......Q."],["....Q...",".......Q","...Q....","Q.......","......Q.",".Q......",".....Q..","..Q....."],[".....Q..","Q.......","....Q...",".Q......",".......Q","..Q.....","......Q.","...Q...."],[".....Q..",".Q......","......Q.","Q.......","..Q.....","....Q...",".......Q","...Q...."],[".....Q..",".Q......","......Q.","Q.......","...Q....",".......Q","....Q...","..Q....."],[".....Q..","..Q.....","Q.......","......Q.","....Q...",".......Q",".Q......","...Q...."],[".....Q..","..Q.....","Q.......",".......Q","...Q....",".Q......","......Q.","....Q..."],[".....Q..","..Q.....","Q.......",".......Q","....Q...",".Q......","...Q....","......Q."],[".....Q..","..Q.....","....Q...","......Q.","Q.......","...Q....",".Q......",".......Q"],[".....Q..","..Q.....","....Q...",".......Q","Q.......","...Q....",".Q......","......Q."],[".....Q..","..Q.....","......Q.",".Q......","...Q....",".......Q","Q.......","....Q..."],[".....Q..","..Q.....","......Q.",".Q......",".......Q","....Q...","Q.......","...Q...."],[".....Q..","..Q.....","......Q.","...Q....","Q.......",".......Q",".Q......","....Q..."],[".....Q..","...Q....","Q.......","....Q...",".......Q",".Q......","......Q.","..Q....."],[".....Q..","...Q....",".Q......",".......Q","....Q...","......Q.","Q.......","..Q....."],[".....Q..","...Q....","......Q.","Q.......","..Q.....","....Q...",".Q......",".......Q"],[".....Q..","...Q....","......Q.","Q.......",".......Q",".Q......","....Q...","..Q....."],[".....Q..",".......Q",".Q......","...Q....","Q.......","......Q.","....Q...","..Q....."],["......Q.","Q.......","..Q.....",".......Q",".....Q..","...Q....",".Q......","....Q..."],["......Q.",".Q......","...Q....","Q.......",".......Q","....Q...","..Q.....",".....Q.."],["......Q.",".Q......",".....Q..","..Q.....","Q.......","...Q....",".......Q","....Q..."],["......Q.","..Q.....","Q.......",".....Q..",".......Q","....Q...",".Q......","...Q...."],["......Q.","..Q.....",".......Q",".Q......","....Q...","Q.......",".....Q..","...Q...."],["......Q.","...Q....",".Q......","....Q...",".......Q","Q.......","..Q.....",".....Q.."],["......Q.","...Q....",".Q......",".......Q",".....Q..","Q.......","..Q.....","....Q..."],["......Q.","....Q...","..Q.....","Q.......",".....Q..",".......Q",".Q......","...Q...."],[".......Q",".Q......","...Q....","Q.......","......Q.","....Q...","..Q.....",".....Q.."],[".......Q",".Q......","....Q...","..Q.....","Q.......","......Q.","...Q....",".....Q.."],[".......Q","..Q.....","Q.......",".....Q..",".Q......","....Q...","......Q.","...Q...."],[".......Q","...Q....","Q.......","..Q.....",".....Q..",".Q......","......Q.","....Q..."]])
    }
    
}
//Leet0051.test()







///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/n-queens-ii/
class Leet0052 {
    func totalNQueens(_ n: Int) -> Int {
        self.size = n
        self.solutions.removeAll()
        var state: [[Character]] = Array(repeating: Array(repeating: ".", count: n), count: n)
        var (colSet, diagonalSet, antiDiagonalSet) : (Set<Int>,  Set<Int>,  Set<Int>) = ([], [], [])
        backtrack(0, &colSet, &diagonalSet, &antiDiagonalSet, &state)
        return solutions.count
    }
    
    private var size = 0
    private var solutions: [[String]] = []
    
    private func createBoardString(_ state: [[Character]]) -> [String] {
        state.map { String($0) }
    }
    
    private func backtrack(_ row: Int, _ colSet: inout Set<Int>, _ diagonalSet: inout Set<Int>, _ antiDiagonalSet: inout Set<Int>, _  state: inout [[Character]]) {
        
        guard row < size else {
            return solutions.append(createBoardString(state))
        }
        
        for col in 0..<size {
            let diagonalIndex = row - col
            let antiDiagonalIndex = row + col
            
            // If the queen is not placeable
            if colSet.contains(col) || diagonalSet.contains(diagonalIndex) || antiDiagonalSet.contains(antiDiagonalIndex) {
                continue
            }
            
            // "Add" the queen to the board
            colSet.insert(col)
            diagonalSet.insert(diagonalIndex)
            antiDiagonalSet.insert(antiDiagonalIndex)
            state[row][col] = "Q"
                        
            // Move on to the next row with the updated board state
            backtrack(row + 1, &colSet, &diagonalSet, &antiDiagonalSet, &state)

            // "Remove" the queen from the board since we have already
            // explored all valid paths using the above function call
            colSet.remove(col)
            diagonalSet.remove(diagonalIndex)
            antiDiagonalSet.remove(antiDiagonalIndex)
            state[row][col] = "."
        }
    }
    
    
    static func test() {
        let solution = Leet0052()
        assert(solution.totalNQueens(1) == 1)
        assert(solution.totalNQueens(2) == 0)
        assert(solution.totalNQueens(3) == 0)
        assert(solution.totalNQueens(4) == 2)
        assert(solution.totalNQueens(5) == 10)
        assert(solution.totalNQueens(6) == 4)
        assert(solution.totalNQueens(7) == 40)
        assert(solution.totalNQueens(8) == 92)
        assert(solution.totalNQueens(9) == 352)
    }
}
//Leet0052.test()





///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/alternating-groups-ii/
class Leet3208 {
    func numberOfAlternatingGroups(_ colors: [Int], _ k: Int) -> Int {
        var (result, size, right) = (0,1,1)
        
        while right < colors.count + k - 1 {
            if colors[right % colors.count] != colors[(right-1) % colors.count] {
                size+=1
            } else {
                size = 1
            }
            if size == k {
                size-=1
                result += 1
            }
            right+=1
        }
        return result
    }
    static func test() {
        let solution = Leet3208()
        assert(solution.numberOfAlternatingGroups([0,1,0,1,0], 3) == 3)
        assert(solution.numberOfAlternatingGroups([0,1,0,0,1,0,1], 6) == 2)
        assert(solution.numberOfAlternatingGroups([1,1,0,1], 4) == 0)
    }
}
//Leet3208.test()



///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-length-substring-with-two-occurrences/
class Leet3090 {
    func maximumLengthSubstring(_ s: String) -> Int {
        let k = 2
        let stringArray = Array(s)
        var maxLength = 0
        var left = 0
        var right = 0
        var window = [Character:Int]()
        
        while right < stringArray.count {
            
            let rightChar = stringArray[right]
            window[rightChar, default: 0] += 1

            // contract window and search for the leftChar that will reduce the window
            while window[rightChar]! > k {
                let leftChar = stringArray[left]
                window[leftChar, default: 0] -= 1
                left += 1
            }
            maxLength = max(maxLength, right - left + 1)
            
            right += 1
        }
        return maxLength
    }
    static func test() {
        let solution = Leet3090()
        assert(solution.maximumLengthSubstring("zadcfdddccb") == 6)
        assert(solution.maximumLengthSubstring("dcfdddccb") == 5)
        assert(solution.maximumLengthSubstring("bcbbbcba") == 4)
        assert(solution.maximumLengthSubstring("aaaa") == 2)
        assert(solution.maximumLengthSubstring("abbbbbbbbbbbbc") == 3)
        assert(solution.maximumLengthSubstring("cdba") == 4)
    }
}
//Leet3090.test()










///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/length-of-longest-subarray-with-at-most-k-frequency/description/
class Leet2958 {
    func maxSubarrayLength(_ nums: [Int], _ k: Int) -> Int {
        var maxLength = 0
        var left = 0
        var right = 0
        var window = [Int:Int]()
        
        while right < nums.count {
            
            let rightInt = nums[right]
            window[rightInt, default: 0] += 1

            // contract window and search for the leftChar that will reduce the window
            while window[rightInt]! > k {
                let leftInt = nums[left]
                window[leftInt, default: 0] -= 1
                left += 1
            }
            maxLength = max(maxLength, right - left + 1)
            right += 1
        }
        return maxLength
    }
    static func test() {
        let solution = Leet2958()
        assert(solution.maxSubarrayLength([1,2,2,1,3], 1) == 3)
        assert(solution.maxSubarrayLength([1,2,3,1,2,3,1,2], 2) == 6)
        assert(solution.maxSubarrayLength([1,2,1,2,1,2,1,2], 1) == 2)
        assert(solution.maxSubarrayLength([5,5,5,5,5,5,5], 4) == 4)
    }
}
//Leet2958.test()
















///
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/robot-room-cleaner/
public class Robot {
    // Returns true if the cell in front is open and robot moves into the cell.
    // Returns false if the cell in front is blocked and robot stays in the current cell.
    public func move() -> Bool { true }

    // Robot will stay in the same cell after calling turnLeft/turnRight.
    // Each turn will be 90 degrees.
    public func turnLeft() {}
    public func turnRight() {}

    // Clean the current cell.
    public func clean() {}
}

class Leet0489 {
    
    enum Direction: Int, CaseIterable {
        case up = 0
        case right = 1
        case down = 2
        case left = 3
        
        var vector: Cell {
            switch self {
            case .up: return Cell(row: -1,col: 0)
            case .right: return Cell(row: 0,col: 1)
            case .down: return Cell(row: 1,col: 0)
            case .left: return Cell(row: 0,col: -1)
            }
        }
    }
    
    struct Cell: Hashable {
        let row: Int
        let col: Int
    }
    var robot: Robot!
    var visited = Set<Cell>()
    
    func cleanRoom(_ robot: Robot) {
        self.robot = robot
        backtrack(0,0,0)
    }
    
    private func backtrack(_ row: Int, _ col: Int, _ direction: Int) {
        visited.insert(Cell(row: row, col: col))
        robot.clean()
        
        for i in Direction.allCases.indices {
            let newDirection = (direction + i) % 4
            let newRow = row + Direction(rawValue: newDirection)!.vector.row
            let newCol = col + Direction(rawValue: newDirection)!.vector.col
            
            if !visited.contains(Cell(row: newRow, col: newCol)) && robot.move() {
                backtrack(newRow, newCol, newDirection)
                goBack()
            }
            robot.turnRight()
        }
    }
    
    private func goBack() {
        robot.turnRight()
        robot.turnRight()
        robot.move()
        robot.turnRight()
        robot.turnRight()
    }
}










///---------------------------------------------------------------------------------------
/// Leetcode 36
///https://leetcode.com/problems/valid-sudoku/

class Leet0036 {

    /*
    func isValidSudoku(_ board: [[Character]]) -> Bool {
        
        var setBox = [Int:Set<Int>]()
        
        for i in 1 ... 3 {
            setBox[i] = Set<Int>()
            setBox[i*10] = Set<Int>()
            setBox[i*100] = Set<Int>()
        }
        
        for i in 0 ... 8 {
            var setRow = Set<Int>()
            var setCol = Set<Int>()
            
            for j in 0 ... 8 {
                
                // check if the i/j combination in the row set
                let charRow = board[i][j]
                if let num = Int(String(charRow)) {
                    if setRow.contains(num) {
                        return false
                    } else {
                        setRow.insert(num)
                    }
                }
                
                // check if the i/j combination is in the column set
                let charCol = board[j][i]
                if let num = Int(String(charCol)) {
                    if setCol.contains(num) {
                        return false
                    } else {
                        setCol.insert(num)
                    }
                }
                
                // check if the i/j combination is in the box set
                // range is 1...9 to calculate the box set
                let boxIndex = Int(pow(Double(10), Double(Int(i/3)))) * (j/3+1)
                let charBox = board[i][j]
                if let num = Int(String(charBox)) {
                    if let setBox = setBox[boxIndex], setBox.contains(num) {
                        return false
                    } else {
                        setBox[boxIndex]?.insert(num)
                    }
                }
            }
        }
        return true
    }
     */
    
    func isValidSudoku(_ board: [[Character]]) -> Bool {
        
        var setRows: [Set<Character>] = Array(repeating: [], count: 9)
        var setCols: [Set<Character>] = Array(repeating: [], count: 9)
        var setBoxes: [Set<Character>] = Array(repeating: [], count: 9)
        
        for row in 0..<9 {
            for col in 0..<9 {
                let val = board[row][col]
                
                guard val != "." else { continue }
                
                if setRows[row].contains(val) {
                    return false
                }
                setRows[row].insert(val)
                
                if setCols[col].contains(val) {
                    return false
                }
                setCols[col].insert(val)
                
                let boxIndex = row/3 * 3 + col/3
                if setBoxes[boxIndex].contains(val) {
                    return false
                }
                setBoxes[boxIndex].insert(val)
            }
        }
        return true
    }
    
    
    static func test() {
        let sut = Leet0036()
        //                       1   1   1   2   2   2   3   3   3
        assert(
            sut.isValidSudoku([["5","3",".",".","7",".",".",".","."]
                               ,["6",".",".","1","9","5",".",".","."]
                               ,[".","9","8",".",".",".",".","6","."]
                               ,["8",".",".",".","6",".",".",".","3"]
                               ,["4",".",".","8",".","3",".",".","1"]
                               ,["7",".",".",".","2",".",".",".","6"]
                               ,[".","6",".",".",".",".","2","8","."]
                               ,[".",".",".","4","1","9",".",".","5"]
                               ,[".",".",".",".","8",".",".","7","9"]]) == true)
        assert(
            sut.isValidSudoku([["8","3",".",".","7",".",".",".","."]
                               ,["6",".",".","1","9","5",".",".","."]
                               ,[".","9","8",".",".",".",".","6","."]
                               ,["8",".",".",".","6",".",".",".","3"]
                               ,["4",".",".","8",".","3",".",".","1"]
                               ,["7",".",".",".","2",".",".",".","6"]
                               ,[".","6",".",".",".",".","2","8","."]
                               ,[".",".",".","4","1","9",".",".","5"]
                               ,[".",".",".",".","8",".",".","7","9"]]) == false)
    }
}
//Leet0036.test()




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sudoku-solver/description/
class Leet0037 {

    func solveSudoku(_ board: inout [[Character]]) {
        
        func isPlaceable(_ newNum: Character, _ row: Int, _ col: Int) -> Bool {
            for i in 0..<9 where board[row][i] == newNum || board[i][col] == newNum || board[(row/3)*3 + i/3][(col/3)*3 + i%3] == newNum {
                return false
            }
            return true
        }

        func solve() -> Bool {
            for row in 0..<9 {
                for col in 0..<9 where board[row][col] == "." {
                    for num in "123456789" where isPlaceable(num, row, col) {
                        board[row][col] = num
                        if solve() {
                            return true
                        }
                        board[row][col] = "."
                    }
                    return false
                }
            }
            return true
        }
        
        solve()
    }
    
    static func test() {
        let sut = Leet0037()
        var board: [[Character]] = [["5","3",".",".","7",".",".",".","."],
                                    ["6",".",".","1","9","5",".",".","."],
                                    [".","9","8",".",".",".",".","6","."],
                                    ["8",".",".",".","6",".",".",".","3"],
                                    ["4",".",".","8",".","3",".",".","1"],
                                    ["7",".",".",".","2",".",".",".","6"],
                                    [".","6",".",".",".",".","2","8","."],
                                    [".",".",".","4","1","9",".",".","5"],
                                    [".",".",".",".","8",".",".","7","9"]]
        sut.solveSudoku(&board)
        assert(board == [["5","3","4","6","7","8","9","1","2"],
                         ["6","7","2","1","9","5","3","4","8"],
                         ["1","9","8","3","4","2","5","6","7"],
                         ["8","5","9","7","6","1","4","2","3"],
                         ["4","2","6","8","5","3","7","9","1"],
                         ["7","1","3","9","2","4","8","5","6"],
                         ["9","6","1","5","3","7","2","8","4"],
                         ["2","8","7","4","1","9","6","3","5"],
                         ["3","4","5","2","8","6","1","7","9"]])
    }
}
//Leet0037.test()
    




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/check-if-every-row-and-column-contains-all-numbers/
class Leet2133 {
    func checkValid(_ matrix: [[Int]]) -> Bool {
        let n = matrix.count
        var setRows: [Set<Int>] = Array(repeating: [], count: n)
        var setCols: [Set<Int>] = Array(repeating: [], count: n)
        for row in 0..<n {
            for col in 0..<n {
                let val = matrix[row][col]
                if setRows[row].contains(val) {
                    return false
                }
                setRows[row].insert(val)
                
                if setCols[col].contains(val) {
                    return false
                }
                setCols[col].insert(val)
            }
        }
        return true
    }
    
    static func test() {
        let sut = Leet2133()
        assert(sut.checkValid([[1,2,3],[3,1,2],[2,3,1]]))
        assert(!sut.checkValid([[1,1,1],[1,2,3],[1,2,3]]))
        assert(!sut.checkValid([[15,7,18,11,19,10,14,16,8,2,3,6,5,1,17,12,9,4,13],[17,15,9,8,11,13,7,6,5,1,3,16,12,19,10,2,4,14,18],[19,14,12,10,8,9,17,16,4,3,13,18,1,5,7,11,2,15,6],[4,2,10,15,19,16,8,9,5,3,1,11,13,14,6,18,12,17,7],[13,19,9,16,5,8,6,12,14,11,18,10,7,2,3,4,15,17,1],[4,7,18,11,17,16,5,12,10,1,15,13,14,6,19,2,3,9,8],[14,5,15,1,18,6,12,7,8,9,3,13,2,10,19,4,11,16,17],[10,3,1,8,14,19,11,18,15,13,9,12,16,17,7,4,5,2,6],[14,13,19,18,7,2,4,8,10,17,12,5,15,1,6,9,11,3,16],[19,8,10,18,16,12,11,17,4,9,7,2,5,13,15,3,6,1,14],[1,10,6,14,7,18,3,9,4,16,5,11,13,17,15,8,19,2,12],[13,10,5,16,1,19,17,3,9,11,7,8,12,6,4,2,14,15,18],[17,2,1,6,9,19,18,14,4,11,12,13,16,5,8,7,3,10,15],[1,4,10,5,13,6,18,11,3,2,15,14,16,12,17,19,8,9,7],[2,14,3,12,16,17,11,9,1,6,5,19,10,13,4,18,7,15,8],[15,9,8,18,14,13,4,12,5,17,6,1,11,16,19,3,7,2,10],[15,8,12,16,13,2,6,19,18,14,10,5,11,9,7,1,3,17,4],[15,6,17,7,5,3,1,9,19,12,10,11,16,14,18,8,2,13,4],[6,11,10,14,2,13,16,1,9,15,8,19,17,3,5,18,7,4,12]]))
        assert(!sut.checkValid([[1,2,3,4],[4,2,3,1],[1,4,2,3],[4,1,2,3]]))
    }
}
//Leet2133.test()












///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/combinations/
class Leet0077 {

    private var (n, k) = (0, 0)
    
    func combine(_ n: Int, _ k: Int) -> [[Int]] {
        self.n = n
        self.k = k
        var result: [[Int]] = []
        backtrack(1, [], &result)
        return result
    }
    
    private func backtrack(_ first: Int, _ current: [Int], _ result: inout [[Int]]) {
        guard current.count < k else {
            result.append(current)
            return
        }
        
        let need = k - current.count
        for num in first...n - need + 1 {
            var next = current
            next.append(num)
            backtrack(num + 1, next, &result)
            next.removeLast()
        }
    }
    

    static func test() {
        let sut = Leet0077()
        assert(sut.combine(4, 2) == [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]])
        assert(sut.combine(1, 1) == [[1]])
    }
}
//Leet0077.test()




     
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/permutations/
class Leet0046 {

    func permute(_ nums: [Int]) -> [[Int]] {
        var result = [[Int]]()
        backtrack(nums, [], &result)
        return result
    }
    
    func backtrack(_ original: [Int], _ current: [Int], _ result: inout [[Int]]) {
        var current = current
        guard current.count < original.count else {
            result.append(current)
            return
        }
        for num in original where !current.contains(num) {
            current.append(num)
            backtrack(original, current, &result)
            current.removeLast()
        }
    }
    
    static func test() {
        let sut = Leet0046()
        assert(sut.permute([1,2,3]) == [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]])
        assert(sut.permute([0,1]) == [[0,1],[1,0]])
        assert(sut.permute([1]) == [[1]])
    }
}
//Leet0046.test()






///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/same-tree/description/
class Leet0100 {
    
    func isSameTree(_ p: TreeNode?, _ q: TreeNode?) -> Bool {
        guard let p = p, let q = q else {
            return p == nil && q == nil
        }
        var stack: [TreeNode?] = [p, q]
        while let a = stack.popLast(), let b = stack.popLast( ) {
            if a == nil && b == nil { continue }
            guard a?.val == b?.val else {
                return false
            }
            stack.append(contentsOf: [a?.left, b?.left, a?.right, b?.right])
        }
        return stack.isEmpty
    }
    
    func isSameTree_Recursion(_ p: TreeNode?, _ q: TreeNode?) -> Bool {
        guard let p = p, let q = q else {
            return p == nil && q == nil
        }
        return p.val == q.val
        && isSameTree(p.left, q.left)
        && isSameTree(p.right, q.right)
    }

    func isSameTree_Equatable(_ p: TreeNode?, _ q: TreeNode?) -> Bool {
        p == q
    }
    
    static func test() {
        let sut = Leet0100()
        assert(sut.isSameTree(TreeNode(1, TreeNode(2), TreeNode(3)), TreeNode(1, TreeNode(2), TreeNode(3))))
        assert(sut.isSameTree(TreeNode(1, TreeNode(2), nil), TreeNode(1, nil, TreeNode(2))) == false)
        assert(sut.isSameTree(TreeNode(1, TreeNode(2), TreeNode(1)), TreeNode(1, TreeNode(1), TreeNode(2))) == false)
        assert(sut.isSameTree(TreeNode(1), TreeNode(1)))
        assert(sut.isSameTree(TreeNode(1), TreeNode(2)) == false)
    }
}
extension TreeNode: Equatable {
    public static func == (lhs: TreeNode, rhs: TreeNode) -> Bool {
        lhs.val == rhs.val && lhs.left == rhs.left && lhs.right == rhs.right
    }
}
//Leet0100.test()







///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/design-circular-queue/
///Leet0622
class MyCircularQueue {

    private var queue: [Int]
    private var (front, capacity, count) = (0, 0, 0)
    
    init(_ k: Int) {
        capacity = k
        queue = Array(repeating: 0, count: k)
    }
    
    func enQueue(_ value: Int) -> Bool {
        guard count < capacity else {
            return false
        }
        queue[(front + count) % capacity] = value
        count += 1
        return true
    }
    
    func deQueue() -> Bool {
        guard count > 0 else {
            return false
        }
        front = (front + 1) % capacity
        count -= 1
        return true
    }
    
    func Front() -> Int {
        count > 0 ? queue[front] : -1
    }
    
    func Rear() -> Int {
        count > 0 ? queue[(front + count - 1) % capacity] : -1
    }
    
    func isEmpty() -> Bool {
        count == 0
    }
    
    func isFull() -> Bool {
        capacity == count
    }
}






///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/design-circular-deque/
///Leet0641
class MyCircularDeque {
    private var front = 0, count = 0, deque: [Int]
    
    init(_ k: Int) {
        deque = Array(repeating: 0, count: k)
    }
    
    func insertFront(_ value: Int) -> Bool {
        insert(value, true)
    }
    
    func insertLast(_ value: Int) -> Bool {
        insert(value, false)
    }
    
    func deleteFront() -> Bool {
        delete(true)
    }
    
    func deleteLast() -> Bool {
        delete(false)
    }
    
    func getFront() -> Int {
        !isEmpty() ? deque[front] : -1
    }
    
    func getRear() -> Int {
        !isEmpty() ? deque[(front + count - 1) % deque.count] : -1
    }
    
    func isEmpty() -> Bool {
        count == 0
    }
    
    func isFull() -> Bool {
        count == deque.count
    }
    
    private func insert(_ value: Int, _ isFront: Bool) -> Bool {
        guard !isFull() else {
            return false
        }
        count += 1
        if isFront {
            front = (front - 1 + deque.count) % deque.count
            deque[front] = value
        } else {
            deque[(front + count - 1) % deque.count] = value
        }
        return true
    }
    
    
    private func delete(_ isFront: Bool) -> Bool {
        guard !isEmpty() else {
            return false
        }
        count -= 1
        if isFront {
            front = (front + 1) % deque.count
        }
        return true
    }
    
    static func test() {
        let sut = MyCircularDeque(3)
        assert(sut.insertLast(1))  // return True
        assert(sut.insertLast(2))  // return True
        assert(sut.insertFront(3)) // return True
        assert(sut.insertFront(4) == false) // return False, the queue is full.
        assert(sut.getRear() == 2)      // return 2
        assert(sut.isFull())       // return True
        assert(sut.deleteLast())   // return True
        assert(sut.insertFront(4)) // return True
        assert(sut.getFront() == 4)     // return 4
        
        /**
         ["MyCircularDeque","insertLast","insertLast","insertFront","insertFront","getRear","isFull","deleteLast","insertFront","getFront"]
         [[3],[1],[2],[3],[4],[],[],[],[4],[]]
         ["MyCircularDeque","insertFront","getRear","deleteLast","getRear","insertFront","insertFront","insertFront","insertFront","isFull","insertFront","isFull","getRear","deleteLast","getFront","getFront","insertLast","deleteFront","getFront","insertLast","getRear","insertLast","getRear","getFront","getFront","getFront","getRear","getRear","insertFront","getFront","getFront","getFront","getFront","deleteFront","insertFront","getFront","deleteLast","insertLast","insertLast","getRear","getRear","getRear","isEmpty","insertFront","deleteLast","getFront","deleteLast","getRear","getFront","isFull","isFull","deleteFront","getFront","deleteLast","getRear","insertFront","getFront","insertFront","insertFront","getRear","isFull","getFront","getFront","insertFront","insertLast","getRear","getRear","deleteLast","insertFront","getRear","insertLast","getFront","getFront","getFront","getRear","insertFront","isEmpty","getFront","getFront","insertFront","deleteFront","insertFront","deleteLast","getFront","getRear","getFront","insertFront","getFront","deleteFront","insertFront","isEmpty","getRear","getRear","getRear","getRear","deleteFront","getRear","isEmpty","deleteFront","insertFront","insertLast","deleteLast"]
         [[77],[89],[],[],[],[19],[23],[23],[82],[],[45],[],[],[],[],[],[74],[],[],[98],[],[99],[],[],[],[],[],[],[8],[],[],[],[],[],[75],[],[],[35],[59],[],[],[],[],[22],[],[],[],[],[],[],[],[],[],[],[],[21],[],[26],[63],[],[],[],[],[87],[76],[],[],[],[26],[],[67],[],[],[],[],[36],[],[],[],[72],[],[87],[],[],[],[],[85],[],[],[91],[],[],[],[],[],[],[],[],[],[34],[44],[]]
         ["MyCircularDeque","insertLast","insertLast","insertLast","deleteFront","deleteLast","deleteFront"]
         [[3],[1],[2],[3],[],[],[]]
         ["MyCircularDeque","insertLast","insertLast","insertLast","deleteLast","deleteLast","deleteLast", "deleteLast"]
         [[3],[1],[2],[3],[],[],[],[]]
         ["MyCircularDeque","insertFront","insertFront","insertFront","deleteLast","deleteLast","deleteLast", "deleteLast"]
         [[3],[1],[2],[3],[],[],[],[]]
         ["MyCircularDeque","insertFront","insertLast","getRear","getFront","getFront","deleteLast","deleteFront","getRear"]
         [[41],[70],[11],[],[],[],[],[],[]]
         ["MyCircularDeque","insertFront","insertFront","getRear","getFront","getFront","deleteLast","deleteFront","getRear"]
         [[41],[70],[11],[],[],[],[],[],[]]
         ["MyCircularDeque","insertFront","insertLast","deleteFront","getFront","deleteLast","insertLast","isEmpty","deleteLast","insertFront","getRear","deleteFront","insertFront","insertLast","deleteLast","getFront","getRear","insertFront","getRear","getFront"]
         [[999],[93],[578],[],[],[],[533],[],[],[913],[],[],[100],[57],[],[],[],[900],[],[]]
         */
    }
}
//MyCircularDeque.test()










/*
class MovingAverageArraySlice {
    private var window: ArraySlice<Int>
    private var sum = 0
    private let size: Int
    
    init(_ size: Int) {
        self.size = size
        window = []
    }
    
    func next(_ val: Int) -> Double {
        window.append(val)
        sum += val
        if window.count > size {
            sum -= window.removeFirst()
        }
        return Double(sum) / Double(window.count)
    }
}
 */

/*
class MovingAverage {
    private var window: [Int]
    private var sum = 0, count = 0, head = 0
    private let size: Int
    
    init(_ size: Int) {
        self.size = size
        window = Array(repeating: 0, count: size)
    }
    
    func next(_ val: Int) -> Double {
        count += 1
        let tail = (head + 1) % size
        sum += val - window[tail]
        head = (head + 1) % size
        window[head] = val
        return Double(sum) / Double(min(count, window.count))
    }
}
*/

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/moving-average-from-data-stream/
///Leet0346
class MovingAverage {
    private var window: [Int]   // Fixed-size array acting as a circular queue
    private var sum = 0         // Keeps track of the sum of the current window elements
    private var head = 0        // Points to the current position in the circular queue
    private var count = 0       // Number of elements currently in the window
    private let size: Int       // Maximum size of the moving window

    init(_ size: Int) {
        self.size = size
        self.window = Array(repeating: 0, count: size)  // Initialize fixed-size array with 0s
    }

    func next(_ val: Int) -> Double {
        let tail = head % size  // Circular index for replacing old values
        
        sum -= window[tail]     // Remove the oldest value from sum
        sum += val              // Add the new value to sum
        
        window[tail] = val      // Replace the oldest value with the new one
        head += 1               // Move head forward
        
        count = min(count + 1, size)  // Ensure count does not exceed size
        
        return Double(sum) / Double(count)  // Compute and return the moving average
    }
    
    static func test() {
        let solution = MovingAverage(3)
        assert(solution.next(1) == 1.0)
        assert(solution.next(10) == 5.5)
        assert(solution.next(3) == 4.666666666666667)
        assert(solution.next(5) == 6.0)
        assert(solution.next(16) == 8.0)
        assert(solution.next(-9) == 4.0)
        assert(solution.next(2) == 3.0)
    }
}
//MovingAverage.test()





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/walls-and-gates/
import DequeModule
class Leet0286 {
        
    private struct Cell: Hashable {
        let row: Int
        let col: Int
        
        enum State: Int {
            case empty = 2147483647 //Int(Int32.max)
            case gate = 0
            case wall = -1
        }
    }
    
    private enum Direction: Int, CaseIterable {
        case down = 0
        case up
        case right
        case left
        var vector: Cell {
            switch self {
            case .up: return Cell(row: -1,col: 0)
            case .right: return Cell(row: 0,col: 1)
            case .down: return Cell(row: 1,col: 0)
            case .left: return Cell(row: 0,col: -1)
            }
        }
    }

    func wallsAndGates(_ rooms: inout [[Int]]) {
        guard !rooms.isEmpty else { return }
        
        let m = rooms.count
        let n = rooms[0].count
        var queue: Deque<Cell> = []
        
        // Queue all gates
        for i in 0..<m {
            for j in 0..<n {
                if rooms[i][j] == Cell.State.gate.rawValue {
                    queue.append(Cell(row: i, col: j))
                }
            }
        }

        while !queue.isEmpty {
            let cell = queue.removeFirst()              // process current cell
            let row = cell.row
            let col = cell.col

            // process a cell by testing all of it's next direction
            for direction in Direction.allCases {
                let r = row + direction.vector.row      // calculate next indeces
                let c = col + direction.vector.col
                let isNextIndexOutOfBounds = (r < 0 || r >= m || c < 0 || c >= n)
                guard !isNextIndexOutOfBounds else { continue }
                guard rooms[r][c] == Cell.State.empty.rawValue else { continue }
                rooms[r][c] = rooms[row][col] + 1       // add the distance from the current
                queue.append(Cell(row: r, col: c))      // add next empty cell to the deque
            }
        }
    }
    
    static func test() {
        let sut = Leet0286()
        
        var rooms: [[Int]] = [[2147483647,-1,0,2147483647],[2147483647,2147483647,2147483647,-1],[2147483647,-1,2147483647,-1],[0,-1,2147483647,2147483647]]
        sut.wallsAndGates(&rooms)
        assert(rooms == [[3, -1, 0, 1], [2, 2, 1, -1], [1, -1, 2, -1], [0, -1, 3, 4]])

        rooms = [[-1]]
        sut.wallsAndGates(&rooms)
        assert(rooms == [[-1]])

        rooms = [[0,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647],[2147483647,0,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647],[2147483647,2147483647,0,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647],[2147483647,2147483647,2147483647,0,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647],[2147483647,2147483647,2147483647,2147483647,0,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647],[2147483647,2147483647,2147483647,2147483647,2147483647,0,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647],[2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,0,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647],[2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,0,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647],[2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,0,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647],[2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,0,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647],[2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,0,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647],[2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,0,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647],[2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,0,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647],[2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,0,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647],[2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,0,2147483647,2147483647,2147483647,2147483647,2147483647],[2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,0,2147483647,2147483647,2147483647,2147483647],[2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,0,2147483647,2147483647,2147483647],[2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,0,2147483647,2147483647],[2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,0,2147483647],[2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,2147483647,0]]
        sut.wallsAndGates(&rooms)
        assert(rooms == [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],[1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],[2,1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],[3,2,1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],[4,3,2,1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[5,4,3,2,1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],[6,5,4,3,2,1,0,1,2,3,4,5,6,7,8,9,10,11,12,13],[7,6,5,4,3,2,1,0,1,2,3,4,5,6,7,8,9,10,11,12],[8,7,6,5,4,3,2,1,0,1,2,3,4,5,6,7,8,9,10,11],[9,8,7,6,5,4,3,2,1,0,1,2,3,4,5,6,7,8,9,10],[10,9,8,7,6,5,4,3,2,1,0,1,2,3,4,5,6,7,8,9],[11,10,9,8,7,6,5,4,3,2,1,0,1,2,3,4,5,6,7,8],[12,11,10,9,8,7,6,5,4,3,2,1,0,1,2,3,4,5,6,7],[13,12,11,10,9,8,7,6,5,4,3,2,1,0,1,2,3,4,5,6],[14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,1,2,3,4,5],[15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,1,2,3,4],[16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,1,2,3],[17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,1,2],[18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,1],[19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]])
    }
}
//Leet0286.test()



 








///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/number-of-islands/
class Leet0200_Number_of_Islands {
        
    private struct Cell: Hashable {
        let row: Int
        let col: Int
        
        enum State: Character {
            case water = "0"
            case island = "1"
        }
    }
    
    private enum Direction: Int, CaseIterable {
        case down = 0
        case up
        case right
        case left
        var vector: Cell {
            switch self {
            case .up: return Cell(row: -1,col: 0)
            case .right: return Cell(row: 0,col: 1)
            case .down: return Cell(row: 1,col: 0)
            case .left: return Cell(row: 0,col: -1)
            }
        }
    }
    
    func numIslands(_ grid: [[Character]]) -> Int {

        // init visited set of cells
        let m = grid.count
        let n = grid[0].count
        var deque: Deque<Cell> = []
        var visited: Set<Cell> = []
        var islandsCount = 0
        
        // look at all chars
        for i in 0..<m {
            for j in 0..<n {
                
                guard grid[i][j] == Cell.State.island.rawValue else { continue }

                let island = Cell(row: i, col: j)
                guard !visited.contains(island) else { continue }
                
                deque.append(island)
                
                while !deque.isEmpty {
                    let cell = deque.removeFirst()                  // process current cell
                    let row = cell.row
                    let col = cell.col
                    
                    guard !visited.contains(cell) else { continue } // skip cell if visited
                    
                    visited.insert(cell)
                    islandsCount += 1
                    
                    // process a cell by testing all of it's next direction
                    for direction in Direction.allCases {
                        let r = row + direction.vector.row      // calculate next indeces
                        let c = col + direction.vector.col
                        
                        let isNextIndexOutOfBounds = (r < 0 || r >= m || c < 0 || c >= n)
                        guard !isNextIndexOutOfBounds else { continue }
                        
                        guard grid[r][c] == Cell.State.island.rawValue else { continue }
                        
                        let nextCell = Cell(row: r, col: c)
                        visited.insert(nextCell)
                        deque.append(nextCell)
                    }
                }
            }
        }
        
        return islandsCount
    }
}



/*
TEST CASES



[["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]
[["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]]
[["1","1","1"],["0","1","0"],["0","1","0"]]
[["1"],["0"],["1"],["0"],["1"],["1"]]
[["1","0","1","1","0","1","1"]]
[["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]]
[["0"]]
[["1"],["0"],["1"],["0"],["1"],["0"],["1"],["1"],["0"],["1"],["1"],["1"],["1"],["1"],["1"],["1"],["1"],["0"],["0"],["1"]]


[["0"]]
[["1"],["0"],["1"],["0"],["1"],["0"],["1"],["1"],["0"],["1"],["1"],["1"],["1"],["1"],["1"],["1"],["1"],["0"],["0"],["1"]]
[["0","1","0","0","1","1","1","1","0","1","0","1","0","0","0","0","0","1","0","0","0","0","1"]]
[["0","1","1","1","1","0","0","0","0","0","1","0","0","1","1","1","0","0","1","0","1","0","0","0","1","1","0","1","1","1"],["1","0","0","0","1","1","1","1","0","1","1","1","0","0","1","1","0","0","0","0","0","1","1","1","1","1","0","1","1","0"]]
[["0","0","0","0","1","0","0","0","0","0","1","1","0","1","1","0","0","1","1","1","1","1","0","1","0","1","0"],["0","1","0","1","0","1","1","1","0","1","0","0","1","1","1","1","0","1","0","0","0","1","1","0","0","0","1"],["0","1","0","0","1","0","0","0","1","0","0","0","1","0","0","1","0","0","1","1","0","0","1","0","1","0","0"],["1","1","1","0","0","1","1","0","0","0","1","1","0","0","1","1","1","1","1","1","0","1","1","0","0","1","0"],["0","1","1","1","1","0","0","1","1","0","0","0","1","1","0","1","0","1","0","1","0","0","0","0","0","0","1"],["1","1","0","0","0","1","1","0","0","0","1","1","0","0","1","1","1","1","1","0","0","0","0","0","1","1","1"],["1","0","1","1","1","0","0","0","1","0","0","1","0","1","0","0","0","0","0","1","1","1","0","1","0","0","0"],["0","0","0","1","0","0","0","0","0","1","0","0","0","0","0","0","0","1","1","0","1","0","0","1","0","1","1"],["0","1","1","0","1","0","1","0","0","0","1","1","0","1","0","1","1","1","0","0","1","0","1","0","0","0","0"],["1","0","1","0","1","1","1","0","0","1","0","0","1","1","1","0","1","1","1","1","1","1","1","1","0","0","1"],["1","0","0","1","0","0","0","0","1","1","1","0","1","0","0","0","0","1","1","0","1","0","0","1","0","0","1"],["1","1","1","0","1","1","0","1","1","0","1","1","0","0","0","1","0","0","1","1","0","0","1","1","0","1","0"],["1","0","1","1","1","1","0","1","0","1","1","1","1","1","1","0","1","1","1","0","1","0","0","1","0","0","0"]]
[["1","0","1","0","0","1","1","0","0","1","1","1","1","1","0","0","1","0","1","0","1","1","1","0","1","1","1","0","1","0"],["0","1","1","0","0","0","1","1","1","0","0","0","0","1","1","0","0","0","1","1","1","0","1","1","1","1","0","0","1","1"],["0","1","1","1","0","0","0","0","1","0","0","0","1","0","1","0","1","0","0","1","0","0","0","0","0","0","1","0","1","0"],["1","1","0","1","1","0","1","0","1","1","1","1","0","1","0","0","1","1","0","0","1","0","0","1","0","0","1","1","1","1"],["0","0","1","0","0","1","0","1","1","0","0","0","1","1","1","0","1","0","0","1","0","0","1","0","1","1","0","0","0","1"],["0","1","0","1","1","1","0","1","0","0","0","0","0","0","0","1","0","0","0","1","0","1","1","0","1","1","0","1","0","0"],["1","0","0","1","0","0","1","1","1","0","0","0","0","1","0","1","0","1","0","1","0","1","0","0","1","0","1","1","0","1"],["1","0","0","1","1","1","1","0","1","1","1","1","0","0","1","0","0","1","0","0","1","1","1","0","1","1","1","1","0","1"],["0","0","1","1","0","1","0","0","1","0","1","1","0","1","1","1","1","1","1","0","0","0","0","0","0","0","0","1","1","1"],["0","0","0","1","0","1","1","0","1","1","0","1","0","1","1","1","1","1","1","1","0","0","0","0","1","1","1","0","1","1"],["1","1","0","1","0","0","1","0","1","0","0","1","1","1","0","0","1","0","0","1","0","0","1","0","1","1","1","0","1","1"],["1","1","1","0","0","1","0","0","0","0","0","1","0","0","1","1","1","1","0","1","0","1","1","0","1","1","1","1","0","0"],["1","0","1","0","0","0","1","1","1","0","0","1","0","1","1","0","0","0","0","0","1","0","0","0","0","0","0","1","1","1"],["1","1","0","0","0","1","0","0","0","0","1","1","0","1","1","0","0","0","1","0","0","0","1","1","1","0","1","0","1","0"],["1","0","0","1","0","1","1","1","1","0","1","1","0","0","1","0","0","1","1","1","0","0","1","0","0","1","0","1","1","1"],["1","0","0","0","0","0","1","0","1","0","1","1","0","1","1","1","0","1","0","1","1","0","0","1","1","1","0","0","0","0"],["1","1","1","0","1","1","0","0","1","0","1","1","0","1","1","1","0","1","0","0","0","1","1","1","1","1","1","1","1","0"],["1","0","0","1","1","1","1","0","1","1","0","1","1","0","1","0","0","1","0","0","1","1","1","1","0","1","1","0","0","0"],["1","1","0","0","1","0","0","0","0","0","1","0","0","1","0","0","0","0","0","1","1","0","0","1","0","1","0","1","1","1"],["0","0","0","0","0","0","0","0","1","0","0","1","1","1","0","1","1","1","0","0","1","0","0","0","1","1","0","0","0","0"],["1","0","1","1","0","1","1","1","0","0","1","1","1","1","0","1","1","0","0","1","0","0","1","0","0","1","1","0","0","1"],["0","0","0","0","0","1","1","1","1","1","0","1","0","1","1","1","1","0","1","1","0","0","0","0","1","0","1","0","1","1"],["1","0","0","1","0","1","0","0","1","1","0","0","1","0","1","0","1","1","1","1","0","0","0","0","0","1","1","0","1","0"],["1","0","0","0","1","0","0","0","0","1","0","0","0","0","0","1","1","1","0","0","1","0","0","1","0","1","0","0","0","0"],["0","0","0","1","1","0","1","0","0","0","1","0","1","0","0","1","0","1","0","0","0","1","0","0","1","1","1","1","1","1"],["1","1","0","1","0","1","0","0","1","1","1","1","0","0","1","0","1","0","0","1","0","0","1","1","1","1","1","0","1","1"],["0","0","0","1","1","1","1","0","1","1","1","1","0","0","1","1","1","1","1","1","0","1","1","1","1","0","0","1","1","1"],["0","1","0","0","0","0","0","0","0","0","1","0","1","1","1","1","0","0","1","1","1","1","0","0","1","1","0","0","1","1"],["0","1","0","0","1","1","0","0","0","1","0","1","1","1","0","0","1","0","1","1","1","0","1","0","1","1","0","0","0","0"],["1","0","1","1","1","1","1","1","1","0","0","1","1","0","0","0","0","1","1","0","1","0","0","0","1","0","0","0","0","0"]]
[["1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0"],["0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1"],["1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0"],["0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1"],["1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0"],["0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1"],["1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0"],["0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1"],["1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0"],["0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1"],["1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0"],["0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1"],["1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0"],["0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1"],["1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0"],["0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1"],["1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0"],["0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1"],["1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0"],["0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1"],["1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0"],["0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1"],["1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0"],["0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1"],["1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0"],["0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1"],["1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0"],["0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1"],["1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0"],["0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1","0","1"]]
[["1","0","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","1"],["1","0","0","0","0","0","1","1","1","1","1","1","1","1","1","1","1","1","1","1"],["1","1","1","1","1","0","0","0","1","1","1","1","1","1","1","1","1","1","1","1"],["1","1","1","1","1","1","1","0","1","1","1","1","1","1","1","1","1","1","1","1"],["1","1","1","1","1","1","1","0","0","1","1","1","1","1","1","1","1","1","1","1"],["1","1","1","1","1","1","1","1","0","1","1","1","1","1","1","1","1","1","1","1"],["1","1","1","1","1","1","1","1","0","0","1","1","1","1","1","1","1","1","1","1"],["1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","1","1","1","1","1"],["1","1","1","1","1","1","1","1","1","0","0","0","0","1","1","1","1","1","1","1"],["1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1","1","1","1"],["1","1","1","1","1","1","1","1","1","1","1","1","0","0","0","0","1","1","1","1"],["1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","0","1","1","1","1"],["1","1","1","1","1","1","1","1","1","1","1","1","1","1","1","0","0","0","0","0"]]
*/







///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/open-the-lock/
class Leet0752 {
    func openLock(_ deadends: [String], _ target: String) -> Int {
        let deadends = Set(deadends.map { $0.compactMap { Int(String($0)) } })
        let target = target.compactMap { Int(String($0)) }
        let start = [0,0,0,0]
        var deque = Deque<[Int]>([start]), visited = Set<[Int]>([start])
        var step = 0
                
        while !deque.isEmpty {
            for _ in 0..<deque.count {
                let combo = deque.removeFirst()
                guard !deadends.contains(combo) else { continue }
                guard combo != target else { return step }
                for i in 0...3 {
                    for delta in [-1, 1] {
                        var newCombo = combo
                        newCombo[i] = ((newCombo[i] + delta) + 10) % 10
                        if !visited.contains(newCombo) && !deadends.contains(newCombo) {
                            deque.append(newCombo)
                            visited.insert(newCombo)
                        }
                    }
                }
            }
            step += 1
        }
        return -1
    }
    
    static func test() {
        let sut = Leet0752()
        assert(sut.openLock(["0201","0101","0102","1212","2002"], "0202") == 6)
        assert(sut.openLock(["8888"], "0009") == 1)
        assert(sut.openLock(["8887","8889","8878","8898","8788","8988","7888","9888"], "8888") == -1)
    }
}
//Leet0752.test()



 





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/perfect-squares/
class Leet0279 {
    func numSquares(_ n: Int) -> Int {
        var squares: [Int] = []
        // list all perfect squares with max square root of n
        for i in 1...Int(sqrt(Double(n)).rounded(.down)) {
            squares.append(i * i)
        }
        var visited: Set<Int> = []
        var deque: Deque<(sum: Int, count: Int)> = [(0, 0)]
        while !deque.isEmpty {
            let (sum, count) = deque.removeFirst()
            guard sum < n else { return count }

            for i in 0..<squares.count {
                let newSum = sum + squares[i]
                if newSum <= n, !visited.contains(newSum) {
                    visited.insert(newSum)
                    deque.append((newSum, count + 1))
                }
            }
        }
        return 0
    }
    static func test() {
        let sut = Leet0279()
        assert(sut.numSquares(12) == 3)
        assert(sut.numSquares(13) == 2)
    }
}
//Leet0279.test()





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/daily-temperatures/
class Leet0739 {
    func dailyTemperatures(_ temperatures: [Int]) -> [Int] {
        var result = Array(repeating: 0, count: temperatures.count)
        var stack = [(index: Int, temp: Int)]()
        
        for (i, temp) in temperatures.enumerated() {
            while !stack.isEmpty, let last = stack.last, last.temp < temp {
                if let (j, _) = stack.popLast() {
                    result[j] = i - j
                }
            }
            stack.append((i, temp))
        }
        return result
    }
    static func test() {
        let sut = Leet0739()
        assert(sut.dailyTemperatures([73,74,75,71,69,72,76,73]) == [1,1,4,2,1,1,0,0])
        assert(sut.dailyTemperatures([30,40,50,60]) == [1,1,1,0])
        assert(sut.dailyTemperatures([30,60,90]) == [1,1,0])
    }
}
//Leet0739.test()






/**
 
 [4,1,2]
 [1,3,4,2]
 [2,4]
 [1,2,3,4]
 [1,3,5,2,4]
 [6,5,4,3,2,1,7]
 [137,79,120,122,131]
 [137,79,120,131,122,236]
 
 
 [4,1,2]
 [1,3,4,2]
 [2,4]
 [1,2,3,4]
 [1,5,3]
 [2,1,5,3,7]
 [6,3,1]
 [3,1,6,8,4]
 [5,1]
 [1,5,6,3,7]
 [9,2]
 [2,7,9,5,10]
 [8,12,7]
 [7,8,9,12,11]
 [3,2]
 [4,3,2,1]
 
 */

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/next-greater-element-i/
class Leet0496 {
    func nextGreaterElement(_ nums1: [Int], _ nums2: [Int]) -> [Int] {
        var stack = [Int]()
        var hash: [Int: Int] = [:]
        for n2 in nums2 {
            while !stack.isEmpty, let last = stack.last, last < n2 {
                if let previousLower = stack.popLast() {
                    hash[previousLower] = n2
                }
            }
            stack.append(n2)
        }
        let result = nums1.map { hash[$0] ?? -1 }
        return result
    }
    static func test() {
        let sut = Leet0496()
        assert(sut.nextGreaterElement([4,1,2], [1,3,4,2]) == [-1,3,-1])
        assert(sut.nextGreaterElement([2,4], [1,2,3,4]) == [3,-1])
    }
}
//Leet0496.test()


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/next-greater-element-ii/
class Leet0503 {
    func nextGreaterElements(_ nums: [Int]) -> [Int] {
        var expandedNums = nums + nums
        var stack = [(index: Int, num: Int)]()
        
        for (i, num) in expandedNums.enumerated() {
            while !stack.isEmpty, let last = stack.last, last.num < num {
                if let (lastIndex, _) = stack.popLast() {
                    expandedNums[lastIndex] = num
                }
            }
            stack.append((i, num))
        }
        let result = nums.enumerated().map { $0.element == expandedNums[$0.offset] ? -1 : expandedNums[$0.offset] }
        return result
    }
    static func test() {
        let sut = Leet0503()
        assert(sut.nextGreaterElements([1,2,1]) == [2,-1,2])
        assert(sut.nextGreaterElements([1,2,3,4,3]) == [2,3,4,-1,4])
    }
}
//Leet0503.test()




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/final-prices-with-a-special-discount-in-a-shop/
class Leet1475 {
    func finalPrices(_ prices: [Int]) -> [Int] {
        var result = prices
        var stack = [(index: Int, price: Int)]()
        
        for (i, price) in prices.enumerated() {
            while !stack.isEmpty, let last = stack.last, last.price >= price {
                if let (lastIndex, lastPrice) = stack.popLast() {
                    result[lastIndex] = lastPrice - price
                }
            }
            stack.append((i, price))
        }
        return result
    }
    static func test() {
        let sut = Leet1475()
        assert(sut.finalPrices([8,4,6,2,3]) == [4,2,4,2,3])
        assert(sut.finalPrices([1,2,3,4,5]) == [1,2,3,4,5])
        assert(sut.finalPrices([10,1,1,6]) == [9,0,1,6])
    }
}
//Leet1475.test()







///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/evaluate-reverse-polish-notation/
class Leet0150 {

    enum Operator: String { case plus = "+", minus = "-", multiply = "*", divide = "/" }

    func evalRPN(_ tokens: [String]) -> Int {
        var stack = [Int]()
        for token in tokens {
            if let op = Operator(rawValue: token), let right = stack.popLast(), let left = stack.popLast() {
                var res: Int
                switch op {
                case .plus: res = left + right
                case .minus: res = left - right
                case .multiply: res = left * right
                case .divide: res = left / ( right == 0 ? 1 : right )
                }
                stack.append(res)
            } else if let num = Int(token) {
                stack.append(num)
            }
        }
        return stack.popLast( ) ?? 0
    }
    
    static func test() {
        let sut = Leet0150()
        assert(sut.evalRPN(["2","1","+","3","*"]) == 9)
        assert(sut.evalRPN(["4","13","5","/","+"]) == 6)
        assert(sut.evalRPN(["10","6","9","3","+","-11","*","/","*","17","+","5","+"]) == 22)
    }
}
//Leet0152.test()


public class NodeWithNeighbors {
    public var val: Int
    public var neighbors: [NodeWithNeighbors?]
    public init(_ val: Int) {
        self.val = val
        self.neighbors = []
    }
}

extension NodeWithNeighbors: Equatable {
    public static func == (lhs: NodeWithNeighbors, rhs: NodeWithNeighbors) -> Bool {
        return lhs.val == rhs.val
    }
}

extension NodeWithNeighbors: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(val)
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/clone-graph/
class Leet0133 {
    
    typealias Node = NodeWithNeighbors
    
    var map: [Node: Node] = [:] // [old:new]

    func cloneGraph(_ node: Node?) -> Node? {
        guard let node else { return nil }
        guard map[node] == nil else { return map[node] }
        let clone = Node(node.val)
        map[node] = clone
        clone.neighbors = node.neighbors.map { cloneGraph($0) }
        return clone
    }
    
    static func test() {
        let sut = Leet0133()
        let n4 = Node(4)
        let n3 = Node(3)
        n3.neighbors = [n4]
        let n2 = Node(2)
        n2.neighbors = [n3]
        n3.neighbors.append(n2)
        let n1 = Node(1)
        n1.neighbors = [n2, n4]
        n2.neighbors.append(n1)
        n4.neighbors = [n1, n3]
        let new1 = sut.cloneGraph(n1)
        assert(n1 == new1)
    }
}
//Leet0133.test()






///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/target-sum/
class Leet0494 {
    
    func findTargetSumWays(_ nums: [Int], _ target: Int) -> Int {
        let total = nums.reduce(0, +)
        var memo = Array(repeating: Array(repeating: Int(Int32.min), count: 2 * total + 1), count: nums.count)
        return calculateWays(nums, 0, 0, target, total, &memo)
    }
    
    private func calculateWays(_ nums: [Int], _ currentIndex: Int, _ currentSum: Int, _ target: Int, _ total: Int, _ memo: inout [[Int]] ) -> Int {
        if currentIndex == nums.count {
            return currentSum == target ? 1 : 0
        } else {
            if memo[currentIndex][currentSum + total] != Int(Int32.min) {
                return memo[currentIndex][currentSum + total]
            }
            let add = calculateWays(nums, currentIndex + 1, currentSum + nums[currentIndex], target, total, &memo)
            let subtract = calculateWays(nums, currentIndex + 1, currentSum - nums[currentIndex], target, total, &memo)
            memo[currentIndex][currentSum + total] = add + subtract
            return memo[currentIndex][currentSum + total]
        }
    }

    func dp_findTargetSumWays(_ nums: [Int], _ target: Int) -> Int {
        let total = nums.reduce(0, +)
        var dp = Array(repeating: 0, count: 2 * total + 1)
     
        // Initialize the first row of the DP table
        dp[nums[0] + total] = 1
        dp[-nums[0] + total] += 1

        // Fill the DP table
        for i in 1..<nums.count {
            var next = Array(repeating: 0, count: 2 * total + 1)
            for sum in -total...total where dp[sum + total] > 0 {
                next[sum + nums[i] + total] += dp[sum + total]
                next[sum - nums[i] + total] += dp[sum + total]
            }
            dp = next
        }
        
        // Return the result if the target is within the valid range
        return abs(target) > total ? 0 : dp[target + total]
    }
    static func test() {
        let sut = Leet0494()
        assert(sut.findTargetSumWays([1,1,1,1,1], 3) == 5)
        assert(sut.findTargetSumWays([1], 1) == 1)
    }
}
//Leet0494.test()






///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/binary-tree-inorder-traversal/
class Leet0094 {
    func inorderTraversal(_ root: TreeNode?) -> [Int] {
        guard let root else { return [] }
        var result: [Int] = []
        var stack: [TreeNode] = []
        var current: TreeNode? = root
        while (current != nil || !stack.isEmpty) {
            while (current != nil) {
                stack.append(current!)
                current = current?.left
            }
            current = stack.popLast()
            result.append(current!.val)
            current = current?.right
        }
        return result
    }
}
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/binary-tree-preorder-traversal/
class Leet0144 {
    func preorderTraversal(_ root: TreeNode?) -> [Int] {
        guard let root else { return [] }
        var result: [Int] = []
        var stack: [TreeNode] = [root]

        while (!stack.isEmpty) {
            let node = stack.removeLast()
            result.append(node.val)
            if let right = node.right {
                stack.append(right)
            }
            if let left = node.left {
                stack.append(left)
            }
        }
        return result
    }
}
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/binary-tree-postorder-traversal/
class Leet0145 {
    func postorderTraversal(_ root: TreeNode?) -> [Int] {
        guard let root else { return [] }
        var result: [Int] = []
        var stack: [TreeNode] = []
        var current: TreeNode? = root
        
        while (current != nil || !stack.isEmpty) {
            if let node = current {
                result.append(node.val)
                stack.append(node)
                current = node.right
            } else {
                current = stack.popLast()?.left
            }
        }
        result.reverse()
        return result
    }
}




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/implement-queue-using-stacks/
class MyQueue {
    
    private var s1: [Int]
    private var s2: [Int]
    private var front: Int?
    
    init() {
        s1 = []
        s2 = []
        front = nil
    }
    
    func push(_ x: Int) {
        if s1.isEmpty {
            front = x
        }
        s1.append(x)
    }
    
    func pop() -> Int {
        if s2.isEmpty {
            while !s1.isEmpty {
                s2.append(s1.removeLast())
            }
        }
        return s2.removeLast()
    }
    
    func peek() -> Int {
        if !s2.isEmpty {
            return s2.last!
        }
        return front ?? 0
    }
    
    func empty() -> Bool {
        s1.isEmpty && s2.isEmpty
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/implement-stack-using-queues/
class MyStack {
    private var q1: Deque<Int>

    init() {
        q1 = []
    }
    
    func push(_ x: Int) {
        q1.append(x)
    }
    
    func pop() -> Int {
        q1.removeLast()
    }
    
    func top() -> Int {
        guard !q1.isEmpty else {
            return 0
        }
        return q1.last!
    }
    
    func empty() -> Bool {
        q1.isEmpty
    }
}




extension String {
    var isFirstLetter: Bool {
        guard let first = first else { return false }
        return first.isLetter
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/decode-string/
class Leet0394 {
    func decodeString(_ s: String) -> String {
        var stack = [String]()
        var string = Deque<Character>(s)
        while !string.isEmpty {
            let char = string.removeFirst()
            if char.isNumber {
                // exhaust the number as it could be 111 or 222
                var number = "\(char)"
                while !string.isEmpty, let first = string.first, first.isNumber {
                    number.append(String(first))
                    string.removeFirst()
                }
                stack.append(number)
            } else if char == "[" {
                stack.append(String(char))
            } else if char.isLetter {
                // exhaust the string
                var str = "\(char)"
                while !string.isEmpty, let first = string.first, first.isLetter {
                    str.append(String(first))
                    string.removeFirst()
                }
                // peek at the top (last) of the stack and combine if a string too
                if !stack.isEmpty, let last = stack.last, last.isFirstLetter {
                    var temp = stack.removeLast()
                    temp.append(contentsOf: str)
                    stack.append(temp)
                } else {
                    stack.append(str)
                }
            } else if char == "]" {
                var str = ""
                // start popping!
                while !stack.isEmpty {
                    guard let last = stack.popLast() else { continue }
                    if last == "[" {
                        continue
                    } else if let number = Int(last) {
                        var temp = ""
                        if let last = stack.last, last.isFirstLetter, let top = stack.popLast() {
                            temp.append(top)
                        }
                        for _ in 1...number {
                            temp.append(str)
                        }
                        stack.append(temp)
                        break
                    } else {
                        str = last
                    }
                }
            }
        }
        return stack.removeLast()
    }
    
    static func test() {
        let sut = Leet0394()
        assert(sut.decodeString("2[a2[b]c]") == "abbcabbc")
        assert(sut.decodeString("11[1[a]]") == "aaaaaaaaaaa")
        assert(sut.decodeString("3[a]2[bc]") == "aaabcbc")
        assert(sut.decodeString("3[a2[c]]") == "accaccacc")
        assert(sut.decodeString("2[abc]3[cd]ef") == "abcabccdcdcdef")
        assert(sut.decodeString("10[ab]") == "abababababababababab")
        assert(sut.decodeString("10[ab]c") == "ababababababababababc")
        assert(sut.decodeString("c3[2[a]1[b]]d") == "caabaabaabd")
        assert(sut.decodeString("2[3[a2[c]]d]e") == "accaccaccdaccaccaccde")
    }
}
//Leet0394.test()








///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/basic-calculator/
class Leet0224 {
    private func evaluate(_ stack: inout [Any]) -> Int {
        if stack.isEmpty || (stack.last as? Int) == nil {
            stack.append(0)
        }
        var result = stack.removeLast() as? Int
        while !stack.isEmpty && stack.last as? Character != ")" {
            let sign = stack.removeLast() as? Character
            if sign == "+" {
                result! += (stack.removeLast() as? Int) ?? 0
            } else {
                result! -= (stack.removeLast() as? Int) ?? 0
            }
        }
        return result ?? 0
    }
    
    func calculate(_ s: String) -> Int {
        var operand = 0
        var n = 0
        var stack: [Any] = []
        for char in s.reversed() where !char.isWhitespace {
            if char.isNumber {
                operand += Int(pow(10.0, Double(n))) * Int(String(char))!
                n += 1
            } else {
                if n > 0 {
                    stack.append(operand)
                    operand = 0
                    n = 0
                }
                if char == "(" {
                    let temp = evaluate(&stack)
                    stack.removeLast()
                    stack.append(temp)
                } else {
                    stack.append(char)
                }
            }
        }
        if n > 0 {
            stack.append(operand)
        }
        return evaluate(&stack)
    }
    
    static func test() {
        let sut = Leet0224()
        assert(sut.calculate("1 + 1") == 2)
        assert(sut.calculate(" 2-1 + 2 ") == 3)
        assert(sut.calculate("(1+(4+5+2)-3)+(6+8)") == 23)
        assert(sut.calculate("-2+ 1") == -1)
        assert(sut.calculate("- (3 + (4 + 5))") == -12)
        assert(sut.calculate("2147483647") == 2147483647)
        assert(sut.calculate("-2147483648") == -2147483648)
    }
}
//Leet0224.test()







///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/flood-fill/
class Leet0733 {
    private struct Cell { let row: Int; let col: Int }
    private enum Direction: CaseIterable {
        case up, right, down, left
        var vector: Cell { self == .up ? Cell(row: -1, col: 0) : self == .right ? Cell(row: 0, col: 1) : self == .down ? Cell(row: 1, col: 0) : Cell(row: 0, col: -1) }
    }
    func floodFill(_ image: [[Int]], _ sr: Int, _ sc: Int, _ color: Int) -> [[Int]] {
        let oldColor = image[sr][sc]
        guard oldColor != color else { return image }
        let m = image.count; let n = image[0].count
        var deque: Deque<Cell> = [Cell(row: sr, col: sc)]
        var image = image
        while !deque.isEmpty {
            let cell = deque.removeFirst()
            guard image[cell.row][cell.col] == oldColor else { continue }
            image[cell.row][cell.col] = color
            for direction in Direction.allCases {
                let newRow = cell.row + direction.vector.row
                let newCol = cell.col + direction.vector.col
                let isNewIndexOutOfBounds = (newRow < 0 || newRow >= m) || (newCol < 0 || newCol >= n)
                guard !isNewIndexOutOfBounds, image[newRow][newCol] == oldColor else { continue }
                let newCell = Cell(row: newRow, col: newCol)
                deque.append(newCell)
            }
        }
        return image
    }
    static func test() {
        let sut = Leet0733()
        assert(sut.floodFill([[1,1,1],[1,1,0],[1,0,1]], 1, 1, 2) == [[2,2,2],[2,2,0],[2,0,1]])
        assert(sut.floodFill([[0,0,0],[0,0,0]], 0, 0, 0) == [[0,0,0],[0,0,0]])
    }
}
//Leet0733.test()





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/rotting-oranges/
class Leet0994 {
    private enum State: Int { case empty = 0, fresh, rotten }
    private struct Cell { let row: Int; let col: Int }
    func orangesRotting(_ grid: [[Int]]) -> Int {
        let (m, n) = (grid.count, grid[0].count)
        var (totalEmptyCount, step, visited) = (0, 0, 0)
        var (deque, grid) = (Deque<Cell>(), grid.map { $0.map { State(rawValue: $0)! } })
        for i in 0..<m { // collect all rottens and queue for processing
            for j in 0..<n {
                if grid[i][j] == .rotten {
                    deque.append(Cell(row: i, col: j))
                    visited += 1
                } else if grid[i][j] == .empty {
                    totalEmptyCount += 1
                }
            }
        }
        let totalOrangeCount = m * n - totalEmptyCount
        while !deque.isEmpty && visited < totalOrangeCount {
            for _ in deque {
                let cell = deque.removeFirst()
                guard grid[cell.row][cell.col] == .rotten else { continue }
                for (dx, dy) in [(-1, 0), (0, 1), (1, 0), (0, -1)] {
                    let (nx, ny) = (cell.row + dx, cell.col + dy)
                    let isNewIndexInBounds = (0..<m).contains(nx) && (0..<n).contains(ny)
                    guard isNewIndexInBounds, grid[nx][ny] == .fresh else { continue }
                    grid[nx][ny] = .rotten
                    deque.append(Cell(row: nx, col: ny))
                    visited += 1
                }
            }
            step += 1
        }
        return totalOrangeCount == visited ? step : -1
    }
    static func test() {
        let sut = Leet0994()
        assert(sut.orangesRotting([[2,1,1],[1,1,0],[0,1,1]]) == 4)
        assert(sut.orangesRotting([[2,1,1],[0,1,1],[1,0,1]]) == -1)
        assert(sut.orangesRotting([[0,2]]) == 0)
        assert(sut.orangesRotting([[2,1,1],[1,1,0],[0,1,2]]) == 2)
    }
}
//Leet0994.test()







///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/island-perimeter/
class Leet0463 {
    private enum State: Int { case water = 0, island, visited }
    private struct Cell { let row: Int; let col: Int }
    func islandPerimeter(_ grid: [[Int]]) -> Int {
        let (m, n) = (grid.count, grid[0].count)
        var (deque, perimeter) = (Deque<Cell>(), 0)
        var grid = grid.map { $0.map { State(rawValue: $0)! } }
        // find the first cell of the island to add in the deque
        all: for i in 0..<m {
            for j in 0..<n {
                if grid[i][j] == .island {
                    deque.append(Cell(row: i, col: j))
                    break all
                }
            }
        }
        while !deque.isEmpty {
            let current = deque.removeFirst()
            guard grid[current.row][current.col] == .island else { continue }
            grid[current.row][current.col] = .visited
            for (dx, dy) in [(-1, 0), (0, 1), (1, 0), (0, -1)] {
                let (nx, ny) = (current.row + dx, current.col + dy)
                let isNewIndexInBounds = (0..<m).contains(nx) && (0..<n).contains(ny)
                guard isNewIndexInBounds else {
                    perimeter += 1
                    continue
                }
                if grid[nx][ny] == .water {
                    perimeter += 1
                    continue
                }
                guard grid[nx][ny] == .island else { continue }
                deque.append(Cell(row: nx, col: ny))
            }
        }
        return perimeter
    }
    static func test() {
        let sut = Leet0463()
        assert(sut.islandPerimeter([[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]) == 16)
    }
}
//Leet0463.test()






///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/max-area-of-island/
class Leet0695 {
    private enum State: Int { case water = 0, island, visited }
    private struct Cell { let row: Int; let col: Int }
    private func areaOfIsland(_ grid: inout [[State]], from cell: Cell) -> Int {
        var (area, deque) = (0, Deque<Cell>([cell]))
        let (m, n) = (grid.count, grid[0].count)
        while !deque.isEmpty {
            for _ in deque {
                let current = deque.removeFirst()
                guard grid[current.row][current.col] == .island else { continue }
                grid[current.row][current.col] = .visited
                area += 1
                for (dx, dy) in [(-1, 0), (0, 1), (1, 0), (0, -1)] {
                    let (nx, ny) = (current.row + dx, current.col + dy)
                    let isNewCellInGrid = (0..<m) ~= nx && (0..<n) ~= ny
                    guard isNewCellInGrid, grid[nx][ny] == .island else { continue }
                    deque.append(Cell(row: nx, col: ny))
                }
            }
        }
        return area
    }
    func maxAreaOfIsland(_ grid: [[Int]]) -> Int {
        var (maxArea, grid) = (0, grid.map { $0.map { State(rawValue: $0)! } })
        let (m, n) = (grid.count, grid[0].count)
        for i in 0..<m {
            for j in 0..<n where grid[i][j] == .island {
                let area = areaOfIsland(&grid, from: Cell(row: i, col: j))
                maxArea = max(maxArea, area)
            }
        }
        return maxArea
    }
    static func test() {
        let sut = Leet0695()
        assert(sut.maxAreaOfIsland([[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]) == 6)
        assert(sut.maxAreaOfIsland([[0,0,0,0,0,0,0,0]]) == 0)
        assert(sut.maxAreaOfIsland([[1,1,0,0,0],[1,1,0,0,0],[0,0,0,1,1],[0,0,0,1,1]]) == 4)
    }
}
//Leet0695.test()





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/battleships-in-a-board/
class Leet0419 {
    private enum State: Character { case empty = "."; case ship = "X"; case visited = "#" }
    private struct Cell { let row: Int; let col: Int }
    private enum Direction: CaseIterable {
        case up, right, down, left
        var vector: Cell { self == .up ? Cell(row: -1, col: 0) : self == .right ? Cell(row: 0, col: 1) : self == .down ? Cell(row: 1, col: 0) : Cell(row: 0, col: -1) }
    }
    private func dfs(_ board: inout [[State]], _ cell: Cell) {
        var stack: [Cell] = [cell]
        let m = board.count; let n = board[0].count
        while !stack.isEmpty {
            let current = stack.removeLast()
            guard  board[current.row][current.col] == .ship else { continue }
            board[current.row][current.col] = .visited
            for direction in Direction.allCases {
                let newRow = current.row + direction.vector.row
                let newCol = current.col + direction.vector.col
                let isNewIndexOutOfBounds = (newRow < 0 || newRow >= m || newCol < 0 || newCol >= n)
                guard !isNewIndexOutOfBounds else { continue }
                guard board[newRow][newCol] == .ship else { continue }
                stack.append(Cell(row: newRow, col: newCol))
            }
        }
    }
    func countBattleships(_ board: [[Character]]) -> Int {
        var count = 0; var board = board.map { row in row.map { State(rawValue: $0)! } }
        let m = board.count; let n = board[0].count
        for i in 0..<m {
            for j in 0..<n where board[i][j] == .ship {
                dfs(&board, Cell(row: i, col: j))
                count += 1
            }
        }
        return count
    }
    static func test() {
        let sut = Leet0419()
        assert(sut.countBattleships([["X",".",".","X"],[".",".",".","X"],[".",".",".","X"]]) == 2)
        assert(sut.countBattleships([["."]]) == 0)
    }
}
//Leet0419.test()





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-number-of-fish-in-a-grid/
class Leet2658 {
    private struct Cell { let row: Int; let col: Int }
    private enum Direction: CaseIterable {
        case up, right, down, left
        var vector: Cell { self == .up ? Cell(row: -1, col: 0) : self == .right ? Cell(row: 0, col: 1) : self == .down ? Cell(row: 1, col: 0) : Cell(row: 0, col: -1) }
    }
    private func catchFish(in grid: inout [[Int]], at cell: Cell) -> Int {
        var queue: [Cell] = [cell]; var (frontIndex, count) = (0, 0)
        let (x, y) = (grid.count, grid[0].count)
        while frontIndex < queue.count {
            let (i, j) = (queue[frontIndex].row, queue[frontIndex].col)
            frontIndex += 1
            guard  grid[i][j] > 0 else { continue }
            count += grid[i][j]
            grid[i][j] = 0
            for d in Direction.allCases {
                let (ni, nj) = (i + d.vector.row, j + d.vector.col)
                let isNewIndexOutOfBounds = (ni < 0 || ni >= x) || (nj < 0 || nj >= y)
                guard !isNewIndexOutOfBounds else { continue }
                guard grid[ni][nj] > 0 else { continue }
                queue.append(Cell(row: ni, col: nj))
            }
        }
        return count
    }
    func findMaxFish(_ grid: [[Int]]) -> Int {
        var (grid, maxCount) = (grid, 0)
        let (m, n) = (grid.count, grid[0].count)
        for i in 0..<m {
            for j in 0..<n where grid[i][j] > 0 {
                maxCount = max(maxCount, catchFish(in: &grid, at: Cell(row: i, col: j)))
            }
        }
        return maxCount
    }
    static func test() {
        let sut = Leet2658()
        assert(sut.findMaxFish([[0,2,1,0],[4,0,0,3],[1,0,0,4],[0,3,2,0]]) == 7)
        assert(sut.findMaxFish([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]) == 1)
    }
}
//Leet2658.test()





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/01-matrix/
class Leet0542 {
    private struct Cell: Hashable { let row: Int; let col: Int }
    private struct State { let cell: Cell; let steps: Int }
    func updateMatrix(_ mat: [[Int]]) -> [[Int]] {
        var (result, visited, deque) = (mat, Set<Cell>(), Deque<State>())
        let (m, n) = (result.count, result[0].count)
        for i in 0..<m {
            for j in 0..<n where result[i][j] == 0 {
                let cell = Cell(row: i, col: j)
                let state = State(cell: cell, steps: 0)
                visited.insert(cell)
                deque.append(state)
            }
        }
        while !deque.isEmpty {
            let current = deque.removeFirst()
            result[current.cell.row][current.cell.col] = current.steps
            for (dx, dy) in [(-1, 0), (0, 1), (1, 0), (0, -1)] {
                let (nx, ny) = (current.cell.row + dx, current.cell.col + dy)
                let isNewIndexOutOfBounds = nx < 0 || nx >= m || ny < 0 || ny >= n
                guard !isNewIndexOutOfBounds else { continue }
                let newCell = Cell(row: nx, col: ny)
                if !visited.contains(newCell) {
                    visited.insert(newCell)
                    let newState = State(cell: newCell, steps: current.steps + 1)
                    deque.append(newState)
                }
            }
        }
        return result
    }
    
    static func test() {
        let sut = Leet0542()
        assert(sut.updateMatrix([[0,0,0],[0,1,0],[0,0,0]]) == [[0,0,0],[0,1,0],[0,0,0]])
        assert(sut.self .updateMatrix([[0,0,0],[0,1,0],[1,1,1]]) == [[0,0,0],[0,1,0],[1,2,1]])
        assert(sut.updateMatrix([[1,1,1,1,0]]) == [[4,3,2,1,0]])
    }
}
//Leet0542.test()







///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/map-of-highest-peak/
class Leet1765 {
    private struct Cell: Hashable { let row: Int; let col: Int }
    private struct State { let cell: Cell; let steps: Int }
    func highestPeak(_ isWater: [[Int]]) -> [[Int]] {
        var result = isWater.map { $0.map { $0 == 1 ? 0 : 1 } }
        var (visited, deque) = (Set<Cell>(), Deque<State>())
        let (m, n) = (result.count, result[0].count)
        for i in 0..<m {
            for j in 0..<n where result[i][j] == 0 {
                let cell = Cell(row: i, col: j)
                let state = State(cell: cell, steps: 0)
                visited.insert(cell)
                deque.append(state)
            }
        }
        while !deque.isEmpty {
            let current = deque.removeFirst()
            result[current.cell.row][current.cell.col] = current.steps
            for (dx, dy) in [(-1, 0), (0, 1), (1, 0), (0, -1)] {
                let (nx, ny) = (current.cell.row + dx, current.cell.col + dy)
                let isNewIndexOutOfBounds = nx < 0 || nx >= m || ny < 0 || ny >= n
                guard !isNewIndexOutOfBounds else { continue }
                let newCell = Cell(row: nx, col: ny)
                if !visited.contains(newCell) {
                    visited.insert(newCell)
                    let newState = State(cell: newCell, steps: current.steps + 1)
                    deque.append(newState)
                }
            }
        }
        return result
    }
    static func test() {
        let sut = Leet1765()
        assert(sut.highestPeak([[0,1],[0,0]]) == [[1,0],[2,1]])
        assert(sut.highestPeak([[0,0,1],[1,0,0],[0,0,0]]) ==  [[1,1,0],[0,1,1],[1,2,2]])
    }
}
//Leet1765.test()





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/number-of-closed-islands/
class Leet1254 {
    private struct Cell { let row: Int; let col: Int }
    private enum State: Int { case island = 0, water = 1, visited = 2 }
    private func addendClosedIsland(from cell: Cell, _ grid: inout [[State]]) -> Int {
        // look for the edges breadth first and if touching boundary of the grid, return 0
        var (deque, isTouchingEdge) = (Deque<Cell>([cell]), false)
        let (m, n) = (grid.count, grid[0].count)
        while !deque.isEmpty {
            let current = deque.removeFirst()
            guard grid[current.row][current.col] == .island else { continue }
            grid[current.row][current.col] = .visited
            
            for (dx, dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
                let (nextRow, nextCol) = (current.row + dx, current.col + dy)
                let isNextCellInGrid = nextRow >= 0 && nextRow < m && nextCol >= 0 && nextCol < n
                guard isNextCellInGrid else {
                    isTouchingEdge = true
                    continue
                }
                deque.append(.init(row: nextRow, col: nextCol))
            }
        }
        return isTouchingEdge ? 0 : 1
    }
    func closedIsland(_ grid: [[Int]]) -> Int {
        let (m, n) = (grid.count, grid[0].count)
        var (count, grid) = (0, grid.map { $0.map { State(rawValue: $0)! } } )
        for i in 0..<m {
            for j in 0..<n where grid[i][j] == .island {
                count += addendClosedIsland(from: .init(row: i, col: j), &grid)
            }
        }
        return count
    }
    static func test() {
        let sut = Leet1254()
        assert(sut.closedIsland([[1,1,1,1,1,1,1,0],[1,0,0,0,0,1,1,0],[1,0,1,0,1,1,1,0],[1,0,0,0,0,1,0,1],[1,1,1,1,1,1,1,0]]) == 2)
        assert(sut.closedIsland([[0,0,1,0,0],[0,1,0,1,0],[0,1,1,1,0]]) == 1)
        assert(sut.closedIsland([[1,1,1,1,1,1,1],
                                 [1,0,0,0,0,0,1],
                                 [1,0,1,1,1,0,1],
                                 [1,0,1,0,1,0,1],
                                 [1,0,1,1,1,0,1],
                                 [1,0,0,0,0,0,1],
                                 [1,1,1,1,1,1,1]]) == 2)
        
    }
}
//Leet1254.test()



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/surrounded-regions/
class Leet0130 {
    private struct Cell { let row: Int; let col: Int }
    private func surround(_ cell: Cell, in board: [[Character]]) -> [[Character]] {
        let (m, n) = (board.count, board[0].count)
        var (result, deque, isTouchingEdge) = (board, Deque<Cell>([cell]), false)
        while !deque.isEmpty {
            let current = deque.removeFirst()
            guard result[current.row][current.col] == "O" else { continue }
            result[current.row][current.col] = "X"
            
            for (dx, dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
                let (nx, ny) = (current.row + dx, current.col + dy)
                let isNewCellInGrid = (0..<m) ~= nx && (0..<n) ~= ny
                guard isNewCellInGrid else {
                    isTouchingEdge = true
                    continue
                }
                deque.append(.init(row: nx, col: ny))
            }
        }
        return isTouchingEdge ? board : result
    }
    func solve(_ board: inout [[Character]]) {
        let (m, n) = (board.count, board[0].count)
        for i in 0..<m {
            for j in 0..<n where board[i][j] == "O" {
                board = surround(.init(row: i, col: j), in: board)
            }
        }
    }
    static func test() {
        let sut = Leet0130()
        var board: [[Character]] = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
        sut.solve(&board)
        assert(board == [["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]])
        
        board = [["X"]]
        sut.solve(&board)
        assert(board == [["X"]])
    }
}
//Leet0130.test()





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/number-of-enclaves/
class Leet1020 {
    private struct Cell { let row: Int; let col: Int }
    private enum State: Int { case visited = -1, water, land }
    private func addendEnclaves(from cell: Cell, _ grid: inout [[State]]) -> Int {
        var (count, deque, isTouchingEdge) = (0, Deque<Cell>([cell]), false)
        let (m, n) = (grid.count, grid[0].count)
        while !deque.isEmpty {
            let current = deque.removeFirst()
            guard grid[current.row][current.col] == .land else { continue }
            count += 1
            grid[current.row][current.col] = .visited
            for (dx, dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
                let (nx, ny) = (current.row + dx, current.col + dy)
                let isNewCellInGrid = (0..<m) ~= nx && (0..<n) ~= ny
                guard isNewCellInGrid else {
                    isTouchingEdge = true
                    continue
                }
                deque.append(.init(row: nx, col: ny))
            }
        }
        return isTouchingEdge ? 0 : count
    }
    func numEnclaves(_ grid: [[Int]]) -> Int {
        let (m, n) = (grid.count, grid[0].count)
        var (count, grid) = (0, grid.map { $0.map { State(rawValue: $0)! }} )
        for i in 0..<m {
            for j in 0..<n where grid[i][j] == .land {
                count += addendEnclaves(from: .init(row: i, col: j), &grid)
            }
        }
        return count
    }
    static func test() {
        let sut = Leet1020()
        assert(sut.numEnclaves([[0,0,0,0],[1,0,1,0],[0,1,1,0],[0,0,0,0]]) == 3)
        assert(sut.numEnclaves([[0,1,1,0],[0,0,1,0],[0,0,1,0],[0,0,0,0]]) == 0)
    }
}
//Leet1020.test()




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/coloring-a-border/
class Leet1034 {
    private struct Cell: Hashable { let row: Int; let col: Int }
    func colorBorder(_ grid: [[Int]], _ row: Int, _ col: Int, _ color: Int) -> [[Int]] {
        let oldColor = grid[row][col]
        guard oldColor != color else { return grid }
        let (m, n) = (grid.count, grid[0].count)
        var (result, deque, seen) = (grid, Deque<Cell>([Cell(row: row, col: col)]), Set<Cell>())
        
        while !deque.isEmpty {
            let cell = deque.removeFirst()
            guard grid[cell.row][cell.col] == oldColor, !seen.contains(cell) else { continue }
            seen.insert(cell)
            var oldColorCount = 0
            for (dx, dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)] {
                let (nx, ny) = (cell.row + dx, cell.col + dy)
                let newCell = Cell(row: nx, col: ny)
                let isNewCellInGrid = (0..<m) ~= nx && (0..<n) ~= ny
                guard isNewCellInGrid, oldColor == grid[nx][ny] else { continue }
                deque.append(newCell)
                oldColorCount += 1
            }
            guard oldColorCount < 4 else { continue }
            result[cell.row][cell.col] = color
        }
        return result
    }
    static func test() {
        let sut = Leet1034()
        assert(sut.colorBorder([[1,1],[1,2]], 0, 0, 3) == [[3,3],[3,2]])
        assert(sut.colorBorder([[1,2,2],[2,3,2]], 0, 1, 3) == [[1,3,3],[2,3,3]])
        assert(sut.colorBorder([[1,1,1],[1,1,1],[1,1,1]], 1, 1, 2) == [[2,2,2],[2,1,2],[2,2,2]])
        assert(sut.colorBorder([[1,2,1,2,1,2],[2,2,2,2,1,2],[1,2,2,2,1,2]], 1, 3, 1) == [[1,1,1,1,1,2],[1,2,1,1,1,2],[1,1,1,1,1,2]])
    }
}
//Leet1034.test()





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/keys-and-rooms/
class Leet0841 {
    func canVisitAllRooms(_ rooms: [[Int]]) -> Bool {
        var visited: Set<Int> = []
        var stack: [Int] = [0]
        while !stack.isEmpty {
            let roomIndex = stack.removeLast()
            guard !visited.contains(roomIndex) else { continue }
            visited.insert(roomIndex)
            for nextRoomIndex in rooms[roomIndex] where !visited.contains(nextRoomIndex) {
                stack.append(nextRoomIndex)
            }
        }
        return visited.count == rooms.count
    }
    static func test() {
        let sut = Leet0841()
        assert(sut.canVisitAllRooms([[1],[2],[3],[]]))
        assert(!sut.canVisitAllRooms([[1,3],[3,0,1],[2],[0]]))
        assert(sut.canVisitAllRooms([[2],[],[1]]))
        assert(sut.canVisitAllRooms([ [1, 3] , [1, 4] , [2, 3, 4, 1] , [ ] , [4, 3, 2] ]))
        assert(!sut.canVisitAllRooms([[1], [ ], [ 0,3 ], [ 1 ]]))
        assert(!sut.canVisitAllRooms([[],[1,15,18],[16],[2,3,9,11,17,5],[15,19,8,12,14],[10,1,6],[12,9,11],[],[7],[13],[3],[16,2],[4],[18,13],[7,17],[6],[14,4],[5],[8,19],[10]]))
        assert(!sut.canVisitAllRooms([[1],[],[0,3],[1]]))
        assert(sut.canVisitAllRooms([[1,3],[1,4],[2,3,4,1],[],[4,3,2]]))
    }
}
//Leet0841.test()




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/word-search
class Leet0079 {
    func exist(_ board: [[Character]], _ word: String) -> Bool {
        let (rows, cols) = (board.count, board[0].count)
        var board = board
        func backtrack(row: Int, col: Int, suffix: String) -> Bool {
            guard suffix.count > 0 else { return true }
            let isIndexInGrid = (0..<rows) ~= row && (0..<cols) ~= col
            guard isIndexInGrid, board[row][col] == suffix.first else { return false }
            let temp = board[row][col]
            board[row][col] = "#"
            let directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for (dx, dy) in directions where backtrack(row: row + dx, col: col + dy, suffix: String(suffix[suffix.index(after: suffix.startIndex)...])) {
                return true
            }
            board[row][col] = temp
            return false
        }
        for row in 0..<rows {
            for col in 0..<cols where backtrack(row: row, col: col, suffix: word) {
                return true
            }
        }
        return false
    }
    static func test() {
        let sut = Leet0079()
        assert(sut.exist([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "ABCCED"))
        assert(sut.exist([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "SEE"))
        assert(!sut.exist([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], "ABCB"))
    }
}
//Leet0079.test()










///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/majority-element/
class Leet0169 {
    func majorityElement(_ nums: [Int]) -> Int {
        var count = 0, candidate = nums[0]
        for num in nums {
            if count == 0 {
                candidate = num
            }
            count += (candidate == num) ? 1 : -1
        }
        return candidate
    }
    static func test() {
        let sut = Leet0169()
        assert(sut.majorityElement([3,2,3]) == 3)
        assert(sut.majorityElement([2,2,1,1,1,2,2]) == 2)
        assert(sut.majorityElement([3,4,2,4,4,2,4]) == 4)
    }
}
//Leet0169.test()








///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/check-if-a-number-is-majority-element-in-a-sorted-array/
class Leet1150 {
    func isMajorityElement(_ nums: [Int], _ target: Int) -> Bool {
        nums.filter { $0 == target }.count > nums.count / 2
    }
    static func test() {
        let sut = Leet1150()
        assert(sut.isMajorityElement([2,4,5,5,5,5,5,6,6], 5) )
        assert(sut.isMajorityElement([10,100,101,101], 101) == false)
    }
}
//Leet1150.test()




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/most-frequent-even-element/
class Leet2404 {
    func mostFrequentEven(_ nums: [Int]) -> Int {
        var freq: [Int: Int] = [:]
        for num in nums where num % 2 == 0 {
            freq[num, default: 0] += 1
        }
        return freq.sorted { $0.value == $1.value ? $0.key < $1.key : $0.value > $1.value }.first?.key ?? -1
    }
    
    static func test() {
        let sut = Leet2404()
        assert(sut.mostFrequentEven([0,1,2,2,4,4,1]) == 2)
        assert(sut.mostFrequentEven([4,4,4,9,2,4]) == 4)
        assert(sut.mostFrequentEven([29,47,21,41,13,37,25,7]) == -1)
        assert(sut.mostFrequentEven([8154,9139,8194,3346,5450,9190,133,8239,4606,8671,8412,6290]) == 3346)
    }
}
//Leet2404.test()




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-index-of-a-valid-split/
class Leet2780 {
    func minimumIndex(_ nums: [Int]) -> Int {
        // get the majority element first
        let dominant = nums.sorted()[nums.count/2]
        var dominantCount = 0, index = -1
                
        // then find the minimum valid split index
        for (i, n) in nums.enumerated() where n == dominant {
            dominantCount += 1
            if dominantCount * 2 > i + 1 {
                index = i
                break
            }
        }
        
        // check the remainder of the array
        guard index > -1, index < nums.count - 1 else { return -1 }
        dominantCount = nums[index+1..<nums.count].count(where: { $0 == dominant })
        guard dominantCount * 2 > nums.count - index - 1 else { return -1 }
        return index
    }
    
    static func test() {
        let sut = Leet2780()
        assert(sut.minimumIndex([1,2,2,2]) == 2)
        assert(sut.minimumIndex([2,1,3,1,1,1,7,1,2,1]) == 4)
        assert(sut.minimumIndex([3,3,3,3,7,2,2]) == -1)
    }
}
//Leet2780.test()





import HeapModule
///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-number-of-points-from-grid-queries/
///
class Leet2503 {
    private struct Cell: Hashable { let x: Int; let y: Int }
    private struct Node: Comparable {
        let cell: Cell; let value: Int
        static func < (lhs: Node, rhs: Node) -> Bool { lhs.value < rhs.value }
    }
    func maxPoints(_ grid: [[Int]], _ queries: [Int]) -> [Int] {
        let (m, n, first) = (grid.count, grid[0].count, Cell(x: 0, y: 0))
        let directions: [(x: Int, y: Int)] = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        let firstNode = Node(cell: first, value: grid[0][0]); var heap = Heap<Node>([firstNode])
        var (visited, answer, points) = (Set([first]), queries.map { _ in 0 }, 0)
        let sortedQueries = queries.enumerated().sorted { $0.1 < $1.1 }
        for query in sortedQueries {
            while !heap.isEmpty, let min = heap.min, min.value < query.element {
                heap.removeMin()
                points += 1
                for (dx, dy) in directions {
                    let (nx, ny) = (min.cell.x + dx, min.cell.y + dy)
                    let newCell = Cell(x: nx, y: ny)
                    let isNewCellInGrid = 0..<m ~= nx && 0..<n ~= ny
                    guard isNewCellInGrid, !visited.contains(newCell) else { continue }
                    let newNode = Node(cell: newCell, value: grid[nx][ny])
                    visited.insert(newCell)
                    heap.insert(newNode)
                }
            }
            answer[query.offset] = points
        }
        return answer
    }
    static func test() {
        let sut = Leet2503()
        assert(sut.maxPoints([[1,2,3],[2,5,7],[3,5,1]], [5,6,2]) == [5,8,1])
        assert(sut.maxPoints([[5,2,1],[1,1,2]], [3]) == [0])
    }
}
//Leet2503.test()




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/k-radius-subarray-averages/
class Leet2090 {
    func getAverages(_ nums: [Int], _ k: Int) -> [Int] {
        var result = nums.map { _ in -1 }; let diameter = k * 2 + 1
        guard nums.count >= diameter else { return result }
        var windowSum = nums[0..<diameter].reduce(0, +)
        result[k] = windowSum / diameter
        for i in k+1..<nums.count - k {
            windowSum += nums[i + k] - nums[i - k - 1]
            result[i] = windowSum / diameter
        }
        return result
    }
    static func test() {
        let sut = Leet2090()
        assert(sut.getAverages([7,4,3,9,1,8,5,2,6], 3) == [-1,-1,-1,5,4,4,-1,-1,-1])
        assert(sut.getAverages([100000], 0) == [100000])
        assert(sut.getAverages([8], 100000) == [-1])
    }
}
//Leet2090.test()







///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/number-of-sub-arrays-of-size-k-and-average-greater-than-or-equal-to-threshold/
class Leet1343 {
    func numOfSubarrays(_ arr: [Int], _ k: Int, _ threshold: Int) -> Int {
        var count = 0, windowSum = arr[0..<k].reduce(0, +)
        count += (windowSum >= threshold * k) ? 1 : 0
        for i in k ..< arr.count {
            windowSum += arr[i] - arr[i - k]
            count += (windowSum >= threshold * k) ? 1 : 0
        }
        return count
    }
    
    static func test() {
        let sut = Leet1343()
        assert(sut.numOfSubarrays([2,2,2,2,5,5,5,8], 3, 4) == 3)
        assert(sut.numOfSubarrays([11,13,17,23,29,31,7,5,2,3], 3, 5) == 6)
    }
}
//Leet1343.test()





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/apply-operations-to-maximize-score/d
class Leet2818 {
        
    private let mod = 1_000_000_007
    
    func maximumScore(_ nums: [Int], _ k: Int) -> Int {
        let n = nums.count, maxElement = nums.max() ?? 0, primes = getPrimes(maxElement)
        
        // calculate prime scores
        var primeScores = [Int](repeating: 0, count: n)
        for index in 0 ..< n {
            var num = nums[index]
            for prime in primes {
                if prime * prime > num { break }
                if num % prime == 0 {
                    primeScores[index] += 1
                    while num % prime == 0 {
                        num /= prime
                    }
                }
            }

            if num > 1 {
                primeScores[index] += 1
            }
        }

        // initialize next and previous dominant index arrays
        var nextDominants = Array(repeating: n, count: n)
        var prevDominants = Array(repeating: -1, count: n)
        var decreasingPrimeScoreStack = [Int]()

        // Calculate the next and previous dominant indices for each number
        for index in 0 ..< n {
            while !decreasingPrimeScoreStack.isEmpty, let last = decreasingPrimeScoreStack.last,
                    primeScores[last] < primeScores[index] {
                
                let topIndex = decreasingPrimeScoreStack.removeLast()
                nextDominants[topIndex] = index
            }
            if !decreasingPrimeScoreStack.isEmpty, let peek = decreasingPrimeScoreStack.last {
                prevDominants[index] = peek
            }
            decreasingPrimeScoreStack.append(index)
        }
        
        // Calculate the number of subarrays in which each element is dominant
        var numOfSubArrays = Array(repeating: 0, count: n)
        for index in 0 ..< n {
            numOfSubArrays[index] = (nextDominants[index] - index) * (index - prevDominants[index])
        }
        
        // Sort elements in decreasing order based on their values
        let sortedArray = nums.enumerated().sorted { $0.element > $1.element }
        var score = 1, processingIndex = 0, k = k
        
        // Process elements while there are operations left
        while (k > 0) {
            let num = sortedArray[processingIndex].element
            let index = sortedArray[processingIndex].offset
            processingIndex += 1
            
            let operations = min(k, numOfSubArrays[index])
            score = (score * power(num, operations)) % mod
            
            k -= operations
        }
        
        return score
    }
    
    private func power(_ base: Int, _ exponent: Int) -> Int {
        var result = 1, base = base, exponent = exponent
        while exponent > 0 {
            if exponent % 2 == 1 {
                result = (result * base) % mod
            }
            base = (base * base) % mod
            exponent /= 2
        }
        return result
    }
    
    private func getPrimes(_ limit: Int) -> [Int] {
        let upper = limit + 1
        var isPrime = Array(repeating: true, count: upper)
        isPrime[0] = false
        isPrime[1] = false
        var primes: [Int] = []
        for number in 2 ..< upper where isPrime[number] {
            primes.append(number)
            for multiple in stride(from: number * number, to: limit + 1, by: number) {
                isPrime[multiple] = false
            }
        }
        return primes
    }
        
    static func test() {
        let sut = Leet2818()
        assert(sut.maximumScore([8,3,9,3,8],2) == 81)
        assert(sut.maximumScore([19,12,14,6,10,18],3) == 4788)
    }
    
}
//Leet2818.test()





extension ClosedRange where Bound: Comparable {
    func merge(_ other: ClosedRange<Bound>) -> ClosedRange<Bound> {
        Swift.min(lowerBound, other.lowerBound)...Swift.max(upperBound, other.upperBound)
    }
}

extension Range where Bound: Comparable {
    func merge(_ other: Range<Bound>) -> Range<Bound> {
        Swift.min(lowerBound, other.lowerBound)..<Swift.max(upperBound, other.upperBound)
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/merge-intervals/
class Leet0056 {
    func merge(_ intervals: [[Int]]) -> [[Int]] {
        let sortedRanges = intervals.sorted { $0[0] < $1[0] || ($0[0] == $1[0] && $0[1] < $1[1]) }.map { $0[0]...$0[1] }
        guard let firstRange = sortedRanges.first else { return [] }
        var resultRanges = [ClosedRange<Int>]([firstRange])
        for i in 1..<sortedRanges.count {
            guard let lastRange = resultRanges.last else { continue }
            let currentRange = sortedRanges[i]
            if currentRange.overlaps(lastRange) {
                let merged = lastRange.merge(currentRange)
                resultRanges[resultRanges.count - 1] = merged
            } else {
                resultRanges.append(currentRange)
            }
        }
        return resultRanges.map { [$0.lowerBound, $0.upperBound] }
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/partition-labels/
class Leet0763 {
    func partitionLabels(_ s: String) -> [Int] {
        let s = Array(s)
        var lowerBounds: [Character: Int] = [:], upperBounds: [Character: Int] = [:], letters = [Character]()
        // fill the lower bounds and upper bounds dictionary of the string
        for char in s.enumerated() where lowerBounds[char.element] == nil {
            lowerBounds[char.element] = char.offset
            letters.append(char.element)
        }
        for char in s.enumerated().reversed() where upperBounds[char.element] == nil {
            upperBounds[char.element] = char.offset
        }
        guard letters.count > 1 else { return [s.count] }
        var resultRanges = [ClosedRange<Int>]([lowerBounds[letters[0]]!...upperBounds[letters[0]]!])
        for i in 1..<letters.count {
            let currChar = letters[i], currRange = lowerBounds[currChar]!...upperBounds[currChar]!
            guard let lastRange = resultRanges.last else { continue }
            if currRange.overlaps(lastRange) {
                let merged = lastRange.merge(currRange)
                resultRanges[resultRanges.count-1] = merged
            } else {
                resultRanges.append(currRange)
            }
        }
        return resultRanges.map(\.count)
    }
    static func test() {
        let sut = Leet0763()
        assert(sut.partitionLabels("ababcbacadefegdehijhklij") == [9,7,8])
        assert(sut.partitionLabels("eccbbbbdec") == [10])
        assert(sut.partitionLabels("qiejxqfnqceocmy") == [13,1,1])
        assert(sut.partitionLabels("caedbdedda") == [1,9])
    }
}
//Leet0763.test()


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/insert-interval/
class Leet0057 {
        
    func insert(_ intervals: [[Int]], _ newInterval: [Int]) -> [[Int]] {
        let intervals = intervals.map { $0[0]...$0[1] }
        var i = 0, resultRanges = [ClosedRange<Int>](), newRange = newInterval[0]...newInterval[1]

        while i < intervals.count, intervals[i].upperBound < newRange.lowerBound {
            resultRanges.append(intervals[i])
            i += 1
        }
        
        while i < intervals.count, intervals[i].overlaps(newRange) {
            newRange = newRange.merge(intervals[i])
            i += 1
        }
        resultRanges.append(newRange)
        
        while i < intervals.count {
            resultRanges.append(intervals[i])
            i += 1
        }
        
        return resultRanges.map { [$0.lowerBound, $0.upperBound] }
    }
    
        
    /*
    func insert(_ intervals: [[Int]], _ newInterval: [Int]) -> [[Int]] {
        guard !intervals.isEmpty else { return [newInterval] }
        let intervals = intervals.map { $0[0]...$0[1] }, newRange = newInterval[0]...newInterval[1]
        var unmerged = [ClosedRange<Int>]()
        
        // insert the new range without merging
        if let first = intervals.first, newRange.lowerBound <= first.lowerBound {
            unmerged.append(newRange)
            unmerged.append(contentsOf: intervals)
        } else if let last = intervals.last, last.lowerBound <= newRange.lowerBound {
            unmerged.append(contentsOf: intervals)
            unmerged.append(newRange)
        } else if intervals.count == 1, let first = intervals.first, first.overlaps(newRange) {
            unmerged.append(first)
            unmerged.append(newRange)
        } else {
            for i in 1..<intervals.count {
                let prev = intervals[i-1], curr = intervals[i]
                if prev.lowerBound...curr.lowerBound ~= newRange.lowerBound {
                    unmerged.append(contentsOf: intervals[0..<i])
                    unmerged.append(newRange)
                    unmerged.append(contentsOf: intervals[i...])
                    break
                }
            }
        }
        
        // 56. merge intervals
        var resultRanges = [ClosedRange<Int>]([unmerged[0]])
        for i in 1..<unmerged.count {
            guard let lastRange = resultRanges.last else { continue }
            let currentRange = unmerged[i]
            if currentRange.overlaps(lastRange) {
                let merged = lastRange.merge(currentRange)
                resultRanges[resultRanges.count - 1] = merged
            } else {
                resultRanges.append(currentRange)
            }
        }
        return resultRanges.map { [$0.lowerBound, $0.upperBound]}
    }
     */
    


    static func test() {
        let sut = Leet0057()
        assert(sut.insert([[2,4],[5,7],[8,10],[11,13]], [3,6]) == [[2,7],[8,10],[11,13]])
        assert(sut.insert([[0,0],[2,4],[9,9]], [0,7]) == [[0,7],[9,9]])
        assert(sut.insert([[0,10],[14,14],[15,20]], [11,11]) == [[0,10],[11,11],[14,14],[15,20]])
        assert(sut.insert([[1,2],[5,6]], [2,5]) == [[1,6]])
        assert(sut.insert([[0,3]], [2,5]) == [[0,5]])
        assert(sut.insert([[2,6],[7,9]], [15,18]) == [[2,6],[7,9],[15,18]])
        assert(sut.insert([[2,3],[5,7]], [0,6]) == [[0,7]] )
    }
}
//Leet0057.test()






///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/put-marbles-in-bags
class Leet2551 {
    func putMarbles(_ weights: [Int], _ k: Int) -> Int {
        let n = weights.count
        var pairWeights = [Int]()
        for i in 0..<n-1 {
            pairWeights.append(weights[i] + weights[i+1])
        }
        pairWeights.sort()
        
        let minScore = pairWeights.prefix(k - 1).reduce(0, +)
        let maxScore = pairWeights.suffix(k - 1).reduce(0, +)

        return maxScore - minScore
    }
    static func test() {
        let sut = Leet2551()
        assert(sut.putMarbles([24,16,62,27,8,3,70,55,13,34,9,29,10], 11) == 168)
        assert(sut.putMarbles([1,4,2,5,2], 3) == 3)
        assert(sut.putMarbles([1,3,5,1], 2) == 4)
        assert(sut.putMarbles([1,3], 2) == 0)
        assert(sut.putMarbles([1,3,5,1], 1) == 0)
    }
}
//Leet2551.test()




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/meeting-rooms/
class Leet0252 {
    func canAttendMeetings(_ intervals: [[Int]]) -> Bool {
        guard !intervals.isEmpty else { return true }
        let ranges = intervals.map { $0[0]..<$0[1] }.sorted { $0.lowerBound < $1.lowerBound }
        for i in 1..<ranges.count where ranges[i].overlaps(ranges[i-1]) {
            return false
        }
        return true
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/meeting-rooms-ii/
class Leet0253 {
    func minMeetingRooms(_ intervals: [[Int]]) -> Int {
        let ranges = intervals.map { $0[0]..<$0[1] }.sorted { $0.lowerBound < $1.lowerBound }
        var maxRooms = 1, heapsort: Heap<Int> = [ranges[0].upperBound]
        for currentRange in ranges.dropFirst() {
            heapsort.insert(currentRange.upperBound)
            // earliest end overlaps, then allocate in heap
            if let earliestEnd = heapsort.min, earliestEnd > currentRange.lowerBound {
                maxRooms = Swift.max(maxRooms, heapsort.count)
            // otherwise the min has finished so remove from heap
            } else {
                heapsort.removeMin()
            }
        }
        return maxRooms
    }
    static func test() {
        let sut = Leet0253()
        assert(sut.minMeetingRooms([[0, 30], [5, 10], [15, 20]]) == 2)
        assert(sut.minMeetingRooms([[7, 10], [2, 4]]) == 1)
        assert(sut.minMeetingRooms([[1,13], [13,14]]) == 1)
        assert(sut.minMeetingRooms([[1,5],[8,9],[8,9]]) == 2)
    }
}
//Leet0253.test()




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/divide-intervals-into-minimum-number-of-groups/
class Solution {
    func minGroups(_ intervals: [[Int]]) -> Int {
        let ranges = intervals.map { $0[0]..<$0[1] }.sorted { $0.lowerBound < $1.lowerBound }
        var maxGroup = 1, heapsort: Heap<Int> = [ranges[0].upperBound]
        for currentRange in ranges.dropFirst() {
            heapsort.insert(currentRange.upperBound)
            if let earliestEnd = heapsort.min, earliestEnd >= currentRange.lowerBound {
                maxGroup = Swift.max(maxGroup, heapsort.count)
            } else {
                heapsort.removeMin()
            }
        }
        return maxGroup
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/teemo-attacking/
class Leet0495 {
    func findPoisonedDuration(_ timeSeries: [Int], _ duration: Int) -> Int {
        guard duration > 0, !timeSeries.isEmpty else { return timeSeries.reduce(0, +) }
        let timeSeries = timeSeries.map { $0...$0+duration-1 }
        guard let firstRange = timeSeries.first else { return 0 }
        var resultRanges = [ClosedRange<Int>]([firstRange])
        for i in 1..<timeSeries.count {
            guard let lastRange = resultRanges.last else { continue }
            let currentRange = timeSeries[i]
            if currentRange.overlaps(lastRange) {
                resultRanges[resultRanges.count - 1] = lastRange.self.merge(currentRange)
            } else {
                resultRanges.append(currentRange)
            }
        }
        return resultRanges.reduce(0) { result, range in result + range.upperBound - range.lowerBound + 1 }
    }
}







extension Array<Character> {
    func find(_ word: [Character]) -> [Range<Int>] {
        guard !word.isEmpty else { return [] }
        var wordIndex = 0, index = 0, result = [Range<Int>]()
        while index < self.count {
            var tempIndex = index
            while wordIndex < word.count, tempIndex < self.count, self[tempIndex] == word[wordIndex] {
                wordIndex += 1; tempIndex += 1
            }
            if wordIndex == word.count {
                result.append(tempIndex - word.count ..< tempIndex)
                wordIndex = 0
            }
            index += 1
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
/// https://leetcode.com/problems/add-bold-tag-in-string/
class Leet0616 {

    func addBoldTag(_ s: String, _ words: [String]) -> String {
        let s = Array(s), n = s.count, startTag = "<b>", endTag = "</b>"
        var bolds = Array(repeating: false, count: s.count), result = ""
        // map the bolds array with the words appearing in string s.
        let words = words.map { Array($0) }
        for i in 0..<n {
            skip: for w in words {            // skip prevents time limit exceeded!
                for (j, c) in w.enumerated() {
                    guard i+j < n else { continue skip } // exceeds string lenth n
                    if s[i+j] != c { continue skip }     // obviously not equal
                }
                for j in 0..<w.count {
                    bolds[i+j] = true
                }
            }
        }
        // construct the result and add the tags
        for i in 0..<n {
            if bolds[i], (i == 0 || !bolds[i-1]) {
                result += startTag
            }
            result += String(s[i])
            if bolds[i], (i == n-1 || !bolds[i+1]) {
                result += endTag
            }
        }
        return result
    }

    static func test() {
        let sut = Leet0616()
        assert(sut.addBoldTag("abcxyz123", ["abc", "123"]) == "<b>abc</b>xyz<b>123</b>")
        assert(sut.addBoldTag("aaabbb", ["aa","b"]) == "<b>aaabbb</b>")
        assert(sut.addBoldTag("aaabbcc", ["aaa","aab","bc"]) == "<b>aaabbc</b>c")
    }
}
//Leet0616.test()






///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/min-cost-climbing-stairs/
class Leet0746 {
    func minCostClimbingStairs(_ cost: [Int]) -> Int {
        let n = cost.count
        var dp = Array(repeating: 0, count: n + 1)
        for i in 2...n {
            dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        }
        return dp[n]
    }
}


///---------------------------------------------------------------------------------------
/// https://leetcode.com/problems/longest-increasing-subsequence/
class Leet0300 {
    func lengthOfLIS(_ nums: [Int]) -> Int {
        let n = nums.count
        var dp = Array(repeating: 1, count: n), result = 1
        for i in 1..<n {
            for j in 0..<i {
                if nums[i] > nums[j] {
                    dp[i] = max(dp[i], dp[j] + 1)
                    result = max(result, dp[i])
                }
            }
        }
        return result
    }
}




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/solving-questions-with-brainpower/
class Leet2140 {
    func mostPoints(_ questions: [[Int]]) -> Int {
        let n = questions.count
        var dp = Array(repeating: 0, count: n + 1)
        for i in stride(from: n - 1, through: 0, by: -1) {
            let j = i + questions[i][1] + 1
            dp[i] = max(questions[i][0] + dp[min(j, n)], dp[i + 1])
        }
        return dp[0]
    }
    static func test() {
        let sut = Leet2140()
        assert(sut.mostPoints([[3,2],[4,3],[4,4],[2,5]]) == 5)
        assert(sut.mostPoints([[1,1],[2,2],[3,3],[4,4],[5,5]]) == 7)
    }
}
//Leet2140.test()





extension ClosedRange where Bound: Comparable {
    func intersection(_ other: ClosedRange<Bound>) -> ClosedRange<Bound>? {
        overlaps(other) ? Swift.max(lowerBound, other.lowerBound)...Swift.min(upperBound, other.upperBound) : nil
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/interval-list-intersections/
class Leet0986 {
    func intervalIntersection(_ firstList: [[Int]], _ secondList: [[Int]]) -> [[Int]] {
        guard !firstList.isEmpty && !secondList.isEmpty else { return [] }
        let firstList = firstList.map { $0[0]...$0[1] }, secondList = secondList.map { $0[0]...$0[1] }
        var ranges: [ClosedRange<Int>] = [], i1 = 0, i2 = 0
        while i1 < firstList.count && i2 < secondList.count {
            let r1 = firstList[i1], r2 = secondList[i2]
            if r1.overlaps(r2), let intersection = r1.intersection(r2) {
                ranges.append(intersection)
            }
            i1 += r1.upperBound <= r2.upperBound ? 1 : 0
            i2 += r2.upperBound <= r1.upperBound ? 1 : 0
        }
        return ranges.map { [$0.lowerBound, $0.upperBound] }
    }
    static func test() {
        let sut = Leet0986()
        assert(sut.intervalIntersection([[0,2],[5,10],[13,23],[24,25]], [[1,5],[8,12],[15,24],[25,26]]) == [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]])
        assert(sut.intervalIntersection([[1,3],[5,9]], []) == [])
    }
}
//Leet0986.test()







///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/determine-if-two-events-have-conflict/
class Leet2446 {
    func haveConflict(_ event1: [String], _ event2: [String]) -> Bool {
        (event1[0]...event1[1]).overlaps(event2[0]...event2[1])
    }
}



/*
 
 ["01:15","02:00"]
 ["02:00","03:00"]
 ["01:00","02:00"]
 ["01:20","03:00"]
 ["10:00","11:00"]
 ["14:00","15:00"]
 ["14:13","22:08"]
 ["02:40","08:08"]
 ["16:53","19:00"]
 ["10:33","18:15"]
 ["15:19","17:56"]
 ["14:11","20:02"]
 ["01:37","14:20"]
 ["05:06","06:17"]
 
 */






///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/non-overlapping-intervals/
class Leet0435 {
    func eraseOverlapIntervals(_ intervals: [[Int]]) -> Int {
        let intervals = intervals.sorted { $0[1] < $1[1] }.map { $0[0]..<$0[1] }
        var overlapCount = 0, prevEnd = intervals[0].upperBound
        for i in intervals[1...] {
            // do not overlapp
            if prevEnd <= i.lowerBound {
                prevEnd = i.upperBound
            } else { // overlaps
                overlapCount += 1
            }
        }
        return overlapCount
    }
    
    static func test() {
        let sut = Leet0435()
        assert(sut.eraseOverlapIntervals([[1,5],[2,3],[3,4]]) == 1)
        assert(sut.eraseOverlapIntervals([[1,2],[1,2],[1,2]]) == 2)
        assert(sut.eraseOverlapIntervals([[1,2],[2,3]]) == 0)
    }
}
//Leet0435.test()


/*
 
 [[1, 5], [2, 3], [3, 4]]
 [[1,9],[2,3],[4,5],[6,7]]
 [[0,1],[3,4],[1,2]]
 
 [[1,2],[2,3],[3,4],[1,3]]
 [[1,2],[1,2],[1,2]]
 [[1,2],[2,3]]
 [[-73,-26],[-65,-11],[-62,-49]]
 [[1,100],[11,22],[1,11],[2,12]]
 [[-52,31],[-73,-26],[82,97],[-65,-11],[-62,-49],[95,99],[58,95],[-31,49],[66,98],[-63,2],[30,47],[-40,-26]]
 [[-100,-87],[-90,-44],[-86,7],[-85,-76],[-70,33]]
 [[-47397,9550],[2720,30659],[-23874,-14936],[-31855,18014],[31222,34030],[-13839,-4283],[-22000,17663],[46119,47768],[-5123,10900],[11410,31472],[43210,46846],[10347,46318],[-37970,9448],[-38500,-8858],[6451,18362],[-11875,16988],[-7651,36023],[20607,21165],[35854,40598],[32093,37885],[21732,40458],[4378,26744],[24895,35981],[3782,27773],[45646,48754],[-6825,-576],[-1356,38013],[-20483,-10761],[41785,44602],[-35607,20895],[24983,39497],[49344,49667],[3809,39757],[-21256,-916],[-24425,20174],[29860,40909],[-18701,-12461]]
 
 */

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/
class Leet0452 {
    func findMinArrowShots(_ points: [[Int]]) -> Int {
        let intervals = points.sorted { $0[1] < $1[1] }.map { $0[0]...$0[1] }
        var arrowsNeeded = 1, overlappingUpperBound = intervals[0].upperBound
        for i in 1..<intervals.count {
            if !(intervals[i] ~= overlappingUpperBound) {
                arrowsNeeded += 1
                overlappingUpperBound = intervals[i].upperBound
            }
        }
        return arrowsNeeded
    }
    static func test() {
        let sut = Leet0452()
        assert(sut.findMinArrowShots([[10,16],[2,8],[1,6],[7,12]]) == 2)
        assert(sut.findMinArrowShots([[1,2],[3,4],[5,6],[7,8]]) == 4)
        assert(sut.findMinArrowShots([[1,2],[2,3],[3,4],[4,5]]) == 2)
    }
}
//Leet0452.test()


///---------------------------------------------------------------------------------------
/// https://leetcode.com/problems/maximum-length-of-pair-chain/
class Leet0646 {
    func findLongestChain(_ pairs: [[Int]]) -> Int {
        let intervals = pairs.sorted { $0[1] < $1[1] }.map { $0[0]..<$0[1] }
        var count = 1, prevEnd = intervals[0].upperBound
        for i in intervals[1...] {
            // if chained, count and update prevEnd. don't care for overlaps
            if prevEnd < i.lowerBound {
                count += 1
                prevEnd = i.upperBound
            }
        }
        return count
    }
    static func test() {
        let sut = Leet0646()
        assert(sut.findLongestChain([[1,2]]) == 1)
    }
}
//Leet0646.test()





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-value-of-an-ordered-triplet-i
class Leet2873 {
    func maximumTripletValue(_ nums: [Int]) -> Int {
        var maxTriplet = Int.min
        for i in 0..<(nums.count - 2) {
            for j in (i+1)..<(nums.count - 1) {
                for k in (j+1)..<nums.count {
                    maxTriplet = max(maxTriplet, (nums[i] - nums[j]) * nums[k])
                }
            }
        }
        return maxTriplet > 0 ? maxTriplet : 0
    }
    func x_maximumTripletValue(_ nums: [Int]) -> Int {
        var result = 0, max_i = 0, maxDiff = 0
        for num in nums {
            result = max(result, maxDiff * num)
            maxDiff = max(maxDiff, max_i - num)
            max_i = max(max_i, num)
        }
        return result
    }
    static func test() {
        let sut = Leet2873()
        assert(sut.maximumTripletValue([1000000,1,1000000]) == 999999000000)
        assert(sut.maximumTripletValue([12,6,1,2,7]) == 77)
    }
}
//Leet2873.test()



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-value-of-an-ordered-triplet-ii
class Leet2874 {
        
    func maximumTripletValue(_ nums: [Int]) -> Int {
        let n = nums.count
        var maxLefts = Array(repeating: 0, count: n)
        for i in 1..<n {
            maxLefts[i] = max(maxLefts[i-1], nums[i-1])
        }
        var maxRights = Array(repeating: 0, count: n)
        for k in stride(from: n-2, through: 0, by: -1) {
            maxRights[k] = max(maxRights[k+1], nums[k+1])
        }
        var maxTripletValue = 0
        for j in 1..<n-1 {
            maxTripletValue = max(maxTripletValue, (maxLefts[j] - nums[j]) * maxRights[j])
        }
        return maxTripletValue
    }
    
    static func test() {
        let sut = Leet2874()
        assert(sut.maximumTripletValue([1000000,1,1000000]) == 999999000000)
        assert(sut.maximumTripletValue([12,6,1,2,7]) == 77)
    }
}
//Leet2874.test()




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-ways-to-group-overlapping-ranges/
class Leet2580 {
    func countWays(_ ranges: [[Int]]) -> Int {
        let intervals = ranges.sorted { $0[0] < $1[0] }.map { $0[0]...$0[1] }
        var prev = intervals[0], merged = [prev]
        // merge overlaps
        for current in intervals[1...] {
            if let last = merged.last, last.overlaps(current) {
                merged[merged.count - 1] = last.merge(current)
            } else {
                merged.append(current)
            }
        }
        return power(2, merged.count) % mod
    }
    private let mod = 1_000_000_007
    private func power(_ base: Int, _ exponent: Int) -> Int {
        var result = 1, base = base, exponent = exponent
        while exponent > 0 {
            if exponent % 2 == 1 {
                result = (result * base) % mod
            }
            base = (base * base) % mod
            exponent /= 2
        }
        return result
    }
    static func test() {
        let sut = Leet2580()
        assert(sut.countWays([[6,10],[5,15]]) == 2)
        assert(sut.countWays([[1,3],[10,20],[2,5],[4,8]]) == 4)
        assert(sut.countWays([[57,92],[139,210],[306,345],[411,442],[533,589],[672,676],[801,831],[937,940],[996,1052],[1113,1156],[1214,1258],[1440,1441],[1507,1529],[1613,1659],[1773,1814],[1826,1859],[2002,2019],[2117,2173],[2223,2296],[2335,2348],[2429,2532],[2640,2644],[2669,2676],[2786,2885],[2923,2942],[3035,3102],[3177,3249],[3310,3339],[3450,3454],[3587,3620],[3725,3744],[3847,3858],[3901,3993],[4100,4112],[4206,4217],[4250,4289],[4374,4446],[4510,4591],[4675,4706],[4732,4768],[4905,4906],[5005,5073],[5133,5142],[5245,5309],[5352,5377],[5460,5517],[5569,5602],[5740,5791],[5823,5888],[6036,6042],[6096,6114],[6217,6262],[6374,6394],[6420,6511],[6564,6587],[6742,6743],[6797,6877],[6909,6985],[7042,7117],[7141,7144],[7276,7323],[7400,7456],[7505,7557],[7690,7720],[7787,7800],[7870,7880],[8013,8031],[8114,8224],[8272,8328],[8418,8435],[8493,8537],[8600,8704],[8766,8812],[8839,8853],[9032,9036],[9108,9189],[9222,9291],[9344,9361],[9448,9502],[9615,9673],[9690,9800],[9837,9868],[85,96],[145,202],[254,304],[372,411],[534,551],[629,692],[727,787],[861,944],[1041,1084],[1133,1174],[1260,1307],[1339,1358],[1478,1548],[1580,1618],[1694,1814],[1848,1891],[1936,1990],[2058,2130]]) == 570065479)
    }
}
//Leet2580.test()





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/points-that-intersect-with-cars/
class Leet2848 {
    func numberOfPoints(_ nums: [[Int]]) -> Int {
        let nums = nums.sorted { $0[0] < $1[0] }.map { $0[0]...$0[1]}
        var merged = [nums[0]]
        for range in nums[1...] {
            if let last = merged.last, last.overlaps(range) {
                merged[merged.count - 1] = last.merge(range)
            } else {
                merged.append(range)
            }
        }
        return merged.reduce(0) { $0 + ($1.upperBound - $1.lowerBound + 1) }
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-days-without-meetings/
class Leet3169 {
    func countDays(_ days: Int, _ meetings: [[Int]]) -> Int {
        let meetings = meetings.sorted { $0[0] < $1[0] }.map { $0[0]...$0[1] }
        var merged = [meetings[0]]
        for range in meetings[1...] {
            if let last = merged.last, last.overlaps(range) {
                merged[merged.count - 1] = last.merge(range)
            } else {
                merged.append(range)
            }
        }
        let m = merged.reduce(into: 0) { $0 += ($1.upperBound - $1.lowerBound + 1) }
        return days - m
    }
}





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/number-of-students-doing-homework-at-a-given-time/
class Leet1450 {
    func busyStudent(_ startTime: [Int], _ endTime: [Int], _ queryTime: Int) -> Int {
        startTime.enumerated().reduce(into: 0) { $0 += ($1.element...endTime[$1.offset]) ~= queryTime ? 1 : 0 }
    }
}







///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimize-connected-groups-by-inserting-interval/
class Leet3323 {
    func minConnectedGroups(_ intervals: [[Int]], _ k: Int) -> Int {
        let intervals = intervals.sorted { $0[0] < $1[0] }.map { $0[0]...$0[1] }
        var merged = [intervals[0]]
        // merge intervals
        for range in intervals[1...] {
            if let last = merged.last, last.overlaps(range) {
                merged[merged.count - 1] = last.merge(range)
            } else {
                merged.append(range)
            }
        }
        // calculate minimum count
        var minCount = Int.max, groups = merged.count, right = 0
        for curr in merged {
            while right < merged.count, curr.upperBound + k >= merged[right].lowerBound {
                groups -= 1
                right += 1
            }
            groups += 1
            minCount = Swift.min(minCount, groups)
        }
        return minCount
    }
    static func test() {
        let sut = Leet3323()
        assert(sut.minConnectedGroups([[1,3],[5,6],[8,10]], 3) == 2)
        assert(sut.minConnectedGroups([[5,10],[1,1],[3,3]], 1) == 3)
        assert(sut.minConnectedGroups([[9,13],[3,7],[17,20],[5,8],[5,7],[5,7],[16,18],[16,19]], 3) == 2)
        assert(sut.minConnectedGroups([[1,3],[15,16],[11,16],[10,15],[14,18],[7,9],[4,5]], 1) == 3)
        assert(sut.minConnectedGroups([[7,11],[19,20],[5,5],[13,16],[17,17],[3,4],[9,9],[9,13],[13,15],[14,14]], 3) == 3)

    }
}
//Leet3323.test()








///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/lowest-common-ancestor-of-deepest-leaves/
class Leet1123 {
    func lcaDeepestLeaves(_ root: TreeNode?) -> TreeNode? {
        dfs(root).node
    }
    func dfs(_ node: TreeNode?) -> (depth: Int, node: TreeNode?) {
        guard let node else { return (0, nil) }
        let left = dfs(node.left)
        let right = dfs(node.right)
        if left.depth > right.depth {
            return (left.depth + 1, left.node)
        }
        if left.depth < right.depth {
            return (right.depth + 1, right.node)
        }
        return (left.depth + 1, node)
    }
    static func test() {
        let sut = Leet1123()
        assert(sut.lcaDeepestLeaves([3,5,1,6,2,0,8,nil,nil,7,4].buildTree())?.val == 2)
        assert(sut.lcaDeepestLeaves([1].buildTree())?.val == 1)
        assert(sut.lcaDeepestLeaves([0,1,3,nil,2].buildTree())?.val == 2)
    }
}
// Leet1123.test()






///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/smallest-subtree-with-all-the-deepest-nodes/
class Leet0865 {
    func subtreeWithAllDeepest(_ root: TreeNode?) -> TreeNode? {
        dfs(root).node
    }
    private func dfs(_ node: TreeNode?) -> (depth: Int, node: TreeNode?) {
        guard let node else { return (0, nil) }
        let left = dfs(node.left)
        let right = dfs(node.right)
        if left.depth > right.depth {
            return (left.depth + 1, left.node)
        }
        if left.depth < right.depth {
            return (right.depth + 1, right.node)
        }
        return (left.depth + 1, node)
    }
    static func test() {
        let sut = Leet0865()
        assert(sut.subtreeWithAllDeepest([3,5,1,6,2,0,8,nil,nil,7,4].buildTree())?.val == 2)
        assert(sut.subtreeWithAllDeepest([1].buildTree())?.val == 1)
        assert(sut.subtreeWithAllDeepest([0,1,3,nil,2].buildTree())?.val == 2)
    }
}
//Leet0865.test()




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/subsets/
class Leet0078 {
    func subsets(_ nums: [Int]) -> [[Int]] {
        let n = nums.count, maskCount = 1 << n
        var result = [[Int]]()
        for i in 0..<maskCount {
            var subset = [Int]()
            for p in Array(String(i, radix: 2)).reversed().enumerated() where p.element == "1" {
                subset.append(nums[p.offset])
            }
            result.append(subset)
        }
        return result
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sum-of-all-subset-xor-totals
class Leet1863 {
    func subsetXORSum(_ nums: [Int]) -> Int {
        let n = nums.count, maskCount = 1 << n
        var result = 0
        for i in 0..<maskCount {
            var subsetXor = 0
            for p in Array(String(i, radix: 2)).reversed().enumerated() where p.element == "1" {
                subsetXor ^= nums[p.offset]
            }
            result += subsetXor
        }
        return result
    }
}







///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/largest-divisible-subset/
class Leet0368 {
    func largestDivisibleSubset(_ nums: [Int]) -> [Int] {
        let n = nums.count
        guard n > 0 else { return [] }
        
        var eds: [[Int]] = Array(repeating: [], count: n)
        let nums = nums.sorted()
        
        for i in 0..<n {
            var maxSubset = [Int]()
            for k in 0..<i {
                if (nums[i] % nums[k] == 0 && maxSubset.count < eds[k].count) {
                    maxSubset = eds[k]
                }
            }
            eds[i] = maxSubset + [nums[i]]
        }
        var answer = [Int]()
        for i in 0..<n {
            if answer.count < eds[i].count {
                answer = eds[i]
            }
        }
        return answer
    }
}










///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/partition-equal-subset-sum/
class Leet0416 {
    func canPartition(_ nums: [Int]) -> Bool {
        let sum = nums.reduce(0, +), n = nums.count
        guard sum % 2 == 0 else { return false }
        var dp: [[Bool]] = Array(repeating: Array(repeating: false, count: sum / 2 + 1), count: n + 1)
        dp[0][0] = true
        for i in 1...n {
            for j in 0...sum/2 {
                dp[i][j] = dp[i-1][j]
                if j >= nums[i-1] {
                    dp[i][j] = dp[i][j] || dp[i-1][j-nums[i-1]]
                }
            }
        }
        return dp[n][sum/2]
    }
    
    static func test() {
        let sut = Leet0416()
        assert(sut.canPartition([1,5,11,5]))
    }
}
//Leet0416.test()






///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iv/
class Leet1676 {
    func lowestCommonAncestor(_ root: TreeNode, _ nodes: [TreeNode]) -> TreeNode {
        guard nodes.count > 1 else { return nodes.first! }
        let children = Set(nodes.map { $0.val })
        var euler = [TreeNode](), depths = [Int](), i = -1, j = 0
        
        // do the euler tour
        func dfs(_ node: TreeNode, _ depth: Int) {
            euler.append(node)
            depths.append(depth)
            if children.contains(node.val) {
                if i == -1 {
                    i = euler.count - 1
                }
                j = euler.count - 1
            }
            if let left = node.left {
                dfs(left, depth + 1)
                euler.append(node)
                depths.append(depth)
            }
            if let right = node.right {
                dfs(right, depth + 1)
                euler.append(node)
                depths.append(depth)
            }
        }
        dfs(root, 0)
        guard let index = depths[i...j].enumerated().min(by: { $0.element < $1.element })?.offset else { return root }
        return euler[i + index]
    }
    static func test() {
        let sut = Leet1676()
        assert(sut.lowestCommonAncestor([3,5,1,6,2,0,8,nil,nil,7,4].buildTree()!, [TreeNode(7),TreeNode(6),TreeNode(2),TreeNode(4)]).val == 5)
        assert(sut.lowestCommonAncestor([0,1,2,nil,3,4,5,nil,nil,6].buildTree()!, [TreeNode(5), TreeNode(6)]).val == 2)
    }
}
//Leet1676.test()



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree
class Leet0236 {
    func lowestCommonAncestor(_ root: TreeNode?, _ p: TreeNode?, _ q: TreeNode?) -> TreeNode? {
        guard let p, let q, let root else { return root }
        var euler = [TreeNode](), depths = [Int](), i = -1, j = -1, children = Set([p.val, q.val])
        
        // do the euler tour
        func dfs(_ node: TreeNode, _ depth: Int) {
            euler.append(node)
            depths.append(depth)
            guard !children.isEmpty else { return }
            if children.contains(node.val) {
                if i == -1 {
                    i = euler.count - 1
                }
                j = euler.count - 1
                children.remove(node.val)
            }
            if let left = node.left {
                dfs(left, depth + 1)
                euler.append(node)
                depths.append(depth)
            }
            if let right = node.right {
                dfs(right, depth + 1)
                euler.append(node)
                depths.append(depth)
            }
        }
        dfs(root, 0)
        guard let index = depths[i...j].enumerated().min(by: { $0.element < $1.element })?.offset else { return root }
        return euler[i + index]
    }
}






///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
class Leet0235 {
    func lowestCommonAncestor(_ root: TreeNode?, _ p: TreeNode?, _ q: TreeNode?) -> TreeNode? {
        guard let parentVal = root?.val, let pVal = p?.val, let qVal = q?.val else { return nil }
        if pVal > parentVal && qVal > parentVal {
            return lowestCommonAncestor(root?.right, p, q)
        } else if pVal < parentVal && qVal < parentVal {
            return lowestCommonAncestor(root?.left, p, q)
        } else {
            return root
        }
    }
    
    static func test() {
        let sut = Leet0235()
        assert(sut.lowestCommonAncestor([6,2,8,0,4,7,9,nil,nil,3,5].buildTree(), TreeNode(2), TreeNode(8))?.val == 6)
        assert(sut.lowestCommonAncestor([6,2,8,0,4,7,9,nil,nil,3,5].buildTree(), TreeNode(2), TreeNode(4))?.val == 2)
        assert(sut.lowestCommonAncestor([2,1].buildTree(), TreeNode(2), TreeNode(1))?.val == 2)
    }
}
//Leet0235.test()






///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-ii/
class Leet1644 {
    func lowestCommonAncestor(_ root: TreeNode?, _ p: TreeNode?, _ q: TreeNode?) -> TreeNode? {
        guard let p, let q, let root else { return root }
        var euler = [TreeNode](), depths = [Int](), i = -1, j = -1, children = Set([p.val, q.val])
        
        // do the euler tour
        func dfs(_ node: TreeNode, _ depth: Int) {
            euler.append(node)
            depths.append(depth)
            guard !children.isEmpty else { return }
            if children.contains(node.val) {
                if i == -1 {
                    i = euler.count - 1
                }
                j = euler.count - 1
                children.remove(node.val)
            }
            if let left = node.left {
                dfs(left, depth + 1)
                euler.append(node)
                depths.append(depth)
            }
            if let right = node.right {
                dfs(right, depth + 1)
                euler.append(node)
                depths.append(depth)
            }
        }
        dfs(root, 0)
        guard children.isEmpty else { return nil }
        guard let index = depths[i...j].enumerated().min(by: { $0.element < $1.element })?.offset else { return root }
        return euler[i + index]
    }
    
    static func test() {
        let sut = Leet1644()
        assert(sut.lowestCommonAncestor([3,5,1,6,2,0,8,nil,nil,7,4].buildTree(), TreeNode(5), TreeNode(10)) == nil)
    }
}
//Leet1644.test()












///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-number-of-operations-to-make-elements-in-array-distinct/
class Leet3396 {
    func minimumOperations(_ nums: [Int]) -> Int {
        var seen: Set<Int> = []
        for i in stride(from: nums.count - 1, through: 0, by: -1) {
            if !seen.contains(nums[i]) {
                seen.insert(nums[i])
            } else {
                return i / 3 + 1
            }
        }
        return 0
    }
}





public class NodeWithParent {
    public var val: Int
    public var left: NodeWithParent?
    public var right: NodeWithParent?
    public var parent: NodeWithParent?
    public init(_ val: Int) {
        self.val = val
        self.left = nil
        self.right = nil
        self.parent = nil
    }
}
 


///---------------------------------------------------------------------------------------
/// https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iii/
class Leet1650 {
    typealias Node = NodeWithParent
    
    func lowestCommonAncestor(_ p: Node?,_ q: Node?) -> Node? {
        var pValueSet: Set<Int> = [], currentNode: Node? = p
        while let node = currentNode {
            pValueSet.insert(node.val)
            currentNode = node.parent
        }
        currentNode = q
        while let node = currentNode {
            if pValueSet.contains(node.val) {
                return node
            }
            currentNode = node.parent
        }
        return nil
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/
class Leet1026 {
    private func dfs(_ node: TreeNode?, _ minValue: Int, _ maxValue: Int) -> Int{
        guard let node else {
            return maxValue - minValue
        }
        let minValue = min(minValue, node.val)
        let maxValue = max(maxValue, node.val)
        let left = dfs(node.left, minValue, maxValue)
        let right = dfs(node.right, minValue, maxValue)
        return max(left, right)
    }
        
    func maxAncestorDiff(_ root: TreeNode?) -> Int {
        guard let root else { return 0 }
        let answer = dfs(root, root.val, root.val)
        return answer
    }
    
    static func test() {
        let sut = Leet1026()
        assert(sut.maxAncestorDiff([8,3,10,1,6,nil,14,nil,nil,4,7,13].buildTree()) == 7)
    }
    
}
//Leet1026.test()









public class NodeWithChildren {
    public var val: Int
    public var children: [NodeWithChildren]
    public init(_ val: Int) {
        self.val = val
        self.children = []
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/n-ary-tree-preorder-traversal/
class Leet0589 {
    typealias Node = NodeWithChildren
    
    func preorder(_ root: Node?) -> [Int] {
        guard let root else { return [] }
        var stack: [Node?] = [root]
        var result: [Int] = []
        while !stack.isEmpty {
            let node = stack.removeLast()
            guard let node else { continue }
            for i in stride(from: node.children.count - 1, through: 0, by: -1) {
                let child = node.children[i]
                stack.append(child)
            }
            result.append(node.val)
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/n-ary-tree-postorder-traversal/
class Leet0590 {
    typealias Node = NodeWithChildren

    private func traverse(_ node: Node, _ result: inout [Int]) {
        guard !node.children.isEmpty else {
            result.append(node.val)
            return
        }
        for child in node.children {
            traverse(child, &result)
        }
        result.append(node.val)
    }
    func postorder_recursion(_ root: Node?) -> [Int] {
        var result = [Int]()
        guard let root else { return result }
        traverse(root, &result)
        return result
    }
    
    func postorder(_ root: Node?) -> [Int] {
        var result = [Int]()
        guard let root else { return result }
        var stack = [root]
        
        while !stack.isEmpty {
            let current = stack.removeLast()
            result.append(current.val)
            for child in current.children {
                stack.append(child)
            }
        }
        result.reverse()
        return result
    }

}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/n-ary-tree-level-order-traversal/
class Leet0429 {
    typealias Node = NodeWithChildren

    func levelOrder(_ root: Node?) -> [[Int]] {
        var result = [[Int]]()
        guard let root else { return result }
        var deque: Deque<Node> = [root]
        while !deque.isEmpty {
            var currentLevel: [Int] = []
            for _ in deque {
                let node = deque.removeFirst()
                currentLevel.append(node.val)
                for child in node.children {
                    deque.append(child)
                }
            }
            result.append(currentLevel)
        }
        return result
    }
}








///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/longest-substring-without-repeating-characters/
class Leet0003 {
    func lengthOfLongestSubstring(_ s: String) -> Int {
        let s = Array(s)
        var lastSeenMap = [Character: Int](), start = 0, maxLength = 0
        for (index, char) in s.enumerated() {
            if let lastSeenIndex = lastSeenMap[char] {
                for i in start...lastSeenIndex {
                    lastSeenMap[s[i]] = nil
                }
                start = lastSeenIndex + 1
            }
            lastSeenMap[char] = index
            maxLength = max(maxLength, index - start + 1)
        }
        return maxLength
    }
    static func test() {
        let sut = Leet0003()
        assert(sut.lengthOfLongestSubstring("abababababababababababababababababababababababababababababababab") == 2)
        assert(sut.lengthOfLongestSubstring("bbbbb") == 1)
        assert(sut.lengthOfLongestSubstring("tmmzuxt") == 5)
    }
}
//Leet0003.test()






///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-erasure-value/
class Leet1695 {
    func maximumUniqueSubarray(_ nums: [Int]) -> Int {
        var lastSeenMap = [Int:Int](), currentSum = 0, maxSum = 0, firstClearedIndex = 0
        for (index, num) in nums.enumerated() {
            if let lastSeenIndex = lastSeenMap[num] {
                for i in firstClearedIndex...lastSeenIndex {
                    lastSeenMap[nums[i]] = nil
                    currentSum -= nums[i]
                }
                firstClearedIndex = lastSeenIndex + 1
            }
            currentSum += num
            lastSeenMap[num] = index
            maxSum = max(maxSum, currentSum)
        }
        return maxSum
    }
    static func test() {
        let sut = Leet1695()
        assert(sut.maximumUniqueSubarray([187,470,25,436,538,809,441,167,477,110,275,133,666,345,411,459,490,266,987,965,429,166,809,340,467,318,125,165,809,610,31,585,970,306,42,189,169,743,78,810,70,382,367,490,787,670,476,278,775,673,299,19,893,817,971,458,409,886,434]) == 16911)
    }
}
//Leet1695.test()




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-operations-to-make-array-values-equal-to-k/
class Leet3375 {
    func minOperations(_ nums: [Int], _ k: Int) -> Int {
        Set(nums).filter({$0 < k}).count > 0 ? -1 : Set(nums).filter({$0 > k}).count
    }
}





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/longest-substring-with-at-most-two-distinct-characters/
class Leet0159 {
    func lengthOfLongestSubstringTwoDistinct(_ s: String) -> Int {
        let s = Array(s)
        var maxLength = 0, windowStart = 0, charCounts = [Character: Int]()
        for i in 0..<s.count {
            let currentChar = s[i]
            charCounts[currentChar, default: 0] += 1
            while charCounts.count > 2 {
                let charAtStart = s[windowStart]
                charCounts[charAtStart]! -= 1
                if charCounts[charAtStart] == 0 {
                    charCounts[charAtStart] = nil
                }
                windowStart += 1
            }
            maxLength = max(maxLength, i - windowStart + 1)
        }
        return maxLength
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/
class Leet0340 {
    func lengthOfLongestSubstringKDistinct(_ s: String, _ k: Int) -> Int {
        let s = Array(s)
        var maxLength = 0, windowStart = 0, charCounts = [Character: Int]()
        for i in 0..<s.count {
            let currentChar = s[i]
            charCounts[currentChar, default: 0] += 1
            while charCounts.count > k {
                let charAtStart = s[windowStart]
                charCounts[charAtStart]! -= 1
                if charCounts[charAtStart] == 0 {
                    charCounts[charAtStart] = nil
                }
                windowStart += 1
            }
            maxLength = max(maxLength, i - windowStart + 1)
        }
        return maxLength
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/fruit-into-baskets/
class Leet0904 {
    func totalFruit(_ fruits: [Int]) -> Int {
        var maxLength = 0, windowStart = 0, fruitCounts = [Int: Int]()
        for i in 0..<fruits.count {
            let currentFruit = fruits[i]
            fruitCounts[currentFruit, default: 0] += 1
            while fruitCounts.count > 2 {
                let fruitAtStart = fruits[windowStart]
                fruitCounts[fruitAtStart]! -= 1
                if fruitCounts[fruitAtStart] == 0 {
                    fruitCounts[fruitAtStart] = nil
                }
                windowStart += 1
            }
            maxLength = max(maxLength, i - windowStart + 1)
        }
        return maxLength
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/fruits-into-baskets-ii/
class Leet3477 {
    func numOfUnplacedFruits(_ fruits: [Int], _ baskets: [Int]) -> Int {
        var baskets = baskets
        fruitLoop: for f in fruits {
            for b in 0..<baskets.count {
                guard f <= baskets[b] else {
                    continue
                }
                baskets[b] = 0 // allocated
                continue fruitLoop
            }
        }
        return baskets.filter { $0 > 0 }.count
    }
    static func test() {
        let sut = Leet3477()
        assert(sut.numOfUnplacedFruits([4,2,5],[3,5,4]) == 1)
    }
}
//Leet3477.test()



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/longest-nice-subarray/
class Leet2401 {
    func longestNiceSubarray(_ nums: [Int]) -> Int {
        var usedBits = 0, maxLength = 1, left = 0
        for right in 0..<nums.count {
            while (usedBits & nums[right]) != 0 {
                usedBits ^= nums[left]
                left += 1
            }
            usedBits |= nums[right]
            maxLength = max(maxLength, right - left + 1)
        }
        return maxLength
    }
}



/*
 
 [8, 4, 2, 1]
 [12, 5, 3, 10]
 [987654321, 123456789, 555555555, 777777777, 999999999, 222222222, 444444444, 666666666, 888888888, 333333333]
 [1023, 2047, 4095, 8191, 16383, 32767, 65535, 131071, 262143, 524287]
 [1, 3, 7, 15, 31, 63, 127, 255, 511, 1023]
 [19, 38, 76, 152, 304, 608, 1216, 2432, 4864, 9728, 19456]
 [429497295, 147483647, 103741823, 53670911, 26843555, 14217727]
 [250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000]
 
 

 
 [1]
 [1, 1, 1, 1]
 [3, 6, 2]
 [5, 10, 5]
 [744437702, 379056602, 145555074, 392756761, 560864007, 934981918, 113312475, 1090, 16384, 33, 217313281, 117883195, 978927664]
 [904163577,321202512,470948612,490925389,550193477,87742556,151890632,655280661,4,263168,32,573703555,886743681,937599702,120293650,725712231,257119393]
 [84139415,693324769,614626365,497710833,615598711,264,65552,50331652,1,1048576,16384,544,270532608,151813349,221976871,678178917,845710321,751376227,331656525,739558112,267703680]
 [178830999,19325904,844110858,806734874,280746028,64,256,33554432,882197187,104359873,453049214,820924081,624788281,710612132,839991691]
 
 
 
 [21,2,8,23,64,128,256,512]
 [4,3,4,3]
 
 
 */









///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-the-number-of-powerful-integers/
class Leet2999 {
    func numberOfPowerfulInt(_ start: Int, _ finish: Int, _ limit: Int, _ s: String) -> Int {
        let high = String(finish), n = high.count, prefixLength = n - s.count, s = Array(s)
        let low = String(start).padded(to: n, with: "0")
        var memo = Array(repeating: -1, count: n)
        func dfs(_ i: Int, _ limitLo: Bool, _ limitHi: Bool) -> Int {
            if i == n {
                return 1
            }
            if (!limitLo && !limitHi && memo[i] != -1) {
                return memo[i]
            }
            let lo = limitLo ? Int(String(Array(low)[i]))! : 0
            let hi = limitHi ? Int(String(Array(high)[i]))! : 9
            var result = 0
            if (i < prefixLength) {
                var digit = lo
                while digit <= (min(hi, limit)) {
                    result += dfs(i + 1, limitLo && digit == lo, limitHi && digit == hi)
                    digit += 1
                }
            } else {
                let x = Int(String(s[i - prefixLength]))!
                if lo <= x && x <= min(hi, limit) {
                    result = dfs(i + 1, limitLo && x == lo, limitHi && x == hi)
                }
            }
            if (!limitLo && !limitHi) {
                memo[i] = result
            }
            return result
        }
        return dfs(0, true, true)
    }
    
    static func test() {
        let sut = Leet2999()
        assert(sut.numberOfPowerfulInt(1829505, 1255574165, 7, "11223") == 5470)
    }
    
}
//Leet2999.test()

/*
 
 1829505
 1255574165
 7
 "11223"
 
 
 20
 1159
 5
 "20"
 1
 971
 9
 "72"
 1
 6000
 4
 "124"
 15398
 1424153842
 8
 "101"
 1
 2000
 8
 "1"
 15
 1440
 5
 "11"
 15398
 1424153842
 8
 "101"
 1114
 1864854501
 7
 "26"
 
 */




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/longest-repeating-character-replacement/
class Leet0424 {
    func characterReplacement(_ s: String, _ k: Int) -> Int {
        let allLetters = Set(s), s = Array(s)
        var maxLength = 0
        for letter in allLetters {
            var start = 0, count = 0
            for (end, char) in s.enumerated() {
                if char == letter {
                    count += 1
                }
                while !isWindowValid(start, end, count, k) {
                    if s[start] == letter {
                        count -= 1
                    }
                    start += 1
                }
                maxLength = max(maxLength, end - start + 1)
            }
        }
        return maxLength
    }
    private func isWindowValid(_ start: Int, _ end: Int, _ charFrequency: Int, _ maxReplacements: Int) -> Bool {
        end - start + 1 - charFrequency <= maxReplacements
    }
}







///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-consecutive-cards-to-pick-up/
class Leet2260 {
    func minimumCardPickup(_ cards: [Int]) -> Int {
        var minimumLength = Int.max, lastSeenMap = [Int: Int]()
        for (index, value) in cards.enumerated() {
            if let lastSeenIndex = lastSeenMap[value] {
                minimumLength = min(minimumLength, index - lastSeenIndex + 1)
            }
            lastSeenMap[value] = index
        }
        guard minimumLength != Int.max else {
            return -1
        }
        return minimumLength
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/optimal-partition-of-string/
class Leet2405 {
    func partitionString(_ s: String) -> Int {
        let s = Array(s)
        var partitionCount = 1, charSet: Set<Character> = []
        for c in s {
            if charSet.contains(c) {
                partitionCount += 1
                charSet.removeAll()
            }
            charSet.insert(c)
        }
        return partitionCount
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/partition-array-into-disjoint-intervals/
class Leet0915 {
    func partitionDisjoint(_ nums: [Int]) -> Int {
        var leftMax = [nums.first!], rightMin = [nums.last!]

        // compute left max
        for i in 1..<nums.count {
            leftMax.append(max(nums[i], leftMax[leftMax.count - 1]))
        }
        // compute right min
        for i in stride(from: nums.count - 2, through: 0, by: -1) {
            rightMin.append(min(nums[i], rightMin[rightMin.count - 1]))
        }
        rightMin.reverse()

        // look for partition
        for i in 1..<nums.count {
            if leftMax[i-1] <= rightMin[i] {
                return i
            }
        }
        return 0
    }
    static func test() {
        let sut = Leet0915()
        assert(sut.partitionDisjoint([1,1,1,0,6,12]) == 4)
        
    }
}
//Leet0915.test()



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sum-of-beauty-in-the-array/
class Leet2012 {
    func sumOfBeauties(_ nums: [Int]) -> Int {
        var leftMax = [nums.first!], rightMin = [nums.last!]

        // compute left max
        for i in 1..<nums.count {
            leftMax.append(max(nums[i], leftMax[leftMax.count - 1]))
        }
        // compute right min
        for i in stride(from: nums.count - 2, through: 0, by: -1) {
            rightMin.append(min(nums[i], rightMin[rightMin.count - 1]))
        }
        rightMin.reverse()

        // compute beauty
        var sum = 0
        for i in 1..<nums.count - 1 {
            if leftMax[i-1] < nums[i] && nums[i] < rightMin[i+1] {
                sum += 2
            } else if nums[i-1] < nums[i] && nums[i] < nums[i+1] {
                sum += 1
            }
        }
        return sum
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-symmetric-integers/
class Leet2843 {
    func countSymmetricIntegers(_ low: Int, _ high: Int) -> Int {
        var result = 0
        for n in low...high {
            let a = String(n).compactMap { Int(String($0)) }, c = a.count
            guard c % 2 == 0 else { continue }
            if c == 2 && n % 11 == 0 || c == 4 && a[0] + a[1] == a[2] + a[3] {
                result += 1
            }
        }
        return result
    }
}






///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-complete-subarrays-in-an-array/
class Leet2799 {
    func countCompleteSubarrays(_ nums: [Int]) -> Int {
        let uniqueNums = Set(nums)
        var l = 0, count = 0, seenMap = [Int:Int]()
        for r in 0..<nums.count {
            let n = nums[r]
            seenMap[n, default: 0] += 1
            while seenMap.count == uniqueNums.count {
                count += nums.count - r
                let m = nums[l]
                seenMap[m, default: 0] -= 1
                if seenMap[m] == 0 {
                    seenMap[m] = nil
                }
                l += 1
            }
        }
        return count
    }
}




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/number-of-substrings-containing-all-three-characters/
class Leet1358 {
    func numberOfSubstrings(_ s: String) -> Int {
        let s = Array(s)
        var left = 0, count = 0, seenMap = [Character:Int]()
        for right in 0 ..< s.count {
            let n = s[right]
            seenMap[n, default: 0] += 1
            while seenMap.count == 3 {
                count += s.count - right
                seenMap[s[left], default: 0] -= 1
                if seenMap[s[left]] == 0 {
                    seenMap[s[left]] = nil
                }
                left += 1
            }
        }
        return count
    }
}





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-of-substrings-containing-every-vowel-and-k-consonants-ii/
class Leet3306 {
    // hard
    func countOfSubstrings(_ word: String, _ k: Int) -> Int {
        let word = Array(word), vowels = Set("aeiou")
        var count = 0, start = 0, end = 0, consonantCount = 0, seenVowels = [Character: Int]()

        var nextConsonant = Array(repeating: 0, count: word.count), nextConsonantIndex = word.count
        for i in stride(from: word.count - 1, through: 0, by: -1) {
            nextConsonant[i] = nextConsonantIndex
            if !vowels.contains(word[i]) {
                nextConsonantIndex = i
            }
        }
        
        while end < word.count {
            let char = word[end]
            if vowels.contains(char) {
                seenVowels[char, default: 0] += 1
            } else {
                consonantCount += 1
            }
            
            // shrink if too many k
            while consonantCount > k {
                let startChar = word[start]
                if vowels.contains(startChar) {
                    seenVowels[startChar]! -= 1
                    if (seenVowels[startChar] == 0) { seenVowels[startChar] = nil }
                } else {
                    consonantCount -= 1
                }
                start += 1
            }
            
            // while valid window shrink
            while start < word.count, seenVowels.count == 5, consonantCount == k {
                
                count += nextConsonant[end] - end
                let startChar = word[start]
                if vowels.contains(startChar) {
                    seenVowels[startChar]! -= 1
                    if (seenVowels[startChar] == 0) { seenVowels[startChar] = nil }
                } else {
                    consonantCount -= 1
                }
                start += 1
            }
            
            end += 1
        }
        return count
    }
    static func test() {
        let sut = Leet3306()
        assert(sut.countOfSubstrings("aadieuoh", 1) == 2)
        assert(sut.countOfSubstrings("aoaiuefi", 1) == 4)
        assert(sut.countOfSubstrings("iqeaouqi", 2) == 3)
        assert(sut.countOfSubstrings("ieaouqqieaouqq", 1) == 3)
    }
}
//Leet3306.test()






///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-the-count-of-good-integers/
class Leet3272 {
    // hard
    func countGoodIntegers(_ n: Int, _ k: Int) -> Int {
        var dict = Set<String>()
        let base = Int(pow(10.0, Double((n - 1) / 2)))
        let skip = n % 2

        for i in base..<(base * 10) {
            var s = String(i)
            let reversed = String(s.reversed())
            let mirrored = String(reversed.dropFirst(skip))
            s += mirrored

            if let palindromicInteger = Int(s), palindromicInteger % k == 0 {
                let sortedS = String(s.sorted())
                dict.insert(sortedS)
            }
        }

        var factorial = Array(repeating: 1, count: n + 1)
        for i in 1...n {
            factorial[i] = factorial[i - 1] * i
        }
        var ans = 0
        for s in dict {
            var cnt = Array(repeating: 0, count: 10)
            for c in s {
                if let digit = c.wholeNumberValue {
                    cnt[digit] += 1
                }
            }

            // Compute permutations: avoid starting with 0
            let totalDigits = n
            let permutations = (totalDigits - cnt[0]) * factorial[totalDigits - 1]
            
            var tot = permutations
            for x in cnt {
                tot /= factorial[x]
            }

            ans += tot
        }
        return ans
    }
}






///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sum-of-digits-in-base-k/
class Leet1837 {
    func sumBase(_ n: Int, _ k: Int) -> Int {
        String(n, radix: k).reduce(into: 0) { $0 += Int(String($1))! }
    }
}




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-sum-of-distinct-subarrays-with-length-k/
class Leet2461 {
    func maximumSubarraySum(_ nums: [Int], _ k: Int) -> Int {
        var seenMap = [Int:Int](), maxSum = 0, windowSum = 0, l = 0
        for (r, n) in nums.enumerated() {
            seenMap[n, default: 0] += 1
            windowSum += n
            var wk = r - l + 1
            if wk > k { // shrink window
                let lv = nums[l]
                seenMap[lv]! -= 1
                windowSum -= lv
                if seenMap[lv] == 0 { seenMap[lv] = nil }
                l += 1
            }
            wk = r - l + 1
            guard wk == k, seenMap.count == k else { continue }
            maxSum = max(maxSum, windowSum)
        }
        return maxSum
    }
    static func test() {
        let sut = Leet2461()
        assert(sut.maximumSubarraySum([1,5,4,2,9,9,9], 3) == 15)
    }
}
//Leet2461.test()



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-sum-of-almost-unique-subarray/
class Leet2841 {
    func maxSum(_ nums: [Int], _ m: Int, _ k: Int) -> Int {
        var seenMap = [Int:Int](), maxSum = 0, windowSum = 0, l = 0
        for (r, n) in nums.enumerated() {
            seenMap[n, default: 0] += 1
            windowSum += n
            var wk = r - l + 1
            if wk > k { // shrink window
                let lv = nums[l]
                seenMap[lv]! -= 1
                windowSum -= lv
                if seenMap[lv] == 0 { seenMap[lv] = nil }
                l += 1
            }
            wk = r - l + 1
            guard wk == k, seenMap.count >= m else { continue }
            maxSum = max(maxSum, windowSum)
        }
        return maxSum
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-the-power-of-k-size-subarrays-i/
class Leet3254 {
    func resultsArray(_ nums: [Int], _ k: Int) -> [Int] {
        var results = [Int](), l = 0, consecutiveCount = 0
        for (r, n) in nums.enumerated() {
            // count consecutives
            if r > 0 {
                if n == nums[r-1] + 1 {
                    consecutiveCount += 1
                } else {
                    consecutiveCount = 0
                }
            }
            var wk = r - l + 1
            if wk > k { // shrink
                l += 1
            }
            wk = r - l + 1
            if wk == k {
                if consecutiveCount == k - 1 {
                    results.append(n)
                    consecutiveCount -= 1
                } else {
                    results.append(-1)
                }
            }
        }
        return results
    }
    static func test() {
        let sut = Leet3254()
        assert(sut.resultsArray([1,2,3,4,3,2,5], 3) == [3,4,-1,-1,-1])
        assert(sut.resultsArray([2,2,2,2,2], 4) == [-1,-1])
        assert(sut.resultsArray([3,2,3,2,3,2], 2) == [-1,3,-1,3,-1])
    }
}
//Leet3254.test()





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/powx-n/
class Leet0050 {
    func myPow(_ x: Double, _ n: Int) -> Double {
        guard n != 0 else { return 1.0 }
        var n = n, x = x, result = 1.0
        if n < 0 {
            n = -n
            x = 1.0 / x
        }
        while n > 0 {
            if n % 2 == 1 {
                result *= x
                n -= 1
            }
            x *= x
            n /= 2
        }
        return result
    }

    func xxx_myPow(_ x: Double, _ n: Int) -> Double {
        guard n != 0 else { return 1.0 }
        if n < 0 {
            return 1.0 / myPow(x, -n)
        }
        if n % 2 == 1 {
            return x * myPow(x * x, (n - 1) / 2 )
        } else {
            return myPow(x * x, n / 2 )
        }
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-good-numbers/
class Leet1922 {
    private let mod = 1_000_000_007
    private func power(_ base: Int, _ exponent: Int) -> Int {
        var result = 1, base = base, exponent = exponent
        while exponent > 0 {
            if exponent % 2 == 1 {
                result = (result * base) % mod
            }
            base = (base * base) % mod
            exponent /= 2
        }
        return result
    }
    func countGoodNumbers(_ n: Int) -> Int {
        power(5, (n + 1) / 2) * power(4, n / 2) % mod
    }
}






///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-good-triplets/
class Leet1534 {
    func countGoodTriplets(_ arr: [Int], _ a: Int, _ b: Int, _ c: Int) -> Int {
        var count = 0
        for i in 0..<arr.count-2 {
            for j in i+1..<arr.count-1 where abs(arr[i] - arr[j]) <= a {
                for k in j+1..<arr.count where abs(arr[j] - arr[k]) <= b && abs(arr[i] - arr[k]) <= c {
                    count += 1
                }
            }
        }
        return count
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/number-of-unequal-triplets-in-array/
class Leet2475 {
    func unequalTriplets(_ nums: [Int]) -> Int {
        var count = 0
        for i in 0..<nums.count-2 {
            for j in i+1..<nums.count-1 {
                for k in j+1..<nums.count where nums[i] != nums[j] && nums[j] != nums[k] && nums[i] != nums[k] {
                    count += 1
                }
            }
        }
        return count
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-square-sum-triples/
class Leet1925 {
    func countTriples(_ n: Int) -> Int {
        var result = 0
        for a in 1...n {
            for b in 1...n {
                for c in 1...n where a * a + b * b  == c * c {
                    result += 1
                }
            }
        }
        return result
    }
}









///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-sum-of-mountain-triplets-i/
class Leet2908 {
    func minimumSum(_ nums: [Int]) -> Int {
        var result = Int.max
        for i in 0..<nums.count-2 {
            for j in i+1..<nums.count-1 {
                for k in j+1..<nums.count {
                    guard nums[i] < nums[j], nums[k] < nums[j] else { continue }
                    result = min(result, nums[i] + nums[j] + nums[k])
                }
            }
        }
        guard result != Int.max else { return -1 }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-sum-of-mountain-triplets-ii/description/
class Leet2909 {
    func minimumSum(_ nums: [Int]) -> Int {
        var leftMin = Array(repeating: nums[0], count: nums.count)
        for i in 1..<(nums.count) {
            leftMin[i] = min(leftMin[i-1], nums[i-1])
        }
        var rightMin = Array(repeating: nums[nums.count-1], count: nums.count)
        for i in stride(from: (nums.count - 2), through: 0, by: -1) {
            rightMin[i] = min(rightMin[i+1], nums[i+1])
        }
        var result = Int.max
        for i in 1..<(nums.count-1) where leftMin[i] < nums[i] && rightMin[i] < nums[i] {
            result = min(result, leftMin[i] + nums[i] + rightMin[i])
        }
        guard result != Int.max else { return -1 }
        return result
    }
    static func test() {
        let sut = Leet2909()
        assert(sut.minimumSum([8,6,1,5,3]) == 9)
    }
}
//Leet2909.test()



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-special-quadruplets/
class Leet1995 {
    func countQuadruplets(_ nums: [Int]) -> Int {
        var result = 0
        for a in 0..<nums.count {
            for b in a+1..<nums.count {
                for c in b+1..<nums.count {
                    for d in c+1..<nums.count {
                        if nums[a] + nums[b] + nums[c] == nums[d] {
                            result += 1
                        }
                    }
                }
            }
        }
        return result
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/number-of-ways-to-split-array/
class Leet2270 {
    func waysToSplitArray(_ nums: [Int]) -> Int {
        let sum = nums.reduce(0, +)
        var currentSum = 0, count = 0
        for i in 0..<nums.count - 1 {
            currentSum += nums[i]
            let diff = sum - currentSum
            if currentSum >= diff {
                count += 1
            }
        }
        return count
    }
}





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-pivot-index/
class Leet0724 {
    func pivotIndex(_ nums: [Int]) -> Int {
        let total = nums.reduce(0, +), n = nums.count
        var leftSum = 0
        for i in 0..<n {
            let curr = nums[i]
            if leftSum == total - leftSum - curr {
                return i
            }
            leftSum += curr
        }
        return -1
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/left-and-right-sum-differences/
class Leet2574 {
    
    func leftRightDifference_o_n(_ nums: [Int]) -> [Int] {
        guard nums.count > 1 else { return [0] }
        let n = nums.count
        var leftSum = [0], rightSum = [0], result = Array(repeating: 0, count: n)
        for i in 1 ..< n   {
            leftSum.append(leftSum[i-1] + nums[i-1])
            rightSum.append(rightSum[i-1] + nums[n-i])
        }
        rightSum.reverse()
        for i in 0 ..< n {
            result[i] = abs(leftSum[i] - rightSum[i])
        }
        return result
    }
    
    func leftRightDifference(_ nums: [Int]) -> [Int] {
        guard nums.count > 1 else { return [0] }
        let n = nums.count
        var result = Array(repeating: 0, count: n), leftSum = 0, rightSum = nums.reduce(0, +)
        for i in 0 ..< n {
            rightSum -= nums[i]
            result[i] = abs(leftSum - rightSum)
            leftSum += nums[i]
        }
        return result
    }
    static func test() {
        let sut = Leet2574()
        assert(sut.leftRightDifference([10,4,8,3]) == [15,1,11,22])
        assert(sut.leftRightDifference([1]) == [0])
    }
}
//        Leet2574.test()





class FenwickTree {
    private var tree: [Int]
    init(_ size: Int) {
        self.tree = Array(repeating: 0, count: size + 1)
    }
    func update(_ index: Int, _ delta: Int) {
        var index = index + 1
        while index < tree.count {
            tree[index] += delta
            index += index & (-index)
        }
    }
    func query(_ index: Int) -> Int {
        var sum = 0
        var index = index + 1
        while index > 0 {
            sum += tree[index]
            index -= index & (-index)
        }
        return sum
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-good-triplets-in-an-array/
class Leet2179 {
    func x_goodTriplets(_ nums1: [Int], _ nums2: [Int]) -> Int {
        let n = nums1.count
        var pos2 = Array(repeating: 0, count: n), reveresedIndexMapping = Array(repeating: 0, count: n)
        for i in 0 ..< n {
            pos2[nums2[i]] = i
        }
        for i in 0 ..< n {
            reveresedIndexMapping[pos2[i]] = i
        }
        var tree = FenwickTree(n)
        var res = 0
        for v in 0 ..< n {
            let pos = reveresedIndexMapping[v]
            let left = tree.query(pos)
            tree.update(pos, 1)
            let right = (n - 1 - pos) - (v - left)
            res += left * right
        }
        return res
    }
    
    func goodTriplets(_ nums1: [Int], _ nums2: [Int]) -> Int {
        let n = nums1.count
        var pos2 = Array(repeating: 0, count: n)
        for i in 0..<n {
            pos2[nums2[i]] = i
        }

        // Convert nums1 into positions in nums2
        var transformed = nums1.map { pos2[$0] }

        var leftSmaller = Array(repeating: 0, count: n)
        var tree = FenwickTree(n)

        for i in 0..<n {
            let val = transformed[i]
            leftSmaller[i] = tree.query(val - 1)
            tree.update(val, 1)
        }

        var rightLarger = Array(repeating: 0, count: n)
        tree = FenwickTree(n) // reset tree

        for i in stride(from: n - 1, through: 0, by: -1) {
            let val = transformed[i]
            rightLarger[i] = tree.query(n - 1) - tree.query(val)
            tree.update(val, 1)
        }

        var res = 0
        for i in 0..<n {
            res += leftSmaller[i] * rightLarger[i]
        }

        return res
    }

}




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-value-to-get-positive-step-by-step-sum/
class Leet1413 {
    func minStartValue(_ nums: [Int]) -> Int {
        var minValue = 0, sum = 0
        for i in 0 ..< nums.count {
            sum += nums[i]
            minValue = min(minValue, sum)
        }
        return 1 - minValue
    }
}





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-size-subarray-sum/
class Leet0209 {
    func minSubArrayLen(_ target: Int, _ nums: [Int]) -> Int {
        var minLength = Int.max, currentSum = 0, l = 0
        for (r, n) in nums.enumerated() {
            currentSum += n
            while currentSum >= target {
                minLength = min(minLength, r - l + 1)
                currentSum -= nums[l]
                l += 1
            }
        }
        return minLength == Int.max ? 0 : minLength
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/
class Leet0325 {
    func maxSubArrayLen(_ nums: [Int], _ k: Int) -> Int {
        var prefixSum = 0, maxLength = 0, sumIndexMap = [Int: Int]()
        for i in 0..<nums.count {
            prefixSum += nums[i]
            if prefixSum == k {
                maxLength = i + 1
            }
            let diff = prefixSum - k
            if let lastIndex = sumIndexMap[diff] {
                maxLength = max(maxLength, i - lastIndex)
            }
            if sumIndexMap[prefixSum] == nil {
                sumIndexMap[prefixSum, default: i] = i
            }
        }
        return maxLength
    }
    static func test() {
        let sut = Leet0325()
        assert(sut.maxSubArrayLen([1,-1,5,-2,3], 3) == 4)
    }
}
//Leet0325.test()





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-the-number-of-good-subarrays/
class Leet2537 {
    func countGood(_ nums: [Int], _ k: Int) -> Int {
        var result = 0, countMap = [Int: Int](), l = 0, pairsCount = 0
        let n = nums.count
        for r in 0..<n {
            let num = nums[r]
            countMap[num, default: 0] += 1
            let c = countMap[num]!
            if c > 1 {
                pairsCount += c - 1
            }
            while pairsCount >= k {
                result += n - r
                let cl = countMap[nums[l]]!
                pairsCount -= cl - 1
                countMap[nums[l]]! -= 1
                if countMap[nums[l]] == 0 {
                    countMap[nums[l]] = nil
                }
                l += 1
            }
        }
        return result
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-subarrays-where-max-element-appears-at-least-k-times/
class Leet2962 {
    func countSubarrays(_ nums: [Int], _ k: Int) -> Int {
        var result = 0, mx = nums.max()!, mxCount = 0, l = 0
        let n = nums.count
        for r in 0..<n {
            let rn = nums[r]
            if rn == mx {
                mxCount += 1
            }
            while mxCount >= k {
                result += n - r
                let ln = nums[l]
                if ln == mx {
                    mxCount -= 1
                }
                l += 1
            }
        }
        return result
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/contiguous-array/
class Leet0525 {
    func findMaxLength(_ nums: [Int]) -> Int {
        let n = nums.count, nums = nums.map { $0 == 0 ? -1 : 1 }
        var result = 0, map = [Int: Int](), sum = 0
        for j in 0..<n {
            let jn = nums[j]
            sum += jn
            let d = sum - jn   // pfx[j] - pfx[i] + nums[i] == 0 ---> pfx[j] == pfx[i] - nums[j]
            if map[d] == nil { // map only the first instance
                map[d] = j
            }
            if let i = map[sum] {
                result = max(result, j - i + 1)
            }
        }
        return result
    }
}





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/best-sightseeing-pair/
class Leet1014 {
    // dp
    func maxScoreSightseeingPair(_ values: [Int]) -> Int {
        let n = values.count
        var maxLeftScore = Array(repeating: 0, count: n), maxScore = 0
        maxLeftScore[0] = values[0]
        for i in 1..<n {
            let currRight = values[i] - i
            maxScore = max(maxScore, maxLeftScore[i-1] + currRight)
            let currLeft = values[i] + i
            maxLeftScore[i] = max(maxLeftScore[i-1], currLeft)
        }
        return maxScore
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-number-of-bad-pairs/
class Leet2364 {
    func countBadPairs(_ nums: [Int]) -> Int {
        let n = nums.count
        var result = 0, map = [Int:Int](), goodPairsCount = 0
        for i in 0..<nums.count {
            let d = i - nums[i]
            if let c = map[d] {
                goodPairsCount += c
            }
            map[d, default: 0] += 1
        }
        result = n * (n-1) / 2 - goodPairsCount
        return result
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-number-of-pairs-with-absolute-difference-k/
class Leet2006 {
    func countKDifference(_ nums: [Int], _ k: Int) -> Int {
        var result = 0
        for i in 0..<nums.count {
            for j in i+1..<nums.count {
                if abs(nums[i] - nums[j]) == k {
                    result += 1
                }
            }
        }
        return result
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-equal-and-divisible-pairs-in-an-array/
class Leet2176 {
    func countPairs(_ nums: [Int], _ k: Int) -> Int {
        var result = 0
        for i in 0..<nums.count {
            for j in i+1..<nums.count {
                if nums[i] == nums[j] && i * j % k == 0 {
                    result += 1
                }
            }
        }
        return result
    }
}






///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-repeating-substring/
class Leet1668 {
    func maxRepeating(_ sequence: String, _ word: String) -> Int {
        guard word.count <= sequence.count else { return 0 }
        let n = sequence.count, m = word.count, sequence = Array(sequence)
        var dp: [Int] = Array(repeating: 0, count: n + 1), result = 0
        for i in m..<n+1 {
            guard String(sequence[i-m..<i]) == word else { continue }
            dp[i] = dp[i-m] + 1
            result = max(result, dp[i])
        }
        return result
    }
    static func test() {
        let sut = Leet1668()
        assert(sut.maxRepeating("aaabaaaabaaabaaaabaaaabaaaabaaaaba", "aaaba") == 5)
        assert(sut.maxRepeating("ababc", "ab") == 2)
    }
}
//Leet1668.test()





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/longest-substring-of-all-vowels-in-order/
class Leet1839 {
    func longestBeautifulSubstring(_ word: String) -> Int {
        let word = Array(word)
        var l = 0, maxLength = 0, window : Set<Character> = [word[0]]
        for r in 1..<word.count {
            if word[r] >= word[r-1] {
                window.insert(word[r])
                if window.count == 5 {
                    maxLength = max(maxLength, r - l + 1)
                }
            } else if word[r] == "a" {
                window = ["a"]
                l = r
            } else {
                window.removeAll()
            }
        }
        return maxLength
    }
    static func test() {
        let sut = Leet1839()
        assert(sut.longestBeautifulSubstring("aeiaaioaaaaeiiiiouuuooaauuaeiu") == 13)
    }
}
//Leet1839.test()






///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-and-say
class Leet0038 {
    func countAndSay(_ n: Int) -> String {
        guard n > 1 else {
            return "1"
        }
        var currentString = "1"
        for _ in 2...n {
            var nextString = ""
            var j = currentString.startIndex
            while j < currentString.endIndex {
                var k = j
                while k < currentString.endIndex && currentString[k] == currentString[j] {
                    k = currentString.index(after: k)
                }
                let count = currentString.distance(from: j, to: k)
                nextString += "\(count)\(currentString[j])"
                j = k
            }
            currentString = nextString
        }
        return currentString
    }
    static func test() {
        let sut = Leet0038()
        assert(sut.countAndSay(1) == "1")
    }
}
//Leet0038.test()


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-the-number-of-fair-pairs/
class Leet2563 {
    private func lowerBound(_ nums: [Int], _ target: Int) -> Int {
        var l = 0, r = nums.count - 1, result = 0
        while l < r {
            let sum = nums[l] + nums[r]
            if sum < target {
                result += r - l
                l += 1
            } else {
                r -= 1
            }
        }
        return result
    }
    
    func countFairPairs(_ nums: [Int], _ lower: Int, _ upper: Int) -> Int {
        let nums = nums.sorted()
        return lowerBound(nums, upper + 1) - lowerBound(nums, lower)
    }
}




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-largest-group/
class Leet1399 {
    func countLargestGroup(_ n: Int) -> Int {
        var largestList = 0, sumMap = [Int: [Int]]() // [sum: [digits]]
        for i in 1...n {
            var num = i, sum = 0
            while num > 0 {
                sum += num % 10
                num /= 10
            }
            sumMap[sum, default: []].append(i)
            largestList = max(largestList, sumMap[sum]!.count)
        }
        return sumMap.values.filter { $0.count == largestList }.count
    }
}






///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/frequency-of-the-most-frequent-element/
class Leet1838 {
    func maxFrequency(_ nums: [Int], _ k: Int) -> Int {
        let nums = nums.sorted()
        var ans = 0, l = 0, curr = 0
        for r in 0..<nums.count {
            let target = nums[r]
            curr += target
            while (r - l + 1) * target - curr > k {
                curr -= nums[l]
                l += 1
            }
            ans = max(ans, r - l + 1)
        }
        return ans
    }
}





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/get-equal-substrings-within-budget
class Leet1208 {
    func equalSubstring(_ s: String, _ t: String, _ maxCost: Int) -> Int {
        let s = Array(s), t = Array(t)
        var result = 0, l = 0, windowCost = 0
        for r in 0..<s.count {
            windowCost += abs(Int(s[r].asciiValue!) - Int(t[r].asciiValue!))
            while windowCost > maxCost {
                windowCost -= abs(Int(s[l].asciiValue!) - Int(t[l].asciiValue!))
                l += 1
            }
            result = max(result, r - l + 1)
        }
        return result
    }
}




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-the-number-of-ideal-arrays/
class Leet2338 {
    
    let mod = 1_000_000_007
    let maxN = 10_010
    let maxP = 15 // 15 prime factors
    var c: [[Int]]
    var sieve: [Int]
    var ps: [[Int]]
    
    init () {
        c = [[Int]](repeating: [Int](repeating: 0, count: maxP + 1), count: maxN + maxP)
        sieve = [Int](repeating: 0, count: maxN)
        ps = [[Int]](repeating: [], count: maxN)
        
        for i in 2..<maxN where sieve[i] == 0 {
            for j in stride(from: i, to: maxN, by: i) where sieve[j] == 0 {
                sieve[j] = i
            }
        }
        
        for i in 2..<maxN {
            var x = i
            while x > 1 {
                let p = sieve[x]
                var count = 0
                while x % p == 0 {
                    x /= p
                    count += 1
                }
                ps[i].append(count)
            }
        }
        
        c[0][0] = 1
        for i in 1..<maxN + maxP {
            c[i][0] = 1
            for j in 1...min(i, maxP) {
                c[i][j] = (c[i - 1][j] + c[i - 1][j - 1]) % mod
            }
        }
    }
    
    func idealArrays(_ n: Int, _ maxValue: Int) -> Int {
        var ans = 0
        for x in 1...maxValue {
            var mul = 1
            for p in ps[x] {
                mul = (mul * c[n + p - 1][p]) % mod
            }
            ans = (ans + mul) % mod
        }
        return ans
    }
}











///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-the-hidden-sequences/
class Leet2415 {
    func numberOfArrays(_ differences: [Int], _ lower: Int, _ upper: Int) -> Int {
        var x = 0, y = 0, curr = 0
        for d in differences {
            curr += d
            x = min(curr, x)
            y = max(curr, y)
            if y - x > upper - lower {
                return 0
            }
        }
        return (upper - lower) - (y - x) + 1
    }
}




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/rabbits-in-forest/
class Leet0781 {
    func numRabbits(_ answers: [Int]) -> Int {
        var freq = [Int:Int](), ans = 0
        for a in answers {
            freq[a, default: 0] += 1
        }
        for (k, v) in freq {
            ans += (k + 1) * ((v  + k) / (k + 1))
        }
        return ans
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/
class Leet1456 {
    func maxVowels(_ s: String, _ k: Int) -> Int {
        let vowels = Set("aeiou"), s = Array(s)
        var count = s[0..<k].count(where: vowels.contains), maxCount = count
        for r in k..<s.count {
            let l  = r - k, cR = s[r], cL = s[l]
            if vowels.contains(cR) {
                count += 1
            }
            if vowels.contains(cL) {
                count -= 1
            }
            maxCount = max(maxCount, count)
        }
        return maxCount
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/range-sum-query-immutable/
class Leet0303 {
    private var prefixSum = [Int]()
    private let nums : [Int]
    init(_ nums: [Int]) {
        self.nums = nums
        prefixSum.append(nums[0])
        for i in 1 ..< nums.count {
            prefixSum.append(nums[i] + prefixSum[i-1])
        }
    }
    func sumRange(_ left: Int, _ right: Int) -> Int {
        prefixSum[right] - prefixSum[left] + nums[left]
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-the-highest-altitude/
class Leet1732 {
    func largestAltitude(_ gain: [Int]) -> Int {
        var sum = 0, maxSum = 0
        for g in gain {
            sum += g
            maxSum = max(maxSum, sum)
        }
        return maxSum
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sum-of-variable-length-subarrays/
class Leet3427 {
    func subarraySum(_ nums: [Int]) -> Int {
        let n = nums.count
        var prefixSum = [nums[0]], sum = 0
        for i in 1..<n {
            prefixSum.append(nums[i] + prefixSum[i-1])
        }
        for i in 0..<n {
            let s = max(0, i - nums[i])
            sum += prefixSum[i] - prefixSum[s] + nums[s]
        }
        return sum
    }
    static func test() {
        let sut = Leet3427()
        assert(sut.subarraySum([2,3,1]) == 11)
    }
}
//Leet3427.test()









///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/n-repeated-element-in-size-2n-array/
class Leet0961 {
    func repeatedNTimes(_ nums: [Int]) -> Int {
        var set = Set<Int>()
        for n in nums {
            if set.contains(n) {
                return n
            }
            set.insert(n)
        }
        return 0
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/degree-of-an-array/
class Leet0697 {
    func findShortestSubArray(_ nums: [Int]) -> Int {
        var freq = [Int: Int](), starts = [Int: Int](), ends = [Int: Int](), maxFreq = 0, shortest = Int.max
        for (i, n) in nums.enumerated() {
            freq[n, default: 0] += 1
            if starts[n] == nil {
                starts[n] = i
            }
            ends[n] = i
            maxFreq = max(maxFreq, freq[n]!)
        }
        for n in nums where freq[n]! == maxFreq {
            shortest = min(shortest, ends[n]! - starts[n]! + 1)
        }
        return shortest
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/subarrays-with-k-different-integers/
class Leet0992 {
    func subarraysWithKDistinct(_ nums: [Int], _ k: Int) -> Int {
        let n = nums.count
        var distinctCounts = [Int](repeating: 0, count: n + 1)
        var total = 0, l = 0, r = 0, curr = 0, k = k
        while r < n {
            distinctCounts[nums[r]] += 1
            if distinctCounts[nums[r]] == 1 {
                k -= 1
            }
            if k < 0 {
                distinctCounts[nums[l]] -= 1
                if distinctCounts[nums[l]] == 0 {
                    k += 1
                }
                l += 1
                curr = 0
            }
            if k == 0 {
                while distinctCounts[nums[l]] > 1 {
                    distinctCounts[nums[l]] -= 1
                    l += 1
                    curr += 1
                }
                total += curr + 1
            }
            r += 1
        }
        return total
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-subarrays-of-length-three-with-a-condition/
class Leet3392 {
    func countSubarrays(_ nums: [Int]) -> Int {
        var result = 0
        for i in 1..<(nums.count-1) where 2 * (nums[i-1] + nums[i+1]) == nums[i]  {
            result += 1
        }
        return result
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/most-frequent-number-following-key-in-an-array/
class Leet2190 {
    func mostFrequent(_ nums: [Int], _ key: Int) -> Int {
        var freq = [Int: Int](), maxFreq = 0
        for i in 0..<nums.count - 1 {
            if nums[i] == key {
                let target = nums[i + 1]
                freq[target, default: 0] += 1
                maxFreq = max(maxFreq, freq[target]!)
            }
        }
        return freq.first(where: { $1 == maxFreq })?.key ?? -1
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sort-the-people/
class Leet2418 {
    func sortPeople(_ names: [String], _ heights: [Int]) -> [String] {
        typealias Person = (name: String, height: Int)
        let people: [Person] = zip(names, heights).map { Person(name: $0, height: $1) }
        return people.sorted { (p1, p2) -> Bool in
            if p1.height != p2.height {
                return p1.height > p2.height
            } else {
                return p1.name < p2.name
            }
        }.map(\.name)
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/counting-elements/
class Leet1426 {
    func countElements(_ arr: [Int]) -> Int {
        var result = 0, set = Set<Int>(arr)
        for n in arr {
            if set.contains(n + 1) {
                result += 1
            }
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/kth-largest-element-in-an-array/
class Leet0215 {
    func findKthLargest(_ nums: [Int], _ k: Int) -> Int {
        var heap = Heap<Int>(nums), result = 0
        for _ in 0..<k {
            if let top = heap.popMax() {
                result = top
            }
        }
        return result
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/top-k-frequent-elements/
class Leet0347 {
    func topKFrequent(_ nums: [Int], _ k: Int) -> [Int] {
       var freq = [Int: Int]()
        for n in nums {
            freq[n, default: 0] += 1
        }
        let sorted = freq.sorted { $0.value > $1.value }
        guard k <= sorted.count else {
            return []
        }
        return Array(sorted[..<k]).map(\.key)
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sort-array-by-increasing-frequency/
class Leet1636 {
    func frequencySort(_ nums: [Int]) -> [Int] {
        var freq: [Int: Int] = [:]
        for num in nums {
            freq[num, default: 0] += 1
        }
        return Array(freq).sorted { (a, b) -> Bool in
            a.value < b.value || (a.value == b.value && a.key > b.key)
        }.flatMap { Array(repeating: $0.key, count: $0.value) }
    }
}





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sort-characters-by-frequency/
class Leet0451 {
    struct HeapItem: Comparable {
        static func < (lhs: HeapItem, rhs: HeapItem) -> Bool {
            lhs.i < rhs.i
        }
        let c: Character
        let i: Int
    }
    
    func frequencySort(_ s: String) -> String {
        let s = Array(s)
        var freq: [Character: Int] = [:], heap = Heap<HeapItem>(), result = [Character]()
        for c in s {
            freq[c, default: 0] += 1
        }
        for f in freq {
            heap.insert(.init(c: f.key, i: f.value))
        }
        while !heap.isEmpty {
            let i = heap.removeMax()
            result.append(contentsOf: Array(repeating: i.c, count: i.i))
        }
        return String(result)
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/first-letter-to-appear-twice/
class Leet2351 {
    func repeatedCharacter(_ s: String) -> Character {
        var set: Set<Character> = []
        for c in s {
            if set.contains(c) {
                return c
            }
            set.insert(c)
        }
        fatalError()
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-subarrays-with-fixed-bounds/
class Leet2444 {
    func countSubarrays(_ nums: [Int], _ minK: Int, _ maxK: Int) -> Int {
        guard minK <= maxK else { return 0 }
        var result = 0, minPos = -1, maxPos = -1, l = -1
        for r in 0..<nums.count {
            let n = nums[r]
            if !(minK...maxK ~= n) {
                l = r
            }
            if n == minK {
                minPos = r
            }
            if n == maxK {
                maxPos = r
            }
            result += max(0, min(maxPos, minPos) - l)
        }
        return result
    }
}




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/range-sum-query-2d-immutable/
class Leet0304 {

    private var matrix: [[Int]]
    init(_ matrix: [[Int]]) {
        self.matrix = matrix
    }
    func sumRegion(_ row1: Int, _ col1: Int, _ row2: Int, _ col2: Int) -> Int {
        matrix[row1...row2].reduce(into: 0) { result, row in
            result += row[col1...col2].reduce(0, +)
        }
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/range-sum-query-2d-mutable/
class Leet0308 {

    private var matrix: [[Int]]
    init(_ matrix: [[Int]]) {
        self.matrix = matrix
    }
    
    func update(_ row: Int, _ col: Int, _ val: Int) {
        matrix[row][col] = val
    }
    
    func sumRegion(_ row1: Int, _ col1: Int, _ row2: Int, _ col2: Int) -> Int {
        matrix[row1...row2].reduce(into: 0) { result, row in
            result += row[col1...col2].reduce(0, +)
        }
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/range-sum-query-mutable/
class Leet0307 {
    private var nums: [Int]
    
    init(_ nums: [Int]) {
        self.nums = nums
    }
    
    func update(_ index: Int, _ val: Int) {
        nums[index] = val
    }
    
    func sumRange(_ left: Int, _ right: Int) -> Int {
        nums[left...right].reduce(0, +)
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-of-interesting-subarrays
/// REVISIT!
class Leet2845 {
    func countInterestingSubarrays(_ nums: [Int], _ modulo: Int, _ k: Int) -> Int {
        let n = nums.count
        var res = 0, prefix = 0, cnt = [Int: Int]()
        cnt[0] = 1
        for i in 0..<n {
            prefix += nums[i] % modulo == k ? 1 : 0
            let m = (prefix - k + modulo) % modulo
            res += cnt[m] ?? 0
            cnt[prefix % modulo, default: 0] += 1
        }
        return res
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/subarray-sums-divisible-by-k/
/// REVISIT!
class Leet0974 {
    func subarraysDivByK(_ nums: [Int], _ k: Int) -> Int {
        var prefixMod = 0, result = 0, modGroups = [Int](repeating: 0, count: k)
        modGroups[0] = 1
        for num in nums {
            prefixMod = (prefixMod + num % k + k) % k
            result += modGroups[prefixMod]
            modGroups[prefixMod] += 1
        }
        return result
    }
}




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/product-of-two-run-length-encoded-arrays/
class Leet1868 {
    func findRLEArray(_ encoded1: [[Int]], _ encoded2: [[Int]]) -> [[Int]] {
        var i1 = 0, i2 = 0, en1 = encoded1, en2 = encoded2, result = [[Int]](), temp = result
        while i1 < en1.count && i2 < en2.count {
            while i1 < en1.count && i2 < en2.count && en1[i1][1] >= en2[i2][1] {
                temp.append([en1[i1][0] * en2[i2][0], en2[i2][1]])
                en1[i1][1] -= en2[i2][1]
                en2[i2][1] = 0
                i2 += 1
                if en1[i1][1] == 0 { i1 += 1 }
            }
            while i1 < en1.count && i2 < en2.count && en1[i1][1] <= en2[i2][1] {
                temp.append([en1[i1][0] * en2[i2][0], en1[i1][1]])
                en2[i2][1] -= en1[i1][1]
                en1[i1][1] = 0
                i1 += 1
                if en2[i2][1] == 0 { i2 += 1 }
            }
        }
        // compress results
        result.append(temp[0])
        guard temp.count > 1 else { return result }
        for i in 1..<temp.count {
            if temp[i][0] == result.last![0] {
                result[result.count - 1][1] += temp[i][1]
            } else {
                result.append(temp[i])
            }
        }
        return result
    }
    
    static func test() {
        let sut = Leet1868()
        assert(sut.findRLEArray([[1,1],[2,1],[1,1],[2,1],[1,1]], [[1,1],[2,1],[1,1],[2,1],[1,1]]) == [[1,1],[4,1],[1,1],[4,1],[1,1]])
        assert(sut.findRLEArray([[1,3],[2,3]], [[6,3],[3,3]]) == [[6,6]])
        assert(sut.findRLEArray([[1,3],[2,1],[3,2]], [[2,3],[3,3]]) == [[2,3],[6,1],[9,2]])
    }
}
//Leet1868.test()





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/subarray-sum-equals-k/
class Leet0560 {
    func subarraySum(_ nums: [Int], _ k: Int) -> Int {
        let n = nums.count
        var count = 0, sum = 0, freq = [Int: Int]()
        freq[0] = 1
        for i in 0..<n {
            sum += nums[i]
            if let c = freq[sum - k] {
                count += c
            }
            freq[sum, default: 0] += 1
        }
        return count
    }
    static func test() {
        let sut = Leet0560()
        assert(sut.subarraySum([1,-1,0], 0) == 3)
        assert(sut.subarraySum([3,4,7,-2,2,1,4,2], 7) == 6)
    }
}
//Leet0560.test()



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-number-of-nice-subarrays/
class Leet1248 {
    func numberOfSubarrays(_ nums: [Int], _ k: Int) -> Int {
        let n = nums.count
        var count = 0, sum = 0, freq = [Int: Int]()
        freq[0] = 1
        for i in 0..<n {
            sum += nums[i] % 2
            if let c = freq[sum - k] {
                count += c
            }
            freq[sum, default: 0] += 1
        }
        return count
    }
    static func test() {
        let sut = Leet1248()
        assert(sut.numberOfSubarrays([1,1,2,1,1], 3) == 2)
        assert(sut.numberOfSubarrays([2,4,6], 1) == 0)
        assert(sut.numberOfSubarrays([2,2,2,1,2,2,1,2,2,2], 2) == 16)
    }
}
//Leet1248.test()

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/binary-subarrays-with-sum/
class Leet0930 {
    func numSubarraysWithSum(_ nums: [Int], _ goal: Int) -> Int {
        let n = nums.count
        var count = 0, sum = 0, map = [Int: Int]()
        map[0] = 1
        
        for i in 0..<n {
            sum += nums[i]
            if let cnt = map[sum - goal] {
                count += cnt
            }
            map[sum, default: 0] += 1
        }
        return count
    }
    
    static func test() {
        let sut = Leet0930()
        assert(sut.numSubarraysWithSum([1,0,1,0,1], 2) == 4)
        assert(sut.numSubarraysWithSum([0,0,0,0,0], 0) == 15)
    }
}
//Leet0930.test()





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/matrix-diagonal-sum/
class Leet1572 {
    func diagonalSum(_ mat: [[Int]]) -> Int {
        var result = 0
        for i in 0..<mat.count {
            // main diagonal sum
            result += mat[i][i]
            
            // anti diagonal sum
            result += mat[i][mat.count - i - 1]
        }
        
        // when odd, remove the middle value
        if !mat.count.isMultiple(of: 2) {
            let mid = mat.count / 2
            result -= mat[mid][mid]
        }
        return result
    }
    
    static func test() {
        let sut = Leet1572()
        assert(sut.diagonalSum([[1,2,3],[4,5,6],[7,8,9]]) == 25)
        assert(sut.diagonalSum([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]) == 8)
        assert(sut.diagonalSum([[5]]) == 5)
    }
}
//Leet1572.test()



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/check-if-matrix-is-x-matrix/
class Leet2319 {
    func checkXMatrix(_ grid: [[Int]]) -> Bool {
        var g2 = [[Int]](repeating: [Int](repeating: 0, count: grid[0].count), count: grid.count)
        for i in 0..<grid.count {
            // check diagonals
            let d = grid[i][i]
            guard d > 0 else { return false }
            g2[i][i] = d
            // check anti-diagonals
            let ad = grid[i][grid.count - i - 1]
            guard ad > 0 else { return false }
            g2[i][grid.count - i - 1] = ad
        }
        return g2 == grid
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/check-if-all-characters-have-equal-number-of-occurrences/
class Leet1941 {
    func areOccurrencesEqual(_ s: String) -> Bool {
        Set(Array(s).reduce(into: [:]) { f, c in f[c, default: 0] += 1 }.values).count == 1
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/rings-and-rods/
class Leet2103 {
    func countPoints(_ rings: String) -> Int {
        let rings = Array(rings), n = rings.count, red = Character("R"), green = Character("G"), blue = Character("B")
        var map: [Character: Set<Character>] = [red: [], green: [], blue: []]
        for i in stride(from: 0, to: n, by: 2) {
            let r = rings[i], g = rings[i + 1]
            map[r]!.insert(g)
        }
        return ( map[red]!.intersection(map[green]!).intersection(map[blue]!) ).count
    }
    static func test() {
        let sut = Leet2103()
        assert(sut.countPoints("B0B6G0R6R0R6G9") == 1)
        assert(sut.countPoints("B0R0G0R9R0B0G0") == 1)
    }
}
//Leet2103.test()


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/intersection-of-multiple-arrays/
class Leet2248 {
    func intersection(_ nums: [[Int]]) -> [Int] {
        Array(nums
            .map(Set.init)
            .reduce(Set(nums[0])) { $0.intersection($1) }
        ).sorted()
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-the-losers-of-the-circular-game/
class Leet2682 {
    func circularGameLosers(_ n: Int, _ k: Int) -> [Int] {
        var receipts = Set<Int>([]), losers = [Int](), prev = 0
        // find winner
        for i in 0..<n {
            let next = (prev + i * k) % n
            prev = next
            if receipts.contains(next) {
                break
            }
            receipts.insert(next)
        }
        // collect the losers
        for i in 0..<n {
            guard !receipts.contains(i) else { continue }
            losers.append(i+1)
        }
        return losers
    }
    static func test() {
        let sut = Leet2682()
        assert(sut.circularGameLosers(3,1) == [3])
        assert(sut.circularGameLosers(2,1) == [])
        assert(sut.circularGameLosers(5,2) == [4,5])
        assert(sut.circularGameLosers(4,4) == [2,3,4])
    }
}
//Leet2682.test()



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/pass-the-pillow/
class Leet2582 {
    func passThePillow(_ n: Int, _ time: Int) -> Int {
        let rounds = time / (n - 1), remainder = time % (n - 1)
        if rounds % 2 == 0 {
            return remainder + 1
        } else {
            return n - remainder
        }
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-the-child-who-has-the-ball-after-k-seconds/
class Leet3178 {
    func numberOfChild(_ n: Int, _ k: Int) -> Int {
        let rounds = k / (n  - 1), remainder = k % (n  - 1)
        if rounds % 2 == 0 {
            return remainder + 1
        } else {
            return n - remainder
        }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/unique-3-digit-even-numbers/
class Leet3483 {
    func totalNumbers(_ digits: [Int]) -> Int {
        var set = Set<Int>()
        let n = digits.count
        for i in 0..<n where digits[i] > 0 {
            for j in 0..<n {
                for k in 0..<n where digits[k].isMultiple(of: 2) {
                    guard i != j, i != k, j != k else { continue }
                    set.insert(digits[i]*100 + digits[j]*10 + digits[k])
                }
            }
        }
        return set.count
    }
    static func test() {
        let sut = Leet3483()
        assert(sut.totalNumbers([1,2,3,4]) == 12)
    }
}
//Leet3483.test()


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/finding-3-digit-even-numbers/
class Leet2094 {
    func findEvenNumbers(_ digits: [Int]) -> [Int] {
        var set = Set<Int>()
        let n = digits.count
        for i in 0..<n where digits[i] > 0 {
            for j in 0..<n {
                for k in 0..<n where digits[k].isMultiple(of: 2) {
                    guard i != j, i != k, j != k else { continue }
                    set.insert(digits[i]*100 + digits[j]*10 + digits[k])
                }
            }
        }
        return set.sorted()
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-if-digit-game-can-be-won/
class Leet3232 {
    func canAliceWin(_ nums: [Int]) -> Bool {
        nums.reduce(0) { $0 + (($1 < 10) ? $1 : 0) } != nums.reduce(0) { $0 + (($1 >= 10) ? $1 : 0) }
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-integers-with-even-digit-sum/
class Leet2180 {
    func countEven(_ num: Int) -> Int {
        (1...num).map { $0 }.filter { $0.sum.isMultiple(of: 2) }.count
    }
}

extension Int {
    var sum: Int {
        var n = self, sum = 0
        while n != 0 {
            sum += n % 10
            n /= 10
        }
        return sum
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/number-of-even-and-odd-bits/
class Leet2595 {
    func evenOddBit(_ n: Int) -> [Int] {
        Array(String(n, radix: 2))
            .reversed()
            .enumerated()
            .reduce(into: [0, 0]) { counts, item in
                guard item.element == "1" else { return }
                if item.offset.isMultiple(of: 2) {
                    counts[0] += 1
                } else {
                    counts[1] += 1
                }
        }
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/intersection-of-three-sorted-arrays/
class Leet1213 {
    func arraysIntersection(_ arr1: [Int], _ arr2: [Int], _ arr3: [Int]) -> [Int] {
        Set(arr1).intersection(Set(arr2)).intersection(Set(arr3)).sorted()
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-the-difference-of-two-arrays/
class Leet2215 {
    func findDifference(_ nums1: [Int], _ nums2: [Int]) -> [[Int]] {
        let n1 = Set(nums1), n2 = Set(nums2), i = n1.intersection(n2)
        return [Array(n1.subtracting(i)), Array(n2.subtracting(i))]
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-smallest-common-element-in-all-rows/
class Leet1198 {
    func smallestCommonElement(_ mat: [[Int]]) -> Int {
        mat.map { Set($0) }.reduce(Set(mat[0])) { $0.intersection($1) }.min() ?? -1
    }
}




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sum-of-digits-of-string-after-convert/
class Leet1945 {
    func getLucky(_ s: String, _ k: Int) -> Int {
        var result = Array(s).map { String(Int($0.asciiValue!) - 96) }.joined().sum
        guard k > 1 else { return result }
        for _ in 1..<k {
            result = result.sum
        }
        return result
    }
    static func test() {
        let sut = Leet1945()
        assert(sut.getLucky("fleyctuuajsr", 5) == 8)
    }
}

extension String {
    var sum: Int { Array(self).reduce(0) { $0 + Int(String($1))! } }
}
//Leet1945.test()



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/calculate-digit-sum-of-a-string/
class Leet2243 {
    func digitSum(_ s: String, _ k: Int) -> String {
        var result = Array(s)
        while result.count > k {
            var round = ""
            for i in stride(from: 0, to: result.count, by: k) {
                let group: String
                if i + k < result.count {
                    group = String(result[i..<(i+k)])
                } else {
                    group = String(result[i...])
                }
                round += String(group.sum)
            }
            result = Array(round)
        }
        return String(result)
    }
    
    static func test() {
        let sut = Leet2243()
        assert(sut.digitSum("11111222223", 3) == "135")
    }
}
//Leet2243.test()



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-sum-of-four-digit-number-after-splitting-digits/
class Leet2160 {
    func minimumSum(_ num: Int) -> Int {
        let a = Array(String(num)).sorted()
        return Int("\(a[0])\(a[3])")! + Int("\(a[1])\(a[2])")!
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/alternating-digit-sum/
class Leet2544 {
    func alternateDigitSum(_ n: Int) -> Int {
        var n = n, list = [Int]()
        while n > 0 {
            list.append(n % 10)
            n /= 10
        }
        return list.reversed().enumerated().reduce(0) { $0 + ($1.offset.isMultiple(of: 2) ? $1.element : -$1.element) }
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-number-of-tasks-you-can-assign/
///REVISIT!!
class Leet2071 {
    func maxTaskAssign(_ tasks: [Int], _ workers: [Int], _ pills: Int, _ strength: Int) -> Int {
        let n = tasks.count, m = workers.count, tasks = tasks.sorted(), workers = workers.sorted()
        var l = 1, r = min(m, n), ans = 0
        while l <= r {
            let mid = (l + r) / 2
            if check(tasks, workers, pills, strength, mid) {
                ans = mid
                l = mid + 1
            } else {
                r = mid - 1
            }
        }
        return ans
    }
    
    private func check(_ tasks: [Int], _ workers: [Int], _ pills: Int, _ strength: Int, _ mid: Int) -> Bool {
        let m = workers.count
        var p = pills, ptr = m - 1, ws = Deque<Int>()
        for i in stride(from: mid-1, through: 0, by: -1) {
            while (ptr >= m - mid && workers[ptr] + strength >= tasks[i]) {
                ws.prepend(workers[ptr])
                ptr -= 1
            }
            if ws.isEmpty {
                return false
            } else if ws.last! >= tasks[i] {
                ws.removeLast()
            } else {
                if p == 0 {
                    return false
                }
                p -= 1
                ws.removeFirst()
            }
        }
        return true
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/separate-the-digits-in-an-array/
class Leet2553 {
    func separateDigits(_ nums: [Int]) -> [Int] {
        nums.map { $0.digitsReversed }.flatMap { $0 }
    }
}

extension Int {
    var digits: [Int] {
        var result: [Int] = [], num = self
        while num != 0 {
            result.append(num % 10)
            num /= 10
        }
        return result
    }
    
    var digitsReversed: [Int] { digits.reversed() }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/ugly-number/
class Leet0263 {
    func isUgly(_ n: Int) -> Bool {
        guard n > 0 else { return false }
        var n = n
        n = keepDividingWhenDivisible(n, 2)
        n = keepDividingWhenDivisible(n, 3)
        n = keepDividingWhenDivisible(n, 5)
        return n == 1
    }
    
    private func keepDividingWhenDivisible(_ dividend: Int, _ divisor: Int) -> Int {
        var dividend = dividend
        while dividend % divisor == 0 {
            dividend /= divisor
        }
        return dividend
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-element-after-replacement-with-digit-sum/
class Leet3300 {
    func minElement(_ nums: [Int]) -> Int {
        nums.map { $0.sum }.min() ?? 0
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-the-digits-that-divide-a-number/
class Leet2520 {
    func countDigits(_ num: Int) -> Int {
        num.digits.compactMap { num % $0 == 0 ? 1 : nil }.count
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/self-dividing-numbers/
class Leet0728 {
    func selfDividingNumbers(_ left: Int, _ right: Int) -> [Int] {
        (left...right).compactMap { num in num.digits.allSatisfy { $0 != 0 && num.isMultiple(of: $0) } ? num : nil  }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/perfect-number/
class Leet0507 {
    func checkPerfectNumber(_ num: Int) -> Bool {
        Set([6, 28, 496, 8128, 33550336]).contains(num)
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/push-dominoes/
class Leet0838 {
        
    func pushDominoes(_ dominoes: String) -> String {
        typealias State = (c: Character, rDistance: Int?)
        var stack = [State](), result = [Character](), dominoes = Array(dominoes)
        for c in dominoes {
            if c == "." {
                if let top = stack.last {
                    if top.c == "R" {
                        stack.append(State(c, 1))
                    } else if top.c == "." {
                        var distance: Int? = nil
                        if let rDistance = top.rDistance {
                            distance = rDistance + 1
                        }
                        stack.append(State(c, distance))
                    }
                } else {
                    stack.append(State(c, nil))
                }
            } else if c == "R" {
                if let top = stack.last {
                    if top.c == "." {
                        if let rDistance = top.rDistance {
                        // "R...R"
                            result.append(contentsOf: Array(repeating: "R", count: rDistance + 1))
                            stack.removeAll()
                        } else {
                        // "...R"
                            result.append(contentsOf: stack.map(\.c))
                            stack.removeAll()
                        }
                        stack.append(State(c, nil))
                    } else if top.c == "R" {
                        result.append(stack.removeLast().c)
                        stack.append(State(c, nil))
                    }
                } else {
                    stack.append(State(c, nil))
                }
            } else if c == "L" {
                // clear the stack and convert dots and Rs
                var stk2 = [Character](["L"])
                if let top = stack.last {
                    if top.c == "R" {
                        stk2.append("R")
                        stack.removeLast()
                    } else if top.c == "." {
                        if let rDistance = top.rDistance {
                            // "R...L", "R.L", "R..L"
                            let mid = rDistance / 2
                            for _ in 0..<mid {
                                stk2.append("L")
                                stack.removeLast()
                            }
                            if rDistance % 2 == 1 {
                                stk2.append(stack.removeLast().c)
                            }
                            for _ in 0..<mid {
                                stk2.append("R")
                                stack.removeLast()
                            }
                            stk2.append(stack.removeLast().c)
                        } else {
                            // "....L"
                            while !stack.isEmpty {
                                if let top2 = stack.last, top2.c == "." {
                                    stk2.append("L")
                                    stack.removeLast()
                                }
                            }
                        }
                    }
                }
                while !stk2.isEmpty {
                    result.append(stk2.removeLast())
                }
            }
        }
        while !stack.isEmpty {
            if let top = stack.last {
                if top.c == "." {
                    if let rDistance = top.rDistance {
                        // "R..."
                        result.append(contentsOf: Array(repeating: "R", count: rDistance + 1))
                    } else {
                        // "...."
                        result.append(contentsOf: stack.map(\.c))
                    }
                    stack.removeAll()
                } else if top.c == "R" {
                    // "R"
                    result.append(stack.removeLast().c)
                }
            }
        }
        return String(result)
    }
    
    static func test(){
        let sut = Leet0838()
        assert(sut.pushDominoes("RL.R..R.") == "RL.RRRRR")
        assert(sut.pushDominoes("R...R") == "RRRRR")
        assert(sut.pushDominoes("L...R") == "L...R")
        assert(sut.pushDominoes("R...L") == "RR.LL")
        assert(sut.pushDominoes("...R") == "...R")
        assert(sut.pushDominoes(".L.R...LR..L..") == "LL.RR.LLRRLL..")
        assert(sut.pushDominoes("RR.L") == "RR.L")
    }
}
//Leet0838.test()


/*
 "RR.L"
 ".L.R...LR..L.."
 "R...L"
 "...."
 "R..."
 "L..."
 "...R"
 "...L"
 
 
 "."
 "..."
 "L.R.L.R"
 "R.L.R.L"
 "LLLLLLLLLL"
 ".......LR...LR.R......L"
 "LL...R.L.......R.LR..L..RL.R..R.L...LRR.LR.L.R...R"
 ".R....RLRR......RL...L..L....R.L.......L..R.....L........RL.L..LR......L...L..RL.R...LRL.....R......"
 
 */



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/check-if-number-has-equal-digit-count-and-digit-value/
class Leet2283 {
    func digitCount(_ num: String) -> Bool {
        let digits = Array(num).compactMap { $0.wholeNumberValue }, freq = digits.reduce(into: [:]) { c, d in c[d, default: 0] += 1 }
        return digits.indices.allSatisfy { digits[$0] == (freq[$0] ?? 0) }
    }
    static func test() {
        let sut = Leet2283()
        assert(sut.digitCount("1210") == true)
        assert(sut.digitCount("030") == false)
    }
}
//Leet2283.test()



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/buddy-strings/
class Leet0859 {
    func buddyStrings(_ s: String, _ goal: String) -> Bool {
        guard s.count == goal.count else { return false }
        var freq = [Character: Int](), s = Array(s), diffs = [Int]() //diffCount = 0
        let goal = Array(goal)
        for i in 0..<s.count {
            let sc = s[i], gc = goal[i]
            freq[sc, default: 0] += 1
            if sc != gc {
                diffs.append(i)
            }
        }
        guard diffs.count <= 2 else { return false }
        if s == goal {
            return Set(freq.values).max()! > 1
        } else {
            guard diffs.count == 2 else { return false }
            s.swapAt(diffs[0], diffs[1])
            return s == goal
        }
    }
}

/*
 Test cases:
 
"abcd"
"abcd"
"aabb"
"aabb"
"abc"
"ab"
"abcd"
"abed"
"abdcgf"
"abcdfg"

"abcaa"
"abcbb"
*/


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/check-if-one-string-swap-can-make-strings-equal/
class Leet1790 {
    func areAlmostEqual(_ s1: String, _ s2: String) -> Bool {
        guard s1 != s2 else { return true }
        guard s1.count == s2.count else { return false }
        var s1 = Array(s1), diffs = [Int]()
        let s2 = Array(s2)
        for i in 0..<s1.count {
            let sc = s1[i], gc = s2[i]
            if sc != gc {
                diffs.append(i)
            }
        }
        guard diffs.count == 2 else { return false }
        s1.swapAt(diffs[0], diffs[1])
        return s1 == s2
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-domino-rotations-for-equal-row/
class Leet1007 {
    
    func minDominoRotations(_ tops: [Int], _ bottoms: [Int]) -> Int {
        min(minRotations(tops, bottoms), minRotations(bottoms, tops))
    }
    private func minRotations(_ tops: [Int], _ bottoms: [Int]) -> Int {
        let n = tops.count
        var topMaxCount = 0
        let topCounts: [Int: Int] = tops.reduce(into: [:]) { counts, top in
            counts[top, default: 0] += 1
            topMaxCount = max(topMaxCount, counts[top]!)
        }
        let topMaxes = topCounts.filter { $1 == topMaxCount }.keys
        for t in topMaxes {
            var result = 0
            for i in 0..<n where tops[i] != t && bottoms[i] == t {
                result += 1
            }
            if n - topMaxCount == result {
                return result
            }
        }
        return -1
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/neither-minimum-nor-maximum/
class Leet2733 {
    func findNonMinOrMax(_ nums: [Int]) -> Int {
        guard nums.count > 2 else { return -1 }
        let minValue = nums.min()!, maxValue = nums.max()!
        for n in nums where n != minValue && n != maxValue {
            return n
        }
        return -1
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/ant-on-the-boundary/
class Leet3028 {
    func returnToBoundaryCount(_ nums: [Int]) -> Int {
        var sum = 0, result = 0
        for num in nums {
            sum += num
            result += sum == 0 ? 1 : 0
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/faulty-keyboard/
class Leet2810 {
    func finalString(_ s: String) -> String {
        let s = Array(s)
        var result = [Character]()
        for c in s {
            if c != "i" {
                result.append(c)
            } else {
                result.reverse()
            }
        }
        return String(result)
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-the-original-typed-string-i/
class Leet3330 {
    func possibleStringCount(_ word: String) -> Int {
        let word = Array(word)
        var result = 1
        guard word.count > 1 else { return result }
        for i in 1..<word.count where word[i] == word[i - 1] {
            result += 1
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/keyboard-row/
class Leet0500 {
    func findWords(_ words: [String]) -> [String] {
        let keyboard = ["qwertyuiop", "asdfghjkl", "zxcvbnm"].map(Set.init), wordsLower = words.map { Array($0.lowercased()) }
        var result = [String]()
        for (i, word) in wordsLower.enumerated() {
            let isWordInRow = keyboard.reduce(false) { (res, row) in res || word.allSatisfy { c in row.contains(c) } }
            guard isWordInRow else { continue }
            result.append(words[i])
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/number-of-equivalent-domino-pairs
class Leet1128 {
    func numEquivDominoPairs(_ dominoes: [[Int]]) -> Int {
        var num = Array(repeating: 0, count: 100), result = 0
        for d in dominoes {
            let v = d[0] < d[1] ? d[0] * 10 + d[1] : d[1] * 10 + d[0]
            result += num[v]
            num[v] += 1
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/domino-and-tromino-tiling
class Leet0790 {
    func numTilings(_ n: Int) -> Int {
        let mod = 1_000_000_007
        guard n > 2 else { return n }
        var fCurr = 5, fPrev = 2, fBeforePrev = 1, k = 4
        while k < n + 1 {
            k += 1
            let tmp = fPrev
            fPrev = fCurr
            fCurr = (2 * fCurr + fBeforePrev) % mod
            fBeforePrev = tmp
        }
        return fCurr % mod
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/build-array-from-permutation/
class Leet1920 {
    func buildArray(_ nums: [Int]) -> [Int] {
        nums.map { nums[$0] }
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-addition-to-make-integer-beautiful/
class Leet2457 {
    func makeIntegerBeautiful(_ n: Int, _ target: Int) -> Int {
        var order = 10, x = 0
        while !isBeautiful(n + x, target) {
            x = order - (n % order)
            order *= 10
        }
        return x
    }
    private func isBeautiful(_ n: Int, _ target: Int) -> Bool {
        n.sum <= target
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/difference-between-element-sum-and-digit-sum-of-an-array/
class Leet2535 {
    func differenceOfSum(_ nums: [Int]) -> Int {
        abs(nums.reduce(0, +) - nums.reduce(0) { $0 + $1.sum })
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/get-maximum-in-generated-array/
class Leet1646 {
    func getMaximumGenerated(_ n: Int) -> Int {
        guard n > 1 else { return n }
        var result = 1, nums = [0, 1]
        for i in 2...n {
            let half = i / 2
            if i.isMultiple(of: 2) {
                nums.append(nums[half])
            } else {
                nums.append(nums[half] + nums[half + 1])
            }
            result = max(result, nums.last!)
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/number-of-ways-to-buy-pens-and-pencils/
class Leet2240 {
    func waysToBuyPensPencils(_ total: Int, _ cost1: Int, _ cost2: Int) -> Int {
        var i = 0, hiCost = max(cost1, cost2), hiTotal = 0, loCost = min(cost1, cost2), result = 0
        while hiTotal <= total {
            result += (total - hiTotal) / loCost + 1
            i += 1
            hiTotal = i * hiCost
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/make-number-of-distinct-characters-equal/
class Leet2531 {
    func isItPossible(_ word1: String, _ word2: String) -> Bool {
        let word1 = Array(word1), word2 = Array(word2)
        var freq1 = word1.reduce(into: [:]) { $0[$1, default: 0] += 1 }
        var freq2 = word2.reduce(into: [:]) { $0[$1, default: 0] += 1 }
        for (k1, v1) in freq1 {
            for (k2, v2) in freq2 {
                var temp1 = freq1, temp2 = freq2

                temp1[k1, default: 0] -= 1
                temp2[k1, default: 0] += 1
                if temp1[k1]! == 0 { temp1[k1] = nil }
                                
                temp2[k2, default: 0] -= 1
                temp1[k2, default: 0] += 1
                if temp2[k2]! == 0 { temp2[k2] = nil }
                                
                if temp1.count == temp2.count {
                    return true
                }
            }
        }
        return false
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-players-with-zero-or-one-losses/
class Leet2225 {
    func findWinners(_ matches: [[Int]]) -> [[Int]] {
        var winsMap = [Int: Int](), losMap = [Int: Int]()
        for m in matches {
            winsMap[m[0], default: 0] += 1
            losMap[m[1], default: 0] += 1
        }
        return [winsMap.keys.filter { losMap[$0] == nil }.sorted(),
                losMap.keys.filter { losMap[$0] == 1 }.sorted()]
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/transformed-array/
class Leet3379 {
    func constructTransformedArray(_ nums: [Int]) -> [Int] {
        var result = nums
        for i in 0..<nums.count where nums[i] != 0 {
            let n = nums[i]
            if n > 0 {
                let j = (i + n) % nums.count
                result[i] = nums[j]
            } else {
                let j = ((i + n) %  nums.count + nums.count) % nums.count
                result[i] = nums[j]
            }
        }
        return result
    }
    static func test() {
        let sut = Leet3379()
        assert(sut.constructTransformedArray([-10,-10,-4]) == [-4,-10,-10])
        assert(sut.constructTransformedArray([3,-2,1,1]) == [1,1,1,3])
        assert(sut.constructTransformedArray([-1,4,-1]) == [-1,-1,4])
        assert(sut.constructTransformedArray([-10,-10]) == [-10,-10])
    }
}
//Leet3379.test()


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-minimum-time-to-reach-last-room-i
class Leet3341 {
    struct State: Comparable {
        let x: Int, y: Int, time: Int
        static func < (lhs: State, rhs: State) -> Bool {
            lhs.time < rhs.time
        }
    }
        
    func minTimeToReach(_ moveTime: [[Int]]) -> Int {
        let n = moveTime.count, m = moveTime[0].count
        var d = [[Int]](repeating: [Int](repeating: Int.max, count: m), count: n)
        var v = [[Bool]](repeating: [Bool](repeating: false, count: m), count: n)
        let dirs = [(1,0), (-1,0), (0,1), (0,-1)]
        d[0][0] = 0
        var q: Heap<State> = [.init(x: 0, y: 0, time: 0)]
        while let s = q.popMin() {
            guard !v[s.x][s.y] else { continue }
            v[s.x][s.y] = true
            for (dx, dy) in dirs {
                let nx = s.x + dx, ny = s.y + dy
                guard 0..<n ~= nx, 0..<m ~= ny else { continue }
                let time = max(d[s.x][s.y], moveTime[nx][ny]) + 1
                guard d[nx][ny] > time else { continue }
                d[nx][ny] = time
                q.insert(.init(x: nx, y: ny, time: time))
            }
        }
        return d[n-1][m-1]
    }
    static func test() {
        let sut = Leet3341()
        assert(sut.minTimeToReach([[56,93], [3,38]]) == 39)
    }
}
//Leet3341.test()

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-minimum-time-to-reach-last-room-ii/
class Leet3342 {
    struct State: Comparable {
        let x: Int, y: Int, time: Int
        static func < (lhs: State, rhs: State) -> Bool {
            lhs.time < rhs.time
        }
    }
        
    func minTimeToReach(_ moveTime: [[Int]]) -> Int {
        let n = moveTime.count, m = moveTime[0].count
        var d = [[Int]](repeating: [Int](repeating: Int.max, count: m), count: n)
        var v = [[Bool]](repeating: [Bool](repeating: false, count: m), count: n)
        let dirs = [(1,0), (-1,0), (0,1), (0,-1)]
        d[0][0] = 0
        var q: Heap<State> = [.init(x: 0, y: 0, time: 0)]
        while let s = q.popMin() {
            guard !v[s.x][s.y] else { continue }
            v[s.x][s.y] = true
            for (dx, dy) in dirs {
                let nx = s.x + dx, ny = s.y + dy, offset = (nx % 2 == 0) ? ((ny % 2 == 0) ? 2 : 1) : ((ny % 2 == 0) ? 1 : 2)
                guard 0..<n ~= nx, 0..<m ~= ny else { continue }
                let time = max(d[s.x][s.y], moveTime[nx][ny]) + offset
                guard d[nx][ny] > time else { continue }
                d[nx][ny] = time
                q.insert(.init(x: nx, y: ny, time: time))
            }
        }
        return d[n-1][m-1]
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-nearest-point-that-has-the-same-x-or-y-coordinate/
class Leet1779 {
    func nearestValidPoint(_ x: Int, _ y: Int, _ points: [[Int]]) -> Int {
        points
            .enumerated()
            .filter { (_, v) in v[0] == x || v[1] == y }
            .map { (i, v) in (i: i, d: abs(v[0] - x) + abs(v[1] - y)) }
            .sorted { $0.d < $1.d }
            .first?.i ?? -1
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-operations-to-make-the-array-increasing/
class Leet1827 {
    func minOperations(_ nums: [Int]) -> Int {
        var result = 0, nums = nums
        guard nums.count > 1 else { return result }
        for i in 1..<nums.count where nums[i] <= nums[i-1] {
            let diff = nums[i-1] - nums[i] + 1
            nums[i] += diff
            result += diff
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-operations-to-make-columns-strictly-increasing/
class Leet3402 {
    
    private func minOperations(_ nums: [Int]) -> Int {
        var result = 0, nums = nums
        guard nums.count > 1 else { return result }
        for i in 1..<nums.count where nums[i] <= nums[i-1] {
            let diff = nums[i-1] - nums[i] + 1
            nums[i] += diff
            result += diff
        }
        return result
    }
    
    func minimumOperations(_ grid: [[Int]]) -> Int {
        var result = 0
        for col in grid[0].indices {
            var columnValues: [Int] = []
            for row in grid.indices {
                columnValues.append(grid[row][col])
            }
            result += minOperations(columnValues)
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/largest-unique-number/
class Leet1133 {
    func largestUniqueNumber(_ nums: [Int]) -> Int {
        nums.reduce(into: [:]) { $0[$1, default: 0] += 1 }.filter { $1 == 1 }.keys.max() ?? -1
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/design-neighbor-sum-service/
class Leet3242 {
    
    private var map = [Int: (x: Int, y: Int)]()
    private let n: Int, m: Int, grid: [[Int]]
    private func sum(_ value: Int, _ vector: [(Int, Int)]) -> Int {
        guard let p = map[value] else { return 0 }
        var result = 0
        for (dx, dy) in vector {
            let nx = p.x + dx, ny = p.y + dy
            guard 0..<n ~= nx, 0..<m ~= ny else { continue }
            result += grid[nx][ny]
        }
        return result
    }
        
    init(_ grid: [[Int]]) {
        n = grid.count
        m = grid[0].count
        self.grid = grid
        for x in 0..<grid.count {
            for y in 0..<grid[x].count {
                map[grid[x][y]] = (x: x, y: y)
            }
        }
    }
    
    func adjacentSum(_ value: Int) -> Int { sum(value, [(0, 1), (0, -1), (1, 0), (-1, 0)]) }
    func diagonalSum(_ value: Int) -> Int { sum(value, [(1, 1), (1, -1), (-1, 1), (-1, -1)]) }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-number-of-balloons/
class Leet1189 {
    func maxNumberOfBalloons(_ text: String) -> Int {
        let counts = text.reduce(into: [:]) { counts, c in counts[c, default: 0] += 1 }.filter { "balloon".contains($0.key) }
        let factors = "balloon".reduce(into: [:]) { counts, c in counts[c, default: 0] += 1 }
        return factors.reduce(into: Int.max) { result, pair in result = min(result, (counts[pair.key] ?? 0) / pair.value) }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/rearrange-characters-to-make-target-string/
class Leet2287 {
    func rearrangeCharacters(_ s: String, _ target: String) -> Int {
        let counts = s.reduce(into: [:]) { counts, c in counts[c, default: 0] += 1 }.filter { target.contains($0.key) }
        let factors = target.reduce(into: [:]) { counts, c in counts[c, default: 0] += 1 }
        return factors.reduce(into: Int.max) { result, pair in result = min(result, (counts[pair.key] ?? 0) / pair.value) }    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-words-that-can-be-formed-by-characters/
class Leet1160 {
    func countCharacters(_ words: [String], _ chars: String) -> Int {
        let charsCount = chars.reduce(into: [Character: Int]()) { counts, c in counts[c, default: 0] += 1 }
        var result = 0
        for w in words {
            let wCount = w.reduce(into: [Character: Int]()) { counts, c in counts[c, default: 0] += 1 }
            guard wCount.allSatisfy({ charsCount[$0.key] ?? 0 >= $0.value }) else { continue }
            result += w.count
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/check-balanced-string/
class Leet3340 {
    func isBalanced(_ num: String) -> Bool {
        let nums = Array(num).map { Int(String($0))! }
        return nums.indices.reduce(into: 0) { (sum, i) in sum += (i % 2 == 0) ? nums[i] : 0 }
            == nums.indices.reduce(into: 0) { (sum, i) in sum += (i % 2 != 0) ? nums[i] : 0 }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sentence-similarity/
class Leet0734 {
    func areSentencesSimilar(_ sentence1: [String], _ sentence2: [String], _ similarPairs: [[String]]) -> Bool {
        guard sentence1.count == sentence2.count else { return false }
        let d = similarPairs.reduce(into: [String: Set<String>]()) { r, p in r[p[0], default: []].insert(p[1]) }
        let isSimilar: (String, String) -> Bool = { d[$0]?.contains($1) ?? false || d[$1]?.contains($0) ?? false || $0 == $1 }
        return zip(sentence1, sentence2).allSatisfy { isSimilar($0, $1) }
    }
    static func test() {
        let sut = Leet0734()
        assert(sut.areSentencesSimilar(["this","summer","thomas","get","actually","actually","rich","and","possess","the","actually","great","and","fine","vehicle","every","morning","he","drives","one","nice","car","around","one","great","city","to","have","single","super","excellent","super","as","his","brunch","but","he","only","eat","single","few","fine","food","as","some","fruits","he","wants","to","eat","an","unique","and","actually","healthy","life"],
            ["this","summer","thomas","get","very","very","rich","and","possess","the","very","fine","and","well","car","every","morning","he","drives","a","fine","car","around","unique","great","city","to","take","any","really","wonderful","fruits","as","his","breakfast","but","he","only","drink","an","few","excellent","breakfast","as","a","super","he","wants","to","drink","the","some","and","extremely","healthy","life"],
            [["good","nice"],["good","excellent"],["good","well"],["good","great"],["fine","nice"],["fine","excellent"],["fine","well"],["fine","great"],["wonderful","nice"],["wonderful","excellent"],["wonderful","well"],["wonderful","great"],["extraordinary","nice"],["extraordinary","excellent"],["extraordinary","well"],["extraordinary","great"],["one","a"],["one","an"],["one","unique"],["one","any"],["single","a"],["single","an"],["single","unique"],["single","any"],["the","a"],["the","an"],["the","unique"],["the","any"],["some","a"],["some","an"],["some","unique"],["some","any"],["car","vehicle"],["car","automobile"],["car","truck"],["auto","vehicle"],["auto","automobile"],["auto","truck"],["wagon","vehicle"],["wagon","automobile"],["wagon","truck"],["have","take"],["have","drink"],["eat","take"],["eat","drink"],["entertain","take"],["entertain","drink"],["meal","lunch"],["meal","dinner"],["meal","breakfast"],["meal","fruits"],["super","lunch"],["super","dinner"],["super","breakfast"],["super","fruits"],["food","lunch"],["food","dinner"],["food","breakfast"],["food","fruits"],["brunch","lunch"],["brunch","dinner"],["brunch","breakfast"],["brunch","fruits"],["own","have"],["own","possess"],["keep","have"],["keep","possess"],["very","super"],["very","actually"],["really","super"],["really","actually"],["extremely","super"],["extremely","actually"]]))
    }
}
//Leet0734.test()

/*
 ["great","acting","skills"]
 ["fine","drama","talent"]
 [["great","fine"],["drama","acting"],["skills","talent"]]
 ["great"]
 ["great"]
 []
 ["great"]
 ["doubleplus","good"]
 [["great","doubleplus"]]
 ["one","excellent","meal"]
 ["one","good","dinner"]
 [["great","good"],["extraordinary","good"],["well","good"],["wonderful","good"],["excellent","good"],["fine","good"],["nice","good"],["any","one"],["some","one"],["unique","one"],["the","one"],["an","one"],["single","one"],["a","one"],["truck","car"],["wagon","car"],["automobile","car"],["auto","car"],["vehicle","car"],["entertain","have"],["drink","have"],["eat","have"],["take","have"],["fruits","meal"],["brunch","meal"],["breakfast","meal"],["food","meal"],["dinner","meal"],["super","meal"],["lunch","meal"],["possess","own"],["keep","own"],["have","own"],["extremely","very"],["actually","very"],["really","very"],["super","very"]]
 ["an","extraordinary","meal"]
 ["a","good","dinner"]
 [["great","good"],["extraordinary","good"],["well","good"],["wonderful","good"],["excellent","good"],["fine","good"],["nice","good"],["any","one"],["some","one"],["unique","one"],["the","one"],["an","one"],["single","one"],["a","one"],["truck","car"],["wagon","car"],["automobile","car"],["auto","car"],["vehicle","car"],["entertain","have"],["drink","have"],["eat","have"],["take","have"],["fruits","meal"],["brunch","meal"],["breakfast","meal"],["food","meal"],["dinner","meal"],["super","meal"],["lunch","meal"],["possess","own"],["keep","own"],["have","own"],["extremely","very"],["actually","very"],["really","very"],["super","very"]]
 ["fine", "apple", "orange"]
 ["well", "food", "food"]
 [["fine","well"],["fine","great"],["apple","food"],["orange","food"]]
 */


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sentence-similarity-ii/
class Leet0737 {
    func areSentencesSimilarTwo(_ sentence1: [String], _ sentence2: [String], _ similarPairs: [[String]]) -> Bool {
        guard sentence1.count == sentence2.count else { return false }
        let d = similarPairs.reduce(into: [String: Set<String>]()) { r, p in
            r[p[0], default: []].insert(p[1])
            r[p[1], default: []].insert(p[0])
        }
        func isSimilar(_ w1: String, _ w2: String, _ seen: inout Set<String>) -> Bool {
            guard w1 != w2 else { return true }
            guard let s = d[w1], !seen.contains(w1) else { return false }
            seen.insert(w1)
            // traverse the "d" dictionary until we find w2
            return s.reduce(into: false) { ok, w in ok = ok || w == w2 || isSimilar(w, w2, &seen) }
        }
        return zip(sentence1, sentence2).allSatisfy {
            var seen: Set<String> = []
            return isSimilar($0, $1, &seen)
        }
    }
    static func test() {
        let sut = Leet0737()
        assert(sut.areSentencesSimilarTwo(["I","love","leetcode"], ["I","love","onepiece"], [["manga","onepiece"],["platform","anime"],["leetcode","platform"],["anime","manga"]]))
        assert(sut.areSentencesSimilarTwo(["great", "acting", "skills"], ["fine","drama","talent"], [["great","good"],["fine","good"],["drama","acting"],["skills","talent"]]))
    }
}
//Leet0737.test()





///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-resultant-array-after-removing-anagrams/
class Leet2273 {
    func removeAnagrams(_ words: [String]) -> [String] {
        var result = [words[0]], last = words[0].sortedCharacters
        guard words.count > 1 else { return result }
        for i in 1..<words.count {
            let current = words[i].sortedCharacters
            if current != last {
                result.append(words[i])
                last = current
            }
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-the-number-of-vowel-strings-in-range/
class Leet2586 {
    func vowelStrings(_ words: [String], _ left: Int, _ right: Int) -> Int {
        words[left...right]
            .count(where: {
                "aeiou".contains(String($0.prefix(1)))
                && "aeiou".contains(String($0.suffix(1))) }
            )
    }
}




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/max-sum-of-a-pair-with-equal-sum-of-digits/
class Leet2342 {
    func maximumSum(_ nums: [Int]) -> Int {
        let digitSums = nums.reduce(into: [Int: Heap<Int>]()) { r, n in
            let s = n.sum
            r[s, default: Heap<Int>()].insert(n)
            guard let c = r[s]?.count, c > 2 else { return }
            r[s]?.removeMin()
        }
        return digitSums.values.compactMap { l in
            guard l.count > 1 else { return nil }
            return l.min! + l.max!
        }.max() ?? -1
    }
    static func test() {
        let sut = Leet2342()
        assert(sut.maximumSum([18,43,36,13,7]) == 54)
        assert(sut.maximumSum([229,398,269,317,420,464,491,218,439,153,482,169,411,93,147,50,347,210,251,366,401]) == 973)
        assert(sut.maximumSum([368,369,307,304,384,138,90,279,35,396,114,328,251,364,300,191,438,467,183]) == 835)
        assert(sut.maximumSum([10,12,19,14]) == -1)
    }
}
//Leet2342.test()


/**
 
 [279,169,463,252,94,455,423,315,288,64,494,337,409,283,283,477,248,8,89,166,188,186,128]
 [5,1,6]
 [4,6,10,6]
 [2,1,5,5,2,4]
 [368,369,307,304,384,138,90,279,35,396,114,328,251,364,300,191,438,467,183]
 [229,398,269,317,420,464,491,218,439,153,482,169,411,93,147,50,347,210,251,366,401]
 [809039901,892618095,699694397,576724044,699515542,831899037,959450091,88124465,102780641,275884357,658771111,660539885,620862925,263622781,778473545,672947452,521367711,970040373,895455228,886907524,781592735,528179525,100136578,646624289,523918444,628999419,931048268,991029445,631668102,667259810,535380751,786735115,971553625,797004919,81520773,137283330,846189211,464238688,713970439,286355524,482704479,527261737,383453409,217307241,601715229,828501551,256079369,567779582,770290899,264325638,778183437,411538949,798508462,831231181,56846075,112379513,259195786,67218178,957517878,911879358,119232266,891855628,438001321,732866407,521986754,5058581,912946383,243362613,899499777,226815070,285361727,44274463]
 [247351159,161472930,511016654,373543290,997057335,290034168,352251270,790223491,680595472,629849481,986800407,361203066,342541499,886976902,644157918,684791042,582956271,311237471,759333957,223323355,732073320,629074391,61613921,10436269,50997727,247270638,408201132,960491117,308429878,351768956,397982620,873519490,874345684,878636461,552793699,326735218,103296513,769241373,146011530,747509017,797985783,653625162,859835868,772594491,771403414,148303734,184662811,551724519,365553847,254886387,697774826,321561332,233009893,449595293,133556693,935763658,588307811,307586772,567625588,125803299,415559316,521261625,742687059,793214170,799825657,38489820,326719544,618977724,553580865,520432021,923282247,210836173,121709091,453459994,878845519,373247016,622359215,426836043,913609355,60241911,840779549,643891938,208097563,809136981,118981,348065550,171913383,140689507,570986579,522302720,984920180,690027819,499226416,8040490,952123274,782283242,858900630,530126601,154648450,246840301,671129922,987388184,553114291,828229076,269871814,381443811,839929528,195450527,469712719,404422948,417132170,591750793,815113068,459856660,213992004,287775658,487682466,212930577,382101913,463014193,522612501,999561513,178244646,707655745,279679447,488344153,849901786,73955229,844237836,314425651,311108634,275955813,706298551,977023048,707748306,840158581,179824276,722480420,66782803,62896215,711396884,330408307,951001756,4763314,721696637,490510096,466243253,330802019,974557355,358265354,986370069,597004861,643050448,811073383,995640531,465083376,113678049,592453849,785986642,636328003,241472863,989352839,337371354,851797007,642464972,771008606,164704241,252512846,203790268,13413542,663959488,146363206,212143534,619898939,183460976,956567217,516634185,837492399,161935364,190081723,864643750,497764706,957355038,790297107,241390788,19126423,74768984,519206343,746839197,891295871,21865890,57238585,424805052,861872421,694981476,819122530,84776696,430880242,585237708,730643743,613731393,796503798,151109911,191633637,640542210,772567830,645336041,205910924,211867675,421223137,278046853,519346455,630345370,462715503,544321891,292368743,837574945,366558573,875571476,603429556,680528699,424250171,864416089,73196594,213595749,192415235,950283357,480321446,504146764,159193881,698933722,986999276,391230988,798261067,883735562,37014257,436782168,173383836,771626291,684349072,279623292,845562552,808686610,833737740,438104156,399066676,454534132,166989547,106915687,24359636,455739644,905147799,932571255,316443334,920913043,722749532,201453365,125971486,474204266,415402611,413858748,305647560,471775551,854675415,900349197,238429188,687251249,168262490,217815778,284297961,788830596,364116302,358209260,48222456,387810637,984022742,159583909,813828412,44660391,19915626,728108302,560650644,215606700,952025678,186826238,943461324,655168284,999004122,439912067,38555375,606532599,581108458,431151350,484188713,964513546,255958055,230379263,449653403,888893423,126878427,626289429,673836979,482235453,758921650,284489043,889230675,397472074,115615232,769736602,814335590,72351755,710439821,43481273,289582772,926824334,324899448,109476165,493354119,843339873,902933404,894735005,368260845,329742111,123680952,76665147,219093823,582836453,272296441,900202553,314807979,752532627,19181861,89906791,612942993,701364947,440233531,610419523,602709585,241859197,598141660,614436106,357903896,221093584,856007433,400530139,588697642,838586006,469031068,42519731,456222723,246298972,993088236,291772262,303917359,817464147,106784877,114773685,74376417,568914797,589078427,618257594,937062551,887763777,340904234,511902361,658050325,994514349,773033283,432812865,304694847,655033928,167211240,716269222,149759343,600693865,556550136,237924866,234788375,111074273,789296565,241051497,486437130,914088509,273500908,477534922,113866063,827720757,371549449,725173433,583660777,387645226,736187954,708352974,570171720,297445964,520492768,413627806,340545276,908409793,387974833,763219578,450285754,733962958,480205371,325944284,672716341,753990836,494725284,940765659,120993275,570973679,795839824,427051218,397528907,374305009,127712543,192752675,244328472,526454339,166766893,396456096,432528101,99131209,436636045,656557464,826009332,767061206,187791961,105030392,350782493,840571067,64623230,848672612,911297564,527834972,526690394,62318650,909920059,733156582,132340155,313975799,874689816,866459099,101977390,789581682,61962843,472704980,389810683,570788485,922689157,707622998,874124775,889234287,886791750,408201618,368396349,465991372,788646237,296026031,678695379,211343101,265686222,65253265,530605089,759612892,333858940,509870611,421086899,597856035,963172161,165815873,245813809,734452944,626674878,37911814,264536674,320213430,988609484,270262961,368038146,990265271,25294514,517564089,454474230,847297484,700960427,486330456,468519121,468898064,802748070,799646716,422032729,667100975,506398565,297578885,915263474,915241289,695831539,788497177,227078612,77020672,272842877,721674971,733543899,815550011,95960565,838895301,808487100,315561484,648416768,660836480,166942942,14475801,919942690,871056305,427431155,660568247,945672767,984561788,561009315,174011136,431924736,714162788,620409526,508022586,33779708,422515993,547488296,148991020,223436958,45936963,552387014,899960414,665442573,703262533,25792038,270542432,453940163,240711304,249675716,842821363,724685354,241773699,271502459,126344279,907242308,416375272,881006446,786113519,343046644,405172610,649830102,362577756,892428340,297073158,197492195,228926781,689720840,952171689,204535551,867843429,863424208,529787052,547097560,228921501,667947897,882524066,930249837,608696677,787199867,747918392,800424854,341668205,620900878,660227032,830619216,661127835,262105055,580144675,87714970,988858283,673805377,943009090,300341073,122511472,271941907,111130114,370159833,206429478,792421884,115978749,689391644,682128139,136696783,656662986,291130047,844484300,921678992,386526164,47705922,513206988,438171892,654385363,491015536,326138347,119302130,59527238,154670399,30274918,242673867,31999692,74430337,421652345,367045275,457349809,491802928,486044677,186741360,401285404,529282044,925469863,557727101,294489338,757482630,985443893,142880767,387307062,127890608,646056718,492114807,175327997,892590488,569014476,542901305,228655711,144976707]
 
 */


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-number-of-balanced-permutations/
///REVITSIT!
class Leet3343 {
    func countBalancedPermutations(_ num: String) -> Int {
        let mod = 1_000_000_007, n = num.count
        var tot = 0, cnt = [Int](repeating: 0, count: 10)
        for c in num {
            guard let d = Int(String(c)) else { continue }
            cnt[d] += 1
            tot += d
        }
        guard tot % 2 == 0 else { return 0 }

        let target = tot / 2, maxOdd = (n + 1) / 2
        var comb = [[Int]](repeating: [Int](repeating: 1, count: maxOdd + 1), count: maxOdd + 1)
        for i in 1...maxOdd {
            comb[i][0] = 1
            comb[i][i] = 1
            for j in 1..<i {
                comb[i][j] = (comb[i - 1][j] + comb[i - 1][j - 1]) % mod
            }
        }

        var f = [[Int]](repeating: [Int](repeating: 0, count: maxOdd + 1), count: target + 1)
        f[0][0] = 1
        var pSum = 0, totSum = 0
        for i in 0...9 {
            pSum += cnt[i]
            totSum += i * cnt[i]
            for oddCnt in stride(from: min(pSum, maxOdd), through: max(0, pSum - (n - maxOdd)), by: -1) {
                let eventCnt = pSum - oddCnt
                for curr in stride(from: min(totSum, target), through: max(0, totSum - target), by: -1) {
                    var res = 0, j = max(0, cnt[i] - eventCnt)
                    while j <= min(cnt[i], oddCnt) && i * j <= curr {
                        if curr - i * j >= 0 && oddCnt - j >= 0 {
                            let ways = (comb[oddCnt][j] * comb[eventCnt][cnt[i] - j]) % mod
                            res = (res + ((ways * f[curr - i * j][oddCnt - j]) % mod)) % mod
                        }
                        j += 1
                    }
                    f[curr][oddCnt] = res % mod
                }
            }
        }
        return f[target][maxOdd]
    }
    static func test() {
        let sut = Leet3343()
        assert(sut.countBalancedPermutations("123") == 2)
    }
}
//Leet3343.test()




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/equal-row-and-column-pairs/
class Leet2352 {
    func equalPairs(_ grid: [[Int]]) -> Int {
        let n = grid.count
        var rows = [String: Int](), cols = [String: Int]()
        for i in 0..<n {
            let r = grid[i], kr = r.map(String.init).joined(separator: ",")
            rows[kr, default: 0] += 1
            let kc = (0..<n).map { "\(grid[$0][i])" }.joined(separator: ",")
            cols[kc, default: 0] += 1
        }
        return rows.reduce(into: 0) { res, r in
            guard let cCount = cols[r.key] else { return }
            res += r.value * cCount
        }
    }
    static func test() {
        let sut = Leet2352()
        assert(sut.equalPairs([[11,1],[1,11]]) == 2)
        assert(sut.equalPairs([[3,1,2,2],[1,4,4,5],[2,4,2,2],[2,4,2,2]]) == 3)
    }
}
//Leet2352.test()


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/jewels-and-stones/
class Leet0771 {
    func numJewelsInStones(_ jewels: String, _ stones: String) -> Int {
        let j = Set(jewels)
        return Array(stones).reduce(into: 0) { cnt, s in cnt += j.contains(String(s)) ? 1 : 0 }
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/delete-greatest-value-in-each-row/
class Leet2500 {
    private func deleteGreatestValue(_ grid: [Heap<Int>]) -> Int {
        var result = 0, grid = grid
        for i in 0..<grid.count {
            guard let m = grid[i].popMax() else { continue }
            result = max(result, m)
        }
        guard grid[0].count > 0 else { return result }
        return deleteGreatestValue(grid) + result
    }
    func deleteGreatestValue(_ grid: [[Int]]) -> Int {
        deleteGreatestValue(grid.map { Heap($0) })
    }
    static func test() {
        let sut = Leet2500()
        assert(sut.deleteGreatestValue([[1,2,4],[3,3,1]]) == 8)
    }
}
//Leet2500.test()



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sum-in-a-matrix/
class Leet2679 {
    private func matrixSum(_ grid: [Heap<Int>]) -> Int {
        var result = 0, grid = grid
        for i in 0..<grid.count {
            guard let m = grid[i].popMax() else { continue }
            result = max(result, m)
        }
        guard grid[0].count > 0 else { return result }
        return matrixSum(grid) + result
    }
    func matrixSum(_ nums: [[Int]]) -> Int {
        matrixSum(nums.map(Heap.init))
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/set-mismatch/
class Leet0645 {
    func findErrorNums(_ nums: [Int]) -> [Int] {
        let n = nums.count, sum = (n * (n + 1)) / 2, actualSum = nums.reduce(0, +)
        let dup = nums.reduce(into: [Int: Int]()) { f, n in f[n, default: 0] += 1 }.filter { $1 > 1 }.keys.first!
        return [dup, sum - (actualSum - dup)]
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-number-of-occurrences-of-a-substring/
class Leet1297 {
    func maxFreq(_ s: String, _ maxLetters: Int, _ minSize: Int, _ maxSize: Int) -> Int {
        let s = Array(s)
        var freq = [Character: Int](), l = 0, strFreqs = [String: Int]()
        func shrink(_ f: inout [Character: Int], _ l: inout Int, _ size: inout Int) {
            f[s[l]]! -= 1
            if f[s[l]] == 0 { f[s[l]] = nil }
            l += 1
            size -= 1
        }
        for r in 0..<s.count {
            let c = s[r]
            freq[c, default: 0] += 1
            var size = r - l + 1
            while freq.keys.count > maxLetters || size > maxSize {
                shrink(&freq, &l, &size)
            }
            guard minSize...maxSize ~= size, freq.keys.count <= maxLetters else { continue }
            strFreqs[String(s[l...r]), default: 0] += 1
                        
            var freq2 = freq, size2 = size, l2 = l
            while size2 > minSize {
                shrink(&freq2, &l2, &size2)
                guard minSize...maxSize ~= size2, freq2.keys.count <= maxLetters else { continue }
                strFreqs[String(s[l2...r]), default: 0] += 1
            }
        }
        return strFreqs.values.max() ?? 0
    }
    static func test() {
        let sut = Leet1297()
        assert(sut.maxFreq("aaaaacbc", 2, 4, 6) == 2)
        assert(sut.maxFreq("aabcabcab", 2, 2, 3) == 3)
        assert(sut.maxFreq("aababcaab", 2, 3, 4) == 2)
        assert(sut.maxFreq("aaaa", 1, 3, 3) == 2)
    }
}
//Leet1297.test()


/*
 "aababcaab"
 2
 3
 4
 "aaaa"
 1
 3
 3
 "ffcbcecaaeaafcb"
 1
 8
 10
 "bccaaabac"
 2
 3
 3
 "bccaaabac"
 2
 2
 2
 "abcdef"
 2
 2
 2
 "aabcabcab"
 2
 2
 3
 "aaaaacbc"
 2
 4
 6
 "ffcbcecaaeaafcb"
 1
 8
 10
 
 */

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-words-containing-character/
class Leet2942 {
    func findWordsContaining(_ words: [String], _ x: Character) -> [Int] {
        words.indices.filter { words[$0].contains(x) }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-target-indices-after-sorting-array/
class Leet2089 {
    func targetIndices(_ nums: [Int], _ target: Int) -> [Int] {
        let nums = nums.sorted { $0 < $1 }
        return nums.indices.filter { nums[$0] == target }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-equal-sum-of-two-arrays-after-replacing-zeros
class Leet2918 {
    
    typealias Counts = (zeroes: Int, total: Int)
    
    private func counts(_ nums: [Int]) -> Counts {
        nums.reduce(into: Counts(zeroes: 0, total: 0)) { r, n in
            r.zeroes += (n == 0 ? 1 : 0)
            r.total  += (n == 0 ? 1 : n)
        }
    }
        
    func minSum(_ nums1: [Int], _ nums2: [Int]) -> Int {
        let c1 = counts(nums1), c2 = counts(nums2)
        guard c1.total != c2.total else { return c1.total }
        let hi = c1.total > c2.total ? c1 : c2, lo = c1.total < c2.total ? c1 : c2
        guard lo.zeroes > 0 else { return -1 }
        return hi.total
    }
}


/*
 
 [3,2,0,1,0]
 [6,5,0]
 [2,0,2,0]
 [1,4]
 [7,3,9,1]
 [11,9,0]
 [7,3,9,1]
 [11,9]
 [7,3,9,1,0]
 [11,9,0]
 [3,2,0,1,0]
 [6,1]
 [0,17,20,17,5,0,14,19,7,8,16,18,6]
 [21,1,27,19,2,2,24,21,16,1,13,27,8,5,3,11,13,7,29,7]
 [0,7,28,17,18]
 [1,2,6,26,1,0,27,3,0,30]
 
 [20,0,18,11,0,0,0,0,0,0,17,28,0,11,10,0,0,15,29]
 [16,9,25,16,1,9,20,28,8,0,1,0,1,27]
 [0,16,28,12,10,15,25,24,6,0,0]
 [20,15,19,5,6,29,25,8,12]
 [0,17,20,17,5,0,14,19,7,8,16,18,6]
 [21,1,27,19,2,2,24,21,16,1,13,27,8,5,3,11,13,7,29,7]
 [8,13,15,18,0,18,0,0,5,20,12,27,3,14,22,0]
 [29,1,6,0,10,24,27,17,14,13,2,19,2,11]
 [1000000,0,0,1000000]
 [0]
 [16,9,25,16,1,9,20,28,8,0,1,0,1,27]
 [20,0,18,11,0,0,0,0,0,0,17,28,0,11,10,0,0,15,29]
 [9,5]
 [15,12,5,21,4,26,27,9,6,29,0,18,16,0,0,0,20]
 [3,2,0,1,0]
 [6,5,0,9,8]
 


 */


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/rank-transform-of-an-array/
class Leet1331 {
    func arrayRankTransform(_ arr: [Int]) -> [Int] {
        guard !arr.isEmpty else { return [] }
        let enumerates = arr.enumerated(), sorteds = enumerates.sorted { $0.element < $1.element }
        var r = 1, ranks: [(r: Int, v: Int)] = [(1, sorteds[0].element)]
        for i in 1..<sorteds.count {
            let c = sorteds[i], p = sorteds[i-1]
            if c.element != p.element {
                r += 1
            }
            ranks.append((r, c.element))
        }
        let ranksMap = ranks.reduce(into: [Int: Int]()) { res, e in res[e.v] = e.r }
        return arr.compactMap { ranksMap[$0] }
    }
    
    static func test() {
        let sut = Leet1331()
        assert(sut.arrayRankTransform([100]) == [1])
        assert(sut.arrayRankTransform([37,12,28,9,100,56,80,5,12]) == [5,3,4,2,8,6,7,1,3])
    }
}
//Leet1331.test()

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/perform-string-shifts/
class Leet1427 {
    func stringShift(_ s: String, _ shift: [[Int]]) -> String {
        let netShift = shift.reduce(into: 0) { res, c in res += c[0] == 0 ? -c[1] : c[1] }
        let n = s.count, s = Array(s), finalShift = ((netShift % n) + n ) % n
        return String(s[(n - finalShift)...] + s[..<(n - finalShift)])
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/three-consecutive-odds/
class Leeet1550 {
    func threeConsecutiveOdds(_ arr: [Int]) -> Bool {
        let n = arr.count, k = 3
        guard n > 2 else { return false }
        return arr[(k-1)...].indices.reduce(into: false) { (res, i) in
            res = res || (arr[i] & 1 == 1) && (arr[i-1] & 1 == 1) && (arr[i-2] & 1 == 1)
        }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/rectangle-overlap/
class Leet0836 {
    func isRectangleOverlap(_ rec1: [Int], _ rec2: [Int]) -> Bool {
        (rec1[0]..<rec1[2]).overlaps(rec2[0]..<rec2[2]) && (rec1[1]..<rec1[3]).overlaps(rec2[1]..<rec2[3])
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/destination-city
class Leet1436 {
    func destCity(_ paths: [[String]]) -> String {
        let hash = paths.reduce(into: [String: String]()) { r, p in r[p[0]] = p[1] }
        var curr = paths[0][0]
        while let next = hash[curr] {
            curr = next
        }
        return curr
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/path-crossing/
class Leet1496 {
    struct Point: Hashable { let x: Int, y: Int }
    
    func isPathCrossing(_ path: String) -> Bool {
        let path = Array(path)
        var p: Point = .init(x: 0, y: 0), seen: Set<Point> = [p]
        for c in path {
            switch c {
            case "N": p = .init(x: p.x, y: p.y + 1)
            case "S": p = .init(x: p.x, y: p.y - 1)
            case "E": p = .init(x: p.x + 1, y: p.y)
            case "W": p = .init(x: p.x - 1, y: p.y)
            default: break
            }
            guard !seen.contains(p) else { return true }
            seen.insert(p)
        }
        return false
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-number-of-words-you-can-type/
class Leet1935 {
    func canBeTypedWords(_ text: String, _ brokenLetters: String) -> Int {
        let brokenSet = Set(brokenLetters)
        return text.split(separator: " ").filter { brokenSet.intersection(Set($0)).count > 0 }.count
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sum-of-unique-elements/
class Leet1748 {
    func sumOfUnique(_ nums: [Int]) -> Int {
        nums.reduce(into: [Int: Int]()) { res, n in res[n, default: 0] += 1 }
            .map { (key: $0, value: $1) }
            .filter { (_, v) in v == 1 }
            .reduce(into: 0) { res, e in res += e.key }
    }
}

///---------------------------------------------------------------------------------------
///
class Leet3005 {
    func maxFrequencyElements(_ nums: [Int]) -> Int {
        let freq = nums.reduce(into: [:]) { counts, n in counts[n, default: 0] += 1 }
        let m = freq.values.max() ?? 0
        return freq.values.reduce(0) { (res, n) in res + (n == m ? m : 0) }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-lucky-integer-in-an-array
class Leet1394 {
    func findLucky(_ arr: [Int]) -> Int {
        let countMap = arr.reduce(into: [Int: Int]()) { res, n in res[n, default: 0] += 1 }
        var result = -1
        for (k, v) in countMap where k == v {
            result = max(result, k)
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/number-of-good-pairs
class Leet1512 {
    func numIdenticalPairs(_ nums: [Int]) -> Int {
        var result = 0
        _ = nums.reduce(into: [Int: Int]()) { counts, num in
            result += counts[num, default: 0]
            counts[num, default: 0] += 1
        }
        return result
    }
    static func test() {
        let sut = Leet1512()
        assert(sut.numIdenticalPairs([1,1,1,1]) == 6)
    }
}
//Leet1512.test()


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/permutation-in-string/
class Leet0567 {
    func checkInclusion(_ s1: String, _ s2: String) -> Bool {
        guard s1.count <= s2.count else { return false }
        let s1 = Array(s1), s2 = Array(s2)
        let s1map = [Character: Int](s1.map { ($0, 1) }, uniquingKeysWith: +)
        var s2map = [Character: Int](s2.prefix(s1.count).map { ($0, 1) }, uniquingKeysWith: +)
        for r in s1.count..<s2.count {
            guard s1map != s2map else { return true }
            let lChar = s2[r - s1.count], rChar = s2[r]
            s2map[rChar, default: 0] += 1
            s2map[lChar, default: 0] -= 1
            guard s2map[lChar]! == 0 else { continue }
            s2map[lChar] = nil
        }
        return s2map == s1map
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-all-anagrams-in-a-string/
class Leet0438 {
    func findAnagrams(_ s: String, _ p: String) -> [Int] {
        guard p.count <= s.count else { return [] }
        let p = Array(p), s = Array(s), pCounts = [Character: Int](p.map { ($0, 1) }, uniquingKeysWith: +)
        var win = [Character: Int](s[..<p.count].map { ($0, 1) }, uniquingKeysWith: +), result = [Int]()
        for r in p.count..<s.count {
            if win == pCounts {
                result.append(r - p.count)
            }
            let lChar = s[r - p.count], rChar = s[r]
            win[rChar, default: 0] += 1
            win[lChar, default: 0] -= 1
            if win[lChar] == 0 {
                win[lChar] = nil
            }
        }
        if win == pCounts {
            result.append(s.count - p.count)
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/isomorphic-strings/
class Leet0205 {
    func isIsomorphic(_ s: String, _ t: String) -> Bool {
        Set(s).count == Set(t).count && Set(t).count == Set(zip(s, t).map { "\($0)\($1)" }).count
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/word-pattern/
class Leet0290 {
    func wordPattern(_ pattern: String, _ s: String) -> Bool {
        let patternArray = Array(pattern), s = s.split(separator: " ").map(String.init)
        var wMap = [String: Character](), cMap = [Character: String]()
        guard patternArray.count == s.count else { return false }
        for i in 0..<s.count {
            let c = patternArray[i], w = s[i]
            if let existingChar = wMap[w], existingChar != c {
                return false
            } else {
                wMap[w] = c
            }
            if let existingWord = cMap[c], existingWord != w {
                return false
            } else {
                cMap[c] = w
            }
        }
        return true
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/total-characters-in-string-after-transformations-i
class Leet3335 {
    func lengthAfterTransformations(_ s: String, _ t: Int) -> Int {
        let s = Array(s), aVal = Character("a").asciiValue!, mod = 1_000_000_007
        var cnt = [Int](repeating: 0, count: 26)
        for ch in s {
            cnt[Int(ch.asciiValue! - aVal)] += 1
        }
        for _ in 0..<t {
            var nxt = [Int](repeating: 0, count: 26)
            nxt[0] = cnt[25]
            nxt[1] = (cnt[25] + cnt[0]) % mod
            for i in 2..<26 {
                nxt[i] = cnt[i - 1]
            }
            cnt = nxt
        }
        var ans = 0
        for i in 0..<26 {
            ans = (ans + cnt[i]) % mod
        }
        return ans
    }
    static func test() {
        let sut = Leet3335()
        assert(sut.lengthAfterTransformations("jqktcurgdvlibczdsvnsg", 7517) == 79033769)
    }
}
//Leet3335.test()

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/total-characters-in-string-after-transformations-ii/
class Leet3337 {
    
    private let mod = 1_000_000_007
    private let k = 26
    private typealias Matrix = [[Int]]
    
    private func newMatrix() -> Matrix {
        [[Int]](repeating: [Int](repeating: 0, count: k), count: k)
    }
    
    private func multiply(_ a: Matrix, _ other: Matrix) -> Matrix {
        var res = newMatrix()
        for i in 0..<k {
            for j in 0..<k {
                for k in 0..<k {
                    res[i][j] = (res[i][j] + Int(Int64(a[i][k]) * Int64(other[k][j]) % Int64(mod))) % mod
                }
            }
        }
        return res
    }
    
    private func identityMatrix() -> Matrix {
        var res = newMatrix()
        for i in 0..<k {
            res[i][i] = 1
        }
        return res
    }
    
    private func quickMultiply(_ x: Matrix, _ y: Int) -> Matrix {
        var ans = identityMatrix(), curr = x, y = y
        while y > 0 {
            if y & 1 == 1 {
                ans = multiply(ans, curr)
            }
            curr = multiply(curr, curr)
            y >>= 1
        }
        return ans
    }
    
    func lengthAfterTransformations(_ s: String, _ t: Int, _ nums: [Int]) -> Int {
        var trans = newMatrix()
        for i in 0..<k {
            for j in 1...nums[i] {
                trans[(i + j) % k][i] = 1
            }
        }
        let aAscii = Int(Character("a").asciiValue!)
        var res = quickMultiply(trans, t), f = [Int](repeating: 0, count: k), ans = 0
        for c in s {
            f[Int(c.asciiValue!) - aAscii] += 1
        }
        for i in 0..<k {
            for j in 0..<k {
                ans = (ans + Int(Int64(res[i][j]) * Int64(f[j]) % Int64(mod))) % mod
            }
        }
        return ans
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/custom-sort-string/
class Leet0791 {
    func customSortString(_ order: String, _ s: String) -> String {
        let sCounts = [Character: Int](s.map { ($0, 1) }, uniquingKeysWith: +), orderSet = Set(order)
        var res = [Character]()
        for c in order {
            guard let count = sCounts[c] else { continue }
            res.append(contentsOf: repeatElement(c, count: count))
        }
        for c in s where !orderSet.contains(c) {
            res.append(c)
        }
        return String(res)
    }
}

/*
 "cba"
 "abcd"
 "bcafg"
 "abcd"
 "abcdefghijklmnopqrstuvwxyz"
 "abcde"
 "xyz"
 "xyzab"
 "pqrs"
 "pqrstuvwx"
 "aeiou"
 "hello"
 "cba"
 "cba"
 "abc"
 "abcd"
 
 
 */

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/determine-if-two-strings-are-close/
class Leet1657 {
    func closeStrings(_ word1: String, _ word2: String) -> Bool {
        guard word1.count == word2.count else { return false }
        guard Set(word1) == Set(word2) else { return false }
        let counts1 = [Character: Int](word1.map { ($0, 1) }, uniquingKeysWith: +)
        let counts2 = [Character: Int](word2.map { ($0, 1) }, uniquingKeysWith: +)
        return counts1.values.sorted() == counts2.values.sorted()
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/remove-duplicates-from-sorted-list/
class Leet0083 {
    func deleteDuplicates(_ head: ListNode?) -> ListNode? {
        var prev: ListNode? = head, curr: ListNode? = head?.next
        while curr != nil {
            if prev?.val == curr?.val {
                prev?.next = curr?.next
            } else {
                prev = curr
            }
            curr = curr?.next
        }
        return head
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/
class Leet0082 {
    func deleteDuplicates(_ head: ListNode?) -> ListNode? {
        var sentinel = ListNode(0), prev: ListNode? = sentinel, head: ListNode? = head
        sentinel.next = head
        while head != nil {
            if head?.val == head?.next?.val {
                while head?.val == head?.next?.val {
                    head = head?.next
                }
                prev?.next = head?.next
            } else {
                prev = prev?.next
            }
            head = head?.next
        }
        return sentinel.next
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/reverse-linked-list-ii/
/// REVISIT!
class Leet0092 {
    func reverseBetween(_ head: ListNode?, _ left: Int, _ right: Int) -> ListNode? {
        guard head != nil else { return nil }
        var cur = head, head = head, prev: ListNode?, m = left, n = right
        while m > 1 {
            prev = cur
            cur = cur?.next
            m -= 1
            n -= 1
        }
        var con = prev, tail = cur, third: ListNode?
        while n > 0 {
            third = cur?.next
            cur?.next = prev
            prev = cur
            cur = third
            n -= 1
        }
        if con != nil {
            con?.next = prev
        } else {
            head = prev
        }
        tail?.next = cur
        return head
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/remove-linked-list-elements/
class Leet0203 {
    func removeElements(_ head: ListNode?, _ val: Int) -> ListNode? {
        let sentinel = ListNode(0)
        sentinel.next = head
        var prev: ListNode? = sentinel, curr = head
        while curr != nil {
            if curr?.val == val {
                prev?.next = curr?.next
            } else {
                prev = curr
            }
            curr = curr?.next
        }
        return sentinel.next
    }
    static func test() {
        let sut = Leet0203()
        assert(sut.removeElements([1,2,6,3,4,5,6].makeListNode(), 6)?.toArray() == [1,2,3,4,5])
        assert(sut.removeElements([1,2,2,1].makeListNode(), 2)?.toArray() == [1,1])
        assert(sut.removeElements([7,7,7,7].makeListNode(), 7) == nil)
    }
}
//Leet0203.test()

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/convert-binary-number-in-a-linked-list-to-integer/
class Leet1290 {
    private func power(_ base: Int, _ exponent: Int) -> Int {
        var result = 1, base = base, exponent = exponent
        while exponent > 0 {
            if exponent % 2 == 1 {
                result = (result * base)
            }
            base = (base * base)
            exponent /= 2
        }
        return result
    }
    func getDecimalValue(_ head: ListNode?) -> Int {
        var reversed: ListNode?, curr = head
        while curr != nil {
            let temp = curr?.next
            curr?.next = reversed
            reversed = curr
            curr = temp
        }
        curr = reversed
        var result = 0, exponent = 0
        while let c = curr {
            result += c.val * power(2, exponent)
            exponent += 1
            curr = c.next
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/swapping-nodes-in-a-linked-list/
class Leet1721 {
    func swapNodes(_ head: ListNode?, _ k: Int) -> ListNode? {
        var i = 1, kFront: ListNode?, prev: ListNode?, curr = head
        while curr != nil {
            if i == k {
                kFront = curr
            }
            i += 1
            let temp = curr?.next
            curr?.next = prev
            prev = curr
            curr = temp
        }
        i = 1; curr = prev; prev = nil
        while curr != nil {
            if i == k, let kFrontValue = kFront?.val {
                kFront?.val = curr?.val ?? 0
                curr?.val = kFrontValue
            }
            i += 1
            let temp = curr?.next
            curr?.next = prev
            prev = curr
            curr = temp
        }
        return head
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/reverse-nodes-in-even-length-groups/
class Leet2074 {
    func reverseEvenLengthGroups(_ head: ListNode?) -> ListNode? {
        var i = 1, groupSize = 1, prev: ListNode?, curr = head
        while curr != nil {
            if i == groupSize || curr?.next == nil {
                if i.isMultiple(of: 2) {
                    let nextGroupHead = curr?.next
                    // reverse from c to nextGroupHead (exlusive)
                    var p = prev, c = prev?.next, newTail = c
                    while c !== nextGroupHead {
                        let temp = c?.next
                        c?.next = p
                        p = c
                        c = temp
                    }
                    prev?.next = p
                    newTail?.next = nextGroupHead
                    curr = newTail
                }
                prev = curr
                groupSize += 1
                i = 1
            } else {
                i += 1
            }
            curr = curr?.next
        }
        return head
    }
    static func test() {
        let sut = Leet2074()
        assert(sut.reverseEvenLengthGroups([5,2,6,3,9,1,7,3,8,4].makeListNode())?.toArray() == [5,6,2,3,9,1,4,8,3,7])
        assert(sut.reverseEvenLengthGroups([1,1,0,6].makeListNode())?.toArray() == [1,0,1,6])
        assert(sut.reverseEvenLengthGroups([1,1,0,6,5].makeListNode())?.toArray() == [1,0,1,5,6])
    }
}
//Leet2074.test()


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/reverse-nodes-in-k-group/
class Leet0025 {
    func reverseKGroup(_ head: ListNode?, _ k: Int) -> ListNode? {
        var i = 1, result: ListNode?, prev: ListNode?, curr = head
        while curr != nil {
            if i == k {
                // reverse from c to nextKHead (exclusive)
                let nextKHead = curr?.next
                var p = prev, c = prev?.next
                
                if result == nil {
                    result = curr
                    c = head
                }
                let newTail = c
                while c !== nextKHead {
                    let temp = c?.next
                    c?.next = p
                    p = c
                    c = temp
                }
                prev?.next = p
                newTail?.next = nextKHead
                curr = newTail
                prev = curr
                i = 1
            } else {
                i += 1
            }
            curr = curr?.next
        }
        return result
    }
    static func test() {
        let sut = Leet0025()
        assert(sut.reverseKGroup([1,2,3,4,5].makeListNode(), 2)?.toArray() == [2,1,4,3,5])
        assert(sut.reverseKGroup([1,2,3,4,5].makeListNode(), 3)?.toArray() == [3,2,1,4,5])
    }
    
}
//Leet0025.test()


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-twin-sum-of-a-linked-list/
class Leet2130 {
    func pairSum(_ head: ListNode?) -> Int {
        var slow = head, fast = head, maxPairSum = 0
        // traverse slow and fast pointers
        while fast?.next?.next != nil {
            slow = slow?.next
            fast = fast?.next?.next
        }
        // split the list and reverse the second half
        var curr = slow?.next, prev: ListNode?
        slow?.next = nil
        while curr != nil {
            let temp = curr?.next
            curr?.next = prev
            prev = curr
            curr = temp
        }
        // traverse the two pointers and get the maximum sum pair
        var p1 = head, p2 = prev
        while p1 != nil && p2 != nil {
            maxPairSum = max(maxPairSum, p1!.val + p2!.val)
            p1 = p1?.next
            p2 = p2?.next
        }
        return maxPairSum
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/longest-unequal-adjacent-groups-subsequence-i/
class Leet2900 {
    func getLongestSubsequence(_ words: [String], _ groups: [Int]) -> [String] {
        zip(words, groups)
            .enumerated()
            .map { (i: $0.offset, w: $0.element.0, g: $0.element.1) }
            .compactMap { (i, w, g) in
                guard i != 0 else { return w }
                guard groups[i - 1] != g else { return nil }
                return w
            }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/odd-even-linked-list/
class Leet0328 {
    func oddEvenList(_ head: ListNode?) -> ListNode? {
        var i = 1, curr = head, result = curr, oddTail: ListNode?, evenHead = curr?.next, evenTail: ListNode?
        while curr != nil {
            let temp = curr?.next
            // when odd
            if i % 2 == 1 {
                oddTail = curr
                oddTail?.next = curr?.next?.next
            // when even
            } else {
                evenTail = curr
                evenTail?.next = curr?.next?.next
            }
            curr = temp
            i += 1
        }
        oddTail?.next = evenHead
        return result
    }
    static func test() {
        let sut = Leet0328()
        assert(sut.oddEvenList([1,2,3,4,5].makeListNode())?.toArray() == [1,3,5,2,4])
    }
}
//Leet0328.test()



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/design-linked-list/
///Leet0707
class MyLinkedList {
    private var head: ListNode?
    private var tail: ListNode?
    private var count = 0
    init() {}
    
    func get(_ index: Int) -> Int {
        guard index >= 0 else { return -1 }
        var curr = head
        for _ in 0..<index {
            curr = curr?.next
        }
        return curr?.val ?? -1
    }
    
    func addAtHead(_ val: Int) {
        count += 1
        let newHead = ListNode(val)
        newHead.next = head
        head = newHead
        if tail == nil {
            tail = newHead
        }
    }
    
    func addAtTail(_ val: Int) {
        count += 1
        let newTail = ListNode(val)
        if head == nil {
            head = newTail
        } else {
            tail?.next = newTail
        }
        tail = newTail
    }
    
    func addAtIndex(_ index: Int, _ val: Int) {
        guard index > 0 else { return addAtHead(val) }
        guard index != count else { return addAtTail(val) }
        guard index < count else { return }
        var prev = head
        for _ in 1..<index {
            prev = prev?.next
        }
        let newNode = ListNode(val)
        let temp = prev?.next
        prev?.next = newNode
        newNode.next = temp
        count += 1
    }
    
    func deleteAtIndex(_ index: Int) {
        guard 0..<count ~= index else { return }
        guard index > 0 else { return head = head?.next }
        var prev = head
        for i in 1..<count {
            guard i < index else { break }
            prev = prev?.next
        }
        let temp = prev?.next?.next
        prev?.next = temp
        if count - 1 == index {
            tail = prev
        }
        count -= 1
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/
class Leet1074 {
    func removeDuplicates(_ s: String) -> String {
        let s = Array(s)
        var stack = [Character]()
        for c in s {
            if let last = stack.last, last == c {
                stack.removeLast()
            } else {
                stack.append(c)
            }
        }
        return String(stack)
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii/
class Leet1209 {
    func removeDuplicates(_ s: String, _ k: Int) -> String {
        typealias C = (char: Character, count: Int)
        let s = Array(s)
        var stack = [C]()
        for c in s {
            guard let last = stack.last else {
                stack.append(C(char: c, count: 1))
                continue
            }
            if last.char == c {
                if last.count + 1 == k {
                    stack.removeLast(k - 1)
                } else {
                    stack.append(C(char: c, count: last.count + 1))
                }
            } else {
                stack.append(C(char: c, count: 1))
            }
        }
        return stack.map { String($0.char) }.joined()
    }
    static func test() {
        let sut = Leet1209()
        assert(sut.removeDuplicates("deeedbbcccbdaa", 3) == "aa")
    }
}
//Leet1209.test()

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimize-string-length/
class Leet2716 {
    func minimizedStringLength(_ s: String) -> Int {
        Set(s).count
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/removing-stars-from-a-string/
class Leet2390 {
    func removeStars(_ s: String) -> String {
        let s = Array(s)
        var stack = [Character]()
        for c in s {
            if c == "*" {
                if !stack.isEmpty {
                    stack.removeLast()
                }
            } else {
                stack.append(c)
            }
        }
        return String(stack)
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/simplify-path/
class Leet0071 {
    func simplifyPath(_ path: String) -> String {
        var stack = [String]()
        for item in path.split(separator: "/") where item != "." {
            if item == ".." {
                if !stack.isEmpty {
                    stack.removeLast()
                }
            } else {
                stack.append(String(item))
            }
        }
        return "/" + stack.joined(separator: "/")
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/goal-parser-interpretation/
class Leet1678 {
    func interpret(_ command: String) -> String {
        command
            .replacingOccurrences(of: "(al)", with: "al")
            .replacingOccurrences(of: "()", with: "o")
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/reformat-date/
class Leet1507 {
    func reformatDate(_ date: String) -> String {
        // remove ordinals from the date string
        let dateWithoutOrdinal = date
            .replacing("st", with: "")
            .replacing("nd", with: "")
            .replacing("rd", with: "")
            .replacing("th", with: "")
        let f = DateFormatter()
        f.dateFormat = "dd MMM yyyy"
        guard let d = f.date(from: dateWithoutOrdinal) else { return "" }
        f.dateFormat = "yyyy-MM-dd"
        return f.string(from: d)
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/shuffle-string/
class Leet1528 {
    func restoreString(_ s: String, _ indices: [Int]) -> String {
        String(zip(Array(s), indices).sorted { $0.1 < $1.1 }.map(\.0))
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/longest-unequal-adjacent-groups-subsequence-ii/
class Leet2901 {
    func getWordsInLongestSubsequence(_ words: [String], _ groups: [Int]) -> [String] {
        let n = words.count
        var dp = [Int](repeating: 1, count: n), prev = Array(repeating: -1, count: n), maxIndex = 0
        for i in 1..<n {
            for j in 0..<i {
                if isValid(words[i], words[j]) && dp[j] + 1 > dp[i] && groups[i] != groups[j] {
                    dp[i] = dp[j] + 1
                    prev[i] = j
                }
            }
            if dp[i] > dp[maxIndex] {
                maxIndex = i
            }
        }
        var result = [String](), i = maxIndex
        while i >= 0 {
            result.append(words[i])
            i = prev[i]
        }
        return result.reversed()
    }
        
    private func isValid(_ s1: String, _ s2: String) -> Bool {
        guard s1.count == s2.count else { return false }
        return zip(s1, s2).filter { $0 != $1 }.count == 1
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/make-the-string-great/
class Leet1544 {
            
    func makeGood(_ s: String) -> String {
        let s = Array(s)
        var stack = [s[0]]
        for c in s[1...] {
            if let last = stack.last, isBad(last, c) {
                stack.removeLast()
                continue
            }
            stack.append(c)
        }
        return String(stack)
    }
    
    private func isBad(_ c1: Character, _ c2: Character) -> Bool {
        c1.isLowercase && c2.isUppercase && c1 == c2.lowercased().first! || c1.isUppercase && c2.isLowercase && c1 == c2.uppercased().first!
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/remove-letter-to-equalize-frequency/
class Leet2423 {
    func equalFrequency(_ word: String) -> Bool {
        let word = Array(word)
        var result = false
        for i in 0..<word.count {
            let lessIthWord = Array(word[0..<i] + word[i+1..<word.count])
            let freq = lessIthWord.reduce(into: [:]) { $0[$1, default: 0] += 1 }
            result = result || Set(freq.values).count == 1
        }
        return result
    }
}



/*
 
 
 "ceeeec"
 "abcabc"
 "abcdefg"
 "cccd"
 "ab"
 "aaaabbbbccc"
 "abbcc"
 "ddaccb"
 
 "abcc"

 "aazz"
 "bac"
 "bacc"
 "abbcc"
 "dddaccc"
 "aca"
 "cbccca"
 "zzzzzzzzzzzzzzzzzzzzz"
 
 
 */


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/check-if-array-is-sorted-and-rotated/
class Leet1752 {
    func check(_ nums: [Int]) -> Bool {
        let min = nums.min()!
        let indexesOfMin = nums.indices.filter { nums[$0] == min }
        let sorted = nums.sorted()
        var result = false
        for indexOfMin in indexesOfMin {
            let leftHalf = nums[0..<indexOfMin]
            let rightHalf = nums[indexOfMin...]
            let combined = Array(rightHalf) + Array(leftHalf)
            result = result || sorted == combined
        }
        return result
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/score-of-parentheses/
class Leet0856 {
    func scoreOfParentheses(_ s: String) -> Int {
        var stack = [0]
        for c in s {
            if c == "(" {
                stack.append(0)
            } else {
                let v = stack.removeLast(), w = stack.removeLast()
                stack.append(w + max(2 * v, 1))
            }
        }
        return stack[0]
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sort-colors/
class Leet0075 {
    func sortColors(_ nums: inout [Int]) {
        var l = 0, r = nums.count - 1, p = 0
        while p <= r {
            if nums[p] == 0 {
                nums.swapAt(l, p)
                l += 1
                p += 1
            } else if nums[p] == 2 {
                nums.swapAt(p, r)
                r -= 1
            } else {
                p += 1
            }
        }
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sort-list/
class Leet0148 {
    func sortList(_ head: ListNode?) -> ListNode? {
        guard head != nil && head?.next != nil else { return head }
        guard let middle = mid(head) else { return head }
        let left = sortList(head)
        let right = sortList(middle)
        let result = mergeTwoLists(left, right)
        return result
        
    }
    
    private func mergeTwoLists(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
        var prehead = ListNode(0), prev: ListNode? = prehead, p1 = l1, p2 = l2
        while p1 != nil && p2 != nil {
            if p1!.val < p2!.val {
                prev?.next = p1
                p1 = p1?.next
            } else {
                prev?.next = p2
                p2 = p2?.next
            }
            prev = prev?.next!
        }
        prev?.next = p1 != nil ? p1 : p2
        return prehead.next
    }

    private func mid(_ head: ListNode?) -> ListNode? {
        var slow = head, fast = head, prev: ListNode?
        while fast != nil && fast?.next != nil {
            prev = slow
            slow = slow?.next
            fast = fast?.next?.next
        }
        prev?.next = nil
        return slow
    }
    
    static func test() {
        let sut = Leet0148()
        assert(sut.sortList([10,1,60,30,5].makeListNode())?.toArray() == [1,5,10,30,60])
    }
}
//Leet0148.test()



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/merge-k-sorted-lists/
class Leet0023 {
    func mergeKLists(_ lists: [ListNode?]) -> ListNode? {
        var result: ListNode?
        for head in lists {
            result = mergeTwoLists(result, head)
        }
        return result
    }
    private func mergeTwoLists(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
        var prehead = ListNode(0), prev: ListNode? = prehead, p1 = l1, p2 = l2
        while p1 != nil && p2 != nil {
            if p1!.val < p2!.val {
                prev?.next = p1
                p1 = p1?.next
            } else {
                prev?.next = p2
                p2 = p2?.next
            }
            prev = prev?.next!
        }
        prev?.next = p1 != nil ? p1 : p2
        return prehead.next
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/check-if-all-as-appears-before-all-bs/
class Leet2124 {
    func checkString(_ s: String) -> Bool {
        !s.contains("ba")
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/check-if-numbers-are-ascending-in-a-sentence/
class Leet2042 {
    func areNumbersAscending(_ s: String) -> Bool {
        let nums = s.split(separator: " ").compactMap { Int(String($0)) }
        return nums.indices.dropFirst(1).allSatisfy { nums[$0 - 1] < nums[$0] }
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sorting-the-sentence/
class Leet1859 {
    func sortSentence(_ s: String) -> String {
        var m = 0, result = ""
        let d = s
            .split(separator: " ")
            .reduce(into: [Int:String]()) { r, w in
                let n = Int(String(w.last!))!
                r[n] = String(w.dropLast())
                m = max(m, n)
            }
        for i in 1...m {
            result += d[i]! + " "
        }
        result.removeLast()
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/painting-a-grid-with-three-different-colors/
class Leet1931 {
    
    func colorTheGrid(_ m: Int, _ n: Int) -> Int {
        let mod = 1_000_000_007, maskEnd = Int(pow(3.0, Double(m)))
        var valid = [Int: [Int]]()
        
        for mask in 0..<maskEnd {
            var color = [Int](), mm = mask, check = true
            for _ in 0..<m {
                color.append(mm % 3)
                mm /= 3
            }
            for i in 0..<m-1 {
                if color[i] == color[i+1] {
                    check = false
                    break
                }
            }
            if check {
                valid[mask] = color
            }
        }
        
        var adjacent = [Int: [Int]]()
        for mask1 in valid.keys {
            for mask2 in valid.keys {
                var check = true
                for i in 0..<m {
                    if valid[mask1]?[i] == valid[mask2]?[i] {
                        check = false
                        break
                    }
                }
                if check {
                    adjacent[mask1, default: []].append(mask2)
                }
                
            }
        }
        var f = [Int: Int]()
        for mask in valid.keys {
            f[mask] = 1
        }
        for _ in 1..<n {
            var g = [Int: Int]()
            for mask2 in valid.keys {
                for mask1 in adjacent[mask2] ?? [] {
                    g[mask2, default: 0] = ((g[mask2] ?? 0) + (f[mask1] ?? 0)) % mod
                }
            }
            f = g
        }
        var ans = 0
        for num in f.values {
            ans = (ans + num) % mod
        }
        return ans
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sort-linked-list-already-sorted-using-absolute-values/
class Leet2046 {
    
    func sortLinkedList(_ head: ListNode?) -> ListNode? {
        var head = head, prev: ListNode?, curr = head
        while curr != nil {
            if let currVal = curr?.val, currVal < 0, head !== curr {
                prev?.next = curr?.next
                curr?.next = head
                head = curr
                curr = prev?.next
            } else {
                prev = curr
                curr = curr?.next
            }
        }
        return head
    }
    
    func xxx_sortLinkedList(_ head: ListNode?) -> ListNode? {
        var prev: ListNode?, curr = head, posHead: ListNode?, posTail: ListNode?, negHead: ListNode?, negTail: ListNode?
        while curr != nil {
            guard let val = curr?.val else { break }
            if val >= 0 {
                if let p = prev, p.val < 0 {
                    p.next = curr?.next
                    posTail?.next = curr
                }
                if posHead == nil {
                    posHead = curr
                }
                posTail = curr
            } else {
                if let p = prev, p.val >= 0 {
                    p.next = curr?.next
                    negTail?.next = curr
                }
                if negHead == nil {
                    negHead = curr
                }
                negTail = curr
            }
            prev = curr
            curr = curr?.next
        }
        posTail?.next = nil
        negTail?.next = nil
        // reverse negHead
        prev = nil; curr = negHead; negTail = negHead
        while curr != nil {
            let temp = curr?.next
            curr?.next = prev
            prev = curr
            curr = temp
        }
        negHead = prev
        negTail?.next = posHead // connect!
        return negHead ?? posHead
    }
    static func test() {
        let sut = Leet2046()
        assert(sut.sortLinkedList([0,-6,-9,-10].makeListNode())?.toArray() == [-10,-9,-6,0])
        assert(sut.sortLinkedList([0,1,2].makeListNode())?.toArray() == [0,1,2])
        assert(sut.sortLinkedList([0,2,-5,5,10,-10].makeListNode())?.toArray() == [ -10, -5, 0, 2, 5, 10 ])
    }
}
//Leet2046.test()


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/rotate-list/
class Leet0061 {
    func rotateRight(_ head: ListNode?, _ k: Int) -> ListNode? {
        guard k > 0 else { return head }
        var n = 0, curr = head, prev: ListNode?, head = head
        while curr != nil {
            prev = curr
            curr = curr?.next
            n += 1
        }
        guard n > 1, k%n > 0 else { return head }
        let tail = prev
        curr = head
        for _ in 0..<(n-k%n) {
            prev = curr
            curr = curr?.next
        }
        tail?.next = head
        head = curr
        prev?.next = nil
        return head
    }
    
    static func test() {
        let sut = Leet0061()
        assert(sut.rotateRight([1,2].makeListNode(), 2)?.toArray() == [1,2])
        assert(sut.rotateRight([1,2].makeListNode(), 0)?.toArray() == [1,2])
        assert(sut.rotateRight([1].makeListNode(), 3)?.toArray() == [1])
        assert(sut.rotateRight([1].makeListNode(), 1)?.toArray() == [1])
        assert(sut.rotateRight([1,2,3,4,5].makeListNode(), 2)?.toArray() == [4,5,1,2,3])
    }
}
//Leet0061.test()


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/split-linked-list-in-parts/
class Leet0725 {
    func splitListToParts(_ head: ListNode?, _ k: Int) -> [ListNode?] {
        guard head != nil else { return [ListNode?](repeating: nil, count: k) }
        var n = 0, curr = head, head = head
        while curr != nil {
            curr = curr?.next
            n += 1
        }
        let size = n / k
        var remainder = n % k, result = [ListNode?](repeating: nil, count: k), count = 1, i = 0, currentSize = size + (remainder > 0 ? 1 : 0)
        curr = head; remainder -= 1
        while curr != nil {
            if count == currentSize {
                count = 1
                currentSize = size + (remainder > 0 ? 1 : 0)
                remainder -= 1
                result[i] = head
                i += 1
                head = curr?.next
                curr?.next = nil
                curr = head
            } else {
                curr = curr?.next
                count += 1
            }
        }
        return result
    }
    static func test() {
        let sut = Leet0725()
        assert(sut.splitListToParts([1,2,3].makeListNode(), 5).map { $0?.toArray() ?? [] } == [[1],[2],[3],[],[]] )
        assert(sut.splitListToParts([1,2,3,4,5,6,7,8,9,10].makeListNode(), 3).map { $0?.toArray() } == [[1,2,3,4],[5,6,7],[8,9,10]] )
    }
}
//Leet0725.test()

/*
 
 [1,2,3,4,5,6,7,8,9,10]
 3
 [1,2,3,4,5,6,7,8,9,10,11]
 3
 [1,2,3]
 5
 
 
 */

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/type-of-triangle
class Leet3024 {
    func triangleType(_ nums: [Int]) -> String {
        guard nums.count == 3 else { return "none" }
        let a = nums[0], b = nums[1], c = nums[2]
        let counts = nums.reduce(into: [:]) { $0[$1, default: 0] += 1 }
        if counts.count == 1 {
            return "equilateral"
        } else if (a + b) <= c || (a + c) <= b || (b + c) <= a {
            return "none"
        } else if counts.count == 2 {
            return "isosceles"
        } else {
            return "scalene"
        }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/insertion-sort-list/
class Leet0147 {
    func insertionSortList(_ head: ListNode?) -> ListNode? {
        var result: ListNode? = ListNode(0), curr = head
        while curr != nil {
            var prev = result
            while prev?.next != nil && prev!.next!.val < curr!.val {
                prev = prev?.next
            }
            let next = curr?.next
            curr?.next = prev?.next
            prev?.next = curr
            curr = next
        }
        return result?.next
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/
class Leet1249 {
    func minRemoveToMakeValid(_ s: String) -> String {
        let s = Array(s)
        var indexExcludes = [Int]() // unpaired closing parenthesis indeces
        var indexStack = [Int]() // index stack of open parenthesis that must match
        for i in s.indices {
            let c = s[i]
            if c == "(" {
                indexStack.append(i)
            } else if c == ")" {
                if indexStack.isEmpty {
                    indexExcludes.append(i)
                } else {
                    indexStack.removeLast()
                }
            }
        }
        let set = Set(indexStack + indexExcludes)
        return String(s.enumerated().filter { !set.contains($0.offset) }.map(\.element))
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/check-if-a-parentheses-string-can-be-valid/description/
class Leet2116 {
    func canBeValid(_ s: String, _ locked: String) -> Bool {
        let n = s.count, s = Array(s), locked = Array(locked)
        guard n.isMultiple(of: 2) else { return false }
        var openStack = [Int](), unlockedStack = [Int]()
        for i in 0..<n {
            let l = locked[i], c = s[i]
            if l == "0" {
                unlockedStack.append(i)
            } else if c == "(" {
                openStack.append(i)
            } else if c == ")" {
                if !openStack.isEmpty {
                    openStack.removeLast()
                } else if !unlockedStack.isEmpty {
                    unlockedStack.removeLast()
                } else {
                    return false
                }
            }
        }
        while !openStack.isEmpty && !unlockedStack.isEmpty && openStack.last! < unlockedStack.last! {
            openStack.removeLast()
            unlockedStack.removeLast()
        }
        guard openStack.isEmpty else { return false }
        return true
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/swap-nodes-in-pairs/
class Leet0024 {
    func swapPairs(_ head: ListNode?) -> ListNode? {
        guard let h = head else { return nil }
        guard let t = h.next else { return h }
        let nt = swapPairs(t.next)
        t.next = h
        h.next = nt
        return t
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/partition-list/
class Leet0086 {
   func partition(_ head: ListNode?, _ x: Int) -> ListNode? {
       var curr = head, prev: ListNode?, rightHead: ListNode?, result: ListNode?, leftTail: ListNode?
       while curr != nil {
           let temp = curr?.next
           if let v = curr?.val, v < x {
               if result == nil {
                   result = curr
               } else {
                   leftTail?.next = curr
               }
               prev?.next = temp
               curr?.next = nil
               leftTail = curr
           } else {
               if rightHead == nil {
                   rightHead = curr
               }
               prev = curr
           }
           curr = temp
       }
       leftTail?.next = rightHead
       guard let result = result else { return rightHead }
       return result
   }
}


/*
 
 [1,4,3,2,5,2]
 3
 [2,1]
 2
 [2,1]
 1
 []
 0
 [3,6,5,2,4,8,1,7]
 4
 [3,6,5,4,8,1,7,2,4,5,6,7,2,1,2,3,4,5,77,88,3,3,3,4,4,4]
 4
 [3,1]
 2
 [2,4,1,1,2,3,4,2,2]
 3
 
 
 [1,1,1,1,1]
 0
 [3,6,5,4,1,2]
 4
 [1,1]
 0
 [1,1]
 1
 [3,1]
 2
 [1,4,3,0,2,5,2]
 3
 [4,3,2,5,2]
 3
 
 
 */


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/counting-bits/
class Leet0338 {
    func countBits(_ n: Int) -> [Int] {
        (0...n).map { $0.nonzeroBitCount }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/convert-date-to-binary/
class Leet3280 {
    func convertDateToBinary(_ date: String) -> String {
        date
            .split(separator: "-")
            .map {  String(Int($0)!, radix: 2) }
            .joined(separator: "-")
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-the-k-or-of-an-array/
class Leet2917 {
    func findKOr(_ nums: [Int], _ k: Int) -> Int {
        let pad = String(nums.max()!, radix: 2).count
        let bins = nums.map { n in Array(String(n, radix: 2).padded(to: pad).map { Int(String($0))! }.reversed()) }
        var result = [Bool](repeating: false, count: pad)
        for i in result.indices {
            result[i] = (k <= bins.reduce(into: 0) { r, b in r += b[i] })
        }
        return Int(result.reversed().map { $0 ? "1" : "0" }.joined(), radix: 2)!
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/zero-array-transformation-i/
class Leet3355 {
    func isZeroArray(_ nums: [Int], _ queries: [[Int]]) -> Bool {
        let n1 = nums.count + 1
        var delta = [Int](repeating: 0, count: n1)
        for q in queries {
            let l = q[0], r = q[1]
            delta[l] += 1
            delta[r + 1] -= 1
        }
        var opCounts = [Int](repeating: 0, count: n1), ops = 0
        for i in 0..<n1 {
            ops += delta[i]
            opCounts[i] = ops
        }
        for i in 0..<nums.count {
            if opCounts[i] < nums[i] {
                return false
            }
        }
        return true
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sum-of-values-at-indices-with-k-set-bits/
class Leet2859 {
    func sumIndicesWithKSetBits(_ nums: [Int], _ k: Int) -> Int {
        (0..<nums.count).reduce(into: 0) { r, i in r += (i.nonzeroBitCount == k) ? nums[i] : 0 }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/kth-distinct-string-in-an-array/
class Leet2053 {
    func kthDistinct(_ arr: [String], _ k: Int) -> String {
        let counts = arr.reduce(into: [:]) { counts, element in counts[element, default: 0] += 1 }
        let distincts = arr.filter { counts[$0]! == 1 }
        guard k <= distincts.count else { return "" }
        return distincts[k - 1]
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-common-words-with-one-occurrence/
class Leet2085 {
    func countWords(_ words1: [String], _ words2: [String]) -> Int {
        let s = words1.count < words2.count ? words1 : words2
        let l = words1.count < words2.count ? words2 : words1
        var hash = l.reduce(into: [String: Int]()) { h, w in h[w, default: 0] += 1 }
        hash = hash.filter { $0.value == 1 }
        for w in s {
            hash[w, default: 0] -= 1
        }
        return hash.count(where: { $0.value == 0 })
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/uncommon-words-from-two-sentences/
class Leet0884 {
    func uncommonFromSentences(_ s1: String, _ s2: String) -> [String] {
        (s1 + " " + s2)
            .split(separator: " ")
            .reduce(into: [String: Int]()) { cnt, w in cnt[String(w), default: 0] += 1 }
            .filter { $1 == 1 }
            .map { String($0.key) }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-subarrays-with-equal-sum/
class Leet2395 {
    func findSubarrays(_ nums: [Int]) -> Bool {
        var sums: Set<Int> = []
        for i in 1..<nums.count {
            let sum = nums[i-1] + nums[i]
            if sums.contains(sum) {
                return true
            }
            sums.insert(sum)
        }
        return false
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/existence-of-a-substring-in-a-string-and-its-reverse/
class Leet3083 {
    func isSubstringPresent(_ s: String) -> Bool {
        let s = Array(s), r = Array(s.reversed()), n = s.count
        let seen = (1..<n).reduce(into: Set<String>()) { m, i in m.insert(String(r[i-1...i])) }
        for i in 1..<n where seen.contains(String(s[i-1...i])) {
            return true
        }
        return false
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/binary-number-with-alternating-bits/
class Leet0693 {
    func hasAlternatingBits(_ n: Int) -> Bool {
        (n & (n >> 1)) == 0 && (n | (n >> 2)) == n
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/prime-number-of-set-bits-in-binary-representation/
class Leet0762 {
    func countPrimeSetBits(_ left: Int, _ right: Int) -> Int {
        let primes = Set([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31])
        return (left...right).count { n in primes.contains(n.nonzeroBitCount) }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/median-of-two-sorted-arrays/
class Leet0004 {
    func findMedianSortedArrays(_ nums1: [Int], _ nums2: [Int]) -> Double {
        Array(nums1 + nums2).median()
    }
}

extension Array where Element == Int {
    func median() -> Double {
        let sortedArray = sorted()
        guard count.isMultiple(of: 2) else { return Double(sortedArray[count / 2]) }
        return Double(sortedArray[count / 2] + sortedArray[count / 2 - 1]) / 2.0
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/median-of-a-row-wise-sorted-matrix/
class Leet2387 {
    func matrixMedian(_ grid: [[Int]]) -> Int {
        let nums = grid.flatMap(\.self).sorted(), n = nums.count
        return nums[n/2]
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/letter-combinations-of-a-phone-number/
class Leet0017 {
        
    private var digits = [Character]()
    private let letters: [Character: [Character]] = ["2": ["a", "b", "c"], "3": ["d", "e", "f"], "4": ["g", "h", "i"], "5": ["j", "k", "l"], "6": ["m", "n", "o"], "7": ["p", "q", "r", "s"], "8": ["t", "u", "v"], "9": ["w", "x", "y", "z"]]
    private var result = [String]()
    
    func letterCombinations(_ digits: String) -> [String] {
        guard !digits.isEmpty else { return [] }
        result = []
        self.digits = Array(digits)
        backtrack(0, [])
        return result
    }
    
    private func backtrack(_ index: Int, _ path: [Character]) {
        guard path.count < digits.count else {
            return result.append(String(path))
        }
        var path = path
        for l in letters[digits[index]]! {
            path.append(l)
            backtrack(index + 1, path)
            path.removeLast()
        }
    }
    
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-number-of-pushes-to-type-word-i/
class Leet3014 {
    func minimumPushes(_ word: String) -> Int {
        let n = word.count
        guard n <= 26 else { fatalError("Should not happen") }
        var pushes = 0, remaining = n, costPerLetter = 1
        while remaining > 0 {
            let chunk = min(8, remaining)
            pushes += chunk * costPerLetter
            remaining -= chunk
            costPerLetter += 1
        }
        return pushes
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/average-value-of-even-numbers-that-are-divisible-by-three/
class Leet2455 {
    func averageValue(_ nums: [Int]) -> Int {
        let sixes = nums.filter { $0.isMultiple(of: 6) }
        guard sixes.count > 0 else { return 0 }
        return sixes.reduce(0, +) / sixes.count
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/binary-prefix-divisible-by-5/
class Leet1018 {
    func prefixesDivBy5(_ nums: [Int]) -> [Bool] {
        var outputArr = [Bool](), num = 0
        for i in 0..<nums.count {
            num = (num * 2 + nums[i]) % 5
            outputArr.append(num == 0)
        }
        return outputArr
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/set-matrix-zeroes/
class Leet0073 {
    func setZeroes(_ matrix: inout [[Int]]) {
        let m = matrix.count, n = matrix[0].count
        var isFirstRow0 = false, isFirstCol0 = false
        for row in 0..<m {
            for col in 0..<n where matrix[row][col] == 0 {
                if row == 0 {
                    isFirstRow0 = true
                }
                if col == 0 {
                    isFirstCol0 = true
                }
                matrix[0][col] = 0
                matrix[row][0] = 0
            }
        }
        // set rows
        for col in 1..<n where matrix[0][col] == 0 {
            for row in 1..<m {
                matrix[row][col] = 0
            }
        }
        // set columns
        for row in 1..<m where matrix[row][0] == 0 {
            for col in 1..<n {
                matrix[row][col] = 0
            }
        }
        // set first row or columns
        if isFirstCol0 {
            for row in 0..<m {
                matrix[row][0] = 0
            }
        }
        if isFirstRow0 {
            for col in 0..<n {
                matrix[0][col] = 0
            }
        }
    }
}


/*
 [[1,1,1],[1,0,1],[1,1,1]]
 [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
 [[1],[0]]
 [[0,1,1],[1,1,1],[1,0,0]]
 [[-4,-2147483648,6,-7,0],[-8,6,-8,-6,0],[2147483647,2,-9,-6,-10]]
 [[2147483647],[2],[3]]
 [[1,2,3,4],[5,0,7,8],[0,10,11,12],[13,14,15,0]]

 
 [[1,1,1],[1,0,1],[1,1,1]]
 [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
 [[-1],[2],[3]]
 [[2147483647],[2],[3]]
 [[-2147483648],[2],[3]]
 [[-2147483647,1,1],[1,0,1],[1,1,1]]
 [[1],[0]]
 [[-4,-2147483648,6,-7,0],[-8,6,-8,-6,0],[2147483647,2,-9,-6,-10]]
 
 */


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/zigzag-conversion/
class Leet0006 {
    func convert(_ s: String, _ numRows: Int) -> String {
        guard numRows > 1 else { return s }
        let s = Array(s), n = s.count
        var result = [[Character]](repeating: [], count: numRows), row = 0, down = false
        for c in s {
            result[row].append(c)
            if row == 0 || row == numRows - 1 {
                down.toggle()
            }
            row += down ? 1 : -1
        }
        return result.compactMap { String($0) }.joined()
    }
}

/*
 "PAYPALISHIRING"
 3
 "PAYPALISHIRING"
 4
 "A"
 1
 "ABC"
 2
 "A"
 1
 "AB"
 1
 "ABCD"
 2
 */


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/percentage-of-letter-in-string/
class Leet2278 {
    func percentageLetter(_ s: String, _ letter: Character) -> Int {
        s.count(where: { $0 == letter }) * 100 / s.count
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/container-with-most-water/
class Leet0011 {
    func maxArea(_ height: [Int]) -> Int {
        var result = 0, l = 0, r = height.count - 1
        while l < r {
            if height[l] < height[r] {
                result = max(result, (r - l) * height[l])
                l += 1
            } else {
                result = max(result, (r - l) * height[r])
                r -= 1
            }
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/trapping-rain-water/
class Leet0042 {
    func trap(_ height: [Int]) -> Int {
        var l = 0, r = height.count - 1, result = 0, lmax = 0, rmax = 0
        while l < r {
            if height[l] < height[r] {
                lmax = max(lmax, height[l])
                result += lmax - height[l]
                l += 1
            } else {
                rmax = max(rmax, height[r])
                result += rmax - height[r]
                r -= 1
            }
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/3sum/
class Leet0015 {
    func threeSum(_ nums: [Int]) -> [[Int]] {
        let numIdxMap = nums.enumerated()
            .map { (i: $0.offset, v: $0.element) }
            .reduce(into: [Int: Set<Int>]()) { r, e in r[e.v, default: []].insert(e.i) }
        var result = Set<[Int]>()
        for i in 0..<nums.count {
            for j in i+1..<nums.count {
                let target = -(nums[i] + nums[j])
                guard let s = numIdxMap[target], s.contains(where: { $0 != i && $0 != j }) else { continue }
                result.insert([nums[i], nums[j], target].sorted())
            }
        }
        return result.map { $0 }
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/3sum-closest/
class Leet0016 {
    func threeSumClosest(_ nums: [Int], _ target: Int) -> Int {
        var diff = Int.max, n = nums.count, i = 0
        let nums = nums.sorted()
        while i < n && diff != 0 {
            var lo = i + 1, hi = n - 1
            while lo < hi {
                let sum = nums[i] + nums[lo] + nums[hi]
                if abs(target - sum) < abs(diff) {
                    diff = target - sum
                }
                if sum < target {
                    lo += 1
                } else {
                    hi -= 1
                }
            }
            i += 1
        }
        return target - diff
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/3sum-smaller/description/
class Leet0259 {
    func threeSumSmaller(_ nums: [Int], _ target: Int) -> Int {
        var n = nums.count, result = 0
        let nums = nums.sorted()
        for i in 0..<n {
            var lo = i + 1, hi = n - 1
            while lo < hi {
                let sum = nums[i] + nums[lo] + nums[hi]
                if sum < target {
                    result += hi - lo
                    lo += 1
                } else {
                    hi -= 1
                }
            }
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/4sum/
class Leet0018 {
    func fourSum(_ nums: [Int], _ target: Int) -> [[Int]] {
        kSum(nums.sorted(), target, 0, 4)
    }
    private func kSum(_ nums: [Int], _ target: Int, _ start: Int, _ k: Int) -> [[Int]] {
        var result = [[Int]]()
        let n = nums.count, ave = target / k
        guard start < n else { return result }
        guard !(nums[start] > ave || ave > nums[n - 1]) else { return result }
        guard k > 2 else { return twoSum(nums, target, start) }
        for i in start..<n where i == start || nums[i] != nums[i - 1]{
            for subset in kSum(nums, target - nums[i], i + 1, k - 1) {
                result.append([nums[i]] + subset)
            }
        }
        return result
    }
    private func twoSum ( _ nums: [Int], _ target: Int, _ start: Int) -> [[Int]] {
        let n = nums.count
        var result = [[Int]](), lo = start, hi = n - 1
        while lo < hi {
            let currSum = nums[lo] + nums[hi]
            if currSum < target || (lo > start && nums[lo] == nums[lo - 1]) {
                lo += 1
            } else if currSum > target || (hi < n - 1 && nums[hi] == nums[hi + 1]) {
                hi -= 1
            } else {
                result.append([nums[lo], nums[hi]])
                lo += 1
                hi -= 1
            }
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/zero-array-transformation-iii/
class Leet3362 {
    func maxRemoval(_ nums: [Int], _ queries: [[Int]]) -> Int {
        let queries = queries.sorted { (q1, q2) -> Bool in q1[0] < q2[0] }, n = nums.count
        var heap = Heap<Int>(), deltas = [Int](repeating: 0, count: n + 1), ops = 0, j = 0
        for i in 0..<n {
            ops += deltas[i]
            while j < queries.count, queries[j][0] == i {
                heap.insert(queries[j][1])
                j += 1
            }
            while ops < nums[i], let top = heap.max, top >= i {
                ops += 1
                deltas[heap.removeMax() + 1] -= 1
            }
            if ops < nums[i] {
                return -1
            }
        }
        return heap.count
    }
    static func test() {
        let sut = Leet3362()
        assert(sut.maxRemoval([0,0,1,1,0], [[3,4],[0,2],[2,3]]) == 2)
        assert(sut.maxRemoval([2,0,2], [[0,2],[0,2],[1,1]]) == 1)
    }
}
//Leet3362.test()

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/4sum-ii/
class Leet0454 {
    func fourSumCount(_ nums1: [Int], _ nums2: [Int], _ nums3: [Int], _ nums4: [Int]) -> Int {
        var m = [Int:Int](), result = 0
        for a in nums1 {
            for b in nums2 {
                m[a+b, default: 0] += 1
            }
        }
        for c in nums3 {
            for d in nums4 {
                result += m[-c-d, default: 0]
            }
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/check-if-array-is-good/
class Leet2784 {
    func isGood(_ nums: [Int]) -> Bool {
        let n = nums.count - 1, m = nums.reduce(into: [:]) { c, n in c[n, default: 0] += 1 }
        guard n > 1 else { return nums == [1,1] }
        return m[n] == 2 && m.count == n && (1...n-1).allSatisfy { m[$0] ?? 0 == 1 }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/generate-parentheses/
class Leet0022 {
    func generateParenthesis(_ n: Int) -> [String] {
        var result = [String]()
        backtrack(n, 0, 0, "", &result)
        return result
    }
    private func backtrack(_ n: Int, _ open: Int, _ close: Int, _ current:  String, _ result: inout [String]) {
        guard current.count < n * 2 else { return result.append(String(current)) }
        if open < n {
            backtrack(n, open + 1, close, current + "(", &result)
        }
        if open > close {
            backtrack(n, open, close + 1, current + ")", &result)
        }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/index-pairs-of-a-string/
class Leet1065 {
    func indexPairs(_ text: String, _ words: [String]) -> [[Int]] {
        let text = Array(text)
        var result = [[Int]]()
        for i in 0..<text.count {
            for w in words {
                let l = w.count
                guard i + l <= text.count else { continue }
                if String(text[i..<i+l]) == w {
                    result.append([i, i+l-1])
                }
            }
        }
        return result.sorted { $0[0] < $1[0] || ($0[0] == $1[0] && $0[1] < $1[1]) }
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-the-maximum-sum-of-node-values/
///REVISIT
class Leet3068 {
    func maximumValueSum(_ nums: [Int], _ k: Int, _ edges: [[Int]]) -> Int {
        let n = nums.count
        var netChange = [Int](repeating: 0, count: n), nodeSum = 0
        for i in 0..<n {
            netChange[i] = (nums[i] ^ k) - nums[i]
            nodeSum += nums[i]
        }
        netChange.sort(by: >)
        for i in stride(from: 0, to: n, by: 2) {
            guard i + 1 < n else { break }
            let pairSum = netChange[i] + netChange[i + 1]
            guard pairSum > 0 else { continue }
            nodeSum += pairSum
        }
        return nodeSum
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/number-of-different-integers-in-a-string/
class Leet1805 {
    func numDifferentIntegers(_ word: String) -> Int {
        word
            .split(whereSeparator: { c in c.isLetter })
            .map { num in num[(num.firstIndex(where: { c in c != "0" }) ?? num.endIndex)...] }
            .reduce(into: Set<Substring>()) { r, n in r.insert(n) }
            .count
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/number-of-recent-calls/
class Leet0933 {
    private var queue: Deque<Int> = []
    init() { queue = [] }
    func ping(_ t: Int) -> Int {
        let before3K = t - 3000
        while !queue.isEmpty, queue.first! < before3K {
            queue.removeFirst()
        }
        queue.append(t)
        return queue.count
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/dota2-senate/
class Leet0649 {
    func predictPartyVictory(_ senate: String) -> String {
        let senate = Array(senate), n = senate.count
        var rq = Deque<Int>(senate.indices.filter { senate[$0] == "R" }), dq = Deque<Int>(senate.indices.filter { senate[$0] == "D" })
        while !rq.isEmpty || !dq.isEmpty {
            guard let ri = rq.first else { return "Dire" }
            guard let di = dq.first else { return "Radiant" }
            // banning
            if ri < di {
                dq.removeFirst()
                rq.append(rq.removeFirst() + n)
            } else {
                rq.removeFirst()
                dq.append(dq.removeFirst() + n)
            }
        }
        fatalError("Incorrect input")
    }
    static func test() {
        let sut = Leet0649()
        assert(sut.predictPartyVictory("RD") == "Radiant")
        assert(sut.predictPartyVictory("DDRRR") == "Dire")
        assert(sut.predictPartyVictory("RR") == "Radiant")
        assert(sut.predictPartyVictory("DDDRRRRR") == "Radiant")
    }
}
//Leet0649.test()

/*
 "RD"
 "RDD"
 "RRDD"
 "DDRRR"
 "DRRDRDRDRDDRDRDR"
 "DDDRRRRR"
 "RRDDDDDDDRRDRRDDRRRR"
 
 "DDDRRRRR"
 "RDD"
 "DRRDRDRDRDDRDRDR"
 "DDRRR"
 "RDRDRDDRDRDRDRDRRDRDRDRDRDRDDDDRRDRDRDRDRDRDRDRRRRRDRDRDRDRDDDDDRDRDRDRDRDRDRDRRDRDRDRDRDRDRRDRDRDRDRDRDRDRDRRDRDRDRDRDRRD"
 "RRDDDDDDDRRDRRDDRRRR"
 "RDRDRDDRDDDDDDDRRDRRDDRRRRDDDDDDRRDRRDDRRRRDDDDDDRRDRRDDRRRRDDDDDDRRDRRDDRRRRDDDDDDRRDRRDDRRRRDDDDDDRRDRRDDRRRRDDDDDDRRDRRDDRRRRDDDDDDRRDRRDDRRRRRDRDDDDDDDRRDRRDDRRRRRDRRDRDRDDDDDDDRRDRRDDRRRRRDRDRDDDDRRDRDRDRDRDRDRDRRRRRDRDRDRDRDDDDDRDRDRDRDRDRDRDRRDRDRDRDRDRDRRDRDRDDDDDDDRRDRRDDRRRRRDRDRDRDRDRRDRDRDRDRDRRD"
 "DDR"
 */

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/longest-palindrome-by-concatenating-two-letter-words/
class Leet2131 {
    func longestPalindrome(_ words: [String]) -> Int {
        var syms = [String: Int](), asym = [String: Int](), result = 0
        // collect symetric and asymetric word counts
        for w in words {
            guard w.count == 2 else { continue }
            guard let f = w.first, let l = w.last else { continue }
            if l == f {
                syms[w, default: 0] += 1
            } else {
                asym[w, default: 0] += 1
            }
        }
        // all asymetric words must have matching words
        for w1 in asym {
            guard w1.value > 0 else { continue }
            guard let f = w1.key.first, let l = w1.key.last else { continue }
            let w2 = "\(l)\(f)"
            guard let c2 = asym[w2] else { continue }
            result += 2 * min(w1.value, c2)
            asym[w2] = 0
        }
        // all symetric words must be a multiple of two to pair each other except for one which can be the middle
        var foundCenter = false
        for v in syms.values.sorted(by: >) {
            if v.isMultiple(of: 2) {
                result += 2 * v
            } else {
                if foundCenter {
                    result += 2 * (v - 1)
                } else { // center can be 1 or any other odd number
                    if v == 1 {
                        result += 2
                    } else {
                        result += 2 * (v - 1) + 2
                    }
                    foundCenter = true
                    continue
                }
            }
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/largest-color-value-in-a-directed-graph/
class Leet1875 {
    func largestPathValue(_ colors: String, _ edges: [[Int]]) -> Int {
        let colors = Array(colors), n = colors.count
        var adj = [Int: [Int]](), indegree = [Int](repeating: 0, count: n)
        for e in edges {
            adj[e[0], default: []].append(e[1])
            indegree[e[1]] += 1
        }
        var count = [[Int]](repeating: [Int](repeating: 0, count: 26), count: n)
        var q = Deque<Int>()
        for i in 0..<n where indegree[i] == 0 {
            q.append(i)
        }
        var result = 1, nodeSeen = 0
        while !q.isEmpty {
            let node = q.removeFirst()
            let color = Int(colors[node].asciiValue! - Character("a").asciiValue!)
            count[node][color] += 1
            result = max(result, count[node][color])
            nodeSeen += 1
            guard let neighbors = adj[node] else { continue }
            for neighbor in neighbors {
                for i in 0..<26 {
                    count[neighbor][i] = max(count[neighbor][i], count[node][i])
                }
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0 {
                    q.append(neighbor)
                }
            }
        }
        return nodeSeen < n ? -1 : result
    }
    static func test() {
        let sut = Leet1875()
        assert(sut.largestPathValue("abaca", [[0,1],[0,2],[2,3],[3,4]]) == 3)
    }
}
//Leet1875.test()


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/longest-palindrome/
class Leet0409 {
    func longestPalindrome(_ s: String) -> Int {
        let counts = s.reduce(into: [Character: Int]()) { r, c in r[c, default: 0] += 1 }
        var isCenterFound = false
        return counts.values.reduce(into: 0) { result, count in
            if !isCenterFound && (count == 1 || !count.isMultiple(of: 2) ) {
                isCenterFound = true
                result += count
                return
            }
            if count.isMultiple(of: 2) {
                result += count
            } else {
                result += count - 1
            }
        }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/palindrome-permutation/
class Leet0266 {
    func canPermutePalindrome(_ s: String) -> Bool {
        Array(s)
            .reduce(into: [Character: Int]()) { r, c in r[c, default: 0] += 1 }
            .values
            .count { !$0.isMultiple(of: 2) } <= 1
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/divisible-and-non-divisible-sums-difference/
class Leet2894 {
    func differenceOfSums(_ n: Int, _ m: Int) -> Int {
        (1...n).reduce(0) { $0 + ($1.isMultiple(of: m) ? $1 : -$1) }
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/palindrome-permutation-ii/
///O(n * k)
class Leet0267 {
    func generatePalindromes(_ s: String) -> [String] {
        var freq = s.reduce(into: [Character: Int]()) { r, c in r[c, default: 0] += 1 }, center = ""
        //start from the center character, if existing
        let oddElems = freq.filter { !$1.isMultiple(of: 2) }, canPermute = oddElems.count <= 1
        guard canPermute else { return [] }
        if let c = oddElems.first?.key {
            freq.subtract(c)
            center = String(c)
        }
        var result: Set<String> = []
        backtrack(&freq, center, &result)
        return Array(result)
        
    }
    private func backtrack(_ freq: inout [Character: Int], _ current: String, _ result: inout Set<String>) {
        guard !freq.isEmpty else {
            result.insert(current)
            return
        }
        for (k, v) in freq {
            for _ in 0..<v/2 {
                freq.subtract(k, 2)
                let c = String(k)
                backtrack(&freq, c + current + c, &result)
                freq[k] = v
            }
        }
    }
    static func test() {
        let sut = Leet0267()
        assert(Set(sut.generatePalindromes("aabb")) == Set(["abba","baab"]))
    }
}

extension Dictionary where Key == Character, Value == Int {
    mutating func subtract(_ k: Character, _ by: Int = 1) {
        self[k, default: 0] -= by
        if let cnt = self[k], cnt <= 0 {
            self[k] = nil
        }
    }
}
//Leet0267.test()



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/largest-palindromic-number/
class Leet2384 {
    func largestPalindromic(_ num: String) -> String {
        let freq = num.reduce(into: [Character: Int]()) { counts, char in counts[char, default: 0] += 1 }
        var oddElems = freq.filter { !$1.isMultiple(of: 2) }, evenElems = freq.filter { $1.isMultiple(of: 2) }, result = "", center = ""
        if let maxOdd = oddElems.max(by: { $0.key < $1.key  }) {
            center = String(maxOdd.key)
        }
        for k in oddElems.keys {
            oddElems.subtract(k)
            guard let cnt = oddElems[k], cnt > 0 else { continue }
            evenElems[k] = oddElems[k]
        }
        let evens = evenElems.map { k, v in (String(k), v / 2) }.sorted(by: { $0.0 > $1.0 })
        for (str, cnt) in evens {
            // no leading zeroes
            guard result.isEmpty && str != "0" || !result.isEmpty else { continue }
            result += String(repeating: str, count: cnt)
        }
        let secondHalf = result.reversed()
        result += String(center) + secondHalf
        guard !result.isEmpty else { return "0" }
        return result
        
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/permutations-ii/
class Leet0047 {
    func permuteUnique(_ nums: [Int]) -> [[Int]] {
        var result = Set<[Int]>()
        backtrack([], nums, &result)
        return Array(result)
    }
    private func backtrack(_ current: [Int], _ nums: [Int], _ result: inout Set<[Int]>) {
        guard !nums.isEmpty else {
            result.insert(current)
            return
        }
        for i in 0..<nums.count {
            let n = nums[i]
            backtrack(current + [n], Array(nums[0..<i] + nums[i+1..<nums.count]), &result)
        }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/next-permutation/
class Leet0031 {
    func nextPermutation(_ nums: inout [Int]) {
        var i = nums.count - 2
        while i >= 0 && nums[i + 1] <= nums[i] {
            i -= 1
        }
        if i >= 0 {
            var j = nums.count - 1
            while j >= 0 && nums[j] <= nums[i] {
                j -= 1
            }
            nums.swapAt(i, j)
        }
        reverse(&nums, i + 1)
    }
    private func reverse(_ nums: inout [Int], _ start: Int) {
        var i = start, j = nums.count - 1
        while i < j {
            nums.swapAt(i, j)
            i += 1
            j -= 1
        }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximize-the-number-of-target-nodes-after-connecting-trees-i
class Leet3372 {
    func maxTargetNodes(_ edges1: [[Int]], _ edges2: [[Int]], _ k: Int) -> [Int] {
        let n = edges1.count + 1
        var count1 = build(edges1, k), count2 = build(edges2, k - 1), maxCount2 = 0
        for c in count2 {
            guard c > maxCount2 else { continue }
            maxCount2 = c
        }
        var result = [Int](repeating: 0, count: n)
        for i in 0..<n {
            result[i] = count1[i] + maxCount2
        }
        return result
    }
    
    private func build(_ edges: [[Int]], _ k: Int) -> [Int] {
        let n = edges.count + 1
        var children = [[Int]](repeating: [], count: n)
        for e in edges {
            let u = e[0], v = e[1]
            children[u].append(v)
            children[v].append(u)
        }
        var result = [Int](repeating: 0, count: n)
        for i in 0..<n {
            result[i] = dfs(i, -1, children, k)
        }
        return result
    }
    
    private func dfs(_ node: Int, _ parent: Int, _ children: [[Int]], _ k: Int) -> Int {
        guard k >= 0 else { return 0 }
        var result = 1
        for c in children[node] where c != parent {
            result += dfs(c, node, children, k - 1)
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximize-the-number-of-target-nodes-after-connecting-trees-ii/
class Leet3373 {
    func maxTargetNodes(_ edges1: [[Int]], _ edges2: [[Int]]) -> [Int] {
        let n = edges1.count + 1, m = edges2.count + 1
        var color1 = [Int](repeating: 0, count: n), color2 = [Int](repeating: 0, count: m)
        var count1 = build(edges1, &color1), count2 = build(edges2, &color2)
        var result = [Int](repeating: 0, count: n)
        for i in 0..<n {
            result[i] = count1[color1[i]] + max(count2[0], count2[1])
        }
        return result
    }
    
    private func build(_ edges: [[Int]], _ color: inout [Int]) -> [Int] {
        let n = edges.count + 1
        var children = [[Int]](repeating: [], count: n)
        for e in edges {
            let u = e[0], v = e[1]
            children[u].append(v)
            children[v].append(u)
        }
        let result = dfs(0, -1, 0, children, &color)
        return [result, n - result]
    }
    
    private func dfs(_ node: Int, _ parent: Int, _ depth: Int, _ children: [[Int]], _ color: inout [Int]) -> Int {
        var result = 1 - depth % 2
        color[node] = depth % 2
        for c in children[node] where c != parent {
            result += dfs(c, node, depth + 1, children, &color)
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sliding-window-maximum/
class Leet0239 {
    func maxSlidingWindow(_ nums: [Int], _ k: Int) -> [Int] {
        var result = [Int](), q = Deque<Int>()
        for i in 0..<nums.count {
            let n = nums[i]
            while let last = q.last, nums[last] < n {
                q.removeLast()
            }
            q.append(i)
            if let first = q.first, first + k == i {
                q.removeFirst()
            }
            if i >= k - 1, let first = q.first {
                result.append(nums[first])
            }
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/
class Leet1438 {
    func longestSubarray(_ nums: [Int], _ limit: Int) -> Int {
        var inc = Deque<Int>(), dec = Deque<Int>(), l = 0, result = 0
        for r in 0..<nums.count {
            // maintain monotonic deques
            while let lastI = inc.last, lastI > nums[r] {
                inc.removeLast()
            }
            while let lastD = dec.last, lastD < nums[r] {
                dec.removeLast()
            }
            inc.append(nums[r])
            dec.append(nums[r])
            // shrink window
            while let mx = dec.first, let mn = inc.first, mx - mn > limit {
                if nums[l] == mx {
                    dec.removeFirst()
                }
                if nums[l] == mn {
                    inc.removeFirst()
                }
                l += 1
            }
            result = max(result, r - l + 1)
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/online-stock-span/
class Leet0901 {

    typealias Span = (price: Int, span: Int)
    var stack: [Span] = []
    init() { stack = [] }
    
    func next(_ price: Int) -> Int {
        var result = 1
        while let top = stack.last, top.price <= price {
            result += top.span
            stack.removeLast()
        }
        stack.append((price: price, span: result))
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/remove-nodes-from-linked-list/
class Leet2487 {
    func removeNodes(_ head: ListNode?) -> ListNode? {
        var stack = [ListNode](), dummy: ListNode? = ListNode(0, head)
        while let curr = dummy?.next {
            while let last = stack.last, curr.val > last.val {
                stack.removeLast()
                last.next = nil
            }
            stack.last?.next = curr
            stack.append(curr)
            dummy = curr
        }
        return stack.first
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/delete-nodes-from-linked-list-present-in-array/
class Leet3217 {
    func modifiedList(_ nums: [Int], _ head: ListNode?) -> ListNode? {
        let s = Set(nums)
        var stack = [ListNode](), curr = head
        while curr != nil {
            let temp = curr?.next
            guard let c = curr else { break }
            if s.contains(c.val) {
                stack.last?.next = nil
                c.next = nil
            } else {
                stack.last?.next = c
                stack.append(c)
            }
            curr = temp
        }
        return stack.first
    }
    static func test() {
        let sut = Leet3217()
        assert(sut.modifiedList([5], [1,2,3,4].makeListNode())?.toArray() == [1,2,3,4])
        assert(sut.modifiedList([1,2,3], [1,2,3,4,5].makeListNode())?.toArray() == [4,5])
    }
}
//Leet3217.test()

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/convert-1d-array-into-2d-array/
class Leet2022 {
    func construct2DArray(_ original: [Int], _ m: Int, _ n: Int) -> [[Int]] {
        guard original.count == m * n else { return [] }
        var result = Array(repeating: Array(repeating: 0, count: n), count: m)
        for i in 0..<original.count {
            result[i / n][i % n] = original[i]
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/reshape-the-matrix/
class Leet0566 {
    func matrixReshape(_ mat: [[Int]], _ r: Int, _ c: Int) -> [[Int]] {
        let m = mat.count
        guard let n = mat.first?.count, m * n == r * c, m != r, n != c else { return mat }
        let flat = mat.flatMap { $0 }
        var result = Array(repeating: Array(repeating: 0, count: c), count: r)
        for i in 0..<flat.count {
            result[i / c][i % c] = flat[i]
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/row-with-maximum-ones/
class Leet2643 {
    func rowAndMaximumOnes(_ mat: [[Int]]) -> [Int] {
        let sums = mat.map { l in l.reduce(0, +) }
        guard let x = sums.max(), let first = sums.enumerated().filter({ e in e.element == x }).first else { return [] }
        return [first.offset, x]
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/most-visited-sector-in-a-circular-track/
class Leet1560 {
    func mostVisited(_ n: Int, _ rounds: [Int]) -> [Int] {
        guard let f = rounds.first, let l = rounds.last else { return [] }
        if f < l {
            return Array(f...l)
        } else if f > l {
            return Array(1...l) + Array(f...n)
        } else {
            return [f]
        }
    }
    static func test() {
        let sut = Leet1560()
        assert(sut.mostVisited(7, [1,3,5,7]) == [1,2,3,4,5,6,7])
        assert(sut.mostVisited(2, [2,1,2,1,2,1,2,1,2]) == [2])
        assert(sut.mostVisited(4, [1,3,1,2]) == [1,2])
    }
}
//Leet1560.test()


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sort-even-and-odd-indices-independently/
class Leet2164 {
    func sortEvenOdd(_ nums: [Int]) -> [Int] {
        var odds = [Int](), evens = [Int]()
        for (i, n) in nums.enumerated() {
            if i.isMultiple(of: 2) {
                evens.append(n)
            } else {
                odds.append(n)
            }
        }
        odds.sort { $0 > $1 }
        evens.sort()
        var result = [Int](repeating: 0, count: nums.count), i = 0, j = 1
        for n in evens {
            result[i] = n
            i += 2
        }
        for n in odds {
            result[j] = n
            j += 2
        }
        return result
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/validate-stack-sequences/
class Leet0946 {
    func validateStackSequences(_ pushed: [Int], _ popped: [Int]) -> Bool {
        var stack = [Int](), i = 0
        for sh in pushed {
            var pop = popped[i]
            if sh == pop {
                i += 1
            } else {
                while let last = stack.last, last == pop {
                    stack.removeLast()
                    i += 1
                    pop = popped[i]
                }
                stack.append(sh)
            }
        }
        for pop in popped[i...] {
            guard let last = stack.last, last == pop else { continue }
            stack.removeLast()
        }
        return stack.isEmpty
    }
    
    static func test() {
        let sut = Leet0946()
        assert(sut.validateStackSequences([3,2,1,5,4], [1,2,3,4,5]))
        assert(sut.validateStackSequences([3,2,1,4,5], [1,2,3,4,5]))
    }
}
//Leet0946.test()

/*
 
 
 [1,2,3,4,5]
 [4,5,3,2,1]
 [1,2,3,4,5]
 [4,3,5,1,2]
 [2,1,0]
 [1,2,0]
 [2,1,0]
 [1,0,2]
 [3,2,1,5,4]
 [1,2,3,4,5]
 [3,2,1,4,5]
 [1,2,3,4,5]

*/


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-closest-node-to-given-two-nodes/
class Leet2359 {
    func closestMeetingNode(_ edges: [Int], _ node1: Int, _ node2: Int) -> Int {
        let n = edges.count
        var distance1 = [Int](repeating: .max, count: n), distance2 = [Int](repeating: .max, count: n)
        bfs(node1, edges, &distance1)
        bfs(node2, edges, &distance2)
        
        var minDistance = Int.max, closestNode = -1
        for i in 0..<n {
            let mx = max(distance1[i], distance2[i])
            if minDistance > mx {
                closestNode = i
                minDistance = mx
            }
        }
        return closestNode
    }
    
    private func bfs(_ start: Int, _ edges: [Int], _ distance: inout [Int]) {
        let n = edges.count
        var q = Deque<Int>([start]), visited = Set<Int>()
        distance[start] = 0
        while !q.isEmpty {
            let n = q.removeFirst()
            guard !visited.contains(n) else { continue }
            visited.insert(n)
            let neighbor = edges[n]
            if neighbor != -1 && !visited.contains(neighbor) {
                distance[neighbor] = distance[n] + 1
                q.append(neighbor)
            }
        }
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/asteroid-collision/
class Leet0735 {
    func asteroidCollision(_ asteroids: [Int]) -> [Int] {
        let isColliding: (Int, Int) -> Bool = { $0 > 0 && $1 < 0 }
        var stack = [Int]()
        outer: for a in asteroids {
            while let last = stack.last, isColliding(last, a) {
                if abs(last) < abs(a) {
                    stack.removeLast()
                } else if abs(last) > abs(a) {
                    continue outer
                } else {
                    stack.removeLast()
                    continue outer
                }
            }
            stack.append(a)
        }
        return stack
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-collisions-on-a-road/
class Leet2211 {
    func countCollisions(_ directions: String) -> Int {
        var d = Deque(directions), stack = [Character](), result = 0
        while !d.isEmpty {
            // "RL" becomes "S"
            while let last = stack.last, last == "R", let f = d.first, f == "L" {
                result += 2
                stack.removeLast()
                d.removeFirst()
                d.prepend(contentsOf: "S")
            }
            // "RS" collision
            while let last = stack.last, last == "R", let f = d.first, f == "S" {
                result += 1
                stack.removeLast()
            }
            // "SL collision"
            while let last = stack.last, last == "S", let f = d.first, f  == "L" {
                result += 1
                d.removeFirst()
            }
            guard let f = d.first else { break }
            stack.append(f)
            d.removeFirst()
        }
        return result
    }
}


/*
 "RSRSLLLRSRSSRRL"
 "RRRRRLLLLL"
 "RLRSLL"
 "LLRR"
 "LLLLLLRSRSLLLRSRSSRRLRRRRRRRRRR"
 "SSRSSRLLRSLLRSRSSRLRRRRLLRRLSSRR"
 "LLRLRLLSLRLLSLSSSS"
 */


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/snakes-and-ladders/
class Leet0909 {
    func snakesAndLadders(_ board: [[Int]]) -> Int {
        let n = board.count
        var cells = [(Int, Int)](repeating: (-1, -1), count: n * n + 1), label = 1, columns = Array(0..<n)
        for row in stride(from: n - 1, through: 0, by: -1) {
            for column in columns {
                cells[label] = (row, column)
                label += 1
            }
            columns.reverse()
        }
        var distance = [Int](repeating: -1, count: n * n + 1), q = Deque<Int>([1])
        distance[1] = 0
        while !q.isEmpty {
            let current = q.removeFirst()
            var next = current + 1
            while next <= min(current + 6, n * n) {
                defer { next += 1 }
                let (r, c) = cells[next], destination = board[r][c] != -1 ? board[r][c] : next
                guard distance[destination] == -1 else { continue }
                distance[destination] = distance[current] + 1
                q.append(destination)
            }
        }
        return distance[n * n]
    }
    static func test() {
        let sut = Leet0909()
        assert(sut.snakesAndLadders([[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,35,-1,-1,13,-1],[-1,-1,-1,-1,-1,-1],[-1,15,-1,-1,-1,-1]]) == 4)
    }
}
//Leet0909.test()


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/distribute-candies-among-children-iii/
class Leet2927 {
    func distributeCandies(_ n: Int, _ limit: Int) -> Int {
        cal(n + 2)
        - 3 * cal(n - limit + 1)
        + 3 * cal(n - (limit + 1) * 2 + 2)
        - cal(n - 3 * (limit + 1) + 2)
    }
    private func cal(_ x: Int) -> Int {
        guard x > 0 else { return 0 }
        return x * (x - 1) / 2
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/distribute-candies-among-children-i/
class Leet2928 {
    func distributeCandies(_ n: Int, _ limit: Int) -> Int {
        Array(0...min(n, limit)).reduce(0) { r, i in
            guard limit * 2 >= n - i else { return r }
            return r + min(n - i, limit) - max(0, n - i - limit) + 1
        }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/distribute-candies-among-children-ii/
class Leet2929 {
    func distributeCandies(_ n: Int, _ limit: Int) -> Int {
        Array(0...min(n, limit)).reduce(0) { r, i in
            guard limit * 2 >= n - i else { return r }
            return r + min(n - i, limit) - max(0, n - i - limit) + 1
        }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/smallest-even-multiple/
class Leet2413 {
    func smallestEvenMultiple(_ n: Int) -> Int {
        n & 1 == 0 ? n << 1 : n
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/greatest-common-divisor-of-strings/
class Leet1071 {
    func gcdOfStrings(_ str1: String, _ str2: String) -> String {
        guard str1 + str2 == str2 + str1 else { return "" }
        let l = gcd(str1.count, str2.count)
        return String(str1[str1.startIndex..<str1.index(str1.startIndex, offsetBy: l)])
    }
    private func gcd(_ x: Int, _ y: Int) -> Int {
        y == 0 ? x : gcd(y, x % y)
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/candy/
class Leet0135 {
    func candy(_ ratings: [Int]) -> Int {
        var candies = [Int](repeating: 1, count: ratings.count)
        for i in 1..<ratings.count where ratings[i] > ratings[i - 1] {
            candies[i] = candies[i - 1] + 1
        }
        guard var result = candies.last else { return 0 }
        for i in (0..<(ratings.count - 1)).reversed() {
            if  ratings[i] > ratings[i + 1] {
                candies[i] = max(candies[i], candies[i + 1] + 1)
            }
            result += candies[i]
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/last-moment-before-all-ants-fall-out-of-a-plank/
class Leet1503 {
    func getLastMoment(_ n: Int, _ left: [Int], _ right: [Int]) -> Int {
        right.reduce(into: left.max() ?? 0) { $0 = max($0, n - $1) }
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/destroying-asteroids/
class Leet2126 {
    func asteroidsDestroyed(_ mass: Int, _ asteroids: [Int]) -> Bool {
        var mass = mass
        for a in asteroids.sorted() {
            guard mass >= a else { return false }
            mass += a
        }
        return true
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/convert-the-temperature/
class Leet2469 {
    func convertTemperature(_ celsius: Double) -> [Double] {
        [celsius + 273.15, celsius * 9.0 / 5.0 + 32.0 ]
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/three-divisors/
class Leet1952 {
    func isThree(_ n: Int) -> Bool {
        Array(1...n).filter { d in n.isMultiple(of: d) }.count == 3
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-cuts-to-divide-a-circle/
class Leet2481 {
    func numberOfCuts(_ n: Int) -> Int {
        n > 1 ? (n.isMultiple(of: 2) ? n/2 : n) : 0
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/car-fleet/
class Leet0853 {
    func carFleet(_ target: Int, _ position: [Int], _ speed: [Int]) -> Int {
        typealias Car = (position: Int, time: Double)
        var cars = zip(position, speed)
            .map { p, s in Car(position: p, time: Double(target - p) / Double(s)) }
            .sorted { c1, c2  in c1.position < c2.position }
        var result = 0, t = position.count
        print(cars)
        while t > 0 {
            t -= 1
            print("\(cars[t]) \(cars[t-1])")
            if cars[t].time < cars[t-1].time {
                result += 1
            } else {
                cars[t-1] = cars[t]
            }
        }
        return result + (t == 0 ? 1 : 0)
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/can-place-flowers/
class Leet0605 {
    func canPlaceFlowers(_ flowerbed: [Int], _ n: Int) -> Bool {
        var count = 0, flowerbed = flowerbed
        for i in 0..<flowerbed.count where flowerbed[i] == 0 {
            let isPrevEmpty = (i == 0) || (flowerbed[i - 1] == 0)
            let isNextEmpty = (i == flowerbed.count - 1) || (flowerbed[i + 1] == 0)
            if isPrevEmpty && isNextEmpty  {
                flowerbed[i] = 1
                count += 1
            }
        }
        return count >= n
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/number-of-valid-subarrays/
class Leet1063 {
    func validSubarrays(_ nums: [Int]) -> Int {
        var result = 0, stack = [Int]()
        for i in 0..<nums.count {
            while let last = stack.last, nums[i] < nums[last] {
                result += i - last
                stack.removeLast()
            }
            stack.append(i)
        }
        while let last = stack.last {
            result += nums.count - last
            stack.removeLast()
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sum-of-subarray-minimums/
class Leet0907 {
    func sumSubarrayMins(_ arr: [Int]) -> Int {
        let mod = 1_000_000_007
        var stack = [Int](), result = 0
        for i in 0...arr.count {
            while let last = stack.last, i == arr.count || arr[last] >= arr[i] {
                let mid = stack.removeLast()
                let l = stack.isEmpty ? -1 : stack.last!
                let r = i
                let count = (mid - l) * (r - mid) % mod
                result += (count * arr[mid]) % mod
                result %= mod
            }
            stack.append(i)
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sum-of-subarray-ranges/
class Leet2104 {
    func subArrayRanges(_ nums: [Int]) -> Int {
        let n = nums.count
        var result = 0, stack = [Int]()
        for r in 0...n {
            while let last = stack.last, r == n || nums[last] >= nums[r] {
                let mid = stack.removeLast()
                let l = stack.last ?? -1
                result -= nums[mid] * (r - mid) * (mid - l)
            }
            stack.append(r)
        }
        stack.removeAll()
        for r in 0...n {
            while let last = stack.last, r == n || nums[last] <= nums[r] {
                let mid = stack.removeLast()
                let l = stack.last ?? -1
                result += nums[mid] * (r - mid) * (mid - l)
            }
            stack.append(r)
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-the-most-competitive-subsequence/
class Leet1673 {
    func mostCompetitive(_ nums: [Int], _ k: Int) -> [Int] {
        var q = Deque<Int>(), additionalCount = nums.count - k
        for i in 0..<nums.count {
            while let last = q.last, last > nums[i], additionalCount > 0 {
                q.removeLast()
                additionalCount -= 1
            }
            q.append(nums[i])
        }
        var result = [Int]()
        for i in 0..<k {
            result.append(q.removeFirst())
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-candies-you-can-get-from-boxes/
class Leet1298 {
    func maxCandies(_ status: [Int], _ candies: [Int], _ keys: [[Int]], _ containedBoxes: [[Int]], _ initialBoxes: [Int]) -> Int {
        let n = status.count
        var canOpen = status.map { $0 == 1 }, hasBox = [Bool](repeating: false, count: n), used = [Bool](repeating: false, count: n)
        var q = Deque<Int>(), result = 0
        for b in initialBoxes {
            hasBox[b] = true
            guard canOpen[b] else { continue }
            q.append(b)
            used[b] = true
            result += candies[b]
        }
        while let b = q.popFirst() {
            for k in keys[b] {
                canOpen[k] = true
                guard !used[k], hasBox[k] else { continue }
                q.append(k)
                used[k] = true
                result += candies[k]
            }
            for b in containedBoxes[b] {
                hasBox[b] = true
                guard !used[b], canOpen[b] else { continue }
                q.append(b)
                used[b] = true
                result += candies[b]
            }
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/number-of-visible-people-in-a-queue/
class Leet1944 {
    func canSeePersonsCount(_ heights: [Int]) -> [Int] {
        var result = [Int](), stack: [Int] = []
        for h in heights.reversed() {
            var count = 0
            while let last = stack.last, last < h {
                count += 1
                stack.removeLast()
            }
            if let last = stack.last, last > h {
                count += 1
            }
            result.append(count)
            stack.append(h)
        }
        return result.reversed()
    }
}

/**
 [10,6,8,5,11,9]
 [11,19,12,15,14,18,7,1,8,9]
 [5,1,2,3,10]
 [3,1,5,8,6]
 [10,8,7,6,11,9]
 [10, 5, 8, 12, 3, 7, 20, 6, 15]
 [5, 1, 4, 2, 3, 6]
 [10, 3, 7, 4, 12]
 [5, 4, 3, 2, 1]
 
 */

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/using-a-robot-to-print-the-lexicographically-smallest-string/
class Leet2434 {
    func robotWithString(_ s: String) -> String {
        let s = Array(s), charZ = Character("z")
        var counts = s.reduce(into: [Character: Int]()) { c, v in c[v, default: 0] += 1 }
        var stack = [Character](), result = [Character](), minChar = Character("a")
        for c in s {
            stack.append(c)
            counts[c, default: 0] -= 1
            if counts[c] == 0 {
                counts[c] = nil
            }
            while minChar < charZ, counts[minChar] ?? 0 == 0 {
                minChar = Character(UnicodeScalar(Int(minChar.asciiValue!) + 1)!)
            }
            while let last = stack.last, last <= minChar {
                result.append(stack.removeLast())
            }
        }
        return String(result)
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-number-of-robots-within-budget/
class Leet2398 {
    func maximumRobots(_ chargeTimes: [Int], _ runningCosts: [Int], _ budget: Int) -> Int {
        var l = 0, result = 0, sum = 0, deq = Deque<Int>()
        for r in 0..<chargeTimes.count {
            let cr = chargeTimes[r]
            while let last = deq.last, chargeTimes[last] < cr {
                deq.removeLast()
            }
            deq.append(r)
            sum += runningCosts[r]
            var k = r - l + 1
            // shrink
            while let f = deq.first, chargeTimes[f] + k * sum > budget {
                sum -= runningCosts[l]
                l += 1
                k -= 1
                if f < l {
                    deq.removeFirst()
                }
            }
            k = r - l + 1
            result = max(result, k)
        }
        return result
    }
    static func test() {
        let sut = Leet2398()
        assert(sut.maximumRobots([19,63,21,8,5,46,56,45,54,30,92,63,31,71,87,94,67,8,19,89,79,25],[91,92,39,89,62,81,33,99,28,99,86,19,5,6,19,94,65,86,17,10,8,42], 85) == 1)
    }
}
//Leet2398.test()



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-the-lexicographically-largest-string-from-the-box-i/
class Leet3403 {
    func answerString(_ word: String, _ numFriends: Int) -> String {
        guard numFriends > 1 else { return word }
        let last = Array(lastSubstring(word)), n = word.count, m = last.count
        return String(last[..<min(m, n - numFriends + 1)])
    }
    private func lastSubstring(_ s: String) -> ArraySlice<Character> {
        let s = Array(s), n = s.count
        var i = 0, j = 1
        while j < n {
            var k = 0
            while j + k < n, s[i + k] == s[j + k] {
                k += 1
            }
            if j + k < n, s[i + k] < s[j + k] {
                let t = i
                i = j
                j = max(j + 1, t + k + 1)
            } else {
                j = j + k + 1
            }
        }
        return s[i...]
    }
    static func test() {
        let sut = Leet3403()
        assert(sut.answerString("yxz", 3) == "z")
    }
}
//Leet3403.test()


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/binary-gap/
class Leet0868 {
    func binaryGap(_ n: Int) -> Int {
        var bin = Array(0..<32).map { i in (n >> i) & 1 == 1 }
        var result = 0, l = 0, window = 0
        for r in 0..<bin.count where bin[r] {
            window += 1
            if window == 1 {
                l = r
            } else if window == 2 {
                result = max(result, r - l)
                l = r
                window = 1
            }
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/add-digits/
class Leet0258 {
    func addDigits(_ num: Int) -> Int {
        1 + (num - 1) % 9
    }
    
    
    func addDigits2(_ num: Int) -> Int {
        guard num > 9 else { return num }
        var num = num, sum = 0
        while num > 0 {
            sum += num % 10
            num /= 10
        }
        return addDigits(sum)
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/subtree-of-another-tree/
class Leet0572 {
    func isSubtree(_ root: TreeNode?, _ subRoot: TreeNode?) -> Bool {
        guard let root, let subRoot else { return false }
        let subList = subRoot.buildArray()
        let candidates = findCandidates(root, subRoot)
        for c in candidates where subList == c.buildArray() {
            return true
        }
        return false
    }
    private func findCandidates(_ root: TreeNode, _ subRoot: TreeNode) -> [TreeNode] {
        var deque: Deque<TreeNode?> = [root], result: [TreeNode] = []
        while !deque.isEmpty {
            for _ in deque {
                let node = deque.removeFirst()
                guard let node = node else { continue }
                guard node.val != subRoot.val else {
                    result.append(node)
                    continue
                }
                deque.append(node.left)
                deque.append(node.right)
            }
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/lexicographically-smallest-equivalent-string/
class Leet1061 {
    var map = [Character: Character](uniqueKeysWithValues: (Character("a").asciiValue!...Character("z").asciiValue!).map { Character(UnicodeScalar($0)) }.map { ($0, $0) })
    func smallestEquivalentString(_ s1: String, _ s2: String, _ baseStr: String) -> String {
        for (c1, c2) in zip(Array(s1), Array(s2)) {
            union(c1, c2)
        }
        return String(baseStr.map(find))
    }
    private func union(_ c: Character, _ d: Character) {
        let c = find(c), d = find(d)
        map[max(c,d)] = min(c,d)
    }
    private func find(_ c: Character) -> Character {
        guard let v = map[c], v != c else { return c }
        map[c] = find(v)
        return map[c]!
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/spiral-matrix/
class Leet0054 {
        
    private struct Cell { let r: Int; let c: Int }
    private enum Direction {
        case right, down, left, up
        var vector: Cell {
            switch self {
            case .right: return Cell(r: 0, c: 1)
            case .down: return Cell(r: 1, c: 0)
            case .left: return Cell(r: 0, c: -1)
            case .up: return Cell(r: -1, c: 0)
            }
        }
        var nexts: [Direction] {
            switch self {
            case .right: return [.right, .down]
            case .down: return [.down, .left]
            case .left: return [.left, .up]
            case .up: return [.up, .right]
            }
        }
    }
    func spiralOrder(_ matrix: [[Int]]) -> [Int] {
        let m = matrix.count, n = matrix[0].count, visited = 101
        var matrix = matrix, result = [Int](), d: Direction = .right, next: Cell? = Cell(r: 0, c: 0)
        while let n = next {
            // a cell will be set to 101 to mark visit
            result.append(matrix[n.r][n.c])
            matrix[n.r][n.c] = visited
            // calculate the next cell. priority is the current direction. the next cell must be in bound and not visited
            if let x = d.nexts.first, let c = nextCandidate(x.vector, n) {
                next = c
                d = x
            } else if let x = d.nexts.last, let c = nextCandidate(x.vector, n) {
                next = c
                d = x
            } else {
                next = nil
            }
        }
        func nextCandidate(_ vector: Cell, _ current: Cell) -> Cell? {
            let o = Cell(r: current.r + vector.r, c: current.c + vector.c)
            guard 0..<m ~= o.r, 0..<n ~= o.c, matrix[o.r][o.c] != visited else { return nil }
            return o
        }
        return result
    }
}


/*
 
 [[1,2,9,9,3,4,5,1],[1,2,9,9,1,4,1,1],[1,10,9,9,3,4,5,1],[1,21,9,9,3,4,5,1],[1,2,9,3,3,4,3,3],[1,2,19,9,13,4,15,1]]
 [[1,2,3,4,3],[5,6,7,8,1],[9,10,11,12,1]]
 [[1,2],[2,5]]
 [[1],[2]]
 [[1]]
 
 
 [[1,2,3,4]]
 [[1,2,3,4],[5,6,7,8]]
 [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]]
 [[1]]
 [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
 [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]]
 [[1],[2],[3],[4]]
 
 */

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/spiral-matrix-ii/
class Leet0059 {
    private struct Cell { let r: Int; let c: Int }
    private enum Direction {
        case right, down, left, up
        var vector: Cell {
            switch self {
            case .right: return Cell(r: 0, c: 1)
            case .down: return Cell(r: 1, c: 0)
            case .left: return Cell(r: 0, c: -1)
            case .up: return Cell(r: -1, c: 0)
            }
        }
        var nexts: [Direction] {
            switch self {
            case .right: return [.right, .down]
            case .down: return [.down, .left]
            case .left: return [.left, .up]
            case .up: return [.up, .right]
            }
        }
    }
    func generateMatrix(_ n: Int) -> [[Int]] {
        var result = [[Int]](repeating: [Int](repeating: 0, count: n), count: n), d: Direction = .right, next: Cell? = Cell(r: 0, c: 0), i = 1
        while let n = next {
            result[n.r][n.c] = i
            i += 1
            if let x = d.nexts.first, let c = nextCandidate(x.vector, n) {
                next = c
                d = x
            } else if let x = d.nexts.last, let c = nextCandidate(x.vector, n) {
                next = c
                d = x
            } else {
                next = nil
            }
        }
        return result
        func nextCandidate(_ vector: Cell, _ current: Cell) -> Cell? {
            let o = Cell(r: current.r + vector.r, c: current.c + vector.c)
            guard 0..<n ~= o.r, 0..<n ~= o.c, result[o.r][o.c] == 0 else { return nil }
            return o
        }
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/spiral-matrix-iii/
class Leet0885 {
    
    private struct Cell { let r: Int; let c: Int }
    private enum Direction {
        case right, down, left, up
        var vector: Cell {
            switch self {
            case .right: return Cell(r: 0, c: 1)
            case .down: return Cell(r: 1, c: 0)
            case .left: return Cell(r: 0, c: -1)
            case .up: return Cell(r: -1, c: 0)
            }
        }
        var nexts: [Direction] {
            switch self {
            case .right: return [.right, .down]
            case .down: return [.down, .left]
            case .left: return [.left, .up]
            case .up: return [.up, .right]
            }
        }
    }
    func spiralMatrixIII(_ rows: Int, _ cols: Int, _ rStart: Int, _ cStart: Int) -> [[Int]] {
        var matrix = [[Int]](repeating: [Int](repeating: 0, count: cols), count: rows), result = [[Int]]()
        var d: Direction = .right, pos: Cell = Cell(r: rStart, c: cStart), i = 1, stepsLimit = 1, steps = 0, changes = 0
        while i <= rows * cols {
            if let c = validCell(pos) {
                result.append([c.r, c.c])
                matrix[c.r][c.c] = i
                i += 1
            }
            if steps < stepsLimit {
                steps += 1
            } else {
                steps = 1
                changes += 1
                if changes == 2 {
                    changes = 0
                    stepsLimit += 1
                }
                d = d.nexts.last!
            }
            pos = nextPosition(d.vector, pos)
        }
        return result
        func nextPosition(_ vector: Cell, _ pos: Cell) -> Cell {
            Cell(r: pos.r + vector.r, c: pos.c + vector.c)
        }
        func validCell(_ pos: Cell) -> Cell? {
            guard 0..<rows ~= pos.r, 0..<cols ~= pos.c, matrix[pos.r][pos.c] == 0 else { return nil }
            return pos
        }
    }
    static func test() {
        let sut = Leet0885()
        assert(sut.spiralMatrixIII(5, 6, 1, 4) == [[1,4],[1,5],[2,5],[2,4],[2,3],[1,3],[0,3],[0,4],[0,5],[3,5],[3,4],[3,3],[3,2],[2,2],[1,2],[0,2],[4,5],[4,4],[4,3],[4,2],[4,1],[3,1],[2,1],[1,1],[0,1],[4,0],[3,0],[2,0],[1,0],[0,0]])
        assert(sut.spiralMatrixIII(1, 4, 0, 0) == [[0,0],[0,1],[0,2],[0,3]])
    }
}
//Leet0885.test()


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/spiral-matrix-iv/
class Leet2326 {
    private struct Cell { let r: Int; let c: Int }
    private enum Direction {
        case right, down, left, up
        var vector: Cell {
            switch self {
            case .right: return Cell(r: 0, c: 1)
            case .down: return Cell(r: 1, c: 0)
            case .left: return Cell(r: 0, c: -1)
            case .up: return Cell(r: -1, c: 0)
            }
        }
        var nexts: [Direction] {
            switch self {
            case .right: return [.right, .down]
            case .down: return [.down, .left]
            case .left: return [.left, .up]
            case .up: return [.up, .right]
            }
        }
    }
    func spiralMatrix(_ m: Int, _ n: Int, _ head: ListNode?) -> [[Int]] {
        var result = [[Int]](repeating: [Int](repeating: -1, count: n), count: m), curr = head
        var d: Direction = .right, next: Cell? = Cell(r: 0, c: 0)
        while let node = curr, let c = next {
            result[c.r][c.c] = node.val
            if let x = d.nexts.first, let c = nextCandidate(x.vector, c) {
                next = c
                d = x
            } else if let x = d.nexts.last, let c = nextCandidate(x.vector, c) {
                next = c
                d = x
            } else {
                next = nil
            }
            curr = node.next
        }
        
        return result
        
        func nextCandidate(_ vector: Cell, _ current: Cell) -> Cell? {
            let o = Cell(r: current.r + vector.r, c: current.c + vector.c)
            guard 0..<m ~= o.r, 0..<n ~= o.c, result[o.r][o.c] == -1 else { return nil }
            return o
        }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/remove-one-element-to-make-the-array-strictly-increasing/
class Leet1909 {
    func canBeIncreasing(_ nums: [Int]) -> Bool {
        var count = 0, hi = nums[0]
        for i in 1 ..< nums.count {
            if hi >= nums[i] { // dip
                count += 1
                if i > 1, nums[i - 2] >= nums[i] {
                    hi = nums[i - 1]
                    continue
                }
            }
            hi = nums[i]
        }
        return count < 2
    }
    
    static func test() {
        let sut = Leet1909()
        assert(sut.canBeIncreasing([3,1,2]))
        assert(sut.canBeIncreasing([1,3,2,3]))
        assert(sut.canBeIncreasing([2,3,2,4]))
        assert(sut.canBeIncreasing([1,2,10,5,7]))
        assert(sut.canBeIncreasing([2,3,1,2]) == false)
        assert(sut.canBeIncreasing([100, 21, 100]))
        assert(sut.canBeIncreasing([105, 924, 32, 968]))
        assert(sut.canBeIncreasing([1]))
        assert(sut.canBeIncreasing([1,1]))
        assert(sut.canBeIncreasing([1,1,1]) == false)
    }
}
//Leet1909.test()

/*
 [1,2,10,5,7]
 [2,3,1,2]
 [100, 21, 100]
 [105, 924, 32, 968]
 [1]
 [1,1]
 [1,1,1]
 [3,1,2]
 [1,3,2,3]
 [2,3,2,4]
 */


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/search-in-rotated-sorted-array/
class Leet0033 {
    func search(_ nums: [Int], _ target: Int) -> Int {
        // binary search for the pivot index
        let n = nums.count
        var low = 0, high = n - 1
        while low <= high {
            let mid = low + (high - low) / 2
            if nums[mid] > nums[n - 1] {
                low = mid + 1
            } else {
                high = mid - 1
            }
        }
        let pivot = low, result = binarySearch(nums, 0, pivot - 1, target)
        if result != -1 {
            return result
        }
        return binarySearch(nums, pivot, n - 1, target)
    }
    private func binarySearch(_ nums: [Int], _ low: Int, _ high: Int, _ target: Int) -> Int {
        var low = low, high = high
        while low <= high {
            let mid = low + (high - low) / 2
            if nums[mid] == target {
                return mid
            } else if target < nums[mid] {
                high = mid - 1
            } else {
                low = mid + 1
            }
        }
        return -1
    }
        
    static func test() {
        let sut = Leet0033()
        assert(sut.search([1,3], 3) == 1)
        assert(sut.search([1], 1) == 0)
        assert(sut.search([1], 0) == -1)
        assert(sut.search([4,5,6,7,0,1,2], 0) == 4)
    }
}
//Leet0033.test()

/*
 [1,3,5]
 5
 [1,3]
 0
 [1,3]
 3
 [3,1]
 3
 [3,1]
 0
 
 
 [4,5,6,7,0,1,2]
 0
 [4,5,6,7,0,1,2]
 3
 [1]
 1
 [1]
 0
 [1,2,3,4,5,6,7,8,9]
 5
 [6,7,1,2,3,4,5]
 6
 [4,5,6,7,8,1,2,3]
 3
 [8,9,10,1,2,3,4,5,6,7]
 11
 
 */


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/search-in-rotated-sorted-array-ii/
class Leet0081 {
    func search(_ nums: [Int], _ target: Int) -> Bool {
        // binary search for the pivot index
        let n = nums.count
        var lo = 0, hi = n - 1
        while lo <= hi {
            let mid = lo + (hi - lo) / 2
            
            if nums[mid] == target {
                return true
            }

            if nums[lo] == nums[mid], nums[mid] == nums[hi] {
                lo += 1
                hi -= 1
            } else if nums[lo] <= nums[mid] {
                if nums[lo] <= target && target < nums[mid] {
                    hi = mid - 1
                } else {
                    lo = mid + 1
                }
            } else {
                if nums[mid] < target && target <= nums[hi] {
                    lo = mid + 1
                } else {
                    hi = mid - 1
                }
            }
        }
        return false
    }
    static func test() {
        let sut = Leet0081()
        assert(sut.search([1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1], 2))
    }
}
//Leet0081.test()




///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
class Leet0153 {
    func findMin(_ nums: [Int]) -> Int {
        // binary search for the pivot index
        let n = nums.count
        var lo = 0, hi = n - 1
        while lo <= hi {
            let mid = lo + (hi - lo) / 2
            if nums[mid] > nums[n - 1] {
                lo = mid + 1
            } else {
                hi = mid - 1
            }
        }
        return nums[lo]
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/
class Leet0154 {
    func findMin(_ nums: [Int]) -> Int {
        let n = nums.count
        var lo = 0, hi = n - 1
        while lo < hi {
            let mid = lo + (hi - lo) / 2
            if nums[mid] < nums[hi] {
                hi = mid
            } else if nums[mid] > nums[hi] {
                lo = mid + 1
            } else {
                hi -= 1
            }
        }
        return nums[lo]
    }
}
 

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/
class Leet0103 {
    func zigzagLevelOrder(_ root: TreeNode?) -> [[Int]] {
        var result = [[Int]](), deque = Deque<TreeNode?>([root]), level = 1
        // breadth first search traverse the tree alternating left to right, and right to left
        while !deque.isEmpty {
            var levelNodes = [Int]()
            for _  in deque {
                guard let node = deque.removeFirst() else { continue }
                levelNodes.append(node.val)
                deque.append(node.left)
                deque.append(node.right)
            }
            if !levelNodes.isEmpty {
                if level.isMultiple(of: 2) {
                    result.append(levelNodes.reversed())
                } else {
                    result.append(levelNodes)
                }
            }
            level += 1
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/binary-tree-level-order-traversal-ii/
class Leet0107 {
    func levelOrderBottom(_ root: TreeNode?) -> [[Int]] {
        var result = [[Int]](), deque = Deque<TreeNode?>([root])
        while !deque.isEmpty {
            var levelNodes = [Int]()
            for _ in deque {
                guard let node = deque.removeFirst() else { continue }
                levelNodes.append(node.val)
                deque.append(node.left)
                deque.append(node.right)
            }
            guard levelNodes.count > 0 else { continue }
            result.append(levelNodes)
        }
        return result.reversed()
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/average-of-levels-in-binary-tree/
class Leet0637 {
    func averageOfLevels(_ root: TreeNode?) -> [Double] {
        var result = [Double](), deque = Deque<TreeNode?>([root])
        while !deque.isEmpty {
            var count = 0, sum = 0.0
            for _ in deque {
                guard let node = deque.removeFirst() else { continue }
                sum += Double(node.val)
                count += 1
                deque.append(node.left)
                deque.append(node.right)
            }
            guard count > 0 else { continue }
            result.append(sum / Double(count))
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/zigzag-grid-traversal-with-skip/
class Leet3417 {
    func zigzagTraversal(_ grid: [[Int]]) -> [Int] {
        var result = [Int]()
        for i in 0..<grid.count {
            if i % 2 == 0 {
                for j in 0..<grid[i].count where j % 2 == 0 {
                    result.append(grid[i][j])
                }
            } else {
                var list = [Int]()
                for j in 0..<grid[i].count where j % 2 == 1 {
                    list.append(grid[i][j])
                }
                result.append(contentsOf: list.reversed())
            }
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/lexicographically-minimum-string-after-removing-stars/
class Leet3170 {
    func clearStars(_ s: String) -> String {
        var s: [Character?] = Array(s), heap = Heap<Letter>()
        for (i, c) in s.enumerated() {
            guard let c else { continue }
            if c == "*" {
                s[i] = nil
                if let min = heap.popMin() {
                    s[abs(min.i)] = nil
                }
            } else {
                heap.insert(.init(c: c, i: -i))
            }
        }
        return String(s.compactMap { $0 })
    }
    struct Letter: Comparable {
        let c: Character
        let i: Int
        static func < (lhs: Letter, rhs: Letter) -> Bool {
            return lhs.c < rhs.c || (lhs.c == rhs.c && lhs.i < rhs.i)
        }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/longest-zigzag-path-in-a-binary-tree/
class Leet1372 {
    struct Item {
        let n: TreeNode
        let depth: Int
        let isParentLeft: Bool
    }
    func longestZigZag(_ root: TreeNode?) -> Int {
        // dfs using stack to traverse the tree
        var stack: [Item] = []
        guard let root = root else { return 0 }
        stack.append(.init(n: root, depth: 0, isParentLeft: false))
        var result = 0
        while !stack.isEmpty {
            let item = stack.removeLast()
            result = max(result, item.depth)
            if let left = item.n.left {
                let depth = item.isParentLeft ? 1 : item.depth + 1
                stack.append(.init(n: left, depth: depth, isParentLeft: true))
            }
            if let right = item.n.right {
                let depth = item.isParentLeft ? item.depth + 1 : 1
                stack.append(.init(n: right, depth: depth, isParentLeft: false))
            }
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/copy-list-with-random-pointer/
class Leet0138 {
    func copyRandomList(_ head: Node?) -> Node? {
        var curr: Node? = head, prev: Node?, result: Node?, resultIds = [ObjectIdentifier: Node](), idMap = [ObjectIdentifier: ObjectIdentifier]()
        while let n = curr {
            let newNode = Node(n.val)
            idMap[ObjectIdentifier(n)] = ObjectIdentifier(newNode)
            resultIds[ObjectIdentifier(newNode)] = newNode
            if result == nil {
                result = newNode
            }
            prev?.next = newNode
            prev = newNode
            curr = n.next
        }
        curr = head
        var curr2 = result
        while let n = curr, let m = curr2 {
            if let r = n.random, let id = idMap[ObjectIdentifier(r)], let randomNode = resultIds[id] {
                m.random = randomNode
            }
            curr = curr?.next
            curr2 = curr2?.next
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/lexicographical-numbers/
class Leet0386 {
    func lexicalOrder(_ n: Int) -> [Int] {
        var result = [Int](), num = 1
        while result.count < n {
            if num <= n {
                result.append(num)
                num *= 10
            } else {
                num /= 10
                while num % 10 == 9 {
                    num /= 10
                }
                num += 1
            }
        }
        return result
    }
}

class Leet0386_StackSolution {
    func lexicalOrder(_ n: Int) -> [Int] {
        var result = [Int](), s = [Int](stride(from: min(9, n), through: 1, by: -1))
        while let num = s.popLast() {
            guard num <= n else { break }
            result.append(num)
            for i in stride(from: 9, through: 0, by: -1) {
                let newNum = num * 10 + i
                guard newNum <= n else { continue }
                s.append(newNum)
            }
        }
        return result
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/clone-n-ary-tree/
class Leet1490 {
    public class Node {
        public var val: Int
        public var children: [Node]
        public init(_ val: Int) {
            self.val = val
            self.children = []
        }
    }
    func cloneTree(_ root: Node?) -> Node? {
        guard let root else { return nil }
        var map = [ObjectIdentifier: ObjectIdentifier](), newMap = [ObjectIdentifier: Node]()
        var deq = Deque<Node>([root]), result: Node?
        while let oldNode = deq.popFirst() {
            let newNode = Node(oldNode.val)
            map[ObjectIdentifier(oldNode)] = ObjectIdentifier(newNode)
            newMap[ObjectIdentifier(newNode)] = newNode
            if result == nil {
                result = newNode
            }
            for c in oldNode.children {
                deq.append(c)
            }
        }
        deq = [root]
        while let oldNode = deq.popFirst(), let newId = map[ObjectIdentifier(oldNode)], let newNode = newMap[newId] {
            newNode.children = oldNode.children.compactMap { newMap[map[ObjectIdentifier($0)]!] }
            for c in oldNode.children {
                deq.append(c)
            }
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/k-th-smallest-in-lexicographical-order/
class Leet0440 {
    func findKthNumber(_ n: Int, _ k: Int) -> Int {
        var curr = 1, k = k - 1
        while k > 0 {
            let steps = countSteps(n, curr, curr + 1)
            if steps <= k {
                curr += 1
                k -= steps
            } else {
                curr *= 10
                k -= 1
            }
        }
        return curr
    }
    private func countSteps(_ n: Int, _ prefix1: Int, _ prefix2: Int) -> Int {
        var result = 0, prefix1 = prefix1, prefix2 = prefix2
        while prefix1 <= n {
            result += min(n + 1, prefix2) - prefix1
            prefix1 *= 10
            prefix2 *= 10
        }
        return result
    }
    
}
 

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-difference-between-even-and-odd-frequency-i/
class Leet3442 {
    func maxDifference(_ s: String) -> Int {
        let map = s.reduce(into: [Character: Int]()) { r, c in r[c, default: 0] += 1 }
        var maxOdd = 1, minEven = s.count
        for v in map.values {
            if v % 2 == 1 {
                maxOdd = max(maxOdd, v)
            } else {
                minEven = min(minEven, v)
            }
        }
        return maxOdd - minEven
    }
    static func test() {
        let sut = Leet3442()
        assert(sut.maxDifference("abbbcccccddddddddeeeeeeeeffffffff") == -3)
        assert(sut.maxDifference("aaaaabbc") == 3)
    }
}
//Leet3442.test()

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-difference-between-even-and-odd-frequency-ii/
class Leet3445 {
    func maxDifference(_ s: String, _ k: Int) -> Int {
        let n = s.count, values = Array("01234"), s = Array(s)
        var result = Int.min
        for a in values {
            for b in values where a != b {
                var best: [Int] = [.max, .max, .max, .max]
                var countA = 0, countB = 0, prevA = 0, prevB = 0, l = -1
                for r in 0..<n {
                    countA += s[r] == a ? 1 : 0
                    countB += s[r] == b ? 1 : 0
                    while r - l >= k, countB - prevB >= 2 {
                        let lStatus = status(prevA, prevB)
                        best[lStatus] = min(best[lStatus], prevA - prevB)
                        l += 1
                        prevA += s[l] == a ? 1 : 0
                        prevB += s[l] == b ? 1 : 0
                    }
                    let rStatus = status(countA, countB)
                    guard best[rStatus ^ 2 ] != .max else { continue }
                    result = max(result, countA - countB - best[rStatus ^ 2 ])
                }
            }
        }
        return result
    }
    private func status(_ countA: Int, _ countB: Int) -> Int {
        (countA & 1) << 1 | (countB & 1)
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-difference-between-adjacent-elements-in-a-circular-array/
class Leet3423 {
    func maxAdjacentDistance(_ nums: [Int]) -> Int {
        let n = nums.count
        var result = abs(nums[0] - nums[n - 1])
        for i in 1..<n {
            result = max(result, abs(nums[i] - nums[i - 1]))
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/clone-binary-tree-with-random-pointer/
class Leet1485 {
    public class Node {
        public var val: Int
        public var left: Node?
        public var right: Node?
        public var random: Node?
        public init() { self.val = 0; self.left = nil; self.right = nil; self.random = nil; }
        public init(_ val: Int) {
            self.val = val
            self.left = nil
            self.right = nil
            self.random = nil
        }
    }
    typealias NodeCopy = Node
    
    
    func copyRandomBinaryTree(_ root: Node?) -> NodeCopy? {
        guard let root = root else { return nil }
        var map = [ObjectIdentifier: ObjectIdentifier](), copyMap = [ObjectIdentifier: NodeCopy]()
        var queue = Deque<Node>([root]), result: NodeCopy?
        while let oldNode = queue.popFirst() {
            let copyNode = NodeCopy(oldNode.val)
            map[ObjectIdentifier(oldNode)] = ObjectIdentifier(copyNode)
            copyMap[ObjectIdentifier(copyNode)] = copyNode
            if result == nil {
                result = copyNode
            }
            if let left = oldNode.left {
                queue.append(left)
            }
            if let right = oldNode.right {
                queue.append(right)
            }
        }
        queue = [root]
        while let oldNode = queue.popFirst(), let copyId = map[ObjectIdentifier(oldNode)], let copyNode = copyMap[copyId] {
            if let left = oldNode.left, let id = map[ObjectIdentifier(left)] {
                copyNode.left = copyMap[id]
                queue.append(left)
            }
            if let right = oldNode.right, let id = map[ObjectIdentifier(right)] {
                copyNode.right = copyMap[id]
                queue.append(right)
            }
            if let random = oldNode.random, let id = map[ObjectIdentifier(random)] {
                copyNode.random = copyMap[id]
            }
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimize-the-maximum-difference-of-pairs/
class Leet2616 {
    func minimizeMax(_ nums: [Int], _ p: Int) -> Int {
        let nums = nums.sorted(), n = nums.count
        var l = 0, r = nums[n - 1] - nums[0]
        while l < r {
            let mid = l + (r - l) / 2
            if countValidPairs(nums, mid) >= p {
                r = mid
            } else {
                l = mid + 1
            }
        }
        return l
    }
    private func countValidPairs(_ nums: [Int], _ threshold: Int) -> Int {
        var result = 0, i = 0
        while i < nums.count - 1 {
            if nums[i + 1] - nums[i] <= threshold {
                i += 1
                result += 1
            }
            i += 1
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/verify-preorder-sequence-in-binary-search-tree/
class Leet0255 {
    func verifyPreorder(_ preorder: [Int]) -> Bool {
        var minLimit = Int.min, stack = [Int]()
        for n in preorder {
            while let last = stack.last, last < n {
                minLimit = stack.removeLast()
            }
            if n <= minLimit {
                return false
            }
            stack.append(n)
        }
        return true
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-difference-by-remapping-a-digit/
class Leet2566 {
    func minMaxDifference(_ num: Int) -> Int {
        var digits = [Int](), mn = 0, mx = 0, num = num, og = num
        while num > 0 {
            let d = num % 10
            digits.append(d)
            num /= 10
        }
        let mapped = digits.enumerated().map { (i: $0.offset, v: $0.element) }.reversed()
        guard let first = mapped.first?.v, let notNine = mapped.first(where: { $0.v != 9 })?.v else { return og }
        for m in mapped {
            mx += Int(pow(10.0, Double(m.i))) * ((m.v == notNine) ? 9 : m.v)
            mn += Int(pow(10.0, Double(m.i))) * ((m.v == first) ? 0 : m.v)
        }
        return mx - mn
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/max-difference-you-can-get-from-changing-an-integer/
class Leet1432 {
    func maxDiff(_ num: Int) -> Int {
        var digits = [Int](), mn = 0, mx = 0, num = num
        while num > 0 {
            let d = num % 10
            digits.append(d)
            num /= 10
        }
        let mapped = digits.enumerated().map { (i: $0.offset, v: $0.element) }.reversed()
        let notNine = mapped.first(where: { $0.v != 9 })?.v, n = mapped.count
        let gtOne = mapped.first(where: { $0.v > 1 })
        for m in mapped {
            var dx = m.v, dn = m.v
            if let notNine, m.v == notNine {
                dx = 9
            }
            mx += Int(pow(10.0, Double(m.i))) * dx
            if let gtOne, gtOne.v == m.v {
                dn = (gtOne.i == n - 1 ) ? 1 : 0
            }
            mn += Int(pow(10.0, Double(m.i))) * dn
        }
        return mx - mn
    }
}


/*
 123
 923-103=820
 
 999
 999-111=888
 
 111
 999-111=888
 
 555
 999-111=888
 */


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/path-sum/
class Leet0112 {
    func hasPathSum(_ root: TreeNode?, _ targetSum: Int) -> Bool {
        typealias State = (node: TreeNode?, sum: Int)
        guard let root else { return false }
        var stack = [State]([State(node: root, sum: targetSum - root.val)])
        while let top = stack.popLast() {
            var (node, sum) = top
            guard let node else { continue }
            if top.sum == 0, node.left == nil, node.right == nil {
                return true
            }
            if let left = node.left {
                stack.append((node: left, sum: sum - left.val))
            }
            if let right = node.right {
                stack.append((node: right, sum: sum - right.val))
            }
        }
        return false
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/path-sum-ii/
class Leet0113 {
    func pathSum(_ root: TreeNode?, _ targetSum: Int) -> [[Int]] {
        typealias State = (node: TreeNode?, path: [Int], sum: Int)
        guard let root else { return [] }
        var stack = [State(node: root, path: [root.val], sum: root.val)], result = [[Int]]()
        while let top = stack.popLast() {
            var (node, path, sum) = top
            guard let node else { continue }
            if top.sum == targetSum, node.left == nil, node.right == nil {
                result.append(path)
            }
            if let left = node.left {
                stack.append((node: left, path: path + [left.val], sum: sum + left.val))
            }
            if let right = node.right {
                stack.append((node: right, path: path + [right.val], sum: sum + right.val))
            }
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-good-nodes-in-binary-tree/
class Leet1448 {
    func goodNodes(_ root: TreeNode?) -> Int {
        typealias State = (node: TreeNode?, maxSoFar: Int)
        guard let root else { return 0 }
        var stack = [State(node: root, maxSoFar: root.val)], result = 0
        while let top = stack.popLast() {
            var (node, maxSoFar) = top
            guard let node else { continue }
            if node.val >= maxSoFar {
                result += 1
            }
            if let right = node.right {
                stack.append((node: right, maxSoFar: max(maxSoFar, right.val)))
            }
            if let left = node.left {
                stack.append((node: left, maxSoFar: max(maxSoFar, left.val)))
            }
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-difference-between-increasing-elements/
class Leet2016 {
    func maximumDifference(_ nums: [Int]) -> Int {
        var minValue = nums[0], result = -1
        for i in 1..<nums.count {
            let diff = nums[i] - minValue
            if diff > 0 {
                result = max(result, diff)
            } else {
                minValue = nums[i]
            }
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/two-furthest-houses-with-different-colors/description/
class Leet2078 {
    func maxDistance(_ colors: [Int]) -> Int {
        var l = 0, r = colors.count - 1
        while let first = colors.first, first == colors[r] {
            r -= 1
        }
        while let last = colors.last, last == colors[l] {
            l += 1
        }
        return max(r, colors.count - 1 - l)
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-depth-of-binary-tree/
class Leet0111 {
    func minDepth(_ root: TreeNode?) -> Int {
        guard let root else { return 0 }
        var deq = Deque<TreeNode>([root]), depth = 1
        while !deq.isEmpty {
            for _ in deq {
                guard let node = deq.popFirst() else { continue }
                if node.left == nil, node.right == nil {
                    return depth
                }
                if let left = node.left {
                    deq.append(left)
                }
                if let right = node.right {
                    deq.append(right)
                }
            }
            depth += 1
        }
        return -1
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-depth-of-n-ary-tree/
class Leet0559 {
    public class Node {
        public var val: Int
        public var children: [Node]
        public init(_ val: Int) {
            self.val = val
            self.children = []
        }
    }
    func maxDepth(_ root: Node?) -> Int {
        guard let root else { return 0 }
        return 1 + root.children.reduce(0) { max($0, maxDepth($1)) }
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/balanced-binary-tree/
class Leet0110 {
    func isBalanced(_ root: TreeNode?) -> Bool {
        guard let root else { return true }
        return isBalanced(root.left) && isBalanced(root.right) && abs(height(root.left) - height(root.right)) <= 1
    }
    private func height(_ root: TreeNode?) -> Int {
        guard let root else { return 0 }
        return max(height(root.left), height(root.right)) + 1
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/diameter-of-binary-tree/
class Leet0543 {
    func diameterOfBinaryTree(_ root: TreeNode?) -> Int {
        var diameter = 0
        height(root)
        return diameter
        
        @discardableResult func height(_ node: TreeNode?) -> Int {
            guard let node else { return -1 }
            let left = height(node.left)
            let right = height(node.right)
            diameter = max(diameter, left + right + 2)
            return max(left, right) + 1
        }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/diameter-of-n-ary-tree/
class Leet1522 {
    public class Node {
        public var val: Int
        public var children: [Node]
        public init(_ val: Int) {
            self.val = val
            self.children = []
        }
    }
    
    func diameter(_ root: Node?) -> Int {
        var diameter = 0
        height(root)
        return diameter
        @discardableResult func height(_ node: Node?) -> Int {
            guard let node, node.children.count > 0 else { return 0 }
            var max1 = 0, max2 = 0
            for c in node.children {
                let h = height(c) + 1
                if h > max1 {
                    max2 = max1
                    max1 = h
                } else if h > max2 {
                    max2 = h
                }
                let distance = max1 + max2
                diameter = max(diameter, distance)
            }
            return max1
        }
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/binary-tree-paths/
class Leet0257 {
    func binaryTreePaths(_ root: TreeNode?) -> [String] {
        typealias State = (node: TreeNode?, path: [Int])
        guard let root else { return [] }
        var stack = [State(node: root, path: [root.val])], result: [String] = []
        while let top = stack.popLast() {
            var (node, path) = top
            guard let node else { continue }
            if node.left == nil, node.right == nil {
                result.append(path.map(String.init).joined(separator: "->"))
            } else {
                if let right = node.right {
                    stack.append((node: right, path: path + [right.val]))
                }
                if let left = node.left {
                    stack.append((node: left, path: path + [left.val]))
                }
            }
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sum-of-root-to-leaf-binary-numbers/
class Leet1022 {
    func sumRootToLeaf(_ root: TreeNode?) -> Int {
        typealias State = (node: TreeNode?, path: [Int])
        guard let root else { return 0 }
        var stack = [State(node: root, path: [root.val])], result = 0
        while let (node, path) = stack.popLast() {
            if node?.left == nil, node?.right == nil {
                result += num(path)
            } else {
                if let right = node?.right {
                    stack.append((node: right, path: path + [right.val]))
                }
                if let left = node?.left {
                    stack.append((node: left, path: path + [left.val]))
                }
            }
        }
        return result
    }
    private func num(_ path: [Int]) -> Int {
        var result = 0
        for bit in path.reversed().enumerated() {
            result += bit.element * Int(pow(2.0, Double(bit.offset)))
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/smallest-string-starting-from-leaf/
class Leet0988 {
    func smallestFromLeaf(_ root: TreeNode?) -> String {
        typealias State = (node: TreeNode?, path: [Int])
        guard let root, let aAscii = Character("a").asciiValue else { return "" }
        var stack = [State(node: root, path: [root.val])], result = "{"
        while let (node, path) = stack.popLast() {
            if node?.left == nil, node?.right == nil {
                let stringifyPath = String(path.reversed().compactMap { Character(UnicodeScalar( UInt8($0) + aAscii)) })
                result = min(result, stringifyPath)
            } else {
                if let right = node?.right {
                    stack.append((node: right, path: path + [right.val]))
                }
                if let left = node?.left {
                    stack.append((node: left, path: path + [left.val]))
                }
            }
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sum-of-left-leaves/
class Leet0404 {
    func sumOfLeftLeaves(_ root: TreeNode?) -> Int {
        typealias State = (node: TreeNode?, isLeftChild: Bool)
        guard let root = root else { return 0 }
        var stack = [State(node: root, isLeftChild: false)], result = 0
        while let (node, isLeftChild) = stack.popLast(), let node {
            if node.left == nil, node.right == nil, isLeftChild {
                result += node.val
            } else {
                if let right = node.right {
                    stack.append((node: right, isLeftChild: false))
                }
                if let left = node.left {
                    stack.append((node: left, isLeftChild: true))
                }
            }
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sum-root-to-leaf-numbers/
class Leet0129 {
    func sumNumbers(_ root: TreeNode?) -> Int {
        typealias State = (node: TreeNode?, path: [Int])
        guard let root else { return 0 }
        var stack = [State(node: root, path: [root.val])], result = 0
        while let (node, path) = stack.popLast(), let node {
            if node.left == nil, node.right == nil {
                result += num(path)
            } else {
                if let right = node.right {
                    stack.append((node: right, path: path + [right.val]))
                }
                if let left = node.left {
                    stack.append((node: left, path: path + [left.val]))
                }
            }
        }
        return result
    }
    private func num(_ path: [Int]) -> Int {
        path.reversed().enumerated().reduce(0) { r, e in r + e.element * Int(pow(10, Double(e.offset))) }
    }
}



///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/count-the-number-of-arrays-with-k-matching-adjacent-elements/
class Leet3405 {
        
    func countGoodArrays(_ n: Int, _ m: Int, _ k: Int) -> Int {
        combination(n - 1, k) * m % mod * power(m - 1, n - k - 1) % mod
    }
    private let mod = 1_000_000_007
    private let mx = 100_000
    private var fact: [Int]
    private var inv: [Int]
    init() {
        fact = Array(repeating: 0, count: mx)
        inv = Array(repeating: 0, count: mx)
        fact[0] = 1
        for i in 1..<mx {
            fact[i] = (fact[i - 1] * i) % mod
        }
        inv[mx - 1] = power(fact[mx - 1], mod - 2)
        for i in stride(from: mx - 1, to: 0, by: -1) {
            inv[i - 1] = (inv[i] * i) % mod
        }
    }
    private func combination(_ n: Int, _ r: Int) -> Int {
        return (fact[n] * inv[r]) % mod * inv[n - r] % mod
    }
    private func power(_ base: Int, _ exponent: Int) -> Int {
        var result = 1, base = base, exponent = exponent
        while exponent > 0 {
            if exponent % 2 == 1 {
                result = (result * base) % mod
            }
            base = (base * base) % mod
            exponent /= 2
        }
        return result
    }
    
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/divide-array-into-arrays-with-max-difference/
class Leet2966 {
    func divideArray(_ nums: [Int], _ k: Int) -> [[Int]] {
        let nums = nums.sorted(), n = nums.count
        var result = [[Int]]()
        for i in stride(from: 0, to: n, by: 3) {
            let temp = nums[i..<i+3]
            guard let first = temp.first, let last = temp.last, last - first <= k else { return [] }
            result.append(Array(temp))
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/partition-array-such-that-maximum-difference-is-k/
class Leet2294 {
    func partitionArray(_ nums: [Int], _ k: Int) -> Int {
        let nums = nums.sorted(), n = nums.count
        var l = 0, result = 1
        for r in 0..<n where  nums[r] - nums[l] > k  {
            l = r
            result += 1
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/maximum-manhattan-distance-after-k-changes/
class Leet3443 {
    func maxDistance(_ s: String, _ k: Int) -> Int {
        let n = s.count, s = Array(s)
        var lat = 0, lon = 0, result = 0
        for i in 0 ..< n {
            switch s[i] {
            case "N": lat += 1; break
            case "S": lat -= 1; break
            case "E": lon += 1; break
            case "W": lon -= 1; break
            default: break
            }
            result = max(result, min(abs(lat) + abs(lon) + k * 2, i + 1))
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/minimum-deletions-to-make-string-k-special/
class Leet3085 {
    func minimumDeletions(_ word: String, _ k: Int) -> Int {
        var count = word.reduce(into: [Character: Int]()) { r, c in r[c, default: 0] += 1 }, result = word.count
        for a in count.values {
            var deleted = 0
            for b in count.values {
                if a > b {
                    deleted += b
                } else if b > a + k {
                    deleted += b - (a + k)
                }
            }
            result = min(result, deleted)
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/divide-a-string-into-groups-of-size-k/
class Leet2138 {
    func divideString(_ s: String, _ k: Int, _ fill: Character) -> [String] {
        let s = Array(s)
        var result = [String]()
        for i in stride(from: 0, to: s.count, by: k) {
            let endIndex = min(i + k, s.count)
            let chunk = s[i..<endIndex]
            let filledChunk = Array(chunk) + Array(repeating: fill, count: k - chunk.count)
            result.append(String(filledChunk))
        }
        return result
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/sum-of-k-mirror-numbers/
class Leet2081 {
    
    var digit = [Int](repeating: 0, count: 100)
    func kMirror(_ k: Int, _ n: Int) -> Int {
        var l = 1, count = 0, result = 0
        while count < n {
            let r = l * 10
            for op in 0..<2 {
                var i = l
                while i < r && count < n {
                    var combined = i, x = (op == 0 ? i / 10 : i)
                    while x > 0 {
                        combined = combined * 10 + (x % 10)
                        x /= 10
                    }
                    if isPalindrome(combined, k) {
                        count += 1
                        result += combined
                    }
                    i += 1
                }
            }
            l = r
        }
        return result
    }
    private func isPalindrome(_ num: Int, _ base: Int) -> Bool {
        var l = -1, num = num
        while num > 0 {
            l += 1
            digit[l] = num % base
            num /= base
        }
        var i = 0, j = l
        while i < j {
            if digit[i] != digit[j] {
                return false
            }
            i += 1
            j -= 1
        }
        return true
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/kth-smallest-product-of-two-sorted-arrays/
class Leet2040 {
    func kthSmallestProduct(_ nums1: [Int], _ nums2: [Int], _ k: Int) -> Int {
        let n1 = nums1.count
        var low = -10_000_000_000, high = 10_000_000_000
        while low <= high {
            let mid = low + (high - low) / 2
            var count = 0
            for i in 0..<n1 {
                count += f(nums2, nums1[i], mid)
            }
            if count < k {
                low = mid + 1
            } else {
                high = mid - 1
            }
        }
        return low
    }
    
    private func f(_ nums2: [Int], _ x1: Int, _ v: Int) -> Int {
        var low = 0
        var high = nums2.count - 1
        while low <= high {
            let mid = low + (high - low) / 2, product = nums2[mid] * x1
            if x1 >= 0 && product <= v || x1 < 0 && product > v {
                low = mid + 1
            } else {
                high = mid - 1
            }
        }
        if x1 >= 0 {
            return low
        } else {
            return nums2.count - low
        }
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/longest-binary-subsequence-less-than-or-equal-to-k/
class Leet2311 {
    func longestSubsequence(_ s: String, _ k: Int) -> Int {
        let s = Array(s.reversed()), bits = String(k, radix: 2).count
        var num = 0, result = 0
        for i in 0..<s.count {
            if s[i] == "1" {
                let temp = 1 << i
                if i < bits && num + temp <= k {
                    num += temp
                    result += 1
                }
            } else {
                result += 1
            }
        }
        return result
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/longest-subsequence-repeated-k-times/
class Leet2014 {
    func longestSubsequenceRepeatedK(_ s: String, _ k: Int) -> String {
        let s = Array(s)
        var freq = s.reduce(into: [Character: Int]()) { r, ch in r[ch, default: 0] += 1 }
        let candidates: [Character] = freq.filter { $1 >= k }.keys.sorted().reversed()
        var q = Deque<String>(candidates.map { String($0)} ), result = ""
        while let curr = q.popFirst() {
            if curr.count > result.count {
                result = curr
            }
            for ch in candidates {
                let next = curr + String(ch)
                guard isKRepeatedSubsequence(s, next, k) else { continue }
                q.append(next)
            }
        }
        return result
    }
    private func isKRepeatedSubsequence(_ s: [Character], _ t: String, _ k: Int) -> Bool {
        let t = Array(t), m = t.count
        var pos = 0, matched = 0
        for ch in s where ch == t[pos] {
            pos += 1
            guard pos == m else { continue }
            pos = 0
            matched += 1
            guard matched == k else { continue }
            return true
        }
        return false
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-subsequence-of-length-k-with-the-largest-sum/
class Leet2099 {
    func maxSubsequence(_ nums: [Int], _ k: Int) -> [Int] {
        nums
            .enumerated()
            .sorted { $0.element < $1.element }
            .suffix(k)
            .sorted { $0.offset < $1.offset }
            .map(\.element)
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/number-of-subsequences-that-satisfy-the-given-sum-condition/
class Leet1498 {
    func numSubseq(_ nums: [Int], _ target: Int) -> Int {
        let n = nums.count, mod = 1_000_000_007, nums = nums.sorted()
        var power = [Int](repeating: 0, count: n), result = 0
        power[0] = 1
        for i in 1..<n {
            power[i] = (power[i-1] * 2) % mod
        }
        for l in 0..<n {
            let r = binarySearchRightmostIndex(nums, target - nums[l]) - 1
            guard r >= l else { continue }
            result += power[r - l]
            result %= mod
        }
        return result
    }
    private func binarySearchRightmostIndex(_ nums: [Int], _ target: Int) -> Int {
        var low = 0, high = nums.count - 1
        while low <= high {
            let mid = low + (high - low) / 2
            if nums[mid] <= target {
                low = mid + 1
            } else {
                high = mid - 1
            }
        }
        return low
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-the-original-typed-string-ii/
class Leet3333 {
    func possibleStringCount(_ word: String, _ k: Int) -> Int {
        let mod = 1_000_000_007, n = word.count, word = Array(word)
        var count = 1, freq = [Int](), result = 1
        for i in 1..<n {
            if word[i] == word[i-1] {
                count += 1
            } else {
                freq.append(count)
                count = 1
            }
        }
        freq.append(count)
        
        for o in freq {
            result = (result * o) % mod
        }
        if freq.count >= k {
            return result
        }
        
        var f = Array(repeating: 0, count: k), g = Array(repeating: 1, count: k)
        f[0] = 1
        for i in 0..<freq.count {
            var fNew = Array(repeating: 0, count: k)
            for j in 1..<k {
                fNew[j] = g[j-1]
                guard (j - freq[i] - 1 >= 0) else { continue }
                fNew[j] = (fNew[j] - g[j - freq[i] - 1] + mod) % mod
            }
            var gNew = Array(repeating: 0, count: k)
            gNew[0] = fNew[0]
            for j in 1..<k {
                gNew[j] = (gNew[j - 1] + fNew[j]) % mod
            }
            f = fNew
            g = gNew
        }
        return (result - g[k - 1] + mod) % mod
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-the-k-th-character-in-string-game-i/
class Leet3304 {
    func kthCharacter(_ k: Int) -> Character {
        Character(UnicodeScalar(Int(Character("a").asciiValue!) + (k - 1).nonzeroBitCount)!)
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/shifting-letters/
class Leet0848 {
    func shiftingLetters(_ s: String, _ shifts: [Int]) -> String {
        let aAscii = Character("a").asciiValue!
        var result = Array(s), shifts = shifts
        for i in stride(from: s.count - 1, through: 0, by: -1) {
            shifts[i] = shifts[i] + (i == shifts.count - 1 ? 0 : shifts[i + 1])
            result[i] = Character(UnicodeScalar((Int(result[i].asciiValue! - aAscii) + shifts[i]) % 26 + Int(aAscii))!)
        }
        return String(result)
    }
}


///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/shifting-letters-ii/
class Leet2381 {
    func shiftingLetters(_ s: String, _ shifts: [[Int]]) -> String {
        let aAscii = Character("a").asciiValue!, n = s.count
        // build difference array
        var diffArray = [Int](repeating: 0, count: n), numbefOfShifts = 0, result = Array(s)
        for shift in shifts {
            if shift[2] == 1 {
                diffArray[shift[0]] += 1
                if shift[1] + 1 < n {
                    diffArray[shift[1] + 1] -= 1
                }
            } else {
                diffArray[shift[0]] -= 1
                if shift[1] + 1 < n {
                    diffArray[shift[1] + 1] += 1
                }
            }
        }
        // shift
        for i in 0..<s.count {
            numbefOfShifts = (numbefOfShifts + diffArray[i]) % 26
            if numbefOfShifts < 0 {
                numbefOfShifts += 26
            }
            result[i] = Character(UnicodeScalar((Int(result[i].asciiValue! - aAscii) + numbefOfShifts) % 26 + Int(aAscii))!)
        }
        return String(result)
    }
}

///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/range-addition/
class Leet0370 {
    func getModifiedArray(_ length: Int, _ updates: [[Int]]) -> [Int] {
        var result = [Int](repeating: 0, count: length), diffArray = [Int](repeating: 0, count: length)
        for op in updates {
            diffArray[op[0]] += op[2]
            if op[1] + 1 < length {
                diffArray[op[1] + 1] -= op[2]
            }
        }
        var sum = 0
        for (i, v) in diffArray.enumerated() {
            sum += v
            result[i] = sum
        }
        return result
    }
}


print("All playground tests passed!")
