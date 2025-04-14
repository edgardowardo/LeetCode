//
//  AppDelegate.swift
//  DebugMe
//
//  Created by EDGARDO AGNO on 17/10/2024.
//

import UIKit
import DequeModule
import HeapModule



///---------------------------------------------------------------------------------------
///

///---------------------------------------------------------------------------------------
///

///---------------------------------------------------------------------------------------
///

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




///---------------------------------------------------------------------------------------
///

///---------------------------------------------------------------------------------------
///

///---------------------------------------------------------------------------------------
///


///---------------------------------------------------------------------------------------
///

///---------------------------------------------------------------------------------------
///

///---------------------------------------------------------------------------------------
///









///---------------------------------------------------------------------------------------
///https://leetcode.com/problems/find-longest-special-substring-that-occurs-thrice-i/
class Leet2981 {
    func maximumLength(_ s: String) -> Int {
        let s = Array(s)
        var maxLength = 0, lastSeenMap = [Character: Int](), start = 0, stringFreq = [String: Int]()
        for (i, c) in s.enumerated() {
            if let lastSeenIndex = lastSeenMap[c] {
                for j in start...lastSeenIndex {
                    lastSeenMap[s[j]] = nil
                }
                start = lastSeenIndex + 1
            }
            lastSeenMap[c] = i
            let substr = String(s[start...i])
            stringFreq[substr, default: 0] += 1

            if stringFreq[substr]! >= 3 {
                maxLength = max(maxLength, substr.count)
            }
        }
        return maxLength
    }
    
    static func test() {
        let sut = Leet2981()
        assert(sut.maximumLength("abcccccddddabcccccddddabcccccdddd") == 3)
    }
    
}


@main
class AppDelegate: UIResponder, UIApplicationDelegate {
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {



                
        Leet2981.test()
        
        
        
        
        print("All tests passed!")
        return true
    }
}




/*
 
 "acc"
 "aaa"
 "abcccccddddabcccccddddabcccccdddd"
 "jinhhhtttttttefffffjjjjjjjjjfffffjjjjjjjjjzvvvvvvg"
 "aaaaaaaaaaaaccccccccccccccccccccccccccaaaaaaaaaaaa"
 "aaaaaaaaaaaaaaaaaaaabbbbbbbbbbaaaaaaaaaaaaaaaaaaaa"
 "zzzzzzzzzzzzzzzzzfffffdddddddddiiiiiiiiiiiiiiiiiii"
 "zzzzzzzzzzzsssssssssssssssssqppppppppppppppnqmosat"
 
 "abcdabcddddabcddddccccbbbbaaaa"
 "abcccccdddd"
 "aaaaabccddrrruuuuuutttt"
 "aaannnnuuuuuuwwwwwwxxzzzzzzzz"
 "dddddddddddggggggggggggvvvvvvvvzzzzzzz"
 "eccdnmcnkl"
 "cbc"
 "lkwwdddddnnnnnynnnnl"
 
 "fafff"
 "cccerrrecdcdccedecdcccddeeeddcdcddedccdceeedccecde"
 "cceeddedceddccceecddoooocdeecddcdddedcceeeeccedccc"
 "ceeeeeeeeeeeebmmmfffeeeeeeeeeeeewww"
 
 
 
 
 
 */
