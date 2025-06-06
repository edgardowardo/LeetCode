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


 














@main
class AppDelegate: UIResponder, UIApplicationDelegate {
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {


        

        
        print("All tests passed!")
        return true
    }
}







