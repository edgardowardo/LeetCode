//
//  Package.swift
//  
//
//  Created by EDGARDO AGNO on 3/13/25.
//

// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "PlaygroundPackage",
    dependencies: [
        .package(url: "https://github.com/apple/swift-collections.git", from: "1.0.0")
    ],
    targets: [
        .target(
            name: "PlaygroundPackage",
            dependencies: [
                .product(name: "DequeModule", package: "swift-collections")
            ]
        )
    ]
)
#if os(Linux)
import Glibc
#else
import Darwin
#endif
