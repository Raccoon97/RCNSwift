/*

            StringExtension
            
            Raccoon 2023-08-24

            Easier Using String 

*/

import Foundation


// String 확장.
extension String {
    
    subscript(idx: Int) -> String? { // idx 에 해당하는 String 의 원소를 반환.
        guard idx >= 0 && idx < count else { return nil }
        let targetIndex = index(startIndex, offsetBy: idx)
        return String(self[targetIndex])
    }
    
    subscript(range: Range<Int>) -> String? { // idx Range 에 해당하는 String 의 원소를 반환. half-open
        guard range.lowerBound >= 0 && range.upperBound <= count else { return nil }
        let startIndex = index(self.startIndex, offsetBy: range.lowerBound)
        let endIndex = index(self.startIndex, offsetBy: range.upperBound)
        return String(self[startIndex..<endIndex])
    }
    
    subscript(range: CountableClosedRange<Int>) -> String? {  // idx Range 에 해당하는 String 의 원소를 반환. closed
        guard range.lowerBound >= 0 && range.upperBound < count else { return nil }
        let startIndex = index(self.startIndex, offsetBy: range.lowerBound)
        let endIndex = index(self.startIndex, offsetBy: range.upperBound)
        return String(self[startIndex...endIndex])
    }

    public func findFirstidx(with target: String) -> [Int]? {  // String 에서 가장 앞에 위치한 target 의 idx 를 반환
        if let range = self.range(of: target) {
            let startIndex = self.distance(from: self.startIndex, to: range.lowerBound)
            let endIndex = self.distance(from: self.startIndex, to: range.upperBound) - 1

            if startIndex == endIndex {
                return [startIndex]
            } else {
                return [startIndex, endIndex]
            }
        } else {
            return nil
        }
    }

    public func findLastidx(with target: String) -> [Int]? {   // String 에서 가장 뒤에 위치한 target 의 idx 를 반환
        if let range = self.range(of: target, options: .backwards) {
            let startIndex = self.distance(from: self.startIndex, to: range.lowerBound)
            let endIndex = self.distance(from: self.startIndex, to: range.upperBound) - 1

            if startIndex == endIndex {
                return [startIndex]
            } else {
                return [startIndex, endIndex]
            }
        } else {
            return nil
        }
    }

    public func findWholeidx(with target: String, op: Bool = false) -> [[Int]]? {  // String 에서 모든 target 의 idx 를 반환
        var results: [[Int]] = []
        var searchRange: Range<String.Index> = self.startIndex..<self.endIndex
        
        while let range = self.range(of: target, options: op ? [] : [.caseInsensitive], range: searchRange) {
            let startIndex = self.distance(from: self.startIndex, to: range.lowerBound)
            let endIndex = self.distance(from: self.startIndex, to: range.upperBound) - 1
            
            if startIndex == endIndex {
                results.append([startIndex])
            } else {
                results.append([startIndex, endIndex])
            }
            
            searchRange = range.upperBound..<self.endIndex
        }
        
        if results.isEmpty {
            return nil
        }

        return results        
    }

    public func getASCIICode() -> [UInt8]? {    // String 을 ASCII 코드로 변경
        var asciiArray: [UInt8] = []
        for char in self {
            if let scalar = char.unicodeScalars.first {
                if scalar.isASCII {
                    asciiArray.append(UInt8(scalar.value))
                } else {
                    return nil
                }
            }
        }
        return asciiArray
    }
}