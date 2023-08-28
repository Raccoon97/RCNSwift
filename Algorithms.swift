/*

            RaccoonSwift
            
            Raccoon 2023-08-24

            Easier Using Algorithms



문자열 처리
KMP 알고리즘 (KMP Algorithm)
보이어-무어 알고리즘 (Boyer-Moore Algorithm)
레벤슈타인 거리 (Levenshtein Distance)

동적 계획법 (Dynamic Programming)
피보나치 수열 (Fibonacci Sequence)
최장 공통 부분열 (Longest Common Subsequence)
동전 교환 문제 (Coin Change Problem)

기하 알고리즘
볼록 껍질 (Convex Hull)
선분 교차 판별 (Line Segment Intersection)

최적화 알고리즘
그리디 알고리즘 (Greedy Algorithm)
분기 한정법 (Branch and Bound)

데이터 압축
Huffman 코딩 (Huffman Coding)
Run-Length Encoding (RLE)

암호화
RSA 알고리즘
AES 알고리즘


*/

import Foundation

protocol RaccoonSwfitProtocol {
    
    // 정렬 알고리즘( Sorting Algorithms )
        // Bool, Optional, Tuple, Dictionary 등 Comparable 프로토콜을 따르지 않는 데이터 타입 제외하고 모두 가능
    func bubbleSort<T: Comparable>(_ array: [T]) -> [T]
    func selectionSort<T: Comparable>(_ array: [T]) -> [T]
    func insertionSort<T: Comparable>(_ array: [T]) -> [T]
    func quickSort<T: Comparable>(_ array: [T]) -> [T]
    func quickSort2<T: Comparable>(_ array: [T]) -> [T]
    func mergeSort<T: Comparable>(_ array: [T]) -> [T]
    func heapSort<T: Comparable>(_ array: [T]) -> [T]

    // 검색 알고리즘
    func linearSearch<T: Equatable>(_ array: [T], _ target: T) -> Int?
    func binarySearch<T: Comparable>(_ array: [T], _ target: T) -> Int?
    func dfs<T: Hashable>(graph: [T: [T]], start: T) -> [T]
    func bfs<T: Hashable>(graph: [T: [T]], start: T) -> [T]

    //그래프 알고리즘
    func dijkstra<T: Hashable>(graph: [T: [(T, Int)]], start: T) -> [T: Int]
    func floydWarshall<T: Hashable>(graph: [T: [(T, Int)]]) -> [T: [T: Int]]
    func bellmanFord<T: Hashable>(graph: [T: [(T, Int)]], start: T) -> [T: Int]?
    func kruskal<T: Hashable>(edges: [(T, T, Int)]) -> [(T, T, Int)]
    func prim<T: Hashable>(graph: [T: [(T, Int)]], start: T) -> [(T, T, Int)]

    // 문자열 처리
    
    // 동적 계획법 (Dynamic Programming)

    // 기하 알고리즘

    //최적화 알고리즘

    // 수학적 알고리즘
    func isPrime(_ n: Int) -> Bool
    func sieveOfEratosthenes(_ n: Int) -> [Int]
    func gcd_recursive(_ a: Int, _ b: Int) -> Int
    func gcd_iterative(_ a: Int, _ b: Int) -> Int

    // 데이터 압축

    // 암호화

}


struct RaccoonSwift: RaccoonSwfitProtocol {

    //MARK: - 정렬 알고리즘 
    func bubbleSort<T: Comparable>(_ array: [T]) -> [T] {  // 버블정렬
        /*
        이 알고리즘의 시간 복잡도는 최악의 경우와 평균적으로 O(n^2)입니다. 
        따라서 데이터 양이 큰 경우에는 더 효율적인 정렬 알고리즘을 사용하는 것이 좋습니다.
        */

        var arr = array // 배열을 복사해서 원본 배열에는 영향을 미치지 않게 합니다.
        let n = arr.count
        
        for i in 0..<n {
            var swapped = false // 교환 여부를 확인하는 변수
            
            // 마지막 i개의 요소는 이미 정렬되어 있으므로, 0부터 (n-i-1)까지만 순회
            for j in 0..<(n-i-1) {
                if arr[j] > arr[j + 1] { // 인접한 두 요소를 비교
                    // 교환
                    (arr[j], arr[j + 1]) = (arr[j + 1], arr[j])
                    swapped = true // 교환했음을 표시
                }
            }
            
            // 교환이 이루어지지 않았다면, 배열은 이미 정렬되어 있는 것이므로 종료
            if !swapped {
                break
            }
        }
        
        return arr
    }

    func selectionSort<T: Comparable>(_ array: [T]) -> [T] {   // 선택정렬
        /*
        이 알고리즘의 시간 복잡도는 최악의 경우와 평균적으로 O(n^2)입니다. 
        따라서 데이터 양이 큰 경우에는 더 효율적인 정렬 알고리즘을 사용하는 것이 좋습니다.
        */

        var arr = array // 배열을 복사해서 원본 배열에는 영향을 미치지 않게 합니다.
        let n = arr.count
        
        for i in 0..<(n - 1) {
            var minIndex = i // 가장 작은 원소의 인덱스를 저장하는 변수
            for j in (i + 1)..<n {
                if arr[j] < arr[minIndex] {
                    minIndex = j // 더 작은 원소를 찾았다면 인덱스를 업데이트합니다.
                }
            }
            
            // i번째와 가장 작은 원소를 교환
            if i != minIndex {
                (arr[i], arr[minIndex]) = (arr[minIndex], arr[i])
            }
        }
        
        return arr
    }

    func insertionSort<T: Comparable>(_ array: [T]) -> [T] {   // 삽입정렬
        /*
        삽입 정렬의 시간 복잡도는 최악의 경우와 평균적으로 O(n^2)입니다. 
        하지만 이미 부분적으로 정렬되어 있는 배열에 대해서는 매우 효율적으로 동작하며, 
        작은 데이터셋에 대해서는 다른 O(nlogn) 알고리즘보다 더 빠를 수 있습니다.
        */

        var arr = array  // 배열을 복사해서 원본 배열에는 영향을 미치지 않게 합니다.
        let n = arr.count
        
        for i in 1..<n {
            var j = i
            let target = arr[i]  // 현재 원소를 target에 저장합니다.
            
            // target 보다 큰 원소들을 뒤로 한 칸씩 이동합니다.
            while j > 0 && arr[j - 1] > target {
                arr[j] = arr[j - 1]
                j -= 1
            }
            
            // target을 적절한 위치에 삽입합니다.
            arr[j] = target
        }
        
        return arr
    }

    func quickSort<T: Comparable>(_ array: [T]) -> [T] {   // 퀵정렬, 고차함수
        /*
        퀵 정렬의 시간 복잡도는 평균적으로 O(nlogn), 최악의 경우에는 O(n^2)입니다. 
        하지만 실제로는 다른 O(nlogn) 알고리즘보다 일반적으로 더 빠르게 동작하며, 
        인플레이스(in-place) 정렬이 가능해서 추가 메모리를 거의 사용하지 않습니다.
        */

        if array.count <= 1 {  // 배열의 크기가 1 이하이면 이미 정렬된 것으로 간주
            return array
        }
        
        let pivot = array[array.count / 2]  // 중간값을 피벗으로 선택
        let less = array.filter { $0 < pivot }  // 피벗보다 작은 원소들
        let equal = array.filter { $0 == pivot }  // 피벗과 같은 원소들
        let greater = array.filter { $0 > pivot }  // 피벗보다 큰 원소들
        
        // 각 부분을 재귀적으로 정렬하고 결과를 합침
        return quickSort(less) + equal + quickSort(greater)
    }

    func quickSort2<T: Comparable>(_ array: [T]) -> [T] {  // 퀵정렬, 구현
        /*
        quickSort2 함수는 배열 arr과 배열의 범위를 나타내는 low와 high 인덱스를 매개변수로 받습니다. 
        partition 함수는 배열을 피벗을 기준으로 두 부분으로 분할하고, 피벗의 새로운 인덱스를 반환합니다.
        */
        var arr = array  // 원본 배열에 영향을 주지 않도록 복사합니다.

        func quickSort2_Inner(_ arr: inout [T], low: Int, high: Int) {
            if low < high {
                // 피벗을 기준으로 배열을 분할
                let pivotIndex = partition(&arr, low: low, high: high)
                
                // 분할된 두 부분을 각각 정렬
                quickSort2_Inner(&arr, low: low, high: pivotIndex - 1)
                quickSort2_Inner(&arr, low: pivotIndex + 1, high: high)
            }
        }

        func partition(_ arr: inout [T], low: Int, high: Int) -> Int {
            let pivot = arr[high]  // 마지막 원소를 피벗으로 선택
            var i = low - 1  // 피벗보다 작은 원소의 인덱스
            
            for j in low..<high {
                if arr[j] <= pivot {
                    i += 1
                    // arr[i]와 arr[j]를 교환
                    (arr[i], arr[j]) = (arr[j], arr[i])
                }
            }
            
            // arr[i + 1]와 arr[high] (피벗)를 교환
            (arr[i + 1], arr[high]) = (arr[high], arr[i + 1])
            return i + 1
        }

        quickSort2_Inner(&arr, low: 0, high: arr.count - 1)
        return arr
    }

    func mergeSort<T: Comparable>(_ array: [T]) -> [T] {   // 병합정렬
        /*
        병합 정렬의 시간 복잡도는 O(nlogn)이며, 공간 복잡도도 O(n)입니다. 
        병합 정렬은 안정적인 정렬 알고리즘이기도 합니다.
        */


        func merge(left: [T], right: [T]) -> [T] {
            var leftIndex = 0
            var rightIndex = 0
            var sortedArray: [T] = []

            while leftIndex < left.count && rightIndex < right.count {
                if left[leftIndex] < right[rightIndex] {
                    sortedArray.append(left[leftIndex])
                    leftIndex += 1
                } else {
                    sortedArray.append(right[rightIndex])
                    rightIndex += 1
                }
            }

            while leftIndex < left.count {
                sortedArray.append(left[leftIndex])
                leftIndex += 1
            }

            while rightIndex < right.count {
                sortedArray.append(right[rightIndex])
                rightIndex += 1
            }

            return sortedArray
        }

        guard array.count > 1 else { return array }

        let middleIndex = array.count / 2
        let leftArray = Array(array[0..<middleIndex])
        let rightArray = Array(array[middleIndex...])

        return merge(left: mergeSort(leftArray), right: mergeSort(rightArray))
    }

    func heapSort<T: Comparable>(_ array: [T]) -> [T] {     // 힙정렬
        /*
        최대 힙으로 만드는 과정(heapify)의 시간 복잡도는 O(logn) 이다. 
        이 heapify의 과정이 n개의 원소를 다 정렬할 때까지 반복되므로 최종 힙 정렬의 시간 복잡도는 O(nlogn) 이 된다.
        */
        var arr = array
        
        func heapify(n: Int, i: Int) {
            var largest = i
            let left = 2 * i + 1
            let right = 2 * i + 2

            if left < n && arr[left] > arr[largest] {
                largest = left
            }

            if right < n && arr[right] > arr[largest] {
                largest = right
            }

            if largest != i {
                arr.swapAt(i, largest)
                heapify(n: n, i: largest)
            }
        }
        
        let n = arr.count

        for i in stride(from: n / 2 - 1, through: 0, by: -1) {
            heapify(n: n, i: i)
        }

        for i in stride(from: n - 1, through: 1, by: -1) {
            arr.swapAt(0, i)
            heapify(n: i, i: 0)
        }
        
        return arr
    }



    //MARK: - 검색 알고리즘
    func linearSearch<T: Equatable>(_ array: [T], _ target: T) -> Int? {    // 선형 검색
        /*
        찾고자 하는 대상(target)이 배열 안에 있으면 해당 인덱스를 반환하고, 없으면 nil을 반환합니다.
        선형 검색의 시간 복잡도는 최악의 경우 O(n)입니다.
        */

        for (index, item) in array.enumerated() {
            if item == target {
                return index
            }
        }
        return nil
    }

    func binarySearch<T: Comparable>(_ array: [T], _ target: T) -> Int? {     // 이진 검색
        /*
        정렬된 배열에서 특정 값을 효율적으로 찾는 방법 중 하나입니다. 
        이 알고리즘의 시간 복잡도는 O(logn)입니다.
        */
        var low = 0
        var high = array.count - 1
        
        while low <= high {
            let mid = (low + high) / 2
            let midValue = array[mid]
            
            if midValue == target {
                return mid
            } else if midValue < target {
                low = mid + 1
            } else {
                high = mid - 1
            }
        }
        
        return nil  // 타겟 값이 배열에 없다면 nil 반환
    }

    func dfs<T: Hashable>(graph: [T: [T]], start: T) -> [T] {      // 깊이 우선 탐색
        var visited: [T] = []                               // visited 기록 남기기
        var stack: [T] = [start]                            // visited 예정인 친구들
        
        while !stack.isEmpty {                              // visited 예정이 없을 때 까지
            let vertex = stack.popLast()!                   // visited 예정 중 마지막 친구
            
            if !visited.contains(vertex) {                  // visited 했던 친구가 아닐 때
                visited.append(vertex)                      // visited 기록에 추가
                
                if let neighbors = graph[vertex] {          // 해당 친구의 친구들이 있을 때
                    for neighbor in neighbors.reversed() {  // 친구들을 뒤에서부터
                        stack.append(neighbor)              // visited 예정인 친구들에 추가
                    }
                }
            }
        }
        
        return visited                                      // visited 기록 반환

        /*
            DFS 동작

            [
                "A": ["B", "C"],
                "B": ["D", "E"],
                "C": ["F", "G", "H"],
                "D": ["I", "J"],
                "E": ["K"]
            ]

            stack       [A]
            vidisted    []
            vertex      A
            visited     [A]
            stack       []
            neighbors   [B, C]
            stack       [C, B]

            vertex      B
            stack       [C]
            visited     [A, B]
            neighbors   [D, E]
            stack       [C, E, D]

            vertex      D
            stack       [C, E]
            visited     [A, B, D]
            neighbors   [I, J]
            stack       [C, E, J, I]

            vertex      I
            stack       [C, E, J]
            visited     [A, B, D, I]
            neighbors   []
            stack       [C, E, J]

            vertex      J
            stack       [C, E]
            visited     [A, B, D, I, J]
            neighbors   []
            stack       [C, E]

            vertex      E
            stack       [C]
            visited     [A, B, D, I, J, E]
            neighbors   [K]
            stack       [C, K]

            vertex      K
            stack       [C]
            visited     [A, B, D, I, J, E, K]
            neighbors   []
            stack       [C]

            vertex      C
            stack       []
            visited     [A, B, D, I, J, E, K, C]
            neighbors   [F, G, H]
            stack       [H, G, F]

            vertex      F
            stack       [H, G]
            visited     [A, B, D, I, J, E, K, C, F]
            neighbors   []
            stack       [H, G]

            vertex      G
            stack       [H]
            visited     [A, B, D, I, J, E, K, C, F, G]
            neighbors   []
            stack       [H]

            vertex      H
            stack       []
            visited     [A, B, D, I, J, E, K, C, F, G, H]
            neighbors   []
            stack       []

            Fin.

        */
    }

    func bfs<T: Hashable>(graph: [T: [T]], start: T) -> [T] {       // 너비 우선 탐색
        var visited: [T] = []                           // visited 기록 남기기
        var queue: [T] = [start]                        // visited 예정인 친구들

        while !queue.isEmpty {                          // visited 예정이 없을 때 까지
            let vertex = queue.removeFirst()            // visited 예정 중 첫 번째 친구
            
            if !visited.contains(vertex) {              // visited 했던 친구가 아닐 때
                visited.append(vertex)                  // visited 기록에 추가
                
                if let neighbors = graph[vertex] {      // 해당 친구의 친구들이 있을 때
                    for neighbor in neighbors {         // 친구들을 앞에서 부터
                        queue.append(neighbor)          // visited 예정인 친구들에 추가
                    }
                }
            }
        }
        return visited                                  // visited 기록 반환

        /*
            BFS 동작

            [
                "A": ["B", "C"],
                "B": ["D", "E"],
                "C": ["F", "G", "H"],
                "D": ["I", "J"],
                "E": ["K"]
            ]

            stack       [A]
            vidisted    []
            vertex      A
            visited     [A]
            stack       []
            neighbors   [B, C]
            stack       [B, C]

            vertex      B
            stack       [C]
            visited     [A, B]
            neighbors   [D, E]
            stack       [C, D, E]

            vertex      C
            stack       [D, E]
            visited     [A, B, C]
            neighbors   [F, G, H]
            stack       [D, E, F, G, H]

            vertex      D
            stack       [E, F, G, H]
            visited     [A, B, C, D]
            neighbors   [I, J]
            stack       [E, F, G, H, I, J]

            vertex      E
            stack       [F, G, H, I, J]
            visited     [A, B, C, D, E]
            neighbors   [K]
            stack       [F, G, H, I, J, K]

            vertex      F
            stack       [G, H, I, J, K]
            visited     [A, B, C, D, E, F]
            neighbors   []
            stack       [G, H, I, J, K]

            vertex      G
            stack       [H, I, J, K]
            visited     [A, B, C, D, E, F, G]
            neighbors   []
            stack       [H, I, J, K]

            vertex      H
            stack       [I, J, K]
            visited     [A, B, C, D, E, F, G, H]
            neighbors   []
            stack       [I, J, K]

            vertex      I
            stack       [J, K]
            visited     [A, B, C, D, E, F, G, H, I]
            neighbors   []
            stack       [J, K]

            vertex      J
            stack       [K]
            visited     [A, B, C, D, E, F, G, H, I, J]
            neighbors   []
            stack       [K]

            vertex      K
            stack       []
            visited     [A, B, C, D, E, F, G, H, I, J, K]
            neighbors   []
            stack       []

            Fin.

        */
    }



    //MARK: - 그래프 알고리즘
    func dijkstra<T: Hashable>(graph: [T: [(T, Int)]], start: T) -> [T: Int] {      // 다익스트라
        /*
        시작 정점에서 다른 모든 정점까지의 최단 거리를 찾는 알고리즘입니다. 
        이 알고리즘은 음의 가중치를 가진 간선이 없는 그래프에서만 작동합니다. 
        시간 복잡도는 O(V^2) 입니다.    V = 노드(정점)의 수
        */
        var distances: [T: Int] = [:]                                   // 최단거리 정보를 저장할 딕셔너리 초기화
        var priorityQueue: [(T, Int)] = [(start, 0)]                    // 우선순위 큐 초기화, 시작 노드와 거리를 추가
        distances[start] = 0                                            // 시작 노드 까지의 거리를 0으로 설정한다.
        
        while !priorityQueue.isEmpty {                                  // 우선순위 큐 빌때까지 반복
            // 우선순위 큐에서 가장 작은 거리를 가진 노드를 선택
            priorityQueue.sort(by: { $0.1 > $1.1 })                     // 큐를 가장 작은 거리가 마지막 요소가 되도록 정렬한다
            let current = priorityQueue.removeLast()                    // 가장 작은 거리를 가진 노드를 current 에 넣고 제거한다.
            let currentVertex = current.0                               // 선택한 노드와 그 노드까지의 거리를 저장한다.
            let currentDistance = current.1
            
            // 인접 노드들의 거리를 업데이트
            for neighbor in graph[currentVertex] ?? [] {                // 선택한 노드의 이웃 노드들을 순회한다.
                let neighborVertex = neighbor.0                         // 이웃 노드와 가중치를 저장한다.
                let weight = neighbor.1
                
                if distances[neighborVertex] == nil || distances[neighborVertex]! > currentDistance + weight {  // 이웃 노드까지의 거리를 업데이트 해야 하는 경우 체크
                    distances[neighborVertex] = currentDistance + weight                                        // 최단 거리를 업데이트한다.
                    priorityQueue.append((neighborVertex, currentDistance + weight))                            // 업데이트된 정보를 우선순위 큐에 다시 추가한다.
                }
            }
        }
        
        return distances                                                // 모든 노드까지의 최단 거리 정보를 담은 distance 딕셔너리를 반환한다.

        /* // 예제 그래프와 테스트
            let graph: [String: [(String, Int)]] = [
                "A": [("B", 1), ("C", 4)],
                "B": [("A", 1), ("C", 2), ("D", 5)],
                "C": [("A", 4), ("B", 2), ("D", 1)],
                "D": [("B", 5), ("C", 1)]
            ]

            let start = "D"
            let distances = rs.dijkstra(graph: graph, start: start)

            print("Shortest distances from \(start): \(distances)")
        */
    }

    func floydWarshall<T: Hashable>(graph: [T: [(T, Int)]]) -> [T: [T: Int]] {      // 플로이드-워셜
        /*
        플로이드-워셜 알고리즘은 모든 노드 쌍 사이의 최단 경로를 찾는 알고리즘입니다. 
        다음은 플로이드-워셜 알고리즘의 기본 구현입니다. 
        그래프의 노드들을 0부터 시작하여 n-1까지의 정수로 표현하고 있습니다. 
        무한 거리는 Int.max로 표현됩니다.
        이 구현은 3중 반복문을 사용하므로 시간 복잡도는 O(n^3)입니다. 
        */
        var dist = [T: [T: Int]]()
        
        // 초기화: 모든 가능한 경로에 대해 dist를 설정
        for (u, edges) in graph {
            for (v, _) in edges {
                dist[u, default: [T: Int]()][v] = Int.max
                dist[v, default: [T: Int]()][u] = Int.max
            }
            dist[u]![u] = 0 // 자기 자신까지의 거리는 0
        }
        
        // 그래프의 간선에 대한 정보를 사용하여 초기 거리를 설정
        for (u, edges) in graph {
            for (v, weight) in edges {
                dist[u]![v] = weight
            }
        }
        
        // 플로이드-워셜 알고리즘
        for k in graph.keys {
            for i in graph.keys {
                for j in graph.keys {
                    if let ik = dist[i]?[k], let kj = dist[k]?[j] {
                        if ik == Int.max || kj == Int.max {
                            continue
                        }
                        if let ij = dist[i]?[j], ij > ik + kj {
                            dist[i]![j] = ik + kj
                        }
                    }
                }
            }
        }
        
        return dist
    }

    func bellmanFord<T: Hashable>(graph: [T: [(T, Int)]], start: T) -> [T: Int]? {      // 벨만-포드
        /*
        주어진 그래프와 시작 정점에서 모든 다른 정점까지의 최단 경로를 찾는 알고리즘입니다. 
        이 알고리즘은 음수 가중치를 갖는 간선도 처리할 수 있고, 음수 가중치의 사이클이 있는 경우에도 감지할 수 있습니다.
        이 알고리즘의 시간 복잡도는 O(VE)입니다, 여기서 V는 정점의 개수이고 E는 간선의 개수입니다.
        */
        var distance: [T: Int] = [start: 0]  // 시작 정점으로부터의 거리를 저장할 딕셔너리

        // 모든 정점에 대한 초기 거리를 무한대로 설정
        for vertex in graph.keys {
            if vertex != start {
                distance[vertex] = Int.max
            }
        }
        
        // 정점의 개수
        let vertexCount = graph.keys.count

        // (vertexCount - 1)번 반복
        for _ in 1..<vertexCount {
            for (u, edges) in graph {
                for (v, weight) in edges {
                    // 시작 정점에서 u까지의 거리가 무한대면 무시
                    guard let distFromStartToU = distance[u] else {
                        continue
                    }

                    // 시작 정점에서 u를 거쳐 v로 가는 거리가 더 짧으면 업데이트
                    if let distFromStartToV = distance[v], distFromStartToU + weight < distFromStartToV {
                        distance[v] = distFromStartToU + weight
                    }
                }
            }
        }

        // 음수 가중치의 사이클 체크
        for (u, edges) in graph {
            for (v, weight) in edges {
                if let distFromStartToU = distance[u], let distFromStartToV = distance[v], distFromStartToU + weight < distFromStartToV {
                    return nil // 음수 가중치의 사이클이 존재
                }
            }
        }

        return distance
    }

    func kruskal<T: Hashable>(edges: [(T, T, Int)]) -> [(T, T, Int)] {      // 크루스칼
        /*
        그래프의 최소 신장 트리(Minimum Spanning Tree, MST)를 찾는 알고리즘 중 하나입니다.
        그래프를 간선의 리스트로 표현하며, 각 간선은 (u, v, weight) 형태의 튜플로 되어 있습니다. 
        알고리즘은 최소 신장 트리의 간선들을 리스트로 반환합니다.
        UnionFind: Union-Find 자료구조를 구현한 것입니다. 이것은 find와 union 연산을 제공합니다.
        kruskal: 크루스칼 알고리즘을 구현한 것입니다. 입력으로 간선의 리스트를 받고, 최소 신장 트리의 간선들을 리스트로 반환합니다.
        이 알고리즘의 시간 복잡도는 O(ElogE)입니다, 여기서 E는 간선의 개수입니다.
        */
        var parent: [T: T] = Dictionary(uniqueKeysWithValues: Set(edges.flatMap { [$0.0, $0.1] }).map { ($0, $0) })
        
        func find(_ element: T) -> T {
            if parent[element] == element {
                return element
            } else {
                parent[element] = find(parent[element]!)
                return parent[element]!
            }
        }
        
        func union(_ a: T, _ b: T) {
            let rootA = find(a)
            let rootB = find(b)
            if rootA != rootB {
                parent[rootA] = rootB
            }
        }

        var result: [(T, T, Int)] = []
        let sortedEdges = edges.sorted { $0.2 < $1.2 }
        
        for edge in sortedEdges {
            let (u, v, weight) = edge
            if find(u) != find(v) {
                result.append(edge)
                union(u, v)
            }
        }
        
        return result
    }

    func prim<T: Hashable>(graph: [T: [(T, Int)]], start: T) -> [(T, T, Int)] {     // 프림
        /*
        그래프에서 최소 신장 트리(Minimum Spanning Tree, MST)를 찾는 알고리즘 중 하나입니다.
        시작 노드에서 시작하여, 이미 연결된 노드 집합에 인접한 노드들 중에서 가장 가중치가 작은 간선을 선택해 나가면서 트리를 확장해 갑니다. 
        이 과정을 모든 노드가 트리에 포함될 때까지 반복합니다.
        배열을 사용한 경우: O(V^2)
        우선순위 큐를 사용한 경우: O(E+VlogV)
        피보나치 힙을 사용한 경우: O(E+VlogV)
        여기서 V는 그래프의 노드 수, E는 간선의 수입니다.
        대부분의 경우에는 우선순위 큐를 사용하여 O(E+VlogV) 시간 복잡도로 문제를 해결할 수 있습니다.
        */
        var visited: Set<T> = [start]
        var priorityQueue: [(Int, T, T)] = []
        var result: [(T, T, Int)] = []
        
        for (neighbor, weight) in graph[start] ?? [] {
            priorityQueue.append((weight, start, neighbor))
        }
        priorityQueue.sort(by: { $0.0 > $1.0 })
        
        while !priorityQueue.isEmpty {
            let (weight, u, v) = priorityQueue.removeLast()
            
            if !visited.contains(v) {
                visited.insert(v)
                result.append((u, v, weight))
                
                for (neighbor, nextWeight) in graph[v] ?? [] {
                    if !visited.contains(neighbor) {
                        priorityQueue.append((nextWeight, v, neighbor))
                    }
                }
                
                priorityQueue.sort(by: { $0.0 > $1.0 })
            }
        }
        
        return result
    }



    //MARK: - 수학적 알고리즘

    func isPrime(_ n: Int) -> Bool { // 소수 판별
        // 소수 판별 √n까지만 검사해서 빠름

        if n < 2 {
            return false
        }

        for i in 2...Int(sqrt(Double(n))) {
            if n % i == 0 {
                return false
            }
        }

        return true
    }
    
    func sieveOfEratosthenes(_ n: Int) -> [Int] { // 에라토스테네스의 체
        // 주어진 범위 내에서 소수 판별 √n까지만 검사해서 빠름

        var isPrime = [Bool](repeating: true, count: n + 1)
        var primes = [Int]()
        
        isPrime[0] = false
        isPrime[1] = false
        
        for i in 2...Int(sqrt(Double(n))) {
            if isPrime[i] {
                for j in stride(from: i * i, to: n + 1, by: i) {
                    isPrime[j] = false
                }
            }
        }
        
        for i in 2...n {
            if isPrime[i] {
                primes.append(i)
            }
        }
        
        return primes
    }
    
    func gcd_recursive(_ a: Int, _ b: Int) -> Int { // 유클리드 재귀
        // 재귀를 이용한 a 와 b 의 최대 공약수

        if b == 0 {
            return a
        }
        return gcd_recursive(b, a % b)
    }

    func gcd_iterative(_ a: Int, _ b: Int) -> Int { // 유클리드 반복문
        // 반복문을 이용한 a 와 b 의 최대 공약수

        var a = a
        var b = b
        while b != 0 {
            let temp = b
            b = a % b
            a = temp
        }
        return a
    }

}


// Test Code
let rs = RaccoonSwift()

// Example usage
let graph: [String: [(String, Int)]] = [
    "A": [("B", 1), ("C", 4)],
    "B": [("A", 1), ("C", 2), ("D", 5)],
    "C": [("A", 4), ("B", 2), ("D", 1)],
    "D": [("B", 5), ("C", 1)]
]

let start = "A"
let mst = rs.prim(graph: graph, start: start)
print("Minimum Spanning Tree:", mst)
