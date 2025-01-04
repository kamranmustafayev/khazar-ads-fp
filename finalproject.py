import time
import tracemalloc
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class GraphModule:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_task(self, task, dependencies=[]):
        self.graph.add_node(task)
        for dep in dependencies:
            self.graph.add_edge(dep, task)

    def detect_cycle(self):
        try:
            cycle = nx.find_cycle(self.graph, orientation='original')
            return True, cycle
        except nx.NetworkXNoCycle:
            return False, None

    def topological_sort(self):
        try:
            order = list(nx.topological_sort(self.graph))
            return order
        except nx.NetworkXUnfeasible:
            raise ValueError("Cycle detected; topological sort not possible.")

    def visualize(self):
        plt.figure(figsize=(8, 6))
        nx.draw(self.graph, with_labels=True, node_color='lightblue', font_weight='bold', node_size=1500)
        plt.show()

class DataOperationsModule:
    @staticmethod
    def merge_sort(arr):
        if len(arr) > 1:
            mid = len(arr) // 2
            left = arr[:mid]
            right = arr[mid:]

            DataOperationsModule.merge_sort(left)
            DataOperationsModule.merge_sort(right)

            i = j = k = 0
            while i < len(left) and j < len(right):
                if left[i] < right[j]:
                    arr[k] = left[i]
                    i += 1
                else:
                    arr[k] = right[j]
                    j += 1
                k += 1

            while i < len(left):
                arr[k] = left[i]
                i += 1
                k += 1

            while j < len(right):
                arr[k] = right[j]
                j += 1
                k += 1

    @staticmethod
    def quick_sort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return DataOperationsModule.quick_sort(left) + middle + DataOperationsModule.quick_sort(right)

    @staticmethod
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1

class StackQueueModule:
    def __init__(self):
        self.array = [None] * 100
        self.stack1_top = -1
        self.stack2_top = 100

    def push_stack1(self, value):
        if self.stack1_top + 1 == self.stack2_top:
            raise OverflowError("Stack Overflow")
        self.stack1_top += 1
        self.array[self.stack1_top] = value

    def push_stack2(self, value):
        if self.stack2_top - 1 == self.stack1_top:
            raise OverflowError("Stack Overflow")
        self.stack2_top -= 1
        self.array[self.stack2_top] = value

    def pop_stack1(self):
        if self.stack1_top == -1:
            raise IndexError("Stack Underflow")
        value = self.array[self.stack1_top]
        self.stack1_top -= 1
        return value

    def pop_stack2(self):
        if self.stack2_top == 100:
            raise IndexError("Stack Underflow")
        value = self.array[self.stack2_top]
        self.stack2_top += 1
        return value

    class LinkedListQueue:
        class Node:
            def __init__(self, value):
                self.value = value
                self.next = None

        def __init__(self):
            self.front = self.rear = None

        def enqueue(self, value):
            new_node = self.Node(value)
            if not self.rear:
                self.front = self.rear = new_node
                return
            self.rear.next = new_node
            self.rear = new_node

        def dequeue(self):
            if not self.front:
                raise IndexError("Queue is empty")
            value = self.front.value
            self.front = self.front.next
            if not self.front:
                self.rear = None
            return value

class PerformanceModule:
    @staticmethod
    def benchmark(func, *args, **kwargs):
        tracemalloc.start()
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Time: {end_time - start_time:.6f}s, Memory: {peak / 10**6:.6f} MB")
        return result

if __name__ == "__main__":
    # Example of usage
    graph = GraphModule()
    graph.add_task("Task1")
    graph.add_task("Task2", ["Task1"])
    graph.add_task("Task3", ["Task2"])

    print("Cycle Detection:", graph.detect_cycle())
    print("Topological Sort:", graph.topological_sort())
    graph.visualize()

    stack_queue = StackQueueModule()
    stack_queue.push_stack1(10)
    stack_queue.push_stack1(20)
    print("Popped from Stack1:", stack_queue.pop_stack1())

    performance = PerformanceModule()
    arr = [3, 2, 1, 5, 4]
    sorted_arr = performance.benchmark(DataOperationsModule.quick_sort, arr)
    print("Sorted Array:", sorted_arr)
