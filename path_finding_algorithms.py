import numpy as np
import time
import heapq
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import random

def generate_weighted_graph(n_vertices: int, n_edges: int, max_weight: int = 100) -> np.ndarray:
    """Generate a random undirected weighted graph."""
    adj_matrix = np.zeros((n_vertices, n_vertices), dtype=int)
    edges_added = 0
    while edges_added < n_edges:
        v1, v2 = np.random.randint(0, n_vertices, 2)
        if v1 == v2 or adj_matrix[v1][v2] != 0:
            continue
        weight = np.random.randint(1, max_weight + 1)
        adj_matrix[v1][v2] = weight
        adj_matrix[v2][v1] = weight
        edges_added += 1
    return adj_matrix

def dijkstra(graph: np.ndarray, start: int) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Dijkstra's algorithm for shortest paths."""
    n = len(graph)
    distances = {i: float('infinity') for i in range(n)}
    distances[start] = 0
    predecessors = {i: None for i in range(n)}
    unvisited = [(0, start)]
    visited = set()

    while unvisited:
        current_distance, current = heapq.heappop(unvisited)
        if current in visited:
            continue
        visited.add(current)

        for neighbor in range(n):
            if graph[current][neighbor] > 0:
                distance = current_distance + graph[current][neighbor]
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current
                    heapq.heappush(unvisited, (distance, neighbor))

    return distances, predecessors

def bellman_ford(graph: np.ndarray, start: int) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Bellman-Ford algorithm for shortest paths."""
    n = len(graph)
    distances = {i: float('infinity') for i in range(n)}
    distances[start] = 0
    predecessors = {i: None for i in range(n)}

    for _ in range(n - 1):
        for u in range(n):
            for v in range(n):
                if graph[u][v] > 0:
                    if distances[u] + graph[u][v] < distances[v]:
                        distances[v] = distances[u] + graph[u][v]
                        predecessors[v] = u

    return distances, predecessors

def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two points."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def generate_grid(rows: int, cols: int, n_obstacles: int) -> np.ndarray:
    """Generate a grid with random obstacles."""
    grid = np.zeros((rows, cols), dtype=int)
    positions = [(i, j) for i in range(rows) for j in range(cols)]
    obstacle_positions = random.sample(positions, n_obstacles)

    for pos in obstacle_positions:
        grid[pos] = 1
    return grid

def get_neighbors(pos: Tuple[int, int], grid: np.ndarray) -> List[Tuple[int, int]]:
    """Get valid neighboring positions."""
    rows, cols = grid.shape
    x, y = pos
    neighbors = []

    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        new_x, new_y = x + dx, y + dy
        if (0 <= new_x < rows and 0 <= new_y < cols and
            grid[new_x, new_y] == 0):
            neighbors.append((new_x, new_y))
    return neighbors

def a_star(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """A* pathfinding algorithm implementation."""
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        current = heapq.heappop(frontier)[1]
        if current == goal:
            break

        for next_pos in get_neighbors(current, grid):
            new_cost = cost_so_far[current] + 1
            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + manhattan_distance(goal, next_pos)
                heapq.heappush(frontier, (priority, next_pos))
                came_from[next_pos] = current

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    path.reverse()
    return path if path[0] == start else []

def visualize_path(grid: np.ndarray, path: List[Tuple[int, int]], start: Tuple[int, int], goal: Tuple[int, int]):
    """Visualize the grid, obstacles, and found path."""
    plt.figure(figsize=(10, 5))
    plt.imshow(grid, cmap='binary')

    if path:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], 'b-', linewidth=2, label='Path')

    plt.plot(start[1], start[0], 'go', label='Start')
    plt.plot(goal[1], goal[0], 'ro', label='Goal')
    plt.grid(True)
    plt.legend()
    plt.title(f'Path Length: {len(path) - 1 if path else "No path found"}')
    plt.show()

if __name__ == "__main__":
    # Example 1: Dijkstra vs Bellman-Ford
    print("Testing Dijkstra vs Bellman-Ford:")
    n_vertices = 100
    n_edges = 500
    graph = generate_weighted_graph(n_vertices, n_edges)
    start_vertex = 0

    start_time = time.time()
    dijkstra_dist, _ = dijkstra(graph, start_vertex)
    dijkstra_time = time.time() - start_time

    start_time = time.time()
    bellman_ford_dist, _ = bellman_ford(graph, start_vertex)
    bellman_ford_time = time.time() - start_time

    print(f"Dijkstra's time: {dijkstra_time:.6f}s")
    print(f"Bellman-Ford time: {bellman_ford_time:.6f}s")

    # Example 2: A* Pathfinding
    print("\nTesting A* Pathfinding:")
    rows, cols = 10, 20
    n_obstacles = 40
    grid = generate_grid(rows, cols, n_obstacles)

    # Find empty positions for start and goal
    empty_positions = [(i, j) for i in range(rows) for j in range(cols)
                      if grid[i, j] == 0]
    start, goal = random.sample(empty_positions, 2)

    start_time = time.time()
    path = a_star(grid, start, goal)
    a_star_time = time.time() - start_time

    print(f"A* time: {a_star_time:.6f}s")
    print(f"Path length: {len(path) - 1 if path else 'No path found'}")

    # Visualize the result
    visualize_path(grid, path, start, goal)
