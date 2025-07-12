from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from queue import PriorityQueue
import random
import numpy as np

app = Flask(__name__)
CORS(app)

ROWS = 40


class Node:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.neighbors = []
        self.distance = float("inf")
        self.g_score = float("inf")
        self.f_score = float("inf")
        self.previous = None
        self.type = "empty"

    def __lt__(self, other):
        return False


def make_grid():
    grid = []
    for i in range(ROWS):
        grid.append([])
        for j in range(ROWS):
            node = Node(i, j)
            grid[i].append(node)
    return grid


def get_neighbors(node, grid):
    neighbors = []
    if node.row < ROWS - 1 and grid[node.row + 1][node.col].type != "wall":
        neighbors.append(grid[node.row + 1][node.col])
    if node.row > 0 and grid[node.row - 1][node.col].type != "wall":
        neighbors.append(grid[node.row - 1][node.col])
    if node.col < ROWS - 1 and grid[node.row][node.col + 1].type != "wall":
        neighbors.append(grid[node.row][node.col + 1])
    if node.col > 0 and grid[node.row][node.col - 1].type != "wall":
        neighbors.append(grid[node.row][node.col - 1])
    return neighbors


def heuristic(a, b):
    return abs(a.row - b.row) + abs(a.col - b.col)


def reconstruct_path(end):
    current = end
    path = []
    while current:
        path.append({"row": current.row, "col": current.col})
        current = current.previous
    return path[::-1]


def dijkstra_algorithm(grid_data, start_pos, end_pos):
    grid = make_grid()
    walls = grid_data.get('walls', [])
    for wall in walls:
        if 0 <= wall['row'] < ROWS and 0 <= wall['col'] < ROWS:
            grid[wall['row']][wall['col']].type = "wall"

    start = grid[start_pos['row']][start_pos['col']]
    end = grid[end_pos['row']][end_pos['col']]
    start.type = "start"
    end.type = "end"

    count = 0
    open_set = PriorityQueue()
    start.distance = 0
    open_set.put((0, count, start))
    visited_order = []
    explanations = []
    nodes_explored = 0
    step_counter = 1

    # Initial explanation
    explanations.append({
        "step": step_counter,
        "title": "🚀 Dijkstra's Algorithm Started",
        "description": f"Beginning search from start ({start.row}, {start.col}) to destination ({end.row}, {end.col})",
        "details": "Dijkstra's algorithm guarantees finding the shortest path by exploring nodes in order of their distance from the start. It systematically visits the closest unvisited node at each step."
    })
    step_counter += 1

    explanations.append({
        "step": step_counter,
        "title": "🔍 Exploring the Grid",
        "description": "Light blue nodes show areas being explored as the algorithm searches for the shortest path",
        "details": "Each blue node represents a location the algorithm has visited and evaluated. The algorithm spreads outward from the start point, guaranteeing it finds the optimal route."
    })
    step_counter += 1

    total_reachable_estimate = ROWS * ROWS - len(walls)
    progress_milestones = [0.25, 0.5, 0.75]
    milestone_index = 0

    while not open_set.empty():
        current = open_set.get()[2]
        nodes_explored += 1

        if current == end:
            path = reconstruct_path(end)
            explanations.append({
                "step": step_counter,
                "title": "✅ Shortest Path Found!",
                "description": f"Successfully found the optimal path with distance {current.distance} after exploring {nodes_explored} nodes",
                "details": f"The yellow line shows the shortest path containing {len(path)} nodes. Dijkstra guarantees this is the optimal route with no shorter alternative possible."
            })
            return {"success": True, "path": path, "visited": visited_order, "message": "Path found!",
                    "explanations": explanations}

        neighbors = get_neighbors(current, grid)
        for neighbor in neighbors:
            temp = current.distance + 1
            if temp < neighbor.distance:
                neighbor.distance = temp
                neighbor.previous = current
                count += 1
                open_set.put((neighbor.distance, count, neighbor))

        if current.type not in ("start", "end"):
            visited_order.append({"row": current.row, "col": current.col})

        # Add progress update at key milestones
        progress_ratio = nodes_explored / total_reachable_estimate
        if (milestone_index < len(progress_milestones) and
                progress_ratio >= progress_milestones[milestone_index]):
            milestone_index += 1
            explanations.append({
                "step": step_counter,
                "title": f"📊 Search Progress: {int(progress_ratio * 100)}% Complete",
                "description": f"Explored {nodes_explored} nodes so far, systematically searching for the shortest path",
                "details": "The algorithm is methodically exploring all possible routes, ensuring no shorter path is missed. Blue areas show the search frontier expanding outward."
            })
            step_counter += 1

    explanations.append({
        "step": step_counter,
        "title": "❌ No Path Available",
        "description": f"Explored {nodes_explored} reachable nodes but destination is completely blocked",
        "details": "All possible routes to the destination are blocked by walls. The blue areas show everywhere the algorithm could reach from the starting point."
    })
    return {"success": False, "path": [], "visited": visited_order, "message": "No path found!",
            "explanations": explanations}


def a_star_algorithm(grid_data, start_pos, end_pos):
    grid = make_grid()
    walls = grid_data.get('walls', [])
    for wall in walls:
        if 0 <= wall['row'] < ROWS and 0 <= wall['col'] < ROWS:
            grid[wall['row']][wall['col']].type = "wall"

    start = grid[start_pos['row']][start_pos['col']]
    end = grid[end_pos['row']][end_pos['col']]
    start.type = "start"
    end.type = "end"

    count = 0
    open_set = PriorityQueue()
    start.g_score = 0
    start.f_score = heuristic(start, end)
    open_set.put((start.f_score, count, start))
    open_set_hash = {start}
    visited_order = []
    explanations = []
    nodes_explored = 0
    step_counter = 1

    # Initial explanation
    explanations.append({
        "step": step_counter,
        "title": "🎯 A* Algorithm Started",
        "description": f"Beginning intelligent search from start ({start.row}, {start.col}) to destination ({end.row}, {end.col})",
        "details": f"A* combines actual distance with estimated remaining distance (heuristic: {start.f_score}) to guide the search more efficiently toward the target than Dijkstra."
    })
    step_counter += 1

    explanations.append({
        "step": step_counter,
        "title": "🧭 Smart Pathfinding in Progress",
        "description": "Light blue nodes show the algorithm's intelligent exploration, focusing on promising directions",
        "details": "Unlike Dijkstra, A* uses heuristic guidance to prioritize nodes closer to the destination, making the search more targeted and efficient while still guaranteeing the optimal path."
    })
    step_counter += 1

    estimated_nodes = heuristic(start, end) * 2  # Rough estimate for A*
    progress_milestones = [0.4, 0.8]
    milestone_index = 0

    while not open_set.empty():
        current = open_set.get()[2]
        open_set_hash.remove(current)
        nodes_explored += 1

        if current == end:
            path = reconstruct_path(end)
            explanations.append({
                "step": step_counter,
                "title": "🏆 Optimal Path Found!",
                "description": f"A* found the optimal path with cost {current.g_score} after exploring only {nodes_explored} nodes",
                "details": f"The yellow line shows the shortest path. A*'s heuristic guidance helped it explore fewer nodes than Dijkstra while still guaranteeing the optimal solution."
            })
            return {"success": True, "path": path, "visited": visited_order, "message": "Path found!",
                    "explanations": explanations}

        neighbors = get_neighbors(current, grid)
        for neighbor in neighbors:
            temp_g_score = current.g_score + 1
            if temp_g_score < neighbor.g_score:
                neighbor.previous = current
                neighbor.g_score = temp_g_score
                neighbor.f_score = temp_g_score + heuristic(neighbor, end)

                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((neighbor.f_score, count, neighbor))
                    open_set_hash.add(neighbor)

        if current.type not in ("start", "end"):
            visited_order.append({"row": current.row, "col": current.col})

        # Add progress update at key milestones
        if (milestone_index < len(progress_milestones) and
                nodes_explored >= estimated_nodes * progress_milestones[milestone_index]):
            milestone_index += 1
            explanations.append({
                "step": step_counter,
                "title": f"🔍 Efficient Search in Progress",
                "description": f"Explored {nodes_explored} nodes using heuristic guidance toward the target",
                "details": "A* is intelligently focusing its search on the most promising paths. The blue areas show where the algorithm has looked, guided by its understanding of the remaining distance to the goal."
            })
            step_counter += 1

    explanations.append({
        "step": step_counter,
        "title": "❌ No Path Available",
        "description": f"Explored {nodes_explored} nodes but destination is unreachable",
        "details": "Even with heuristic guidance, no valid path exists through the obstacles. The blue areas show all the reachable locations from the starting point."
    })
    return {"success": False, "path": [], "visited": visited_order, "message": "No path found!",
            "explanations": explanations}


def kmeans_algorithm(points, k, max_iterations):
    if len(points) < k:
        return {"success": False, "message": "Not enough points for clustering", "explanations": []}

    # Initialize centroids randomly
    centroids = random.sample(points, k)
    iterations_data = []
    explanations = []
    step_counter = 1

    explanations.append({
        "step": step_counter,
        "title": "🎯 K-Means Initialization",
        "description": f"Starting K-Means with {k} clusters and {len(points)} data points.",
        "details": f"Randomly selected {k} initial centroids (black X marks). Goal: minimize within-cluster sum of squares by grouping similar points together."
    })
    step_counter += 1

    for iteration in range(max_iterations):
        # Assign points to clusters
        clusters = [[] for _ in range(k)]
        for point in points:
            distances = [np.sqrt((point['x'] - c['x']) ** 2 + (point['y'] - c['y']) ** 2) for c in centroids]
            cluster_idx = distances.index(min(distances))
            clusters[cluster_idx].append(point)

        explanations.append({
            "step": step_counter,
            "title": f"📍 Iteration {iteration + 1}: Point Assignment",
            "description": f"Assigned each point to its nearest centroid using Euclidean distance.",
            "details": f"Cluster sizes: {[len(cluster) for cluster in clusters]}. Points change color to match their assigned cluster."
        })
        step_counter += 1

        # Update centroids
        new_centroids = []
        for i, cluster in enumerate(clusters):
            if cluster:
                avg_x = sum(p['x'] for p in cluster) / len(cluster)
                avg_y = sum(p['y'] for p in cluster) / len(cluster)
                new_centroids.append({'x': avg_x, 'y': avg_y})
            else:
                new_centroids.append(centroids[i])

        explanations.append({
            "step": step_counter,
            "title": f"🎯 Iteration {iteration + 1}: Centroid Update",
            "description": "Moved each centroid to the mean position of its assigned points.",
            "details": "New centroids (black X marks) are positioned at the geometric center of their clusters, representing the 'average' location of all points in that group."
        })
        step_counter += 1

        iterations_data.append({
            'clusters': clusters,
            'centroids': new_centroids.copy()
        })

        # Check for convergence
        converged = True
        total_movement = 0
        for i in range(k):
            movement = abs(centroids[i]['x'] - new_centroids[i]['x']) + abs(centroids[i]['y'] - new_centroids[i]['y'])
            total_movement += movement
            if movement > 1:
                converged = False

        if converged:
            explanations.append({
                "step": step_counter,
                "title": "✅ Convergence Achieved!",
                "description": f"Centroids stopped moving significantly (total movement: {total_movement:.2f}).",
                "details": "Algorithm has converged - clusters are stable and optimal. Further iterations would not change the groupings."
            })
            break

        centroids = new_centroids

    final_message = f"K-Means completed in {len(iterations_data)} iterations"
    if not converged:
        explanations.append({
            "step": step_counter,
            "title": "⏱️ Maximum Iterations Reached",
            "description": f"Stopped after {max_iterations} iterations without full convergence.",
            "details": "Centroids may still be moving slightly, but we've reached the iteration limit. The current clustering is likely very close to optimal."
        })

    return {"success": True, "iterations": iterations_data, "message": final_message, "explanations": explanations}


def agglomerative_clustering_algorithm(points, target_clusters):
    if len(points) < 2:
        return {"success": False, "message": "Need at least 2 points for clustering", "explanations": []}

    if target_clusters >= len(points):
        return {"success": False, "message": "Target clusters must be less than number of points", "explanations": []}

    # Initialize - each point is its own cluster
    clusters = [[point] for point in points]
    merge_history = []
    explanations = []
    step_counter = 1

    explanations.append({
        "step": step_counter,
        "title": "🌳 Agglomerative Clustering Started",
        "description": f"Starting with {len(points)} individual clusters, merging down to {target_clusters} final clusters.",
        "details": "Agglomerative clustering uses a bottom-up approach: start with each point as its own cluster, then repeatedly merge the two closest clusters until reaching the desired number."
    })
    step_counter += 1

    explanations.append({
        "step": step_counter,
        "title": "📊 Initial State",
        "description": f"Each of the {len(points)} points starts as its own cluster (each with unique color).",
        "details": "This hierarchical approach builds clusters by progressively combining the closest pairs, creating a tree-like structure of nested groupings."
    })
    step_counter += 1

    merge_count = 0
    while len(clusters) > target_clusters:
        # Find the two closest clusters
        min_distance = float('inf')
        merge_indices = (0, 1)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Calculate distance between clusters (complete linkage)
                max_dist = 0
                for point1 in clusters[i]:
                    for point2 in clusters[j]:
                        dist = np.sqrt((point1['x'] - point2['x']) ** 2 + (point1['y'] - point2['y']) ** 2)
                        max_dist = max(max_dist, dist)

                if max_dist < min_distance:
                    min_distance = max_dist
                    merge_indices = (i, j)

        # Merge the closest clusters
        i, j = merge_indices
        merged_cluster = clusters[i] + clusters[j]

        # Remove the old clusters and add the merged one
        new_clusters = []
        for k, cluster in enumerate(clusters):
            if k != i and k != j:
                new_clusters.append(cluster)
        new_clusters.append(merged_cluster)
        clusters = new_clusters

        merge_count += 1

        # Record this merge step
        merge_history.append({
            'clusters': [cluster.copy() for cluster in clusters],
            'merged_indices': merge_indices,
            'distance': min_distance,
            'remaining_clusters': len(clusters)
        })

        explanations.append({
            "step": step_counter,
            "title": f"🔗 Merge {merge_count}: Combining Closest Clusters",
            "description": f"Merged two closest clusters (distance: {min_distance:.1f}). Now have {len(clusters)} clusters remaining.",
            "details": f"Used complete linkage: measured distance as the maximum distance between any two points in different clusters. This creates compact, spherical clusters."
        })
        step_counter += 1

    explanations.append({
        "step": step_counter,
        "title": "✅ Clustering Complete!",
        "description": f"Successfully created {target_clusters} final clusters through {merge_count} merge operations.",
        "details": f"The hierarchical clustering process has formed {target_clusters} distinct groups. Each group contains points that were progressively determined to be most similar to each other."
    })

    return {
        "success": True,
        "merge_history": merge_history,
        "final_clusters": clusters,
        "message": f"Agglomerative clustering completed with {target_clusters} clusters",
        "explanations": explanations
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/dijkstra', methods=['POST'])
def run_dijkstra():
    try:
        data = request.get_json()
        if not data or 'start' not in data or 'end' not in data:
            return jsonify({"success": False, "message": "Missing start or end position"}), 400
        result = dijkstra_algorithm(data.get('grid', {}), data['start'], data['end'])
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "message": "Error running algorithm"}), 500


@app.route('/api/astar', methods=['POST'])
def run_astar():
    try:
        data = request.get_json()
        if not data or 'start' not in data or 'end' not in data:
            return jsonify({"success": False, "message": "Missing start or end position"}), 400
        result = a_star_algorithm(data.get('grid', {}), data['start'], data['end'])
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "message": "Error running algorithm"}), 500


@app.route('/api/kmeans', methods=['POST'])
def run_kmeans():
    try:
        data = request.get_json()
        if not data or 'points' not in data:
            return jsonify({"success": False, "message": "No points provided"}), 400

        points = data['points']
        k = data.get('k', 3)
        max_iterations = data.get('max_iterations', 10)

        result = kmeans_algorithm(points, k, max_iterations)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "message": "Error running K-Means"}), 500


@app.route('/api/agglomerative', methods=['POST'])
def run_agglomerative():
    try:
        data = request.get_json()
        if not data or 'points' not in data:
            return jsonify({"success": False, "message": "No points provided"}), 400

        points = data['points']
        target_clusters = data.get('target_clusters', 3)

        result = agglomerative_clustering_algorithm(points, target_clusters)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "message": "Error running Agglomerative Clustering"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)