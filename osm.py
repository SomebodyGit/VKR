import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import time as tm
import tkinter as tk
from tkinter import messagebox
from geopy.distance import great_circle

class RouteFinder:
    def __init__(self, start_coords, target_coords):
        self.start_coords = start_coords
        self.target_coords = target_coords
        self.G = None
        self.start_node = None
        self.target_node = None
        self.node_to_index = None
        self.index_to_node = None
        self.graph_dict = None
        self.Q = None
        self.path_nodes = None
        self.agent_path_length = 0

    def load_graph(self):
        center_point = (
            (self.start_coords[0] + self.target_coords[0]) / 2,
            (self.start_coords[1] + self.target_coords[1]) / 2
        )
        dist = great_circle(self.start_coords, self.target_coords).meters
        self.G = ox.graph_from_point(center_point, dist=dist, network_type='drive')

        strongly_connected_components = list(nx.strongly_connected_components(self.G))
        largest_component = max(strongly_connected_components, key=len)
        self.G = self.G.subgraph(largest_component).copy()

    def prepare_data(self):
        self.start_node = ox.distance.nearest_nodes(self.G, self.start_coords[1], self.start_coords[0])
        self.target_node = ox.distance.nearest_nodes(self.G, self.target_coords[1], self.target_coords[0])

        if not nx.has_path(self.G, self.start_node, self.target_node):
            raise ValueError("Целевой узел недостижим из начального узла.")

        nodes = list(self.G.nodes)
        self.node_to_index = {node: idx for idx, node in enumerate(nodes)}
        self.index_to_node = {idx: node for node, idx in self.node_to_index.items()}
        self.graph_dict = {
            self.node_to_index[node]: [self.node_to_index[neighbor] for neighbor in self.G.neighbors(node)]
            for node in self.G.nodes
        }

    def euclidean_distance(self, node1, node2):
        x1, y1 = self.G.nodes[node1]['x'], self.G.nodes[node1]['y']
        x2, y2 = self.G.nodes[node2]['x'], self.G.nodes[node2]['y']
        return sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def train_agent(self):
        self.Q = {node: {act: 0.0 for act in self.graph_dict[node]} for node in self.graph_dict}
        learning_rate = 0.1
        discount_factor = 0.9
        num_episodes = 10000
        epsilon = 0.1
        episode_rewards = []

        for episode in range(num_episodes):
            current_node = self.node_to_index[self.start_node]
            done = False
            steps = 0
            total_reward = 0
            while not done and steps < 10000:
                possible_actions = self.graph_dict[current_node]
                if not possible_actions:
                    break
                if np.random.random() < epsilon:
                    action = np.random.choice(possible_actions)
                else:
                    action = max(possible_actions, key=lambda a: self.Q[current_node][a])
                if action == self.node_to_index[self.target_node]:
                    reward = 100
                    done = True
                else:
                    reward = -self.euclidean_distance(self.index_to_node[current_node], self.target_node)
                total_reward += reward
                next_node = action
                best_next_action = max(self.Q[next_node].values()) if self.Q[next_node] else 0
                self.Q[current_node][action] += learning_rate * (
                    reward + discount_factor * best_next_action - self.Q[current_node][action]
                )
                current_node = next_node
                steps += 1
            episode_rewards.append(total_reward)
        return episode_rewards

    def test_agent(self):
        current_node = self.node_to_index[self.start_node]
        path = [current_node]
        max_steps = 10000000
        steps = 0
        visited = set()

        while current_node != self.node_to_index[self.target_node] and steps < max_steps:
            if current_node in visited:
                print("Зацикливание обнаружено.")
                break
            visited.add(current_node)
            possible_actions = self.graph_dict[current_node]
            if not possible_actions:
                break
            action = max(possible_actions, key=lambda a: self.Q[current_node][a])
            current_node = action
            path.append(current_node)
            steps += 1

        self.path_nodes = [self.index_to_node[idx] for idx in path]
        self._calculate_path_length()
        return current_node == self.node_to_index[self.target_node]

    def _calculate_path_length(self):
        self.agent_path_length = 0
        for i in range(len(self.path_nodes) - 1):
            u = self.path_nodes[i]
            v = self.path_nodes[i + 1]
            edge_data = self.G.get_edge_data(u, v)
            if edge_data:
                if isinstance(edge_data, dict) and 'length' in edge_data:
                    self.agent_path_length += edge_data['length']
                elif all(isinstance(val, dict) for val in edge_data.values()):
                    lengths = [data['length'] for data in edge_data.values() if 'length' in data]
                    if lengths:
                        self.agent_path_length += min(lengths)

    def visualize(self):
        fig, ax = ox.plot_graph(self.G, node_size=0, bgcolor='k', show=False, close=False, figsize=(15, 15))
        for i in range(len(self.path_nodes) - 1):
            u = self.path_nodes[i]
            v = self.path_nodes[i + 1]
            edges = self.G.get_edge_data(u, v)
            if edges:
                edge_data = list(edges.values())[0]
                if 'geometry' in edge_data:
                    geom = edge_data['geometry']
                    xs, ys = geom.xy
                    ax.plot(xs, ys, color='blue', linewidth=2)
                else:
                    x1, y1 = self.G.nodes[u]['x'], self.G.nodes[u]['y']
                    x2, y2 = self.G.nodes[v]['x'], self.G.nodes[v]['y']
                    ax.plot([x1, x2], [y1, y2], color='blue', linewidth=2)
        ax.plot([], [], color='blue', label=f'Путь агента ({self.agent_path_length:.2f} м)')
        ax.legend()
        padding = 0.001
        min_x = min(self.G.nodes[node]['x'] for node in self.path_nodes)
        max_x = max(self.G.nodes[node]['x'] for node in self.path_nodes)
        min_y = min(self.G.nodes[node]['y'] for node in self.path_nodes)
        max_y = max(self.G.nodes[node]['y'] for node in self.path_nodes)
        ax.set_xlim(min_x - padding, max_x + padding)
        ax.set_ylim(min_y - padding, max_y + padding)
        plt.savefig('route.png')
        plt.close()

    def run(self):
        start_time = tm.time()
        self.load_graph()
        self.prepare_data()
        episode_rewards = self.train_agent()
        success = self.test_agent()
        if success:
            print(f"Длина пути: {self.agent_path_length:.2f} метров")
            self.visualize()
        else:
            print("Путь не найден.")

        end_time = tm.time()
        print(f"Общее время выполнения: {end_time - start_time:.2f} секунд")
        return success

def run_program():
    try:
        start_lat = float(start_lat_entry.get())
        start_lon = float(start_lon_entry.get())
        target_lat = float(target_lat_entry.get())
        target_lon = float(target_lon_entry.get())
        start_coords = (start_lat, start_lon)
        target_coords = (target_lat, target_lon)

        wait_window = tk.Toplevel(root)
        wait_window.title("Обработка")
        wait_window.geometry("400x200")
        tk.Label(wait_window, text="Обработка запроса. Пожалуйста, подождите...").pack(pady=10)
        wait_window.update()
        root.update()
        tm.sleep(1)

        route_finder = RouteFinder(start_coords, target_coords)
        success = route_finder.run()

        wait_window.destroy()
        if success:
            messagebox.showinfo("Успех", "Программа выполнена успешно!")
        else:
            messagebox.showerror("Ошибка", "Путь не найден.")
    except ValueError as e:
        messagebox.showerror("Ошибка", f"Некорректные данные: {str(e)}")
        wait_window.destroy()
    except Exception as e:
        messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")
        wait_window.destroy()

root = tk.Tk()
root.title("Маршрутизация на улично-дорожной сети")
root.geometry("300x300")

tk.Label(root, text="Широта начальной точки:").pack()
start_lat_entry = tk.Entry(root)
start_lat_entry.insert(0, "55.810208")
start_lat_entry.pack()

tk.Label(root, text="Долгота начальной точки:").pack()
start_lon_entry = tk.Entry(root)
start_lon_entry.insert(0, "37.498321")
start_lon_entry.pack()

tk.Label(root, text="Широта конечной точки:").pack()
target_lat_entry = tk.Entry(root)
target_lat_entry.insert(0, "55.739727")
target_lat_entry.pack()

tk.Label(root, text="Долгота конечной точки:").pack()
target_lon_entry = tk.Entry(root)
target_lon_entry.insert(0, "37.408067")
target_lon_entry.pack()

tk.Button(root, text="Запустить", command=run_program).pack(pady=10)

root.mainloop()