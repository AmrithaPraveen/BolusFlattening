import numpy as np
import heapq
import math

class MeshFastMarching:
    def __init__(self, mesh):
        self.vertices = mesh.vertices
        self.faces = mesh.faces
        self.n_vertices = len(self.vertices)
        self.build_mesh_data()

    #Build the mesh data as required for the fast marching
    def build_mesh_data(self):
        self.vertex_neighbors = [[] for _ in range(self.n_vertices)]
        self.edge_lengths = {}
        
        print("Building mesh data:")
        for face in self.faces:
            #print(f"Processing face: {face}")
            for i in range(3):
                v1 = face[i]
                v2 = face[(i + 1) % 3]
                
                # Add neighbor if it is not already present
                if v2 not in self.vertex_neighbors[v1]:
                    self.vertex_neighbors[v1].append(v2)
                    self.vertex_neighbors[v2].append(v1)
                    
                    # Compute edge length
                    length = np.linalg.norm(self.vertices[v1] - self.vertices[v2])
                    self.edge_lengths[(v1, v2)] = length
                    self.edge_lengths[(v2, v1)] = length
        
        self.vertex_faces = [[] for _ in range(self.n_vertices)]
        for i, face in enumerate(self.faces):
            for v in face:
                self.vertex_faces[v].append(i)
        


    def compute_geodesic_distance_matrix(self):
        print("Computing geodesic distances...")
        D = np.zeros((self.n_vertices, self.n_vertices))
        
        for i in range(self.n_vertices):
            #print(f"\nComputing distances from source vertex {i}")
            distances = self.compute_distance(i)
            D[i, :] = distances
            if i % 200 == 0:
                print(f"Processed {i} out of {self.n_vertices} vertices.")
            #print(f"Distances from vertex {i}: {distances}")
        
        print("\nRaw distance matrix:")
        print(D)
        
        # Ensure symmetry
        D = (D + D.T) / 2
        
        print("\nSymmetrized distance matrix:")
        print(D)
        
        return D

    def compute_distance(self, source):
        distances = np.full(self.n_vertices, np.inf)
        distances[source] = 0
        
        state = np.zeros(self.n_vertices, dtype=int)
        
        narrow_band = []
        heapq.heappush(narrow_band, (0, source))
        state[source] = 1

        while narrow_band:
            dist, vertex = heapq.heappop(narrow_band)
            
            if state[vertex] == 2:  # 2 is Already known
                continue
            
            # Update state to known and calculate final distance
            state[vertex] = 2
            distances[vertex] = dist
            
            
            # Update neighbors
            for neighbor in self.vertex_neighbors[vertex]:
                if state[neighbor] != 2:  
                    new_dist = self.update_vertex(neighbor, vertex, distances, state)
                    
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(narrow_band, (new_dist, neighbor))
                        state[neighbor] = 1
        
        return distances



    def update_vertex(self, vertex, current_vertex, distances, state):
        # Initialize distance 
        min_distance = distances[vertex]

        # Iterate over all neighbors of the vertex
        for neighbor in self.vertex_neighbors[vertex]:
            
            if state[neighbor] == 2:  
                
                edge_length = self.edge_lengths.get((vertex, neighbor), float('inf'))

                # Calculate new distance 
                potential_distance = distances[neighbor] + edge_length

                # Update the shortest distance 
                if potential_distance < min_distance:
                    min_distance = potential_distance

        return min_distance


