
import numpy as np
import trimesh
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from validationmetrics import MeshValidationMetrics
from fastmarching import MeshFastMarching
#from dijikstra import MeshDijkstra
from scipy.spatial import cKDTree
from meshaligner import MeshAligner
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
import time
import cv2
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon
class MeshFlattener:
    def __init__(self, mesh_path):
        self.mesh = trimesh.load_mesh(mesh_path)
      
        self.vertices_3d = self.mesh.vertices
        self.faces = self.mesh.faces
        self.n_vertices = len(self.vertices_3d)
        self.vertices_2d = None
        self.original_area = self.mesh.area

    def extract_and_plot_contours(self, image_path, output_pdf_path):
        # Load the image 
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Image not loaded correctly. Check the file path.")
        
        # Convert the image to binary image
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Finding contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_img = np.zeros_like(img, dtype=np.uint8)  
        
        # Draw the contour 
        cv2.drawContours(contour_img, contours, -1, (255), 3)  

        # Invert the colors so the contours are black and the background is white
        inverted_img = 255 - contour_img
        CM_TO_INCHES = 0.393701
        dpi = 100
        height, width = img.shape
        height = height * CM_TO_INCHES
        width = width * CM_TO_INCHES 
        
        figsize = (width / dpi, height / dpi)  # Size in inches

        # Use PdfPages to save the figure as a PDF
        with PdfPages(output_pdf_path) as pdf:
            fig = plt.figure()
            fig.set_size_inches(210/25.4, 297/25.4)  # A4 in inches
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(inverted_img, cmap='gray', aspect='equal')
            #plt.title('Inverted Image with Contours')
            #plt.axis('off')
            #plt.tight_layout(pad=0)
            
            # Save the figure to a PDF
            pdf.savefig(fig, bbox_inches=None, pad_inches=0)
            plt.close(fig)

        print("PDF saved successfully.")




    def save_2d_mesh_image_filled(self, filename):
        try:
            import matplotlib.pyplot as plt
            
            # Get the bounds of your mesh
            x_min = min(v[0] for f in self.faces for v in self.vertices_2d[f])
            x_max = max(v[0] for f in self.faces for v in self.vertices_2d[f])
            y_min = min(v[1] for f in self.faces for v in self.vertices_2d[f])
            y_max = max(v[1] for f in self.faces for v in self.vertices_2d[f])
            
            # Create figure with exact size
            fig = plt.figure(frameon=False)
            ax = fig.add_subplot(111)
            # Calculate physical size based on mesh coordinates
            CM_TO_INCHES = 0.393701
            A4_width_in = 8.27
            A4_height_in = 11.69
            width_inches = (x_max - x_min) * CM_TO_INCHES
            height_inches = (y_max - y_min) * CM_TO_INCHES
            fig.set_size_inches(A4_width_in, A4_height_in)
            scale_x = min(1, A4_width_in / width_inches)
            scale_y = min(1, A4_height_in / height_inches)
            scale = min(scale_x, scale_y)            
            # Fill triangles
            for face in self.faces:
                verts = self.vertices_2d[face]
                x = [(verts[i][0]-x_min) * CM_TO_INCHES  for i in range(3)]
                y = [(verts[i][1] - y_min) * CM_TO_INCHES for i in range(3)]
                ax.fill(x, y, 'k', edgecolor='k')
            
            # Set precise limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
          
            # Remove all spacing
            ax.set_aspect('equal')
            plt.axis('off')
            ax.set_position([0, 0, 1, 1])

            # Save with exact dimensions
            plt.savefig(filename, 
                       dpi=100, 
                       bbox_inches='tight',  
                       pad_inches=0,      
                       facecolor='white')
            plt.close()
        except Exception as e:
            print(f"Failed to save image: {str(e)}")


    def save_2d_mesh_pdf(self, filename):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            from matplotlib.transforms import Affine2D
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            # Plot at exact size (1 unit = 1mm)
            for face in self.faces:
                verts = self.vertices_2d[face]
                for i in range(3):
                    #ax.plot([verts[i][0], verts[(i+1)%3][0]], 
                     #      [verts[i][1], verts[(i+1)%3][1]], 
                    #       'k-', linewidth=0.5)
                    x = [verts[i][0] for i in range(3)]
                    y = [verts[i][1] for i in range(3)]
                    ax.fill(x, y, 'k', edgecolor='k')
            
            ax.set_aspect('equal')
            plt.axis('off')
            
            # Set figure size explicitly in mm
            fig.set_size_inches(210/25.4, 297/25.4)  # A4 in inches
            
            with PdfPages(filename) as pdf:
                pdf.savefig(fig, dpi=25.4)  # 1 dot per mm
                
            plt.close()
            
        except Exception as e:
            print(f"Failed to save PDF: {str(e)}")



    
    def _compute_area(self, vertices):
        """Compute mesh area with error checking"""
        try:
            mesh_2d = trimesh.Trimesh(
                vertices=np.column_stack((vertices, np.zeros(len(vertices)))),
                faces=self.faces)
            return mesh_2d.area
        except Exception as e:
            print(f"Warning: Area computation failed: {str(e)}")
            return 0

    def flatten_mesh(self):
        """Main flattening procedure"""
        # 1. Compute geodesic distances
        print("vertices")
        print (self.vertices_3d)
        print("faces")
        print (self.faces)

        fmm = MeshFastMarching(self.mesh)

        D = fmm.compute_geodesic_distance_matrix()
        print ("Fast Marching")
        print (D)

        # 2. Center distance matrix
        print("Computing centered matrix...")
        n = D.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ (D * D) @ H
        
        # 3. Compute eigendecomposition
        print("Computing 2D embedding...")
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        print("Get 2D coordinates...")

        #scale = np.sqrt(np.abs(eigenvalues[:2]))
        #vertices_2d = eigenvectors[:, :2] * scale[None, :]
        vertices_2d = eigenvectors[:, :2] * np.sqrt(np.abs(eigenvalues[:2]))
        area_2d = self._compute_area(vertices_2d)
        if area_2d > 0: 
            scale = np.sqrt(self.original_area / area_2d)
            vertices_2d = vertices_2d * scale
        print("Center the flattened mesh...")
        aarea_2d_after = self._compute_area(vertices_2d)
        # 6. Center the flattened mesh
        self.vertices_2d = vertices_2d - np.mean(vertices_2d, axis=0)
        area_2d_center = self._compute_area(self.vertices_2d)
        #x_size = np.max(self.vertices_2d[:,0]) - np.min(self.vertices_2d[:,0])
        #print(f"2D mesh width: {x_size}")
        return self.vertices_2d
    def visualize(self):
        """Visualize original and flattened meshes with black edges only and consistent scaling"""
        try:
            fig = plt.figure(figsize=(15, 5))
            
            # Original mesh
            ax1 = fig.add_subplot(121, projection='3d')
            # Plot only the edges in black
            ax1.plot_trisurf(self.vertices_3d[:, 0],
                            self.vertices_3d[:, 1],
                            self.vertices_3d[:, 2],
                            triangles=self.faces,
                            color='white',
                            edgecolor='black',
                            linewidth=0.5)
            ax1.set_title('Original 3D Mesh')
            
            all_ranges = [
                np.ptp(self.vertices_3d[:, 0]),
                np.ptp(self.vertices_3d[:, 1]),
                np.ptp(self.vertices_3d[:, 2]),
                np.ptp(self.vertices_2d[:, 0]),
                np.ptp(self.vertices_2d[:, 1])
            ]
            plot_radius = max(all_ranges) / 2
            
            center_3d = np.mean(self.vertices_3d, axis=0)
            center_2d = [np.mean(self.vertices_2d[:, 0]), np.mean(self.vertices_2d[:, 1])]
            
            # Set equal scaling for 3D plot
            ax1.set_xlim([center_3d[0] - plot_radius, center_3d[0] + plot_radius])
            ax1.set_ylim([center_3d[1] - plot_radius, center_3d[1] + plot_radius])
            ax1.set_zlim([center_3d[2] - plot_radius, center_3d[2] + plot_radius])
            
            # Adjust the viewing angle for better perspective
            ax1.view_init(elev=20, azim=-45)
            ax1.dist = 7
            
            # Remove grid and background color
            ax1.grid(False)
            ax1.xaxis.pane.fill = False
            ax1.yaxis.pane.fill = False
            ax1.zaxis.pane.fill = False
            
            # Flattened mesh
            ax2 = fig.add_subplot(122)
            # Plot only the edges in black
            ax2.triplot(self.vertices_2d[:, 0],
                       self.vertices_2d[:, 1],
                       self.faces,
                       color='black',
                       linewidth=0.5)
            
            # Set equal scaling for 2D plot
            ax2.set_xlim([center_2d[0] - plot_radius, center_2d[0] + plot_radius])
            ax2.set_ylim([center_2d[1] - plot_radius, center_2d[1] + plot_radius])
            
            ax2.set_title('Flattened 2D Mesh')
            ax2.set_aspect('equal')
            ax2.grid(False)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Visualization failed: {str(e)}")

 

    def visualize_color(self):
        """Visualize original and flattened meshes"""
        try:
            fig = plt.figure(figsize=(15, 5))
            
            # Original mesh
            ax1 = fig.add_subplot(121, projection='3d')
            surf = ax1.plot_trisurf(self.vertices_3d[:, 0],
                           self.vertices_3d[:, 1],
                           self.vertices_3d[:, 2],
                           triangles=self.faces,
                           cmap='viridis')
            ax1.set_title('Original 3D Mesh')
            
            # Set equal aspect ratio for all dimensions
            x_limits = ax1.get_xlim3d()
            y_limits = ax1.get_ylim3d()
            z_limits = ax1.get_zlim3d()
            x_range = abs(x_limits[1] - x_limits[0])
            x_middle = np.mean(x_limits)
            y_range = abs(y_limits[1] - y_limits[0])
            y_middle = np.mean(y_limits)
            z_range = abs(z_limits[1] - z_limits[0])
            z_middle = np.mean(z_limits)
            plot_radius = 0.5*max([x_range, y_range, z_range])
            
            ax1.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
            ax1.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
            ax1.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
            
            # Add a color bar
            fig.colorbar(surf, ax=ax1, label='Height')

            # Flattened mesh with height coloring
            ax2 = fig.add_subplot(122)
            heights = self.vertices_3d[:, 2]  # Use z-coordinate for coloring
            tri = ax2.tripcolor(self.vertices_2d[:, 0],
                              self.vertices_2d[:, 1],
                              self.faces,
                              heights,
                              cmap='viridis')
            plt.colorbar(tri, ax=ax2, label='Height')
            ax2.set_title('Flattened 2D Mesh')
            ax2.set_aspect('equal', adjustable='datalim')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Visualization failed: {str(e)}")


def process_mesh(input_path):
    start_time = time.time()
    print(f"Processing mesh: {input_path}")

    try:
        flattener = MeshFlattener(input_path)
        flattener.flatten_mesh()
        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")
        flattener.visualize()
        name = input_path.rsplit('.', 1)[0]
        black_pdf_path = f'{name}_black_contour.pdf'
        png_path = f'{name}.png'
        white_pdf_path = f'{name}_white_contours.pdf'
        output_path = f'{name}_2D.stl'
        flattener.save_2d_mesh_pdf(black_pdf_path)
        flattener.save_2d_mesh_image_filled(png_path)
        flattener.extract_and_plot_contours(png_path,white_pdf_path)
        flattened_mesh = trimesh.Trimesh(
            vertices=np.column_stack((flattener.vertices_2d, 
                                    np.zeros(flattener.n_vertices))),
            faces=flattener.faces)
        flattened_mesh.export(output_path)

 
        
    except Exception as e:
        print(f"Error processing mesh: {str(e)}")
        return None

if __name__ == "__main__":
    metrics = process_mesh("test_shapes/Bolus_chin.stl")
