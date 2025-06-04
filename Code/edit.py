#!/usr/bin/env python
# improved_stl_editor.py - Standalone 3D STL editing script

import sys
import re
import argparse
import numpy as np
import trimesh

# Global debug flag
DEBUG = False

def scale_mesh(mesh, factor, mode='uniform', specified_size=None):
    """
    Scale mesh with improved handling
    
    Parameters:
    - factor: either a single number (uniform scaling) or [x,y,z] factors
    - mode: 'uniform', 'absolute' (for specific size extensions), or 'axis' (for single-axis scaling)
    - specified_size: if mode is 'absolute', this is the target size in mm for specific dimensions
    """
    vertices = np.array(mesh.vertices)
    center = np.mean(vertices, axis=0)
    
    if mode == 'uniform':
        # Uniform scaling - all dimensions scaled equally
        if isinstance(factor, (int, float)):
            factor = [factor, factor, factor]
        
        scaled_vertices = np.zeros_like(vertices)
        for i in range(3):
            scaled_vertices[:, i] = (vertices[:, i] - center[i]) * factor[i] + center[i]
            
    elif mode == 'absolute':
        # Extend by absolute amount (e.g., "extend width by 1cm")
        # Here we add the specified amount to the dimension
        extents = mesh.extents  # [x_size, y_size, z_size]
        
        # Calculate scaling factor needed to achieve the target size
        scaling = [1.0, 1.0, 1.0]
        
        if specified_size is not None:
            axis_idx = specified_size[0]  # Which axis (0=x, 1=y, 2=z)
            target_size = specified_size[1]  # Target size in mm
            current_size = extents[axis_idx]
            
            # Calculate how much to scale to reach the target
            scaling[axis_idx] = target_size / current_size
            
            print(f"Extending axis {axis_idx} from {current_size:.2f}mm to {target_size:.2f}mm (scale factor: {scaling[axis_idx]:.4f})")
            
            # Apply the scaling to the vertices
            scaled_vertices = np.zeros_like(vertices)
            for i in range(3):
                scaled_vertices[:, i] = (vertices[:, i] - center[i]) * scaling[i] + center[i]
    
    elif mode == 'axis':
        # Scale single axis
        scaled_vertices = vertices.copy()
        axis_idx = factor[0]  # Which axis (0=x, 1=y, 2=z)
        scale_factor = factor[1]  # Scale factor for this axis
        
        scaled_vertices[:, axis_idx] = (vertices[:, axis_idx] - center[axis_idx]) * scale_factor + center[axis_idx]
    
    new_mesh = trimesh.Trimesh(vertices=scaled_vertices, faces=mesh.faces)
    return new_mesh

def rotate_mesh(mesh, angle_degrees, axis):
    """Rotate mesh around an axis by angle_degrees"""
    # Convert to radians
    angle_radians = np.radians(angle_degrees)
    
    # Normalize the axis vector
    axis = np.array(axis)
    if np.linalg.norm(axis) > 0:
        axis = axis / np.linalg.norm(axis)
    
    # Create rotation matrix
    rotation = trimesh.transformations.rotation_matrix(angle_radians, axis)
    
    # Apply rotation around center
    center = np.mean(mesh.vertices, axis=0)
    translate1 = trimesh.transformations.translation_matrix(-center)
    translate2 = trimesh.transformations.translation_matrix(center)
    transform = trimesh.transformations.concatenate_matrices(translate2, rotation, translate1)
    
    # Apply transformation
    new_mesh = mesh.copy()
    new_mesh.apply_transform(transform)
    return new_mesh

def translate_mesh(mesh, direction):
    """Translate mesh in the specified direction"""
    # Apply translation
    translation = trimesh.transformations.translation_matrix(direction)
    
    new_mesh = mesh.copy()
    new_mesh.apply_transform(translation)
    return new_mesh

def extrude_face(mesh, direction, distance):
    """
    Extrude a face of the mesh in the specified direction by the given distance
    
    Parameters:
    - direction: 'front', 'back', 'left', 'right', 'top', 'bottom'
    - distance: distance to extrude in mm
    """
    # Define direction vectors for each face
    direction_vectors = {
        'front': [0, 0, 1],    # +Z direction
        'back': [0, 0, -1],    # -Z direction
        'left': [-1, 0, 0],    # -X direction
        'right': [1, 0, 0],    # +X direction
        'top': [0, 1, 0],      # +Y direction
        'bottom': [0, -1, 0],  # -Y direction
    }
    
    # Get direction vector
    if direction not in direction_vectors:
        raise ValueError(f"Unknown direction: {direction}. Must be one of: {', '.join(direction_vectors.keys())}")
    
    dir_vector = np.array(direction_vectors[direction])
    
    # Get the vertices and faces
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    
    # Get the bounds of the mesh
    bounds = mesh.bounds
    min_bound = bounds[0]
    max_bound = bounds[1]
    
    # Determine which vertices are on the face to extrude
    # We use a larger tolerance to better capture vertices on the face
    tolerance = 0.1  # Increased tolerance to catch more vertices
    face_mask = None
    
    # Determine which coordinate corresponds to which axis based on mesh orientation
    # This is to ensure we're selecting the correct face regardless of model orientation
    if direction == 'front':
        face_mask = np.abs(vertices[:, 2] - max_bound[2]) < tolerance
    elif direction == 'back':
        face_mask = np.abs(vertices[:, 2] - min_bound[2]) < tolerance
    elif direction == 'right':
        face_mask = np.abs(vertices[:, 0] - max_bound[0]) < tolerance
    elif direction == 'left':
        face_mask = np.abs(vertices[:, 0] - min_bound[0]) < tolerance
    elif direction == 'top':
        face_mask = np.abs(vertices[:, 1] - max_bound[1]) < tolerance
    elif direction == 'bottom':
        face_mask = np.abs(vertices[:, 1] - min_bound[1]) < tolerance
    
    # Check if we found any vertices to extrude
    if face_mask is None or not np.any(face_mask):
        print(f"Warning: No vertices found on {direction} face. Trying alternative detection method...")
        
        # Alternative method - try detecting using faces
        # Find the faces that have all vertices near the bound
        face_vertices = vertices[faces]
        if direction == 'front':
            # Check if max z-coordinate of each face is close to the max bound
            face_z_max = np.max(face_vertices[:, :, 2], axis=1)
            face_mask = np.abs(face_z_max - max_bound[2]) < tolerance
            vertices_to_move = np.unique(faces[face_mask].flatten())
        elif direction == 'back':
            face_z_min = np.min(face_vertices[:, :, 2], axis=1)
            face_mask = np.abs(face_z_min - min_bound[2]) < tolerance
            vertices_to_move = np.unique(faces[face_mask].flatten())
        elif direction == 'right':
            face_x_max = np.max(face_vertices[:, :, 0], axis=1)
            face_mask = np.abs(face_x_max - max_bound[0]) < tolerance
            vertices_to_move = np.unique(faces[face_mask].flatten())
        elif direction == 'left':
            face_x_min = np.min(face_vertices[:, :, 0], axis=1)
            face_mask = np.abs(face_x_min - min_bound[0]) < tolerance
            vertices_to_move = np.unique(faces[face_mask].flatten())
        elif direction == 'top':
            face_y_max = np.max(face_vertices[:, :, 1], axis=1)
            face_mask = np.abs(face_y_max - max_bound[1]) < tolerance
            vertices_to_move = np.unique(faces[face_mask].flatten())
        elif direction == 'bottom':
            face_y_min = np.min(face_vertices[:, :, 1], axis=1)
            face_mask = np.abs(face_y_min - min_bound[1]) < tolerance
            vertices_to_move = np.unique(faces[face_mask].flatten())
        
        # Create a mask for the vertices we need to move
        new_mask = np.zeros(len(vertices), dtype=bool)
        new_mask[vertices_to_move] = True
        face_mask = new_mask
    
    # Debug info
    print(f"Extruding {direction} face by {distance}mm")
    print(f"Found {np.sum(face_mask)} vertices to extrude")
    
    # Move the vertices on the face
    new_vertices = vertices.copy()
    new_vertices[face_mask] += dir_vector * distance
    
    # Create a new mesh with the new vertices
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=faces)
    return new_mesh

def process_command(cmd, mesh):
    """Process a natural language command and apply it to the mesh"""
    cmd = cmd.lower().strip()
    operation = None
    
    # Scale/size operations
    if re.search(r'(scale|resize|extend|enlarge|reduce|shrink)', cmd):
        # Extract scale factor or dimensions
        scale_match = re.search(r'(by|to)\s+(\d+(\.\d+)?)\s*(cm|mm|percent|%)?', cmd)
        
        if scale_match:
            factor = float(scale_match.group(2))
            unit = scale_match.group(4) or 'cm'
            
            # Check for directional scaling
            is_directional = False
            axis_idx = None
            
            # First check for explicit dimension terms before checking for axis names
            if 'width' in cmd:
                axis_idx = 0
                is_directional = True
                print("Detected width extension - using X axis (0)")
            elif 'height' in cmd:
                axis_idx = 1
                is_directional = True
                print("Detected height extension - using Y axis (1)")
            elif 'depth' in cmd:
                axis_idx = 2
                is_directional = True
                print("Detected depth extension - using Z axis (2)")
            # If no dimension terms, check for axis names
            elif 'x' in cmd or 'horizontal' in cmd:
                axis_idx = 0
                is_directional = True
                print("Detected X-axis extension (0)")
            elif 'y' in cmd or 'vertical' in cmd:
                axis_idx = 1
                is_directional = True
                print("Detected Y-axis extension (1)")
            elif 'z' in cmd:
                axis_idx = 2
                is_directional = True
                print("Detected Z-axis extension (2)")
            
            # Handle "extend by X" differently than "scale by X"
            if re.search(r'(extend|enlarge)', cmd) and is_directional:
                # Convert to mm if necessary
                if unit == 'cm':
                    factor *= 10
                
                # Get current size of that dimension
                current_size = mesh.extents[axis_idx]
                # Target size = current + extension
                target_size = current_size + factor
                
                # Scale using absolute mode
                mesh = scale_mesh(mesh, None, mode='absolute', specified_size=(axis_idx, target_size))
                operation = f"Extended {['width', 'height', 'depth'][axis_idx]} from {current_size:.2f}mm to {target_size:.2f}mm"
                
            else:
                # Regular scaling operation
                if unit in ['percent', '%']:
                    factor = factor / 100
                
                # Apply different scaling based on command type
                if re.search(r'(reduce|shrink)', cmd):
                    if factor > 1:  # If they say "reduce by 50%" they likely mean reduce TO 50%
                        factor = 1 / factor
                    else:
                        factor = factor
                
                if is_directional:
                    # Scale just one axis
                    mesh = scale_mesh(mesh, (axis_idx, factor), mode='axis')
                    axis_name = ['width', 'height', 'depth'][axis_idx]
                    operation = f"Scaled {axis_name} by factor {factor:.4f}"
                else:
                    # Uniform scaling
                    mesh = scale_mesh(mesh, factor, mode='uniform')
                    operation = f"Scaled model uniformly by factor {factor:.4f}"
    
    # Rotation operations
    elif re.search(r'rotate|turn|spin', cmd):
        angle_match = re.search(r'(\d+)\s*degrees', cmd)
        if angle_match:
            angle = float(angle_match.group(1))
            
            # Determine axis
            axis = [0, 0, 1]  # Default to z-axis
            if any(x_term in cmd for x_term in ['x', 'horizontal']):
                axis = [1, 0, 0]
            elif any(y_term in cmd for y_term in ['y', 'vertical']):
                axis = [0, 1, 0]
            
            # Determine direction
            if 'clockwise' in cmd or 'right' in cmd:
                angle = -angle
            
            mesh = rotate_mesh(mesh, angle, axis)
            axis_name = 'z' if axis == [0, 0, 1] else ('x' if axis == [1, 0, 0] else 'y')
            operation = f"Rotated model {angle} degrees around {axis_name}-axis"
    
    # Translation operations
    elif re.search(r'move|shift|translate|position', cmd):
        dist_match = re.search(r'(\d+(\.\d+)?)\s*(cm|mm)', cmd)
        if dist_match:
            distance = float(dist_match.group(1))
            unit = dist_match.group(3)
            
            # Convert to mm if necessary
            if unit == 'cm':
                distance *= 10
            
            # Determine direction
            direction = [0, 0, 0]
            if 'up' in cmd:
                direction[1] = distance
            elif 'down' in cmd:
                direction[1] = -distance
            elif 'left' in cmd:
                direction[0] = -distance
            elif 'right' in cmd:
                direction[0] = distance
            elif 'forward' in cmd or 'front' in cmd:
                direction[2] = distance
            elif 'back' in cmd or 'backward' in cmd:
                direction[2] = -distance
            
            mesh = translate_mesh(mesh, direction)
            dir_name = "x" if direction[0] != 0 else ("y" if direction[1] != 0 else "z")
            dir_value = direction[0] if direction[0] != 0 else (direction[1] if direction[1] != 0 else direction[2])
            operation = f"Moved model {abs(dir_value)}{unit} along {dir_name}-axis"
    
    # Extrusion operations - NEW FUNCTIONALITY
    elif re.search(r'extrude', cmd):
        dist_match = re.search(r'(\d+(\.\d+)?)\s*(cm|mm)', cmd)
        if dist_match:
            distance = float(dist_match.group(1))
            unit = dist_match.group(3) or 'cm'  # Default to cm if not specified
            
            # Convert to mm if necessary
            if unit == 'cm':
                distance *= 10
                
            print(f"Extrusion distance: {distance}mm")
            
            # Determine which face to extrude
            face = None
            # Check for explicit face names
            for face_name in ['front', 'back', 'left', 'right', 'top', 'bottom']:
                if face_name in cmd:
                    face = face_name
                    break
            
            # Additional checks for common terms
            if face is None:
                if any(term in cmd for term in ['forward', 'ahead']):
                    face = 'front'
                elif any(term in cmd for term in ['rear', 'backward', 'behind']):
                    face = 'back'
                elif 'side' in cmd:
                    if 'left' in cmd:
                        face = 'left'
                    elif 'right' in cmd:
                        face = 'right'
                    else:
                        # Default to right if just "side" is mentioned
                        face = 'right'
                elif 'upper' in cmd or 'above' in cmd:
                    face = 'top'
                elif 'lower' in cmd or 'below' in cmd or 'under' in cmd:
                    face = 'bottom'
            
            if face:
                try:
                    # Print the mesh bounds for debugging
                    if DEBUG:
                        print(f"Mesh bounds before extrusion:")
                        print(f"Min: {mesh.bounds[0]}")
                        print(f"Max: {mesh.bounds[1]}")
                        print(f"Extents: {mesh.extents}")
                    
                    mesh = extrude_face(mesh, face, distance)
                    operation = f"Extruded {face} face by {distance}mm"
                    
                    # Print the mesh bounds after extrusion for debugging
                    if DEBUG:
                        print(f"Mesh bounds after extrusion:")
                        print(f"Min: {mesh.bounds[0]}")
                        print(f"Max: {mesh.bounds[1]}")
                        print(f"Extents: {mesh.extents}")
                except Exception as e:
                    print(f"⚠️ Extrusion failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return mesh, False
            else:
                print("⚠️ Please specify which face to extrude (e.g., 'extrude front by 1cm')")
                return mesh, False
    
    else:
        print("⚠️ Unsupported command. Try operations like 'scale by 50%', 'extend width by 1cm', 'rotate 90 degrees', 'move up 2cm', or 'extrude front by 1cm'")
        return mesh, False
    
    print(f"✅ {operation}")
    return mesh, True

def main():
    parser = argparse.ArgumentParser(description='Process STL files with natural language commands')
    parser.add_argument('input_stl', help='Input STL file path')
    parser.add_argument('output_stl', help='Output STL file path')
    parser.add_argument('--command', '-c', help='Natural language edit command')
    parser.add_argument('--info', action='store_true', help='Display mesh information')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    # Set debug flag
    global DEBUG
    DEBUG = args.debug
    
    try:
        # Load the mesh
        print(f"Loading mesh from {args.input_stl}...")
        mesh = trimesh.load_mesh(args.input_stl)
        
        # Show mesh information if requested
        if args.info:
            print("\nMesh Information:")
            print(f"- Vertices: {len(mesh.vertices)}")
            print(f"- Faces: {len(mesh.faces)}")
            print(f"- Size (mm): {mesh.extents[0]:.2f} x {mesh.extents[1]:.2f} x {mesh.extents[2]:.2f}")
            print(f"- Bounding box center: [{mesh.centroid[0]:.2f}, {mesh.centroid[1]:.2f}, {mesh.centroid[2]:.2f}]")
            print(f"- Bounding box min: [{mesh.bounds[0][0]:.2f}, {mesh.bounds[0][1]:.2f}, {mesh.bounds[0][2]:.2f}]")
            print(f"- Bounding box max: [{mesh.bounds[1][0]:.2f}, {mesh.bounds[1][1]:.2f}, {mesh.bounds[1][2]:.2f}]")
        
        if args.command:
            # Process command from command line argument
            mesh, success = process_command(args.command, mesh)
            if not success:
                return 1
        else:
            # Interactive mode
            print("\nSTL Editor - Enter commands or 'exit' to quit")
            print("Examples: 'scale by 150%', 'extend width by 1cm', 'extend height by 2cm', 'rotate 90 degrees', 'move up 2cm', 'extrude front by 1cm'")
            
            while True:
                cmd = input("\nEnter command: ")
                if cmd.lower() in ['exit', 'quit', 'q']:
                    break
                elif cmd.lower() == 'info':
                    print(f"\nCurrent mesh size (mm): {mesh.extents[0]:.2f} x {mesh.extents[1]:.2f} x {mesh.extents[2]:.2f}")
                    print(f"Bounding box min: [{mesh.bounds[0][0]:.2f}, {mesh.bounds[0][1]:.2f}, {mesh.bounds[0][2]:.2f}]")
                    print(f"Bounding box max: [{mesh.bounds[1][0]:.2f}, {mesh.bounds[1][1]:.2f}, {mesh.bounds[1][2]:.2f}]")
                    continue
                elif cmd.lower() == 'axis info':
                    print("\nCoordinate system explanation:")
                    print("X-axis (0): Width - Left to Right")
                    print("Y-axis (1): Height - Bottom to Top")
                    print("Z-axis (2): Depth - Back to Front")
                    continue
                
                mesh, _ = process_command(cmd, mesh)
        
        # Save the mesh
        mesh.export(args.output_stl)
        print(f"Mesh saved to {args.output_stl}")
        return 0
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())