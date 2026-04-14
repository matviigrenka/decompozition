import argparse
import sys
from typing import List

import bpy
import numpy as np


PALETTE = np.array(
    [
        [231, 76, 60],
        [52, 152, 219],
        [46, 204, 113],
        [241, 196, 15],
        [155, 89, 182],
        [230, 126, 34],
        [26, 188, 156],
        [149, 165, 166],
        [52, 73, 94],
        [243, 156, 18],
        [127, 140, 141],
        [192, 57, 43],
    ],
    dtype=np.float32,
) / 255.0


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Project segmentation labels onto a Blender mesh.")
    parser.add_argument("--seg", required=True, help="NPZ file created by inference.py via --blender-output")
    parser.add_argument("--object", default=None, help="Existing Blender object name. Uses active object if omitted.")
    parser.add_argument("--mesh", default=None, help="Optional OBJ file to import when no mesh object is loaded.")
    parser.add_argument("--separate", action="store_true", help="Separate the mesh into multiple objects by predicted cluster.")
    parser.add_argument("--vertex-groups", action="store_true", help="Create a vertex group per cluster.")
    parser.add_argument("--material-prefix", default="Part", help="Prefix for generated materials.")
    parser.add_argument("--output-blend", default=None, help="Optional .blend path to save after processing.")
    return parser.parse_args(argv)


def import_mesh_if_needed(mesh_path: str):
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=mesh_path)
        return bpy.context.selected_objects[-1]
    bpy.ops.import_scene.obj(filepath=mesh_path)
    return bpy.context.selected_objects[-1]


def get_target_object(args: argparse.Namespace):
    if args.object is not None:
        obj = bpy.data.objects.get(args.object)
        if obj is None:
            raise ValueError(f"Object '{args.object}' not found")
        return obj
    if bpy.context.active_object is not None and bpy.context.active_object.type == "MESH":
        return bpy.context.active_object
    if args.mesh is not None:
        return import_mesh_if_needed(args.mesh)
    raise ValueError("No mesh object available. Select a mesh in Blender or pass --mesh path/to/model.obj")


def chunked_nearest_labels(vertices: np.ndarray, sampled_points: np.ndarray, sampled_labels: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    labels = np.empty(vertices.shape[0], dtype=np.int32)
    for start in range(0, vertices.shape[0], chunk_size):
        end = min(start + chunk_size, vertices.shape[0])
        chunk = vertices[start:end]
        distances = np.sum((chunk[:, None, :] - sampled_points[None, :, :]) ** 2, axis=-1)
        nearest_ids = np.argmin(distances, axis=1)
        labels[start:end] = sampled_labels[nearest_ids]
    return labels


def ensure_materials(obj, num_clusters: int, prefix: str) -> List[bpy.types.Material]:
    materials = []
    obj.data.materials.clear()
    for label in range(num_clusters):
        mat = bpy.data.materials.new(name=f"{prefix}_{label}")
        mat.use_nodes = True
        principled = mat.node_tree.nodes.get("Principled BSDF")
        color = PALETTE[label % len(PALETTE)]
        if principled is not None:
            principled.inputs["Base Color"].default_value = (float(color[0]), float(color[1]), float(color[2]), 1.0)
        obj.data.materials.append(mat)
        materials.append(mat)
    return materials


def assign_vertex_groups(obj, vertex_labels: np.ndarray, num_clusters: int) -> None:
    for group in list(obj.vertex_groups):
        if group.name.startswith("cluster_"):
            obj.vertex_groups.remove(group)
    for label in range(num_clusters):
        group = obj.vertex_groups.new(name=f"cluster_{label}")
        indices = np.where(vertex_labels == label)[0].tolist()
        if indices:
            group.add(indices, 1.0, "REPLACE")


def assign_materials_and_colors(obj, vertex_labels: np.ndarray, num_clusters: int, prefix: str) -> None:
    ensure_materials(obj, num_clusters, prefix)
    mesh = obj.data

    if hasattr(mesh, "color_attributes"):
        color_layer = mesh.color_attributes.get("part_colors")
        if color_layer is None:
            color_layer = mesh.color_attributes.new(name="part_colors", type="BYTE_COLOR", domain="CORNER")
    else:
        color_layer = mesh.vertex_colors.get("part_colors")
        if color_layer is None:
            color_layer = mesh.vertex_colors.new(name="part_colors")

    for poly in mesh.polygons:
        face_labels = vertex_labels[np.array(poly.vertices, dtype=np.int64)]
        bincount = np.bincount(face_labels, minlength=num_clusters)
        face_label = int(np.argmax(bincount))
        poly.material_index = face_label
        color = PALETTE[face_label % len(PALETTE)]
        for loop_index in poly.loop_indices:
            color_layer.data[loop_index].color = (float(color[0]), float(color[1]), float(color[2]), 1.0)


def separate_by_material(obj) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="MATERIAL")
    bpy.ops.object.mode_set(mode="OBJECT")


def save_blend(output_blend: str) -> None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blend)


def main() -> None:
    args = parse_args()
    data = np.load(args.seg)
    sampled_points = data["sampled_points"].astype(np.float32)
    sampled_labels = data["labels"].astype(np.int32)
    num_clusters = int(sampled_labels.max()) + 1 if sampled_labels.size > 0 else 0
    if num_clusters == 0:
        raise ValueError("Segmentation file contains no labels")

    obj = get_target_object(args)
    if obj.type != "MESH":
        raise ValueError(f"Target object '{obj.name}' is not a mesh")

    vertices = np.array([vertex.co[:] for vertex in obj.data.vertices], dtype=np.float32)
    vertex_labels = chunked_nearest_labels(vertices, sampled_points, sampled_labels)

    assign_materials_and_colors(obj, vertex_labels, num_clusters, args.material_prefix)
    if args.vertex_groups:
        assign_vertex_groups(obj, vertex_labels, num_clusters)
    if args.separate:
        separate_by_material(obj)
    if args.output_blend:
        save_blend(args.output_blend)
        print(f"Saved Blender scene to {args.output_blend}")

    print(f"Applied segmentation to Blender object '{obj.name}' with {num_clusters} clusters")


if __name__ == "__main__":
    main()
