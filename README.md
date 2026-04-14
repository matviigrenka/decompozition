# 3D Decomposition Pipeline

## What This Project Does
This project trains an unsupervised 3D segmentation model on meshes or point clouds, predicts part clusters for a new `.obj`, and can project the result back into Blender as colored parts, vertex groups, or separate objects.

## Main Files
- `train.py`: trains the embedding model
- `inference.py`: predicts point-level segmentation from a trained checkpoint
- `blender_decompose.py`: projects predicted labels onto a Blender mesh
- `process_obj_to_blend.py`: full pipeline from `.obj` and `.pt` to `.blend`
- `run_obj_to_blend.sh`: bash wrapper around `process_obj_to_blend.py`
- `prepare_datasets.py`: prepares datasets into unified `.npz` format

## Installation
CPU:
```bash
pip install -r requirements.txt
```

RTX 4060 / CUDA:
```bash
pip install -r requirements-rtx4060.txt
```

## Training
Example:
```bash
python train.py \
  --data-dir unified_datasets/modelnet40 \
  --epochs 20 \
  --batch-size 4 \
  --num-points 2048 \
  --num-clusters 6 \
  --use-normals \
  --device auto \
  --num-workers 2 \
  --amp \
  --save-path checkpoint.pt
```

## Inference
Example:
```bash
python inference.py \
  --input path/to/model.obj \
  --checkpoint checkpoint.pt \
  --num-points 2048 \
  --num-clusters 6 \
  --use-normals \
  --device auto \
  --output segmented_output.ply
```

This creates:
- `segmented_output.ply`: colored point cloud
- `segmented_output_blender.npz`: Blender projection data

## Full OBJ -> BLEND Pipeline
### Python entrypoint
```bash
python process_obj_to_blend.py \
  --input path/to/model.obj \
  --checkpoint path/to/checkpoint.pt \
  --blender "/path/to/blender" \
  --output-blend path/to/result.blend \
  --num-points 2048 \
  --num-clusters 6 \
  --use-normals \
  --device auto \
  --vertex-groups \
  --separate
```

### Bash wrapper
```bash
chmod +x run_obj_to_blend.sh
./run_obj_to_blend.sh \
  path/to/model.obj \
  path/to/checkpoint.pt \
  /path/to/blender \
  path/to/result.blend \
  --num-points 2048 \
  --num-clusters 6 \
  --use-normals \
  --device auto \
  --vertex-groups \
  --separate
```

## Parameters You Will Use Most
- `--num-points`: number of sampled points for inference/training
- `--num-clusters`: number of part clusters in the final decomposition
- `--use-normals`: include normals in model input
- `--device`: `auto`, `cuda`, or `cpu`
- `--vertex-groups`: create a Blender vertex group per cluster
- `--separate`: split the mesh into multiple Blender objects by cluster

## Blender Output Behavior
`process_obj_to_blend.py` runs Blender in background mode and saves a `.blend` file.

If `--separate` is enabled:
- the mesh is split into multiple objects by predicted part

If `--vertex-groups` is enabled:
- Blender vertex groups `cluster_0`, `cluster_1`, ... are created

## Notes
- The model predicts labels for sampled points, then Blender projects them back to mesh vertices using nearest sampled point assignment.
- For very dense meshes, increasing `--num-points` may improve projection quality.
- For noisy or over-segmented results, reduce `--num-clusters`.
