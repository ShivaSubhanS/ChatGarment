#!/usr/bin/env python3
"""
Generate 3D mesh from ChatGarment output.txt locally
"""
import os
import sys
import json
import re
from pathlib import Path

# Add custom Warp fork to path
if '/tmp/NvidiaWarp-GarmentCode' not in sys.path:
    sys.path.insert(0, '/tmp/NvidiaWarp-GarmentCode')

# Add GarmentCodeRC to path
garmentcode_path = '/home/sss/project/pose_3d/GarmentCodeRC/'
if garmentcode_path not in sys.path:
    sys.path.insert(1, garmentcode_path)

# Import necessary modules
from llava.garment_utils_v2 import try_generate_garments

def extract_json_from_output(output_txt_path):
    """Extract the JSON garment specification from output.txt"""
    with open(output_txt_path, 'r') as f:
        content = f.read()
    
    # Find the JSON portion (look for the dict after "ASSISTANT:")
    # The model outputs something like: wholebody_garment': {...}
    
    # Try to find the complete JSON structure
    # Look for patterns like {'meta': ... or "meta":
    start_idx = content.find("{'meta':")
    if start_idx == -1:
        start_idx = content.find('{"meta":')
    
    if start_idx == -1:
        print("ERROR: Could not find garment specification in output.txt")
        print("Content preview:", content[:500])
        return None
    
    # Extract from the first '{' to the matching '}'
    brace_count = 0
    end_idx = start_idx
    for i, char in enumerate(content[start_idx:], start=start_idx):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break
    
    json_str = content[start_idx:end_idx]
    
    # Clean up the JSON string
    json_str = json_str.replace("'", '"')  # Python dict to JSON
    json_str = json_str.replace('True', 'true')
    json_str = json_str.replace('False', 'false')
    json_str = json_str.replace('None', 'null')
    
    try:
        garment_spec = json.loads(json_str)
        return garment_spec
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON: {e}")
        print("Extracted string:", json_str[:200])
        return None


def generate_mesh_from_spec(garment_spec, output_dir, garment_type='wholebody'):
    """Generate 3D mesh from garment specification"""
    print(f"\n{'='*60}")
    print(f"Generating {garment_type} garment...")
    print(f"{'='*60}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate 2D pattern and JSON spec
    try:
        try_generate_garments(
            None, 
            garment_spec, 
            garment_type, 
            output_dir, 
            invnorm_float=True, 
            float_dict=None
        )
        print(f"✓ 2D pattern generated successfully")
        
        # Find the generated specification file
        spec_file = os.path.join(
            output_dir, 
            f'valid_garment_{garment_type}', 
            f'valid_garment_{garment_type}_specification.json'
        )
        
        if os.path.exists(spec_file):
            print(f"✓ Specification saved: {spec_file}")
            return spec_file
        else:
            print(f"✗ Specification file not found: {spec_file}")
            return None
            
    except Exception as e:
        print(f"✗ Error generating pattern: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_simulation(spec_file, output_dir):
    """Run cloth simulation to generate 3D mesh"""
    print(f"\n{'='*60}")
    print(f"Running cloth simulation...")
    print(f"{'='*60}\n")
    
    import subprocess
    
    cmd = [
        'python', 'run_garmentcode_sim.py',
        '--json_spec_file', spec_file
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Simulation completed successfully!")
        
        # Find the generated mesh
        garment_name = Path(spec_file).stem.replace('_specification', '')
        mesh_file = os.path.join(
            os.path.dirname(spec_file),
            garment_name,
            f'{garment_name}_sim.obj'
        )
        
        if os.path.exists(mesh_file):
            print(f"✓ 3D mesh saved: {mesh_file}")
            return mesh_file
        else:
            print(f"⚠ Mesh file not found: {mesh_file}")
            return None
    else:
        print(f"✗ Simulation failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate 3D mesh from ChatGarment output.txt')
    parser.add_argument('--output_txt', type=str, required=True,
                       help='Path to output.txt from ChatGarment inference')
    parser.add_argument('--output_dir', type=str, default='./local_mesh_output',
                       help='Directory to save generated meshes')
    parser.add_argument('--skip_simulation', action='store_true',
                       help='Only generate 2D pattern, skip 3D simulation')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("ChatGarment Local Mesh Generation")
    print("="*60 + "\n")
    
    # Step 1: Extract garment specification
    print("Step 1: Extracting garment specification from output.txt...")
    garment_spec = extract_json_from_output(args.output_txt)
    
    if garment_spec is None:
        print("✗ Failed to extract garment specification")
        return
    
    print(f"✓ Extracted garment specification")
    print(f"  Keys: {list(garment_spec.keys())}")
    
    # Step 2: Generate 2D pattern
    print("\nStep 2: Generating 2D sewing pattern...")
    spec_file = generate_mesh_from_spec(garment_spec, args.output_dir)
    
    if spec_file is None:
        print("✗ Failed to generate 2D pattern")
        return
    
    # Step 3: Run simulation (optional)
    if not args.skip_simulation:
        print("\nStep 3: Running 3D cloth simulation...")
        mesh_file = run_simulation(spec_file, args.output_dir)
        
        if mesh_file:
            print("\n" + "="*60)
            print("✓ SUCCESS! Mesh generated successfully!")
            print("="*60)
            print(f"\n📁 Output files:")
            print(f"  - 2D Pattern: {os.path.dirname(spec_file)}")
            print(f"  - 3D Mesh: {mesh_file}")
            print(f"\n💡 Import {mesh_file} into Blender!")
        else:
            print("\n✗ Simulation failed, but 2D pattern is available")
    else:
        print("\n⏭ Skipping simulation (--skip_simulation flag set)")
        print(f"📁 2D pattern saved: {spec_file}")


if __name__ == "__main__":
    main()
