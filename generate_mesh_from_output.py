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
if '/home/sss/project/pose_3d/NvidiaWarp-GarmentCode' not in sys.path:
    sys.path.insert(0, '/home/sss/project/pose_3d/NvidiaWarp-GarmentCode')

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
    
    # The output can be in two formats:
    # 1. Single garment: {'meta': ...}
    # 2. Multiple garments (list): [{'meta': ...}, {'meta': ...}]
    
    # Look for the LAST occurrence (after ASSISTANT:) of either format
    # First try to find list format
    list_start = content.rfind('[{')
    
    if list_start != -1:
        # Found list format - extract it
        bracket_count = 0
        brace_count = 0
        end_idx = list_start
        
        for i, char in enumerate(content[list_start:], start=list_start):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    end_idx = i + 1
                    break
            elif char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
        
        json_str = content[list_start:end_idx]
    else:
        # Single garment - look for {'meta':
        start_idx = content.rfind("{'meta':")
        if start_idx == -1:
            start_idx = content.rfind('{"meta":')
        
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
        print("Extracted string (first 300 chars):", json_str[:300])
        print("Extracted string (last 100 chars):", json_str[-100:])
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
    
    # Check if it's a list of garments (upperbody + lowerbody) or single garment
    garments_to_process = []
    if isinstance(garment_spec, list):
        print(f"✓ Extracted {len(garment_spec)} separate garments")
        for i, spec in enumerate(garment_spec):
            print(f"  Garment {i+1} keys: {list(spec.keys())}")
            # Determine garment type from meta
            if 'meta' in spec:
                meta = spec['meta']
                if meta.get('upper') and not meta.get('bottom'):
                    garment_type = 'upperbody'
                elif meta.get('bottom') and not meta.get('upper'):
                    garment_type = 'lowerbody'
                else:
                    garment_type = 'wholebody'
                garments_to_process.append((spec, garment_type))
    else:
        print(f"✓ Extracted garment specification")
        print(f"  Keys: {list(garment_spec.keys())}")
        garments_to_process.append((garment_spec, 'wholebody'))
    
    # Step 2: Generate 2D patterns for all garments
    print(f"\nStep 2: Generating 2D sewing patterns for {len(garments_to_process)} garment(s)...")
    spec_files = []
    for spec, garment_type in garments_to_process:
        spec_file = generate_mesh_from_spec(spec, args.output_dir, garment_type)
        if spec_file:
            spec_files.append(spec_file)
        else:
            print(f"✗ Failed to generate 2D pattern for {garment_type}")
    
    if not spec_files:
        print("✗ Failed to generate any 2D patterns")
        return
    
    # Step 3: Run simulation (optional)
    if not args.skip_simulation:
        print(f"\nStep 3: Running 3D cloth simulation for {len(spec_files)} garment(s)...")
        mesh_files = []
        for spec_file in spec_files:
            mesh_file = run_simulation(spec_file, args.output_dir)
            if mesh_file:
                mesh_files.append(mesh_file)
        
        if mesh_files:
            print("\n" + "="*60)
            print(f"✓ SUCCESS! Generated {len(mesh_files)} mesh(es)!")
            print("="*60)
            print(f"\n📁 Output files:")
            for i, mesh_file in enumerate(mesh_files):
                print(f"  Garment {i+1}:")
                print(f"    - 2D Pattern: {os.path.dirname(spec_files[i])}")
                print(f"    - 3D Mesh: {mesh_file}")
            print(f"\n💡 Import the .obj files into Blender!")
        else:
            print("\n✗ All simulations failed, but 2D patterns are available")
    else:
        print("\n⏭ Skipping simulation (--skip_simulation flag set)")
        for spec_file in spec_files:
            print(f"📁 2D pattern saved: {spec_file}")


if __name__ == "__main__":
    main()
