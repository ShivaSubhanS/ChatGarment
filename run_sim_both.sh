#!/bin/bash

cd /home/sss/project/pose_3d/ChatGarment

# Dynamically find all pattern JSON files
echo "=== Finding generated patterns ==="
json_files=(garment_patterns/valid_garment_*/valid_garment_*/*_specification.json)

if [ ! -e "${json_files[0]}" ]; then
    echo "❌ No pattern files found in garment_patterns/"
    echo "   Please ensure you have extracted the patterns from Kaggle"
    exit 1
fi

echo "Found ${#json_files[@]} pattern(s)"
echo ""

# Run simulation for each garment
for json_file in "${json_files[@]}"; do
    garment_name=$(basename $(dirname "$json_file"))
    garment_type=$(echo "$garment_name" | grep -o "upper\|lower")
    
    echo "=== Running 3D simulation for $garment_type garment ==="
    python run_garmentcode_sim.py --json_spec_file "$json_file"
    
    if [ $? -eq 0 ]; then
        echo "✓ $garment_type garment completed"
    else
        echo "⚠ $garment_type garment failed"
    fi
    echo ""
done

echo "=== Simulation complete! ==="
echo "OBJ files generated in:"
find garment_patterns -name "*_sim.obj" -type f | while read obj_file; do
    echo "  - $obj_file"
done
