for pattern in color_size size_grid color_grid color_hexagon color_overlap_squares polygon_sides_color rectangle_height_color shape_reflect shape_size_grid size_cycle ; do
    python gen_data/data_generation.py create_data $pattern example_data --limit 1 --seed 42 --target_size "(1280, 704)"
done