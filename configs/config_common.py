

motion_presets = {
    "regular": ["fwd", "side", "updown", "xy", "yz", "xz", "xyz", "xy_v2", "yz_v2", "xz_v2", "xyz_v2", "xyz_v3", "xyz_v4"],
    "fancy": ["demo_speed", "bouncy", "rotate_bounce", "spiral", "spiral_upward"],  # removed snake because it is too long
    "full": ["fwd", "side", "updown", "xy", "yz", "xz", "xyz", "xy_v2", "yz_v2", "xz_v2", "xyz_v2", "xyz_v3", "xyz_v4", "demo_speed", "bouncy", "rotate_bounce", "spiral", "spiral_upward"],
    "debug": ["fwd"]
    # "debug": ["bouncy"]
}

scene_sets = {
    "real": [f"scenes/R{i}.json" for i in range(1, 10)],
    "synthetic": [f"scenes/S{i}_res32_synthetic.json" for i in range(1, 5)]
}