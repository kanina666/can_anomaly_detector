BBOX = {
    "first_pos": (980, 2100, 350, 1480),
    "second_pos": (950, 2060, 180, 1150),
}

EXCLUDE_ZONES = {
    "first_pos": [
        (350, 700, "right", 1950),
        (700, 1100, "right", 2030),
    ],
    "second_pos": [
        (180, 600, "right", 1950),
        (180, 650, "left", 1100),
        (650, 950, "left", 1060),
    ],
}


def bbox_for(folder):
    return BBOX["first_pos"] if folder.startswith("first_pos") else BBOX["second_pos"]


def zones_for(folder):
    return EXCLUDE_ZONES["first_pos"] if folder.startswith("first_pos") else EXCLUDE_ZONES["second_pos"]


def in_matrix(x, y, folder):
    X0, X1, Y0, Y1 = bbox_for(folder)
    if not (X0 <= x <= X1 and Y0 <= y <= Y1):
        return False
    for y0, y1, side, cutoff in zones_for(folder):
        if y0 <= y < y1:
            if side == "right" and x > cutoff:
                return False
            if side == "left" and x < cutoff:
                return False
    return True
