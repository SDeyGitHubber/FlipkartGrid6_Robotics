def get_color_name(r, g, b):
    if r > 200 and g < 100 and b < 100:
        return 'Red'
    elif r < 100 and g > 200 and b < 100:
        return 'Green'
    elif r < 100 and g < 100 and b > 200:
        return 'Blue'
    elif r > 200 and g > 200 and b < 100:
        return 'Yellow'
    elif r > 200 and g < 100 and b > 200:
        return 'Magenta'
    elif r < 100 and g > 200 and b > 200:
        return 'Cyan'
    elif r > 200 and g > 200 and b > 200:
        return 'White'
    elif r < 50 and g < 50 and b < 50:
        return 'Black'
    else:
        return f"R: {r}, G: {g}, B: {b}"