def generate_temp_name(base="temp"):
    """Generate unique temporary dataset names"""
    import time
    return f"{base}_{int(time.time())}"
