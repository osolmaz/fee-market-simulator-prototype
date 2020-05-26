def integrate_cumulative(x, y):
    slice_areas = []
    for i in range(len(x) - 1):
        area = (x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2
        slice_areas.append(area)

    result = [0.0]
    for i in slice_areas:
        result.append(result[-1] + i)

    return result
