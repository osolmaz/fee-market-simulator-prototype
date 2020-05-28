# Points for the demand curve
# P, Q
POINTS = [
    (100, 0),
    (80, 1000),
    (60, 3000),
    (40, 6000),
    (20, 10000),
    (5, 15000),
    (0.1, 20000),
    (0, 100000),
]

POINTS = sorted(POINTS, key=lambda x: x[0])

P = [i[0] for i in POINTS]
Q = [i[1] for i in POINTS]
