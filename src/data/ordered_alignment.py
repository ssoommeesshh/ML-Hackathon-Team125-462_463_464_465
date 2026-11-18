"""Offline weak supervision against chronological route occurrences."""
import numpy as np
import pandas as pd


def align(xy, starts, vectors, route, segment, geometry):
    lengths = np.linalg.norm(vectors[route], axis=1)
    position = np.cumsum(lengths) - lengths / 2
    # A small cost per metre advanced discourages jumping ahead on loops.
    travel = .02 * position
    n, m = len(xy), len(route)
    distances = np.empty((n, m), dtype=np.float32)
    for rows in np.array_split(np.arange(n), max(1, int(np.ceil(n*m/100000)))):
        distances[rows] = geometry(xy[rows], starts[route], vectors[route])[0]
    cost = np.minimum(distances, 50)
    forward = np.empty_like(cost)
    forward[0] = cost[0]
    for i in range(1,n):
        offset = travel if segment[i]==segment[i-1] else 0
        forward[i] = cost[i]+np.minimum.accumulate(forward[i-1]-offset)+offset
        forward[i] -= forward[i].min()
    path = np.empty(n,dtype=int)
    margin = np.empty(n)
    backward = np.zeros(m)
    for i in range(n-1,-1,-1):
        if i==n-1:
            path[i] = forward[i].argmin()
        else:
            end = path[i+1]+1
            offset = travel[:end] if segment[i]==segment[i+1] else 0
            path[i] = (forward[i,:end]-offset).argmin()
        total = forward[i]+backward
        alternatives = total[route!=route[path[i]]]
        margin[i] = alternatives.min()-total[path[i]] if len(alternatives) else np.inf
        offset = travel if i and segment[i]==segment[i-1] else 0
        backward = np.minimum.accumulate((cost[i]+backward+offset)[::-1])[::-1]-offset
        backward -= backward.min()
    dist = distances[np.arange(n),path]
    assert np.all(np.diff(path)>=0)
    return pd.DataFrame(dict(proxy_arc_id=route[path],route_index=path,reference_distance_m=dist,
        alternative_margin=margin,ambiguous=(margin<=1)|(dist>50)))
