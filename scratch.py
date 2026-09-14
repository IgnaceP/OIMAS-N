import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

n_phds = 10
n_postdocs = 3

n_memberships_phd = 0
n_memberships_inst = 1

phd_price_member = 314.6
phd_price = 453.75
postdoc_price_member = 635.25
postdoc_price = 877.25

membership_inst_price = 115.45
membership_phd_price = 34.62

#%% calculate price

import math


def angle_from_coordinate(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    d_lon = lon2 - lon1

    y = math.sin(d_lon) * math.cos(lat2)
    x = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    )

    bearing = math.degrees(math.atan2(y, x))
    bearing = (bearing + 360) % 360
    bearing = 360 - bearing

    return bearing

x = angle_from_coordinate(50.019246,4.736813, 50.017007,4.591408)

