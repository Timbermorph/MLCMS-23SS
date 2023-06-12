import json


"""
Here build up the scenario_6 , and export to a json file
"""

# create an empty dictionary to store positions
positions = {"scenario":6,
            "width": 24,
            "height": 24,
            "target": [], 
             "obstacles": [], 
             "pedestrians": []}

# targets positions 
for i in range(20, 24):        
    positions["target"].append( [i,0] )

# obstacle positions
for i in range(0,20):
    for j in range(18, 20):
        positions["obstacles"].append( [i,j] )

for j in range(0,20):
    for i in range(18, 20):
        positions["obstacles"].append( [i,j] )

# pedestrians positions (20 pedestrians)
for i in range(0, 5):
    for j in range(20, 24):
        positions["pedestrians"].append( [i,j] )

# write positions to json file
with open("scenario-6.json", "w") as f:
    json.dump(positions, f)

