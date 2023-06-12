import json


"""
Here build up the scenario_6 , and export to a json file
"""

# create an empty dictionary to store positions
positions = {"scenario":1,
            "width": 100,
            "height": 100,
            "target": [], 
             "obstacles": [], 
             "pedestrians": [ ]}

for i in range(0,100):
    for j in range(39, 48):
        positions["obstacles"].append( [i,j] )

    for j in range(53, 62):
        positions["obstacles"].append( [i,j] )

        # target positions
for i in range(48, 53):
    positions["target"].append( [99,i] )

positions["pedestrians"].append( [0,50] )



# write positions to json file
with open("scenario-1.json", "w") as f:
    json.dump(positions, f)

