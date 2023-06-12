import json


"""
Here build up the scenario_6 , and export to a json file
"""

# create an empty dictionary to store positions
positions = {
            "scenario":7,
            "width": 50,
            "height": 50,
            "target": [], 
             "obstacles": [], 
             "pedestrians": []}


for i in range(0, 50):
            positions["target"].append( [49,i] )
            positions["pedestrians"].append( [0,i] )



# write positions to json file
with open("scenario-7.json", "w") as f:
    json.dump(positions, f)

