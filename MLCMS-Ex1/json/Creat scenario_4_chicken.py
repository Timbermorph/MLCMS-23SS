import json


"""
Here build up the scenario_6 , and export to a json file
"""

# create an empty dictionary to store positions
## scenario 4-1 for chicken, 4-2 for bottlen
positions = {"scenario":4-1,
            "width": 100,
            "height": 100,
            "target": [], 
             "obstacles": [], 
             "pedestrians": []}

# targets positions 
for i in range(20, 24):        
    positions["target"].append( [i,0] )

# obstacle positions
for i in range(0,20):
    for j in range(0, 20):
        positions["obstacles"].append( [i,j] )

##########
for i in range(40):
            positions["obstacles"].append( [i,0] )
            positions["obstacles"].append( [60+i,0] )
            positions["obstacles"].append( [i,40] )
            positions["obstacles"].append( [60+i,40] )
            positions["obstacles"].append( [0,i] ) 
            positions["obstacles"].append( [99,i] )
            
            positions["obstacles"].append( [30+i,50] )
            positions["obstacles"].append( [30+i,90] )
            positions["obstacles"].append( [69,50+i] )


for i in range(20):
            positions["obstacles"].append( [40+i,17] )
            positions["obstacles"].append( [40+i,23] )
        
for i in range(18):
    positions["obstacles"].append( [40,i] )
    positions["obstacles"].append( [40,i+23] )
    positions["obstacles"].append( [60,i] )
    positions["obstacles"].append( [60,i+23] )

for i in range(6):
    positions["target"].append( [99,17+i] )
    positions["target"].append( [95,67+i] )


# for i in range(1,20):
#     for j in range(1, 40):
#         positions["pedestrians"].append( [i,j] )

for i in range(10,30):
    for j in range(60, 90):
        positions["pedestrians"].append( [i,j] )


# pedestrians positions (20 pedestrians)
#positions["pedestrians"].append( [i,j] )


# write positions to json file
with open("Task-4-1.json", "w") as f:
    json.dump(positions, f)

