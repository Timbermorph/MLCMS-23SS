import json
import warnings

# def read_scenario(path="D:/RCI/mlcms/vadere/Scenarios/ModelTests/TestOSM/scenarios/rimea_06_corner.scenario"):
def read_scenario(path):
    """
    Loads a scenario file with json
    :param path: the path of the scenario file
    :return: the dictionary containing the scenario's data
    """
    with open(path, 'r') as f:
        scenario = json.load(f)
        return scenario


def add_pedestrian(scenario=None, scenario_path=None, out_scen_name=None, output_path=None, id=None,
                   find_min_free_id=True, targetIds=[], radius=0.2, densityDependentSpeed=False,
                   speedDistributionMean=1.34, speedDistributionStandardDeviation=0.26, minimumSpeed=0.5,
                   maximumSpeed=2.2, acceleration=2.0, footstepHistorySize=4, searchRadius=1.0,
                   walkingDirectionCalculation="BY_TARGET_CENTER", walkingDirectionSameIfAngleLessOrEqual=45.0,
                   nextTargetListIndex=0, position=(0, 0), velocity=(0, 0), freeFlowSpeed=1.8522156059160915,
                   followers=[], idAsTarget=-1, infectionStatus="SUSCEPTIBLE", lastInfectionStatusUpdateTime=-1.0,
                   pathogenAbsorbedLoad=0.0, groupIds=[], groupSizes=[], agentsInGroup=[], traj_footsteps=[]):
    """
    Add a pedestrian in a scenario.
    :param scenario: the dictionary containing the data of the scenario where a pedestrian needs to be added
    :param scenario_path: the path to a scenario file to be read (alternative to :param scenario:)
    :param output_path: the path where to save the new scenario file
    :param out_scen_name: name of the output scenario
    All the other parameters are the ones to be written in the pedestrian's json file section.
    """
    # 'scenario' and 'scenario_path' cannot be both None
    if scenario_path is None and scenario is None:
        raise AttributeError("One of 'scenario' and 'scenario_path' must be not None, got both None")

    # if the scenario path is not passed, than an output path is mandatory (otherwise the scenario path is used as output path too)
    elif scenario_path is None and output_path is None:
        raise AttributeError("One of 'scenario_path' and 'output_path' must be not None, got both None")

    # if both the scenario and its path are passed, only the scenario is going to be used and the path is ignored
    elif scenario_path is not None and scenario is not None:
        msg = "Both the scenario and the path to its file were passed to the function 'add_pedestrian'. " \
              "Only the scenario is going to be used, it will not be read again from file"
        warnings.warn(msg, RuntimeWarning)

    # if scenario_path is not None, read the scenario from file
    if scenario_path is not None:
        scenario = read_scenario(scenario_path)

    # if the pedestrian's id is not passed, find a free one
    if id is None:
        id = find_free_id(scenario, find_min_free_id=True)

    # if target id not provided, if there is only 1 target, use its id, otherwise don't set one (pedestrian won't move)
    if not targetIds and len(scenario['scenario']['topography']['targets']) == 1:
        targetIds = [scenario['scenario']['topography']['targets'][0]['id']]

    # create a dictionary with the pedestrian's data (in the format used in the scenario's json file)
    ped = {
        "attributes": {
            "id": id,
            "shape":{
                "x": 0.0,
                "y": 0.0,
                "width": 1.0,
                "height": 1.0,
                "type": "RECTANGLE"
            },
            "visible": True,
            "radius": radius,
            "densityDependentSpeed": densityDependentSpeed,
            "speedDistributionMean": speedDistributionMean,
            "speedDistributionStandardDeviation": speedDistributionStandardDeviation,
            "minimumSpeed": minimumSpeed,
            "maximumSpeed": maximumSpeed,
            "acceleration": acceleration,
            "footstepHistorySize": footstepHistorySize,
            "searchRadius": searchRadius,
            "walkingDirectionCalculation": walkingDirectionCalculation,
            "walkingDirectionSameIfAngleLessOrEqual": walkingDirectionSameIfAngleLessOrEqual
        },
        "source": None,
        "targetIds": targetIds,
        "nextTargetListIndex": nextTargetListIndex,
        "isCurrentTargetAnAgent": False,
        "position": {
            "x": float(position[0]),
            "y": float(position[1])
        },
        "velocity": {
            "x": velocity[0],
            "y": velocity[1]
        },
        "freeFlowSpeed": freeFlowSpeed,
        "followers": followers,
        "idAsTarget": idAsTarget,
        "isChild": False,
        "isLikelyInjured": False,
        "psychologyStatus": {
            "mostImportantStimulus": None,
            "threatMemory": {
                "allThreats": [],
                "latestThreatUnhandled": False
            },
            "selfCategory": "TARGET_ORIENTED",
            "groupMembership": "OUT_GROUP",
            "knowledgeBase": {
                "knowledge": [],
                "informationState": "NO_INFORMATION"
            },
            "perceivedStimuli": [],
            "nextPerceivedStimuli": []
        },
        "healthStatus": None,
        # "infectedStatus": None,
        "groupIds": [],
        "groupSizes": [],
        "agentsInGroup": [],
        "trajectory": {
            "footSteps": []
        },
        "modelPedestrianMap": None,
        "type": "PEDESTRIAN"



    }

    # "scenario['scenario']['topography']['dynamicElements']" gives the list of the pedestrians in the scenario;
    # append to it the new pedestrian just created
    scenario['scenario']['topography']['dynamicElements'].append(ped)

    if output_path is None:  # if output_path is None, use scenario_path
        output_path = scenario_path
    elif not output_path.endswith(".scenario"):  # add ".scenario" suffix to the output path if not present
        output_path += ".scenario"

    if out_scen_name is not None:
        scenario['name'] = out_scen_name

    # write the scenario file with the new pedestrian
    with open(output_path, 'w') as f:
        json.dump(scenario, f, indent='  ')


def find_free_id(scenario: dict, find_min_free_id=True):
    """
    Find a free id for a new pedestrian/target
    :param scenario: dictionary containing a scenario's data
    :param find_min_free_id: if True, finds the minimum free id (less efficient), otherwise simply a free id (more efficient)
    :return: a free id (int)
    """
    busy_ids = set()
    # iterate over pedestrians to collect their (already busy) ids
    dynamic_elems = scenario['scenario']['topography']['dynamicElements']
    for elem in dynamic_elems:
        if elem['type'] == 'PEDESTRIAN':
            busy_ids.add(elem['attributes']['id'])

    # iterate over targets to collect their (already busy) ids
    targets = scenario['scenario']['topography']['targets']
    for t in targets:
        busy_ids.add(t['id'])

    if not find_min_free_id:
        return max(busy_ids) + 1  # simply return the max busy id + 1 (which will be free)

    # otherwise sort the busy ids and find the minimum free one
    sorted_ids = sorted(list(busy_ids))
    try:
        # in case sorted_ids is empty, this would cause an IndexError
        prev_id = sorted_ids[0]
        for id in sorted_ids[1:]:
            if abs(id - prev_id) > 1:
                return prev_id + 1
        # if the end of the list has been reached without finding a free id, return the max id + 1
        return sorted_ids[-1] + 1
    except IndexError:
        # it means the list of ids is empty, so return simply 1
        return 1


if __name__ == '__main__':
    add_pedestrian(
        scenario_path="D:/RCI/mlcms/vadere/Scenarios/ModelTests/TestOSM/scenarios/rimea_06_corner.scenario",
        out_scen_name="task3",
        output_path="D:/RCI/mlcms/vadere/output/task3.scenario",
        position=(12, 2),
        targetIds=[1]
    )