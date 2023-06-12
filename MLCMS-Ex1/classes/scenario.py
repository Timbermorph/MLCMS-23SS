import scipy.spatial.distance
from PIL import Image, ImageTk
import numpy as np
import math
import random
# pip install scikit-fmm
#import skfmm


class Scenario:
    """
    A scenario for a cellular automaton.
    """
    GRID_SIZE = (600, 600)
    ID2NAME = {
        0: 'EMPTY',
        1: 'TARGET',
        2: 'OBSTACLE',
        3: 'PEDESTRIAN'
    }
    NAME2COLOR = {
        'EMPTY': (255, 255, 255),
        'PEDESTRIAN': (255, 0, 0),
        'TARGET': (0, 0, 255),
        'OBSTACLE': (255, 0, 255)
    }
    NAME2ID = {
        ID2NAME[0]: 0,
        ID2NAME[1]: 1,
        ID2NAME[2]: 2,
        ID2NAME[3]: 3
    }
    max_target_distance = 1
    measuring_points = []
    measuring_records = {}

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid_image = None
        self.grid = np.zeros((width, height))
        self.pedestrians = []
        self.recompute_method = 'FMM'
        self.target_distance_grids = self.recompute_target_distances()
        self.target_grid = False
        self.pedestrian_records = [['id', 'age', 'expected_speed', 'real_speed']]
    
    def copy(self):
        """
        Makes a copy of the scenario instance.

        Return:
            A copy of the scenario instance.
        """
        scenario = Scenario(self.width, self.height)
        for x in range(self.width):
            for y in range(self.height):
                scenario.grid[x, y] = self.grid[x, y]
        for pedestrian in self.pedestrians:
            scenario.pedestrians.append(pedestrian.copy())
        scenario.recompute_method = self.recompute_method
        scenario.target_distance_grids = scenario.recompute_target_distances()
        scenario.target_grid = self.target_grid
        return scenario

    


    def recompute_target_distances(self):
        """
        Recomputes the target distances with the algorithm specified in the attribute `recompute_method`.
         
        Here adding  self.recompute_method
        """
        if self.recompute_method == 'BASIC':
            self.target_distance_grids = self.update_target_grid()
        elif self.recompute_method == 'DIJKSTRA':
            self.target_distance_grids = self.dijkstra_flood_update_target_grid((self.width+self.height)*10)
        elif self.recompute_method == 'FMM':
             self.target_distance_grids = self.update_target_grid()
        return self.target_distance_grids

    def update_target_grid(self):
        """
        Computes the shortest distance from every grid point to the nearest target cell.
        This does not take obstacles into account.

        Return:
            The distance for every grid cell, as a np.ndarray.
        """
        targets = []
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x, y] == Scenario.NAME2ID['TARGET']:
                    targets.append([y, x])  # y and x are flipped because they are in image space.
        if len(targets) == 0:
            return np.zeros((self.width, self.height))

        targets = np.row_stack(targets)
        x_space = np.arange(0, self.width)
        y_space = np.arange(0, self.height)
        xx, yy = np.meshgrid(x_space, y_space)
        positions = np.column_stack([xx.ravel(), yy.ravel()])

        # after the target positions and all grid cell positions are stored,
        # compute the pair-wise distances in one step with scipy.
        distances = scipy.spatial.distance.cdist(targets, positions)

        # now, compute the minimum over all distances to all targets.
        distances = np.min(distances, axis=0)

        self.max_target_distance = distances.max()

        return distances.reshape((self.width, self.height))
    def assign_ages(self):
        """
        Randomly assignment of pedestrian ages
        """
        for pedestrian in self.pedestrians:
            pedestrian.age = random.randint(5, 80)

    def assign_speeds(self):
        """
        Random assignment of walking speeds to the list of pedestrians depending on their ages with a normal distribution,
        according to the figure shown in RiMEA scenario 7.
        """
        for pedestrian in self.pedestrians:
            age = pedestrian.age
            if 3 <= age <= 10:
                pedestrian._desired_speed = np.random.uniform(0.6, 1.1)
            elif 11 <= age <= 20:
                pedestrian._desired_speed = np.random.uniform(1.1, 1.6)
            elif 21 <= age <= 50:
                pedestrian._desired_speed = np.random.uniform(1.6, 1.3)
            elif 51 <= age <= 70:
                pedestrian._desired_speed = np.random.uniform(1.3, 1.1)
            else:
                pedestrian._desired_speed = np.random.uniform(1.1, 0.7)





    """         DIJKSTRA RELATED        """
    
    def get_open_neighbors(self, closed, position):
        """
        Compute all neighbors in a 9 cell neighborhood of the current position.
        :param closed_positions: 2D binary table marking closed positions
        :param position: A tuple position (x,y)
        :return: A list of neighbor cell indices (x,y) around the current position.
        """
        neighbors = []
        for x in [position[0] - 1, position[0], position[0] + 1]:
            for y in [position[1] -1, position[1], position[1] + 1]:
                if 0 <= x < self.width and 0 <= y < self.height and closed[x, y] == 0:
                    closed[x, y] = 1
                    neighbors.append((x, y))
        return neighbors

    def get_min_distance(self, distances, position):
        """
        Compute the min distance at current position considering all neighbors in a 9 cell neighborhood of the current position on the distance map.
        :param distances: 2D nparray distance table
        :param position: A tuple position (x,y)
        :return: A list of neighbor cell indices (x,y) around the current position.
        """
        new_distance = distances[position[0], position[1]]
        new_position = (position[0], position[1])
        for x in [position[0] - 1, position[0], position[0] + 1]:
            for y in [position[1] -1, position[1], position[1] + 1]:
                if 0 <= x < self.width and 0 <= y < self.height:
                    temp_distance = distances[x, y] + (1 if x==position[0] or y==position[1] else math.sqrt(2))
                    if temp_distance < new_distance:
                        new_distance = temp_distance
                        new_position = (x, y)
        return new_distance
    
    def dijkstra_flood_update_target_grid(self, limit):
        """
        Computes the shortest distance from every grid point to the nearest target cell.
        This does not take obstacles into account.
        :param limit: work as max distance, after which flooding will stop.
        :returns: The distance for every grid cell, as a np.ndarray.
        """
        targets = []
        closed = np.zeros((self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x, y] == Scenario.NAME2ID['TARGET']:
                    targets.append((x, y))
                elif self.grid[x, y] == Scenario.NAME2ID['OBSTACLE']:
                    closed[x, y] = 1
        if len(targets) == 0:
            return np.zeros((self.width, self.height))
        
        distances = np.matrix(np.ones((self.width, self.height)) * np.inf)
        open_positions = []
        for x, y in targets:
            distances[x, y] = 0
            closed[x, y] = 1
            open_positions += self.get_open_neighbors(closed, (x, y))
        
        counter = 1
        while counter < limit and open_positions:
            next_open_positions = []
            for x, y in open_positions:
                distances[x, y] = self.get_min_distance(distances, (x, y))
                next_open_positions += self.get_open_neighbors(closed, (x, y))
            open_positions = next_open_positions
            counter += 1
        
        for x, y in open_positions:
            distances[x, y] = counter
            
        distances = distances.reshape((self.width, self.height))            
        
        self.max_target_distance = counter

        return distances    

    def update_step(self):
        """
        Updates the position of all pedestrians.
        This does not take obstacles or other pedestrians into account.
        Pedestrians can occupy the same cell.
        """
        updated = False
        for pedestrian in self.pedestrians:
            moved = pedestrian.update_step(self)   ###### until here treated like step
            if moved:
                updated = True                ### Keep run
        return updated

    @staticmethod
    def cell_to_color(_id):
        return Scenario.NAME2COLOR[Scenario.ID2NAME[_id]]

    def target_grid_to_image(self, canvas, old_image_id):
        """
        Creates a colored image based on the distance to the target stored in
        self.target_distance_gids.

        Parameters
            canvas: the canvas that holds the image.
            old_image_id: the id of the old grid image.
        """
        im = Image.new(mode="RGB", size=(self.width, self.height))
        pix = im.load()
        for x in range(self.width):
            for y in range(self.height):
                target_distance = self.target_distance_grids[x, y]
                if target_distance == np.inf:
                    target_distance = self.width * self.height
                pix[x, y] = (max(0, min(255, int(1000 * target_distance / self.max_target_distance) - 0 * 255)),
                             max(0, min(255, int(1000 * target_distance / self.max_target_distance) - 1 * 255)),
                             max(0, min(255, int(1000 * target_distance / self.max_target_distance) - 2 * 255)))
        im = im.resize(Scenario.GRID_SIZE, Image.NONE)
        self.grid_image = ImageTk.PhotoImage(im)
        canvas.itemconfigure(old_image_id, image=self.grid_image)

    def to_image(self, canvas, old_image_id):
        """
        Creates a colored image based on the ids stored in self.grid.
        Pedestrians are drawn afterwards, separately.

        Parameters:
            canvas: the canvas that holds the image.
            old_image_id: the id of the old grid image.
        """
        if self.target_grid:
            self.target_grid_to_image(canvas, old_image_id)
        else:
            im = Image.new(mode="RGB", size=(self.width, self.height))
            pix = im.load()
            for x in range(self.width):
                for y in range(self.height):
                    pix[x, y] = self.cell_to_color(self.grid[x, y])
            for pedestrian in self.pedestrians:
                x, y = pedestrian.position
                if self.grid[x,y] != Scenario.NAME2ID['TARGET']:
                    pix[x, y] = Scenario.NAME2COLOR['PEDESTRIAN']
            im = im.resize(Scenario.GRID_SIZE, Image.NONE)
            self.grid_image = ImageTk.PhotoImage(im)
            canvas.itemconfigure(old_image_id, image=self.grid_image)


def euclidean_distance(x1, y1, x2, y2):
    """
    Calculates the euclidean distance of two points.

    Parameters:
        x1: x parameter of point 1
        y1: y parameter of point 1
        x2: x parameter of point 2
        y2: y parameter of point 2
    """
    x = x1-x2
    y = y1-y2
    return math.sqrt(x*x+y*y)


class Pedestrian:
    """
    Setting a single pedestrian.
    """

    def __init__(self, position, desired_speed):
        self._position = position
        self._desired_speed = desired_speed
        self.accumulated_distance = 0
        self.id = -1
        self.age = -1
        self.distance_walked = 0
        self.ticks = 0
        self.finished = False
        self.saved = False

    def copy(self):
        return Pedestrian(self._position, self._desired_speed)

    @property
    def position(self):
        return self._position

    @property
    def desired_speed(self):
        return self._desired_speed

    def get_neighbors(self, scenario):
        """
        Compute all neighbors in a 9 cell neighborhood of the current position,and returen a list of neighbor cell indices (x,y) around the current position.   
        """
        return [
            (int(x + self._position[0]), int(y + self._position[1]))
            for x in [-1, 0, 1]
            for y in [-1, 0, 1]
            if 0 <= x + self._position[0] < scenario.width and 0 <= y + self._position[1] < scenario.height and np.abs(x) + np.abs(y) > 0
        ]

    def update_step(self, scenario):
        """
        Moves to the cell with the lowest distance to the target.
        """
        if self.finished:
            if not self.saved:
                self.saved = True
                if self.ticks == 0:
                    self.ticks = 1
                if self.id != -1:
                    scenario.pedestrian_records.append([self.id, self.age, self.desired_speed, self.distance_walked/self.ticks])
            return False
        
        moved = False
        available_distance = self._desired_speed + self.accumulated_distance
        self.accumulated_distance = 0
        while available_distance > 0:
            neighbors = self.get_neighbors(scenario)
            p_x = self._position[0]
            p_y = self._position[1]
            next_cell_distance = scenario.target_distance_grids[p_x, p_y]
            next_pos = self._position
            # Goes to the neighbor position that minimizes the distance to the nearest target
            for (n_x, n_y) in neighbors:
                if scenario.grid[n_x, n_y] != Scenario.NAME2ID['OBSTACLE'] and scenario.grid[n_x, n_y] != Scenario.NAME2ID['PEDESTRIAN']:
                    if next_cell_distance > scenario.target_distance_grids[n_x, n_y]:
                        next_pos = (n_x, n_y)
                        next_cell_distance = scenario.target_distance_grids[n_x, n_y]
                    
                    elif next_cell_distance == scenario.target_distance_grids[n_x, n_y]:
                        distance_to_next = euclidean_distance(p_x, p_y, next_pos[0], next_pos[1])
                        distance_to_neighbor = euclidean_distance(p_x, p_y, n_x, n_y)
                        if distance_to_neighbor < distance_to_next:
                            next_pos = (n_x, n_y)
                            next_cell_distance = scenario.target_distance_grids[n_x, n_y]
                    
            if self._position != next_pos:
                '''
                    measuring related points 
                '''
                for x, y, size in scenario.measuring_points:
                    if  next_pos[0] >= x and next_pos[0] <= x + size and next_pos[1] >= y and next_pos[1] <= y + size and self._position[0] < x:
                        scenario.measuring_records[(x, y, size)] += [self.ticks]

                moved = True
                distance_to_travel = euclidean_distance(p_x, p_y, next_pos[0], next_pos[1])
                if distance_to_travel <= available_distance:
                    available_distance -= distance_to_travel
                    self.distance_walked += distance_to_travel
                    if scenario.grid[next_pos] != Scenario.NAME2ID['TARGET']:
                        scenario.grid[next_pos] = Scenario.NAME2ID['PEDESTRIAN']
                    elif scenario.grid[next_pos] == Scenario.NAME2ID['TARGET']:
                        self.finished = True
                    scenario.grid[self._position] = Scenario.NAME2ID['EMPTY']
                    self._position = next_pos
                else:
                    self.accumulated_distance += available_distance
                    available_distance = 0
            else:
                break
        self.ticks += 1
        return moved


