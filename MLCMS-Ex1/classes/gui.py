import sys
import tkinter
import json
import copy
import time
from tkinter import ttk
from tkinter import *
import numpy as np
from tkinter import *
from tkinter import filedialog 
import os
from .scenario import Scenario,Pedestrian
import random



class MainGUI():
    
    def __init__data(self):
        """
         self.data used to store the data imported from Json file
        """
        self.data = None 
        
    def step_forward(self):
        """
        If no backup exists, a duplicate of the current scenario is saved. Then, the simulation advances by one step 
        and the outcome is displayed.
        """
        if self.first_round == True:
            self.counter = 0
            
        if self.backup_scenario == None:
            self.update_reset_status(True)
        self.scenario.update_step()
        self.scenario.to_image(self.canvas, self.canvas_image)
        self.counter += 1
        if self.first_round:
            self.first_round = False          
        self.label_text.set('{} step'.format(self.counter))
    
    def continuous_step_forward(self,root):
        """
        corresponding to the run button.In this function, the step_forward() is looping until the update come to the end or the 
        loop is interrupted.
        """
        counter = 0
        self.running_signal = True
        def counting():
            if self.first_round:
                updated = True
            else:
                updated = self.scenario.update_step()         
            if updated == True: ##still updating P
                if self.running_signal == False:
                    return
                self.counter += 1                    
                self.step_forward()
                if updated == False:
                    return
                # looping after 100ms
                root.after(100, counting)
            else:                
                self.running_signal = False
                return            
        counting()
    ### Edit botten part###
    
    def edit_scenario_gui(self):
        """
        Generates a graphical user interface that permits the user to modify the components of the scenario. 
        Capabilities include the addition or deletion of pedestrians, targets, and obstacles.
        """
        root = Tk()
        root.resizable(False, False) 

        root.title('Edit Elements of the Scenario')
        root.geometry('')

        label = Label(root, text="Pedestrian")
        label.grid(row=0,column=0,sticky=W, padx=5, pady=5)

        self.edit_pedestrians_gui(root)

        label2 = Label(root, text="Target")
        label2.grid(row=5,column=0,sticky=W, padx=5, pady=5)     

        self.edit_targets_gui(root)

        label3 = Label(root, text="Obstacle")
        label3.grid(row=9,column=0,sticky=W, padx=5, pady=5)     

        self.edit_obstacles_gui(root)

        button = Button(root, text="Done", command=root.destroy)
        button.grid(row=13,column=2, padx=5, pady=5)

        root.mainloop()  
        
    def reset_scenarios(self):
        self.reset_button.config( height = 1, width =10 )
        self.reset_button.grid(row=1, column=2)
        self.reset_button["state"] = DISABLED
        
        #set the step number
        self.running_signal = False
        self.first_round = True
        self.counter = 0
        self.label_text.set('{} step'.format(self.counter))
        
        if self.backup_scenario != None:
            self.scenario = self.backup_scenario.copy()
            self.scenario.to_image(self.canvas, self.canvas_image)
            self.update_reset_status(False)

    def add_pedestrian(self, position, desired_speed):
        """
        Adds a pedestrian to the scenario with the specified position and desired_speed

        Parameters:
            position: Position of the pedestrian
            desired_speed: Desired speed of the pedestrian
        """
        try:
            coordinate_x = int(position[0])
            coordinate_y = int(position[1])
            desiredSpeed = float(desired_speed)
        except:
            print("ERROR:")
            print("  Coordinates and Desired Speed must be integer and float values respectively")
            return

        if (coordinate_x < 0 or coordinate_x >= self.scenario.width) or (
                coordinate_y < 0 or coordinate_y >= self.scenario.height):
            print("ERROR:")
            print("  Coordinate x must in the range [ 0,", (self.scenario.width - 1), "]")
            print("  and")
            print("  Coordinate y must in the range [ 0,", (self.scenario.height - 1), "]")
            return

        self.scenario.pedestrians.append(Pedestrian((coordinate_x, coordinate_y), desiredSpeed))
        self.scenario.grid[(coordinate_x, coordinate_y)] = Scenario.NAME2ID['PEDESTRIAN']
        self.scenario.to_image(self.canvas, self.canvas_image)

        self.update_reset_status(False)

    def remove_pedestrian(self, position):
        """
        Removes a pedestrian in the specified position from the scenario

        Parameters:
            position: Position of the pedestrian to remove
        """

        try:
            coordinate_x = int(position[0])
            coordinate_y = int(position[1])
        except:
            print("ERROR:")
            print("  Coordinates must be integer values")
            return

        if (coordinate_x < 0 or coordinate_x >= self.scenario.width) or (
                coordinate_y < 0 or coordinate_y >= self.scenario.height):
            print("ERROR:")
            print("  Coordinate x must in the range [ 0,", (self.scenario.width - 1), "]")
            print("  Coordinate y must in the range [ 0,", (self.scenario.height - 1), "]")
            return

        for i in self.scenario.pedestrians:
            if (coordinate_x, coordinate_y) == i.position:
                self.scenario.pedestrians.remove(i)
                self.scenario.grid[(coordinate_x, coordinate_y)] = Scenario.NAME2ID['EMPTY']

        self.scenario.to_image(self.canvas, self.canvas_image)
        self.update_reset_status(False)

    def edit_pedestrians_gui(self, root):
        """
        Places the widgets required for editing pedestrians to the specified window `root`

        Parameters:
            root: The window where the widgets will be placed
        """

        label = Label(root, text="Coordinate X")
        label.grid(row=0, column=1, sticky=W, padx=5, pady=5)

        entry = Entry(root)
        entry.grid(row=0, column=2)


        label2 = Label(root, text="Coordinate Y")
        label2.grid(row=1, column=1, sticky=W, padx=5, pady=5)

        entry2 = Entry(root)
        entry2.grid(row=1, column=2)

        label3 = Label(root, text="Speed")
        label3.grid(row=2, column=1, sticky=W, padx=5, pady=5)

        entry3 = Entry(root)
        entry3.grid(row=2, column=2)

        button_frame = Frame(root)
        button_frame.grid(row=3, column=2)

        button = Button(button_frame, text='Add',
                        command=lambda: self.add_pedestrian((entry.get(), entry2.get()), entry3.get()))
        button.config(height=1, width=6)
        button.grid(row=0, column=0, padx=2, pady=2)

        button = Button(button_frame, text='Remove',
                        command=lambda: self.remove_pedestrian((entry.get(), entry2.get())))
        button.config(height=1, width=6)
        button.grid(row=0, column=1, padx=2, pady=2)


    def edit_target_or_obstacle(self, position_x, position_y, thing):
        """
        Adds or removes a target or an obstacle from the specified position, depending on the parameter `thing`

        Parameters:
            position_x: x position
            position_y: y position
            thing: the element to place
        """
        try:
            coordinate_x = int(position_x)
            coordinate_y = int(position_y)
        except:
            print("ERROR:")
            print("  Coordinates must be integer values")
            return

        if (coordinate_x < 0 or coordinate_x >= self.scenario.width) or (
                coordinate_y < 0 or coordinate_y >= self.scenario.height):
            print("ERROR:")
            print("  Coordinate x must in the range [ 0,", (self.scenario.width - 1), "]")
            print("  and")
            print("  Coordinate y must in the range [ 0,", (self.scenario.height - 1), "]")
            return

        self.scenario.grid[coordinate_x, coordinate_y] = Scenario.NAME2ID[thing]
        self.scenario.recompute_target_distances()
        self.scenario.to_image(self.canvas, self.canvas_image)

        self.update_reset_status(False)

    def edit_targets_gui(self, root):
        """
        Places the widgets required for editing targets to the specified window `root`

        Parameters:
            root: The window where the widgets will be placed
        """
        label = Label(root, text="Coordinate X")
        label.grid(row=0, column=1, sticky=W, padx=5, pady=5)

        entry = Entry(root)
        entry.grid(row=0, column=2)

        label2 = Label(root, text="Coordinate Y")
        label2.grid(row=1, column=1, sticky=W, padx=5, pady=5)

        entry2 = Entry(root)
        entry2.grid(row=1, column=2)

        label3 = Label(root, text="Speed")
        label3.grid(row=2, column=1, sticky=W, padx=5, pady=5)

        entry3 = Entry(root)
        entry3.config(state='readonly')

        button_frame = Frame(root)
        button_frame.grid(row=3, column=2)

        button = Button(button_frame, text='Add',
                        command=lambda: self.edit_target_or_obstacle(entry.get(), entry2.get(), 'TARGET'))
        button.config(height=1, width=6)
        button.grid(row=0, column=0, padx=2, pady=2)

        button = Button(button_frame, text='Remove',
                        command=lambda: self.edit_target_or_obstacle(entry.get(), entry2.get(), 'EMPTY'))
        button.config(height=1, width=6)
        button.grid(row=0, column=1, padx=2, pady=2)

    def edit_obstacles_gui(self, root):
        """
        Places the widgets required for editing obstacles to the specified window `root`

        Parameters:
            root: The window where the widgets will be placed
        """
        label = Label(root, text="Coordinate X")
        label.grid(row=0, column=1, sticky=W, padx=5)

        entry = Entry(root)
        entry.grid(row=0, column=2)

        label2 = Label(root, text="Coordinate Y")
        label2.grid(row=1, column=1, sticky=W, padx=5)

        entry2 = Entry(root)
        entry2.grid(row=1, column=2)

        label3 = Label(root, text="Speed")
        label3.grid(row=2, column=1, sticky=W, padx=5, pady=5)

        entry3 = Entry(root)
        entry3.config(state='readonly')

        button_frame = Frame(root)
        button_frame.grid(row=3, column=2)

        button = Button(button_frame, text='Add',
                        command=lambda: self.edit_target_or_obstacle(entry.get(), entry2.get(), 'OBSTACLE'))
        button.config(height=1, width=6)
        button.grid(row=0, column=0, padx=2, pady=2)

        button = Button(button_frame, text='Remove',
                        command=lambda: self.edit_target_or_obstacle(entry.get(), entry2.get(), 'EMPTY'))
        button.config(height=1, width=6)
        button.grid(row=0, column=1, padx=2, pady=2)

    def edit_scenario_gui(self):
        """
        Creates and shows a user interface which allows the user to edit the elements of the scenario.
        Supported functions are adding or removing pedestrians, targets and obstacles.
        """

        root = Tk()
        root.resizable(False, False)

        root.title('Edit Elements of the Scenario')
        root.geometry('')

        v = tkinter.IntVar() #set a value to choose function


        # label = Label(root, text="Pedestrian")
        label3 = tkinter.Radiobutton(root, text='Pedestrians',variable=v, value=0,command=lambda: self.edit_pedestrians_gui(root),indicatoron=False)
        label3.grid(row=2, column=0, sticky=W, padx=5, pady=5)

        # self.edit_pedestrians_gui(root)

        # label2 = Label(root, text="Target")
        label1 = tkinter.Radiobutton(root, text='Targets',variable=v, value=1,command=lambda: self.edit_targets_gui(root),indicatoron=False)
        label1.grid(row=0, column=0, sticky=W, padx=5, pady=5)


        # self.edit_targets_gui(root)

        # label3 = Label(root, text="Obstacle")
        label2 = tkinter.Radiobutton(root, text='Obstacles',variable=v, value=2,command=lambda: self.edit_obstacles_gui(root),indicatoron=False)
        label2.grid(row=1, column=0, sticky=W, padx=5, pady=5)


        # self.edit_obstacles_gui(root)

        button = Button(root, text="Done", command=root.destroy)
        button.grid(row=3, column=3, padx=5, pady=5)

        root.mainloop()    


    def update_reset_status(self, active):
        """
        Updates the reset status depending on the boolean value 'active'.
        If 'active', a copy of the current scenario is saved as a backup scenario and the state of the reset button changes its 
        state to NORMAL, which means that it can be clicked. Otherwise, the backup scenario is removed and the state changes to
        DISABLED, so it cannot be clicked.

        Parameters:
            active (bool): The reset status to be changed to.
        """
        if active:
            self.backup_scenario = self.scenario.copy()
            self.reset_button["state"] = NORMAL
        else:
            self.backup_scenario = None
            self.reset_button["state"] = DISABLED
    
    ### Scenario4 button###
                
    def scenario_4(self):
        """
      Assuming that every 3*3 pixel represents an area of 1 square meter, scenario 4 is loaded
      and the simulation is parameterized by the pedestrian density to be simulated..
        """
        target_grid = self.scenario.target_grid
        recompute_method = self.scenario.recompute_method
        self.scenario = Scenario(300, 300)
        w = 300
        h = 300

        self.scenario.target_grid = target_grid
        self.scenario.recompute_method = recompute_method
        self.update_reset_status(False)

        for i in range(300):
            self.scenario.grid[i, 134] = Scenario.NAME2ID['OBSTACLE']
            self.scenario.grid[i, 165] = Scenario.NAME2ID['OBSTACLE']

        for i in range(30):
            self.scenario.grid[w-2,135+i] = Scenario.NAME2ID['TARGET']

        self.scenario.recompute_target_distances()

        self.scenario.pedestrians = []
        for i in range(30):
            for j in range(5):
                pedestrians_num = int(self.density * 4)
                self.scenario.pedestrians += self.fill_pedestrians((i*6, j*6+135), (5, 5), pedestrians_num, 3.9)
        
        self.scenario.measuring_points = [(120,147,6),(270,147,6),(270,153,6)]
        for measuring_point in self.scenario.measuring_points:
            self.scenario.measuring_records[measuring_point] = []
            
        self.scenario.to_image(self.canvas, self.canvas_image)



    def change_algorithm(self, *args):
        """
        Modifies the algorithm utilized for calculating target distances based on the choice made in the option menu. 
        The available options could be 'Basic' or 'Dijkstra'. The function accepts a parameter named 'args' 
        which contains the option chosen from the option menu.
        """
        self.scenario.recompute_method = args[0].upper()
        self.scenario.recompute_target_distances()
        self.scenario.to_image(self.canvas, self.canvas_image)

        if self.backup_scenario != None:
            self.backup_scenario.recompute_method = args[0].upper()
            self.backup_scenario.recompute_target_distances()

    def change_view(self, value):
        """
        Changes the view of the scenario depending on the value of the check button.
        If value = 1 then the view of the scenario changes to the target distance-based view, otherwise changes to normal view.
        Parameters:
            value (int): the value of the check button.
        """
        if value == 1:
            self.scenario.target_grid = True
        else:
            self.scenario.target_grid = False
        if self.backup_scenario != None:
            self.backup_scenario.target_grid = self.scenario.target_grid
            
        self.scenario.to_image(self.canvas, self.canvas_image)

    def change_density(self, *args):
        self.density = float(args[0])
            
    
        
    ### import the json file ###            
    
    def load_task(self):  
        """open and read json data

        Returns
        -------
        dict:
            json data

        """
        ## Renew the label step
        self.reset_button.config( height = 1, width =10 )
        self.reset_button.grid(row=1, column=2)
        self.reset_button["state"] = DISABLED
        
        #set the step number
        self.counter = 0
        self.first_round = True
        self.running_signal = False
        
        
        self.label_text.set('{} step'.format(self.counter))
        
        
        self.input_file = filedialog.askopenfilename(filetypes=[("Json", '*.json'), ("All files", "*.*")])
        if not self.input_file:
            return
        print('Loading file from', self.input_file)
        with open(self.input_file) as jf:
            self.data = json.load(jf)
        
        target_grid = self.scenario.target_grid
        recompute_method = self.scenario.recompute_method
        ##
        width = self.data['width']
        height = self.data['height']
        self.scenario = Scenario(width, height)
        self.scenario.target_grid = target_grid
        self.scenario.recompute_method = recompute_method
        self.update_reset_status(False)
        
        ## Set T,P,O
        for row, col in self.data['target']:
            self.scenario.grid[row, col] = Scenario.NAME2ID['TARGET']
        
        for row, col in self.data['obstacles']:
            self.scenario.grid[row, col] = Scenario.NAME2ID['OBSTACLE']

        self.scenario.recompute_target_distances()
        
        self.scenario.pedestrians = []
        for row, col in self.data['pedestrians']:
            self.scenario.pedestrians.append(Pedestrian((row, col), 1))

        #In Scenario 1 , we should edit the Pedestrians' speed to 3.33 pixels per second
        if self.data['scenario']==1:
            self.scenario.pedestrians = []
            for row, col in self.data['pedestrians']:
                self.scenario.pedestrians.append(Pedestrian((row, col), 3.33))

        ##works in scenario7 , adding ages, speeds
        if self.data['scenario']==7:
            self.scenario.assign_ages()
            self.scenario.assign_speeds()
            self.scenario.recompute_target_distances()

        # can be used to show pedestrians and targets
        self.scenario.to_image(self.canvas, self.canvas_image)
        
      
        
    ### Applied in Task7 , random setting pedestrians and their speed
    def fill_pedestrians(self, start_position, size, pedestrians_num, pedestrians_speed):
        pedestrians = []
        for pos in random.sample(range(size[0]*size[1]), pedestrians_num):
            pedestrians.append(Pedestrian((start_position[0]+pos%size[0], start_position[1]+pos%size[0]), pedestrians_speed))
        return pedestrians
    
    def start_gui(self):

        # Initialize Tkinter window
        win = tkinter.Tk()
        win.geometry("")
        win.resizable(False, False)
        win.title('Cellular Automata GUI')

        # Initialize scenario
        self.scenario = Scenario(100, 100)
        self.backup_scenario = None
        self.backup_scenario_4 = None
        self.running_signal = False
        
        #create menu
        menu = Menu(win)
        win.config(menu=menu)
        file_menu = Menu(menu)
        menu.add_cascade(label='Simulation', menu=file_menu)
        file_menu.add_command(label='Edit',command=lambda:self.edit_scenario_gui())
        file_menu.add_command(label='Load',command=lambda: self.load_task())


        win.config(menu=menu)      
        # Set up frames for button,canvas,and label 
        button_frame = Frame(win)
        button_frame.pack(side='top', expand=True, fill=BOTH)
        label_frame = Frame(win)
        label_frame.pack(side='top', expand=True, fill=BOTH)
        canvas_frame = Frame(win)
        canvas_frame.pack(side='top', expand=True, fill=BOTH)   
        
        # add the label with number of steps
        
        self.label_text = StringVar()
        self.label_text.set(' step')
        self.label = Label(label_frame, textvariable=self.label_text)
        self.label.pack(side='left',expand=True,fill='both', padx=2, pady=2)
        self.counter = 0
        self.first_round = True
        
        button = Button(button_frame, text='Run', command=lambda: self.continuous_step_forward(win))
        button.config( height = 1, width =10 )
        button.grid(row=1, column=0)

        button = Button(button_frame, text='Step', command=lambda: self.step_forward())
        button.config( height = 1, width =10 )
        button.grid(row=1, column=1)
        
        #### Set self.reset_button
        self.reset_button = Button(button_frame, text='Reset', command=lambda: self.reset_scenarios())
        self.reset_button.config( height = 1, width =10 )
        self.reset_button.grid(row=1, column=2)
        self.reset_button["state"] = DISABLED
        

        
        algorithms = ['Basic','Dijkstra' ]
        selected_algorithm = tkinter.StringVar(button_frame)
        selected_algorithm.set(algorithms[0])
        option_menu = OptionMenu(button_frame, selected_algorithm, *algorithms, command=self.change_algorithm)
        option_menu.config(height=1, width=5)
        option_menu.grid(row=1, column=3)
        

        
        
        button = Button(button_frame, text='Scenario 4', command=lambda: self.scenario_4())
        button.config( height = 1, width =10 )
        button.grid(row=2, column=0)
        
        scenario_4_frame = Frame(button_frame)
        scenario_4_frame.grid(row=2, column=1)
        
        self.density = 0.5
        densities = ['0.5', '1', '2', '3', '4', '5', '6']
        selected_density = tkinter.StringVar(button_frame)
        selected_density.set(densities[0])
        option_menu = OptionMenu(scenario_4_frame, selected_density, *densities, command=self.change_density)
        option_menu.config(height=1, width=5)
        option_menu.pack(side='left')


        # Canvas-related functions
        canvas_width, canvas_height = Scenario.GRID_SIZE[0]+5, Scenario.GRID_SIZE[1]+5
        self.canvas = Canvas(canvas_frame, bd=0, width=canvas_width, height=canvas_height)  # creating the canvas
        self.canvas_image = self.canvas.create_image(canvas_width/2,canvas_height/2, image=None, anchor=tkinter.CENTER)
        self.canvas.pack(side=LEFT,expand=True,fill=BOTH)

        self.scenario.to_image(self.canvas, self.canvas_image)
        
        win.mainloop()
    
    def exit_gui(self):
        """
        Closes the GUI.
        """
        sys.exit()