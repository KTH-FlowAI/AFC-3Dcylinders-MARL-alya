# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 14:08:40 2021

@author: Maxence
"""
## Import section
import sys
import os
import numpy as np
cwd = os.getcwd()
sys.path.append(cwd + "/../")
from parameters import dimension

##
class geom_functions:
    def __init__(self,file):
        global geometry_file
        self.file = file
        geometry_file = self.file
        
    def Add_point(self,point_number,x,y,size):
        string = "Point(" + str(point_number) + ") = {" + str(x) + ", " + str(y) + ", " + "0, "+ str(size) +"};\n"
        geometry_file.write(string)
        
    def Rotate_point_2D(self,point_number,angle):
        string = "Rotate{{0,0,1},{0,0,0},"+str(angle)+"}{Point{"+point_number+"};}\n"
        geometry_file.write(string)
        
    def Add_line(self,line_number,P1,P2, Transfinite = False, NbTr = 10, Progression = 1):
        string = "Line(" + str(line_number) + ") = {" + str(P1) + ", "+ str(P2) + "};"
        if Transfinite:
            string += " Transfinite Line {" + str(line_number) + "} "+ "= " + str(NbTr) + " Using Progression " + str(Progression) + ";"
        string += "\n"
        geometry_file.write(string)
        
    def Add_spline(self,line_number,spline):
        # Spline is a vector that contains the Points to be included in the spline
        string = "Spline("+str(line_number)+") = {"
        for point in spline[0:-1]:
            string = string+str(point)+", "
        string = string+str(spline[-1])+"};\n"
        geometry_file.write(string)
        
    def Add_Circle(self,curve_number,PCenter,PRadius1,PRadius2, Transfinite = False, NbTr = 10, Progression = 1):
        string = "Circle(" + str(curve_number) + ") = { " + str(PRadius1) + ", "+ str(PCenter) + ", " + str(PRadius2) + "};"
        if Transfinite:
            string += " Transfinite Line {" + str(curve_number) + "} "+ "= " + str(NbTr) + " Using Progression " + str(Progression) + ";"
        string += "\n"
        geometry_file.write(string)
        
    def Add_CurveLoop(self,curve_number,line_list):
        string = "Curve Loop(" + str(curve_number) + ") = {"+ str(line_list)[1:-1] + "};\n"
        geometry_file.write(string)
        
    def Add_LineLoop(self,line_number,line_list):
        string = "Line Loop(" + str(line_number) + ") = {"+ str(line_list)[1:-1] + "};\n"

        geometry_file.write(string)
        
    def Add_PlaneSurface(self,surface_number,Loop_number, Transfinite = False, Recombine = False):
        if not type(Loop_number)==tuple:
            string = "Plane Surface(" + str(surface_number) + ") = {"+ str(Loop_number) + "};"
        else:
            string = "Plane Surface(" + str(surface_number) + ") = {"+ str(Loop_number[0]) + ', '+ str(Loop_number[1]) + "};"
        
        if Transfinite:
                string += " Transfinite Surface {" + str(Loop_number) + "}"+ ";"
                
        if Recombine:
            string += " Recombine Surface {" + str(Loop_number) + "}"+ ";"
        
        string += "\n"
            
        geometry_file.write(string)
        
    def Add_Transfinite_Line(self,Lines_list,Transfinite_number, Progression = False, Progression_number = 1.):
        string = "Transfinite Line {" 
        for line in Lines_list:
            string += str(line) + ","
        string = string[:-1]
        string += '} = '+str(Transfinite_number)
        if Progression:
            string += ' Using Progression ' + str(Progression_number)
        string += ";\n"
        
        geometry_file.write(string)

    def Add_Transfinite_Surface(self,Surface_list,Transfinite_number = 1, Progression = False, Progression_number = 1.):
        string = "Transfinite Surface {" 
        for line in Surface_list:
            string += str(line) + ","
        string = string[:-1]
        string += "};\n"

        geometry_file.write(string)          
            
    def Add_Recombine_Surface(self,Surface_list):
        string = "Recombine Surface {" 
        for line in Surface_list:
            string += str(line) + ","
        string = string[:-1]
        string += "};\n"            
        
        geometry_file.write(string)
    
    def Add_Extrude(self,extrude,Surface_list,Layers_number = 1):
        string = "Extrude {0,0," + str(extrude) + "} { Surface{"
        
        for surface in Surface_list:
            string += str(surface) + ","
        string = string[:-1]
        string += "}; Layers{" + str(Layers_number) + "}; Recombine;}"
        
        geometry_file.write(string)

    def Add_Physical_Point(self,name,ID,Point_list):
        string = """Physical Point (" """ 
        string = string[:-1]
        string += str(name) 
        string += """", """
        string += str(ID) + ") = {"
        for point in Point_list:
            string += str(point) + ","
        string = string[:-1]
        string += "};\n"          
        
        geometry_file.write(string)
    
    def Add_Physical_Line(self,name,Line_list):
        string = """Physical Line (" """ 
        string = string[:-1]
        string += str(name) 
        string += """") = {"""
        for line in Line_list:
            string += str(line) + ","
        string = string[:-1]
        string += "};\n"            
        
        geometry_file.write(string)
       

    def Add_Physical_Curve(self,name,ID,Curve_list):
        string = "Physical Curve (\""+str(name)+"\""
        if type(ID) == str:
            string += ","+str(ID)
            
        string += ") = {"
        for curve in Curve_list:
            string += str(curve) + ","
        string = string[:-1]
        string += "};\n"          
        
        geometry_file.write(string)
      
    def Add_Physical_Surface(self,name,ID,Surface_list):
        string = "Physical Surface (\""+str(name)+"\""
        if type(ID) == str:
            string += ","+str(ID)
            
        string += ") = {"
        for surface in Surface_list:
            string += str(surface) + ","
        string = string[:-1]
        string += "};\n"          
        
        geometry_file.write(string)
        
    def Add_Physical_Volume(self,name,ID,Volume_list):
        string = """Physical Volume (" """ 
        string = string[:-1]
        string += str(name) 
        string += """", """
        string += str(ID) + ") = {"
        for volume in Volume_list:
            string += str(volume) + ","
        string = string[:-1]
        string += "};\n"          
        
        geometry_file.write(string)
        

        
    def Clear_file(self):
        geometry_file.truncate(0)
        

class figure_functions:
    def __init__(self):
        pass

    def polygon(filepath):
        ## import section
        from parameters import Dict_domain,dp,Transfinite_number,Progression_number
        
        ## open the file
        geometry_file = open(filepath, 'w')
        fig = geom_functions(file = geometry_file)
          
        
        ## Clear the file in case it is not empty
        fig.Clear_file()
        
        
        ## Add the points of the domain
        geometry_file.write("// Define Domain Points\n")
        
        
        domain_points =     [
                            Dict_domain["downleft"],
                            Dict_domain["downright"],
                            Dict_domain["upright"],
                            Dict_domain["upleft"]
                            ]
        
        
        ## Add the points
        geometry_file.write("// Define Points\n")
        point_number = 0
        for point in domain_points:
            point_number += 1
            fig.Add_point(point_number = point_number,
                          x = point[0],
                          y = point[1],
                          size = dp)
        geometry_file.write("\n\n")
        
                
        ## Add the lines
        geometry_file.write("// Define Lines\n")
        for line_number in range(point_number-1):
            fig.Add_line(line_number = line_number+1,
                         P1 = line_number +1,
                         P2 = line_number +2
                         )
        fig.Add_line(line_number = point_number,
                     P1 = point_number,
                     P2 = 1
                     )
        geometry_file.write("\n\n")


        ## Add the Curve
        geometry_file.write("// Define Curves\n")
        fig.Add_CurveLoop(curve_number = 1,
                          line_list = list(range(1,5))
                          )
        geometry_file.write("\n\n")
        
        
        ## Refine the mesh in y
        geometry_file.write("// Refine the mesh in y\n")
        fig.Add_Transfinite_Line(Lines_list =  [2,4],
                                 Transfinite_number = Transfinite_number,
                                 Progression = True,
                                 Progression_number = Progression_number)
        
        ## Create the surface
        geometry_file.write("// Define Surface\n")
        fig.Add_PlaneSurface(surface_number = 1,
                             Loop_number = 1)
        geometry_file.write("\n\n")
        
        
        ## Mesh these surfaces in a structured manner
        geometry_file.write("// Mesh these surfaces in a structured manner\n")
        Surface_list = [
                        1
                        ]
        fig.Add_Transfinite_Surface(Surface_list = Surface_list)
        geometry_file.write("\n\n")
        
        
        ## Turn into quads
        geometry_file.write("// Turn into quads\n")
        fig.Add_Recombine_Surface(Surface_list)
        geometry_file.write("\n\n")
        
        
        if dimension == "2D":
            geometry_file.write("// Create Physical surface\n")
            physical_number = 100
            fig.Add_Physical_Curve(name = "inlet",
                                   ID = physical_number,
                                   Curve_list = [4]
                                   )
            physical_number += 1
            fig.Add_Physical_Curve(name = "outlet",
                                   ID = physical_number,
                                   Curve_list = [2]
                                   )
            physical_number += 1
            fig.Add_Physical_Curve(name = "top",
                                   ID = physical_number,
                                   Curve_list = [3]
                                   )
            physical_number += 1
            fig.Add_Physical_Curve(name = "bottom",
                                   ID = physical_number,
                                   Curve_list = [1]
                                   )
            physical_number += 1
            fig.Add_Physical_Surface(name = "fluid",
                                     ID = physical_number,
                                     Surface_list = [1]
                                     )
        
            
            geometry_file.write("\n\n")
        if dimension == "3D":
            ## Create the mesh
            from parameters import Lz
            #from parameters import Layers_number
            
            
            geometry_file.write("// Create 3D surface\n")
            
            fig.Add_Extrude(extrude = Lz, 
                            Surface_list = [1]
                            )
            geometry_file.write("\n\n")
            
            
            geometry_file.write("// Create Physical surface\n")
            physical_number = 100
            fig.Add_Physical_Surface(name = "inlet",
                                     ID = physical_number,
                                     Surface_list = [25]
                                     )
            physical_number += 1
            fig.Add_Physical_Surface(name = "outlet",
                                     ID = physical_number,
                                     Surface_list = [17]
                                     )
            physical_number += 1
            fig.Add_Physical_Surface(name = "top",
                                     ID = physical_number,
                                     Surface_list = [21]
                                     )
            physical_number += 1
            fig.Add_Physical_Surface(name = "bottom",
                                     ID = physical_number,
                                     Surface_list = [13]
                                     )
            physical_number += 1
            fig.Add_Physical_Surface(name = "FrontAndBack",
                                     ID = physical_number,
                                     Surface_list = [1, 26]
                                     )
                
            physical_number += 1
            fig.Add_Physical_Volume(name = "internal",
                                    ID = physical_number,
                                    Volume_list = [1]
                                    )
        
            
            geometry_file.write("\n\n")
        
        ## Close the file
        geometry_file.close()
        
    def cylinder(filepath):
        ## import section
        from parameters import radius

        from parameters import dp_left,dp_right,dp_cyl,dp_karman
        
        from parameters import boundary_layer
        
        from utils_cylinder_points import add_angle_circle_points_dict
        
        if boundary_layer:
            from parameters import Transfinite_number, Progression_number,outer_radius
        
        from parameters import xkarman, Dict_domain, cylinder_coordinates, geometry_params
   
        ## open the file
        geometry_file = open(filepath, 'w')
        fig = geom_functions(file = geometry_file)
        
        
        ## Clear the file in case it is not empty
        fig.Clear_file()
       
        ## Parameters
        point_number = 0
        
        lines_number = 0
        
        surface_number = 0
        
        Domain_lines = {}
        
        meshing_points = {}
        
        
        ## Add the points of the domain
        geometry_file.write("// Define Domain Points\n")
        
        
        domain_points =     [
                            Dict_domain["downleft"],
                            Dict_domain["downright"],
                            Dict_domain["upright"],
                            Dict_domain["upleft"]
                            ]
        
                
        point_number += 1 # downleft
        meshing_points["domain_downleft"] = point_number
        fig.Add_point(point_number = point_number,
                      x = domain_points[0][0],
                      y = domain_points[0][1],
                      size = dp_left)

        point_number += 1 # downright
        meshing_points["domain_downright"] = point_number
        fig.Add_point(point_number = point_number,
                      x = domain_points[1][0],
                      y = domain_points[1][1],
                      size = dp_right)

        point_number += 1 # upright
        meshing_points["domain_upright"] = point_number
        fig.Add_point(point_number = point_number,
                      x = domain_points[2][0],
                      y = domain_points[2][1],
                      size = dp_right)

        point_number += 1 # upleft
        meshing_points["domain_upleft"] = point_number
        fig.Add_point(point_number = point_number,
                      x = domain_points[3][0],
                      y = domain_points[3][1],
                      size = dp_left)
        
        geometry_file.write("\n\n")
        
        
        ## Add the Cylinder
        geometry_file.write("// Define Cylinder Points\n")
        
        # Define the center point
        Circle_points_number = {} #Center,Up,Right,Down,Left
        
        point_number += 1 # Center
        coordinates = (cylinder_coordinates[0],cylinder_coordinates[1])
        Circle_points_number["Center"] = [point_number,coordinates]
        fig.Add_point(point_number = point_number,
                      x = coordinates[0],
                      y = coordinates[1],
                      size = dp_cyl)
        
                
        # Define the inner radius points
        point_number += 1 # Up
        coordinates = (cylinder_coordinates[0],cylinder_coordinates[1] + radius)
        Circle_points_number["Up"] = [point_number,coordinates]
        fig.Add_point(point_number = point_number,
                      x = coordinates[0],
                      y = coordinates[1],
                      size = dp_cyl)
        
        point_number += 1 # Left
        coordinates = (cylinder_coordinates[0] - radius,cylinder_coordinates[1])
        Circle_points_number["Left"] = [point_number,coordinates]
        fig.Add_point(point_number = point_number,
                      x = coordinates[0],
                      y = coordinates[1],
                      size = dp_cyl)
        
        point_number += 1  # Down
        coordinates = (cylinder_coordinates[0],cylinder_coordinates[1] - radius)
        Circle_points_number["Down"] = [point_number,coordinates]
        fig.Add_point(point_number = point_number,
                      x = coordinates[0],
                      y = coordinates[1],
                      size = dp_cyl)
        
        point_number += 1 # Right
        coordinates = (cylinder_coordinates[0] + radius,cylinder_coordinates[1])
        Circle_points_number["Right"] = [point_number,coordinates]
        fig.Add_point(point_number = point_number,
                      x = coordinates[0],
                      y = coordinates[1],
                      size = dp_cyl)
        
        # Define jets points
        jet_angle = geometry_params['jet_positions_angle']
        for i in range(len(jet_angle)):
            jet_angle[i] = jet_angle[i]*np.pi/180
        jet_width = geometry_params['jet_width']*np.pi/180
        
        point_number += 1 # Jet 1.1
        jet_x = radius*np.cos(jet_angle[0] + jet_width/2)
        jet_y = radius*np.sin(jet_angle[0] + jet_width/2)
        coordinates = (cylinder_coordinates[0] + jet_x, cylinder_coordinates[1] + jet_y)
        Circle_points_number["Jet_1.1"] = [point_number,coordinates]
        fig.Add_point(point_number = point_number,
                      x = coordinates[0],
                      y = coordinates[1],
                      size = dp_cyl)
    
        point_number += 1 # Jet 1.2
        jet_x = radius*np.cos(jet_angle[0] - jet_width/2)
        jet_y = radius*np.sin(jet_angle[0] - jet_width/2)
        coordinates = (cylinder_coordinates[0] + jet_x, cylinder_coordinates[1] + jet_y)
        Circle_points_number["Jet_1.2"] = [point_number,coordinates] 
        fig.Add_point(point_number = point_number,
                      x = coordinates[0],
                      y = coordinates[1],
                      size = dp_cyl)
        
        point_number += 1 # Jet 2.1
        jet_x = radius*np.cos(jet_angle[1] + jet_width/2)
        jet_y = radius*np.sin(jet_angle[1] + jet_width/2)
        coordinates = (cylinder_coordinates[0] + jet_x, cylinder_coordinates[1] + jet_y)
        Circle_points_number["Jet_2.1"] = [point_number,coordinates] 
        fig.Add_point(point_number = point_number,
                      x = coordinates[0], 
                      y = coordinates[1],
                      size = dp_cyl)
        
        point_number += 1 # Jet 2.2
        jet_x = radius*np.cos(jet_angle[1] - jet_width/2)
        jet_y = radius*np.sin(jet_angle[1] - jet_width/2)
        coordinates = (cylinder_coordinates[0] + jet_x, cylinder_coordinates[1] + jet_y)
        Circle_points_number["Jet_2.2"] = [point_number,coordinates]
        fig.Add_point(point_number = point_number,
                      x = coordinates[0],
                      y = coordinates[1],
                      size = dp_cyl)
        
        geometry_file.write("\n")
        
        Circle_points_number = add_angle_circle_points_dict(dict_circle_point = Circle_points_number, 
                                                            cylinder_coordinates = cylinder_coordinates)
        
        # pri(nt(Circle_points_number)
        
        if boundary_layer:
        # Define the outer radius points
            point_number += 1 # Up
            coordinates = (cylinder_coordinates[0],cylinder_coordinates[1] + outer_radius)
            Circle_points_number["Outer_Up"] = (point_number,coordinates) 
            fig.Add_point(point_number = point_number,
                          x = coordinates[0],
                          y = coordinates[1],
                          size = dp_cyl)
            
            point_number += 1 # Left
            coordinates = (cylinder_coordinates[0] - outer_radius,cylinder_coordinates[1])
            Circle_points_number["Outer_Left"] = (point_number,coordinates) 
            fig.Add_point(point_number = point_number,
                          x = coordinates[0],
                          y = coordinates[1],
                          size = dp_cyl)
            
            point_number += 1  # Down
            coordinates = (cylinder_coordinates[0],cylinder_coordinates[1] - outer_radius)
            Circle_points_number["Outer_Down"] = (point_number,coordinates)
            fig.Add_point(point_number = point_number,
                          x = coordinates[0],
                          y = coordinates[1],
                          size = dp_cyl)
            
            point_number += 1 # Right
            coordinates = (cylinder_coordinates[0] + outer_radius,cylinder_coordinates[1])
            Circle_points_number["Outer_Right"] = (point_number,coordinates) 
            fig.Add_point(point_number = point_number,
                          x = coordinates[0],
                          y = coordinates[1],
                          size = dp_cyl)
            geometry_file.write("\n\n")
        
        
        # Define the Karman points
        geometry_file.write("// Define Karman Points\n")
        Karman_points = {}
        point_number += 1 # Top
        xdistance = xkarman
        coordinates = (xdistance,domain_points[2][1])
        Karman_points["Top"] = (point_number,coordinates) 
        fig.Add_point(point_number = point_number,
                      x = coordinates[0],
                      y = coordinates[1],
                      size = dp_karman)
        
        point_number += 1 # Bottom
        coordinates = (xdistance,domain_points[0][1])
        Karman_points["Bottom"] = (point_number,coordinates) 
        fig.Add_point(point_number = point_number,
                      x = coordinates[0],
                      y = coordinates[1],
                      size = dp_karman)
        geometry_file.write("\n\n")
        
        
        # Add the Circles
        geometry_file.write("// Draw the cylinder\n")
        
        Circle_lines = {}
        
        first_key_done = False
        temp_key = list(Circle_points_number)
        jet_n = 0
        cylinder_n = 0
        # print(temp_key)
        for key in temp_key:
            lines_number += 1
            try:
                next_key = temp_key[temp_key.index(key) + 1]
                previous_key = temp_key[temp_key.index(key) - 1]
                # print('\n\n',key,next_key,'\n')
                temp_key_lines = list(Circle_lines) + ["start_of_the_dict"]
                previous_key_line = temp_key_lines[-1]
                if jet_n == 0 and (not first_key_done):
                            first_key = key
                            first_key_done = True
                            # print("firstkey:",Circle_points_number[first_key][0])
                if key[:3] == 'Jet':
    
                    if next_key[:3] == 'Jet':
                        del temp_key_lines
                        
                        jet_n = lines_number
                        Circle_lines['Jet_ss_'+str(jet_n)] = lines_number
                        # print(Circle_lines)
                        temp_key_lines = list(Circle_lines)
                        previous_key_line = temp_key_lines[-1]
    
                    elif previous_key[:3] != 'Jet' and previous_key_line[:8] != 'Jet_sta_':
                        del temp_key_lines
                        
                        jet_n = lines_number
                        Circle_lines['Jet_st_'+str(jet_n)] = lines_number
                        # print(Circle_lines)
                        temp_key_lines = list(Circle_lines)
                        previous_key_line = temp_key_lines[-1]
                        # print(temp_key_lines)
                        # print(previous_key_line[:3])
                        

                elif key[:3]!='Jet' and previous_key_line[:3] == 'Jet':
                    del temp_key_lines

                    cylinder_n = lines_number
                    Circle_lines['Cylinder_'+str(cylinder_n)] = lines_number
                    # print(Circle_lines)
                    temp_key_lines = list(Circle_lines)
                    previous_key_line = temp_key_lines[-1]
                    # print(temp_key_lines)
                
                if next_key != 'Center':
                    fig.Add_Circle(curve_number = lines_number,
                                   PCenter = Circle_points_number["Center"][0], 
                                   PRadius1 = Circle_points_number[key][0],
                                   PRadius2 = Circle_points_number[next_key][0],
                                   )
                else:
                    fig.Add_Circle(curve_number = lines_number,
                                   PCenter = Circle_points_number["Center"][0], 
                                   PRadius1 = Circle_points_number[key][0],
                                   PRadius2 = Circle_points_number[first_key][0],
                                   )
                    # print("center:",Circle_points_number[next_key][0])
            except:
                pass # which help to exclude the last point overlapse
                # print(lines_number)
                # fig.Add_Circle(curve_number = lines_number,
                #                    PCenter = Circle_points_number["Center"][0], 
                #                    PRadius1 = Circle_points_number[key][0],
                #                    PRadius2 = Circle_points_number[next_key][0],
                #                    )
        
        geometry_file.write("\n\n")
        
        
        ## Add the lines of the left block domain
        geometry_file.write("// Define the block domain line\n")
        
        lines_number += 1
        Domain_lines["Top_left"] = lines_number
        fig.Add_line(line_number = lines_number,
                      P1 = meshing_points["domain_upleft"],
                      P2 = Karman_points["Top"][0]
                      )
        
        lines_number += 1
        Domain_lines["Top_right"] = lines_number
        fig.Add_line(line_number = lines_number,
                      P1 = Karman_points["Top"][0],
                      P2 = meshing_points["domain_upright"]
                      )        
        
        lines_number += 1
        Domain_lines["Right"] = lines_number
        fig.Add_line(line_number = lines_number,
                      P1 = meshing_points["domain_upright"],
                      P2 = meshing_points["domain_downright"]
                      )
        
        lines_number += 1
        Domain_lines["Down_right"] = lines_number
        fig.Add_line(line_number = lines_number,
                      P1 = meshing_points["domain_downright"],
                      P2 = Karman_points["Bottom"][0]
                      )
        
        lines_number += 1
        Domain_lines["Down_left"] = lines_number
        fig.Add_line(line_number = lines_number,
                      P1 = Karman_points["Bottom"][0],
                      P2 = meshing_points["domain_downleft"]
                      )
        
        lines_number += 1
        Domain_lines["Left"] = lines_number
        fig.Add_line(line_number = lines_number,
                      P1 = meshing_points["domain_downleft"],
                      P2 = meshing_points["domain_upleft"]
                      )
        
        
        # ## Add the lines of the right block domain
        # geometry_file.write("// Define the right block domain line\n")
        
        # lines_number += 1
        # Domain_lines["Karman_down"] = lines_number
        # fig.Add_line(line_number = lines_number,
        #               P2 = Karman_points["Bottom"][0],
        #               P1 = meshing_points["domain_downright"]
        #               ) 
        
        # lines_number += 1
        # Domain_lines["Karman_right"] = lines_number
        # fig.Add_line(line_number = lines_number,
        #               P2 = meshing_points["domain_downright"],
        #               P1 = meshing_points["domain_upright"]
        #               ) 
        
        # lines_number += 1
        # Domain_lines["Karman_top"] = lines_number
        # fig.Add_line(line_number = lines_number,
        #               P2 = meshing_points["domain_upright"],
        #               P1 = Karman_points["Top"][0]
        #               ) 
    
        
        geometry_file.write("\n\n")


        geometry_file.write("// Define the Line Loop\n")
        LinesLoop = {}
        if boundary_layer:
            #CylinderLoops
            lines_number += 1
            LinesLoop["FirstQuarter"] = lines_number
            Lines = [
                        Circle_lines["FirstQuarter"],
                        -Domain_lines["Inner_left"],
                        -Circle_lines["Outer_FirstQuarter"],
                        Domain_lines["Inner_up"]           
                    ]
            fig.Add_LineLoop(   
                                line_number = lines_number,
                                line_list = Lines
                            )
            
            lines_number += 1
            LinesLoop["SecondQuarter"] = lines_number
            Lines = [
                        Circle_lines["SecondQuarter"],
                        -Domain_lines["Inner_down"],
                        -Circle_lines["Outer_SecondQuarter"],
                        Domain_lines["Inner_left"]             
                    ]
            fig.Add_LineLoop(   
                                line_number = lines_number,
                                line_list = Lines
                            )
            
            lines_number += 1
            LinesLoop["ThirdQuarter"] = lines_number
            Lines = [
                        Circle_lines["ThirdQuarter"],
                        -Domain_lines["Inner_right"],
                        -Circle_lines["Outer_ThirdQuarter"],
                        Domain_lines["Inner_down"]             
                    ]
            fig.Add_LineLoop(   
                                line_number = lines_number,
                                line_list = Lines
                            )
            
            lines_number += 1
            LinesLoop["LastQuarter"] = lines_number
            Lines = [
                        Circle_lines["LastQuarter"],
                        -Domain_lines["Inner_up"],
                        -Circle_lines["Outer_LastQuarter"],
                        Domain_lines["Inner_right"]              
                    ]
            fig.Add_LineLoop(   
                                line_number = lines_number,
                                line_list = Lines
                            )        
        else:
            Jets_lines = []
            Cylinder_lines = []
            
            # print(Circle_lines)
            
            actual_case = 'none'
            # number_of_jets = 0
            actual_line_number = 0
            last_value = 0
            temp_key = list(Circle_lines)
            first_key = temp_key[0]
            last_key = temp_key[-1]
            for key, value in Circle_lines.items():
                last_value = value
                # print('key:',key)
                if key[:7] == 'Jet_ss_':
                    Jets_lines.append([value])
                    last_value = actual_line_number
                    del actual_line_number
                    actual_line_number = value
                    lines = list(range(last_value+1, actual_line_number))
                    Cylinder_lines.append(lines)
                    # print(Jets_lines)
                    # print(Cylinder_lines)
                    actual_case = 'done'
                elif key[:7] == 'Jet_st_':
                    if actual_case == 'none':
                        # print(actual_case)
                        if key != last_key:
                            actual_line_number = value
                        actual_case = 'jet'
                        # number_of_jets += 1
                    elif actual_case == 'jet':
                        lines = list(range(actual_line_number,value))
                        # print('jet:',lines)
                        Jets_lines.append(lines)
                        if key != last_key:
                            actual_line_number = value
                        actual_case = 'cylinder'
                        
                    elif actual_case == 'cylinder':
                        lines = list(range(actual_line_number,value))
                        # print('cylinder:',lines)
                        Cylinder_lines.append(lines)
                        if key != last_key:
                            actual_line_number = value
                        actual_case = 'none'
                else:
                    print('error')
            # print('sortie dict;actual_line:',actual_line_number,'lastvalue',last_value)
            
            if actual_case == 'jet':
                lines = list(range(actual_line_number,last_value))
                # print('jet:',lines)
                Jets_lines.append(lines)
                actual_line_number = last_value
                actual_case = 'cylinder'
                    
            elif actual_case == 'cylinder':
                lines = list(range(actual_line_number,last_value))
                # print('cylinder:',lines)
                Cylinder_lines.append(lines)
                actual_line_number = last_value
                actual_case = 'none'
                
            elif actual_case == 'done':
                pass
                
            if actual_case == 'jet':
                lines = [actual_line_number,Circle_lines[last_key]+1]
                # print('jet:',lines)
                Jets_lines.append(lines)

                    
            elif actual_case == 'cylinder':
                lines = [actual_line_number,Circle_lines[last_key]+1]
                # print('cylinder:',lines)
                Cylinder_lines.append(lines)
            elif actual_case == 'done':
                lines = list(range(actual_line_number+1,len(Circle_points_number)))
                Cylinder_lines.append(lines)
                # pass
            # print('jets:',Jets_lines,'cyls:',Cylinder_lines)

        ## Add the Line Loop 
            lines_number += 1 # cylinder
            # print((Circle_lines[first_key],
            #                     Circle_lines[last_key]+2))
            LinesLoop["cylinder"] = lines_number
            lines = list( range(1,
                                len(Circle_points_number)) )
            
            fig.Add_LineLoop(   
                                    line_number = lines_number,
                                    line_list = lines
                                )

        lines_number += 1 #Domain_left
        LinesLoop["Domain"] = lines_number
        Lines = [
                    Domain_lines["Top_left"],
                    Domain_lines["Top_right"],
                    Domain_lines["Right"],
                    Domain_lines["Down_right"],
                    Domain_lines["Down_left"],
                    Domain_lines["Left"],
                    ]
        
        fig.Add_LineLoop(   
                            line_number = lines_number,
                            line_list = Lines
                        )
        
        # lines_number += 1 #Domain_right
        # LinesLoop["Domain_right"] = lines_number
        # Lines = [
        #             Domain_lines["Middle"],
        #             Domain_lines["Karman_top"],
        #             Domain_lines["Karman_right"],
        #             Domain_lines["Karman_down"]                  
        #         ]
        # fig.Add_LineLoop(   
        #                     line_number = lines_number,
        #                     line_list = Lines
        #                 )
        
        geometry_file.write("\n\n")

                
        ## Create the surfaces
        geometry_file.write("// Define the Surfaces\n")
        
        Surface_number = {}
        
        if boundary_layer:
            surface_number += 1 # First Quarter Surface
            Surface_number["FirstQuarter"] = surface_number
            fig.Add_PlaneSurface(surface_number = surface_number,
                                 Loop_number = (
                                                 LinesLoop["FirstQuarter"]
                                                )
                                 )
            
            surface_number += 1 # Second Quarter Surface
            Surface_number["SecondQuarter"] = surface_number
            fig.Add_PlaneSurface(surface_number = surface_number,
                                 Loop_number = (
                                                 LinesLoop["SecondQuarter"]
                                                )
                                 )
            
            surface_number += 1 # Third Quarter Surface
            Surface_number["ThirdQuarter"] = surface_number
            fig.Add_PlaneSurface(surface_number = surface_number,
                                 Loop_number = (
                                                 LinesLoop["ThirdQuarter"]
                                                )
                                 )
            
            surface_number += 1 # Last Quarter Surface
            Surface_number["LastQuarter"] = surface_number
            fig.Add_PlaneSurface(surface_number = surface_number,
                                 Loop_number = (
                                                 LinesLoop["LastQuarter"]
                                                )
                                 )
        else :
            
            for Loop in LinesLoop:
                # print(Loop)
                # print(LinesLoop[Loop])
                index = 0
                if Loop[:8] == 'cylinder':
                    index +=1
                    surface_number += 1 
                    Surface_number[Loop] = surface_number
                    
                if Loop[:3] == 'jet':
                    surface_number += 1
                    Surface_number[Loop] = surface_number
                    fig.Add_PlaneSurface(surface_number = surface_number,
                                         Loop_number = (
                                                         LinesLoop[Loop],
                                                         LinesLoop["Domain_left"]
                                                        )
                                         )
        surface_number += 1 # Left domain
        Surface_number["Domain_left"] = surface_number
        fig.Add_PlaneSurface(surface_number = surface_number,
                             Loop_number = (
                                             LinesLoop["cylinder"],
                                             LinesLoop["Domain"]
                                            )
                             )
        
        # surface_number += 1
        # Surface_number["Domain_right"] = surface_number
        # fig.Add_PlaneSurface(surface_number = surface_number,
        #                      Loop_number = LinesLoop["Domain_right"])
        
        geometry_file.write("\n\n")
        
        if boundary_layer:
            ## Mesh these surfaces in a structured manner
            geometry_file.write("// Mesh these surfaces in a structured manner\n")
            Surface_list = [
                            Surface_number["FirstQuarter"],
                            Surface_number["SecondQuarter"],
                            Surface_number["ThirdQuarter"],
                            Surface_number["LastQuarter"]
                            ]
            fig.Add_Transfinite_Surface(Surface_list = Surface_list)
            geometry_file.write("\n\n")
            
            
            ## Turn into quads
            geometry_file.write("// Turn into quads____________\n")
            fig.Add_Recombine_Surface(Surface_list)
            geometry_file.write("\n\n")
        
        ## Apply boundary conditions 
        
        
        
        if dimension == '2D':
            geometry_file.write("// Create Physical surface\n")
            
            if boundary_layer:
                inlet = [16]
                outlet = [18]
                cylinder_wall = [1, 4, 2, 3]
                jets = [6,8]
                top = [15, 19]
                bottom = [13, 17]
                fluid = [1, 4, 3, 2, 5, 6]

            else:
                inlet = [15]
                outlet = [12]
                cylinder_wall = []#[3,4,7,8]
                jets = []#[[1,2],[5,6]]
                for Line_list in Cylinder_lines:
                    for i in range(len(Line_list)):
                        cylinder_wall.append(Line_list[i])
                # print(cylinder_wall)
                jets = []
                for Line_list in Jets_lines:
                        jets.append(Line_list)
                # print(jets)                
                top = [10,11]
                bottom = [13,14]
                # kaarman_line = [11]
                fluid = [2,3]
                
            physical_number = 100
            
            fig.Add_Physical_Curve(name = "inlet",
                                   ID = False,
                                   Curve_list = inlet
                                   )
            physical_number += 1
            fig.Add_Physical_Curve(name = "outlet",
                                   ID = False,
                                   Curve_list = outlet
                                   )
            physical_number += 1
            fig.Add_Physical_Curve(name = "cylinder_wall",
                                   ID = False,
                                   Curve_list = cylinder_wall
                                   )
            physical_number += 1
            fig.Add_Physical_Curve(name = "top",
                                   ID = False,
                                   Curve_list = top
                                   )
            physical_number += 1
            fig.Add_Physical_Curve(name = "bottom",
                                   ID = False,
                                   Curve_list = bottom
                                   )
            
            # physical_number += 1
            # fig.Add_Physical_Curve(name = "kaarman_line",
            #                        ID = physical_number,
            #                        Curve_list = kaarman_line
            #                        )
            
            
            physical_number += 1
            fig.Add_Physical_Curve(name = "jet_bottom",
                                   ID = False,
                                   Curve_list = jets[0]
                                   )
            
            physical_number += 1
            fig.Add_Physical_Curve(name = "jet_top",
                                   ID = False,
                                   Curve_list = jets[1]
                                   )
            
            physical_number += 1
            fig.Add_Physical_Surface(name = "fluid",
                                     ID = False,
                                     Surface_list = fluid
                                     )
            

            
        elif dimension == '3D':
                                   
            from parameters import height
            #from parameters import Layers_number
            
            
            geometry_file.write("// Create 3D surface\n")
            Surfaces = list(range(1,len(Surface_number)+1))
            fig.Add_Extrude(extrude = height, 
                            Surface_list = Surfaces
                            )
            geometry_file.write("\n\n")
            
            
            geometry_file.write("// Create Physical surface\n")
            physical_number = 178
            fig.Add_Physical_Surface(name = "inlet",
                                     ID = physical_number,
                                     Surface_list = [126]
                                     )
            physical_number += 1
            fig.Add_Physical_Surface(name = "outlet",
                                     ID = physical_number,
                                     Surface_list = [172]
                                     )
            physical_number += 1
            fig.Add_Physical_Surface(name = "top",
                                     ID = physical_number,
                                     Surface_list = [138, 168]
                                     )
            physical_number += 1
            fig.Add_Physical_Surface(name = "bottom",
                                     ID = physical_number,
                                     Surface_list = [130, 176]
                                     )
            physical_number += 1
            fig.Add_Physical_Surface(name = "FrontAndBack",
                                     ID = physical_number,
                                     Surface_list = [1, 4, 3, 2, 5, 6,
                                                     47, 69, 91, 113, 155, 177]
                                     )
            physical_number += 1
            fig.Add_Physical_Surface(name = "cylinderWalls",
                                     ID = physical_number,
                                     Surface_list = [34, 100, 78, 56]
                                     )
            
            physical_number += 1
            fig.Add_Physical_Volume(name = "internal",
                                     ID = physical_number,
                                     Volume_list = Surfaces
                                     )

            geometry_file.write("\n\n")

        ## Close the file
        geometry_file.close()
 
    def airfoil(filepath):
        ## import section
        from parameters import sizeJet, sizeNaca, sizeWake, sizeFar, \
                               aoa, rotate_airfoil, Dict_domain, \
                               geometry_params
        
        ## open the file
        geometry_file = open(filepath, 'w')
        fig = geom_functions(file = geometry_file)
        
        ## Clear the file in case it is not empty
        fig.Clear_file()
       
        ## Parameters
        point_number = 0
        lines_number = 0
        loop_number = 0
        surface_number = 0
        physical_number = 0
        Domain_points = {}
        Airfoil_points = {}
        Jet_points = {}
        Domain_lines = {}
        Domain_loops = {}
        Domain_surfaces = {}
        jet_coordinate_x1 = geometry_params['jet_positions_x1']
        jet_coordinate_y1 = geometry_params['jet_positions_y1']
        jet_coordinate_x2 = geometry_params['jet_positions_x2']
        jet_coordinate_y2 = geometry_params['jet_positions_y2']
        jet_side = geometry_params['jet_side']
        n_jets = len(jet_coordinate_x1)
        
        
        ## Add the points of the domain
        geometry_file.write("// Define Domain Points\n")
        
        domain_points =     [
                            Dict_domain["downleft"],
                            Dict_domain["downright"],
                            Dict_domain["upright"],
                            Dict_domain["upleft"]
                            ]
        
        point_number += 1 # downleft
        Domain_points["domain_downleft"] = point_number
        fig.Add_point(point_number = point_number,
                      x = domain_points[0][0],
                      y = domain_points[0][1],
                      size = sizeFar)

        point_number += 1 # downright
        Domain_points["domain_downright"] = point_number
        fig.Add_point(point_number = point_number,
                      x = domain_points[1][0],
                      y = domain_points[1][1],
                      size = sizeFar)

        point_number += 1 # upright
        Domain_points["domain_upright"] = point_number
        fig.Add_point(point_number = point_number,
                      x = domain_points[2][0],
                      y = domain_points[2][1],
                      size = sizeFar)

        point_number += 1 # upleft
        Domain_points["domain_upleft"] = point_number
        fig.Add_point(point_number = point_number,
                      x = domain_points[3][0],
                      y = domain_points[3][1],
                      size = sizeFar)
        
        geometry_file.write("\n")
        
        
        ## Add the Airfoil
        ## Open the file with the airfoil coordinates
        with open("../alya_files/case/mesh/airfoil.dat", 'r') as airfoil_coordinates_file:
            airfoil_coordinate_x = []
            airfoil_coordinate_y = []
            for line in airfoil_coordinates_file:
                p = line.split()
                airfoil_coordinate_x.append(float(p[0]))
                airfoil_coordinate_y.append(float(p[1]))
        
        leading_edge_point = airfoil_coordinate_x.index(min(airfoil_coordinate_x))
        suction_jet_coordinate_x1 = []
        suction_jet_coordinate_y1 = []
        suction_jet_coordinate_x2 = []
        suction_jet_coordinate_y2 = []
        pressure_jet_coordinate_x1 = []
        pressure_jet_coordinate_y1 = []
        pressure_jet_coordinate_x2 = []
        pressure_jet_coordinate_y2 = []
        # Separate upper and lower jets
        for i in range(n_jets):
            Airfoil_points["Airfoil_sector_{}".format(i+1)] = []
            Jet_points["Jet_{}".format(i+1)] = []
            if jet_side[i] == 1:  # Suction side
                suction_jet_coordinate_x1.append(jet_coordinate_x1[i])
                suction_jet_coordinate_y1.append(jet_coordinate_y1[i])
                suction_jet_coordinate_x2.append(jet_coordinate_x2[i])
                suction_jet_coordinate_y2.append(jet_coordinate_y2[i])
            elif jet_side[i] == -1:  # Pressure side
                pressure_jet_coordinate_x1.append(jet_coordinate_x1[i])
                pressure_jet_coordinate_y1.append(jet_coordinate_y1[i])
                pressure_jet_coordinate_x2.append(jet_coordinate_x2[i])
                pressure_jet_coordinate_y2.append(jet_coordinate_y2[i])
        Airfoil_points["Airfoil_sector_{}".format(n_jets+1)] = []
        Airfoil_points["Airfoil_sector_{}".format(n_jets+2)] = []
        
        points_tolerance = 0.003  # TODO: this has to be introduced because of a bug in gmsh. To delete a point close but outside the jet
        suction_jet_coordinate_x1.append(airfoil_coordinate_x[leading_edge_point]-0.00001-points_tolerance)
        pressure_jet_coordinate_x1.append(airfoil_coordinate_x[-1]+0.00001+points_tolerance)
        
        # Write the suction side
        geometry_file.write("// Airfoil Points\n")
        jet_counter_suction = 0
        airfoil_counter_jump = 0
        for airfoil_counter in range(leading_edge_point + 1):
            if airfoil_counter_jump > airfoil_counter:  # Skip this point because it is inside or very close to the jet
                pass
            elif airfoil_coordinate_x[airfoil_counter] > suction_jet_coordinate_x1[jet_counter_suction] + points_tolerance:
                point_number += 1
                Airfoil_points["Airfoil_sector_{}".format(jet_counter_suction+1)].append(point_number)
                fig.Add_point(point_number = point_number,
                              x = airfoil_coordinate_x[airfoil_counter],
                              y = airfoil_coordinate_y[airfoil_counter],
                              size = sizeNaca)
                airfoil_counter_jump += 1
            else:
                # Now write the jet points
                geometry_file.write("// Points of Jet {}\n".format(jet_counter_suction + 1))
                point_number += 1
                Jet_points["Jet_{}".format(jet_counter_suction+1)].append(point_number)
                fig.Add_point(point_number = point_number,
                              x = suction_jet_coordinate_x1[jet_counter_suction],
                              y = suction_jet_coordinate_y1[jet_counter_suction],
                              size = sizeJet)
                
                point_number += 1
                Jet_points["Jet_{}".format(jet_counter_suction+1)].append(point_number)
                fig.Add_point(point_number = point_number,
                              x = suction_jet_coordinate_x2[jet_counter_suction],
                              y = suction_jet_coordinate_y2[jet_counter_suction],
                              size = sizeJet)
                
                # skip possible airfoil point that are inside the jet line
                while airfoil_coordinate_x[airfoil_counter_jump] > suction_jet_coordinate_x2[jet_counter_suction] - points_tolerance:
                    airfoil_counter_jump += 1
                    
                geometry_file.write("// Airfoil Points\n")
                jet_counter_suction += 1
                point_number += 1
                Airfoil_points["Airfoil_sector_{}".format(jet_counter_suction+1)].append(point_number)
                fig.Add_point(point_number = point_number,
                              x = airfoil_coordinate_x[airfoil_counter_jump],
                              y = airfoil_coordinate_y[airfoil_counter_jump],
                              size = sizeNaca)
                airfoil_counter_jump += 1
                
        # Write the pressure side
        jet_counter_pressure = 0
        airfoil_counter_jump = leading_edge_point+1
        for airfoil_counter in range(leading_edge_point+1,len(airfoil_coordinate_x)):
            if airfoil_counter_jump > airfoil_counter:  # Skip this point because it is inside or very close to the jet
                pass
            elif airfoil_coordinate_x[airfoil_counter] < pressure_jet_coordinate_x1[jet_counter_pressure] - points_tolerance:
                point_number += 1
                Airfoil_points["Airfoil_sector_{}".format(jet_counter_suction+jet_counter_pressure+2)].append(point_number)
                fig.Add_point(point_number = point_number,
                              x = airfoil_coordinate_x[airfoil_counter],
                              y = airfoil_coordinate_y[airfoil_counter],
                              size = sizeNaca)
                airfoil_counter_jump += 1
            else:
                # Now write the jet points
                geometry_file.write("// Points of Jet {}\n".format(jet_counter_suction + jet_counter_pressure + 1))
                point_number += 1
                Jet_points["Jet_{}".format(jet_counter_suction+jet_counter_pressure+1)].append(point_number)
                fig.Add_point(point_number = point_number,
                              x = pressure_jet_coordinate_x1[jet_counter_pressure],
                              y = pressure_jet_coordinate_y1[jet_counter_pressure],
                              size = sizeJet)
                
                point_number += 1
                Jet_points["Jet_{}".format(jet_counter_suction+jet_counter_pressure+1)].append(point_number)
                fig.Add_point(point_number = point_number,
                              x = pressure_jet_coordinate_x2[jet_counter_pressure],
                              y = pressure_jet_coordinate_y2[jet_counter_pressure],
                              size = sizeJet)
                
                # skip possible airfoil point that are inside the jet line
                while airfoil_coordinate_x[airfoil_counter_jump] < pressure_jet_coordinate_x2[jet_counter_pressure] + points_tolerance:
                    airfoil_counter_jump += 1
                    
                geometry_file.write("// Airfoil Points\n")
                jet_counter_pressure += 1
                point_number += 1
                Airfoil_points["Airfoil_sector_{}".format(jet_counter_suction+jet_counter_pressure+2)].append(point_number)
                fig.Add_point(point_number = point_number,
                              x = airfoil_coordinate_x[airfoil_counter_jump],
                              y = airfoil_coordinate_y[airfoil_counter_jump],
                              size = sizeNaca)
                airfoil_counter_jump += 1
                
                
        geometry_file.write("\n")
        
        
        ## Add the auxiliar box to refine the wake
        geometry_file.write("// Define auxiliar box\n")
        ## Open the file with the auxiliar box coordinates
        with open("../alya_files/case/mesh/auxiliar_box.dat", 'r') as auxiliar_box_coordinates_file:
            auxiliar_box_coordinate_x = []
            auxiliar_box_coordinate_y = []
            for line in auxiliar_box_coordinates_file:
                p = line.split()
                auxiliar_box_coordinate_x.append(float(p[0]))
                auxiliar_box_coordinate_y.append(float(p[1]))
        
        Auxiliar_box_points = []
        
        for point in range(len(auxiliar_box_coordinate_x)):
            point_number += 1
            Auxiliar_box_points.append(point_number)
            fig.Add_point(point_number = point_number,
                          x = auxiliar_box_coordinate_x[point],
                          y = auxiliar_box_coordinate_y[point],
                          size = sizeWake)
        geometry_file.write("\n")
        
        
        if rotate_airfoil == 0:
            ## Rotate the auxiliar box if rotate_airfoil = 0
            geometry_file.write("// Rotate auxiliar box\n")
            points_rotate = str(Auxiliar_box_points[0])+"}; "
            for point_rotate in Auxiliar_box_points[1:-1]:
                points_rotate += "Point{"+str(point_rotate)+"}; "
            points_rotate += "Point{"+str(Auxiliar_box_points[-1])
            fig.Rotate_point_2D(points_rotate, aoa*np.pi/180)
        else:
            ## Rotate the airfoil if rotate_airfoil = 1
            geometry_file.write("// Rotate airfoil\n")
            points_rotate = str(Airfoil_points["Airfoil_sector_1"][0])+"}; "
            for i in range(1,len(Airfoil_points)+1):
                if i > 1:
                    points_rotate += "Point{"+str(Airfoil_points["Airfoil_sector_{}".format(i)][0])+"}; "
                for j in range(1,len(Airfoil_points["Airfoil_sector_{}".format(i)])-1):
                    points_rotate += "Point{"+str(Airfoil_points["Airfoil_sector_{}".format(i)][j])+"}; "
                if i < len(Airfoil_points):
                    points_rotate += "Point{"+str(Airfoil_points["Airfoil_sector_{}".format(i)][-1])+"}; "
            points_rotate += "Point{"+str(Airfoil_points["Airfoil_sector_{}".format(len(Airfoil_points))][-1])
            for i in range(1,n_jets+1):
                points_rotate += "}; Point{"+str(Jet_points["Jet_{}".format(i)][0])
                points_rotate += "}; Point{"+str(Jet_points["Jet_{}".format(i)][1])
                
            fig.Rotate_point_2D(points_rotate, -aoa*np.pi/180)
            
        geometry_file.write("\n")
        
        
        ## Add the lines of the block domain
        geometry_file.write("// Define the domain lines\n")
        
        Domain_lines = {}
        lines_number += 1
        Domain_lines["Down"] = lines_number
        fig.Add_line(line_number = lines_number,
                      P1 = Domain_points["domain_downleft"],
                      P2 = Domain_points["domain_downright"])
        
        lines_number += 1
        Domain_lines["Right"] = lines_number
        fig.Add_line(line_number = lines_number,
                      P1 = Domain_points["domain_downright"],
                      P2 = Domain_points["domain_upright"])
        
        lines_number += 1
        Domain_lines["Up"] = lines_number
        fig.Add_line(line_number = lines_number,
                      P1 = Domain_points["domain_upright"],
                      P2 = Domain_points["domain_upleft"])
        
        lines_number += 1
        Domain_lines["Left"] = lines_number
        fig.Add_line(line_number = lines_number,
                      P1 = Domain_points["domain_upleft"],
                      P2 = Domain_points["domain_downleft"])
        geometry_file.write("\n")
        
        
        ## Add the airfoil splines 
        geometry_file.write("// Define the airfoil splines\n")
        # Suction side
        for i in range(1,jet_counter_suction+2):
            lines_number += 1
            Domain_lines["Airfoil_sector_{}".format(i)] = lines_number
            if i == 1:  # First sector of the suction side
                spline = Airfoil_points["Airfoil_sector_1"]
                if jet_counter_suction > 0:
                    spline.append(Jet_points["Jet_1"][0])
            elif i < jet_counter_suction+1:  # Middle sector sof the suction side
                spline = [Jet_points["Jet_{}".format(i-1)][1]]
                for j in range(len(Airfoil_points["Airfoil_sector_{}".format(i)])):
                    spline.append(Airfoil_points["Airfoil_sector_{}".format(i)][j])
                spline.append(Jet_points["Jet_{}".format(i)][0])
            elif i == jet_counter_suction+1:  # Last sector of the suction side
                spline = [Jet_points["Jet_{}".format(i-1)][1]]
                for j in range(len(Airfoil_points["Airfoil_sector_{}".format(i)])):
                    spline.append(Airfoil_points["Airfoil_sector_{}".format(i)][j])
            fig.Add_spline(line_number = lines_number, spline = spline)
        # Pressure side
        for i in range(jet_counter_suction+2,jet_counter_suction+jet_counter_pressure+3):
            lines_number += 1
            Domain_lines["Airfoil_sector_{}".format(i)] = lines_number
            if i == jet_counter_suction+2:  # First sector of the pressure side
                spline = [Airfoil_points["Airfoil_sector_{}".format(i-1)][-1]]
                for j in range(len(Airfoil_points["Airfoil_sector_{}".format(i)])):
                    spline.append(Airfoil_points["Airfoil_sector_{}".format(i)][j])
                if jet_counter_pressure > 0:
                    spline.append(Jet_points["Jet_{}".format(jet_counter_suction+1)][0])
                else:
                    spline.append(Airfoil_points["Airfoil_sector_1"][0])
            elif i < jet_counter_suction+jet_counter_pressure+2:  # Middle sector sof the pressure side
                spline = [Jet_points["Jet_{}".format(i-2)][1]]
                for j in range(len(Airfoil_points["Airfoil_sector_{}".format(i)])):
                    spline.append(Airfoil_points["Airfoil_sector_{}".format(i)][j])
                spline.append(Jet_points["Jet_{}".format(i-1)][0])
            elif i == jet_counter_suction+jet_counter_pressure+2:  # Last sector of the pressure side
                spline = [Jet_points["Jet_{}".format(i-2)][1]]
                for j in range(len(Airfoil_points["Airfoil_sector_{}".format(i)])):
                    spline.append(Airfoil_points["Airfoil_sector_{}".format(i)][j])
                spline.append(Airfoil_points["Airfoil_sector_1"][0])
            fig.Add_spline(line_number = lines_number, spline = spline)
        
        geometry_file.write("\n")
        
        ## Add the jet lines
        geometry_file.write("// Define the jet lines\n")
        for i in range(1,n_jets+1):
            lines_number += 1
            Domain_lines["Jet_{}".format(i)] = lines_number
            fig.Add_line(line_number = lines_number,
                         P1 = Jet_points["Jet_{}".format(i)][0],
                         P2 = Jet_points["Jet_{}".format(i)][1])
        
        geometry_file.write("\n")
        
        ## Add the splines of the auxiliar box
        geometry_file.write("// Define the auxiliar box splines\n")
        
        # Curved part
        lines_number += 1
        Domain_lines["auxiliar_curved"] = lines_number
        fig.Add_spline(line_number = lines_number, spline = Auxiliar_box_points[0:-2])
        # Straight bottom part
        lines_number += 1
        spline_points = [Auxiliar_box_points[-3]]
        spline_points.append(Auxiliar_box_points[-1])
        Domain_lines["auxiliar_bottom"] = lines_number
        fig.Add_spline(line_number = lines_number, spline = spline_points)
        # Straight right part
        lines_number += 1
        spline_points = [Auxiliar_box_points[-1]]
        spline_points.append(Auxiliar_box_points[-2])
        Domain_lines["auxiliar_right"] = lines_number
        fig.Add_spline(line_number = lines_number, spline = spline_points)
        # Straight upper part
        lines_number += 1
        spline_points = [Auxiliar_box_points[-2]]
        spline_points.append(Auxiliar_box_points[0])
        Domain_lines["auxiliar_up"] = lines_number
        fig.Add_spline(line_number = lines_number, spline = spline_points)
        
        geometry_file.write("\n")
        
        
        ## Add the line loops        
        geometry_file.write("// Define the line loops\n")
        
        # Computational domain
        loop_number += 1
        Domain_loops["Computational_domain"] = loop_number
        Lines = [Domain_lines["Down"], Domain_lines["Right"],
                 Domain_lines["Up"]  , Domain_lines["Left"]]
        fig.Add_CurveLoop(curve_number = loop_number,
                         line_list = Lines)
        
        # Auxiliar box
        loop_number += 1
        Domain_loops["Auxiliar_box"] = loop_number
        Lines = [Domain_lines["auxiliar_curved"], Domain_lines["auxiliar_bottom"],
                 Domain_lines["auxiliar_right"] , Domain_lines["auxiliar_up"]]
        fig.Add_CurveLoop(curve_number = loop_number,
                         line_list = Lines)
        
        # Airfoil
        loop_number += 1
        Domain_loops["Airfoil"] = loop_number
        Lines = []
        for i in range(1,jet_counter_suction+1):
            Lines.append(Domain_lines["Airfoil_sector_{}".format(i)])
            Lines.append(Domain_lines["Jet_{}".format(i)])
        Lines.append(Domain_lines["Airfoil_sector_{}".format(jet_counter_suction+1)])
        for i in range(jet_counter_suction+1,jet_counter_suction+jet_counter_pressure+1):
            Lines.append(Domain_lines["Airfoil_sector_{}".format(i+1)])
            Lines.append(Domain_lines["Jet_{}".format(i)])
        Lines.append(Domain_lines["Airfoil_sector_{}".format(jet_counter_suction+jet_counter_pressure+2)])
        fig.Add_CurveLoop(curve_number = loop_number,
                         line_list = Lines)
        
        geometry_file.write("\n")
        
        
        ## Create the surfaces
        geometry_file.write("// Define the surfaces\n")
        
        surface_number += 1
        Domain_surfaces["Far_flow"] = surface_number
        fig.Add_PlaneSurface(surface_number = surface_number,
                             Loop_number = (Domain_loops["Computational_domain"],
                                            Domain_loops["Auxiliar_box"]))
        
        surface_number += 1
        Domain_surfaces["Close_flow"] = surface_number
        fig.Add_PlaneSurface(surface_number = surface_number,
                             Loop_number = (Domain_loops["Auxiliar_box"],
                                            Domain_loops["Airfoil"]))
        
        geometry_file.write("\n")
        
        
        ## Create physical properties
        geometry_file.write("// Create Physical properties\n")
        
        if rotate_airfoil == 0:
            if aoa >= 0:
                # In this case left and bottom are inlet, while top and right
                # are outlet
                # Inlet
                physical_number += 1
                fig.Add_Physical_Curve(name = "inlet", ID = physical_number,
                                       Curve_list = [Domain_lines["Left"],
                                                     Domain_lines["Down"]])
                # Outlet
                physical_number += 1
                fig.Add_Physical_Curve(name = "outlet", ID = physical_number,
                                       Curve_list = [Domain_lines["Right"],
                                                     Domain_lines["Up"]])
            else:
                # In this case left and top are inlet, while top and bottom
                # are outlet
                # Inlet
                physical_number += 1
                fig.Add_Physical_Curve(name = "inlet", ID = physical_number,
                                       Curve_list = [Domain_lines["Left"],
                                                     Domain_lines["Up"]])
                # Outlet
                physical_number += 1
                fig.Add_Physical_Curve(name = "outlet", ID = physical_number,
                                       Curve_list = [Domain_lines["Right"],
                                                     Domain_lines["Down"]])
        else:
            # In this case only left is inlet and right outlet, top and bottom
            # are free flow
            # Inlet
            physical_number += 1
            fig.Add_Physical_Curve(name = "inlet", ID = physical_number,
                                   Curve_list = [Domain_lines["Left"]])
            # Outlet
            physical_number += 1
            fig.Add_Physical_Curve(name = "outlet", ID = physical_number,
                                   Curve_list = [Domain_lines["Right"]])
            # Top
            physical_number += 1
            fig.Add_Physical_Curve(name = "upper", ID = physical_number,
                                   Curve_list = [Domain_lines["Up"]])
            # Bottom
            physical_number += 1
            fig.Add_Physical_Curve(name = "bottom", ID = physical_number,
                                   Curve_list = [Domain_lines["Down"]])
            
        # Airfoil
        physical_number = 5
        Curves = []
        for i in range(1,len(Airfoil_points)+1):
            Curves.append(Domain_lines["Airfoil_sector_{}".format(i)])
        fig.Add_Physical_Curve(name = "airfoil", ID = physical_number,
                               Curve_list = Curves)
        
        # Jets
        for i in range(1,n_jets+1):
            physical_number += 1
            fig.Add_Physical_Curve(name = "jet_{}".format(i), ID = physical_number,
                                   Curve_list = [Domain_lines["Jet_{}".format(i)]])
        
        # Fluid
        physical_number += 1
        fig.Add_Physical_Surface(name = "fluid", ID = physical_number,
                               Surface_list = [Domain_surfaces["Far_flow"],
                                               Domain_surfaces["Close_flow"]])
        
        geometry_file.write("\n")
        
        
        ## Write the mesh format (needed for Alya)
        geometry_file.write("Mesh.MshFileVersion = 2.2;\n")
        
        
        ## Close the file
        geometry_file.close()
        
        
def main():
    from parameters import case
    filepath = "../alya_files/case/mesh/{}.geo".format(case)
    
    if case == 'bubble':
        figure_functions.polygon(filepath)
    elif case == "cylinder":
        figure_functions.cylinder(filepath)
    elif case == 'airfoil':
        figure_functions.airfoil(filepath)
    
    raw_dimension = dimension[0]
    os.system('gmsh -{0} ../alya_files/case/mesh/{1}.geo -o ../alya_files/case/mesh/{1}.msh -format msh2'.format(raw_dimension, case))
    
## Run
if __name__ == "__main__":
   main()

