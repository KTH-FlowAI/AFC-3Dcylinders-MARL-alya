# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:13:42 2021

@author: Maxence
"""
import numpy as np

def angle(vector_1,vector_2): #TODO: This function is not being used
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arctan2(unit_vector_1,unit_vector_2)#np.arccos(dot_product)

def operation_list(L1,L2,operation='-'):
    L3 = []
    for i in range(len(L1)):
        if operation == '+':
            L3.append(L1[i]+L2[i])
        if operation == '-':
            L3.append(L1[i]-L2[i])
    return L3

def normalize_angle(angle):
    normalized_angle = angle
    if angle < 0:
        normalized_angle += 360
    return normalized_angle    

def without_key(d, key):
    return {x: d[x] for x in d if x not in key}
   
def add_angle_circle_points_dict(dict_circle_point, cylinder_coordinates):
    """ dict__circle_point = { 
                        [
                        ID,
                        coordinates                        
                        ]
                      }
    """
    
    horizontal_vector = [1,0]

    for key, value in dict_circle_point.items():
        if key == 'Center':
            actual_angle = None
        else:
            V1,V2 = operation_list(value[1],cylinder_coordinates),horizontal_vector
            actual_angle = np.arctan2(V1[1], V1[0]) - np.arctan2(V2[1], V2[0])
            actual_angle = actual_angle * 180/np.pi
        
            actual_angle = normalize_angle(actual_angle)
        
        dict_circle_point[key] += [actual_angle]

    sorted_x = without_key(d = dict_circle_point, key = {"Center"})
    
    sorted_x = reversed(sorted(sorted_x.items(), key=lambda kv: kv[1][2]))
    sorted_x = dict(sorted_x)

    center_value = dict_circle_point["Center"]
    sorted_x["Center"] = center_value

    return sorted_x



















