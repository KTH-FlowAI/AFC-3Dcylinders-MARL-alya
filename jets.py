#!/bin/env python
#
# DEEP REINFORCEMENT LEARNING WITH ALYA
#
# Maxence Deferrez, Pol Suarez, Arnau Miro
# 07/07/2022
from __future__ import print_function, division

import os, numpy as np

from alya       import write_jet_file



# Function to build and return the jets
def build_jets(jet_class,jets_definition,delta_t_smooth):
    '''
    This helper function is used to build and return the dictionary that
    contains information on the jets.
    For that one has to give the kind of jet class (directly the class object)
    as well as the jet names and jet geometric parameters.
    '''	
    names = list(jets_definition.keys())
    # Build jets dictionary
    jets = {}
    for name in names:
        jets[name] = jet_class(name, jets_definition[name], T_smoo=delta_t_smooth)
    return jets

# Function to write atan2 as a string
def atan2_str(X,Y):
    return '2*atan({1}/({0} + sqrt({0}^2+{1}^2)))'.format(X,Y)

# Smoothing functions
def Q_smooth_linear(Qnew,Qpre,timestart,Tsmooth):
    '''
    Linear smoothing law:
        Q(t) = (Qn - Qs)*(t - ts)/Tsmooth + Qs
    '''
    deltaQ = Qnew - Qpre
    return '({0}/{1}*(t-{2}) + ({3}))'.format(deltaQ, Tsmooth, timestart, Qpre) 


def Q_smooth_exp(ts,Tsmooth):
    '''
        Exponential smoothing law: from (https://en.wikipedia.org/wiki/Non-analytic_smooth_function#Smooth_transition_functions)

        f(x) = e^(-1/x) if x > 0
             = 0        if x <= 0

        Between two points:

        'x' => (x-a)/(b-a)

        g(x) = f(x)/(f(x) + f(1-x))

    '''
    t1 = ts
    t2 = ts + Tsmooth
    
    xp = '(pos((t-%.2f)/%.2f))' % (t1,t2-t1)
    f1 = 'exp(-1/%s)'%xp
    f2 = 'exp(-1/pos(1-%s))'%xp
    h  = '%s/(%s+%s)' % (f1,f1,f2)

    #return '((%f) + ((%s)*(%f)))' % (Q1,h,Q2-Q1)
    return h

def heav_func(position_z, delta_z):
    '''
    Define the heaviside function in spanwise to change the Q in diferent locations at z axis
    takes de z position and activates the Q inside range [z-delta,z+delta]
    '''        
    return 'heav((z-%.3f)*(%.3f-z))' %(position_z-delta_z*0.5, position_z+delta_z*0.5)

    
class Jet(object):
    '''
    Parent class to implement jets on the DRL.

    Implements a generic class constructor which calls specialized functions from
    children classes in order to set up the jet.

    It also implements the following generic methods:
        - 
    '''
    def __init__(self,name, params, Q_pre = 0., Q_new = 0., time_start = 0., dimension = 2, T_smoo = 0.2, smooth_func = ""):
        '''
        Class initializer, generic.
        Sets up the basic parameters and starts the class.
        After creating the class we should initialize the geometry manually for
        each of the specialized jets.
        '''
        from parameters import dimension, Qs_position_z, delta_Q_z, short_spacetime_func, nb_inv_per_CFD

        # Basic jet variables
        self.name      = name
        self.T_smoo    = T_smoo
        self.dimension = dimension
        self.theta     = 0 # to be updated during DRL
        # Jet velocity functions (to be updated during DRL)
        self.Vx        = ''
        self.Vy        = ''
        self.Vz        = ''
        # Call specialized method to set up the jet geometry
        self.set_geometry(params)
        # Update to this current timestep
        self.Qs_position_z = Qs_position_z
        self.delta_Q_z = delta_Q_z
        self.short_spacetime_func =  short_spacetime_func
        self.nb_inv_per_CFD       =  nb_inv_per_CFD
        self.update(Q_pre,Q_new,time_start,smooth_func)

    def update(self,Q_pre,Q_new,time_start,smooth_func):
        '''
        Updates a jet for a given epoch of the DRL, generic.
        Calls the specialized method smoothfunc to set up the get geometry
        per each of the child classes.
        '''
        # Up
        self.Q_pre      = Q_pre
        self.Q_new      = Q_new
        self.time_start = time_start
        self.smooth_func = smooth_func 
        # Call the specialized method that creates a smoothing function for the current time
        smooth_fun      = self.create_smooth_funcs(self.Q_new, self.Q_pre, self.time_start, self.T_smoo, self.smooth_func, self.Qs_position_z, self.delta_Q_z)
        # Create the velocities using the smoothing functions, 
        if self.dimension == 2:
            # For 2D jets set Vx and Vy
            self.Vx         = '{0}*cos({1})'.format(smooth_fun, self.theta)
            self.Vy         = '{0}*sin({1})'.format(smooth_fun, self.theta)
        else:
            # For 3D jets raise an error
            self.Vx         = '{0}*cos({1})'.format(smooth_fun, self.theta)
            self.Vy         = '{0}*abs(sin({1}))'.format(smooth_fun, self.theta) #TODO: temporal fix for component y (not opposite? check update_jet)
            self.Vz         = '0'

    def update_file(self,filepath):
        '''
        Replaces the jets path file for a new one, generic.
        The name of the file must be the same of that of the jet.
        '''
        functions = [self.Vx,self.Vy] if self.dimension == 2 else [self.Vx,self.Vy,self.Vz]
        write_jet_file(filepath,self.name,functions)

    def set_geometry(self,geometry_params):
        '''
        Placeholder for specialized function that sets the jet geometry
        per each of the independent cases.
        '''
        raise NotImplementedError('Must specialize this method for each specific jet kind')

    def create_smooth_funcs(self, Q_new, Q_pre, time_start, T_smoo):
        '''
        Placeholder for specialized function that sets the jet geometry
        per each of the independent cases.
        '''
        raise NotImplementedError('Must specialize this method for each specific jet kind')        


class JetCylinder(Jet):
    '''
    Specialized jet class to deal with jets specificed in cylindrical coordinates.
    '''
    def set_geometry(self,params):
        '''
        Specialized method that sets up the geometry of the jet
        '''
        from parameters import cylinder_coordinates, Qs_position_z, delta_Q_z
        # Sanity check
        # TODO: asserts are dangerous... we need a function that stops everything!!
        if params['width']  <= 0.:          raise ValueError('Invalid jet width=%f' %params['width'])
        if params['radius'] <= 0.:          raise ValueError('Invalid jet radius=%f'%params['radius'])
        if params['positions_angle'] <= 0.: raise ValueError('Invalid jet angle=%f' %params['positions_angle'])
        # Recover parameters from dictionary
        self.radius = params['radius']
        self.width  = params['width']
        self.theta0 = self.normalize_angle(np.deg2rad(params['positions_angle']))
        self.theta  = self.get_theta(cylinder_coordinates)

    def create_smooth_funcs(self, Q_new, Q_pre, time_start, T_smoo, smooth_func, Qs_position_z, delta_Q_z):
        '''
        Specialized method that creates the smooth functions
        '''
        w        = self.width*(np.pi/180) # deg2rad
        scale    = np.pi/(2.*w*self.radius)  #### FIX: NOT R**2 --> D

        string_all_Q_pre = "0"
        string_all_Q_new = "0"
        string_heav      = ""

        if smooth_func == "EXPONENTIAL":            

            ## Q_pre and Q_new --> list! with nz_Qs dimensions
            string_h = Q_smooth_exp(time_start, T_smoo)

            # create the new Q string 
            string_heav = heav_func(Qs_position_z[0], delta_Q_z)
            string_all_Q_pre = "%s*(%.4f)" % (string_heav, Q_pre[0])
            string_all_Q_new = "%s*(%.4f)" % (string_heav, Q_new[0])

            for i in range(1,self.nb_inv_per_CFD):
                string_heav = heav_func(Qs_position_z[i], delta_Q_z)
                string_all_Q_pre += "+ %s*(%.4f)" % (string_heav, Q_pre[i])
                string_all_Q_new += "+ %s*(%.4f)" % (string_heav, Q_new[i])
            string_Q = '((%s) + (%s)*((%s)-(%s)))' % (string_all_Q_pre, string_h, string_all_Q_new, string_all_Q_pre)

        else:
            string_Q = Q_smooth_linear(Q_new, Q_pre, time_start, T_smoo)

        if self.short_spacetime_func == True:
            #just with Qnorm*Qi -- no projection or smoothing in time/space
            return '(%.1f)*(%s)'%(scale, string_all_Q_new)
        else: 
            string_C = 'cos(%.3f/%.3f*(%s-(%.3f)))'%(np.pi, w, self.theta, self.theta0)
            return '(%.1f)*(%s)*(%s)' %(scale, string_Q, string_C)

    @staticmethod
    def normalize_angle(angle):
        '''
        Normalize angle between [-pi,pi]
        '''
        # TODO: check this... not very clear to me
        if angle > np.pi:    angle -= 2*np.pi
        if angle < 2.*np.pi: angle = -((2.*np.pi) - angle)
        return angle
	
    @staticmethod
    def get_theta(cylinder_coordinates):
        '''
        TODO: documentation!
        '''
        X = '(x - {0})'.format(cylinder_coordinates[0])
        Y = '(y - {0})'.format(cylinder_coordinates[1])
        return atan2_str(X,Y)


class JetAirfoil(Jet):
    '''
    Specialized jet class to deal with jets specificed in cartesian coordinates.
    '''
    def set_geometry(self,params):
        '''
        Specialized method that sets up the geometry of the jet
        '''
        from parameters import rotate_airfoil, aoa
        
        # Get jet positions
        self.x1 = params['x1']
        self.x2 = params['x2']
        self.y1 = params['y1']
        self.y2 = params['y2']

        if rotate_airfoil:
            self.x1 = self.x1*np.cos(np.deg2rad(aoa)) + self.y1*np.sin(np.deg2rad(aoa))
            self.y1 = self.y1*np.cos(np.deg2rad(aoa)) - self.x1*np.sin(np.deg2rad(aoa))
            self.x2 = self.x2*np.cos(np.deg2rad(aoa)) + self.y2*np.sin(np.deg2rad(aoa))
            self.y2 = self.y2*np.cos(np.deg2rad(aoa)) - self.x2*np.sin(np.deg2rad(aoa))
        
        # Get the angle of the slope normal to the surface
        self.theta = self.get_slope(self)

    def create_smooth_funcs(self, Q_new, Q_pre, time_start, T_smoo, smooth_func):
        '''
        Specialized method that creates the smooth functions
        '''
        w        = np.sqrt((self.x1-self.x2)**2 + (self.y1-self.y2)**2)
        scale    = np.pi/(2*w)

        if smooth_func == "EXPONENTIAL":
            string_Q = Q_smooth_exp(Q_new, Q_pre, time_start, T_smoo)

        else:
            string_Q = Q_smooth_linear(Q_new, Q_pre, time_start, T_smoo)

        #delta_Q  = Q_new - Q_pre
        #string_Q = '{}*({}/{}*(t-{}) + ({}))'.format(scale, delta_Q, T_smoo, time_start, Q_pre) # Change this for Xavi's approach
        string_S = 'sin({}*(x-{})/({}-{}))'.format(np.pi, self.x1, self.x2, self.x1)
        return '({})*({})'.format(string_Q, string_S)

    @staticmethod
    def get_slope(self): 
        '''
        We are actually getting the angle of the slope
        ''' 
        X = '({}-({}))'.format(self.y2,self.y1)
        Y = '({}-({}))'.format(self.x1,self.x2)
        return atan2_str(X,Y)
