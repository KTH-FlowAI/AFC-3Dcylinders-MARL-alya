
import numpy as np
import math

from parameters import variational_input, case
if case == 'airfoil':
    from parameters import aoa, rotate_airfoil


### Read input data from file
def _readInputdata():
    bound = []
    variabound = []
    porous = False

    d = 0
    D_exp = 0
    S_exp = 0
    L_exp = 0
    R_exp = 0
    P_exp = 0
    Y_exp = 0
    u = [1,1,1]
    phi = 0
    sin_phi = 0
    cos_phi = 0

    filename = variational_input["filename"]
    bound = variational_input["bound"]
    porous = variational_input["porous"]
    
    density = variational_input["density"]
    veloc = variational_input["veloc"]
    scale_area = variational_input["scale_area"]
    d = variational_input["d"]
    time = variational_input["time"]
    initial_time = variational_input["initial_time"]
    
    rotation_vector = variational_input["rotation_vector"]
    phi = variational_input["phi"]
    
    D_exp = variational_input["D_exp"]
    S_exp = variational_input["S_exp"]
    L_exp = variational_input["L_exp"]
    R_exp = variational_input["R_exp"]
    P_exp = variational_input["P_exp"]
    Y_exp = variational_input["Y_exp"]

    if time > 0:
        initial_time = float(input('Initial time to postprocess:'));
        superTitle = 'Averaged from ' + str(abs(initial_time)) + ' to ' + str(abs(initial_time + time))
    elif time < 0:
        superTitle = 'Averaged over the last ' + str(abs(time))
    elif time == 0:
        superTitle = 'Averaged over all the runtime'
    vecro = np.asarray(rotation_vector)
    npvro = vecro.astype(np.float)
    u = npvro/(math.sqrt(((npvro[0])**2)+((npvro[1])**2)+((npvro[2])**2)))

    sin_phi = math.sin(math.radians(phi))
    cos_phi = math.cos(math.radians(phi))
    
    return (filename, bound, variabound, porous, density, veloc, scale_area, d, initial_time, time, superTitle, u, phi, sin_phi, cos_phi, D_exp, S_exp, L_exp, R_exp, P_exp, Y_exp)

# Read header
def _readHeader(entrada):
    F_visc_columns = []
    F_pres_columns = []
    F_vari_columns = []
    F_poro_columns = []
    M_visc_columns = []
    M_pres_columns = []

    for i in range(3):
        line = entrada.readline().strip().split()

    header = 0
    while line[1] != 'START':
        line = entrada.readline().strip().split()
        if line[1] == 'FORCE':
            F_visc_columns.append(int(line[5])-1)
            F_visc_columns.append(int(line[5]))
            F_visc_columns.append(int(line[5])+1)
        if line[1] == 'F_p_x':
            F_pres_columns.append(int(line[5])-1)
            F_pres_columns.append(int(line[5]))
            F_pres_columns.append(int(line[5])+1)
        if line[1] == 'INTFX':
            F_vari_columns.append(int(line[5])-1)
            F_vari_columns.append(int(line[5]))
            F_vari_columns.append(int(line[5])+1)
        if line[1] == 'FPORX':
            F_poro_columns.append(int(line[5])-1)
            F_poro_columns.append(int(line[5]))
            F_poro_columns.append(int(line[5])+1)
        if line[1] == 'TORQU':
            M_visc_columns.append(int(line[5])-1)
            M_visc_columns.append(int(line[5]))
            M_visc_columns.append(int(line[5])+1)
        if line[1] == 'T_p_x':
            M_pres_columns.append(int(line[5])-1)
            M_pres_columns.append(int(line[5]))
            M_pres_columns.append(int(line[5])+1)
        if line[1] == 'NUMSETS':
            totalBound = int(line[3])
        header+=1
    return (F_visc_columns, F_pres_columns, F_vari_columns, F_poro_columns, M_visc_columns, M_pres_columns, header, totalBound)

# Read File 
def _readFile(entrada, header):
    lines=entrada.readlines()
    entrada.close()
    nline=len(lines)
    fileLines=[]
    for i in range(2,nline):
        line = lines[i]
        line = line.split()
        fileLines.append(line)
    return (fileLines)

# Ṛead unique lines
def _uniqueLines(fileLines):
    steps = []
    index = []
    for i in range (5,len(fileLines)):
        if fileLines[i][1] == 'Iterations':
            if fileLines[i][3] != '0':
                step = int(fileLines[i][3])
                steps.append(step)
                index.append(i+1)
    npStep=np.asarray(steps)
    uniqueStep , uniqueIndex=np.unique(npStep, return_index=True)
    return (uniqueIndex, index)

# Time Arrays
def _timeArrays(uniqueIndex, index, fileLines, totalBound):
    time_steps = []
    accumulated_array =[]
    for i in range(len(uniqueIndex)):
        line = index[uniqueIndex[i]]
        time_step = float(fileLines[line][3]) - float(fileLines[line-2-totalBound][3])
        time_steps.append(time_step)
        accumulated_time = float(fileLines[line][3])
        accumulated_array.append(accumulated_time)
    return (time_steps, accumulated_array)

# Rotation fileLines
def _rotationMatrix(u, sin_phi, cos_phi):
    R = np.zeros((3,3))

    R[0,0] = cos_phi + (1 - cos_phi)*(u[0])**2
    R[1,1] = cos_phi + (1 - cos_phi)*(u[1])**2
    R[2,2] = cos_phi + (1 - cos_phi)*(u[2])**2

    R[0,1] = (1 - cos_phi)*u[0]*u[1] - sin_phi*u[2]
    R[1,2] = (1 - cos_phi)*u[1]*u[2] - sin_phi*u[0]
    R[2,0] = (1 - cos_phi)*u[2]*u[0] - sin_phi*u[1]

    R[0,2] = (1 - cos_phi)*u[0]*u[2] + sin_phi*u[1]
    R[1,0] = (1 - cos_phi)*u[1]*u[0] + sin_phi*u[2]
    R[2,1] = (1 - cos_phi)*u[2]*u[1] + sin_phi*u[0]

    return (R)

# Forces types
def _forcesTypes(bound, variabound, porous):
    typfo = 0
    if len(bound) != 0:
        typfo+=2
    if len(variabound)!= 0:
        typfo+=1
    if porous:
        typfo+=1
    return (typfo)

# Calcule forces and momentum coefficients
def _calculateFnM(uniqueIndex, fileLines, bound, variabound, F_visc_columns, F_pres_columns, F_vari_columns,
    F_poro_columns, M_visc_columns, M_pres_columns, phi, density, index, porous, typfo, scale_area, veloc, d,
    R, case):
    array_drag = []
    array_lift = []
    array_side = []

    array_roll  = []
    array_pitch = []
    array_yaw   = []

    for i in range (len(uniqueIndex)):
        line = index[uniqueIndex[i]]
        Forces = []
        Momentum =[]
        if len(bound) != 0:
            ViFor = [0,0,0]
            PrFor = [0,0,0]
            for j in range(len(bound)):
                jb = bound[j]
                ViFor[0] += (float(fileLines[line+jb][F_visc_columns[0]]))
                ViFor[1] += (float(fileLines[line+jb][F_visc_columns[1]]))
                ViFor[2] += (float(fileLines[line+jb][F_visc_columns[2]]))
                    
                PrFor[0] += (float(fileLines[line+jb][F_pres_columns[0]]))
                PrFor[1] += (float(fileLines[line+jb][F_pres_columns[1]]))
                PrFor[2] += (float(fileLines[line+jb][F_pres_columns[2]]))
            Forces.append(ViFor)
            Forces.append(PrFor)

        if len(variabound) != 0:
            VaFor = [0,0,0]
            for j in range(len(variabound)):
                jb = variabound[j]
                VaFor[0] += -(float(fileLines[line+jb][F_vari_columns[0]]))
                VaFor[1] += -(float(fileLines[line+jb][F_vari_columns[1]]))
                VaFor[2] += -(float(fileLines[line+jb][F_vari_columns[2]]))
            Forces.append(VaFor)
        
        if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
            if len(variabound) != 0 or len(bound) != 0:
                ViMom = [0,0,0]
                PrMom = [0,0,0]
                M_bound = bound + variabound
                for j in range(len(M_bound)):
                    jb = M_bound[j]
                    ViMom[0] += (float(fileLines[line+jb][M_visc_columns[0]]))
                    ViMom[1] += (float(fileLines[line+jb][M_visc_columns[1]]))
                    ViMom[2] += (float(fileLines[line+jb][M_visc_columns[2]]))
                        
                    PrMom[0] += (float(fileLines[line+jb][M_pres_columns[0]]))
                    PrMom[1] += (float(fileLines[line+jb][M_pres_columns[1]]))
                    PrMom[2] += (float(fileLines[line+jb][M_pres_columns[2]]))
                Momentum.append(ViMom)
                Momentum.append(PrMom)

        if porous:
            PoFor = [0,0,0]
            PoFor[0] += (float(fileLines[line+jb][F_poro_columns[0]]))
            PoFor[1] += (float(fileLines[line+jb][F_poro_columns[1]]))
            PoFor[2] += (float(fileLines[line+jb][F_poro_columns[2]]))

        FD = 0
        FS = 0
        FL = 0
        RM = 0
        PM = 0
        YM = 0

        for i in range(typfo):
            FD += Forces[i][0]
            FS += Forces[i][1]
            FL += Forces[i][2]

        drag = -FD*2/(float(density)*float(scale_area)*float(veloc)**2)
        side = -FS*2/(float(density)*float(scale_area)*float(veloc)**2)
        lift = -FL*2/(float(density)*float(scale_area)*float(veloc)**2)

        if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
            for i in range(2):
                RM += Momentum[i][0]
                PM += Momentum[i][1]
                YM += Momentum[i][2]

            roll  = -RM*2/(float(density)*float(scale_area)*float(veloc)**2)*d
            pitch = -PM*2/(float(density)*float(scale_area)*float(veloc)**2)*d
            yaw   = -YM*2/(float(density)*float(scale_area)*float(veloc)**2)*d

        if phi != 0:
            sumForces = np.array([drag, lift, side])
            drag, lift, side = R.dot(sumForces)
            if case == 'airfoil':
                if rotate_airfoil == 0:
                    drag_O = drag
                    lift_O = lift
                    drag = drag_O*np.cos(np.deg2rad(aoa)) + lift_O*np.sin(np.deg2rad(aoa))
                    lift = lift_O*np.cos(np.deg2rad(aoa)) - drag_O*np.sin(np.deg2rad(aoa))
                
            if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
                sumMomentum = np.array([roll, pitch, yaw])
                roll, pitch, yaw = R.dot(sumMomentum)

        array_drag.append(drag)
        array_side.append(side)
        array_lift.append(lift)

        if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
            array_roll.append(roll)
            array_pitch.append(pitch)
            array_yaw.append(yaw)
    return(array_drag, array_lift, array_side, array_roll, array_pitch, array_yaw)

# Summation of coefficients depending on the time
def _timeSummation(initial_time, time, time_steps, array_drag, array_lift, array_side, array_roll, array_pitch, array_yaw,
    M_visc_columns, M_pres_columns):
    sum_drag = 0
    sum_side = 0
    sum_lift = 0

    sum_roll  = 0
    sum_pitch = 0
    sum_yaw   = 0

    if time < 0:
        tu = 0
        k = 1
        while tu <= abs(time):
            tu += time_steps[len(time_steps)-k]
            k += 1
        for i in range(len(time_steps)-k, len(time_steps)):
            sum_drag  += (array_drag[i]*time_steps[i]) #el drag en cada instante de la sim y hace integral para luego la media
            sum_side  += (array_side[i]*time_steps[i])
            sum_lift  += (array_lift[i]*time_steps[i])
            if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
                sum_roll  += (array_roll[i]*time_steps[i])
                sum_pitch += (array_pitch[i]*time_steps[i])
                sum_yaw   += (array_yaw[i]*time_steps[i])

        initial_step = len(time_steps)-k
        final_step = len(time_steps)

    elif time == 0:
        for i in range(len(time_steps)):
            sum_drag  += (array_drag[i]*time_steps[i])
            sum_side  += (array_side[i]*time_steps[i])
            sum_lift  += (array_lift[i]*time_steps[i])
            if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
                sum_roll  += (array_roll[i]*time_steps[i])
                sum_pitch += (array_pitch[i]*time_steps[i])
                sum_yaw   += (array_yaw[i]*time_steps[i])
        initial_step = 0
        final_step = len(time_steps)
        tu = accumulated_time

    else:
        tu = 0
        k = 0
        ti = 0 
        j = 0

        while ti < initial_time:
            ti += time_steps[j]
            j += 1

        while tu <= time:
            tu += time_steps[j+k]
            k += 1

        for i in range(j,j+k):
            sum_drag  += (array_drag[i]*time_steps[i])
            sum_side  += (array_side[i]*time_steps[i])
            sum_lift  += (array_lift[i]*time_steps[i])
            if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
                sum_roll  += (array_roll[i]*time_steps[i])
                sum_pitch += (array_pitch[i]*time_steps[i])
                sum_yaw   += (array_yaw[i]*time_steps[i])
        initial_step = j
        final_step = j+k
    return (sum_drag, sum_side, sum_lift, sum_roll, sum_pitch, sum_yaw, tu, initial_step, final_step)

# Average coefficients
def _avNrms(sum_drag, sum_side, sum_lift, sum_roll, sum_pitch, sum_yaw, tu, M_visc_columns, M_pres_columns,
            initial_step, final_step, array_drag, array_side, array_lift, array_roll, array_pitch, array_yaw):
    average_drag = sum_drag/tu
    average_side = sum_side/tu
    average_lift = sum_lift/tu
    if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
        average_roll  = sum_roll/tu
        average_pitch = sum_pitch/tu
        average_yaw   = sum_yaw/tu

    sum_drag_rms = 0
    sum_side_rms = 0
    sum_lift_rms = 0

    if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
        sum_roll_rms  = 0
        sum_pitch_rms = 0
        sum_yaw_rms   = 0

    # RMS coefficients

    for i in range(initial_step, final_step):
        sum_drag_rms  += (array_drag[i]-average_drag)**2
        sum_side_rms  += (array_side[i]-average_side)**2
        sum_lift_rms  += (array_lift[i]-average_lift)**2
        if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
            sum_roll_rms  += (array_roll[i]-average_roll)**2
            sum_pitch_rms += (array_pitch[i]-average_pitch)**2
            sum_yaw_rms   += (array_yaw[i]-average_yaw)**2

    n = final_step - initial_step

    rms_drag = (sum_drag_rms/n)**0.5
    rms_side = (sum_side_rms/n)**0.5
    rms_lift = (sum_lift_rms/n)**0.5
    if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
        rms_roll  = (sum_roll_rms/n)**0.5
        rms_pitch = (sum_pitch_rms/n)**0.5
        rms_yaw   = (sum_yaw_rms/n)**0.5
    if len(M_visc_columns) != 0 or len(M_pres_columns) != 0:
        return(average_drag, average_side, average_lift, average_roll, average_pitch, average_yaw,
            rms_drag, rms_side, rms_lift, rms_roll, rms_pitch, rms_yaw)
    else:
        return(average_drag, average_side, average_lift, False, False, False,
            rms_drag, rms_side, rms_lift, False, False, False)


def compute_avg_lift_drag(ep_num, cpuid = '', nb_inv=-1, global_rew = False):
    
    filename, bound, variabound, porous, density, veloc, scale_area, d, initial_time, time, superTitle, u, phi, sin_phi, cos_phi, D_exp, S_exp, L_exp, R_exp, P_exp, Y_exp = _readInputdata()

    file = '-boundary.nsi.set'

    #rewrite the boundary from the list --> all the invariants in order
    # in parameters --> array of all possible bounds
    # in global reward mode --> the function access to all bound + change the scale area
    # global: nb_inv = total num of invs
    # local:  nb_inv = position of the actual boundary

    if not global_rew:
        bound = [bound[nb_inv-1]]
        print("Computing DRAG & LIFT of invariant surface: ", bound)
    else:
        bound = bound
        scale_area = scale_area*nb_inv
        print("Computing DRAG & LIFT of global cylinder: ", bound)
       
    if ep_num == 0:
        entrada = open('alya_files/baseline/'+ filename + file,'r')
    else:
        entrada = open('alya_files/{0}/1/EP_{1}/'.format(cpuid, ep_num)+ filename + file,'r')
    
    F_visc_columns, F_pres_columns, F_vari_columns, F_poro_columns, M_visc_columns, M_pres_columns, header, totalBound = _readHeader(entrada)
    
    fileLines = _readFile(entrada, header)
    
    uniqueIndex, index = _uniqueLines(fileLines)
    
    time_steps, accumulated_array = _timeArrays(uniqueIndex, index, fileLines, totalBound)
    
    R = _rotationMatrix(u, sin_phi, cos_phi)
    
    typfo = _forcesTypes(bound, variabound, porous)
    
    array_drag, array_lift, array_side, array_roll, array_pitch, array_yaw = _calculateFnM(uniqueIndex, fileLines,
     	bound, variabound, F_visc_columns, F_pres_columns, F_vari_columns, F_poro_columns, M_visc_columns,
     	M_pres_columns, phi, density, index,porous, typfo, scale_area, veloc, d, R, case)
    
    sum_drag, sum_side, sum_lift, sum_roll, sum_pitch, sum_yaw, tu, initial_step, final_step = _timeSummation(initial_time, time,
     	time_steps, array_drag, array_lift, array_side, array_roll, array_pitch, array_yaw,M_visc_columns, M_pres_columns)
    
    average_drag, average_side, average_lift, average_roll, average_pitch, average_yaw, rms_drag, rms_side, rms_lift, rms_roll, rms_pitch, rms_yaw = _avNrms(sum_drag, sum_side, sum_lift, sum_roll, sum_pitch, sum_yaw, tu, M_visc_columns, M_pres_columns,
            initial_step, final_step, array_drag, array_side, array_lift, array_roll, array_pitch, array_yaw)
    entrada.close()
    
    return average_drag,average_lift
