import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import convolve1d
from vispy import app
from vispy import gloo
from functools import partial
from netCDF4 import Dataset

# Shader source code
# --------------------------------------------------------

vertex = """
uniform float scale;
uniform vec2 rel_pos;
attribute vec3 color;
attribute vec2 position;
varying vec3 v_color;
void main()
{
    gl_Position = vec4((position+rel_pos)*scale, 0.0, 1.0);
    v_color = color;
}
"""

fragment = """
uniform float scale;
uniform vec2 rel_pos;
varying vec3 v_color;
void main()
{
    gl_FragColor.rgb = v_color;
}
"""

vertex2 = """
uniform float scale;
uniform vec2 rel_pos;
attribute vec3 color;
attribute vec2 position;
varying vec3 v_color;
void main()
{
    gl_Position = vec4((position+rel_pos)*scale, 0.0, 1.0);
    v_color = color;
}
"""

fragment2 = """
uniform float scale;
uniform vec2 rel_pos;
varying vec3 v_color;
void main()
{
    gl_FragColor.rgb = v_color;
}
"""

vertex3 = """
uniform float scale;
uniform vec2 rel_pos;
attribute vec3 color;
attribute vec2 position;
attribute float invis1;
varying float invis;
varying vec3 v_color;
void main()
{
    gl_Position = vec4((position+rel_pos)*scale, 0.0, 1.0);
    invis = invis1;
    v_color = color;
}
"""

fragment3 = """
uniform float scale;
uniform vec2 rel_pos;
varying vec3 v_color;
varying float invis;
void main()
{
    if( invis == -2.)
        discard;
    else
    gl_FragColor.rgb = v_color;
}
"""

# vispy Canvas
# --------------------------------------------------------

class RadarCanvas(app.Canvas):

    def __init__(self, radfile, cmap, field):
        app.Canvas.__init__(self, app='pyqt4', keys='interactive')

        self.program = gloo.Program(vertex, fragment)
        self.program2 = gloo.Program(vertex2, fragment2)
        self.program3 = gloo.Program(vertex3, fragment3)
        self.rad = radfile

        self.program['rel_pos'] = (0.0, 0.0)
        self.program['scale'] = 1.0
        self.program2['rel_pos'] = (0.0, 0.0)
        self.program2['scale'] = 1.0
        self.program3['rel_pos'] = (0.0, 0.0)
        self.program3['scale'] = 1.0

        self.sweepNum = 0
        self.azi = radfile.get_azimuth(self.sweepNum)
        self.ran = radfile.range['data']
        self.numAzi = self.azi.shape[0]
        self.numRange = self.ran.shape[0]
        self.numTotal = 4*(self.numAzi+2)*self.numRange
        self.timeUnits = self.rad.time['units']
        self.yyyy = self.timeUnits.split(' ')[-1][0:4]
        self.mm = self.timeUnits.split(' ')[-1][5:7]
        self.dd = self.timeUnits.split(' ')[-1][8:10]
        self.hh = self.timeUnits.split(' ')[-1][11:13]
        self.mn = self.timeUnits.split(' ')[-1][14:16]

        # Create vertex data container
        self.data = None
        self.data2 = None
        self.data3 = None

        self.colorTable = cmap
        self.fieldData = None
        self.kdp = None
        self.commonFieldMask = None
        self.fieldGrid = None
        self.colorData = None
        self.currentFields = None
        self.currentField = field

        self.createCommonMask()
        self.extractData(self.currentField)
        self.createGrid()
        self.colorGrid(-10.0, 80.0)

        self.numSLines = 10
        self.nodesPerLine = 100
        self.createLineGrid()
        self.createCounties()
        gloo.gl.glLineWidth(1.0)

        self.show()

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.size) 

    def on_draw(self, event):
        gloo.clear((1,1,1,1))
        self.program.draw('triangle_strip')
        #self.program2.draw('lines')
        self.program3.draw('line_strip')

    def on_mouse_move(self, event):
        win_size_x = self.size[0]
        win_size_y = self.size[1]

        rel_pos_init = self.program['rel_pos']
        sc = self.program['scale']

        if event.is_dragging and event.buttons[0] == 1:
            x0, y0 = event.last_event.pos[0], event.last_event.pos[1]
            x1, y1 = event.pos[0], event.pos[1]    

            rightXCond = (x1 > x0) and rel_pos_init[0] < -1
            leftXCond = (x1 < x0) and rel_pos_init[0] > 1
            upYCond = (y1 > y0) and rel_pos_init[1] > 1
            downYCond = (y1 < y0) and rel_pos_init[1] < -1

            if all(abs(rel_pos_init) < 1) or upYCond or downYCond or leftXCond or rightXCond:

                norml_x0 = (2.0*x0/float(win_size_x))-1
                norml_y0 = (2.0*y0/float(win_size_y))-1
                norml_x1 = (2.0*x1/float(win_size_x))-1
                norml_y1 = (2.0*y1/float(win_size_y))-1

                self.program['rel_pos'] = (rel_pos_init[0]+(norml_x1-norml_x0)/sc, 
                                           rel_pos_init[1]-(norml_y1-norml_y0)/sc)
                self.program2['rel_pos'] = self.program['rel_pos']
                self.program3['rel_pos'] = self.program['rel_pos']
                self.update()

    def on_mouse_wheel(self, event):

        win_size_x = self.size[0]
        win_size_y = self.size[1]

        cur_pos_x = event.pos[0]
        cur_pos_y = event.pos[1]

        norm_x_pos = (2.0*cur_pos_x/win_size_x)-1.0
        norm_y_pos = (2.0*cur_pos_y/win_size_y)-1.0

        sign_scroll = event.delta[1]/abs(event.delta[1])
        zoomInCond = self.program['scale'] < 12. and sign_scroll > 0
        zoomOutCond = self.program['scale'] > 0.9 and sign_scroll < 0

        if zoomInCond or zoomOutCond:
            self.program['scale'] = self.program['scale']*1.1**sign_scroll
            sc = self.program['scale'][0]

            self.program['rel_pos'] = self.program['rel_pos']+ \
                                      (-norm_x_pos*0.1*sign_scroll/sc, 
                                      norm_y_pos*0.1*sign_scroll/sc)

            self.program2['scale'] = self.program['scale']
            self.program2['rel_pos'] = self.program['rel_pos']
            self.program3['scale'] = self.program['scale']
            self.program3['rel_pos'] = self.program['rel_pos']

            self.update()

    def createCommonMask(self):
        radar_sweep = self.rad.extract_sweeps([self.sweepNum])
        zh = radar_sweep.fields['reflectivity']['data']
        self.commonFieldMask = np.ma.getmaskarray(zh)
        dims = self.commonFieldMask.shape

    def extractData(self, fieldName):

        radar_sweep = self.rad.extract_sweeps([self.sweepNum])
        if fieldName == 'KDP':
            phiDP = radar_sweep.fields['differential_phase']['data']
            self.fieldData = calculateKDP(phiDP, radar_sweep.range['data'])
        else:
            self.fieldData = radar_sweep.fields[fieldName]['data']

        #self.fieldData[self.commonFieldMask] = -999.
        #print self.fieldData
        self.currentFields = radar_sweep.fields.keys()
        self.currentField = fieldName

    def createGrid(self):

        #Get information about the dimension of the radar data

        self.azi = self.rad.get_azimuth(self.sweepNum)
        self.numAzi = self.azi.shape[0]
        self.numTotal = 4*(self.numAzi+2)*self.numRange
        del(self.data)
        self.data = np.zeros(self.numTotal, [('position', np.float32, 2),
               ('color', np.float32, 3)])

        azimuth = self.azi
        range_rad = self.ran
        azi_start = azimuth[0]

        min_range = range_rad[0].astype(float)/ \
                    range_rad[self.numRange-1].astype(float)

        gspace = (1.0-min_range)/(self.numRange-1)
        azispace = 2.0/(self.numAzi-1)*np.pi

        #Created normalized arrays of range and azimuth

        ranall = np.linspace(min_range, 1.000, self.numRange)
        theta = np.pi*(90.0-(np.linspace(0.0, 360.0, 
                self.numAzi)+azi_start))/180.0

        #Expand azimuth arrays to account for extra points

        exp_thet = np.empty([self.numAzi+2])
        exp_thet[:self.numAzi] = theta
        exp_thet[self.numAzi:self.numAzi+2] = theta[self.numAzi-1]

        #Create big arrays to merge all triangle vertex locations

        posx = np.empty([self.numTotal])
        posy = np.empty([self.numTotal])

        #Create array indicies that correspond to the triangle locations

        ind1 = np.arange(0, self.numAzi+1)*4
        ind2 = ind1+1
        ind3 = ind1+2
        ind4 = ind1+3

        #Make repeating 1d arrays of range and azimuth

        #Create 1d arrays of dimension num_ran*num_azi for
        # each of the four triangle points comprising a gate

        for i in range(0, self.numRange-1):

            ind1 = ind1+(self.numAzi+2)*4
            ind2 = ind1+1
            ind3 = ind1+2
            ind4 = ind1+3

            posx.flat[ind1] = (ranall[i]-gspace/2)*np.cos(exp_thet-azispace/2)
            posx.flat[ind2] = (ranall[i]+gspace/2)*np.cos(exp_thet-azispace/2)
            posx.flat[ind3] = (ranall[i]-gspace/2)*np.cos(exp_thet+azispace/2)
            posx.flat[ind4] = (ranall[i]+gspace/2)*np.cos(exp_thet+azispace/2)

            posy.flat[ind1] = (ranall[i]-gspace/2)*np.sin(exp_thet-azispace/2)
            posy.flat[ind2] = (ranall[i]+gspace/2)*np.sin(exp_thet-azispace/2)
            posy.flat[ind3] = (ranall[i]-gspace/2)*np.sin(exp_thet+azispace/2)
            posy.flat[ind4] = (ranall[i]+gspace/2)*np.sin(exp_thet+azispace/2)

            end_inds = (self.numAzi+2)*4

            posx[(i+1)*end_inds-6] = (ranall[i]-gspace/2)* \
                                     np.cos(theta[self.numAzi-1]+azispace/2)
            posx[(i+1)*end_inds-5] = (ranall[i]-gspace/2)* \
                                     np.cos(theta[self.numAzi-1]+azispace/2)
            posx[(i+1)*end_inds-4] = (ranall[i]-gspace/2)* \
                                     np.cos(theta[self.numAzi-1]+azispace/2)
            posx[(i+1)*end_inds-3] = (ranall[i]-gspace/2)* \
                                     np.cos(theta[self.numAzi-1]+azispace/2)
            posx[(i+1)*end_inds-2] = (ranall[i]-gspace/2)* \
                                     np.cos(theta[self.numAzi-1]+azispace/2)
            posx[(i+1)*end_inds-1] = (ranall[i]-gspace/2)* \
                                     np.cos(theta[self.numAzi-1]+azispace/2)

            posy[(i+1)*end_inds-6] = (ranall[i]-gspace/2)* \
                                     np.sin(theta[self.numAzi-1]+azispace/2)
            posy[(i+1)*end_inds-5] = (ranall[i]-gspace/2)* \
                                     np.sin(theta[self.numAzi-1]+azispace/2)
            posy[(i+1)*end_inds-4] = (ranall[i]-gspace/2)* \
                                     np.sin(theta[self.numAzi-1]+azispace/2)
            posy[(i+1)*end_inds-3] = (ranall[i]-gspace/2)* \
                                     np.sin(theta[self.numAzi-1]+azispace/2)
            posy[(i+1)*end_inds-2] = (ranall[i]-gspace/2)* \
                                     np.sin(theta[self.numAzi-1]+azispace/2)
            posy[(i+1)*end_inds-1] = (ranall[i]-gspace/2)* \
                                     np.sin(theta[self.numAzi-1]+azispace/2)

        #Merge x and y vertex arrays together as a tuple

        self.fieldGrid = zip(posx, posy)

        self.data['position'] = self.fieldGrid
        self.program.bind(gloo.VertexBuffer(self.data))

    def colorGrid(self, valmin, valmax):

        raw_dat = self.fieldData[:,:self.numRange]
        mindat = raw_dat.min()
        maxdat = raw_dat.max()
        norm_dat = (raw_dat-valmin)/(valmax-valmin)
        #norm_dat = (raw_dat-mindat)/(maxdat-mindat)

        rc = np.empty([self.numTotal])
        gc = np.empty([self.numTotal])
        bc = np.empty([self.numTotal])

        #Get RGB values for each data value

        ind1 = np.arange(0, self.numAzi+1)*4
        ind2 = ind1+1
        ind3 = ind1+2
        ind4 = ind1+3

        #Create 1d arrays of dimension num_ran*num_azi for
        # each of the four triangle points comprising a gate

        for i in range(0, self.numRange-1):

            ind1 = ind1+(self.numAzi+2)*4
            ind2 = ind1+1
            ind3 = ind1+2
            ind4 = ind1+3

            #missing_data_ind = np.where(raw_dat[:,i]=='--')
            missing_data_ind = self.commonFieldMask[:,i]
            #print missing_data_ind
            maxBoundData = np.where(norm_dat[:,i]>1.)
            minBoundData = np.where(norm_dat[:,i]<0.)
            norm_dat[maxBoundData,i] = 1.
            norm_dat[minBoundData,i] = 0.
            #print norm_dat.max(), norm_dat.min()
            index = (norm_dat[:,i]*255).astype(int)
            #print self.colorTable[index,0]

            rc.flat[ind1] = self.colorTable[index,0]
            subInd1 = ind1[0:self.numAzi]
            rc.flat[subInd1[missing_data_ind]] = -1.
            rc.flat[ind1[self.numAzi]] = rc.flat[ind1[self.numAzi-1]]
            rc[ind2] = rc[ind1]
            rc[ind3] = rc[ind1]
            rc[ind4] = rc[ind1]

            gc.flat[ind1] = self.colorTable[index,1]
            gc[ind2] = gc[ind1]
            gc[ind3] = gc[ind1]
            gc[ind4] = gc[ind1]

            bc.flat[ind1] = self.colorTable[index,2]
            bc[ind2] = bc[ind1]
            bc[ind3] = bc[ind1]
            bc[ind4] = bc[ind1]
        
        missing_col_ind1 = np.where(rc==-1.)

        #missing_col_ind2 = np.where(rc==self.colorTable[0,0]) and \
        #                   np.where(gc==self.colorTable[0,1]) and \
        #                   np.where(bc==self.colorTable[0,2])
        rc[missing_col_ind1] = 1.0
        gc[missing_col_ind1] = 1.0
        bc[missing_col_ind1] = 1.0
        #rc[missing_col_ind2] = 1.0
        #gc[missing_col_ind2] = 1.0
        #bc[missing_col_ind2] = 1.0
        
        '''
        missing_data_indfull = np.where(raw_dat==-999.)
        rc[missing_data_indfull] = 1.
        gc[missing_data_indfull] = 1.
        bc[missing_data_indfull] = 1.
        '''

        self.colorData = zip(rc, gc, bc)
        self.data['color'] = self.colorData
        self.program.bind(gloo.VertexBuffer(self.data))

    def createLineGrid(self):

        numPoints = 4*self.numSLines

        #nodes to make lines dashed

        xPos = np.linspace(-1., 1., self.numSLines)
        yPos = np.linspace(-1., 1., self.numSLines)

        nodeArr = np.linspace(-1., 1., self.nodesPerLine)

        xLinePos = np.zeros((numPoints, self.nodesPerLine))
        yLinePos = np.zeros((numPoints, self.nodesPerLine))

        for i in range(2*(self.numSLines-2)):
            xLinePos[i,:] = xPos[(self.numSLines-1)*(i%2)]*nodeArr
            yLinePos[i,:] = yPos[int((i)/2)+1]
            yLinePos[i+2*self.numSLines,:] = yPos[(self.numSLines-1)*(i%2)]*nodeArr
            xLinePos[i+2*self.numSLines,:] = xPos[int((i)/2)+1]

        rColArr = np.zeros((numPoints, self.nodesPerLine)).flatten()
        gColArr = rColArr
        bColArr = rColArr

        del(self.data2)
        self.data2 = np.zeros(numPoints*self.nodesPerLine, [('position', np.float32, 2),
               ('color', np.float32, 3)])

        self.data2['position'] = zip(xLinePos.flatten(), yLinePos.flatten())
        self.data2['color'] = zip(rColArr, gColArr, bColArr)

        self.program2.bind(gloo.VertexBuffer(self.data2))
        self.program2['rel_pos'] = self.program['rel_pos']
        self.program2['scale'] = self.program['scale']

    def createCounties(self):

        # set map domain (to start)

        latmin = 31.
        latmax = 50.
        lonmin = -120.
        lonmax = -60.

        # read netcdf file with state coordinates

        us_counties = Dataset('../us_counties.nc')

        lat_points = us_counties.variables['latitude'][:]
        lon_points = us_counties.variables['longitude'][:]
        gl_mask = us_counties.variables['gl_mask'][:]

        # convert to screen coordinates
        radlon = self.rad.longitude['data'][0]
        radlat = self.rad.latitude['data'][0]
        maxRange = np.max(self.ran)
        win_size_x = self.size[0]
        win_size_y = self.size[1]

        print radlon, radlat

        scrX, scrY = lonLat2Screen(lon_points, lat_points, radlon, radlat,
                                   maxRange, win_size_x, win_size_y)

        # zip lat and lon arrays together

        county_points = zip(scrX, scrY)

        # create colors
        rColArr = np.zeros([len(lat_points),1])
        gColArr = rColArr
        bColArr = rColArr

        del(self.data3)
        self.data3 = np.zeros(len(lat_points), [('position', np.float32, 2),
               ('color', np.float32, 3), ('invis1', np.float32, 1)])

        self.data3['position'] = county_points
        self.data3['color'] = zip(rColArr, gColArr, bColArr)
        self.data3['invis1'] = gl_mask

        self.program3.bind(gloo.VertexBuffer(self.data3))
        self.program3['rel_pos'] = self.program['rel_pos']
        self.program3['scale'] = self.program['scale']

# Miscellaneous radar functions
#--------------------------------------------------------
def beamHeight(ran, elev):
    radz = 10.
    erad = np.pi*elev/180.

    ke = 4./3.
    a = 6378137.

    z = np.sqrt(ran**2.+(ke*a)**2.+2.*ran*ke*a*np.sin(erad))-ke*a+radz
    return z

def beamDistance(ran, elev):
    radz = 10.
    erad = np.pi*elev/180.

    ke = 4./3.
    a = 6378137.

    z = np.sqrt(ran**2.+(ke*a)**2.+2.*ran*ke*a*np.sin(erad))-ke*a+radz
    beamDis = float(ke*a*np.arcsin(ran*np.cos(erad)/(ke*a+z)))
    return beamDis

def lonLat2Screen(lon, lat, radlon, radlat, maxRange, win_size_x, win_size_y):

    re = 6.371e6

    ylcs = np.pi*re*(lat-radlat)/(180.)
    xlcs = np.pi*re*np.cos(np.pi*lat/180.)*(lon-radlon)/(180.)

    normX = xlcs/(maxRange)
    normY = ylcs/(maxRange)

    return normX, normY

def calculateKDP(phiDP, ran):
    # smooth phiDP field and take derivative
    # calculate lanczos filter weights
    numRan = ran.shape[0]
    numK = 51
    fc = 0.04
    kt = np.linspace(-(numK-1)/2, (numK-1)/2, numK)
    w = np.sinc(2.*kt*fc)*(2.*fc)*np.sinc(kt/(numK/2))

    kdp = np.ma.masked_all(phiDP.shape)
    smoothPhiDP = convolve1d(phiDP, w, axis=1, mode='constant', cval=-999.)
    smoothPhiDP = np.ma.masked_where(smoothPhiDP==-999., smoothPhiDP)
    # take derivative of kdp field
    #flatPhiDP = smoothPhiDP.reshape(numRan, -1)

    winLen = 10
    rprof = ran[0:winLen*2-1]/1000.

    for i in range(numRan-winLen*3):
        kdp[:,i+winLen] = 0.5*np.polyfit(rprof, smoothPhiDP[:,
                           winLen+i:i+winLen*3-1].transpose(), 1)[0]
    return smoothPhiDP
