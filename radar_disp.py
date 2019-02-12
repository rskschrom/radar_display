import numpy as np
from vispy import app
from PyQt4 import QtGui, QtCore
from netCDF4 import Dataset
from numpy import genfromtxt
import pyart
import sys
from functools import partial
import time
from radarcanvas import RadarCanvas, beamHeight, beamDistance

# Open radar color tables
# --------------------------------------------------------

fil_zh = open("/home/robert/python/zh2_map.rgb")
cdata = genfromtxt(fil_zh,skip_header=2)
zh_map = cdata/255

fil_zdr = open("/home/robert/python/zdr_map.rgb")
cdata = genfromtxt(fil_zdr,skip_header=2)
zdr_map = cdata/255

fil_phv = open("/home/robert/python/phv_map.rgb")
cdata = genfromtxt(fil_phv,skip_header=2)
phv_map = cdata/255

fil_vel = open("/home/robert/python/vel2_map.rgb")
cdata = genfromtxt(fil_vel,skip_header=2)
vel_map = cdata/255

fil_kdp = open("/home/robert/python/kdp_map.rgb")
cdata = genfromtxt(fil_kdp,skip_header=2)
kdp_map = cdata/255

colorMaps = {'reflectivity': {'ctable': zh_map, 'minval': -10.,
            'maxval': 80., 'units': 'dBZ'},
            'differential_reflectivity': {'ctable': zdr_map, 'minval': -2.,
            'maxval': 8., 'units': 'dB'},
            'velocity': {'ctable': vel_map, 'minval': -50.,
            'maxval': 50., 'units': 'm/s'},
            'differential_phase': {'ctable': zh_map, 'minval': 0., 
            'maxval': 180., 'units': 'deg.'},
            'cross_correlation_ratio': {'ctable': phv_map, 
            'minval': 0.20, 'maxval': 1.10, 'units': ''}, 
            'spectrum_width': {'ctable': zdr_map, 'minval': 0.0,
            'maxval': 20., 'units': 'm/s'},
            'KDP': {'ctable': kdp_map, 'minval': -4.,
            'maxval': 8., 'units': 'deg./km'}}

# Open radar file
# --------------------------------------------------------

print "opening file"

#filename = "/media/robert/research_stuff/nwsRadarFiles/KBOX20150127_134232_V06"
filename = "/home/robert/radar/KTLX20130531_232345_V06"
#radar = pyart.io.read_nexrad_archive(filename)

#print radar.info()

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

# PyQt window
# --------------------------------------------------------
class ColorWindow(QtGui.QMainWindow):

    def __init__(self, parent=None,):
        QtGui.QMainWindow.__init__(self, parent)

        self.firstTime = True
        self.containWidget = QtGui.QWidget()
        self.grid = QtGui.QGridLayout()
        self.grid.setSpacing(0)

        self.mainMenu = self.menuBar()
        self.createFileMenu()

        '''self.radCan = rad_can
        self.grid.addWidget(self.radCan.native, 1, 1)

        self.colBar = QtGui.QWidget()

        if self.radCan.rad != None:
            self.colBar = ColorBar(self.radCan.currentField)
            self.grid.addWidget(self.colBar, 1, 0)
            self.updateTitle()

            self.createSweepMenu()
            self.createFieldsMenu()
            self.createOptionsMenu()
        '''
        self.containWidget.setLayout(self.grid) 
        self.setCentralWidget(self.containWidget)

        self.setFixedSize(920, 800)
        
        self.show()

    def createFileMenu(self):

        fileMenu = self.mainMenu.addMenu('File')

        openButton = QtGui.QAction('Open', self)
        openButton.triggered.connect(self.openFile)
        screenButton = QtGui.QAction('Screenshot', self)
        screenButton.triggered.connect(self.screenShot)
        quitButton = QtGui.QAction('Quit', self)
        quitButton.triggered.connect(self.close)

        fileMenu.addAction(openButton)
        fileMenu.addAction(screenButton)
        fileMenu.addAction(quitButton)

    def createOptionsMenu(self):

        optionsMenu = self.mainMenu.addMenu('Options')

        colorRange = QtGui.QAction('Color table range', self)
        colorRange.triggered.connect(self.changeColorRange)
        gridLines = QtGui.QAction('Change grid lines', self)
        gridLines.triggered.connect(self.changeGridLines)
        qvpPlot = QtGui.QAction('Plot QVP', self)
        qvpPlot.triggered.connect(self.plotQVP)

        optionsMenu.addAction(colorRange)
        optionsMenu.addAction(gridLines)
        optionsMenu.addAction(qvpPlot)

    def createFieldsMenu(self):

        #Fields menu
        fieldsMenu = self.mainMenu.addMenu('Fields')

        flds = self.radCan.currentFields

        for key in flds:
            menuItem = QtGui.QAction(key, self)
            menuItem.triggered.connect(partial(self.fieldSelect, key))
            fieldsMenu.addAction(menuItem)

        # add option to calculate KDP
        menuItem = QtGui.QAction('Calculate KDP', self)
        menuItem.triggered.connect(partial(self.fieldSelect, 'KDP'))
        fieldsMenu.addAction(menuItem)

    def createSweepMenu(self):

        #Sweep menu
        sweepMenu = self.mainMenu.addMenu('Sweeps')

        elevs = self.radCan.rad.fixed_angle['data']

        for i in range(len(elevs)):

            menuItem = QtGui.QAction('&'+str(elevs[i]), self)
            menuItem.triggered.connect(partial(self.sweepSelect, i))
            sweepMenu.addAction(menuItem)

    def screenShot(self):

        time.sleep(0.2)
        p = QtGui.QPixmap.grabWindow(self.containWidget.winId())
        elev = self.radCan.rad.fixed_angle['data'][self.radCan.sweepNum]
        yyyy = self.radCan.yyyy
        mm = self.radCan.mm
        dd = self.radCan.dd
        hh = self.radCan.hh
        mn = self.radCan.mn

        roundEl = str(round(elev, 1))
        screenFile = self.radCan.currentField+'_'+roundEl+'_'+\
                     self.radSite+'_'+yyyy+mm+dd+'_'+hh+mn+'.png'
        p.save(screenFile, 'png')

    def changeColorRange(self):

        fieldName = self.radCan.currentField

        curmin = colorMaps[fieldName]['minval']
        curmax = colorMaps[fieldName]['maxval']

        self.ti = QtGui.QWidget()
        self.ti.setWindowTitle("Color table range")

        self.ti.e1 = QtGui.QLineEdit(str(curmin))
        self.ti.e2 = QtGui.QLineEdit(str(curmax))
        self.ti.fieldName = fieldName

        self.ti.pb = QtGui.QPushButton("Enter")
        self.ti.pb.clicked.connect(self.rangeButtonClicked)

        layout = QtGui.QFormLayout()
        layout.addRow("Minimum value", self.ti.e1)
        layout.addRow("Maximum value", self.ti.e2)
        layout.addRow(self.ti.pb)
        
        self.ti.setLayout(layout)
        self.ti.show()  

    def changeGridLines(self):

        self.tiGrid = QtGui.QWidget()
        self.tiGrid.setWindowTitle("Grid line spacing")

        # calculate grid line spacing

        curElev = self.radCan.rad.fixed_angle['data'][self.radCan.sweepNum]
        maxRan = self.radCan.ran.max()
        maxDis = beamDistance(maxRan, curElev)/1000.
        gridSpacing = 2.*maxDis/float(self.radCan.numSLines-1)
        nodeSpacing = 2.*maxDis/float(self.radCan.nodesPerLine-1)

        self.tiGrid.e1 = QtGui.QLineEdit("{0:.1f}".format(gridSpacing))
        self.tiGrid.e2 = QtGui.QLineEdit("{0:.1f}".format(nodeSpacing))

        self.tiGrid.pb = QtGui.QPushButton("Enter")
        self.tiGrid.pb.clicked.connect(self.gridLinesButtonClicked)

        layout = QtGui.QFormLayout()
        layout.addRow("Grid spacing (km)", self.tiGrid.e1)
        layout.addRow("Dash spacing (km)", self.tiGrid.e2)
        layout.addRow(self.tiGrid.pb)
        
        self.tiGrid.setLayout(layout)
        self.tiGrid.show()  

    def plotQVP(self):

        import matplotlib.pyplot as plt

        curElev = self.radCan.rad.fixed_angle['data'][self.radCan.sweepNum]
        curField = self.radCan.currentField
        units = colorMaps[curField]['units']

        sweepData = self.radCan.fieldData
        qvp = np.mean(sweepData, axis=0, dtype=np.float64)
        ran = self.radCan.ran

        z = beamHeight(ran, curElev)/1000.

        fig = plt.gcf()
        fig.canvas.set_window_title(str(curElev)+' deg. QVP')

        ax = fig.add_subplot(111)
        ax.set_title(curField+' profile')
        ax.set_ylabel('height (km)')
        ax.set_xlabel(curField+' ('+units+')')
        ax.plot(qvp, z)
        plt.show()

    def rangeButtonClicked(self):

        fieldName = self.radCan.currentField

        mival = float(self.ti.e1.text())
        maval = float(self.ti.e2.text())

        colorMaps[fieldName]['minval'] = mival
        colorMaps[fieldName]['maxval'] = maval

        self.ti.close()
        self.ti.deleteLater()

        self.colBar.deleteLater()
        self.colBar = ColorBar(fieldName)
        self.grid.addWidget(self.colBar, 1, 0)
        self.radCan.colorGrid(mival, maval)

    def gridLinesButtonClicked(self):

        newGridSpacing = float(self.tiGrid.e1.text())
        newNodeSpacing = float(self.tiGrid.e2.text())

        curElev = self.radCan.rad.fixed_angle['data'][self.radCan.sweepNum]
        maxRan = self.radCan.ran.max()
        maxDis = beamDistance(maxRan, curElev)/1000.

        newNumLines = int(2.*maxDis/newGridSpacing+1)
        newNumNodes = int(2.*maxDis/newNodeSpacing+1)

        self.tiGrid.close()
        self.tiGrid.deleteLater()
        self.radCan.numSLines = newNumLines
        self.radCan.nodesPerLine = newNumNodes
        self.radCan.createLineGrid()

    def fieldSelect(self, fieldName):

        mival = colorMaps[fieldName]['minval']
        maval = colorMaps[fieldName]['maxval']

        self.colBar.deleteLater()
        self.colBar = ColorBar(fieldName)
        self.grid.addWidget(self.colBar, 1, 0)
        self.radCan.colorTable = colorMaps[fieldName]['ctable']
        self.radCan.extractData(fieldName)
        self.radCan.colorGrid(mival, maval)
        self.updateTitle()

    def sweepSelect(self, sweepNum):

        fieldName = self.radCan.currentField

        mival = colorMaps[fieldName]['minval']
        maval = colorMaps[fieldName]['maxval']

        self.radCan.sweepNum = sweepNum
        self.radCan.colorTable = colorMaps[fieldName]['ctable']
        self.radCan.createGrid()
        self.radCan.createCommonMask()
        self.radCan.extractData(fieldName)
        self.radCan.colorGrid(mival, maval)
        self.updateTitle()

    def updateTitle(self):

        curElev = self.radCan.rad.fixed_angle['data'][self.radCan.sweepNum]
        curField = self.radCan.currentField
        yyyy = self.radCan.yyyy
        mm = self.radCan.mm
        dd = self.radCan.dd
        hh = self.radCan.hh
        mn = self.radCan.mn

        dateForm = mm+'/'+dd+'/'+yyyy
        timeForm = hh+':'+mn+' UTC'

        self.setWindowTitle(self.radSite+' WSR-88D - '+
             str(round(curElev, 1))+' deg. PPI - '+dateForm+' @ '+
             timeForm+' - '+curField)

    def openFile(self, event):
        
        filename = QtGui.QFileDialog.getOpenFileName(None,
                    'Open File', '/home/robert/radx')
        print "opening file ", filename

        radar = pyart.io.read_nexrad_archive(filename)

        if not (self.firstTime):
            self.radCan.native.deleteLater()
            self.colBar.deleteLater()
        else:
            self.firstTime = False

        self.radSite = filename.split('/')[-1][0:4]
        self.radCan = RadarCanvas(radfile=radar, cmap=zh_map, field='reflectivity')
        self.colBar = ColorBar(self.radCan.currentField)
        self.grid.addWidget(self.colBar, 1, 0)
        self.grid.addWidget(self.radCan.native, 1, 1)
        self.mainMenu.clear()
        self.createFileMenu()
        self.createSweepMenu()
        self.createFieldsMenu()
        self.createOptionsMenu()
        self.updateTitle()

# Class for colorbar
#--------------------------------------------------------
class ColorBar(QtGui.QWidget):

    def __init__(self, fieldName):
        super(ColorBar, self).__init__()

        self.fieldName = fieldName
        self.initUI()

    def initUI(self):

        #self.setGeometry(300, 300, 50, 800)
        self.setWindowTitle('Colors')
        self.show()

    def paintEvent(self, event):

        qp = QtGui.QPainter()
        qp.begin(self)
        self.drawColorTable(qp)
        qp.end()

    def drawColorTable(self, qp):

        posx = 5
        posy = 800-200
        dy = -2
        width = 40
        height = dy

        numLabels = 15
        dxTxt = 256*dy/numLabels

        minVal = colorMaps[self.fieldName]['minval']
        maxVal = colorMaps[self.fieldName]['maxval']
        units = colorMaps[self.fieldName]['units']

        qp.setFont(QtGui.QFont('Helvetica', 12))
        qp.drawText(25, 70, units)

        prec = (maxVal-minVal)/10.
        if prec < 0.4:
            preStr = "{:.2f}"
        else:
            preStr = "{:.1f}"

        for i in range(numLabels):
            intVal = i*(maxVal-minVal)/numLabels+minVal
            strVal = preStr.format(intVal)
            rect1 = QtCore.QRect(55, i*dxTxt+posy-14, 40, 30)
            qp.drawText(rect1, QtCore.Qt.AlignRight, strVal)

        color = QtGui.QColor(1, 1, 1, 0)
        qp.setPen(color)

        color_data = colorMaps[self.fieldName]['ctable']
        num_colors = len(color_data)
        for color in color_data:
            rgb = (color*255).astype(int)
            qp.setBrush(QtGui.QColor(rgb[0], rgb[1], rgb[2]))
            qp.drawRect(posx, posy, width, height)

            posy = posy+dy

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    w = ColorWindow()
    sys.exit(app.exec_())
    #app.run()
