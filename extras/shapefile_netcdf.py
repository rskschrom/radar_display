import numpy as np
from vispy import app
from vispy import gloo
from netCDF4 import Dataset

# set map domain (to start)

latmin = 31.
latmax = 50.
lonmin = -120.
lonmax = -60.

# read netcdf file with state coordinates

us_states = Dataset('../us_states.nc')

lat_points = us_states.variables['latitude'][:]
lon_points = us_states.variables['longitude'][:]
gl_mask = us_states.variables['gl_mask'][:]

# zip lat and lon arrays together

state_points = zip((lon_points-lonmin)/(lonmax-lonmin),(lat_points-latmin)/(latmax-latmin))

print "rendering"

c = app.Canvas(keys='interactive')

vertex = """
uniform float scale;
uniform float hold;
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

fragment = """
uniform float scale;
uniform float hold;
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

program = gloo.Program(vertex, fragment, count=len(state_points))
program['invis1'] = gl_mask
program['position'] = state_points
program['scale']    = 1.0
program['rel_pos']    = (0.0, 0.0)
program['hold'] = 0.0

@c.connect
def on_resize(event):
    gloo.set_viewport(0, 0, *event.size)

@c.connect
def on_draw(event):
    gloo.clear((1,1,1,1))
    program.draw('line_strip')

#Panning!

init_x = 0.0
init_y = 0.0

rel_pos_init = program['rel_pos']

@c.events.mouse_press.connect
def mouse_event(event):
    #print event.button

    global init_x
    global init_y

    global rel_pos_init

    lst_pos_x = event.pos[0]
    lst_pos_y = event.pos[1]

    win_size_x = c.size[0]
    win_size_y = c.size[1]

    init_x = (2.0*lst_pos_x/float(win_size_x))-1
    init_y = (2.0*lst_pos_y/float(win_size_y))-1

    program['hold'] = 1.0
    rel_pos_init = program['rel_pos']

@c.events.mouse_move.connect
def mouse_event(event):

    lst_pos_x = event.pos[0]
    lst_pos_y = event.pos[1]

    win_size_x = c.size[0]
    win_size_y = c.size[1]

    norml_x_pos = (2.0*lst_pos_x/float(win_size_x))-1
    norml_y_pos = (2.0*lst_pos_y/float(win_size_y))-1

    sc = program['scale'][0]
    hl = program['hold'][0]

    program['rel_pos'] = (rel_pos_init[0]+hl*(norml_x_pos-init_x)/sc, rel_pos_init[1]-hl*(norml_y_pos-init_y)/sc)

    c.update()

@c.events.mouse_release.connect
def mouse_event(event):

    global rel_pos_init

    program['hold'] = 0.0
    rel_pos_init = program['rel_pos']

#Zoom!

@c.events.mouse_wheel.connect
def mouse_event(event):
    #print "clicked at:", event.pos

    global rel_pos_init

    win_size_x = c.size[0]
    win_size_y = c.size[1]

    cur_pos_x = event.pos[0]
    cur_pos_y = event.pos[1]

    norm_x_pos = (2.0*cur_pos_x/win_size_x)-1.0
    norm_y_pos = (2.0*cur_pos_y/win_size_y)-1.0

    sign_scroll = event.delta[1]/abs(event.delta[1])

    program['scale'] = program['scale']*1.1**sign_scroll
    sc = program['scale'][0]
    #print sc
    program['rel_pos'] = program['rel_pos']+(-norm_x_pos*0.1*sign_scroll/sc, norm_y_pos*0.1*sign_scroll/sc)
    #print program['rel_pos']

    rel_pos_init = program['rel_pos']

    c.update()

c.show()
app.run();

