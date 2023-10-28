def lerp(a,b,w):
    return a*(1-w) + b*w

def lerp3(a,b,w):
    return [
        a[0]*(1-w) + b[0]*w,
        a[1]*(1-w) + b[1]*w,
        a[2]*(1-w) + b[2]*w
    ]

def rgb(r,g,b):
    return [r/255.0, g/255.0, b/255.0]

def sample01(sample_points, w):
    if w<=0.0001:
        return sample_points[0][1]
    if w>=0.9999:
        return sample_points[-1][1]
    a,b=0,0
    t = 0
    for i in range(len(sample_points)):
        h, c = sample_points[i]
        if w < h:
            a,b = i-1,i
            if a<0: a=0
            t = (w - sample_points[a][0])/( sample_points[b][0] - sample_points[a][0])
            break
    return lerp3( sample_points[a][1], sample_points[b][1], t )

################################################################
### colormappings ##
def get_valid_color_mappings():
    return ['metalheat', 'grayscale', 'grayscale2', 'rainbow', 'highcontrast','green',
        'red', 'blue', 'plasma', 'ratio', 'vik']

def colormap_vik(inten_min, inten_max, value):
    inten_range = inten_max - inten_min
    perc = (value - inten_min) / inten_range
    sample_points=[
        [0.00, rgb(0,16,95)],
        [0.10, rgb(1,60,123)],
        [0.20, rgb(29,110,156)],
        [0.30, rgb(111,167,194)],
        [0.40, rgb(200,220,229)],
        [0.50, rgb(255,255,255)],
        [0.60, rgb(233,204,188)],
        [0.70, rgb(210,150,115)],
        [0.80, rgb(188,100,50)],
        [0.90, rgb(138,38,4)],
        [1.00, rgb(88,0,6)]
    ]
    return sample01( sample_points, perc )   

def colormap_ratio(inten_min, inten_max, value):
    inten_range = inten_max - inten_min
    perc = (value - inten_min) / inten_range
    sample_points=[
        [0.00, rgb(0,0,255)],
        [0.50, rgb(255,255,255)],
        [1.00, rgb(255,0,0)]
    ]
    return sample01( sample_points, perc )   

def colormap_plasma(inten_min, inten_max, value):
    inten_range = inten_max - inten_min
    perc = (value - inten_min) / inten_range
    sample_points=[
        [0.00, rgb(13,8,135)],
        [0.14, rgb(84,2,163)],
        [0.29, rgb(139,10,165)],
        [0.43, rgb(185,50,137)],
        [0.57, rgb(219,92,104)],
        [0.71, rgb(244,136,73)],
        [0.86, rgb(254,188,43)],
        [1.00, rgb(240,249,33)],
    ]
    return sample01( sample_points, perc )   

def colormap_red(inten_min, inten_max, value):
    inten_range = inten_max - inten_min
    perc = (value - inten_min) / inten_range
    sample_points=[
        [0.00, [1.00,1.00,1.00]],
        [1.00, [0.86,0.31,0.31]]
    ]
    return sample01( sample_points, perc )    

def colormap_blue(inten_min, inten_max, value):
    inten_range = inten_max - inten_min
    perc = (value - inten_min) / inten_range
    sample_points=[
        [0.00, [1.00,1.00,1.00]],
        [1.00, [0.16,0.31,0.67]]
    ]
    return sample01( sample_points, perc )    

def colormap_highcontrast(inten_min, inten_max, value):
    inten_range = inten_max - inten_min
    perc = (value - inten_min) / inten_range
    sample_points=[
        [0.00, [0,0,0]],
        [0.99, [0,1,1]],
        [1.00, [1,0,0]]
    ]
    return sample01( sample_points, perc )

def colormap_metalheat(inten_min, inten_max, value):
    inten_range = inten_max - inten_min
    perc = (value - inten_min) / inten_range
    sample_points=[
        [0.00, [0,0,0]],
        [0.17, [0,0,1]],
        [0.44, [1,0,0]],
        [0.74, [1,1,0]],
        [1.00, [1,1,1]]
    ]
    return sample01( sample_points, perc )

def colormap_grayscale(inten_min, inten_max, value):
    inten_range = inten_max - inten_min
    perc = (value - inten_min) / inten_range
    sample_points=[
        [0.00, [0,0,0]],
        [1.00, [1,1,1]]
    ]
    return sample01( sample_points, perc )

def colormap_grayscale2(inten_min, inten_max, value):
    inten_range = inten_max - inten_min
    perc = (value - inten_min) / inten_range
    sample_points=[
        [0.00, [0,0,1]],
        [0.01, [0,0,0]],
        [0.99, [1,1,1]],
        [1.00, [1,0,0]]
    ]
    return sample01( sample_points, perc )

def colormap_rainbow(inten_min, inten_max, value):
    inten_range = inten_max - inten_min
    perc = (value - inten_min) / inten_range
    sample_points=[
        [0.00, [0,0,0.5]],
        [37/255, [0,0,1]],
        [98/255, [0,1,1]],
        [159/255, [1,1,0]],
        [222/255, [1,0,0]],
        [1.00, [0.5,0,0]]
    ]
    return sample01( sample_points, perc )

def colormap_green(inten_min, inten_max, value):
    inten_range = inten_max - inten_min
    perc = (value - inten_min) / inten_range
    sample_points=[
        [0.00, [0,68/255,27/255]],
        [1.00, [200/255,233/255,200/255]]
    ]
    return sample01( sample_points, perc )

def find_colormap_func(colormap):
    assert colormap in get_valid_color_mappings(), 'Invalid color mapping name "%s". Must be one of the '\
        'followings: %s.' % (colormap, str(get_valid_color_mappings()))
    if colormap == 'metalheat':
        return colormap_metalheat
    elif colormap == 'grayscale':
        return colormap_grayscale
    elif colormap == 'grayscale2':
        return colormap_grayscale2
    elif colormap == 'rainbow':
        return colormap_rainbow
    elif colormap == 'highcontrast':
        return colormap_highcontrast
    elif colormap == 'green':
        return colormap_green
    elif colormap == 'red':
        return colormap_red
    elif colormap == 'blue':
        return colormap_blue
    elif colormap == 'plasma':
        return colormap_plasma
    elif colormap == 'ratio':
        return colormap_ratio
    elif colormap == 'vik':
        return colormap_vik



