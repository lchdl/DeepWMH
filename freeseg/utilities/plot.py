from typing import Union, Tuple
import warnings
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import matplotlib.ticker
from freeseg.utilities.file_ops import file_exist, gd, mkdir

#####################################################
## define some plot functions for general purposes ##
#####################################################

def single_curve_plot(x,y,save_file,fig_size=(8,6),dpi=80,
    title=None,xlabel=None,ylabel=None,log_x=False,log_y=False,
    xlim=None, ylim=None):

    plt.figure('figure',figsize=fig_size,dpi=dpi)
    plt.plot(x, y, color=[235/255,64/255,52/255])
    if title is not None: plt.title(title)
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
    if log_x:
        plt.xscale('log')
        locmin = matplotlib.ticker.LogLocator(base=10,subs=(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),numticks=100)
        plt.gca().xaxis.set_minor_locator(locmin)
        plt.gca().xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if log_y: 
        plt.yscale('log')
        locmin = matplotlib.ticker.LogLocator(base=10,subs=(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),numticks=100)
        plt.gca().yaxis.set_minor_locator(locmin)
        plt.gca().yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    plt.grid(which='both',ls='--',lw=1,color=[200/255,200/255,200/255])
    plt.savefig(save_file)
    plt.close('figure')

def multi_curve_plot(curve_dict,save_file,fig_size=(8,6),dpi=80,
    title=None,xlabel=None,ylabel=None,log_x=False,log_y=False,
    xlim=None, ylim=None, selected_keys:Union[list, None]=None):
    '''
    multi_curve_plot: display multiple 2d curves in a single plot

    curve_dict: containing data for drawing. Each key-value pair in this dictionary 
    represents a single curve, the properties are: 
    'x': list of x coordinates
    'y': list of y coordinates
    'color': curve color
    'label': show label in legend, can be True or False
    'ls': line style
    'lw': line width
    'selected_keys': list of curve names that need to be drawn, set to "None" will select all curves.
    '''

    plt.figure('figure',figsize=fig_size,dpi=dpi)

    curve_names = list(curve_dict.keys())
    if selected_keys is not None:
        curve_names0 = [item for item in curve_names if item in selected_keys]
        curve_names = curve_names0

    for cname in curve_names:
        x = curve_dict[cname]['x']
        y = curve_dict[cname]['y']
        color = curve_dict[cname]['color']
        ls = '-' if 'ls' not in curve_dict[cname] else curve_dict[cname]['ls']
        label = cname if ('label' not in curve_dict[cname]) or (curve_dict[cname]['label']==True) else None
        lw = 1.5 if 'lw' not in curve_dict[cname] else curve_dict[cname]['lw']
        plt.plot(x, y, color=color,label=label,ls=ls,lw=lw)
    plt.legend()
    if title is not None: plt.title(title)
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
    if log_x:
        plt.xscale('log')
        locmin = matplotlib.ticker.LogLocator(base=10,subs=(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),numticks=100)
        plt.gca().xaxis.set_minor_locator(locmin)
        plt.gca().xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if log_y: 
        plt.yscale('log')
        locmin = matplotlib.ticker.LogLocator(base=10,subs=(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),numticks=100)
        plt.gca().yaxis.set_minor_locator(locmin)
        plt.gca().yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    plt.grid(which='both',ls='--',lw=1,color=[200/255,200/255,200/255])
    plt.savefig(save_file)
    plt.close('figure')

############################################
## Simple PDF renderer based on reportlab ##
############################################

from typing import Union
from reportlab.graphics import renderPDF
from reportlab.pdfgen import canvas
from svglib.svglib import svg2rlg
from reportlab.lib.units import mm, cm, inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

class PlotCanvas:
    '''
    * A class that draws things in vector format and saved in PDF files.
    '''

    @staticmethod
    def parse_unit(s: str) -> float:
        '''
        parse mm, cm, inch to pixel
        '''
        if s.find('mm')!=-1:
            s = s.replace('mm','').strip()
            return float(s) * mm
        elif s.find('cm')!=-1:
            s = s.replace('cm','').strip()
            return float(s) * cm
        elif s.find('inch')!=-1 or s.find('in')!=-1 :
            s = s.replace('inch','').replace('in','').strip()
            return float(s) * inch
        else:
            # defaults to cm
            return float(s) * cm
    @staticmethod
    def parse_position(s: Union[str,tuple]) -> Tuple[float, float]:
        '''
        s = "5cm, 3.4cm" or 
        s = "4mm, 1mm"   or 
        s = (5, 3.4). 
        default unit is cm
        '''
        pos = None
        if isinstance(s, str):
            pos = (PlotCanvas.parse_unit(s.split(',')[0]),
                PlotCanvas.parse_unit(s.split(',')[1]))
        elif isinstance(s, tuple):
            pos = (s[0] * cm, s[1] * cm)
        else:
            raise RuntimeError('unknown position: "%s"' % str(s))
        return pos
        
    def __init__(self, output_file:str="output.pdf", pagesize:str="21.0cm*29.7cm"):
        # 1 pixel = 1/72 inch (72 DPI)
        self.output_file = output_file
        self.pagesize_desc = pagesize
        self.pagesize_in_px = ( 
            PlotCanvas.parse_unit(pagesize.split('*')[0]), 
            PlotCanvas.parse_unit(pagesize.split('*')[1]) )
        self.canvas = canvas.Canvas(self.output_file, pagesize=self.pagesize_in_px)

    def save(self):
        '''
        * Save the current canvas.
        '''
        mkdir(gd(self.output_file))
        self.canvas.save()
    
    def add_svg(self, svg_file:str, position: Union[str,tuple] = "0cm, 0cm"):
        '''
        * Adding a "*.svg" file onto the canvas.
        '''
        pos = PlotCanvas.parse_position(position)
        renderPDF.draw(svg2rlg(svg_file), self.canvas, pos[0], pos[1])

    def register_font(self, font_file, font_name):
        '''
        * Register a font to the library and use it later.

        for example: 
        >>> canvas.register_font("./arial.ttf", "Arial") # register font
        >>> canvas.text("Hello World!", (1,1), "Arial")  # then use it
        '''
        pdfmetrics.registerFont(TTFont(font_name, font_file))

    def text(self, s, position, font_name, font_size, font_color=[0,0,0], alpha=1.0):
        '''
        * Draw a line of text onto the canvas.

        s: the text needs to be drawn
        position: text position
        font_name: builtin font or a user font that is properly registered
            using register_font(...). Font names are case sensitive!
        font_size: text size
        '''
        pos = PlotCanvas.parse_position(position)
        self.canvas.setFont(font_name, font_size)
        self.canvas.setFillColorRGB(font_color[0],font_color[1], font_color[2], alpha = alpha)
        self.canvas.setStrokeColorRGB(font_color[0],font_color[1], font_color[2], alpha = alpha)
        self.canvas.drawString(pos[0],pos[1],s)

    def line(self, position_start, position_end, line_width, line_color=[0,0,0], alpha=1.0,
        dashed = False, dash_pattern = (3,3)):

        ps = PlotCanvas.parse_position(position_start)
        pe = PlotCanvas.parse_position(position_end)
        self.canvas.setLineWidth(line_width)
        self.canvas.setStrokeColorRGB(line_color[0], line_color[1], line_color[2], alpha=alpha)
        if dashed:
            self.canvas.setDash(dash_pattern[0], dash_pattern[1])
        else:
            self.canvas.setDash() # dash off
        self.canvas.line(ps[0],ps[1],pe[0],pe[1])

    def rect(self, position_start, position_end, line_width, line_color = [0,0,0], fill_color=[1,1,1], 
        line_alpha = 1.0, fill_alpha = 1.0):

        ps = PlotCanvas.parse_position(position_start)
        pe = PlotCanvas.parse_position(position_end)
        self.canvas.setLineWidth(line_width)
        if line_color is not None:
            self.canvas.setStrokeColorRGB(line_color[0], line_color[1], line_color[2], alpha=line_alpha)
        if fill_color is not None:
            self.canvas.setFillColorRGB(fill_color[0], fill_color[1], fill_color[2], alpha=fill_alpha)
        self.canvas.rect(ps[0],ps[1], pe[0]-ps[0], pe[1]-ps[1],
            stroke=1 if line_color is not None else 0,
            fill = 1 if fill_color is not None else 0)

    def image(self, position_start, position_end, image_path:str):
        '''
        if position_end == None, then image will be drawn at a scale of 1 point to 1 pixel
        '''

        assert file_exist(image_path), 'Image "%s" not exists.' % image_path

        ps = PlotCanvas.parse_position(position_start)
        if position_end is not None:
            pe = PlotCanvas.parse_position(position_end)
        else:
            pe = (None,None)

        self.canvas.drawImage(image_path, ps[0], ps[1], 
            width=pe[0]-ps[0] if pe[0] is not None else None, 
            height=pe[1]-ps[1] if pe[1] is not None else None,
            mask=None)


from freeseg.utilities.colormaps import find_colormap_func
import numpy as np

def plot_mat(m: np.ndarray, save_file: str = "mat.pdf", cmap: str = 'grayscale', normalize_data: bool = True):
    if normalize_data:
        m = (m - np.min(m)) / (np.max(m) - np.min(m) + 0.00000001)
    else:
        if np.min(m) < -0.00001 or np.max(m) > 1.0000001:
            warnings.warn('Detected out of range value in matrix. When "normalize_data=False", '
                'we assume the data are in range [0,1], got[%f,%f].' % (np.min(m),np.max(m)))
    
    colormap_func = find_colormap_func(cmap)

    rows, cols = m.shape[0], m.shape[1]
    blocksize = (0.5,0.5) # (height, width), unit: cm
    pagesize =(blocksize[1]*cols, blocksize[0]*rows)

    cv = PlotCanvas(save_file, "%fcm*%fcm" % (pagesize[0],pagesize[1]))
    for row in range(rows):
        for col in range(cols):
            val = m[row,col]
            color = colormap_func(0,1,val)
            p_start = (col*blocksize[1],(rows-1-row)*blocksize[0])
            p_end = (p_start[0]+blocksize[1], p_start[1]+blocksize[0])
            cv.rect(p_start, p_end,0,line_color=None,fill_color=color)

    cv.save()
    


