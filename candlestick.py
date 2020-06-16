# -*- coding: utf-8 -*-
"""
CandlestickVisual

"""

import numba
import numpy as np
from vispy import gloo, scene
from vispy.util.profiler import Profiler
from vispy.util.svg.color import Color
from vispy.visuals import CompoundVisual, Visual
from vispy.visuals.shaders import Function

_BAR_POINTS = 12


@numba.jit()
def bar_points(x, w, o, h, l, c, align="r", d=0.05):
    U, B = max(o, c), min(o, c)
    if align == "m":
        M = x
    elif align == "l":
        M = x + w / 2
    else:
        M = x - w / 2
    d = w * d
    L = M - w / 2 + d
    R = M + w / 2 - d
    return [[L, U], [R, U], [R, U], [R, B], [R, B], [L, B], [L, B], [L, U],
            [M, h], [M, U], [M, B], [M, l]
            ]


vec2to4 = Function("""
    vec4 vec2to4(vec2 inp) {
        return vec4(inp, 0, 1);
    }
""")

vec3to4 = Function("""
    vec4 vec3to4(vec3 inp) {
        return vec4(inp, 1);
    }
""")


class _GLLine2Visual(Visual):
    """

    """
    VERTEX_SHADER = """
        #version 120
        varying vec4 v_color;

        void main()
        {
           gl_Position = $transform($to_vec4($position));
           v_color = $color;
        }
    """

    FRAGMENT_SHADER = """
        #version 120
        varying vec4 v_color;

        void main()
        {
            gl_FragColor = v_color;
        }
    """

    def __init__(self, parent):
        self._parent = parent
        self._pos_vbo = gloo.VertexBuffer()
        self._color_vbo = gloo.VertexBuffer()

        Visual.__init__(self, vcode=self.VERTEX_SHADER,
                        fcode=self.FRAGMENT_SHADER, gcode=None)

        self._draw_mode = 'lines'
        self._connect = 'segments'
        self.set_gl_state('translucent', depth_test=False)
        self.freeze()

    def _prepare_transforms(self, view):
        xform = view.transforms.get_transform()
        view.view_program.vert['transform'] = xform

    def _prepare_draw(self, view):
        prof = Profiler()

        if self._parent._changed['pos']:
            if self._parent._pos is None or len(self._parent._pos) == 0:
                return False
            # pos = np.ascontiguousarray(self._parent._pos.astype(np.float32))
            pos = self._parent._pos
            self._pos_vbo.set_data(pos)
            self._program.vert['position'] = self._pos_vbo
            if pos.shape[-1] == 2:
                self._program.vert['to_vec4'] = vec2to4
            elif pos.shape[-1] == 3:
                self._program.vert['to_vec4'] = vec3to4
            else:
                raise TypeError("Got bad position array shape: %r"
                                % (pos.shape,))

            self._color_vbo.set_data(self._parent._color)
            self._program.vert['color'] = self._color_vbo

        try:
            import OpenGL.GL as GL
            # Turn on line smooth and/or line width
            if GL:
                if self._parent._antialias:
                    GL.glEnable(GL.GL_LINE_SMOOTH)
                else:
                    GL.glDisable(GL.GL_LINE_SMOOTH)
                px_scale = self.transforms.pixel_scale
                width = px_scale * self._parent._border_width
                GL.glLineWidth(max(width, 1.))
        except Exception:  # can be other than ImportError sometimes
            pass

        prof('prepare')
        # Draw
        self._connect = 'segments'
        self._draw_mode = 'lines'
        prof('draw')


class CandlestickVisual(CompoundVisual):
    """
    Candlestick visual

    """

    def __init__(self, colorup="#00ff00ff", colordown="#ff0000ff", log: bool = False,
                 align="m", padding=.05, borderwidth: float = .5, antialias=False):
        """

        Args:
            colorup: the color of the lines where close >= open
            colordown:the color of the lines where close < open
            antialias:
            log: y log scale
            padding: bar padding-left and padding-right, %
            borderwidth: border width
            align: 'r','l','m'

        """
        assert align in ['r', 'l', 'm']

        self._line_visual = _GLLine2Visual(self)

        self._changed = {'pos': False, }

        self._pos = None
        self._color = None
        self._border_width = borderwidth
        self._bounds = None
        self._antialias = antialias

        self._log = log
        self._data = None
        self._colorup = Color(colorup).rgba
        self._colordown = Color(colordown).rgba
        self._align = align
        self._padding = padding

        CompoundVisual.__init__(self, [self._line_visual, ])

    @property
    def antialias(self):
        return self._antialias

    @antialias.setter
    def antialias(self, aa):
        self._antialias = bool(aa)
        self.update()

    @property
    def log(self):
        """ The height of the rectangle.
        """
        return self._log

    @log.setter
    def log(self, log: bool):
        if self._log == log:
            return
        self._log = log
        self._regen_pos()
        self.update()

    def _regen_pos(self):
        vertices = [bar_points(*i, align=self._align, d=self._padding) for i in self._data]
        self._pos = np.array(vertices).astype(np.float32)

        v_color = np.repeat([self._colorup], len(self._data), axis=0)
        f = self._data["c"] < self._data["o"]
        v_color[f] = self._colordown
        self._color = np.repeat(v_color, _BAR_POINTS, axis=0).astype(np.float32)

    def set_data(self, xpos, opens, highs, lows, closes, width=None):
        """Set the data used to draw this visual.

        Args:
            xpos:   X-axes
            opens:  sequence of opening values
            highs:  sequence of high values
            lows:   sequence of low values
            closes: sequence of closing values
            width:  bar width,None or float value

        """
        if xpos is None:
            xpos = np.arange(len(opens))

        assert len(xpos) == len(opens) == len(highs) == len(lows) == len(closes)

        if width is None:
            w = np.full(len(xpos), 1)
            w[1:] = np.diff(xpos)
            w[0] = w[1]
            _bar_width = w
        else:
            _bar_width = np.full(len(xpos), width)

        self._data = np.zeros((len(xpos),),
                              dtype=[('x', 'f4'), ('w', 'f4'),
                                     ('o', 'f4'), ('h', 'f4'), ('l', 'f4'), ('c', 'f4')])

        self._data['x'] = xpos
        self._data['w'] = _bar_width
        self._data['o'] = np.log(opens) if self._log else opens
        self._data['h'] = np.log(highs) if self._log else highs
        self._data['l'] = np.log(lows) if self._log else lows
        self._data['c'] = np.log(closes) if self._log else closes

        f_nan=np.isnan(self._data['o']) |np.isnan(self._data['h']) |np.isnan(self._data['l'])|np.isnan(self._data['c'])
        self._data=self._data[~f_nan]

        self._bounds = None
        self._regen_pos()
        self._changed['pos'] = True
        self.update()

    @property
    def pos(self):
        return self._pos

    # def _interpret_connect(self):
    #     return "segments"

    def _compute_bounds(self, axis, view):
        """Get the bounds

        Parameters
        ----------
        mode : str
            Describes the type of boundary requested. Can be "visual", "data",
            or "mouse".
        axis : 0, 1, 2
            The axis along which to measure the bounding values, in
            x-y-z order.
        """
        if (self._bounds is None) and self._data is not None:
            self._bounds = [
                [self._data["x"].min(), self._data["x"].max()],
                [self._data["l"].min(), self._data["h"].max()],
            ]

        if self._bounds is None:
            return
        else:
            if axis < len(self._bounds):
                return self._bounds[axis]
            else:
                return (0, 0)

    def _prepare_draw(self, view):
        if self._border_width == 0:
            return False
        CompoundVisual._prepare_draw(self, view)


Candlestick = scene.visuals.create_visual_node(CandlestickVisual)
