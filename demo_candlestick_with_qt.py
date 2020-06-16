import os
import sys

import numpy as np
import pandas as pd
import vispy
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from vispy import (app, scene)
from vispy.color import Color

sys.path.append(os.path.dirname(__file__))
from candlestick import Candlestick

df = pd.read_pickle("y2009-m1.df")
df["x"] = np.arange(len(df))
print(f"data len:{len(df)}")

class TTMainWnd(QFrame):
    def __init__(self, parent=None):
        super(TTMainWnd, self).__init__(parent=parent)
        self._ui()

    def _ui(self):
        l1 = QVBoxLayout()
        self.setLayout(l1)
        l1.setContentsMargins(0, 0, 0, 0)

        self.canvas = scene.SceneCanvas(title="",
                                        show=False,
                                        create_native=False,
                                        px_scale=1,
                                        bgcolor=Color("#101010"),
                                        dpi=None

                                        )
        self.canvas.create_native()
        l1.addWidget(self.canvas.native, )  # not set alignment

        #
        grid = self.canvas.central_widget.add_grid()
        grid.spacing = 0

        yaxis = scene.AxisWidget(orientation='left',
                                 # axis_label='Y Axis',
                                 axis_font_size=12,
                                 axis_label_margin=50,
                                 tick_label_margin=5)
        yaxis.width_max = 50
        yaxis.stretch = (0.05, 1)
        grid.add_widget(yaxis, row=0, col=0)

        xaxis = scene.AxisWidget(orientation='bottom',
                                 axis_label='X Axis',
                                 axis_font_size=12,
                                 axis_label_margin=100,
                                 tick_label_margin=10, )

        xaxis.height_max = 100
        xaxis.stretch = (1, .1)
        grid.add_widget(xaxis, row=1, col=1)

        view: scene.ViewBox = grid.add_view(row=0, col=1, border_color='white')
        view.camera = scene.PanZoomCamera()
        view.border_color = "#ffffff"

        xaxis.link_view(view)
        yaxis.link_view(view)

        global df
        # Candlestick
        kline1 = Candlestick(borderwidth=.2, padding=.1)
        kline1.set_data(df.x.values, df.o.values, df.h.values, df.l.values, df.c.values)
        view.add(kline1)

        # MA(CLOSE,22)
        try:
            import talib
            pos = np.empty((len(df), 2), dtype=np.float32)
            pos[:, 0] = np.arange(len(df))
            pos[:, 1] = talib.MA(df.c.values, timeperiod=22)
            ma_line = scene.visuals.Line(pos, color=(1, 1, 1, 1), method='gl', width=1)
            view.add(ma_line)
        except:
            pass

        # view.camera.rect = (0, 0, 800, 7000)
        view.camera.set_range()


if __name__ == '__main__':
    DEBUG = 0
    if DEBUG:
        sys.argv.append("--vispy-fps")
        vispy.set_log_level("info")
        vispy.use(app="PyQt5", gl="pyopengl2")  # pyopengl2 for test
    else:
        vispy.set_log_level("info")
        vispy.use(app="PyQt5", gl="gl2")

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    vispy_app = vispy.app.use_app()
    vispy.app.create()
    _qt_app = vispy_app.native

    # qt style
    try:
        import qdarkstyle

        _qt_app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    except:
        pass

    _main_wnd = TTMainWnd()
    _main_wnd.show()
    vispy.app.run()
    vispy.app.quit()
