from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
# not compatible with matplotlib-2.2.3, matplotlib-2.0.2 required
import numpy
import math
from render_gui import RenderGUI

def rot(theta):
    R = numpy.zeros((2,2))
    R[0,0] = math.cos(theta)
    R[0,1] = -math.sin(theta)
    R[1,0] = math.sin(theta)
    R[1,1] = math.cos(theta)
    return R


class SimulatedChainRobotGUI(RenderGUI):
    def __init__(self, render_rate=None, record_fps=None, record_sim=None, time_frame=None, lim_x=None,
                 lim_y=None, subject=None):
        RenderGUI.__init__(self, render_rate, record_fps, record_sim, time_frame, lim_x, lim_y, subject)

    def render_frame(self, subject, color='b'):
        p = numpy.zeros((2, 1))
        R = numpy.eye(2)
        # state = subject.dynamics.get_chain_state_from_robot_state(subject.state)
        state = subject.state
        q = subject.dynamics.get_q(state)
        pos_0 = subject.dynamics.get_pos_0(state)
        off_x = pos_0[0]
        off_y = pos_0[1]
        plt.xlim(self.lim_x[0], self.lim_x[1])
        plt.ylim(self.lim_y[0], self.lim_y[1])
        link_lengths = subject.dynamics.get_link_lengths()
        for i in range(0, subject.dynamics.get_num_links()):
            R = numpy.dot(R, rot(q[i]))
            l = numpy.zeros((2, 1))
            l[0, 0] = link_lengths[i]
            p_next = p + numpy.dot(R, l)
            self._ax1.add_line(mlines.Line2D((off_x+p[0], off_x+p_next[0]),
                                             (off_y+p[1], off_y+p_next[1]), color=color))
            p = p_next
