from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# not compatible with matplotlib-2.2.3, matplotlib-2.0.2 required
import math
import numpy as np
import time
import threading
import ast
from six import iteritems, itervalues

def rot(theta):
    R = np.zeros((2,2))
    R[0,0] = math.cos(theta)
    R[0,1] = -math.sin(theta)
    R[1,0] = math.sin(theta)
    R[1,1] = math.cos(theta)
    return R

class RenderGUI(object):

    def __init__(self, render_rate=None, record_fps=None, record_sim=None, time_frame=None,
                 lim_x=None, lim_y=None, subject=None):
        # self.subject = subject
        self.subject_dict = {}
        if subject:
            self.add_subject(subject)
        self.start_time = time.time()
        self.figure_created = False

        # recordingZ
        self.render_dir = ''
        self.prev_frame_time = 0
        self.frame_count = 0

        self.point_dict = None
        self.vel_dir = None

        self.rate = render_rate
        self.record_fps = record_fps
        self.record_sim = record_sim
        self.time_frame = time_frame
        self.lim_x = lim_x
        self.lim_y = lim_y

    def initialize_from_config(self, config_data, section_name):
        self.rate = config_data.getfloat(section_name, 'render_rate')
        self.record_fps = config_data.getfloat(section_name, 'record_fps')
        self.record_sim = config_data.getboolean(section_name, 'record_sim')
        self.time_frame = config_data.get(section_name, 'time_frame')
        self.lim_x = [float(x) for x in ast.literal_eval(config_data.get(section_name, 'lim_x'))]
        self.lim_y = [float(x) for x in ast.literal_eval(config_data.get(section_name, 'lim_y'))]

    def add_subject(self, subject):
        len_dict = len(self.subject_dict)
        self.subject_dict[len_dict] = subject

    def setup_figure(self):
        if self.figure_created is False:
            self._fig = plt.figure(figsize=(10, 10))
            self._ax1 = self._fig.add_subplot(111)
            self.figure_created = True
            self._fig.canvas.draw()
            plt.show(block=False)
        plt.figure(self._fig.number)

    def render(self):
        self.setup_figure()
        self._ax1.clear()
        lines = []
        labels = []
        color_idx = 0
        #color_idx_list = np.linspace(0, 1, len(self.subject_dict))
        colors = cm.winter(np.linspace(0, 1, len(self.subject_dict)))
        for name, subject in iteritems(self.subject_dict):
            self.render_frame(subject, color=colors[color_idx])
            labels.append(name)
            line, = self._ax1.plot([0], [0], color=colors[color_idx], lw=4)
            lines.append(line)
            color_idx += 1
        self.add_legend(lines, labels, loc='upper left')

        if self.point_dict is not None:
            lines, labels = self.plot_point_dict(self.point_dict)
            self.add_legend(lines, labels, loc='upper right')

        self.add_time_info(self.subject)

        self._fig.canvas.draw()

    def plot_point_dict(self, point_dict):
        lines = []
        labels = []
        for label, point in point_dict.items():
            labels.append(label)
            line, = self._ax1.plot(point[0], point[1], marker='*')
            lines.append(line)
        return lines, labels

    def add_legend(self, lines, labels, loc):
        legend = plt.legend(lines, labels, loc=loc)
        self._ax1.add_artist(legend)

    def plot_arrow(self, pos):
        plt.arrow(x = 2*pos[0], y = 2*pos[1], dx = 0.1*pos[0], dy = 0.1*pos[1], color = 'r', width = 0.01)

    def render_frame(self, subject, color='b'):
        raise NotImplementedError

    def save_frame_based_on_fps(self, save_path):
        current_frame_time = self.get_time()

        if (current_frame_time - self.prev_frame_time) > (1.0 / self.record_fps):
            self.save_frame(save_path)
            self.prev_frame_time = current_frame_time
            self.frame_count += 1
            # print('saving frame..')

    def save_frame(self, save_path):
        plt.savefig(save_path)

    def loop(self):
        prev_time = time.time()
        while True:
            self.render()
            curr_time = time.time()
            if (curr_time-prev_time) < self.rate:
                time.sleep(self.rate - (curr_time-prev_time))
            prev_time = curr_time
            if self.record_sim is True:
                save_path = PathGenerator.get_gui_render_path(self.render_dir, self.frame_count)
                self.save_frame_based_on_fps(save_path)

    def start(self):
        #Cannot actually run gui in separate thread
        self.my_thread = threading.Thread(target=self.loop)
        self.my_thread.daemon = True
        self.my_thread.start()

    def set_render_dir(self, render_dir):
        self.render_dir = render_dir

    def set_point_dict(self, point_dict):
        self.point_dict = point_dict

    def set_goal(self, goal):
        if self.point_dict is None:
            self.point_dict = {}
        self.point_dict['goal'] = goal

    def show_vel_dir(self, heading_angle):
        heading_angle = math.radians(heading_angle)
        self.vel_dir = [math.cos(heading_angle), math.sin(heading_angle)]

    def reset(self):
        self.prev_frame_time = 0.0

    def add_time_info(self, subject):
        time_str = ('model : ' + str(round(subject.get_time(), 3)) + 's\n'
                    + '    cpu : ' + str(round(time.time() - self.start_time, 1)) + 's')
        plt.text(0.825, 0.05, time_str, ha='left', va='top', transform=self._ax1.transAxes)

    def get_time(self):
        subjects = [subject for subject in itervalues(self.subject_dict)]
        if self.time_frame == 'robot':
            return subjects[0].get_time()
        elif self.time_frame == 'cpu':
            return time.time()

    @property
    def subject(self):
        return self.subject_dict[0]
