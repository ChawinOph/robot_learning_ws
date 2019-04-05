#!/usr/bin/env python

import numpy as np
import time
import rospy

from newtonian_chain_numpy_simulated_dynamics import NewtonianArmDynamics
from simulated_robot import SimulatedRobot
from friction import Friction
from simulated_chain_robot_gui import SimulatedChainRobotGUI

from robot_sim.srv import RobotAction
from robot_sim.srv import RobotActionRequest
from robot_sim.srv import RobotActionResponse



(lambda __g, __print: [[[(lambda __mod: [(lambda __mod: [(lambda __mod: [(lambda __mod: [(lambda __mod: [(lambda __mod: [(lambda __mod: [[(lambda __after: (rospy.init_node('real_robot', anonymous=True), [(__print('Real robot now spinning'), (rospy.spin(), __after())[1])[1] for __g['r'] in [(l1l11ll1l_opy_())]][0])[1] if (__name__ == '__main__') else __after())(lambda: None) for __g['l1l11ll1l_opy_'] in [((lambda b, d: d.get('__metaclass__', getattr(b[0], '__class__', type(b[0])))('l1l11ll1l_opy_', b, d))((object,), (lambda __l: [[[__l for __l['l1l1l1111_opy_'], __l['l1l1l1111_opy_'].__name__ in [(lambda self, l1lll1l11_opy_: (lambda __l: [[(lambda __after: (__l['self'].l1l11llll_opy_(), __after())[1] if (__l['l1lll1l11_opy_'].reset == True) else (__l['self'].l1l1llll1_opy_.take_action(np.asarray(__l['l1lll1l11_opy_'].action).reshape((-1, 1))), __after())[1])(lambda: [__l['l1l11ll1ll1l_opy_'] for __l['l1l11ll1ll1l_opy_'].robot_state in [(__l['self'].l1l1llll1_opy_.get_state()[0])]][0]) for __l['l1l11ll1ll1l_opy_'] in [(RobotActionResponse())]][0] for __l['self'], __l['l1lll1l11_opy_'] in [(self, l1lll1l11_opy_)]][0])({}), 'l1l1l1111_opy_')]][0] for __l['l1l11llll_opy_'], __l['l1l11llll_opy_'].__name__ in [(lambda self: (lambda __l: [(__l['self'].l1l1llll1_opy_.set_state(np.asarray(__l['self'].l1l1l1l11_opy_).reshape((-1, 1))), None)[1] for __l['self'] in [(self)]][0])({}), 'l1l11llll_opy_')]][0] for __l['__init__'], __l['__init__'].__name__ in [(lambda self: (lambda __l: [[[[(__l['self'].l1l11llll_opy_(), [None for __l['self'].service in [(rospy.Service('real_robot', RobotAction, __l['self'].l1l1l1111_opy_))]][0])[1] for __l['self'].l1l1l1l11_opy_ in [([-1.57, 0.0, 0.0, 0.0, 0.0, 0.0])]][0] for __l['self'].l1l1llll1_opy_ in [(SimulatedRobot(dynamics=__l['self'].dynamics, data_names=None, steps_per_action=1))]][0] for __l['self'].dynamics in [(NewtonianArmDynamics(delta_t=0.01, num_links=3, link_mass=0.1, link_length=0.5, torques_max=1.0, friction=Friction(apply_viscous_joint_friction=True, mu_viscous_joint=0.3, apply_dry_friction=True, mu_dry_forward=0.0, mu_dry_sideways=0.0, apply_viscous_friction=False, mu_viscous_forward=0.0, mu_viscous_sideways=0.0), gravity_y=-9.8, gravity_z=0, solver_type='euler', wrap_angles=False, residue_limit=0.0001))]][0] for __l['self'] in [(self)]][0])({}), '__init__')]][0])({'__module__': __name__})))]][0] for __g['RobotActionResponse'] in [(__mod.RobotActionResponse)]][0])(__import__('robot_sim.srv', __g, __g, ('RobotActionResponse',), 0)) for __g['RobotActionRequest'] in [(__mod.RobotActionRequest)]][0])(__import__('robot_sim.srv', __g, __g, ('RobotActionRequest',), 0)) for __g['RobotAction'] in [(__mod.RobotAction)]][0])(__import__('robot_sim.srv', __g, __g, ('RobotAction',), 0)) for __g['SimulatedChainRobotGUI'] in [(__mod.SimulatedChainRobotGUI)]][0])(__import__('simulated_chain_robot_gui', __g, __g, ('SimulatedChainRobotGUI',), 0)) for __g['Friction'] in [(__mod.Friction)]][0])(__import__('friction', __g, __g, ('Friction',), 0)) for __g['SimulatedRobot'] in [(__mod.SimulatedRobot)]][0])(__import__('simulated_robot', __g, __g, ('SimulatedRobot',), 0)) for __g['NewtonianArmDynamics'] in [(__mod.NewtonianArmDynamics)]][0])(__import__('newtonian_chain_numpy_simulated_dynamics', __g, __g, ('NewtonianArmDynamics',), 0)) for __g['rospy'] in [(__import__('rospy', __g, __g))]][0] for __g['time'] in [(__import__('time', __g, __g))]][0] for __g['np'] in [(__import__('numpy', __g, __g))]][0])(globals(), __import__('__builtin__', level=0).__dict__['print'])


if __name__ == '__main__':
    rospy.init_node('real_robot', anonymous=True)
    r = l1l11ll1l_opy_()
    print "Real robot now spinning"
    rospy.spin()
