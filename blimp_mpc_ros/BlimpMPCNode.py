import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Quaternion, Vector3
from mocap_msgs.msg import RigidBody
import numpy as np

from . NonlinearBlimpSim import NonlinearBlimpSim
from . operators import *
from . utilities import *

import time

class BlimpMPCNode(Node):

    def __init__(self, controller, blimp_id=0):
        super().__init__(f'blimp_mpc_{blimp_id}')

        # Create periodic controller "interrupt"
        self.dT = controller.dT
        self.timer = self.create_timer(self.dT, self.compute_control)

        # Create publisher for sending commands to blimp    
        self.publisher_ = self.create_publisher(
            Quaternion,
            f"/agents/blimp{blimp_id}/motion_command",
            10
        )

        # Create subscribers for reading state data

        self.mocap_last_message = None
        self.gyros_last_message = None
        self.accels_last_message = None

        self.update_mocap_subscription = self.create_subscription(
            RigidBody,
            f"/rigid_bodies",
            self.read_mocap,
            1
        )

        self.update_gyros_subscription = self.create_subscription(
            Vector3,
            f"agents/blimp{blimp_id}/gyros",
            self.read_gyros,
            1
        )

        self.update_accels_subscription = self.create_subscription(
            Vector3,
            f"agents/blimp{blimp_id}/accels",
            self.read_accels,
            1
        )

        # Create controller and "simulator" variables

        self.controller = controller

        # The "simulator" is just a dummy object that passes along data from
        # the actual blimp. The blimp controller classes require a "simulator"
        # to exist, from which they read the state data.
        self.sim = NonlinearBlimpSim(self.dT)
        self.controller.init_sim(self.sim)

        # Used for computing velocity
        self.prev_x = None
        self.prev_y = None
        self.prev_z = None
        self.t_current = None
        self.t_last = None

    def read_gyros(self, msg):
        self.gyros_last_message = msg

    def read_accels(self, msg):
        self.accels_last_message = msg

    def read_mocap(self, msg):
        if self.mocap_last_message != None:
            self.prev_x = self.mocap_last_message.pose.position.x
            self.prev_y = self.mocap_last_message.pose.position.y
            self.prev_z = self.mocap_last_message.pose.position.z
            self.t_last = self.t_current

        self.mocap_last_message = msg
        self.t_current = time.time()

    def compute_control(self):

        if self.t_last == None:
            return

        start = time.time()

        x = self.mocap_last_message.pose.position.x
        y = self.mocap_last_message.pose.position.y
        z = self.mocap_last_message.pose.position.z

        v_x = (x - self.prev_x) / (self.t_current - self.t_last)
        v_y = (y - self.prev_y) / (self.t_current - self.t_last)
        v_z = (z - self.prev_z) / (self.t_current - self.t_last)
        
        w_x = self.gyros_last_message.x
        w_y = self.gyros_last_message.y
        w_z = self.gyros_last_message.z

        angles = quat2euler(np.array([self.mocap_last_message.pose.orientation.x,
                                      self.mocap_last_message.pose.orientation.y,
                                      self.mocap_last_message.pose.orientation.z,
                                      self.mocap_last_message.pose.orientation.w]))
        
        phi = angles[0]
        theta = angles[1]
        psi = angles[2]

        v_vect = np.array([v_x, v_y, v_z])
        v_rot = R_b__n(phi, theta, psi) @ v_vect

        w_vect = np.array([w_x, w_y, w_z])
        w_rot = T(phi, theta) @ w_vect

        self.sim.set_var('x', x)
        self.sim.set_var('y', y)
        self.sim.set_var('z', z)
        self.sim.set_var('vx', v_rot[0])
        self.sim.set_var('vy', v_rot[1])
        self.sim.set_var('vz', v_rot[2])
        self.sim.set_var('phi', phi)
        self.sim.set_var('theta', theta)
        self.sim.set_var('psi', psi)
        self.sim.set_var('wx', w_rot[0])
        self.sim.set_var('wy', w_rot[1])
        self.sim.set_var('wz', w_rot[2])
        
        ctrl = self.controller.get_ctrl_action(self.sim)
        fx = ctrl[0].item()
        fy = ctrl[1].item()
        fz = ctrl[2].item()
        tau_z = ctrl[3].item()

        self.write_command(fx, fy, fz, tau_z)
        
        print("Control: " + str(fx) + ", " + str(fy) + ", " + str(fz) + ", " + str(tau_z))

    def write_command(self, fx, fy, fz, tau_z):

        msg = Quaternion()

        msg.x = fx
        msg.y = fy
        msg.z = fz
        msg.w = tau_z
        
        self.publisher_.publish(msg)