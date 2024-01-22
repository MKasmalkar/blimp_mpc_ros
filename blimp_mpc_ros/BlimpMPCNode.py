import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Quaternion, Vector3
from mocap_msgs.msg import RigidBody
import numpy as np

from . NonlinearBlimpSim import NonlinearBlimpSim
from . operators import *
from . utilities import *

from . BlimpLogger import BlimpLogger

import time

class BlimpMPCNode(Node):

    def __init__(self, controller, logfile, blimp_id=0):
        super().__init__(f'blimp_mpc_{blimp_id}')

        self.logfile = logfile

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

        # Create controller and "simulator" variables

        self.controller = controller

        # The "simulator" is just a dummy object that passes along data from
        # the actual blimp. The blimp controller classes require a "simulator"
        # to exist, from which they read the state data.
        self.sim = NonlinearBlimpSim(self.dT)
        self.controller.init_sim(self.sim)

        # Used for computing derivatives
        self.position_history = None    # [x, y, z]
        self.velocity_history = None    # [vx, vy, vz]
        self.angle_history = None       # [phi, theta, psi]
        self.ang_vel_history = None     # [wx, wy, wz]

        self.pos_dot_history = None
        self.vel_dot_history = None
        self.ang_dot_history = None
        self.ang_vel_dot_history = None

        self.full_rotations = 0
        self.prev_mocap_psi = None

        self.last_mocap_timestamp = None
        self.last_gyro_timestamp = None
        self.mocap_k = -1   # index of most recent mocap msg in state history arrays
        self.gyro_k = -1    # index of most recent gyro msg in angle history array

        self.state_variables_valid = False

    def read_velocities(self, msg):
        self.vel_last_message = msg

    def read_angles(self, msg):
        self.angles_last_message = msg

    def read_gyros(self, msg):
        current_time = time.time()

        self.gyro_k += 1

        # Gyros give you wx, wy, wz
        w_x = msg.x
        w_y = msg.y
        w_z = msg.z

        # TODO: figure out if w_x, etc. are world or body frame

        if self.ang_vel_history is None:
            self.ang_vel_history = np.array([w_x, w_y, w_z]).reshape((3,1))
        else:
            self.ang_vel_history = np.hstack((self.ang_vel_history,
                                              np.array([w_x, w_y, w_z]).reshape((3,1))))
            
        if self.last_gyro_timestamp == None:
            self.last_gyro_timestamp = current_time
            w_x_dot = 0
            w_y_dot = 0
            w_z_dot = 0
            
            self.ang_vel_dot_history = np.array([w_x_dot,
                                                 w_y_dot,
                                                 w_z_dot]).reshape((3,1))
        else:
            deltaT = current_time - self.last_gyro_timestamp
            self.last_gyro_timestamp = current_time

            ang_vel_dot = (np.array([w_x, w_y, w_z]) - self.ang_vel_history[:, self.gyro_k-1]) / deltaT
            self.ang_vel_dot_history = np.hstack((self.ang_vel_dot_history, ang_vel_dot.reshape((3,1))))

    def read_mocap(self, msg):
        current_time = time.time()

        self.mocap_k += 1

        # Mocap gives you x, y, z, phi, theta, psi
        # Can compute vx, vy, vz

        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        angles = quat2euler(np.array([msg.pose.orientation.x,
                                      msg.pose.orientation.y,
                                      msg.pose.orientation.z,
                                      msg.pose.orientation.w]))
        
        phi = angles[0]
        theta = angles[1]
        mocap_psi = angles[2]

        psi = None
        
        if self.mocap_k == 0:
            self.prev_mocap_psi = mocap_psi
            psi = mocap_psi

        elif self.mocap_k > 0:
            # mocap angles are from -pi to pi, whereas the angle state variable
            # in the MPC is an absolute angle (i.e. no modulus)
            
            # I correct for this discrepancy here

            if self.prev_mocap_psi > np.pi*0.9 and mocap_psi < -np.pi*0.9:
                # Crossed 180 deg, CCW
                self.full_rotations += 1

            elif self.prev_mocap_psi < -np.pi*0.9 and mocap_psi > np.pi*0.9:
                # Crossed 180 deg, CW
                self.full_rotations -= 1

            psi = mocap_psi + 2*np.pi * self.full_rotations

            self.prev_mocap_psi = mocap_psi


        current_pos_vector = np.array([x, y, z]).reshape((3,1))
        current_ang_vector = np.array([phi, theta, psi]).reshape((3,1))

        if self.position_history is None:
            self.position_history = np.array(current_pos_vector).reshape((3, 1))
        else:
            self.position_history = np.hstack((self.position_history, current_pos_vector))

        if self.angle_history is None:
            self.angle_history = np.array(current_ang_vector).reshape((3, 1))
        else:
            self.angle_history = np.hstack((self.angle_history, current_ang_vector))

        if self.last_mocap_timestamp is None:
            self.last_mocap_timestamp = current_time

            self.velocity_history = np.array([0, 0, 0]).reshape((3,1))

            self.vel_dot_history = np.array([0, 0, 0]).reshape((3,1))
            self.pos_dot_history = np.array([0, 0, 0]).reshape((3,1))
            self.ang_dot_history = np.array([0, 0, 0]).reshape((3,1))

        else:
            deltaT = current_time - self.last_mocap_timestamp
            self.last_mocap_timestamp = current_time

            v_x_n = (self.position_history[0][self.mocap_k]
                    - self.position_history[0][self.mocap_k-1]) / deltaT
            v_y_n = (self.position_history[1][self.mocap_k]
                    - self.position_history[1][self.mocap_k-1]) / deltaT
            v_z_n = (self.position_history[2][self.mocap_k]
                    - self.position_history[2][self.mocap_k-1]) / deltaT
            
            self.pos_dot_history = np.hstack((self.pos_dot_history,
                                              np.array([v_x_n, v_y_n, v_z_n]).reshape((3,1))))

            phi_dot = (self.angle_history[0][self.mocap_k]
                       - self.angle_history[0][self.mocap_k-1]) / deltaT
            theta_dot = (self.angle_history[1][self.mocap_k]
                       - self.angle_history[1][self.mocap_k-1]) / deltaT
            psi_dot = (self.angle_history[2][self.mocap_k]
                       - self.angle_history[2][self.mocap_k-1]) / deltaT

            self.ang_dot_history = np.hstack((self.ang_dot_history,
                                              np.array([phi_dot, theta_dot, psi_dot]).reshape((3,1))))

            vel_vect_n = np.array([v_x_n, v_y_n, v_z_n]).T
            vel_vect_b = R_b__n_inv(phi, theta, psi) @ vel_vect_n
            self.velocity_history = np.hstack((self.velocity_history, vel_vect_b.reshape((3,1))))

            vel_dot = (self.velocity_history[:, self.mocap_k] - self.velocity_history[:, self.mocap_k-1]) / deltaT
            self.vel_dot_history = np.hstack((self.vel_dot_history, vel_dot.reshape((3,1))))

            self.state_variables_valid = True

    def compute_control(self):
        
        if not self.state_variables_valid:
            return

        x = self.position_history[0][self.mocap_k]
        y = self.position_history[1][self.mocap_k]
        z = self.position_history[2][self.mocap_k]

        v_x = self.velocity_history[0][self.mocap_k]
        v_y = self.velocity_history[1][self.mocap_k]
        v_z = self.velocity_history[2][self.mocap_k]

        phi = self.angle_history[0][self.mocap_k]
        theta = self.angle_history[1][self.mocap_k]
        psi = self.angle_history[2][self.mocap_k]

        w_x = self.ang_vel_history[0][self.gyro_k]
        w_y = self.ang_vel_history[1][self.gyro_k]
        w_z = self.ang_vel_history[2][self.gyro_k]

        self.sim.set_var('x', x)
        self.sim.set_var('y', y)
        self.sim.set_var('z', z)
        self.sim.set_var('vx', v_x)
        self.sim.set_var('vy', v_y)
        self.sim.set_var('vz', v_z)
        self.sim.set_var('phi', phi)
        self.sim.set_var('theta', theta)
        self.sim.set_var('psi', psi)
        self.sim.set_var('wx', w_x)
        self.sim.set_var('wy', w_y)
        self.sim.set_var('wz', w_z)
        
        self.sim.set_var_dot('x', self.pos_dot_history[0][self.mocap_k])
        self.sim.set_var_dot('y', self.pos_dot_history[1][self.mocap_k])
        self.sim.set_var_dot('z', self.pos_dot_history[2][self.mocap_k])
        self.sim.set_var_dot('phi', self.ang_dot_history[0][self.mocap_k])
        self.sim.set_var_dot('theta', self.ang_dot_history[1][self.mocap_k])
        self.sim.set_var_dot('psi', self.ang_dot_history[2][self.mocap_k])
        self.sim.set_var_dot('vx', self.vel_dot_history[0][self.mocap_k])
        self.sim.set_var_dot('vy', self.vel_dot_history[1][self.mocap_k])
        self.sim.set_var_dot('vz', self.vel_dot_history[2][self.mocap_k])
        self.sim.set_var_dot('wx', self.ang_vel_history[0][self.gyro_k])
        self.sim.set_var_dot('wy', self.ang_vel_history[1][self.gyro_k])
        self.sim.set_var_dot('wz', self.ang_vel_history[2][self.gyro_k])

        self.sim.update_history()
        
        ctrl = self.controller.get_ctrl_action(self.sim)
        fx = ctrl[0].item()
        fy = ctrl[1].item()
        fz = ctrl[2].item()
        tau_z = ctrl[3].item()

        self.write_command(fx, fy, fz, tau_z)

    def write_command(self, fx, fy, fz, tau_z):

        print(f"Control: {round(fx, 6)}, {round(fy, 6)}, {round(fz, 6)}, {round(tau_z, 6)}")

        msg = Quaternion()

        msg.x = fx
        msg.y = fy
        msg.z = fz
        msg.w = tau_z
        
        self.publisher_.publish(msg)

    def destroy_node(self):
        print("Logging data...")
        logger = BlimpLogger(self.logfile)
        logger.log(self.sim)
        print("Logging done!")