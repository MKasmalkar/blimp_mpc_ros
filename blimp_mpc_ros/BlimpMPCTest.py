import rclpy
import numpy as np
import signal # for interrupts
import time # for measuring execution time
from mocap_msgs.msg import RigidBody, RigidBodies

from rclpy.node import Node

from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion

from . NonlinearBlimpSim import NonlinearBlimpSim
from . BlimpLogger import BlimpLogger
from . utilities import *

class BlimpMPCTest(Node):
    def __init__(self, filename):
        super().__init__('blimp_test')

        self.logfile = filename

        blimp_id = 0
        self.dT = 0.05
        
        self.command_subscriber = self.create_subscription(
            Quaternion,
            f"/agents/blimp{blimp_id}/motion_command",
            self.command_cb,
            2
        )
        
        # inputs
        self.command = np.zeros(4, dtype=np.double)

        self.mocap_publisher = self.create_publisher(
            RigidBodies,
            f'/rigid_bodies',
            1
        )

        self.gyro_publisher = self.create_publisher(
            Vector3,
            f'/agents/blimp{blimp_id}/gyros',
            1
        )

        self.mocap_timer = self.create_timer(
            self.dT,
            self.mocap_publisher_cb
        )

        self.gyro_timer = self.create_timer(
            self.dT,
            self.gyro_publisher_cb
        )

        self.sim = NonlinearBlimpSim(self.dT)
        self.sim.set_var('x', 1.0)
        self.sim.set_var('y', 2.0)
        self.sim.set_var('z', -3.0)
        # self.sim.set_var('phi', np.pi/10)
        # self.sim.set_var('theta', np.pi/4)
        self.sim.set_var('psi', np.pi/2)

        self.k = 0
        
    # Function: command_cb
    # Purpose: callback for the command subscriber.
    # Note that inputs are in R4, hence why we (ab)use the Quaternion message type.
    def command_cb(self, quaternion_message):
        print(str(self.k) + ": vy = " + str(self.sim.state[1]))
        print("State before command: " + str((self.sim.get_var('x'),
                                              self.sim.get_var('y'),
                                              self.sim.get_var('z'),
                                              self.sim.get_var('vy'))))

        self.sim.update_model(np.array([quaternion_message.x,
                                        quaternion_message.y,
                                        quaternion_message.z,
                                        quaternion_message.w]))
        self.k += 1

        print("State after command: " + str((self.sim.get_var('x'),
                                              self.sim.get_var('y'),
                                              self.sim.get_var('z'),
                                              self.sim.get_var('vy'))))
        # print(f"Current state: {round(self.sim.get_var('x'), 6)}, {round(self.sim.get_var('y'), 6)}, {round(self.sim.get_var('z'), 6)}, {round(self.sim.get_var('psi'), 6)}")
        
    # Publishes the gyro readings to the topic /agents/blimp{blimp_id}/gyros.
    # NED frame.
    def gyro_publisher_cb(self):
        msg = Vector3()
        msg.x = self.sim.get_var('wx')
        msg.y = self.sim.get_var('wy')
        msg.z = self.sim.get_var('wz')
        
        self.gyro_publisher.publish(msg)

    def mocap_publisher_cb(self):
        msg = RigidBodies()

        blimp = RigidBody()
        blimp.rigid_body_name = "blimp0"

        blimp.pose.position.x = self.sim.get_var('x')
        blimp.pose.position.y = self.sim.get_var('y')
        blimp.pose.position.z = self.sim.get_var('z')

        print("Publishing state: " + str((blimp.pose.position.x,
                                          blimp.pose.position.y,
                                          blimp.pose.position.z)))

        dynamics_psi = self.sim.get_var('psi')
        mocap_psi = dynamics_psi % (2*np.pi)
        if mocap_psi > np.pi:
            mocap_psi -= 2*np.pi

        orientation = euler2quat(np.array([self.sim.get_var('phi'),
                                           self.sim.get_var('theta'),
                                           mocap_psi]))
        
        blimp.pose.orientation.x = orientation[0]
        blimp.pose.orientation.y = orientation[1]
        blimp.pose.orientation.z = orientation[2]
        blimp.pose.orientation.w = orientation[3]

        msg.rigidbodies = [blimp]

        self.mocap_publisher.publish(msg)


    def destroy_node(self):
        print("Logging data...")
        logger = BlimpLogger(self.logfile)
        logger.log(self.sim)
        print("Logging done!")