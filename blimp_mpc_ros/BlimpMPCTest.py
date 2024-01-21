import rclpy
import numpy as np
import signal # for interrupts
import time # for measuring execution time
from mocap_msgs.msg import RigidBody

from rclpy.node import Node

from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion

from . NonlinearBlimpSim import NonlinearBlimpSim
from . BlimpLogger import BlimpLogger
from . utilities import *

class BlimpMPCTest(Node):
    def __init__(self):
        super().__init__('blimp_test')

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
            RigidBody,
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
        self.sim.set_var('psi', np.pi/2)

    # Function: command_cb
    # Purpose: callback for the command subscriber.
    # Note that inputs are in R4, hence why we (ab)use the Quaternion message type.
    def command_cb(self, quaternion_message):
        print("State: " + str(self.sim.get_var('x')) + ", " + 
                          str(self.sim.get_var('y')) + ", " +
                          str(self.sim.get_var('z')))
        
        self.sim.update_model(np.array([quaternion_message.x,
                                        quaternion_message.y,
                                        quaternion_message.z,
                                        quaternion_message.w]))
        
    # Publishes the gyro readings to the topic /agents/blimp{blimp_id}/gyros.
    # NED frame.
    def gyro_publisher_cb(self):
        msg = Vector3()
        msg.x = self.sim.get_var('wx')
        msg.y = self.sim.get_var('wy')
        msg.z = self.sim.get_var('wz')
        self.gyro_publisher.publish(msg)

    def mocap_publisher_cb(self):
        msg = RigidBody()
        msg.rigid_body_name = "blimp0"

        msg.pose.position.x = self.sim.get_var('x')
        msg.pose.position.y = self.sim.get_var('y')
        msg.pose.position.z = self.sim.get_var('z')
        
        orientation = euler2quat(np.array([self.sim.get_var('phi'),
                                           self.sim.get_var('theta'),
                                           self.sim.get_var('psi')]))
        
        msg.pose.orientation.x = orientation[0]
        msg.pose.orientation.y = orientation[1]
        msg.pose.orientation.z = orientation[2]

        self.mocap_publisher.publish(msg)


    def destroy_node(self):
        print("Logging data...")
        logger = BlimpLogger('log.csv')
        logger.log(self.sim)
        print("Logging done!")