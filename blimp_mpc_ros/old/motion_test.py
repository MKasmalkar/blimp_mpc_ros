import rclpy

from . MotionTestingController import MotionTestingController
from . BlimpMPCNode import BlimpMPCNode

import sys

def main(args=sys.argv):

    if len(args) < 2:
        print("Please run with log file name as argument.")
        sys.exit(0)

    print('Running motion test, logging to ' + args[1])

    try:
        rclpy.init(args=args)
        dT = 0.05
        controller = MotionTestingController(dT)
        node = BlimpMPCNode(controller, args[1])
        
        rclpy.spin(node)

    finally:
        node.destroy_node()

    

if __name__ == '__main__':
    main()
