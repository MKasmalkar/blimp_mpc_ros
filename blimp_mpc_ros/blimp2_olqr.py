import rclpy

from . OriginLQRController import OriginLQRController
from . BlimpMPCNode2 import BlimpMPCNode2

import sys

def main(args=sys.argv):

    if len(args) < 2:
        print("Please run with log file name as argument.")
        sys.exit(0)

    print('Running blimp v2 origin LQR, logging to ' + args[1])

    try:
        rclpy.init(args=args)
        dT = 0.05
        controller = OriginLQRController(dT)
        node = BlimpMPCNode2(controller, args[1])
        
        rclpy.spin(node)

    finally:
        node.destroy_node()

    

if __name__ == '__main__':
    main()