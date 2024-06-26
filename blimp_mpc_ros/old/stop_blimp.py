import rclpy

from . StopBlimp import StopBlimp
from . BlimpMPCNode2 import BlimpMPCNode2

import sys

def main(args=sys.argv):

    if len(args) < 2:
        print("Please run with log file name as argument.")
        sys.exit(0)

    print('Stopping blimp, logging to ' + args[1])

    try:
        rclpy.init(args=args)
        dT = 0.05
        controller = StopBlimp(dT)
        node = BlimpMPCNode2(controller, args[1])
        
        rclpy.spin(node)

    finally:
        node.destroy_node()

    

if __name__ == '__main__':
    main()
