import rclpy

from . TrackingNoDamping import TrackingNoDamping
from . BlimpMPCNode import BlimpMPCNode

import sys

def main(args=sys.argv):
    print('Running feedback linearized helix follower')

    if len(args) < 2:
        print("Please run with log file name as argument.")
        sys.exit(0)
    
    try:
        rclpy.init(args=args)

        dT = 0.05
        controller = TrackingNoDamping(dT)
        node = BlimpMPCNode(controller, args[1])
        
        rclpy.spin(node)

    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()
