import rclpy

from . BlimpMPCTest import BlimpMPCTest

import sys

def main(args=sys.argv):

    if len(args) < 2:
        print("Please run with log file name as argument.")
        sys.exit(0)

    print('Simulating blimp node, logging to ' + args[1])

    try:
        rclpy.init(args=args)

        node = BlimpMPCTest(args[1])
        
        rclpy.spin(node)

    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()
