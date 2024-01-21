import rclpy

from . MPCHelix import MPCHelix
from . BlimpMPCNode import BlimpMPCNode

def main(args=None):
    print('Running helix MPC')

    try:
        rclpy.init(args=args)

        dT = 0.05
        controller = MPCHelix(dT)
        node = BlimpMPCNode(controller)
        
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == '__main__':
    main()
