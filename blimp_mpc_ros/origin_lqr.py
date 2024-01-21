import rclpy

from . OriginLQRController import OriginLQRController
from . BlimpMPCNode import BlimpMPCNode

def main(args=None):
    print('Running origin LQR')

    try:
        rclpy.init(args=args)

        dT = 0.05
        controller = OriginLQRController(0.05)
        node = BlimpMPCNode(controller)
        
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == '__main__':
    main()
