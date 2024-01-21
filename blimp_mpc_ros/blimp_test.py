import rclpy

from . BlimpMPCTest import BlimpMPCTest

def main(args=None):
    print('Running origin LQR')

    try:
        rclpy.init(args=args)

        node = BlimpMPCTest()
        
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == '__main__':
    main()
