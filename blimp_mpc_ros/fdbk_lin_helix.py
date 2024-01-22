import rclpy

from . TrackingNoDamping import TrackingNoDamping
from . BlimpMPCNode import BlimpMPCNode

def main(args=None):
    print('Running feedback linearized helix follower')

    try:
        rclpy.init(args=args)

        dT = 0.05
        controller = TrackingNoDamping(dT)
        node = BlimpMPCNode(controller)
        
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == '__main__':
    main()
