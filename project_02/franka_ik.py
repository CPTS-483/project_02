import numpy as np
import math
import rclpy
import tf2_ros
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf_transformations import quaternion_matrix


def rotation_matrix_x(theta):
    """
    Compute the 3D rotation matrix around the x-axis.
    
    Parameters:
    theta (float): Rotation angle in radians.
    
    Returns:
    np.array: 3x3 rotation matrix.
    """
    return np.array([
        [1, 0, 0],
        [0, math.cos(theta), -math.sin(theta)],
        [0, math.sin(theta), math.cos(theta)]
    ])

def rotation_matrix_y(theta):
    """
    Compute the 3D rotation matrix around the y-axis.
    
    Parameters:
    theta (float): Rotation angle in radians.
    
    Returns:
    np.array: 3x3 rotation matrix.
    """
    return np.array([
        [math.cos(theta), 0, math.sin(theta)],
        [0, 1, 0],
        [-math.sin(theta), 0, math.cos(theta)]
    ])

def rotation_matrix_z(theta):
    """
    Compute the 3D rotation matrix around the z-axis.
    
    Parameters:
    theta (float): Rotation angle in radians.
    
    Returns:
    np.array: 3x3 rotation matrix.
    """
    return [np.array([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1]
        ])]

def rotation_matrix_to_axis_angle(R):
    """
    Convert a 3x3 rotation matrix to axis-angle representation
    
    Args:
        R: 3x3 rotation matrix (numpy array)
    
    Returns:
        (axis, angle) where:
        - axis is a unit vector (numpy array)
        - angle is in radians
    """
    # Ensure the matrix is a valid rotation matrix
    assert R.shape == (3, 3), "Input must be a 3x3 matrix"
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
        raise ValueError("Matrix is not a valid rotation matrix")
    
    # TODO: Compute the rotation angle
    angle = 0
    
    # Handle special cases
    if abs(angle) < 1e-6:
        # No rotation (identity matrix)
        return np.array([1, 0, 0]), 0.0
    elif abs(angle - math.pi) < 1e-6:
        # 180 degree rotation - special handling needed
        # Axis is the normalized vector from non-diagonal elements
        axis = np.array([
            math.sqrt((R[0, 0] + 1)/2),
            math.sqrt((R[1, 1] + 1)/2),
            math.sqrt((R[2, 2] + 1)/2)
        ])
        # Determine signs of axis components
        if R[0, 2] - R[2, 0] < 0:
            axis[1] = -axis[1]
        if R[1, 0] - R[0, 1] < 0:
            axis[2] = -axis[2]
        if R[2, 1] - R[1, 2] < 0:
            axis[0] = -axis[0]
    else:
        # General case
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * math.sin(angle))
    
    # Normalize the axis (should already be unit vector, but just in case)
    axis = axis / np.linalg.norm(axis)
    
    return axis, angle

def transformation_matrix(rotation_matrix, translation_vector):
    """
    Create a 4x4 homogeneous transformation matrix from a 3x3 rotation matrix and a 3x1 translation vector.
    
    Parameters:
    rotation_matrix (np.array): 3x3 rotation matrix.
    translation_vector (np.array): 3x1 translation vector.
    
    Returns:
    np.array: 4x4 homogeneous transformation matrix.
    """
    # Ensure the inputs are numpy arrays
    rotation_matrix = np.array(rotation_matrix)
    translation_vector = np.array(translation_vector)
    
    # Create the transformation matrix
    transformation_matrix = np.eye(4)  # Start with a 4x4 identity matrix
    transformation_matrix[:3, :3] = rotation_matrix  # Set the top-left 3x3 block to the rotation matrix
    transformation_matrix[:3, 3] = translation_vector  # Set the top-right 3x1 block to the translation vector
    
    return transformation_matrix

def forward_kinematics_franka(joint_angles):
    """
    Compute the forward kinematics for franka arm (panda_ee frame) using transformation matrices.
    
    Parameters:
    joint_angles (list): List of joint angles in radians [theta1, ..., theta7].
    
    Returns:
    np.array: transformation matrix of the end-effector.
    """
    
    # Rotation matrices for each joint
    theta1, theta2, theta3, theta4, theta5, theta6, theta7 = joint_angles
    R01 = rotation_matrix_z(theta1)  # First joint rotates around z-axis
    R12 = rotation_matrix_x(-math.pi/2) @ rotation_matrix_z(theta2)
    R23 = rotation_matrix_x(math.pi/2) @ rotation_matrix_z(theta3)
    R34 = rotation_matrix_x(math.pi/2) @ rotation_matrix_z(theta4)
    R45 = rotation_matrix_x(-math.pi/2) @ rotation_matrix_z(theta5)
    R56 = rotation_matrix_x(math.pi/2) @ rotation_matrix_z(theta6)
    R67 = rotation_matrix_x(math.pi/2) @ rotation_matrix_z(theta7)
    R78 = np.eye(3)
    R8ee = rotation_matrix_z(-0.785)
    
    # Translation vectors for each link
    p01 = np.array([0, 0, 0.333])  # First link translates along z-axis
    p12 = np.array([0, 0, 0])
    p23 = np.array([0, -0.316, 0])
    p34 = np.array([0.0825, 0, 0])
    p45 = np.array([-0.0825, 0.384, 0])
    p56 = np.array([0, 0, 0])
    p67 = np.array([0.088, 0, 0])
    p78 = np.array([0, 0, 0.107])
    p8ee = np.array([0, 0, 0.103])

    # Transformation matrix for each link
    T01 = transformation_matrix(R01, p01)
    T12 = transformation_matrix(R12, p12)
    T23 = transformation_matrix(R23, p23)
    T34 = transformation_matrix(R34, p34)
    T45 = transformation_matrix(R45, p45)
    T56 = transformation_matrix(R56, p56)
    T67 = transformation_matrix(R67, p67)
    T78 = transformation_matrix(R78, p78)
    T8ee = transformation_matrix(R8ee, p8ee)
    
    # Compute the end-effector position
    ee_T = T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ T67 @ T78 @ T8ee
    
    return ee_T


def error(current_T, ref_T):
    """
    Compute the difference between two homogeneous transformation matrices.
    
    Parameters:
    current_T (np.array): 4 x 4 transformation matrix.
    ref_T (np.array): 4 x 4 transformation matrix.
    
    Returns:
    np.array: 6x1 error term, first three term for rotational error, last three term for translational error.
    """
    axis, angle = rotation_matrix_to_axis_angle(np.transpose(current_T[:3, :3])@ref_T[:3, :3])
    rot_error = axis * angle
    vec_error = current_T[:3, 3] - ref_T[:3, 3] 
    return np.concatenate((rot_error, vec_error))


def jacobian_fd(angles):
    """
    Compute the Jacobian matrix numerically using finite difference
    Parameters:
    angles (list): joint angles
    
    Returns:
    np.array: 6xlen(angles) jacobian matrix
    """
    n_joints = len(angles)
    epsilon = 0.001
    J = np.zeros((6, n_joints))

    # TODO: compute jacobian using finite difference. You can compare the results from jacobian_fd and jacobian. 

    return J

def jacobian(angles):
    """
    Compute the Jacobian matrix
    Parameters:
    angles (list): joint angles
    
    Returns:
    np.array: 6xlen(angles) jacobian matrix
    """
    n_joints = len(angles)
    J = np.zeros((6, n_joints))

    # TODO: compute Jacobian using the formula for revolute joints. 

    return J

def shift_angle(angles):
    # shift angles to between 0 and 2*pi
    return np.mod(angles + np.pi, 2*np.pi) - np.pi

def inverse_kinematics_franka(ref_T):
    """
    Compute IK using Gauss-Newton Algorithm
    Parameters:
    ref_T (np.array): 4 x 4 reference transformation matrix.
    
    Returns:
    np.array: 7 x 1 joint angles
    """

    # IK Gauss-Newton Algorithm
    angles = np.array([0.0] * 7)  # you can change this part to a random guess. 
    max_iter = 100
    tol = 0.01
    
    for iter in range(max_iter):
        current_T = forward_kinematics_franka(angles)
        # TODO: compute the error between the current transformation matrix and the ref_T.
        cur_error = np.zeros((6, 1))

        if np.linalg.norm(cur_error) < tol:
            print(f"Converged after {iter} iterations")
            return shift_angle(angles)
        
        J = jacobian_fd(angles)  # compute jacobian with jacobian_fd or jacobian depending on which one you choose to complete
        
        # TODO: Update joint angles
        
    
    print("Warning: Did not converge to desired tolerance.")

    return shift_angle(angles)


class IKNode(Node):

    def __init__(self):
        super().__init__('ik_calculation_node')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Call on_timer function every second
        self.timer = self.create_timer(1.0, self.on_timer)

    def on_timer(self):
        try:
            # Get the transform
            transform = self.tf_buffer.lookup_transform(
                "panda_link0", "panda_ee", rclpy.time.Time())
            
            # Extract translation and rotation
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            
            # Convert to transformation matrix (using numpy)
            self.get_logger().info(f"Translation: [{translation.x}, {translation.y}, {translation.z}]")
            self.get_logger().info(f"Rotation (quaternion): [{rotation.x}, {rotation.y}, {rotation.z}, {rotation.w}]")
            
            matrix = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
            matrix[0:3, 3] = [translation.x, translation.y, translation.z]

            self.get_logger().info(f'EE Transformation Matrix:: \n {np.round(matrix, 3)}')

            # call franka IK function
            angles = inverse_kinematics_franka(matrix)
            self.get_logger().info(f'Joints angles:: \n {np.round(angles, 3)}')

            # validate with FK function
            ee_FK = forward_kinematics_franka(angles)
            self.get_logger().info(f'EE Transformation Matrix from FK:: \n {np.round(ee_FK, 3)}')

            
        except tf2_ros.LookupException as e:
            self.get_logger().error(f"Lookup failed: {e}")
        except tf2_ros.ConnectivityException as e:
            self.get_logger().error(f"Connectivity issue: {e}")
        except tf2_ros.ExtrapolationException as e:
            self.get_logger().error(f"Extrapolation issue: {e}")


def main(args=None):
    rclpy.init(args=args)
    calculation_node = IKNode()
    rclpy.spin(calculation_node)
    calculation_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
