import pybullet as p

# Create the PyBullet client
client = p.connect(p.GUI)

# Load the robot into the simulation
robot = p.loadURDF("C:/Users/bouff/RoboticsProject/panda-gym/mesh/doosan-robot2/dsr_description2/urdf/a0509.blue_gripper.urdf", [0, 0, 0])
print(robot)

# Get the indices of the joints that control the end effector
joint_indices = [p.getNumJoints(robot) - 1]

# Set the joint motor control mode to position control
control_mode = p.POSITION_CONTROL

# Set the target positions of the joints to the desired end effector coordinate
target_positions = [0,0,0]

# Set the joint motor control array
p.setJointMotorControlArray(robot, joint_indices, control_mode, targetPositions=target_positions)
