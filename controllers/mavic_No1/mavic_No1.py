import sys
import os
# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

# Now you can import mavic2pro
import controllers.mavic2pro.mavic2pro as mavic2pro


# Main execution
robot = mavic2pro.Mavic()
robot.set_id(0)
robot.run()