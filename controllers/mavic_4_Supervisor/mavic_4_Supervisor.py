import sys
import os
# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

# Now you can import mavic_Supervisor
import controllers.mavic_Supervisor.mavic_Supervisor as mavic_Supervisor

# Main execution
robot = mavic_Supervisor.SuperMavic("Mavic_2_PRO_4")
robot.run()