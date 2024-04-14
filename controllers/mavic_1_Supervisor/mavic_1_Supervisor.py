import sys
import os
sys.path.append(os.path.abspath(r"C:\Users\MikeTheMater\Desktop\Landing_Site_Detection\controllers\mavic_Supervisor"))
import mavic_Supervisor

# Main execution
robot = mavic_Supervisor.SuperMavic("Mavic_2_PRO")
robot.run()