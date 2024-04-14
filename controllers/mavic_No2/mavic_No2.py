import sys
import os
sys.path.append(os.path.abspath(r"C:\Users\MikeTheMater\Desktop\Landing_Site_Detection\controllers\mavic2pro"))
import mavic2pro

# Main execution
robot = mavic2pro.Mavic()
robot.set_id(0)
robot.run()