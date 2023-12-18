import cv2
import os

def generate_and_save_aruco_marker(aruco_dict, marker_id):
    marker_size = 400

    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

    folder_name = "generated_markers"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    marker_path = os.path.join(folder_name, "marker_{}.png".format(marker_id))
    cv2.imwrite(marker_path, marker_img)

    marker_img = cv2.imread(marker_path)

    cv2.imshow("Marker", marker_img)

    print("Dimensions:", marker_img.shape)

    cv2.waitKey(0)

if __name__ == "__main__":
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Generate and save ArUco markers with IDs 0 to 9 and size 200x200 pixels
    for id in range(10):
        generate_and_save_aruco_marker(aruco_dict, id)
        

    print("ArUco markers generated and saved in the 'generated_markers' folder.")
