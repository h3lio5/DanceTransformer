import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

joint_name = [
                "pelvis",
                "left_hip",
                "left_knee",
                "left_ankle",
                "left_foot",
                "right_hip",
                "right_knee",
                "right_ankle",
                "right_foot",
                "spine",
                "bottom_neck",
                "bottom_head",
                "head_top",
                "left_shoulder",
                "left_elbow",
                "left_waist",
                "left_hand",
                "right_shoulder",
                "right_elbow",
                "right_wrist",
                "right_hand"
            ]

PARENT_LIMBS = {
        "PARENT_LIMBS" : [16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13],
             "indcies" : [0,1,2,3,4,6,7,8,9,11,13,15,16,17,18,20,21,24,25,27,28]
        }

connections = [(0,1), (0,1),(0,5), (1,2), (0,9),(2,3), (3,4), (5,6), (6,7), (7,8), (9,10), (10,11),(11,12), (10,17),(10,13), (13,14), (14,15), (15,16),(10,17), (17,18), (18,19), (19,20)]

class Visualizer:
    def __init__(self, exp_name, connections=connections):
        self.connections = connections
        self.save_path = os.path.join('visualizations', exp_name)
        self.frames_path = os.path.join(self.save_path, 'frames')
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        os.mkdir(self.save_path)
        os.mkdir(self.frames_path)

    def generate_and_save_avi(self, frames, framerate):
        framecount = frames.shape[0]
        fig = plt.figure()
        i=0
        for frame in frames:
            bones = []
            for jointA, jointB in self.connections:
                bone = ([frame[jointA][0], frame[jointB][0]], [frame[jointA][1], frame[jointB][1]], [frame[jointA][2], frame[jointB][2]])
                bones.append(bone)
            
            ax = Axes3D(fig)
            ax.set_xlim3d(-50, 10)
            ax.set_ylim3d(-20, 40)
            ax.set_zlim3d(-20, 40)

            plt.plot(frame[:, 2], frame[:, 0], frame[:, 1], "r.")
            for bone in bones:
                plt.plot(bone[2], bone[0], bone[1])
            #plt.show()
            fig.savefig(os.path.join(self.save_path, 'frames', f"fig_{i:05d}.png"))
            i+=1
            fig.clf()
        
        video_name = 'animation.avi'
        images = [img for img in os.listdir(self.frames_path) if img.endswith(".png")]

        frame = cv2.imread(os.path.join(self.frames_path, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(os.path.join(self.save_path, video_name), 0, 60, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(self.frames_path, image)))

        cv2.destroyAllWindows()
        video.release()
