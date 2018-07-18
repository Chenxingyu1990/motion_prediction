from PIL import Image
import sys
import cv2
import ipdb
import imageio
import os
import os.path
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize


def get_images(path, num):
    frames = []
    for i in range(num):
        filename = '{}/{}.png'.format( path, i)
        frames.append(filename)
    return frames
def make_video(images, outvid=None, fps=25, size=None,
               is_color=True, format="XVID"):
    """
    Create a video from a list of images.
 
    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
 
    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid

def main():
    video_name = 'case1.avi'
    path = '/home/xingyu/EXP/case1'   
   
    images = get_images(path, 50)
    make_video(images, video_name)

if __name__ == "__main__":
    main()
    
