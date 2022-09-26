
import cv2
import numpy as np


# function [R, T, cliqueSize, outliers, resnorm] = visodo(I1_l, I1_r, I2_l, I2_r, P1, P2)
# %% The core visual odometry function
# %INPUT:
# % I1_l -> a grayscale image, from camera 1 (left) at some time t
# % I1_r -> a grayscale image, from camera 2 (right) at some time t
# % I2_l -> a grayscale image, from camera 1 (left) at time t+1
# % I2_r -> a grayscale image, from camera 2 (right) at time t+1
# % P1 -> the [3x4] projection matrix of camera 1 (left)
# % P2 -> the [3x4] projection matrix of camera 2 (right)
# %
# %OUTPUT:
# % R-> The rotation matrix [3x3] describing the change in orientation from t to
# % t+1
# % T-> The tranlsation vector [3x1] describing the change in cartesian
# % coordinates of the vehicle from t to t+1
# % cliqueSize -> the size of the cliques used in the Levenberg-Marquardt 
# % optimization (checkout the documentation/blog for more info)
# % 



def visodo(I1_l, I1_r, I2_l, I2_r, P1, P2):
    
    
    I1_l_gray = cv2.cvtColor(I1_l, cv2.COLOR_BGR2GRAY)
    I1_r_gray = cv2.cvtColor(I1_r, cv2.COLOR_BGR2GRAY)
    I2_l_gray = cv2.cvtColor(I2_l, cv2.COLOR_BGR2GRAY)
    I2_r_gray = cv2.cvtColor(I2_r, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoBM_create()
    disparity_1 = stereo.compute(I1_l_gray, I1_r_gray)
    disparity_2 = stereo.compute(I2_l_gray, I2_r_gray)

    disparity_1 = disparity_1.astype(np.float32)
    disparity_2 = disparity_2.astype(np.float32)

    # Scaling down the disparity values and normalizing them
    #disparity = (disparity/16.0 - minDisparity)/numDisparities

    # Displaying the disparity map

    #cv2.imshow("disp1",disparity_1)
    #cv2.waitKey(1)
    #cv2.imshow("disp2",disparity_2)
    #cv2.waitKey(1)


    fast = cv2.FastFeatureDetector_create()
    # find and draw the keypoints
    I1_l_kp = fast.detect(I1_l_gray,None)
    I2_l_kp = fast.detect(I2_l_gray,None)    
    feature_params = dict( maxCorners = 500,   # How many pts. to locate
                       qualityLevel = 0.1,  # b/w 0 & 1, min. quality below which everyone is rejected
                       minDistance = 7,   # Min eucledian distance b/w corners detected
                       blockSize = 3 ) # Size of an average block for computing a derivative covariation matrix over each pixel neighborhood


    I1_l_pts = cv2.goodFeaturesToTrack(I1_l_gray, mask=None, **feature_params)
    # calculate optical flow
    I2_l_pts, st, err = cv2.calcOpticalFlowPyrLK(I1_l_gray, I2_l_gray, I1_l_pts, None)

    # Select good points
    good_new = I2_l_pts[st==1]
    good_old = I1_l_pts[st==1]

    mask = np.zeros_like(I2_l)
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel() #tmp new value
        c,d = old.ravel() #tmp old value
        #draws a line connecting the old point with the new point
        print(a,b,c,d)
        mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), (0,255,0), 1)
        #draws the new point
        I2_l_tracked = cv2.circle(I2_l,(int(a),int(b)),2,(0,0,255), -1)
    I2_l_tracked = cv2.add(I2_l_tracked,mask)
    cv2.imshow("flow tracked",I2_l_tracked)
    cv2.waitKey(1)


    #I1_l_kp = cv2.drawKeypoints(I1_l, kp, None, color=(255,0,0))
    #cv2.imshow("Features detected",I1_l_kp)
    #cv2.waitKey(1)
    #cv2.imshow("disp2",disparity_2)
    #cv2.waitKey(1)
    # Close window using esc key


def main():
    
    images_path = "/Users/pranayspeed/Work/git_repos_other/00"
    images_left = images_path +"/image_2"
    images_right = images_path +"/image_3"
    for i in range(500):
        curr_image = str(i).zfill(6)+".png"
        images_left_i = images_left+"/"+curr_image
        images_right_i = images_right+"/"+curr_image
        print(images_left_i)
        I1_l = cv2.imread(images_left_i)#, cv2.COLOR_BGR2GRAY)
        I1_r = cv2.imread(images_right_i)#, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("image_left", I1_l)
        #cv2.imshow("image-right", I1_r)
        if i ==0:
            I2_l= I1_l
            I2_r= I1_r
            continue 

        P1=[]
        P2 = []
        visodo(I1_l, I1_r, I2_l, I2_r, P1, P2)
        I2_l= I1_l
        I2_r= I1_r

if __name__ == "__main__":
    main()
