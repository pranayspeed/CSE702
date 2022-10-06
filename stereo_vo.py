
import numpy as np

import cv2

from scipy.optimize import least_squares

def poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret 

def get_grayscale_img(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def load_calib(filepath):
    print(filepath)
    with open(filepath, 'r') as f:
        f.readline()
        f.readline()
        str_val = f.readline().split(":")[1]
        params = np.fromstring(str_val, dtype=np.float64, sep=' ')
        print(params)
        P_l = np.reshape(params, (3, 4))
        K_l = P_l[0:3, 0:3]
        str_val = f.readline().split(":")[1]
        params = np.fromstring(str_val, dtype=np.float64, sep=' ')
        print(params)
        P_r = np.reshape(params, (3, 4))
        K_r = P_r[0:3, 0:3]
    return K_l, P_l, K_r, P_r

class Visual_Odometry_Stereo:
    def __init__(self, calib_file, min_num_feat, focal=1, pp=(0.,0.)):
        self.rotation= np.identity(3)
        self.translation = np.zeros((3,1))
        self.min_num_feat = min_num_feat
        self.focal = focal
        self.pp = pp
        self.feature_detector = cv2.FastFeatureDetector_create()

        block=11
        P1 = block* block * 8
        P2 = block* block * 32
        self.stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=block, P1=P1, P2=P2)
        
        self.K_l, self.P_l, self.K_r, self.P_r = load_calib(calib_file)

        self.lk_params = dict(winSize=(15, 15),
                              flags=cv2.MOTION_AFFINE,
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

        self.Q = np.identity(4)
        self.Q

    def compute_disparity(self, img_l_gray, img_r_gray):
        disparity = self.stereo.compute( img_l_gray, img_r_gray)
        return np.divide(disparity.astype(np.float32), 16)

    def extract_features(self, curr_image):
        # find and draw the keypoints
        return self.feature_detector.detect(curr_image,None)


    def track_keypoints(self, img1, img2, kp1, max_error=4):
        # Convert the keypoints into a vector of points and expand the dims so we can select the good ones
        trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)

        # Use optical flow to find tracked counterparts
        trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **self.lk_params)

        # Convert the status vector to boolean so we can use it as a mask
        trackable = st.astype(bool)

        # Create a maks there selects the keypoints there was trackable and under the max error
        under_thresh = np.where(err[trackable] < max_error, True, False)

        # Use the mask to select the keypoints
        trackpoints1 = trackpoints1[trackable][under_thresh]
        trackpoints2 = np.around(trackpoints2[trackable][under_thresh])

        # Remove the keypoints there is outside the image
        h, w = img1.shape
        in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
        trackpoints1 = trackpoints1[in_bounds]
        trackpoints2 = trackpoints2[in_bounds]

        return trackpoints1, trackpoints2


    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
        def get_idxs(q, disp):
            q_idx = q.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)
        
        # Get the disparity's for the feature points and mask for min_disp & max_disp
        disp1, mask1 = get_idxs(q1, disp1)
        disp2, mask2 = get_idxs(q2, disp2)
        
        # Combine the masks 
        in_bounds = np.logical_and(mask1, mask2)
        
        # Get the feature points and disparity's there was in bounds
        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]
        
        # Calculate the right feature points 
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2
        
        return q1_l, q1_r, q2_l, q2_r



    def calc_3d(self, q1_l, q1_r, q2_l, q2_r, depth_threshold=50):
         # Triangulate points from i-1'th image
        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)
        # Un-homogenize
        Q1 = np.transpose(Q1[:3] / Q1[3])

        # Triangulate points from i'th image
        Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)
        # Un-homogenize
        Q2 = np.transpose(Q2[:3] / Q2[3])

        print(Q1.shape, Q2.shape)

        # # filter points greater than depth_threshold 
        mask1 = np.where(np.logical_or(np.logical_or(Q1[:,0] < depth_threshold, Q1[:,1] < depth_threshold), Q1[:,2] < depth_threshold ), True, False)
        mask2 = np.where(np.logical_or(np.logical_or(Q2[:,0] < depth_threshold, Q2[:,1] < depth_threshold), Q2[:,2] < depth_threshold ), True, False)
        #print(mask1)
        #print(mask2)
        in_bounds = np.logical_and(mask1, mask2)
        Q1, Q2 = Q1[in_bounds], Q2[in_bounds]


        print("-> ",Q1.shape, Q2.shape)
        return Q1, Q2, in_bounds                      



    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=1500):
        early_termination_threshold = 5

        # Initialize the min_error and early_termination counter
        min_error = float('inf')
        early_termination = 0

        for _ in range(max_iter):
            # Choose 6 random feature points
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            # Make the start guess
            in_guess = np.zeros(6)
            # Perform least squares optimization
            opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                    args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

            # Calculate the error for the optimized transformation
            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))

            # Check if the error is less the the current min error. Save the result if it is
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_threshold:
                # If we have not fund any better result in early_termination_threshold iterations
                break

        # Get the rotation vector
        r = out_pose[:3]
        # Make the rotation matrix
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = out_pose[3:]
        # Make the transformation matrix
        transformation_matrix = poseRt(R, t)

        print("Min error: ", min_error)
        return transformation_matrix

    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        # Get the rotation vector
        r = dof[:3]
        # Create the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = dof[3:]
        # Create the transformation matrix from the rotation matrix and translation vector
        transf = poseRt(R, t)

        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = np.matmul(self.P_l, transf)
        b_projection = np.matmul(self.P_l, np.linalg.inv(transf))

        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        # Project 3D points from i'th image to i-1'th image
        q1_pred = Q2.dot(f_projection.T)
        # Un-homogenize
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        # Project 3D points from i-1'th image to i'th image
        q2_pred = Q1.dot(b_projection.T)
        # Un-homogenize
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        # Calculate the residuals
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        return residuals

    def track(self, I1_l, I1_r, I2_l, I2_r):
        I1_l_grayscale = get_grayscale_img(I1_l)
        I2_l_grayscale = get_grayscale_img(I2_l)
        I1_r_grayscale = get_grayscale_img(I1_r)
        I2_r_grayscale = get_grayscale_img(I2_r)

        I1_disparity = self.compute_disparity(I1_l_grayscale, I1_r_grayscale)
        I2_disparity = self.compute_disparity(I2_l_grayscale, I2_r_grayscale)

        I1_l_kps = self.extract_features(I2_l_grayscale)

        I1_pts, I2_pts = self.track_keypoints(I1_l_grayscale, I2_l_grayscale, I1_l_kps)

        # Calculate the right keypoints
        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(I1_pts, I2_pts, I1_disparity, I2_disparity)

        # Calculate the 3D points
        Q1, Q2, inbound = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)

        tp1_l, tp2_l = tp1_l[inbound], tp2_l[inbound]

        # Estimate the transformation matrix
        transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2)
        return transformation_matrix



def display_traj(traj, curr_pose, frame):
    #print(curr_pose)
    x_trans = 800
    y_trans = 400
    curr_2d_pos = (int(curr_pose[0, 3])+x_trans,y_trans - int(curr_pose[2, 3]))
    #print(curr_2d_pos)
    traj = cv2.circle(traj,curr_2d_pos,2,(0, 0,255), 1)

    print(frame.dtype, traj.dtype, frame.shape, traj.shape)
    frame_new = cv2.vconcat([frame, traj.astype(np.uint8)])
    cv2.imshow("traj", frame_new)
    cv2.waitKey(1)

def main():
    
    calib_file = "/Users/pranayspeed/Work/git_repos_other/00/calib.txt"
    images_path = "/Users/pranayspeed/Work/git_repos_other/00"
    images_left = images_path +"/image_2"
    images_right = images_path +"/image_3"

    vo = Visual_Odometry_Stereo(calib_file, 500)


    traj = np.zeros((600,1241,3))
    estimated_path =[]
    curr_pose = np.identity(4)


    absolute_transf = np.identity(4)
    for i in range(2000):
        curr_image = str(i).zfill(6)+".png"
        images_left_i = images_left+"/"+curr_image
        images_right_i = images_right+"/"+curr_image

        I1_l = cv2.imread(images_left_i)#, cv2.COLOR_BGR2GRAY)
        I1_r = cv2.imread(images_right_i)#, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("image_left", I1_l)
        #cv2.imshow("image-right", I1_r)

        print(I1_l.shape)
        if i ==0:
            I2_l= I1_l
            I2_r= I1_r
            continue 
        
        transf = vo.track(I2_l, I2_r, I1_l, I1_r)
        curr_pose = np.matmul(curr_pose, transf)

        absolute_transf = transf @absolute_transf 
        estimated_path.append((curr_pose[0, 3], curr_pose[2, 3]))

        display_traj(traj, curr_pose, I1_l)


        I2_l= I1_l
        I2_r= I1_r


if __name__ == "__main__":
    main()
