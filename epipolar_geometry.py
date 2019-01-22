import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

img1 = cv2.imread("data/tsucuba_left.png", 0)
img2 = cv2.imread("data/tsucuba_right.png", 0)

sift = cv2.xfeatures2d.SIFT_create()

(kps1, descs1) = sift.detectAndCompute(img1, None)
(kps2, descs2) = sift.detectAndCompute(img2, None)

sift_img1 = cv2.drawKeypoints(img1, kps1, color=(0,255,0), outImage=np.array([]))
sift_img2 = cv2.drawKeypoints(img2, kps2, color=(0,255,0), outImage=np.array([]))

print("# kps: {}, descriptors: {}".format(len(kps1), descs1.shape))
print("# kps: {}, descriptors: {}".format(len(kps2), descs2.shape))

cv2.imwrite("Result/task2_sift1.jpg", sift_img1)
cv2.imwrite("Result/task2_sift2.jpg", sift_img2)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)


flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descs1,descs2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)

src_pts = np.float32([ kps1[m.queryIdx].pt for m in good ])
dst_pts = np.float32([ kps2[m.trainIdx].pt for m in good ])

F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()


draw_params = dict(matchColor = (0,200,100), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask,
                   flags = 2)

img3 = cv2.drawMatches(img1,kps1,img2,kps2,good,None,**draw_params)
cv2.imwrite("Result/task2_matches_knn.jpg", img3)


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(descs1,descs2,k=2)

print("The Fundamental Matrix is:")
print(F)

stereoMatcher = cv2.StereoBM_create()
stereoMatcher.setMinDisparity(16)
stereoMatcher.setNumDisparities(112)
stereoMatcher.setBlockSize(17)

stereoMatcher.setSpeckleRange(32)
stereoMatcher.setSpeckleWindowSize(100)


stereo = stereoMatcher.compute(img1, img2)

cv2.imwrite("Result/disparity.jpg", stereo)
plt.imshow(stereo)
plt.imsave("Result/task2_disparity.png",stereo)

src_pts = np.int32(src_pts)
dst_pts = np.int32(dst_pts)

pts1 = src_pts
pts2 = dst_pts

pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

list_inliers = []
for i in range(0,11):
    ran_number = random.randint(0,271)
    list_inliers.append(ran_number)

lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)

new_lines = np.copy(lines1)

count=0
for i in list_inliers:
    new_lines[count,:] = lines1[i,:]
new_lines1 = np.copy(new_lines[0:10])


img5,img6 = drawlines(img1,img2,new_lines1,pts1,pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)


new_lines_1 = np.copy(lines2)
count=0
for i in list_inliers:
    new_lines_1[count,:] = lines2[i,:]
new_lines_1_1 = np.copy(new_lines_1[0:10])

img3,img4 = drawlines(img2,img1,new_lines_1_1,pts2,pts1)
cv2.imwrite("Result/task2_epi_left.jpg", img5)
cv2.imwrite("Result/task2_epi_right.jpg", img3)
