import cv2
import math
import cv2.cv as cv
from matplotlib import pyplot as plt
from operator import itemgetter, attrgetter
from pylab import array, plot, show, axis, arange, figure, uint8 
import numpy as np
from scipy import stats
from MachineLearning import *
img_path = '../../../images/'

image_file_L1 = '2014-04-05 15.04.33.jpg'
image_file_L2 = '2014-04-05 15.05.15.jpg'
image_file_L3 = '2014-04-05 15.05.22.jpg'
image_file_S1 = '2014-04-05 15.06.01.jpg'
image_file_S2 = '2014-04-05 15.06.09.jpg'
image_file_B1 = '2014-04-05 15.04.59.jpg'
image_file_B2 = '2014-04-05 15.05.09.jpg'



def main():
    #ch = 0
    original_img = cv2.imread(img_path + image_file_L2, 0)
    #sf = 2;
    #height, width = img.shape
    #small_img = cv.CreateImage((int(height/sf), int(width/sf)), 0, ch)
    #cv2.Resize(img, small_img, interpolation = cv.CV_INTER_CUBIC)

    #img = cv2.medianBlur(img,5)
    img = cv2.blur(original_img,(5,5))
    imgc = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    

    
    # Find the edges of an image
    edges = cv2.Canny(img,50,150)
    #plt.subplot(121),plt.imshow(img,cmap = 'gray')
    #plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(123),plt.imshow(original_img,cmap = 'gray')
    #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #plt.show()

    #return



    maxIntensity = 255.0 # depends on dtype of image data
    x = arange(maxIntensity) 

    # Parameters for manipulating image data
    phi = 2
    theta = 2

    # Increase intensity such that
    # dark pixels become much brighter, 
    # bright pixels become slightly bright
    newImage0 = (maxIntensity/phi)*(img/(maxIntensity/theta))**0.5
    newImage0 = array(newImage0,dtype=uint8)

    y = (maxIntensity/phi)*(x/(maxIntensity/theta))**0.5

    # Decrease intensity such that
    # dark pixels become much darker, 
    # bright pixels become slightly dark 
    newImage1 = (maxIntensity/phi)*(img/(maxIntensity/theta))**2
    newImage1 = array(newImage1,dtype=uint8)

    #figure()
    #plt.imshow(img)
    #figure()
    #plt.imshow(newImage1)
    #figure()
    #plt.imshow(newImage0)
    #plt.show()
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    original_images = newImage0, newImage1, img, 
    all_circle_images = []
    for image in original_images:
        all_circle_images.append(image.copy())

    circles_in_image = []

    #find circles
    for image in all_circle_images:
        max_r = 80
        circles = cv2.HoughCircles(image,cv2.cv.CV_HOUGH_GRADIENT,1,max_r*1.5,param1=50,param2=25,minRadius=50,maxRadius=max_r)
        if(circles != None):
            circles = np.uint16(np.around(circles))
            circles_in_image.append(circles)

            
    final_circles = []
    #combine circles
    CombineCircles(circles_in_image, final_circles)

                
    lines = []
    # find which circles line up
    FindLinesFromCircles(final_circles, lines)
    lines = SortLines(lines)
    
    # find if any lines intersect at 90°
    # put a circle there if there isn't one already
    # add it to the two lines that are intersecting at it
    # fill in circles are are missing between it and the line at average intervals





    #cv2.waitKey(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = original_images[0]
    for i, circle in enumerate(final_circles):
        # draw the outer circle
        cv2.circle(image,(circle[0],circle[1]),circle[2],(0,255,0),2)
        # draw the center of the circle
        j = i+1
        cv2.circle(image,(circle[0],circle[1]),2,(0,0,255),3)
        cv2.putText(image,str(circle),(circle[0]+100,circle[1]+100),font,2.0,(255,255,255),3)
        cv2.line(image,(circle[0]+100,circle[1]+100),(circle[0],circle[1]),(255,255,255),3)
        
    for line in lines:
        pt_max = (line[0][len(line[1])-1],line[1][len(line[1])-1])
        pt_min = (line[0][0],line[1][0])
        color = (0,255,255,0)
        cv2.line(image,pt_max, pt_min, (255,255,255),5)

    figure()
    plt.imshow(image)


    #cv2.destroyAllWindows()

    
    print "Number of lines =", len(lines)
    print "Number of circles =", len(final_circles)
    plt.show()
    return




def CircleCompare(circle1, circle2):
    d_th = 55
    r_th = 30
    i_coords = np.array((circle1[0],circle1[1]),dtype=long)
    j_coords = np.array((circle2[0],circle2[1]),dtype=long)
    i_r = np.long(circle1[2])
    j_r = np.long(circle2[2])
    delta_r = np.linalg.norm(i_r-j_r)
    dist = np.linalg.norm(i_coords-j_coords)
    if((dist < d_th) and delta_r < r_th):
        return 1
    return 0

def Normalize(pt1, pt2):
    a = np.array(pt1-pt2, dtype = np.float)
    b = np.float(np.linalg.norm(pt1-pt2))
    c = 0
    if(b!=0):
        c = np.divide(a,b) 
    return c

def LineEq(x, y):
    x_prime = x
    y_prime = y
    if(x_prime[0] == x_prime[1]):
        x_prime[0] = x_prime[0]+1
    if(y_prime[0] == y_prime[1]):
        y_prime[0] = y_prime[0]+1
    A = np.vstack([x_prime, np.ones(len(x_prime))]).T
    m, c = np.linalg.lstsq(A, y_prime)[0]
    #vars = stats.linregress(x,y)
    #m = vars[0]
    #c = vars[1]
    #stderr = vars[2]
    return m, c #, stderr

def DistanceFromPtToLine(m, c, pt):
    cos_theta = np.dot(Normalize(np.array([0,c]), np.array([1,(m+c)])), Normalize(np.array([0,c]), pt))
    dist = np.linalg.norm(pt-np.array([0,c]))*math.sin(math.acos(cos_theta))
    return dist

def Consolidate(lines):
    changes = False
    master_list =lines
    while(changes == False):
        changes, master_list = CheckAndMerge(master_list)
    return master_list

def CheckAndMerge(lines):
    for n in range(len(lines)):
        for m in range(n+1,len(lines)):
            if(PointsMatch(lines[n], lines[m])):
                temp = Merge([lines[n], lines[m]])
                lines.remove(lines[m])
                lines.remove(lines[n])
                lines.append(temp)
                return False, lines
    return True, lines

def PointsMatch(line, target_line):
    hit_count = 0
    for n in range(len(target_line[0])):
        pt = ([target_line[0][n],target_line[1][n]])
        if((pt[0] in line[0]) and (pt[1] in line[1])):
            hit_count = hit_count + 1
        if (hit_count > 1):
            return True
    return False

def DistBetweenPts(pt1, pt2):
    dif = [long(pt1[0]) - long(pt2[0]), long(pt1[1]) - long(pt2[1])]
    return math.sqrt(dif[0]*dif[0]+dif[1]*dif[1])

def Merge(lines_to_merge):
    master_list = [[],[]]
    for line in lines_to_merge:
        for n in range(len(line[0])):
            pt = ([line[0][n],line[1][n]])
            if((pt[0] not in master_list[0]) or (pt[1] not in master_list[1])):
                master_list[0].append(pt[0])
                master_list[1].append(pt[1])
    return master_list

def CombineCircles(circles_in_image, final_circles):
    #assume no circles within an image overlap
    previously_used = []
    #for each set of circles in an image
    for src_i in range(len(circles_in_image)):
        src = circles_in_image[src_i][0]
        #for each circle in an image's set of circles
        for i in src[0:]:
            avg_list = [i]
            #for each set of circles in future images
            for dst_j in range(src_i+1, len(circles_in_image)):
                dst = circles_in_image[dst_j][0]
                #for each circle in an image's set of circles
                for j in range(len(dst)):
                    circle = dst[j]
                    res = False
                    if((dst_j, j) in previously_used): res = True
                    if(CircleCompare(i,circle) and not res):
                        avg_list.append(circle)
                        previously_used.append((dst_j, j))
                        break
            if(len(avg_list) > 1):
                sum = np.sum(avg_list,0)
                result = np.divide(sum, len(avg_list))
                final_circles.append(result)


def FindLinesFromCircles(final_circles, lines):
    #line_threshold = 0.005
    line_threshold = 10
    if len(final_circles) > 2:
        count = len(final_circles)
        for c_i, circle in enumerate(final_circles):
            if(c_i+1 < count):
                for start in range(c_i+1, count):
                    x = []
                    y = []
                    x.append(final_circles[c_i][0])
                    y.append(final_circles[c_i][1])
                    x.append(final_circles[start][0])
                    y.append(final_circles[start][1])
                    #m, c, stderr = LineEq(x, y)
                    m, c = LineEq(x, y)
                    for i in range(c_i+2, count):
                        pt = np.array(np.int32([final_circles[i][0], final_circles[i][1]]))
                        if i == start:
                            continue
                        #t_x = list(x)
                        #t_y = list(y)
                        #t_x.append(pt[0])
                        #t_y.append(pt[1])
                        #new_m, new_c, new_stderr = LineEq(t_x, t_y)
                        dist = DistanceFromPtToLine(m, c, pt)
                        if dist < line_threshold:
                       # temp = abs(1-new_stderr)
                        #if temp < line_threshold:
                            x.append(pt[0])
                            y.append(pt[1])
                            #m, c, stderr = LineEq(x, y)
                            m, c, = LineEq(x, y)
                    if(len(x) > 2):
                        lines.append([x, y])
                        lines = Consolidate(lines)

def ConvertLineToPts(line):
    pts = []
    for i in range(len(line[0])):
        pts.append([line[0][i], line[1][i]])
    return pts

def ConvertPtsToLine(pts):
    line = [[],[]]
    for x, y in pts:
        line[0].append(x)
        line[1].append(y)
    return line

def SortLines(lines):
    lenths = []
    sorted_lines = []
    for line in lines:
        pts = ConvertLineToPts(line)
        sorted = []
        max_d = 0
        max_pt = []
        lengths = []
        # measuring from zero could start with a point in the middle of the line
        # starting from the furthest point in the line ensures we follow allong it
        for i, pt in enumerate(pts):
            dist = DistBetweenPts([0,0], pt)
            if(dist > max_d):
                max_d = dist
                max_pt = pt   
        for i, pt in enumerate(pts):
            dist = DistBetweenPts(max_pt, pt)
            lengths.append(dist)
        max_delta = max(lengths)+1
        for i in range(len(lengths)):
            min_i = lengths.index(min(lengths))
            next_pt = pts[min_i]
            lengths[min_i] = max_delta
            sorted.append(next_pt)
        sorted_lines.append(ConvertPtsToLine(sorted))
    return sorted_lines

            


if __name__ == "__main__":
    main()