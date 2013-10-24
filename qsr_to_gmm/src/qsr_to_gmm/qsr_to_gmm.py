#!/usr/bin/env python

from qsr_msgs.srv import *
from qsr_msgs.msg import *
import rospy
import getopt
import json
import math
from operator import itemgetter
import tf

import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.patches as mpatches
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class BBox():
    """ Bounding box of an object with getter functions.
    """
    def __init__(self, bbox):
        # Calc x_min and x_max for obj1
        x_sorted = sorted(bbox, key=itemgetter(0))
        self.x_min = x_sorted[0][0]
        self.x_max = x_sorted[7][0]

        # Calc y_min and y_max for obj
        y_sorted = sorted(bbox, key=itemgetter(1))
        self.y_min = y_sorted[0][1]
        self.y_max = y_sorted[7][1]

        # Calc z_min and z_max for obj
        z_sorted = sorted(bbox, key=itemgetter(2))
        self.z_min = z_sorted[0][2]
        self.z_max = z_sorted[7][2]
        
    def get_x_min(self):
        return self.x_min

    def get_x_max(self):
        return self.x_max

    def get_y_min(self):
        return self.y_min

    def get_y_max(self):
        return self.y_max

    def get_z_min(self):
        return self.z_min

    def get_z_max(self):
        return self.z_max
    



class QSR2GMM():
    "A class for transforming QSRs in GMMs"

    def __init__(self, qsr_model):
        # Store model
        self.qsr_model = qsr_model
        
        # Start service
        rospy.init_node('qsr_to_gmm_server')
        self.service = rospy.Service('/qsr_to_gmm', QSRToGMM, self.qsr_to_gmm_service)
        rospy.loginfo("Ready to transform QSRs into a GMM.")
        rospy.spin()


    def qsr_to_hist(self,qsr,position):

        qsr_lst = qsr[self.landmark.lower()][self.obj.lower()]

        direction = list()
        distance = list()

        for rel in qsr_lst:
            if rel in ['left','right','front','behind']:
                direction.append(rel)
            elif rel in ['close','distant']:
                distance.append(rel)
            else:
                rospy.logerror("Wrong relation:", rel)
                

        if len(direction) == 1:
            if direction[0] in ['left','right']:
                relation = direction[0] + '_center'
            elif direction[0] in ['front','behind']:
                relation = 'center_' + direction[0] 
            else:
                rospy.logerror("Wrong relation:", direction[0])
        elif len(direction) == 2:
            if direction[0] in ['left','right']:
                relation = direction[0] + '_' + direction[1]
            else:
                relation = direction[1] + '_' + direction[0]
        else:
            rospy.logerror("Wrong direction:", direction)
                

        self.dir_count[relation] = self.dir_count[relation] + 1

        for rel in distance:
            if rel in ['close','distant']:
                self.dist_count[rel] = self.dist_count[rel] + 1

        x_rel = position[self.obj.lower()][0] - position[self.landmark.lower()][0]
        y_rel = position[self.obj.lower()][1] - position[self.landmark.lower()][1]

        x_rel_rotated = x_rel * math.cos(math.pi * 3/2) - y_rel * math.sin(math.pi * 3/2)
        y_rel_rotated = x_rel * math.sin(math.pi * 3/2) + y_rel * math.cos(math.pi * 3/2)
        
        relative_pos = [x_rel_rotated, y_rel_rotated]
        
        if distance[0] == 'close':
            self.dir_close_pos[relation].append(relative_pos)
            self.dir_close_count[relation] = self.dir_close_count[relation] + 1
        else:
            self.dir_distant_pos[relation].append(relative_pos)
            self.dir_distant_count[relation] = self.dir_distant_count[relation] + 1
        

    def qsr_to_gmm_service(self,req):

        # Get arguments from request
        self.obj = req.object
        self.landmark = req.landmark
        self.qsr = req.qsr

        self.dir_count = dict()
        self.dist_count = dict()

        self.dir_close_count = dict()
        self.dir_distant_count = dict()
        
        self.dir_close_pos = dict()
        self.dir_distant_pos = dict()
        
        for rel in ['left_center', 'left_front', 'left_behind',
                    'right_center','right_front', 'right_behind',
                    'center_front','center_behind']:
            self.dir_count[rel] = 0
            self.dir_close_count[rel] = 0
            self.dir_distant_count[rel] = 0
            self.dir_close_pos[rel] = list()
            self.dir_distant_pos[rel] = list()

        for rel in ['close','distant']:
            self.dist_count[rel] = 0

        landmark_bbox = None
        landmark_orientation = None

        count_obj = 0
        count_obj_landmark = 0
        for scn in self.qsr_model:
            scn_types = scn[1]['type'].values()
            if self.obj in scn_types:
                count_obj += 1
            if self.obj in scn_types and self.landmark in scn_types:
                count_obj_landmark += 1
                #rospy.loginfo(scn[0])
                self.qsr_to_hist(scn[1]['qsr'],scn[1]['position'])
                if landmark_bbox == None:
                    landmark_bbox = scn[1]['bbox'][self.landmark.lower()]
                    landmark_orientation = scn[1]['orientation'][self.landmark.lower()]



        x_close = list()
        x_dist =  list()
        y_close = list()
        y_dist =  list()
        for rel in self.dir_close_pos:
            for pos in self.dir_close_pos[rel]:
                x_close.append(pos[0])
                y_close.append(pos[1])
                
        for rel in self.dir_distant_pos:
            for pos in self.dir_distant_pos[rel]:
                x_dist.append(pos[0])
                y_dist.append(pos[1])


        # fig, ax = plt.subplots()
        # ax.plot(x_close, y_close, 'o')
        # ax.plot(x_dist,  y_dist, 'x')
        # ax.set_title(self.obj + ' WRT ' + self.landmark)
        # ci = Circle((0,0), 0.1,facecolor='r', alpha=0.2)
        # ax.add_artist(ci)
        # ci.set_clip_box(ax.bbox)
        # plt.show()

        # fig, ax = plt.subplots()
        # im = ax.hexbin(x_close, y_close, gridsize=20)
        # fig.colorbar(im, ax=ax)
        # plt.show()
        
        weights, gaussians = self.calc_GMM()

        num_of_samples = 1000

        tmp_x = list()
        tmp_y = list()
        for i in range(len(weights)):
            cov = gaussians[i].covariance           
            x,y = np.random.multivariate_normal(gaussians[i].mean,
                                                [[cov[0],cov[1]], [cov[2],cov[3]]],
                                                int(weights[i] * num_of_samples)).T
            tmp_x.append(x)
            tmp_y.append(y)
            
        gmm_x = [val for subl in tmp_x for val in subl]
        gmm_y = [val for subl in tmp_y for val in subl] 

        # hist,xedges,yedges = np.histogram2d(gmm_x,gmm_y,bins=40, range=[[-0.75, 0.75], [-0.75, 0.75]])
        # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1] ]
        # plt.imshow(hist.T,extent=extent,interpolation='nearest',origin='lower')
        # plt.colorbar()
        # plt.show()

        #print("ENTROPY:",self.entropy(gmm_x,gmm_y))

        
        # uniform sampling instead of samples from data set
        gmm_x = np.random.uniform(-2.2,2.2,10000)
        gmm_y = np.random.uniform(-1.4,1.4,10000)
        
        gmm_z = [0] * len(gmm_x)

        
        
        for i in range(len(gmm_z)):
            for j in range(len(weights)):
                cov = gaussians[j].covariance
                gmm_z[i] += weights[j] * matplotlib.mlab.bivariate_normal(gmm_x[i],
                                                                         gmm_y[i],
                                                                         math.sqrt(cov[0]),
                                                                         math.sqrt(cov[3]),
                                                                         gaussians[j].mean[0],
                                                                         gaussians[j].mean[1],
                                                                         cov[1])

        ### CALC ENTROPY
        ex = seq(-2.0, 2.0, 0.05)
        ey = seq(-1.2, 1.2, 0.05)
        exy = list()

        for xx in ex:
            for yy in ey:
                exy.append([xx,yy])
                
        ez = [0.0] * len(exy)
        for i in range(len(ez)): 
            for j in range(len(weights)):
                cov = gaussians[j].covariance
                ez[i] += weights[j] * matplotlib.mlab.bivariate_normal(exy[i][0],
                                                                          exy[i][1],
                                                                          math.sqrt(cov[0]),
                                                                          math.sqrt(cov[3]),
                                                                          gaussians[j].mean[0],
                                                                          gaussians[j].mean[1],
                                                                          cov[1])


        pmf = [float(ez[i])/sum(ez) for zz in ez]

        entr = entropy(pmf)
        p_ol = float(count_obj_landmark)/count_obj

        print(self.obj, self.landmark)
        print("ENTROPY(L,O) = ", entr)
        print("P(L|O)       = ", p_ol)
        score = p_ol * (float(1.0)/float(entr))
        print("SCORE =      = ", score)
        print("log(SCORE)   = ", math.log(score))

        

        # OLD FIGURES
        #print(self.landmark, landmark_bbox)        
        #print len(gmm_x), len(gmm_y), len(gmm_z)
        # fig = plt.figure()
        # ax = fig.add_subplot(111,projection='3d')
        # #ax.plot_trisurf(gmm_x, gmm_y, gmm_z, cmap=cm.jet, linewidth=0.2)
        # ax.scatter(gmm_x, gmm_y, gmm_z, c='r', marker='o')
        # plt.show()

        # MOST RECENT FIGURE!!!!!!!!!!!!!!!!!!!!!!!
                
        lm_yaw =  tf.transformations.euler_from_quaternion(landmark_orientation,axes='sxyz')
        print("landmark orientation: ", landmark_orientation)
        print("landmark orientation: ", lm_yaw)

        rotated_bbox = list()
        for corner in landmark_bbox:
            rx = corner[0] * math.cos(-lm_yaw[0]) - corner[1] * math.sin(-lm_yaw[0])
            ry = corner[0] * math.sin(-lm_yaw[0]) + corner[1] * math.cos(-lm_yaw[0])
            rz = corner[2]
            rotated_bbox.append([rx,ry,rz])
        
        landmark_local_bbox = BBox(rotated_bbox)
        

        x_dim = (landmark_local_bbox.get_x_max() - landmark_local_bbox.get_x_min()) 
        y_dim = (landmark_local_bbox.get_y_max() - landmark_local_bbox.get_y_min()) 
               
        #rect   =  mpatches.Rectangle([-y_dim/2, -x_dim/2], y_dim, x_dim, facecolor='white')
        rect   =  plt.Circle((0,0), 0.05, facecolor='white')   

        plt.gca().add_patch(rect)
        
        # Define grid.
        xi = np.linspace(-2.0,2.0,1000)
        yi = np.linspace(-1.2,1.2,1000)
        # grid the data.
        zi = griddata(gmm_x,gmm_y,gmm_z,xi,yi,interp='linear')
        # contour the gridded data, plotting dots at the nonuniform data points.
        CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
        CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet ,vmax=abs(zi).max(), vmin=-abs(zi).max())
        plt.colorbar() # draw colorbar
        # plot data points.
        #plt.scatter(gmm_x,gmm_y,marker='o',c='b',s=5,zorder=10)
        #plt.plot(gmm_x,  gmm_y, 'o')
        plt.xlim(-2.0,2.0)
        plt.ylim(-1.2,1.2)
        plt.show()

        # END MOST RECENT FIGURE!!!!!!!!!!!!!!!!!!!!!!!
        

        response = QSRToGMMResponse()

        response.gaussian = gaussians
        response.weight = weights

        return response


    def calc_GMM(self):

        print("Directions:", self.dir_count, sum(self.dir_count.values()))
        print("Distances", self.dist_count, sum(self.dist_count.values()))
        print("Dir/Close:", self.dir_close_count, sum(self.dir_close_count.values()))
        print("Dir/Distant", self.dir_distant_count, sum(self.dir_distant_count.values()))
        #print("Rel position (close)", self.dir_close_pos)
        #print("Rel position (distant)", self.dir_distant_pos)

        rospy.loginfo(self.obj)
        rospy.loginfo(self.landmark)
        rospy.loginfo(self.qsr)

        threshold = 0.03
        total_count = sum(self.dir_count.values())
        
        gmm_count = 0
        gmm_dir_close = list()
        gmm_dir_distant = list()
        for rel in self.dir_close_count.keys():
            if float(self.dir_close_count[rel])/total_count  > threshold:
                gmm_count += self.dir_close_count[rel]
                gmm_dir_close.append(rel)
            if float(self.dir_distant_count[rel])/total_count  > threshold:
                gmm_count += self.dir_distant_count[rel]
                gmm_dir_distant.append(rel)

        print("Total count: ", total_count)
        print("GMM count", gmm_count)
        print("GMM dir close", gmm_dir_close)
        print("GMM dir distant", gmm_dir_distant)


        weights = list()
        gaussians = list()
        
        for rel in gmm_dir_close:
            x = list()
            y = list()
            for pos in self.dir_close_pos[rel]:
                x.append(pos[0])
                y.append(pos[1])

            x_mean = np.mean(x)
            y_mean = np.mean(y)
            cov = np.cov(x,y)
            gaussian = Gaussian2D()
            gaussian.mean = [x_mean, y_mean]
            cov_lst = [val for subl in cov.tolist() for val in subl] 
            gaussian.covariance = cov_lst
            
            weight =  float(self.dir_close_count[rel])/gmm_count
            
            weights.append(weight)
            gaussians.append(gaussian)

        for rel in gmm_dir_distant:
            x = list()
            y = list()
            for pos in self.dir_distant_pos[rel]:
                x.append(pos[0])
                y.append(pos[1])

            x_mean = np.mean(x)
            y_mean = np.mean(y)
            cov = np.cov(x,y)
            gaussian = Gaussian2D()
            gaussian.mean = [x_mean, y_mean]
            cov_lst = [val for subl in cov.tolist() for val in subl] 
            gaussian.covariance = cov_lst
            
            weight =  float(self.dir_distant_count[rel])/gmm_count
            
            weights.append(weight)
            gaussians.append(gaussian)
        
        print("Sum of weights:", sum(weights))
        return weights, gaussians 



def entropy(pmf):
    return -(pmf * np.log(pmf)/np.log(2)).sum()


def seq(start, stop, step=1):
    n = int(round((stop - start)/float(step)))
    if n > 1:
        return([round(start + step*i,2) for i in range(n+1)])
    else:
        return([])


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def help_msg():
    return """
  Usage: qsr_to_gmm.py [-h] <qsrmodel> 

    qsrmodel        file including the QSR model for generationg the GMMs 

    -h, --help for seeing this msg
"""
    

if __name__ == "__main__":

    argv = None
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "h", ["help"])
        except getopt.error as msg:
            raise Usage(msg)

        if ('-h','') in opts or ('--help', '') in opts or len(args) != 1:
            raise Usage(help_msg())

        #print('Parsing QSR model')

        with open(args[0]) as qsr_file:    
            qsr_model = json.load(qsr_file)
    
            qsr_to_gmm = QSR2GMM(qsr_model)

    except Usage as err:
        print(err.msg)
        print("for help use --help")
