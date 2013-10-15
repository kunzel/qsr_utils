#!/usr/bin/env python

from qsr_msgs.srv import *
from qsr_msgs.msg import *
import rospy
import getopt
import json
import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle



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
                    
        for scn in self.qsr_model:
            scn_types = scn[1]['type'].values()
            if self.obj in scn_types and self.landmark in scn_types:
                #rospy.loginfo(scn[0])
                self.qsr_to_hist(scn[1]['qsr'],scn[1]['position'])


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
