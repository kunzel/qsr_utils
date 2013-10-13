#!/usr/bin/env python

from qsr_msgs.srv import *
import rospy
import getopt
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
        relative_pos = [x_rel, y_rel]
        
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
                

        fig, ax = plt.subplots()
        ax.plot(x_close, y_close, 'o')
        ax.plot(x_dist, y_dist, 'x')
        ax.set_title(self.obj + 'WRT' + self.landmark)
        plt.show()

        calc_GMM()

                
        print("Directions:", self.dir_count)
        print("Distances", self.dist_count)
        print("Dir/Close:", self.dir_close_count)
        print("Dir/Distant", self.dir_distant_count)
        #print("Rel position (close)", self.dir_close_pos)
        #print("Rel position (distant)", self.dir_distant_pos)

        rospy.loginfo(self.obj)
        rospy.loginfo(self.landmark)
        rospy.loginfo(self.qsr)

# Which QSRs are relevant? ALL or only parts of it?
        
        return QSRToGMMResponse()

    def calc_GMM(self):
        pass


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
