#!/usr/bin/env python

from qsr_msgs.srv import *
import rospy

def handle_qsr_to_gmm(req):
    print(req)
    return QSRToGMMResponse()

def qsr_to_gmm_server():
    rospy.init_node('qsr_to_gmm_server')
    s = rospy.Service('qsr_to_gmm', QSRToGMM, handle_qsr_to_gmm)
    print "Ready to transform QSRs into a GMM."
    rospy.spin()

if __name__ == "__main__":
    qsr_to_gmm_server()
