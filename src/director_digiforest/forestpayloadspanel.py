import PythonQt
from PythonQt import QtCore, QtGui, QtUiTools
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from director import applogic as app
from director.utime import getUtime
from director import transformUtils
from director import objectmodel as om
from director import visualization as vis
from director.thirdparty import transformations
from director.debugpolydata import DebugData
from director import ioutils
from director import filterUtils
from director import vtkNumpy as vnp
from director import applogic as app
from director import segmentation
from director import vtkNumpy
from matplotlib import image as matimage
import matplotlib.pyplot as plt
import vtkAll as vtk
from director_digiforest.objectpicker import ObjectPicker
from vtk.util import numpy_support

import forest_nav_py as df
import pcl

import os
import re
import numpy as np
import functools

def addWidgetsToDict(widgets, d):

    for widget in widgets:
        if widget.objectName:
            d[str(widget.objectName)] = widget
        addWidgetsToDict(widget.children(), d)


class WidgetDict(object):
    def __init__(self, widgets):
        addWidgetsToDict(widgets, self.__dict__)


class ForestPayloadsPanel(QObject):

    def __init__(self, imageManager):
        QObject.__init__(self)

        loader = QtUiTools.QUiLoader()
        uifile = QtCore.QFile(
            os.path.join(os.path.dirname(__file__), "ui/ddForestPayloads.ui")
        )
        assert uifile.open(uifile.ReadOnly)

        self.widget = loader.load(uifile)
        self.fileData = np.array([])
        self.view = app.getDRCView()
        self.ui = WidgetDict(self.widget.children())
        
        self.ui.loadGraphButton.connect(
            "clicked()", self.onChooseRunInputDir
        )
        self.ui.startPickingButton.connect(
            "clicked()", self._startNodePicking
        )
        self.ui.stopPickingButton.connect(
            "clicked()", self._stopNodePicking
        )
        self.ui.startTreePickingButton.connect(
            "clicked()", self._startTreePicking
        )
        self.ui.stopTreePickingButton.connect(
            "clicked()", self._stopNodePicking   # not used
        )
        self.ui.pickButtonCombinedPointCloud.connect(
            "clicked()", self._startPicking
        )
        self.dataDir = None
        self.imageManager = imageManager
        self.treeData = np.array([])
        
        # Variable to help with automatic height colorization
        self.medianPoseHeight = 0
        
    def runInputDirectory(self):
        return os.path.expanduser(self.ui.loadGraphText.text)

    def chooseDirectory(self):
        return QtGui.QFileDialog.getExistingDirectory(
            app.getMainWindow(), "Choose directory...", self.runInputDirectory()
        )    

    def getShorterNameLast(self, name):
        if len(name) > 30:
            name = "..." + name[len(name) - 20 :]

        return name

    def onChooseRunInputDir(self):
        newDir = self.chooseDirectory()
        if newDir:
            self.dataDir = newDir
            self.ui.loadGraphText.text = self.getShorterNameLast(newDir)
            self.parsePoseGraph(newDir)
            
    def parsePoseGraph(self, directory):
        poseGraphFile = os.path.join(directory, "slam_poses.csv")
        if os.path.isfile(poseGraphFile):
            self.loadCsvFile(poseGraphFile)
        else:
            poseGraphFile = os.path.join(directory, "slam_pose_graph.g2o")
            if not os.path.isfile(poseGraphFile):
                print("Cannot find slam_poses.csv or slam_pose_graph.g2o in", directory)
            else:
                self.loadg2oFile(poseGraphFile)

    def findTree(self, pickedCoords):

        def points_in_cylinder(center, r, length, q):
            '''
            given a cylinder defined by center, length and radius, return whether q is inside it
            '''
            dist = np.linalg.norm(center[0:2]-q[0:2]) # 2d projection
            eps = 0.1
            return (dist <= (r+eps) and np.abs(center[2]-q[2]) <= (0.5*length+eps))


        for tree in self.treeData:
            axis = tree[3:6]
            center = tree[0:3]
            length = tree[6]
            radius = tree[7]
            q = np.array(pickedCoords)
            if points_in_cylinder(center, radius, length, q):
                # found tree
                self.ui.labelLength.text = length
                self.ui.labelRadius.text = radius
                break


    def findNodeData(self, pickedCoords):
        '''
        Find and load the data ( point cloud, images ) stored in a node
        '''
        print("LoadPointCloud", pickedCoords)
        nodeFound = False
        for row in self.fileData:
            if np.isclose(pickedCoords[0], row[3]) and np.isclose(pickedCoords[1], row[4]) \
                    and np.isclose(pickedCoords[2], row[5]):
                expNum = int(row[0]) // 1000 + 1
                sec = int(row[1])
                nsec = int(row[2])
                trans = (row[3], row[4], row[5])
                quat = (row[9], row[6], row[7], row[8]) # wxyz
                nodeFound = True
                break

        if not nodeFound:
            print("Error : Cannot find picked node")
            return

        #localPointCloudDir = os.path.join(self.dataDir, "../exp"+str(expNum), "individual_clouds")
        localPointCloudDir = os.path.join(self.dataDir, "individual_clouds")
        localPointCloud = "cloud_"+str(sec)+"_"+self._convertNanoSecsToString(nsec)+".pcd"
        payloadCloudDir = os.path.join(self.dataDir, "payload_clouds_in_map")
        imagesDir = os.path.join(self.dataDir, "individual_images")
        treeDescriptionFile = os.path.join(self.dataDir, "trees.csv")

        localPointCloudFile = os.path.join(localPointCloudDir, localPointCloud)
        payloadPointCloudFile = os.path.join(payloadCloudDir, localPointCloud)
        if os.path.isfile(localPointCloudFile):
            self.loadPointCloud(localPointCloudFile, trans, quat)
        elif os.path.isfile(payloadPointCloudFile):
            self.loadPointCloud(payloadPointCloudFile, trans, quat)

        if os.path.isfile(treeDescriptionFile):
            self.loadCylinders(treeDescriptionFile)

        if os.path.isdir(imagesDir):
            self.loadImages(imagesDir, sec, self._convertNanoSecsToString(nsec))

    def loadCylinders(self, fileName):
        '''
        From a csv file describing the trees as cylinders, load and display them
        '''
        print("Loading ", fileName)
        self.treeData = np.loadtxt(fileName, delimiter=" ", dtype=np.float)
        id = 0
        for tree in self.treeData:
            if tree.size < 7:
                continue

            d = DebugData()
            d.addCylinder(center=tree[0:3], axis=tree[3:6], length=tree[6], radius=tree[7])
            polyData = d.getPolyData()
            if not polyData or not polyData.GetNumberOfPoints():
                continue

            # store some information about the trees in the polydata data structure
            length = np.asfortranarray(np.ones(polyData.GetNumberOfPoints())*tree[6])
            vnp.addNumpyToVtk(polyData, length, "length")
            radius = np.asfortranarray(np.ones(polyData.GetNumberOfPoints()) * tree[7])
            vnp.addNumpyToVtk(polyData, radius, "radius")
            # tmp = polyData.GetPointData().GetArray("length")
            # print("tmp", tmp, type(tmp))
            # for i in range(0, tmp.GetNumberOfTuples()):
            #     print(i, tmp.GetTuple(i))
            #

            # print tree info next to the loaded trees
            obj = vis.showPolyData(
                polyData, "tree_"+str(id), parent="trees"
            )
            vis.addChildFrame(obj)

            id += 1

    def loadImages(self, directory, sec, nsec):
        listSubfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
        for imageFolder in listSubfolders:
            imageFile = os.path.join(imageFolder, "image_"+str(sec)+"_"+self._convertNanoSecsToString(nsec)+".png")
            if not os.path.isfile(imageFile):
                print("Cannot load image file", imageFile)
                return

            camNum = os.path.basename(os.path.normpath(imageFolder))
            self._loadImage(imageFile, camNum)



    def loadPointCloud(self, fileName, trans, quat):
        print("Loading : ", fileName)
        if not os.path.isfile(fileName):
            print("File doesn't exist", fileName)
            return

        polyData = ioutils.readPolyData(fileName, ignoreSensorPose=True)

        if not polyData or not polyData.GetNumberOfPoints():
            print("Error cannot load file")
            return

        #transformedPolyData = self.transformPolyData(polyData, trans, quat)
        obj = vis.showPolyData(polyData, os.path.basename(fileName), parent="slam")
        vis.addChildFrame(obj)
        self.terrainMapping(fileName)

    def transformPolyData(self, polyData, translation, quat):
        nodeTransform = transformUtils.transformFromPose(translation, quat)

        transformedPolyData = filterUtils.transformPolyData(polyData, nodeTransform)
        return transformedPolyData

    def loadg2oFile(self, fileName):
        print("loading", fileName)
        self.fileData = np.loadtxt(fileName, delimiter=" ", dtype='<U21', usecols=np.arange(0,11))
        #only keep vertex SE3 rows
        self.fileData = np.delete(self.fileData, np.where(
                       (self.fileData[:, 0] == "EDGE_SE3:QUAT"))[0], axis=0)

        #rearrange colums
        self.fileData = np.delete(self.fileData, 0, axis=1)
        sec = self.fileData[:, 8]
        nsec = self.fileData[:, 9]
        # removing sec and nsec columns
        self.fileData = np.delete(self.fileData, 8, axis=1)
        self.fileData = np.delete(self.fileData, 8, axis=1)
        # reinsert them at correct location
        self.fileData = np.insert(self.fileData, 1, sec, axis=1)
        self.fileData = np.insert(self.fileData, 2, nsec, axis=1)

        self.fileData = self.fileData.astype(float, copy=False)
        self._loadFileData(fileName)

    def loadCsvFile(self, fileName):
        print("loading", fileName)
        self.fileData = np.loadtxt(fileName, delimiter=",", dtype=np.float, skiprows=1)
        self._loadFileData(fileName)

    def terrainMappingTest(self):
        fileName = os.getenv('HOME') + "/vilens_slam_offline_data/data/2022-09-00-finland-reference/1.2_id_1/payload_clouds_transformed_in_map_frame/cloud_1663668471_346755000.pcd"
        self.terrainMapping(fileName)

    def convertHeightsToMesh(self,parent):
        pcd=pcl.PointCloud()
        pcd.from_list(self.heights_array_raw)
        pcd.to_file(b'/tmp/height_map.pcd')
        os.system("rosrun forest_nav generate_mesh") # running a ROS node to convert heights to mesh - nasty!
        self.height_mesh = ioutils.readPolyData("/tmp/height_map.ply")
        self.height_mesh = segmentation.addCoordArraysToPolyDataXYZ( self.height_mesh )
        vis.showPolyData(self.height_mesh,'Height Mesh','Color By','z',colorByRange=[self.medianPoseHeight-4,self.medianPoseHeight+4], parent=parent)

    def terrainMapping(self, fileName):
        cloud_pc = pcl.PointCloud_PointNormal()
        cloud_pc._from_pcd_file(fileName.encode('utf-8'))
        self._showPCLXYZNormal(cloud_pc, "Cloud Raw", visible=False, parent=os.path.basename(fileName))

        # remove non-up points
        cloud = df.filterUpNormal(cloud_pc, 0.95)
        self._showPCLXYZNormal(cloud, "Cloud Up Normals only", visible=False, parent=os.path.basename(fileName))

        # drop from xyznormal to xyz
        array_xyz = cloud.to_array()[:, 0:3]
        cloud = pcl.PointCloud()
        cloud.from_array(array_xyz)

        # get the terrain height
        self.heights_array_raw = df.getTerrainHeight(cloud)
        self.convertHeightsToMesh(os.path.basename(fileName))

        self.heights_pd = vnp.getVtkPolyDataFromNumpyPoints(self.heights_array_raw)
        obj = vis.showPolyData(self.heights_pd, 'Heights', color=[0, 1, 0], visible=False,
                               parent=os.path.basename(fileName))
        obj.setProperty('Point Size', 10)

        # filter NaNs *** these could have been removed in function ***
        self.heights_array = self.heights_array_raw[self.heights_array_raw[:, 2] < 10]
        self.heights_pd = vnp.getVtkPolyDataFromNumpyPoints(self.heights_array)
        obj = vis.showPolyData(self.heights_pd, 'Heights Filtered', visible=False, color=[0, 0, 1],
                               parent=os.path.basename(fileName))
        obj.setProperty('Point Size', 10)

    def _convertPclToPolyData(self, cloud):
        polyData=vnp.getVtkPolyDataFromNumpyPoints(cloud.to_array())
        return polyData

    def _showPCLXYZNormal(self, cloud_pc, name, visible, parent):
        array_xyz = cloud_pc.to_array()[:,0:3]
        cloud_pc = pcl.PointCloud()
        cloud_pc.from_array(array_xyz)
        cloud_pd= self._convertPclToPolyData(cloud_pc)
        vis.showPolyData(cloud_pd, name, visible=visible, parent=parent)

    def _startNodePicking(self, ):
        picker = ObjectPicker(numberOfPoints=1, view=app.getCurrentRenderView())
        segmentation.addViewPicker(picker)
        picker.enabled = True
        picker.start()
        picker.annotationFunc = functools.partial(self.findNodeData)

    def _startTreePicking(self, ):
        picker = ObjectPicker(numberOfPoints=1, view=app.getCurrentRenderView(), pickType="cells")
        segmentation.addViewPicker(picker)
        picker.enabled = True
        picker.start()
        picker.annotationFunc = functools.partial(self.findTree)

    def _stopNodePicking(self):
        pass

    def _shallowCopy(self, dataObj):
        newData = dataObj.NewInstance()
        newData.ShallowCopy(dataObj)
        return newData

    def _convertNanoSecsToString(self, nsec):
        """Returns a 9 characters string of a nano sec value"""
        s = str(nsec)
        if len(s) == 9:
            return s

        for i in range(len(s) + 1, 10):
            s = '0' + s
        return s

    def _loadImage(self, imageFile, imageId):
        polyData = self._loadImageToVtk(imageFile)
        self.imageManager.addImage(imageId, polyData)

    def _loadImageToVtk(self, fileName):
        imageNumpy = matimage.imread(fileName)
        imageNumpy = np.multiply(imageNumpy, 255).astype(np.uint8)
        imageColor = self._convertImageToColor(imageNumpy)
        image = vtkNumpy.numpyToImageData(imageColor)
        return image

    def _convertImageToColor(self, image):
        if len(image.shape) > 2:
            return image

        colorImg = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        colorImg[:, :, 0] = image
        colorImg[:, :, 1] = image
        colorImg[:, :, 2] = image
        return colorImg

    def _getImageDir(self, expNum):
        return os.path.join(self.dataDir, "../exp" + str(expNum), "individual_images")

    def _getImageFileName(self, imagesDir, sec, nsec):
        return os.path.join(imagesDir, "image_" + str(sec) + "_" + self._convertNanoSecsToString(nsec) + ".png")

    def _loadFileData(self, fileName):
        colors = [QtGui.QColor(0, 255, 0), QtGui.QColor(255, 0, 0), QtGui.QColor(0, 0, 255),
                  QtGui.QColor(255, 255, 0), QtGui.QColor(255, 0, 255), QtGui.QColor(0, 255, 255)]
        expNum = 1
        data = np.array([]) # point coordinates
        timestamps = np.array([], dtype=np.int64) # sec, nsec
        indexRow = 1
        for row in self.fileData:
            # assumes that an experiment has less than 10000 elements
            if row[0] > expNum*10000 or indexRow == self.fileData.shape[0]:
                # drawing the pose graph

                # finding the payload nodes
                payloadDir = os.path.join(self.dataDir, "payload_clouds_in_map")
                if os.path.isdir(payloadDir):
                    payloadFiles = [f for f in os.listdir(payloadDir) if os.path.isfile(os.path.join(payloadDir, f))]
                    dataPayload = np.array([])  # point coordinates
                    indexPayload = np.array([], dtype=np.int64)
                    for file in payloadFiles:
                        split = re.split('\W+|_', file)
                        if len(split) >= 3:
                            sec = int(split[1])
                            nsec= int(split[2])
                            index = np.where(np.logical_and(timestamps[:, 0] == sec, timestamps[:, 1] == nsec))

                            if len(index) >= 1 and index[0].size == 1:
                                indexPayload = np.append(indexPayload, int(index[0][0]))

                    if indexPayload.size > 0:
                        dataPayload = data[indexPayload, :]
                        data = np.delete(data, indexPayload, axis=0)
                        timestamps = np.delete(timestamps, indexPayload, axis=0)
                        polyData = vnp.numpyToPolyData(dataPayload)

                        if not polyData or not polyData.GetNumberOfPoints():
                            print("Failed to read data from file: ", fileName)
                            return

                        zvalues = vtkNumpy.getNumpyFromVtk(polyData, "Points")[:, 2]
                        self.medianPoseHeight = np.median(zvalues)

                        obj = vis.showPolyData(
                            polyData, "payload_" + str(expNum), parent="slam"
                        )
                        obj.setProperty("Point Size", 8)
                        obj.setProperty("Color", colors[(expNum) % len(colors)])
                        vis.addChildFrame(obj)

                # loading the non-payload nodes

                polyData = vnp.numpyToPolyData(data)

                if not polyData or not polyData.GetNumberOfPoints():
                    print("Failed to read data from file: ", fileName)
                    return

                obj = vis.showPolyData(
                    polyData, "experiment_"+str(expNum), parent="slam"
                )
                obj.setProperty("Point Size", 6)
                obj.setProperty("Color", colors[(expNum-1) % len(colors)])
                vis.addChildFrame(obj)
                expNum += 1
                data = np.array([])
                timestamps = np.array([])
            else:
                position = np.array([row[3], row[4], row[5]])
                timestamp = np.array([row[1], row[2]], dtype=np.int64)
                if data.shape[0] != 0:
                    data = np.vstack((data, position))
                    timestamps = np.vstack((timestamps, timestamp))
                else:
                    data = position
                    timestamps = timestamp
            indexRow += 1

    def _startPicking(self):
        picker = segmentation.PointPicker(numberOfPoints=1, polyDataName='combined_cloud.pcd')
        picker.view = app.getDRCView()
        segmentation.addViewPicker(picker)
        picker.enabled = True
        picker.drawLines = False
        picker.start()
        picker.annotationFunc = functools.partial(self._selectBestImage)

    def _selectBestImage(self, pickedPoint):
        '''
        Givena point, displays the best image stored in the pose graph that can see the point
        '''
        print("Picked point :", pickedPoint)
        combinedPointCloud = om.findObjectByName('combined_cloud.pcd').polyData
        if not combinedPointCloud or not combinedPointCloud.GetNumberOfPoints():
            print("combined_cloud is not found")
            return

        viewpoint = [pickedPoint[0] - self.view.camera().GetPosition()[0],
                     pickedPoint[1] - self.view.camera().GetPosition()[1], 0]
        viewpoint = viewpoint / np.linalg.norm(viewpoint)

        ## browse all nodes
        if self.fileData.size == 0:
            return
        searchRadius = 10
        bestNode = (-1, [0, 0, 0], [0, 0, 0], None, 0, 0)
        for row in self.fileData:
            dist = np.sqrt((pickedPoint[0]-row[3])**2 + (pickedPoint[1]-row[4])**2 + (pickedPoint[2]-row[5])**2)
            if dist < searchRadius:
                expNum = int(row[0]) // 1000 + 1
                sec = int(row[1])
                nsec = int(row[2])
                imagesDir = self._getImageDir(expNum)
                nodePosition = (row[3], row[4], row[5])
                quat = (row[9], row[6], row[7], row[8]) # wxyz
                roll, pitch, yaw = transformUtils.quaternionToRollPitchYaw(quat)
                nodeOrientation = [np.cos(yaw), np.sin(yaw), 0]
                # viewpoint and nodeOrientation must be as aligned as possible
                dotProduct = np.absolute(np.dot(viewpoint, nodeOrientation))
                if dotProduct > bestNode[0] and self._isPointVisible(nodePosition, nodeOrientation, pickedPoint):
                    bestNode = (dotProduct, nodeOrientation, nodePosition, imagesDir, sec, nsec)


        # display best images
        if bestNode[3] == None:
            print("Cannot find suitable node")
            return
        self.loadImages(bestNode[3], bestNode[4], bestNode[5])


        #debug : visualize best fit
        d = DebugData()
        d.addSphere(bestNode[2], 1.0)
        item = om.findObjectByName("best_node")
        if item:
            om.removeFromObjectModel(item)
        vis.showPolyData(d.getPolyData(), "best_node", parent="slam")
        #
        d = DebugData()
        d.addArrow(pickedPoint, pickedPoint+np.array(viewpoint), 0.3)
        item = om.findObjectByName("viewpoint")
        if item:
            om.removeFromObjectModel(item)
        vis.showPolyData(d.getPolyData(), "viewpoint", parent="slam")
        d = DebugData()
        d.addArrow(bestNode[2], bestNode[2]+np.array(bestNode[1]), 0.3, color=[0, 1, 1])
        item = om.findObjectByName("node_orientation")
        if item:
            om.removeFromObjectModel(item)
        vis.showPolyData(d.getPolyData(), "node_orientation", parent="slam")


    def _isPointVisible(self, origin, nodeOrientation, pt):
        '''
        Return true if a point pt can be visible from a node, given the position and orientation of the node
        '''
        if np.dot(nodeOrientation, np.array([pt[0]-origin[0], pt[1]-origin[1], pt[2]-origin[2]])) > 0.3:
            return True
        else:
            return False



        
def init(imageManager):

    global panels
    global docks

    if "panels" not in globals():
        panels = {}
    if "docks" not in globals():
        docks = {}

    panel = ForestPayloadsPanel(imageManager)
    action = app.addDockAction(
        "ForestPayloadsPanel",
        "Forest Payloads",
        os.path.join(os.path.dirname(__file__), "images/forest.png"),
    )
    dock = app.addWidgetToDock(
        panel.widget, action=action
    )

    dock.hide()

    return panel
