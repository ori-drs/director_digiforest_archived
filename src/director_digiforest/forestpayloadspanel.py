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
import shutil

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
        self.data_dir = None
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
            self.data_dir = newDir
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

        local_pointcloud_dir = os.path.join(self.data_dir, "individual_clouds")
        height_map_dir = os.path.join(self.data_dir, "height_maps")
        local_height_map = "height_map_"+str(sec)+"_"+self._convertNanoSecsToString(nsec)+".ply"
        height_map_file = os.path.join(height_map_dir, local_height_map)

        local_cloud = "cloud_"+str(sec)+"_"+self._convertNanoSecsToString(nsec)+".pcd"
        payload_cloud_dir = os.path.join(self.data_dir, "payload_clouds_in_map")
        tree_description_file = os.path.join(self.data_dir, "trees.csv")

        local_cloud_file = os.path.join(local_pointcloud_dir, local_cloud)
        payload_cloud_file = os.path.join(payload_cloud_dir, local_cloud)
        if os.path.isfile(payload_cloud_file):
            self.loadPointCloud(payload_cloud_file, trans, quat)
        elif os.path.isfile(local_cloud_file):
            self.loadPointCloud(local_cloud_file, trans, quat)

        if os.path.isfile(payload_cloud_file):
            self.terrain_mapping(payload_cloud_file, height_map_file)
        elif os.path.isfile(local_cloud_file):
            self.terrain_mapping(local_cloud_file, height_map_file)

        if os.path.isfile(tree_description_file):
            self.loadCylinders(tree_description_file)

        #images_dir = os.path.join(self.data_dir, "individual_images")
        # if os.path.isdir(images_dir):
        #     self.loadImages(images_dir, sec, self._convertNanoSecsToString(nsec))

    def loadCylinders(self, filename):
        '''
        From a csv file describing the trees as cylinders, load and display them
        '''
        print("Loading ", filename)
        self.treeData = np.loadtxt(filename, delimiter=" ", dtype=np.float)
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



    def loadPointCloud(self, filename, trans, quat):
        print("Loading : ", filename)
        if not os.path.isfile(filename):
            print("File doesn't exist", filename)
            return

        polyData = ioutils.readPolyData(filename, ignoreSensorPose=True)

        if not polyData or not polyData.GetNumberOfPoints():
            print("Error cannot load file")
            return

        #transformedPolyData = self.transformPolyData(polyData, trans, quat)
        obj = vis.showPolyData(polyData, os.path.basename(filename), parent="slam")
        vis.addChildFrame(obj)

    def transformPolyData(self, polyData, translation, quat):
        nodeTransform = transformUtils.transformFromPose(translation, quat)

        transformedPolyData = filterUtils.transformPolyData(polyData, nodeTransform)
        return transformedPolyData

    def loadg2oFile(self, filename):
        print("loading", filename)
        self.fileData = np.loadtxt(filename, delimiter=" ", dtype='<U21', usecols=np.arange(0,11))
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
        self._loadFileData(filename)

    def loadCsvFile(self, filename):
        print("loading", filename)
        self.fileData = np.loadtxt(filename, delimiter=",", dtype=np.float, skiprows=1)
        self._loadFileData(filename)

    def convert_heights_mesh(self, parent, height_map_file):
        if not os.path.isfile(height_map_file):
            pcd=pcl.PointCloud()
            pcd.from_list(self.heights_array_raw)
            pcd.to_file(b'/tmp/height_map.pcd')
            os.system("rosrun forest_nav generate_mesh") # running a ROS node to convert heights to mesh - nasty!
            height_maps_dir = os.path.dirname(height_map_file)
            if not os.path.isdir(height_maps_dir):
                os.makedirs(height_maps_dir)
            shutil.copyfile('/tmp/height_map.ply', height_map_file)
        else:
            print("Loading height_map", height_map_file)

        self.height_mesh = ioutils.readPolyData(height_map_file)
        self.height_mesh = segmentation.addCoordArraysToPolyDataXYZ( self.height_mesh )
        vis.showPolyData(self.height_mesh, 'Height Mesh', 'Color By', 'z',
                         colorByRange=[self.medianPoseHeight-4,self.medianPoseHeight+4], parent=parent)

    def terrain_mapping(self, filename, height_map_file):
        cloud_pc = pcl.PointCloud_PointNormal()
        cloud_pc._from_pcd_file(filename.encode('utf-8'))
        self._showPCLXYZNormal(cloud_pc, "Cloud Raw", visible=False, parent=os.path.basename(filename))

        # remove non-up points
        cloud = df.filterUpNormal(cloud_pc, 0.95)
        self._showPCLXYZNormal(cloud, "Cloud Up Normals only", visible=False, parent=os.path.basename(filename))

        # drop from xyznormal to xyz
        array_xyz = cloud.to_array()[:, 0:3]
        cloud = pcl.PointCloud()
        cloud.from_array(array_xyz)

        # get the terrain height
        self.heights_array_raw = df.getTerrainHeight(cloud)
        self.convert_heights_mesh(os.path.basename(filename), height_map_file)

        self.heights_pd = vnp.getVtkPolyDataFromNumpyPoints(self.heights_array_raw)
        obj = vis.showPolyData(self.heights_pd, 'Heights', color=[0, 1, 0], visible=False,
                               parent=os.path.basename(filename))
        obj.setProperty('Point Size', 10)

        # filter NaNs *** these could have been removed in function ***
        self.heights_array = self.heights_array_raw[self.heights_array_raw[:, 2] < 10]
        self.heights_pd = vnp.getVtkPolyDataFromNumpyPoints(self.heights_array)
        obj = vis.showPolyData(self.heights_pd, 'Heights Filtered', visible=False, color=[0, 0, 1],
                               parent=os.path.basename(filename))
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

    def _loadImageToVtk(self, filename):
        imageNumpy = matimage.imread(filename)
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
        return os.path.join(self.data_dir, "../exp" + str(expNum), "individual_images")

    def _getImageFileName(self, imagesDir, sec, nsec):
        return os.path.join(imagesDir, "image_" + str(sec) + "_" + self._convertNanoSecsToString(nsec) + ".png")

    def _loadFileData(self, filename):
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
                payloadDir = os.path.join(self.data_dir, "payload_clouds_in_map")
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
                            print("Failed to read data from file: ", filename)
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
                    print("Failed to read data from file: ", filename)
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
