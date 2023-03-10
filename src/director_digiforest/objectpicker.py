from director import visualization as vis
from director.timercallback import TimerCallback
from director import objectmodel as om
from director.debugpolydata import DebugData
from PythonQt import QtCore, QtGui, QtUiTools

class ObjectPicker(TimerCallback):
    def __init__(self, view, pickType="points", numberOfPoints=3):
        TimerCallback.__init__(self)
        self.targetFps = 30
        self.enabled = False
        self.pickType = pickType
        self.numberOfPoints = numberOfPoints
        self.annotationObj = None
        self.drawLines = True
        self.view = view
        self.clear()

    def clear(self):
        self.points = [None for i in range(self.numberOfPoints)]
        self.hoverPos = None
        self.annotationFunc = None
        self.lastMovePos = [0, 0]

    def onMouseMove(self, displayPoint, modifiers=None):
        self.lastMovePos = displayPoint

    def onMousePress(self, displayPoint, modifiers=None):
        for i in range(self.numberOfPoints):
            if self.points[i] is None:
                self.points[i] = self.hoverPos
                break

        if self.points[-1] is not None:
            self.finish()

    def finish(self):

        self.enabled = False
        om.removeFromObjectModel(self.annotationObj)

        points = [p.copy() for p in self.points]
        if self.annotationFunc is not None:
            self.annotationFunc(*points)

    def handleRelease(self, displayPoint):
        pass

    def draw(self):

        d = DebugData()

        points = [p if p is not None else self.hoverPos for p in self.points]

        # draw points
        for p in points:
            if p is not None:
                d.addSphere(p, radius=0.08)

        self.annotationObj = vis.updatePolyData(
            d.getPolyData(), "annotation", parent=om.findObjectByName("slam")
        )
        self.annotationObj.setProperty("Color", QtGui.QColor(0, 255, 0))
        self.annotationObj.actor.SetPickable(False)

    def tick(self):
        if not self.enabled:
            return

        pickedPointFields = vis.pickPoint(
            self.lastMovePos, self.view, pickType=self.pickType
        )
        self.hoverPos = pickedPointFields.pickedPoint
        self.draw()