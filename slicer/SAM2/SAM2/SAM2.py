import logging
import os
from typing import Annotated, Optional

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode

import numpy as np
import tempfile
import threading
import requests
import time


#
# SAM2
#


class SAM2(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SAM2")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Reza Asakereh (University Health Network)", "Sumin Kim (University of Toronto)", "Jun Ma (University Health Network)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#SAM2">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete



#
# SAM2ParameterNode
#


@parameterNodeWrapper
class SAM2ParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


#
# SAM2Widget
#


class SAM2Widget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SAM2.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = SAM2Logic()
        self.logic.widget = self

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Preprocessing
        self.ui.cmbPrepOptions.addItems(['Manual', 'Abdominal CT', 'Lung CT', 'Brain CT', 'Mediastinum CT', 'MR'])
        self.ui.cmbPrepOptions.currentTextChanged.connect(lambda new_text: self.setManualPreprocessVis(new_text == 'Manual'))
        self.ui.pbApplyPrep.connect('clicked(bool)', lambda: self.logic.applyPreprocess(self.ui.cmbPrepOptions.currentText, self.ui.sldWinLevel.value, self.ui.sldWinWidth.value))

        self.ui.cmbCheckpoint.addItems(['tiny', 'small', 'base_plus', 'large'])
        
        # Setting icons
        # Icons used here are downloaded from flaticon's free icons package. Detailed attributes can be found in slicer/SAM2/SAM2/Resources/Icons/attribute.html 
        from PythonQt.QtGui import QIcon
        iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')
        self.ui.pbApplyPrep.setIcon(QIcon(os.path.join(iconsPath, 'verify.png')))
        self.ui.btnStart.setIcon(QIcon(os.path.join(iconsPath, 'start.png')))
        self.ui.btnEnd.setIcon(QIcon(os.path.join(iconsPath, 'the-end.png')))
        self.ui.btnROI.setIcon(QIcon(os.path.join(iconsPath, 'bounding-box.png')))
        self.ui.btnMiddleSlice.setIcon(QIcon(os.path.join(iconsPath, 'target.png')))
        self.ui.btnRefine.setIcon(QIcon(os.path.join(iconsPath, 'performance.png')))
        self.ui.btnSegment.setIcon(QIcon(os.path.join(iconsPath, 'body-scan.png')))

        # Buttons
        self.ui.btnStart.connect("clicked(bool)", lambda: self.setROIboundary(lower=True))
        self.ui.btnEnd.connect("clicked(bool)", lambda: self.setROIboundary(lower=False))
        self.ui.btnROI.connect("clicked(bool)", self.drawBBox)
        self.ui.btnMiddleSlice.connect("clicked(bool)", self.logic.getMiddleMask)
        self.ui.btnRefine.connect("clicked(bool)", self.logic.refineMiddleMask)
        self.ui.btnSegment.connect("clicked(bool)", self.logic.segment)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
    
    def setManualPreprocessVis(self, visible):
        self.ui.lblLevel.setVisible(visible)
        self.ui.lblWidth.setVisible(visible)
        self.ui.sldWinLevel.setVisible(visible)
        self.ui.sldWinWidth.setVisible(visible)

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[SAM2ParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
    
    def setROIboundary(self, lower):
        if self.logic.boundaries is None:
            self.logic.boundaries = [None, None]
        curr_slice = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceOffset()
        self.logic.boundaries[int(not lower)] = curr_slice

        if None not in self.logic.boundaries:
            slicer.app.layoutManager().sliceWidget("Red").sliceLogic().SetSliceOffset(sum(self.logic.boundaries)/2)

        print(self.logic.boundaries)
    
    def drawBBox(self):
        # Adopted from https://github.com/bingogome/samm/blob/7da10edd7efe44d10369aa13eddead75a7d3a38a/samm/SammBase/SammBaseLib/WidgetSammBase.py
        planeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsROINode').GetID()
        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        selectionNode.SetReferenceActivePlaceNodeID(planeNode)
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        placeModePersistence = 0
        interactionNode.SetPlaceModePersistence(placeModePersistence)
        # mode 1 is Place, can also be accessed via slicer.vtkMRMLInteractionNode().Place
        interactionNode.SetCurrentInteractionMode(1)

        slicer.mrmlScene.GetNodeByID(planeNode).GetDisplayNode().SetGlyphScale(0.5)
        slicer.mrmlScene.GetNodeByID(planeNode).GetDisplayNode().SetInteractionHandleScale(1)



#
# SAM2Logic
#


class SAM2Logic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    boundaries = None
    volume_node = None
    image_data = None
    widget = None
    middleMaskNode = None
    allSegmentsNode = None
    segmentation_res_path = '/home/rasakereh/Desktop'

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return SAM2ParameterNode(super().getParameterNode())
    
    def captureImage(self):
        self.volume_node = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')[0]
        self.image_data = slicer.util.arrayFromVolume(self.volume_node)  ################ Only one node?
    
    def get_bounding_box(self, make2d=False):
        self.captureImage()
        roiNodes = slicer.util.getNodesByClass('vtkMRMLMarkupsROINode')

        # If volume node is transformed, apply that transform to get volume's RAS coordinates
        transformRasToVolumeRas = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, self.volume_node.GetParentTransformNode(), transformRasToVolumeRas)

        bboxes = []
        for roiNode in roiNodes:
            if make2d:
                ############ making it 2D
                roi_size = roiNode.GetSize()
                roiNode.SetSize(roi_size[0], roi_size[1], 1)
                roi_center = np.array(roiNode.GetCenter())
                roiNode.SetCenter([roi_center[0], roi_center[1], slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceOffset()])
                ############ making it 2D

            bounds = np.zeros(6)
            roiNode.GetBounds(bounds)
            point1 = bounds[::2].copy()
            point2 = bounds[1::2].copy()
            point1[2] = min(self.boundaries)
            point2[2] = max(self.boundaries)
            
            ijk_points = []

            for curr_point in [point1, point2]:
                # Get point coordinate in RAS
                point_VolumeRas = transformRasToVolumeRas.TransformPoint(curr_point)

                # Get voxel coordinates from physical coordinates
                volumeRasToIjk = vtk.vtkMatrix4x4()
                self.volume_node.GetRASToIJKMatrix(volumeRasToIjk)
                point_Ijk = [0, 0, 0, 1]
                volumeRasToIjk.MultiplyPoint(np.append(point_VolumeRas,1.0), point_Ijk)
                point_Ijk = [ int(round(c)) for c in point_Ijk[0:3] ]

                ijk_points.append(point_Ijk)

            zrange = [ijk_points[0][2], ijk_points[1][2]]
            slice_idx = int((zrange[0] + zrange[1]) / 2) # it is not accurate
            if ijk_points[0][0] > ijk_points[1][0]:
                ijk_points[0], ijk_points[1] = ijk_points[1], ijk_points[0]
            bbox = np.hstack([ijk_points[0][:2], ijk_points[1][:2]])
            bboxes.append(bbox)

        return slice_idx, bboxes, zrange
    
    def run_on_background(self, target, args, title):
        self.progressbar = slicer.util.createProgressDialog(autoClose=False)
        self.progressbar.minimum = 0
        self.progressbar.maximum = 0
        self.progressbar.setLabelText(title)
        
        job_event = threading.Event()
        paral_thread = threading.Thread(target=target, args=(*args, job_event,))
        paral_thread.start()
        while not job_event.is_set():
            slicer.app.processEvents()
        paral_thread.join()

        self.progressbar.close()
    

    def segment_helper(self, img_path, gts_path, result_path, ip, port, job_event):
        if self.widget.ui.pathModel.currentPath == '':
            checkpoint = 'sam2.1_hiera_%s.pt'%(self.widget.ui.cmbCheckpoint.currentText,)
        else:
            model_name = os.path.basename(self.widget.ui.pathModel.currentPath).split('.')[0]
            checkpoint = os.path.join(model_name, os.path.basename(self.widget.ui.pathModel.currentPath))

        self.progressbar.setLabelText(' uploading ground truth... ')
        upload_url = 'http://%s:%s/upload'%(ip, port)

        with open(gts_path, 'rb') as file:
            files = {'file': file}
            response = requests.post(upload_url, files=files)


        self.progressbar.setLabelText(' segmenting... ')
        run_script_url = 'http://%s:%s/run_script'%(ip, port)
        
        print('data sent is: ', {
                'checkpoint': checkpoint,
                'input': os.path.basename(img_path),
                'gts': os.path.basename(gts_path),
                'propagate': 'Y',
                'size': self.widget.ui.cmbCheckpoint.currentText,
            })

        response = requests.post(
            run_script_url,
            data={
                'checkpoint': checkpoint,
                'input': os.path.basename(img_path),
                'gts': os.path.basename(gts_path),
                'propagate': 'Y',
                'size': self.widget.ui.cmbCheckpoint.currentText,
            }
        )


        self.progressbar.setLabelText(' downloading results... ')
        download_file_url = 'http://%s:%s/download_file'%(ip, port)

        response = requests.get(download_file_url, data={'output': 'data/video/segs_tiny/%s'%os.path.basename(img_path)})

        with open(result_path, 'wb') as f: #TODO: arbitrary file name
            f.write(response.content)
        
        job_event.set()
    
    def showSegmentation(self, segmentation_mask, set_middle_mask=False):
        if self.allSegmentsNode is None:
            self.allSegmentsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")

        current_seg_group = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode") if set_middle_mask else self.allSegmentsNode
        current_seg_group.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        labels = np.unique(segmentation_mask)[1:] # all labels except background(0)

        for idx, label in enumerate(labels, start=1):
            curr_object = np.zeros_like(segmentation_mask)
            curr_object[segmentation_mask == idx] = idx
            segment_volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", 'segment_'+str(idx)+'_'+str(int(time.time())))
            slicer.util.updateVolumeFromArray(segment_volume, curr_object)

            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(segment_volume, current_seg_group)
            slicer.util.updateSegmentBinaryLabelmapFromArray(curr_object, current_seg_group, segment_volume.GetName(), self.volume_node)
            
            slicer.mrmlScene.RemoveNode(segment_volume)

        if set_middle_mask:
            self.middleMaskNode = current_seg_group
        else:
            try:
                slicer.mrmlScene.RemoveNode(self.middleMaskNode)
            except:
                pass


    def segment(self):
        self.captureImage()
        slice_idx, bboxes, zrange = self.get_bounding_box(make2d=False)
        with tempfile.TemporaryDirectory() as tmpdirname:
            img_path = "%s/img_data.npz"%(tmpdirname,)
            np.savez(img_path, imgs=self.image_data, boxes=bboxes, z_range=[*zrange, slice_idx])
            print(f"image file saved at:", img_path)
            gts_path = '%s/gts.npz'%(tmpdirname,)
            result_path = '%s/result.npz'%(tmpdirname,)
            np.savez(gts_path, segs=self.getSegmentationArray(self.middleMaskNode))
            self.run_on_background(self.segment_helper, (img_path, gts_path, result_path, self.widget.ui.txtIP.plainText, self.widget.ui.txtPort.plainText), 'Segmenting...')

            # loading results
            segmentation_mask = np.load(result_path, allow_pickle=True)['segs']
            self.showSegmentation(segmentation_mask)

        roiNodes = slicer.util.getNodesByClass('vtkMRMLMarkupsROINode')
        for roiNode in roiNodes:
            slicer.mrmlScene.RemoveNode(roiNode)
        self.boundaries = None
    
    def middle_mask_helper(self, img_path, result_path, ip, port, job_event):
        if self.widget.ui.pathModel.currentPath == '':
            checkpoint = 'sam2.1_hiera_%s.pt'%(self.widget.ui.cmbCheckpoint.currentText,)
        else:
            model_name = os.path.basename(self.widget.ui.pathModel.currentPath).split('.')[0]
            checkpoint = os.path.join(model_name, os.path.basename(self.widget.ui.pathModel.currentPath))
        
        if self.widget.ui.pathModel.currentPath != '':
            # TODO: Check if model is valid
            self.progressbar.setLabelText(' uploading model... ')
            upload_url = 'http://%s:%s/upload_model'%(ip, port)

            with open(self.widget.ui.pathModel.currentPath, 'rb') as file:
                files = {'file': file}
                response = requests.post(upload_url, files=files)

        self.progressbar.setLabelText(' uploading image... ')
        upload_url = 'http://%s:%s/upload'%(ip, port)

        with open(img_path, 'rb') as file:
            files = {'file': file}
            response = requests.post(upload_url, files=files)


        self.progressbar.setLabelText(' segmenting... ')
        run_script_url = 'http://%s:%s/run_script'%(ip, port)

        print('data sent is: ', {
                'checkpoint': checkpoint,
                'input': os.path.basename(img_path),
                'gts': 'X',
                'propagate': 'N',
                'size': self.widget.ui.cmbCheckpoint.currentText,
            })

        response = requests.post(
            run_script_url,
            data={
                'checkpoint': checkpoint,
                'input': os.path.basename(img_path),
                'gts': 'X',
                'propagate': 'N',
                'size': self.widget.ui.cmbCheckpoint.currentText,
            }
        )


        self.progressbar.setLabelText(' downloading results... ')
        download_file_url = 'http://%s:%s/download_file'%(ip, port)

        response = requests.get(download_file_url, data={'output': 'data/video/segs_tiny/%s'%os.path.basename(img_path)})

        with open(result_path, 'wb') as f:
            f.write(response.content)
        
        job_event.set()
    

    def getMiddleMask(self):
        self.captureImage()
        slice_idx, bboxes, zrange = self.get_bounding_box(make2d=True)
        with tempfile.TemporaryDirectory() as tmpdirname:
            img_path = "%s/img_data.npz"%(tmpdirname,)
            result_path = "%s/result.npz"%(tmpdirname,)
            np.savez(img_path, imgs=self.image_data, boxes=bboxes, z_range=[*zrange, slice_idx])
            self.run_on_background(self.middle_mask_helper, (img_path, result_path, self.widget.ui.txtIP.plainText, self.widget.ui.txtPort.plainText), 'Segmenting...')
            
            # loading results
            segmentation_mask = np.load(result_path, allow_pickle=True)['segs']
            self.showSegmentation(segmentation_mask, set_middle_mask=True)
        
        roiNodes = slicer.util.getNodesByClass('vtkMRMLMarkupsROINode')
        for roiNode in roiNodes:
            roiNode.SetDisplayVisibility(False)
    
    def getSegmentationArray(self, segmentationNode):
        segmentIds = segmentationNode.GetSegmentation().GetSegmentIDs()
        result = np.zeros_like(self.image_data)

        for idx, segmentId in enumerate(segmentIds, start=1):
            print('getting segmentation array for', idx, segmentId)
            segmentArray = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId)
            result[segmentArray != 0] = idx
        
        return result

    
    def refineMiddleMask(self):
        slicer.util.selectModule("SegmentEditor") 
    
    def preprocess_CT(self, win_level=40.0, win_width=400.0):
        self.captureImage()
        lower_bound, upper_bound = win_level - win_width/2, win_level + win_width/2
        image_data_pre = np.clip(self.image_data, lower_bound, upper_bound)
        image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
        image_data_pre = np.uint8(image_data_pre)

        self.volume_node.GetDisplayNode().SetAutoWindowLevel(False)
        self.volume_node.GetDisplayNode().SetWindowLevelMinMax(0, 255)
        
        return image_data_pre
    
    def preprocess_MR(self, lower_percent=0.5, upper_percent=99.5):
        self.captureImage()
        
        lower_bound, upper_bound = np.percentile(self.image_data[self.image_data > 0], lower_percent), np.percentile(self.image_data[self.image_data > 0], upper_percent)
        image_data_pre = np.clip(self.image_data, lower_bound, upper_bound)
        image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
        image_data_pre = np.uint8(image_data_pre)

        self.volume_node.GetDisplayNode().SetAutoWindowLevel(False)
        self.volume_node.GetDisplayNode().SetWindowLevelMinMax(0, 255)

        return image_data_pre
    
    def updateImage(self, new_image):
        self.image_data[:,:,:] = new_image
        slicer.util.arrayFromVolumeModified(self.volume_node)
    
    def applyPreprocess(self, method, win_level, win_width):
        if method == 'MR':
            prep_img = self.preprocess_MR()
        elif method == 'Manual':
            prep_img = self.preprocess_CT(win_level = win_level, win_width = win_width)
        else:
            conversion = {
                'Abdominal CT': (400.0, 40.0),
                'Lung CT': (1500.0, -600.0),
                'Brain CT': (80.0, 40.0),
                'Mediastinum CT': (350.0, 50.0),
            }
            ww, wl = conversion[method]
            prep_img = self.preprocess_CT(win_level = wl, win_width = ww)

        self.updateImage(prep_img)

        



#
# SAM2Test
#


class SAM2Test(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_SAM21()

    def test_SAM21(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        self.delayDisplay("Test passed")
