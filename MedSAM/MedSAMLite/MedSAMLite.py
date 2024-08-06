import logging
import os
from typing import Annotated, Optional
import time
import json

import vtk
import requests
import SimpleITK as sitk
import numpy as np
import subprocess
import atexit
import os
import signal
import threading
import zipfile

import slicer
from slicer import vtkMRMLMarkupsROINode, vtkMRMLSegmentationNode
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from urllib.request import urlopen

from slicer import vtkMRMLScalarVolumeNode

from PythonQt.QtCore import QTimer, QByteArray
from PythonQt.QtGui import QIcon, QPixmap, QMessageBox

try:
    from medsam_interface import MedSAM_Interface # FIXME
except:
    pass # no installation anymore, shorter plugin load

MEDSAMLITE_VERSION = 'v0.1'

#
# MedSAMLite
#

class MedSAMLite(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "MedSAM Lite"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Reza Asakereh (University Health Network)", "Andrew Qiao (University Health Network)", "Jun Ma (University Health Network)"]
        self.parent.helpText = """
This module is an implementation of the semi-automated segmentation method, MedSAM.
See more information in <a href="https://github.com/bowang-lab/MedSAMSlicer">module github page</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # MedSAMLite1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='MedSAMLite',
        sampleName='MedSAMLite1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'MedSAMLite1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='MedSAMLite1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='MedSAMLite1'
    )

    # MedSAMLite2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='MedSAMLite',
        sampleName='MedSAMLite2',
        thumbnailFileName=os.path.join(iconsPath, 'MedSAMLite2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='MedSAMLite2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='MedSAMLite2'
    )


#
# MedSAMLiteParameterNode
#

@parameterNodeWrapper
class MedSAMLiteParameterNode:
    roiNode: vtkMRMLMarkupsROINode
    segmentationNode: vtkMRMLSegmentationNode
    modelPath: str
    prepMethod: str
    prepWinLevel: float = 40.0
    prepWinWidth: float = 400.0


#
# MedSAMLiteWidget
#

class MedSAMLiteWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self.parameterSetNode = None

    def setup(self) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = MedSAMLiteLogic()
        self.logic.widget = self

        self.logic.server_dir = os.path.join(os.path.dirname(__file__), 'Resources/server_essentials')

        DEPENDENCIES_AVAILABLE = False

        # Initial Dependency Setup
        try:
            from segment_anything.modeling import MaskDecoder
            DEPENDENCIES_AVAILABLE = True
        except:
            DEPENDENCIES_AVAILABLE = False
        
        if not DEPENDENCIES_AVAILABLE:
            from PythonQt.QtGui import QLabel, QPushButton, QSpacerItem, QSizePolicy, QCheckBox
            import ctk
            install_instruction = QLabel('This package requires some dependencies to be installed prior to use.')
            restart_instruction = QLabel('Restart 3D Slicer after all dependencies are installed!')

            install_btn = QPushButton('Install dependencies')
            install_btn.clicked.connect(self.logic.install_dependencies)

            self.layout.addWidget(install_instruction)
            self.layout.addWidget(install_btn)
            self.layout.addWidget(restart_instruction)
            
            return

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/MedSAMLite.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        ############################################################################
        # Model Selection
        self.model_path_widget = self.ui.ctkPathModel
        self.model_path_widget.currentPath = os.path.join(self.logic.server_dir, 'medsam_lite.pth')
        self.logic.new_model_loaded = True
        ############################################################################


        ############################################################################
        # Segmentation Module
        import qSlicerSegmentationsModuleWidgetsPythonQt
        self.editor = qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget()
        self.editor.setMaximumNumberOfUndoStates(10)
        self.selectParameterNode()
        self.editor.setMRMLScene(slicer.mrmlScene)
        self.ui.clbtnOperation.layout().addWidget(self.editor, 4, 0, 1, 2)
        ############################################################################

        ###########################################################################
        # Volume load/close tracker
        from PythonQt.qMRMLWidgets import qMRMLNodeComboBox

        self.qNodeSelect = qMRMLNodeComboBox()
        self.qNodeSelect.addEnabled = False
        self.qNodeSelect.removeEnabled = False
        self.qNodeSelect.nodeTypes = ['vtkMRMLScalarVolumeNode']
        self.qNodeSelect.setMRMLScene(slicer.mrmlScene)
        self.qNodeSelect.currentNodeChanged.connect(self.logic.volumeChanged)
        self.logic.volumeChanged()
        ###########################################################################

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.pbUpgrade.connect('clicked(bool)', lambda: self.logic.run_on_background(self.logic.upgrade, (True,), 'Checking for updates...'))
        self.ui.pbSendImage.connect('clicked(bool)', lambda: self.logic.sendImage(partial=False))
        self.ui.pbSegment.connect('clicked(bool)', lambda: self.logic.applySegmentation())

        # Preprocessing
        self.ui.cmbPrepOptions.addItems(['Manual', 'Abdominal CT', 'Lung CT', 'Brain CT', 'Mediastinum CT', 'MR'])
        self.ui.cmbPrepOptions.currentTextChanged.connect(lambda new_text: self.setManualPreprocessVis(new_text == 'Manual'))
        self.ui.pbApplyPrep.connect('clicked(bool)', lambda: self.logic.applyPreprocess(self.ui.cmbPrepOptions.currentText, self.ui.sldWinLevel.value, self.ui.sldWinWidth.value))

        # Hide unnecessary ROI controls
        self.ui.widgetROI.findChild("QLabel", "label").hide()
        self.ui.widgetROI.findChild("QCheckBox", "insideOutCheckBox").hide()
        self.ui.widgetROI.findChild("QLabel", "label_10").hide()
        self.ui.widgetROI.findChild("QComboBox", "roiTypeComboBox").hide()

        # Segmentation Speed
        self.ui.cmbSpeed.addItems(['Normal Speed - Highest Quality', 'Faster Segmentation - High Quality', 'Fastest Segmentation - Moderate Quality'])
        
        self.ui.pbAttach.connect('clicked(bool)', lambda: self._createAndAttachROI())
        self.ui.pbTwoDim.connect('clicked(bool)', lambda: self.makeROI2D())
        self.ui.pbLowerSelection.connect('clicked(bool)', lambda: self.setROIboundary(lower=True))
        self.ui.pbUpperSelection.connect('clicked(bool)', lambda: self.setROIboundary(lower=False))

        self.model_path_widget.connect('currentPathChanged(const QString&)', lambda: setattr(self.logic, 'new_model_loaded', True))

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
    
    def setManualPreprocessVis(self, visible):
        self.ui.lblLevel.setVisible(visible)
        self.ui.lblWidth.setVisible(visible)
        self.ui.sldWinLevel.setVisible(visible)
        self.ui.sldWinWidth.setVisible(visible)
    

    def selectParameterNode(self):
        # Select parameter set node if one is found in the scene, and create one otherwise
        segmentEditorSingletonTag = "SegmentEditor"
        segmentEditorNode = slicer.mrmlScene.GetSingletonNode(segmentEditorSingletonTag, "vtkMRMLSegmentEditorNode")
        if segmentEditorNode is None:
            segmentEditorNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSegmentEditorNode")
            segmentEditorNode.UnRegister(None)
            segmentEditorNode.SetSingletonTag(segmentEditorSingletonTag)
            segmentEditorNode = slicer.mrmlScene.AddNode(segmentEditorNode)
        if self.parameterSetNode == segmentEditorNode:
            # nothing changed
            return
        self.parameterSetNode = segmentEditorNode
        self.editor.setMRMLSegmentEditorNode(self.parameterSetNode)
    

    def _createAndAttachROI(self):
        # Make sure there is only one 'R'
        if self.logic.volume_node is None:
            self.logic.volume_node = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')[0]
        volumeNode = self.logic.volume_node

        # Create a new ROI that will be fit to volumeNode
        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "R")

        cropVolumeParameters = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLCropVolumeParametersNode")
        cropVolumeParameters.SetInputVolumeNodeID(volumeNode.GetID())
        cropVolumeParameters.SetROINodeID(roiNode.GetID())
        slicer.modules.cropvolume.logic().SnapROIToVoxelGrid(cropVolumeParameters)  # optional (rotates the ROI to match the volume axis directions)
        slicer.modules.cropvolume.logic().FitROIToInputVolume(cropVolumeParameters)
        slicer.mrmlScene.RemoveNode(cropVolumeParameters)

        self.scaleROI(.85)

        self.ui.widgetROI.setMRMLMarkupsNode(slicer.util.getNode("R"))
        self.updateAllParameters()


    
    def scaleROI(self, ratio):
        # Make sure there is exactly one 'R'
        roiNode = slicer.util.getNode('R')
        roi_size = roiNode.GetSize()
        roiNode.SetSize(int(roi_size[0] * ratio), int(roi_size[1] * ratio), int(roi_size[2] * ratio))
    
    def makeROI2D(self):
        # Make sure there is exactly one 'R'
        roiNode = slicer.util.getNode('R')
        roi_size = roiNode.GetSize()
        roiNode.SetSize(roi_size[0], roi_size[1], 1)
        roi_center = np.array(roiNode.GetCenter())
        roiNode.SetCenter([roi_center[0], roi_center[1], slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceOffset()])
    
    def setROIboundary(self, lower):
        roiNode = slicer.util.getNode('R')
        
        bounds = np.zeros(6)
        roiNode.GetBounds(bounds)
        p1 = bounds[::2]
        p2 = bounds[1::2]

        curr_slice = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceOffset()

        center = (p2[2] + curr_slice)/2 if lower else (p1[2] + curr_slice)/2
        depth = p2[2] - curr_slice if lower else curr_slice - p1[2]

        roi_center = np.array(roiNode.GetCenter())
        roiNode.SetCenter([roi_center[0], roi_center[1], center])
        
        roi_size = roiNode.GetSize()
        roiNode.SetSize(roi_size[0], roi_size[1], depth + 1)

    
    def toggleLocalInstall(self, checkbox, file_selector):
        import ctk
        file_selector.currentPath = ''
        if checkbox.isChecked():
            file_selector.filters = ctk.ctkPathLineEdit.Files
            file_selector.nameFilters = ['server_essentials.zip']
        else:
            file_selector.filters = ctk.ctkPathLineEdit.Dirs


    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self) -> None:
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None

    def onSceneStartClose(self, caller, event) -> None:
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode: Optional[MedSAMLiteParameterNode]) -> None:
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
            self.renderAllParameters()
    
    
    def renderAllParameters(self):
        if self._parameterNode.modelPath:
            self.model_path_widget.currentPath = self._parameterNode.modelPath
        if self._parameterNode.roiNode:
            self.ui.widgetROI.setMRMLMarkupsNode(slicer.util.getNode("R"))
        if self._parameterNode.segmentationNode:
            self.logic.segment_res_group = self._parameterNode.segmentationNode
        

        if self._parameterNode.prepMethod:
            self.ui.cmbPrepOptions.currentText = self._parameterNode.prepMethod
        if self._parameterNode.prepWinLevel:
            self.ui.sldWinLevel.value = self._parameterNode.prepWinLevel
        if self._parameterNode.prepWinWidth:
            self.ui.sldWinWidth.value = self._parameterNode.prepWinWidth
    

    def updateAllParameters(self):
        self._parameterNode.modelPath = self.model_path_widget.currentPath
        try:
            self._parameterNode.roiNode = slicer.util.getNode('R')
        except:
            pass
        self._parameterNode.segmentationNode = self.logic.segment_res_group
        self._parameterNode.prepMethod = self.ui.cmbPrepOptions.currentText
        self._parameterNode.prepWinLevel = self.ui.sldWinLevel.value
        self._parameterNode.prepWinWidth = self.ui.sldWinWidth.value
            


#
# MedSAMLiteLogic
#

class MedSAMLiteLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """
    image_data = None
    segment_res_group = None
    server_ready = False
    server_process = None
    volume_node = None
    timer = None
    progressbar = None
    server_dir = None
    widget = None
    new_model_loaded = True
    backend = None

    def __init__(self) -> None:
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return MedSAMLiteParameterNode(super().getParameterNode())

    
    def installTorch(self, version):
        # Install PyTorch
        try:
          import PyTorchUtils
        except ModuleNotFoundError as e:
          raise RuntimeError("This module requires PyTorch extension. Install it from the Extensions Manager.")

        minTorch, minTorchVision = version.split(' ')
        minTorch, minTorchVision = minTorch.split('==')[1], minTorchVision.split('==')[1]

        torchLogic = PyTorchUtils.PyTorchUtilsLogic()
        if not torchLogic.torchInstalled():
            print('PyTorch Python package is required. Installing... (it may take several minutes)')
            torch = torchLogic.installTorch(askConfirmation=True, torchVersionRequirement = f">={minTorch}", torchvisionVersionRequirement=f">={minTorchVision}")
            if torch is None:
                raise ValueError('PyTorch extension needs to be installed to use this module.')
        else:
            # torch is installed, check version
            from packaging import version
            if version.parse(torchLogic.torch.__version__) < version.parse(minTorch) or version.parse(torchLogic.torchvisionVersionInformation.split(' ')[-1]) < version.parse(minTorchVision):
                raise ValueError(f'PyTorch version {torchLogic.torch.__version__} or PyTorch Vision version {torchLogic.torchvisionVersionInformation.split(" ")[-1]} is not compatible with this module.'
                                 + f' Minimum required version is Torch v{minTorch} and TorchVision v{minTorchVision}. You can use "PyTorch Util" module to install PyTorch'
                                 + f' with version requirement set to: >={minTorch} and >={minTorchVision} for Torch and TorchVision respectively.')
    
    def pip_install_wrapper(self, command, event):
        slicer.util.pip_install(command)
        event.set()
    
    def download_wrapper(self, url, filename, download_needed, event):
        if download_needed:
            with urlopen(url) as r:
                with open(filename, "wb") as f:
                    while True:
                        chunk = r.read(1024)
                        if chunk is None:
                            continue
                        elif chunk == b"":
                            break
                        f.write(chunk)
            
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(filename))
        
        event.set()
    
    def install_dependencies(self):
        dependencies = {
            'PyTorch': 'torch==2.0.1 torchvision==0.15.2',
            'MedSam Lite Server': '-e "%s"'%(self.server_dir)
        }

        for dependency in dependencies:
            if 'torch==' in dependencies[dependency]:
                self.installTorch(dependencies[dependency])
            else:
                self.run_on_background(self.pip_install_wrapper, (dependencies[dependency],), 'Installing dependencies: %s'%dependency)
    

    def upgrade(self, download, event):
        try:
            self.progressbar.setLabelText('Checking for updates...')
            latest_version_req = requests.get('https://github.com/bowang-lab/MedSAMSlicer/releases/latest')
            latest_version = latest_version_req.url.split('/')[-1]
            latest_version = float(latest_version[1:])
            curr_version = float(MEDSAMLITE_VERSION[1:])
            print('Latest version identified:', latest_version)
            print('Current version is:', curr_version)
            if latest_version > curr_version:
                print('Upgrade available')
                github_base = 'https://raw.githubusercontent.com/bowang-lab/MedSAMSlicer/v%.2f/'%latest_version
                server_url = github_base + 'server/server.py'
                module_url = github_base + 'MedSAM/MedSAMLite/MedSAMLite.py'
                ui_url = github_base + 'MedSAM/MedSAMLite/Resources/UI/MedSAMLite.ui'

                server_file_path = os.path.join(self.server_dir, 'server.py')
                module_file_path = __file__
                ui_file_path = os.path.join(os.path.dirname(__file__), 'Resources/UI/MedSAMLite.ui')

                self.progressbar.setLabelText('Downloading updates...')
                server_req = requests.get(server_url)
                module_req = requests.get(module_url)
                ui_req = requests.get(ui_url)

                with open(server_file_path, 'w') as server_file:
                    server_file.write(server_req.text)
                with open(module_file_path, 'w') as module_file:
                    module_file.write(module_req.text)
                with open(ui_file_path, 'w') as ui_file:
                    ui_file.write(ui_req.text)
                self.progressbar.setLabelText('Upgraded successfully, please restart Slicer.')

            else:
                self.progressbar.setLabelText('Already using the latest version')
        except:
            self.progressbar.setLabelText('Error happened while upgrading')
        
        time.sleep(3)

        event.set()        
    
    
    def run_on_background(self, target, args, title, progress_check=None):
        self.progressbar = slicer.util.createProgressDialog(autoClose=False)
        self.progressbar.minimum = 0
        self.progressbar.maximum = 0 if progress_check is None else 100
        self.progressbar.setLabelText(title)

        if progress_check is not None:
            self.timer = QTimer()
            self.timer.timeout.connect(progress_check)
            self.timer.start(1000)
        
        parallel_event = threading.Event()
        dep_thread = threading.Thread(target=target, args=(*args, parallel_event,))
        dep_thread.start()
        while not parallel_event.is_set():
            slicer.app.processEvents()
        dep_thread.join()

        self.progressbar.close()
    
    def run_server(self):
        #FIXME show that 'Backend is loading...'
        self.backend = MedSAM_Interface()
        self.widget.renderAllParameters()
        self.backend.MedSAM_CKPT_PATH = self.widget.model_path_widget.currentPath
        self.backend.load_model()
        self.server_ready = True
    
    def progressCheck(self, partial=False):
        slicer.app.processEvents()
        progress_data = self.backend.get_progress()
        self.progressbar.value = progress_data['generated_embeds']

        if progress_data['layers'] <= progress_data['generated_embeds']:
            self.progressbar.close()
            self.timer.stop()
            self.widget.ui.pbSegment.setEnabled(True)
            if partial:
                segmentation_mask = self.inferSegmentation()
                self.showSegmentation(segmentation_mask)
                self.widget.ui.pbSegment.setText('Single Segmentation')
            else:
                self.widget.ui.pbSegment.setText('Segmentation')

    def volumeChanged(self, node=None):
        self.widget.ui.pbSegment.setText('Single Segmentation')


    def captureImage(self):
        ######## Set your image path here
        self.volume_node = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')[0]
        self.image_data = slicer.util.arrayFromVolume(self.volume_node)  ################ Only one node?
    
    def sendImage(self, partial=False):
        self.widget.ui.pbSegment.setEnabled(False)
        self.widget.ui.pbSegment.setText('Sending image, please wait...')

        if self.new_model_loaded or not self.server_ready:
            self.run_server()
            self.new_model_loaded = False
        
        ############ Partial segmentation
        if partial:
            _, _, zrange = self.get_bounding_box()
            zmin, zmax = zrange
        else:
            zmin, zmax = -1, -1

        self.captureImage()
        
        self.progressbar = slicer.util.createProgressDialog(autoClose=False)
        self.progressbar.minimum = 0
        self.progressbar.maximum = 100
        self.progressbar.setLabelText("Preparing image embeddings...")

        self.timer = QTimer()
        self.timer.timeout.connect(lambda: self.progressCheck(partial))
        self.timer.start(1000)

        self.backend.speed_level = 1 if 'Normal' in self.widget.ui.cmbSpeed.currentText else 2 if 'Faster' in self.widget.ui.cmbSpeed.currentText else 3
        self.backend.set_image(self.image_data, -160, 240, zmin, zmax, recurrent_func=slicer.app.processEvents)

        self.widget.updateAllParameters()
        # self.run_on_background(self.embedding_prep_wrapper, (self.image_data, -160, 240, zmin, zmax), "Preparing image embeddings...", lambda: self.progressCheck(partial))
    
    def embedding_prep_wrapper(self, arr, wmin, wmax, zmin, zmax, event):
        print('before sending image')
        self.backend.set_image(arr, wmin, wmax, zmin, zmax)
        print('after sending image')
        event.set()
    
    def inferSegmentation(self):
        print('sending infer request...')
        ################ DEBUG MODE ################
        if self.volume_node is None:
            self.captureImage()
        ################ DEBUG MODE ################

        slice_idx, bbox, zrange = self.get_bounding_box()
        seg_data = self.backend.infer(slice_idx, bbox, zrange)
        frames = list(seg_data.keys())
        seg_result = np.zeros_like(self.image_data)
        for frame in frames:
            seg_result[frame, :, :] = seg_data[frame]

        return seg_result
    
    def showSegmentation(self, segmentation_mask):
        segment_volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", 'segment_'+str(int(time.time())))
        slicer.util.updateVolumeFromArray(segment_volume, segmentation_mask)
        
        current_seg_group = self.widget.editor.segmentationNode()
        if current_seg_group is None:
            if self.segment_res_group is None:
                self.segment_res_group = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
                self.segment_res_group.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)
            current_seg_group = self.segment_res_group

        try:
            check_if_node_is_removed = slicer.util.getNode(current_seg_group.GetID()) # if scene is closed and reopend, this line will raise an error
        except:
            self.segment_res_group = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            self.segment_res_group.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)
            current_seg_group = self.segment_res_group
        

        current_seg_group.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(segment_volume, current_seg_group)
        slicer.util.updateSegmentBinaryLabelmapFromArray(segmentation_mask, current_seg_group, segment_volume.GetName(), self.volume_node)

        slicer.mrmlScene.RemoveNode(segment_volume)

        self.widget.updateAllParameters()
    
    def singleSegmentation(self):
        self.sendImage(partial=True)

    
    def applySegmentation(self):
        if self.widget.ui.pbSegment.text == 'Single Segmentation':
            continueSingle = QMessageBox.question(None,'', "You are using single segmentation option which is faster but is not advised if you want large or multiple regions be segmented in one image. In that case click 'Send Image' button. Do you wish to continue with single segmentation?", QMessageBox.Yes | QMessageBox.No)
            if continueSingle == QMessageBox.No: return
            self.singleSegmentation()

            return
        segmentation_mask = self.inferSegmentation()
        self.showSegmentation(segmentation_mask)
    
    def get_bounding_box(self):
        self.captureImage()
        roiNode = slicer.util.getNode("R") # multiple bounding box?

        # If volume node is transformed, apply that transform to get volume's RAS coordinates
        transformRasToVolumeRas = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, self.volume_node.GetParentTransformNode(), transformRasToVolumeRas)

        bounds = np.zeros(6)
        roiNode.GetBounds(bounds)
        p1 = bounds[::2]
        p2 = bounds[1::2]

        ijk_points = []

        for curr_point in [p1, p2]:
            # Get point coordinate in RAS
            point_VolumeRas = transformRasToVolumeRas.TransformPoint(curr_point)

            # Get voxel coordinates from physical coordinates
            volumeRasToIjk = vtk.vtkMatrix4x4()
            self.volume_node.GetRASToIJKMatrix(volumeRasToIjk)
            point_Ijk = [0, 0, 0, 1]
            volumeRasToIjk.MultiplyPoint(np.append(point_VolumeRas,1.0), point_Ijk)
            point_Ijk = [ int(round(c)) for c in point_Ijk[0:3] ]

            ijk_points.append(point_Ijk)

        x_min, x_max = min(ijk_points[0][0], ijk_points[1][0]), max(ijk_points[0][0], ijk_points[1][0])
        y_min, y_max = min(ijk_points[0][1], ijk_points[1][1]), max(ijk_points[0][1], ijk_points[1][1])
        z_min, z_max = min(ijk_points[0][2], ijk_points[1][2]), max(ijk_points[0][2], ijk_points[1][2])
        bbox = [x_min, y_min, x_max, y_max]
        zrange = [z_min, z_max]
        slice_idx = int((zrange[0] + zrange[1]) / 2) # it is not accurate

        return slice_idx, bbox, zrange
    
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

        self.widget.updateAllParameters()


#
# MedSAMLiteTest
#

class MedSAMLiteTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_MedSAMLite1()

    def test_MedSAMLite1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
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
        slicer.util.loadVolume('/home/rasakereh/Desktop/wanglab/MedSam/slicer-plugin/MedSAM-Slicer/HCC_004_0000.nii.gz')
        logic = MedSAMLiteLogic()
        logic.sendImage()
        input()
        logic.inferSegmentation()
        return

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('MedSAMLite1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = MedSAMLiteLogic()

        # Test algorithm with non-inverted threshold
        # logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        # logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
