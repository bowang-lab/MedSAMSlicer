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
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from urllib.request import urlopen

from slicer import vtkMRMLScalarVolumeNode

from PythonQt.QtCore import QTimer, QByteArray
from PythonQt.QtGui import QIcon, QPixmap

try:
    from numpysocket import NumpySocket
except:
    pass # no installation anymore, shorter plugin load

MEDSAMLITE_VERSION = 'v0.03'

#
# MedSAMLite
#

class MedSAMLite(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "MedSAMLite"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Segmentation"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Reza Asakereh", "Andrew Qiao", "Jun Ma"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#MedSAMLite">module documentation</a>.
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

        DEPENDENCIES_AVAILABLE = False

        # Initial Dependency Setup
        if self.is_setting_available():
            try:
                from segment_anything.modeling import MaskDecoder
                DEPENDENCIES_AVAILABLE = True
            except:
                DEPENDENCIES_AVAILABLE = False
        else:
            DEPENDENCIES_AVAILABLE = False

        if not DEPENDENCIES_AVAILABLE:
            from PythonQt.QtGui import QLabel, QPushButton, QSpacerItem, QSizePolicy, QCheckBox
            import ctk
            path_instruction = QLabel('Choose a folder to install module dependencies in')
            restart_instruction = QLabel('Restart 3D Slicer after all dependencies are installed!')

            ctk_install_path = ctk.ctkPathLineEdit()
            ctk_install_path.filters = ctk.ctkPathLineEdit.Dirs
            
            local_install = QCheckBox("Install from local server_essentials.zip")
            local_install.toggled.connect(lambda:self.toggleLocalInstall(local_install, ctk_install_path))

            install_btn = QPushButton('Install dependencies')
            install_btn.clicked.connect(lambda: self.logic.install_dependencies(ctk_install_path))

            self.layout.addWidget(path_instruction)
            self.layout.addWidget(local_install)
            self.layout.addWidget(ctk_install_path)
            self.layout.addWidget(install_btn)
            self.layout.addWidget(restart_instruction)
            
            return
        
        self.logic.server_dir = os.path.join(self.read_setting(), 'server_essentials')

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/MedSAMLite.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)


        ############################################################################
        # Segmentation Module
        import qSlicerSegmentationsModuleWidgetsPythonQt
        self.editor = qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget()
        self.editor.setMaximumNumberOfUndoStates(10)
        self.selectParameterNode()
        self.editor.setMRMLScene(slicer.mrmlScene)
        # print(self.ui.clbtnOperation.layout().__dict__)
        self.ui.clbtnOperation.layout().addWidget(self.editor)
        # self.layout.addWidget(self.editor)
        # self.editor.currentSegmentIDChanged.connect(print)
        ############################################################################

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
        self.ui.pbSendImage.connect('clicked(bool)', lambda: self.logic.sendImage())
        self.ui.pbSegment.connect('clicked(bool)', lambda: self.logic.applySegmentation())

        self.ui.pbCTprep.setIcon(QIcon(os.path.join(self.logic.server_dir, 'CT.jpg')))
        self.ui.pbCTprep.connect('clicked(bool)', lambda: self.logic.applyPreprocess(self.logic.preprocess_CT))
        self.ui.pbMRprep.setIcon(QIcon(os.path.join(self.logic.server_dir, 'MR.png')))
        self.ui.pbMRprep.connect('clicked(bool)', lambda: self.logic.applyPreprocess(self.logic.preprocess_MR))
        
        self.ui.pbAttach.connect('clicked(bool)', lambda: self._createAndAttachROI())
        self.ui.pbTwoDim.connect('clicked(bool)', lambda: self.makeROI2D())

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
    

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

        self.ui.widgetROI.setMRMLMarkupsNode(slicer.util.getNode("R"))
    
    def makeROI2D(self):
        # Make sure there is exactly one 'R'
        roiNode = slicer.util.getNode('R')
        roi_size = roiNode.GetSize()
        roiNode.SetSize(roi_size[0], roi_size[1], 1)
        roi_center = np.array(roiNode.GetCenter())
        roiNode.SetCenter([roi_center[0], roi_center[1], slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceOffset()])
    
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

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

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
        
    def is_setting_available(self):
        if not (os.path.isfile('.medsam_info') or os.path.isfile(os.path.expanduser('~/.medsam_info'))):
            return False

        setting_file = '.medsam_info' if os.path.isfile('.medsam_info') else os.path.expanduser('~/.medsam_info')
        server_file = os.path.join(self.read_setting(), 'server_essentials/server.py')

        return os.path.isfile(server_file)
    
    def read_setting(self):
        setting_file = '.medsam_info' if os.path.isfile('.medsam_info') else os.path.expanduser('~/.medsam_info')
        with open(setting_file, 'r') as settings:
            server_essentials_root = settings.read()
            return server_essentials_root

    def write_setting(self, setting):
        try:
            with open('.medsam_info', 'w') as settings:
                settings.write(setting)
        except:
            with open(os.path.expanduser('~/.medsam_info'), 'w') as settings:
                settings.write(setting)


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

    def __init__(self) -> None:
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return MedSAMLiteParameterNode(super().getParameterNode())

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            'InputVolume': inputVolume.GetID(),
            'OutputVolume': outputVolume.GetID(),
            'ThresholdValue': imageThreshold,
            'ThresholdType': 'Above' if invert else 'Below'
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')
    
    def pip_install_wrapper(self, command, event):
        slicer.util.pip_install(command)
        event.set()
    
    def download_wrapper(self, url, filename, download_needed, event):
        if download_needed:
            with urlopen(url) as r:
                # self.setTotalProgress.emit(int(r.info()["Content-Length"]))
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
    
    def install_dependencies(self, ctk_path):
        if ctk_path.currentPath == '':
            print('Installation path is empty')
            return
        
        if os.path.isfile(ctk_path.currentPath) and os.path.basename(ctk_path.currentPath) == 'server_essentials.zip':
            install_path = os.path.abspath(os.path.dirname(ctk_path.currentPath))
            download_needed = False
        elif os.path.isdir(ctk_path.currentPath):
            install_path = ctk_path.currentPath
            download_needed = True
        else:
            print('Invalid installation path')
            return
        
        print('Installation will happen in %s'%install_path)
        
        self.widget.write_setting(install_path)

        file_url = 'https://github.com/rasakereh/medsam-3dslicer/raw/master/server_essentials.zip'
        filename = os.path.join(install_path, 'server_essentials.zip')
        
        self.run_on_background(self.download_wrapper, (file_url, filename, download_needed), 'Downloading additional files...')

        self.server_dir = os.path.join(install_path + '/', 'server_essentials')

        dependencies = {
            'PyTorch': 'torch==2.0.1 torchvision==0.15.2',
            'Numpy Socket': 'numpysocket',
            'FastAPI': 'fastapi',
            'Uvicorn': 'uvicorn',
            'MedSam Lite Server': '-e "%s"'%(self.server_dir)
        }

        for dependency in dependencies:
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

                self.progressbar.setLabelText('Downloading updates...')
                server_req = requests.get(server_url)
                module_req = requests.get(module_url)

                with open(os.path.join(self.server_dir, 'server.py'), 'w') as server_file:
                    server_file.write(server_req.text)
                with open(__file__, 'w') as module_file:
                    module_file.write(module_req.text)
                self.progressbar.setLabelText('Upgraded successfully, please restart Slicer.')

            else:
                self.progressbar.setLabelText('Already using the latest version')
        except:
            self.progressbar.setLabelText('Error happened while upgrading')
        
        time.sleep(3)

        event.set()
    
    
    def run_on_background(self, target, args, title):
        self.progressbar = slicer.util.createProgressDialog(autoClose=False)
        self.progressbar.minimum = 0
        self.progressbar.maximum = 0
        self.progressbar.setLabelText(title)
        
        pip_install_event = threading.Event()
        dep_thread = threading.Thread(target=target, args=(*args, pip_install_event,))
        dep_thread.start()
        while not pip_install_event.is_set():
            slicer.app.processEvents()
        dep_thread.join()

        self.progressbar.close()
    
    def run_server(self):
        print('Running server...')
        
        # buggy_file_path = os.getcwd() + '/lib/Python/lib/python3.9/site-packages/typing_extensions.py'

        # with open(buggy_file_path, 'r') as file:
        #     lines = file.readlines()

        # # Update the value in line 173
        # buggy_line_num = 173
        # new_value = '\t\t\tt, (typing._GenericAlias, _types.GenericAlias, _types.UnionType)'

        # if 1 <= buggy_line_num <= len(lines):
        #     lines[buggy_line_num - 1] = f"{new_value}\n"

        # # Write the updated content back to the file
        # with open(buggy_file_path, 'w') as file:
        #     file.writelines(lines)
        

        self.server_process = subprocess.Popen(['PythonSlicer', os.path.join(self.server_dir, 'server.py')])#, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, start_new_session=True)
        def cleanup():
            timeout_sec = 5

            p_sec = 0
            for second in range(timeout_sec):
                if self.server_process.poll() == None:
                    time.sleep(1)
                    p_sec += 1
            if p_sec >= timeout_sec:
                self.server_process.kill()
        # atexit.register(cleanup)

        self.server_ready = True

        time.sleep(4) #Change

    
    def progressCheck(self, serverUrl='http://127.0.0.1:5555'):
        response = requests.post(f'{serverUrl}/getProgress')
        progress_data = json.loads(response.json())
        self.progressbar.value = progress_data['generated_embeds']
        # slicer.app.processEvents()

        if int(progress_data['layers']) == int(progress_data['generated_embeds']):
            self.progressbar.close()
            self.timer.stop()

    
    def captureImage(self):
        ######## Set your image path here
        self.volume_node = slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')[0]
        self.img_path = self.volume_node.GetStorageNode().GetFullNameFromFileName()
        self.img_sitk = sitk.ReadImage(self.img_path)
        self.image_data = slicer.util.arrayFromVolume(self.volume_node)  ################ Only one node?
    
    def sendImage(self, serverUrl='http://127.0.0.1:5555', numpyServerAddress=("127.0.0.1", 5556)):
        if not self.server_ready:
            self.run_server()
        print('sending setImage request...')
        response = requests.post(f'{serverUrl}/setImage', json={"wmin": -160, "wmax": 240}) # wmin, wmax as input?
        print('Response from setImage:', response.text)

        self.captureImage()
        with NumpySocket() as s:
            s.connect(numpyServerAddress)
            print("sending numpy array:")
            s.sendall(self.image_data)
        
        ###########################
        # Timer
        if self.timer is None:
            self.timer = QTimer()
            self.timer.timeout.connect(self.progressCheck)
        ###########################
        self.progressbar = slicer.util.createProgressDialog(autoClose=False)
        self.progressbar.minimum = 0
        self.progressbar.maximum = self.image_data.shape[0]
        self.progressbar.setLabelText('Preparing image embeddings...')

        self.timer.start(1000)

    def inferSegmentation(self, serverUrl='http://127.0.0.1:5555'):
        print('sending infer request...')
        ################ DEBUG MODE ################
        if self.volume_node is None:
            self.captureImage()
        ################ DEBUG MODE ################

        slice_idx, bbox, zrange = self.get_bounding_box()

        response = requests.post(f'{serverUrl}/infer', json={"slice_idx": slice_idx, "bbox": bbox, "zrange": zrange})
        response.raise_for_status()
        seg_data = json.loads(response.json())
        frames = sorted(list(map(int, seg_data.keys())))
        seg_result = np.zeros_like(self.image_data)
        for frame in frames:
            seg_result[frame, :, :] = seg_data[str(frame)]

        return seg_result
    
    def showSegmentation(self, segmentation_mask):
        segmentation_res_file = os.path.dirname(self.img_path) + '/lite_seg_' + os.path.basename(self.img_path)
        seg_sitk = sitk.GetImageFromArray(segmentation_mask)
        seg_sitk.CopyInformation(self.img_sitk)
        sitk.WriteImage(seg_sitk, segmentation_res_file) ########## Set your segmentation output here
        loaded_seg_file = slicer.util.loadSegmentation(segmentation_res_file)

        segment_volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(loaded_seg_file, segment_volume, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY)

        current_seg_group = self.widget.editor.segmentationNode()
        if current_seg_group is None:
            if self.segment_res_group is None:
                self.segment_res_group = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
                self.segment_res_group.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)
            current_seg_group = self.segment_res_group

        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(segment_volume, current_seg_group)
        slicer.mrmlScene.RemoveNode(segment_volume)
        slicer.mrmlScene.RemoveNode(loaded_seg_file)
    
    def applySegmentation(self, serverUrl='http://127.0.0.1:5555'):
        segmentation_mask = self.inferSegmentation(serverUrl)
        self.showSegmentation(segmentation_mask)
    
    def get_bounding_box(self):
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
        # self.volume_node.GetDisplayNode().SetThreshold(0, 255)
        # self.volume_node.GetDisplayNode().ApplyThresholdOn()
        
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
    
    def applyPreprocess(self, method):
        self.updateImage(method())


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
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
