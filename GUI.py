import sys
from PIL import Image
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QSlider
from PyQt5.QtWidgets import QPushButton, QLineEdit, QTextBrowser
from PyQt5.QtWidgets import QFileDialog, QFrame, QCheckBox, QRadioButton
from PyQt5.QtGui import QIcon, QPixmap, QFont, QImage
from PyQt5.QtCore import Qt
import Models
import Visual


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # Load Pretrained Models at the startup
        self.DepthModel = Models.LoadAdabins()
        self.YOLOModel, self.YOLOModel_exp = Models.LoadYOLOX()

        self.DepthThreshold = 10.0
        self.MinConfidence = 50
        self.RawImage = 0
        self.RawImageSize = (640, 480)
        self.Depth = 0
        self.BoundingBox = 0
        self.RadioButton = 'Raw Image'
        self.InitUi()
        self.show()

    def InitUi(self):
        self.setGeometry(0, 0, 1181, 719)
        self.setWindowTitle("Sharif Deep Vision")
        self.setWindowIcon(QIcon('./Images/logo.png'))
        self.setWindowFlags(Qt.WindowCloseButtonHint |
                            Qt.WindowMinimizeButtonHint)

        ImageFrameFont = QFont()
        ImageFrameFont.setFamily("Times New Roman")
        ImageFrameFont.setPointSize(11)
        ImageFrameFont.setBold(True)
        ImageFrameFont.setItalic(True)
        ImageFrameFont.setWeight(75)

        MainFont = QFont()
        MainFont.setFamily("Times New Roman")
        MainFont.setPointSize(10)

        self.ImageFrame = QLabel(self)
        self.ImageFrame.setGeometry(10, 20, 640, 480)
        self.ImageFrame.setFont(ImageFrameFont)
        self.ImageFrame.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.ImageFrame.setFrameShape(QFrame.Panel)
        self.ImageFrame.setAlignment(Qt.AlignCenter)
        self.ImageFrame.setText("Please load an image!")

        self.MessageBox = QTextBrowser(self)
        self.MessageBox.setGeometry(10, 520, 641, 192)
        self.MessageBox.append(
            "Object Detection and Depth Estimation Apllication [Version 1.0.0]")
        self.MessageBox.append(
            "Developed by: Sina Nabigol & Amin Dehnavi. (c) All Rights Reserved.")

        self.VerticalLine = QFrame(self)
        self.VerticalLine.setGeometry(650, 20, 20, 691)
        self.VerticalLine.setFrameShape(QFrame.VLine)
        self.VerticalLine.setFrameShadow(QFrame.Sunken)

        self.FilePathLabel = QLabel(self)
        self.FilePathLabel.setGeometry(670, 33, 71, 21)
        self.FilePathLabel.setFont(MainFont)
        self.FilePathLabel.setText("File Path")

        self.FilePathLineEdit = QLineEdit(self)
        self.FilePathLineEdit.setGeometry(750, 30, 381, 28)
        self.FilePathLineEdit.setEnabled(False)
        self.FilePathLineEdit.setStyleSheet(
            "background-color: rgb(255, 255, 255);")

        self.FilePathButton = QPushButton(self)
        self.FilePathButton.setGeometry(1140, 30, 31, 28)
        self.FilePathButton.setText("...")
        self.FilePathButton.clicked.connect(self.FilePathRoutine)

        self.SavePathLabel = QLabel(self)
        self.SavePathLabel.setGeometry(670, 73, 71, 21)
        self.SavePathLabel.setFont(MainFont)
        self.SavePathLabel.setText("Save Path")

        self.SavePathLineEdit = QLineEdit(self)
        self.SavePathLineEdit.setGeometry(750, 70, 381, 28)
        self.SavePathLineEdit.setEnabled(False)
        self.SavePathLineEdit.setStyleSheet(
            "background-color: rgb(255, 255, 255);")

        self.SavePathButton = QPushButton(self)
        self.SavePathButton.setGeometry(1140, 70, 31, 28)
        self.SavePathButton.setText("...")
        self.SavePathButton.clicked.connect(self.SavePathRoutine)

        self.PredictButton = QPushButton(self)
        self.PredictButton.setGeometry(750, 130, 101, 28)
        self.PredictButton.setText("Predict")
        self.PredictButton.setEnabled(False)
        self.PredictButton.clicked.connect(self.PredictRoutine)

        self.SaveButton = QPushButton(self)
        self.SaveButton.setGeometry(890, 130, 101, 28)
        self.SaveButton.setText("Save")
        self.SaveButton.setEnabled(False)
        self.SaveCounter = 0
        self.SaveButton.clicked.connect(self.SaveRoutine)

        self.ResetButton = QPushButton(self)
        self.ResetButton.setGeometry(1032, 130, 101, 28)
        self.ResetButton.setText("Reset")
        self.ResetButton.clicked.connect(self.ResetRoutine)

        self.OptionsLabel = QLabel(self)
        self.OptionsLabel.setGeometry(890, 190, 61, 19)
        self.OptionsLabel.setFont(ImageFrameFont)
        self.OptionsLabel.setAlignment(Qt.AlignCenter)
        self.OptionsLabel.setText("Options")

        self.HorizontalLeftLine = QFrame(self)
        self.HorizontalLeftLine.setGeometry(670, 190, 211, 20)
        self.HorizontalLeftLine.setFrameShape(QFrame.HLine)
        self.HorizontalLeftLine.setFrameShadow(QFrame.Sunken)

        self.HorizontalRightLine = QFrame(self)
        self.HorizontalRightLine.setGeometry(960, 190, 211, 20)
        self.HorizontalRightLine.setFrameShape(QFrame.HLine)
        self.HorizontalRightLine.setFrameShadow(QFrame.Sunken)

        self.RadioButton1 = QRadioButton(self)
        self.RadioButton1.setGeometry(680, 230, 121, 20)
        self.RadioButton1.setFont(MainFont)
        self.RadioButton1.setText("Raw Image")
        self.RadioButton1.toggled.connect(self.RadioButtonRoutine)

        self.RadioButton2 = QRadioButton(self)
        self.RadioButton2.setGeometry(680, 260, 121, 20)
        self.RadioButton2.setFont(MainFont)
        self.RadioButton2.setText("Depth Image")
        self.RadioButton2.toggled.connect(self.RadioButtonRoutine)

        self.RadioButton3 = QRadioButton(self)
        self.RadioButton3.setGeometry(680, 290, 231, 20)
        self.RadioButton3.setFont(MainFont)
        self.RadioButton3.setText("Raw Image + Bounding Boxes")
        self.RadioButton3.toggled.connect(self.RadioButtonRoutine)

        self.RadioButton4 = QRadioButton(self)
        self.RadioButton4.setGeometry(680, 320, 251, 20)
        self.RadioButton4.setFont(MainFont)
        self.RadioButton4.setText("Depth Image + Bounding Boxes")
        self.RadioButton4.toggled.connect(self.RadioButtonRoutine)

        self.DepthThresholdLabel = QLabel(self)
        self.DepthThresholdLabel.setGeometry(680, 360, 471, 21)
        self.DepthThresholdLabel.setFont(MainFont)

        self.DepthThresholdSlider = QSlider(self)
        self.DepthThresholdSlider.setGeometry(680, 390, 491, 22)
        self.DepthThresholdSlider.setOrientation(Qt.Horizontal)
        self.DepthThresholdSlider.setMaximum(100)
        self.DepthThresholdSlider.setMinimum(0)
        self.DepthThresholdSlider.valueChanged.connect(
            self.DepthThresholdSliderRoutine)

        self.MinConfidenceLabel = QLabel(self)
        self.MinConfidenceLabel.setGeometry(680, 430, 471, 21)
        self.MinConfidenceLabel.setFont(MainFont)

        self.MinConfidenceSlider = QSlider(self)
        self.MinConfidenceSlider.setGeometry(680, 460, 491, 22)
        self.MinConfidenceSlider.setOrientation(Qt.Horizontal)
        self.MinConfidenceSlider.setMaximum(100)
        self.MinConfidenceSlider.setMinimum(0)
        self.MinConfidenceSlider.valueChanged.connect(
            self.MinConfidenceSliderRoutine)

        self.ApplyButton = QPushButton(self)
        self.ApplyButton.setGeometry(1080, 680, 91, 28)
        self.ApplyButton.setText("Apply")
        self.ApplyButton.clicked.connect(self.ApplyRoutine)

        self.EnableOption(False)

    def FilePathRoutine(self):
        self.MessageBox.append('\n>>> FilePath')
        FileName, _ = QFileDialog.getOpenFileName(
            self, 'Select image', './', 'Image files (*.jpg *.jpeg)')
        if FileName == '':
            self.MessageBox.append('No Image Selected!')
            return
        self.FilePathLineEdit.setText(FileName)
        self.RawImage = Image.open(FileName)
        self.RawImageSize = self.RawImage.size
        self.RawImage = self.RawImage.resize((640, 480))
        self.RawImage = np.asarray(self.RawImage)
        self.Image = self.RawImage
        self.ShowImage(self.Image)
        self.PredictButton.setEnabled(True)
        if(len(self.FilePathLineEdit.text()) > 1 and len(self.SavePathLineEdit.text()) > 1):
            self.SaveButton.setEnabled(True)
        self.EnableOption(False)
        self.MessageBox.append('Image Loaded')

    def SavePathRoutine(self):
        self.MessageBox.append('\n>>> SavePath')
        FolderName = QFileDialog.getExistingDirectory(
            self, 'Select folder', './')
        if FolderName == '':
            self.MessageBox.append('No Folder Selected!')
            return
        self.SavePathLineEdit.setText(FolderName)
        if(len(self.FilePathLineEdit.text()) > 1 and len(self.SavePathLineEdit.text()) > 1):
            self.SaveButton.setEnabled(True)
        self.MessageBox.append('Ok')

    def SaveRoutine(self):
        self.MessageBox.append('\n>>> Saving')
        img = Image.fromarray(self.Image)
        img = img.resize(self.RawImageSize)
        img.save(self.SavePathLineEdit.text() +
                 f"/Image_{self.SaveCounter}.jpg")
        self.SaveCounter += 1
        self.MessageBox.append('Ok')

    def PredictRoutine(self):
        self.MessageBox.append('\n>>> Predict')
        self.Depth = Models.PredictDepth(self.DepthModel, self.RawImage)
        self.YOLO_Out, self.Img_Info = Models.PredictBoundingBox(
            self.YOLOModel, self.YOLOModel_exp, self.RawImage, fp16=False, device='cpu')
        self.EnableOption(True)
        self.MessageBox.append('Ok')

    def ResetRoutine(self):
        self.MessageBox.append('\n>>> Reset')
        self.FilePathLineEdit.setText(' ')
        self.SavePathLineEdit.setText(' ')
        self.ImageFrame.setText("Please load an image!")
        self.PredictButton.setEnabled(False)
        self.SaveButton.setEnabled(False)
        self.EnableOption(False)
        self.MessageBox.append('Ok')

    def RadioButtonRoutine(self, _):
        button = self.sender()
        if button.isChecked() == True:
            self.RadioButton = button.text()

    def DepthThresholdSliderRoutine(self):
        self.DepthThreshold = self.DepthThresholdSlider.value()/10
        self.DepthThresholdLabel.setText(
            f"Max. Depth : {self.DepthThreshold} m")

    def MinConfidenceSliderRoutine(self):
        self.MinConfidence = self.MinConfidenceSlider.value()
        self.MinConfidenceLabel.setText(
            f"Min. Confidence : {self.MinConfidence}%")

    def ApplyRoutine(self):
        self.MessageBox.append('\n>>> Apply')
        self.Image = Visual.visualize(np.copy(self.RawImage), np.copy(self.Depth),
                                      self.YOLO_Out.copy(), self.Img_Info, self.DepthThreshold,
                                      self.MinConfidence/100, self.RadioButton)
        self.ShowImage(self.Image)
        self.MessageBox.append('Ok')

    def Numpy2Qt(self, Image):
        return QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)

    def ShowImage(self, Image):
        Image = QPixmap.fromImage(self.Numpy2Qt(Image))
        Image = Image.scaled(640, 480)
        self.ImageFrame.setPixmap(Image)

    def EnableOption(self, state):
        self.RadioButton1.setChecked(True)
        self.RadioButton1.setEnabled(state)
        self.RadioButton2.setEnabled(state)
        self.RadioButton3.setEnabled(state)
        self.RadioButton4.setEnabled(state)

        self.DepthThresholdSlider.setValue(1000)
        self.DepthThresholdLabel.setEnabled(state)
        self.DepthThresholdSlider.setEnabled(state)

        self.MinConfidenceSlider.setValue(50)
        self.MinConfidenceLabel.setEnabled(state)
        self.MinConfidenceSlider.setEnabled(state)

        self.ApplyButton.setEnabled(state)


if __name__ == '__main__':
    App = QApplication(sys.argv)
    GUI = MainWindow()
    sys.exit(App.exec())
