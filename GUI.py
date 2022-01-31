import sys
from PIL import Image
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtWidgets import QPushButton, QLineEdit, QTextBrowser
from PyQt5.QtWidgets import QFileDialog, QFrame, QCheckBox
from PyQt5.QtGui import QIcon, QPixmap, QFont, QImage
from PyQt5.QtCore import Qt
import Models
import Visual

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.DepthModel = Models.LoadAdabins()
        self.InitUi()
        self.show()

    def InitUi(self):
        self.setGeometry(0, 0, 1181, 719)
        self.setWindowTitle("Sharif Vision")
        self.setWindowIcon(QIcon('./Images/Icon.svg'))
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)

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
        self.MessageBox.append("Object Detection and Depth Estimation Apllication [Version 1.0.0]")
        self.MessageBox.append("Developed by: Sina Nabigol & Amin Dehnavi. (c) All Rights Reserved.")

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
        self.FilePathLineEdit.setStyleSheet("background-color: rgb(255, 255, 255);")

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
        self.SavePathLineEdit.setStyleSheet("background-color: rgb(255, 255, 255);")

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

        self.DepthImageCheck = QCheckBox(self)
        self.DepthImageCheck.setGeometry(680, 250, 131, 20)
        self.DepthImageCheck.setFont(MainFont)
        self.DepthImageCheck.setText("Depth Image")
        self.BoundingBoxCheck = QCheckBox(self)

        self.BoundingBoxCheck.setGeometry(680, 280, 141, 20)
        self.BoundingBoxCheck.setFont(MainFont)
        self.BoundingBoxCheck.setText("Bounding Boxes")
        self.DepthInfoCheck = QCheckBox(self)

        self.DepthInfoCheck.setGeometry(680, 310, 161, 20)
        self.DepthInfoCheck.setFont(MainFont)
        self.DepthInfoCheck.setText("Depth Information")

        self.ApplyButton = QPushButton(self)
        self.ApplyButton.setGeometry(1080, 680, 91, 28)
        self.ApplyButton.setText("Apply")
        self.ApplyButton.setEnabled(False)
        self.ApplyButton.clicked.connect(self.ApplyRoutine)

    def FilePathRoutine(self):
        self.MessageBox.append('\n>>> FilePath')
        FileName, _ = QFileDialog.getOpenFileName(self, 'Select image', './' , 'Image files (*.png *.jpg)')
        if FileName == '':
            self.MessageBox.append('No Image Selected!')
            return
        self.FilePathLineEdit.setText(FileName)
        self.RawImage = Image.open(FileName)
        self.RawImage = np.asarray(self.RawImage)
        self.Image = self.RawImage
        self.ShowImage(self.Image)
        self.PredictButton.setEnabled(True)
        if( len(self.FilePathLineEdit.text())>1 and len(self.SavePathLineEdit.text())>1 ):
            self.SaveButton.setEnabled(True)
        self.MessageBox.append('Image Loaded')

    def SavePathRoutine(self):
        self.MessageBox.append('\n>>> SavePath')
        FolderName = QFileDialog.getExistingDirectory(self, 'Select folder', './')
        if FolderName == '':
            self.MessageBox.append('No Folder Selected!')
            return
        self.SavePathLineEdit.setText(FolderName)
        if( len(self.FilePathLineEdit.text())>1 and len(self.SavePathLineEdit.text())>1 ):
            self.SaveButton.setEnabled(True)
        self.MessageBox.append('Ok')

    def SaveRoutine(self):
        self.MessageBox.append('\n>>> Saving')
        img = Image.fromarray(self.Image)
        img.save(self.SavePathLineEdit.text() + f"/Image_{self.SaveCounter}.png")
        self.SaveCounter += 1
        self.MessageBox.append('Ok')
        
    def PredictRoutine(self):
        self.MessageBox.append('\n>>> Predict')
        self.Depth = Models.PredictDepth(self.DepthModel, self.RawImage)
        self.ApplyButton.setEnabled(True)
        self.MessageBox.append('Ok')
          
    def ResetRoutine(self):
        self.MessageBox.append('\n>>> Reset')
        self.FilePathLineEdit.setText(' ')
        self.SavePathLineEdit.setText(' ')
        self.ImageFrame.setText("Please load an image!")
        self.PredictButton.setEnabled(False)
        self.SaveButton.setEnabled(False)
        self.ApplyButton.setEnabled(False)
        self.MessageBox.append('Ok')
    
    def ApplyRoutine(self):
        self.MessageBox.append('\n>>> Apply')
        if self.DepthImageCheck.isChecked():
            depthMap = Visual.DepthMap(self.Depth)
            self.Image = depthMap
        else:
            self.Image = self.RawImage
        self.ShowImage(self.Image)
        self.MessageBox.append('Ok')

    def Numpy2Qt(self, Image):
        return QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
    
    def ShowImage(self, Image):
        Image = QPixmap.fromImage(self.Numpy2Qt(Image))
        Image = Image.scaled(640, 480)
        self.ImageFrame.setPixmap(Image)
        


if __name__  == '__main__':
	App = QApplication(sys.argv)
	GUI = MainWindow()
	sys.exit(App.exec())
