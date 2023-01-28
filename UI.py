# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.test_table = QtWidgets.QTableWidget(self.centralwidget)
        self.test_table.setGeometry(QtCore.QRect(520, 300, 451, 451))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.test_table.setFont(font)
        self.test_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.test_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.test_table.setAutoScroll(False)
        self.test_table.setTabKeyNavigation(False)
        self.test_table.setProperty("showDropIndicator", False)
        self.test_table.setDragDropOverwriteMode(False)
        self.test_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.test_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.test_table.setShowGrid(False)
        self.test_table.setWordWrap(False)
        self.test_table.setCornerButtonEnabled(False)
        self.test_table.setRowCount(18)
        self.test_table.setColumnCount(18)
        self.test_table.setObjectName("test_table")
        self.test_table.horizontalHeader().setVisible(False)
        self.test_table.horizontalHeader().setDefaultSectionSize(25)
        self.test_table.horizontalHeader().setMinimumSectionSize(25)
        self.test_table.verticalHeader().setVisible(False)
        self.test_table.verticalHeader().setDefaultSectionSize(25)
        self.test_table.verticalHeader().setMinimumSectionSize(25)
        self.train_path_lb = QtWidgets.QLabel(self.centralwidget)
        self.train_path_lb.setGeometry(QtCore.QRect(230, 40, 251, 51))
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(12)
        self.train_path_lb.setFont(font)
        self.train_path_lb.setObjectName("train_path_lb")
        self.train_btn = QtWidgets.QPushButton(self.centralwidget)
        self.train_btn.setGeometry(QtCore.QRect(30, 40, 171, 51))
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(12)
        self.train_btn.setFont(font)
        self.train_btn.setObjectName("train_btn")
        self.test_path_lb = QtWidgets.QLabel(self.centralwidget)
        self.test_path_lb.setGeometry(QtCore.QRect(720, 40, 251, 51))
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(12)
        self.test_path_lb.setFont(font)
        self.test_path_lb.setObjectName("test_path_lb")
        self.test_btn = QtWidgets.QPushButton(self.centralwidget)
        self.test_btn.setGeometry(QtCore.QRect(520, 40, 171, 51))
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(12)
        self.test_btn.setFont(font)
        self.test_btn.setObjectName("test_btn")
        self.train_table = QtWidgets.QTableWidget(self.centralwidget)
        self.train_table.setGeometry(QtCore.QRect(30, 300, 451, 451))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.train_table.setFont(font)
        self.train_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.train_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.train_table.setAutoScroll(False)
        self.train_table.setTabKeyNavigation(False)
        self.train_table.setProperty("showDropIndicator", False)
        self.train_table.setDragDropOverwriteMode(False)
        self.train_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.train_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.train_table.setShowGrid(False)
        self.train_table.setWordWrap(False)
        self.train_table.setCornerButtonEnabled(False)
        self.train_table.setRowCount(18)
        self.train_table.setColumnCount(18)
        self.train_table.setObjectName("train_table")
        self.train_table.horizontalHeader().setVisible(False)
        self.train_table.horizontalHeader().setDefaultSectionSize(25)
        self.train_table.horizontalHeader().setMinimumSectionSize(25)
        self.train_table.verticalHeader().setVisible(False)
        self.train_table.verticalHeader().setDefaultSectionSize(25)
        self.train_table.verticalHeader().setMinimumSectionSize(25)
        self.slider = QtWidgets.QSlider(self.centralwidget)
        self.slider.setGeometry(QtCore.QRect(650, 140, 321, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.slider.setFont(font)
        self.slider.setMaximum(1)
        self.slider.setSingleStep(0)
        self.slider.setPageStep(1)
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.slider.setObjectName("slider")
        self.epoch_lb = QtWidgets.QLabel(self.centralwidget)
        self.epoch_lb.setGeometry(QtCore.QRect(40, 130, 81, 51))
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(12)
        self.epoch_lb.setFont(font)
        self.epoch_lb.setObjectName("epoch_lb")
        self.start_btn = QtWidgets.QPushButton(self.centralwidget)
        self.start_btn.setGeometry(QtCore.QRect(520, 130, 101, 51))
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(12)
        self.start_btn.setFont(font)
        self.start_btn.setObjectName("start_btn")
        self.test_box = QtWidgets.QComboBox(self.centralwidget)
        self.test_box.setGeometry(QtCore.QRect(720, 220, 251, 51))
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(12)
        self.test_box.setFont(font)
        self.test_box.setCurrentText("")
        self.test_box.setObjectName("test_box")
        self.train_box = QtWidgets.QComboBox(self.centralwidget)
        self.train_box.setGeometry(QtCore.QRect(230, 220, 251, 51))
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(12)
        self.train_box.setFont(font)
        self.train_box.setCurrentText("")
        self.train_box.setObjectName("train_box")
        self.epoch_box = QtWidgets.QSpinBox(self.centralwidget)
        self.epoch_box.setGeometry(QtCore.QRect(140, 130, 101, 51))
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(12)
        self.epoch_box.setFont(font)
        self.epoch_box.setMinimum(1)
        self.epoch_box.setMaximum(20)
        self.epoch_box.setObjectName("epoch_box")
        self.train_data_lb = QtWidgets.QLabel(self.centralwidget)
        self.train_data_lb.setGeometry(QtCore.QRect(40, 220, 161, 51))
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(12)
        self.train_data_lb.setFont(font)
        self.train_data_lb.setObjectName("train_data_lb")
        self.test_data_lb = QtWidgets.QLabel(self.centralwidget)
        self.test_data_lb.setGeometry(QtCore.QRect(520, 220, 161, 51))
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(12)
        self.test_data_lb.setFont(font)
        self.test_data_lb.setObjectName("test_data_lb")
        self.sync_check = QtWidgets.QCheckBox(self.centralwidget)
        self.sync_check.setGeometry(QtCore.QRect(270, 135, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Comic Sans MS")
        font.setPointSize(12)
        self.sync_check.setFont(font)
        self.sync_check.setObjectName("sync_check")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 18))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.train_path_lb.setText(_translate("MainWindow", "..."))
        self.train_btn.setText(_translate("MainWindow", "Train Data"))
        self.test_path_lb.setText(_translate("MainWindow", "..."))
        self.test_btn.setText(_translate("MainWindow", "Test Data"))
        self.epoch_lb.setText(_translate("MainWindow", "Epoch"))
        self.start_btn.setText(_translate("MainWindow", "Train"))
        self.train_data_lb.setText(_translate("MainWindow", "Train Data"))
        self.test_data_lb.setText(_translate("MainWindow", "Test Data"))
        self.sync_check.setText(_translate("MainWindow", "Synchronization"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
