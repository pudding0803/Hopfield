import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog

from UI import Ui_MainWindow


def ordinal(n: int) -> str:
    if n % 100 // 10 == 1 or (n - 1) % 10 > 2:
        return str(n) + 'th pattern'
    else:
        return str(n) + ['st', 'nd', 'rd'][(n - 1) % 10] + ' pattern'


class Controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.setFocus()
        self.ui.setupUi(self)
        self.setWindowTitle('NN HW3 - Hopfield')
        self.setWindowIcon(QtGui.QIcon('pudding.png'))

        self.BLACK = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        self.BLACK.setStyle(QtCore.Qt.SolidPattern)
        self.WHITE = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        self.WHITE.setStyle(QtCore.Qt.SolidPattern)
        self.GRAY = QtGui.QBrush(QtGui.QColor(127, 127, 127))
        self.GRAY.setStyle(QtCore.Qt.SolidPattern)

        self.status = self.w = self.h = self.size = 0
        self.train_path = self.test_path = ''
        self.train_data = self.test_data = self.result = None

        self.setup_control()

    def setup_control(self):
        self.reset()
        self.ui.train_btn.clicked.connect(self.choose_train)
        self.ui.test_btn.clicked.connect(self.choose_test)
        self.ui.start_btn.clicked.connect(self.start)
        self.ui.train_box.currentIndexChanged.connect(lambda: self.change_pattern(True))
        self.ui.test_box.currentIndexChanged.connect(lambda: self.change_pattern(False))
        self.ui.slider.valueChanged.connect(lambda: self.change_pattern(False))
        self.ui.epoch_box.valueChanged.connect(self.reset)
        self.ui.sync_check.stateChanged.connect(self.reset)

    def choose_train(self):
        self.ui.train_path_lb.setText('...')
        self.train_path, _ = QFileDialog.getOpenFileName(self, '選擇訓練資料集', './', '文字文件 (*.txt)')
        if self.train_path == '':
            self.update_status(0)
            return
        self.ui.train_path_lb.setText(self.train_path.split('/')[-1])
        if self.test_path:
            self.update_status(1)

    def choose_test(self):
        self.ui.test_path_lb.setText('...')
        self.test_path, _ = QFileDialog.getOpenFileName(self, '選擇測試資料集', './', '文字文件 (*.txt)')
        if self.test_path == '':
            self.update_status(0)
            return
        self.ui.test_path_lb.setText(self.test_path.split('/')[-1])
        if self.train_path:
            self.update_status(1)

    def update_status(self, status: int):
        self.status = status
        self.ui.start_btn.setEnabled(self.status >= 1)
        self.ui.train_box.setEnabled(self.status == 2)
        self.ui.test_box.setEnabled(self.status == 2)
        self.ui.slider.setEnabled(self.status == 2)

    def reset(self):
        self.update_status(int(self.train_path != '' and self.test_path != ''))
        self.ui.train_box.clear()
        self.ui.test_box.clear()
        self.ui.slider.setValue(0)
        self.ui.slider.setMaximum(self.ui.epoch_box.value())
        for i in range(18):
            for j in range(18):
                item = QtWidgets.QTableWidgetItem()
                item.setBackground(self.GRAY)
                self.ui.train_table.setItem(i, j, item)
                item = QtWidgets.QTableWidgetItem()
                item.setBackground(self.GRAY)
                self.ui.test_table.setItem(i, j, item)

    def start(self):
        self.reset()
        self.update_status(2)
        self.train_data = self.get_data(self.train_path)
        self.test_data = self.get_data(self.test_path)
        self.result = [[self.test_data[i].copy()] for i in range(self.test_data.shape[0])]
        self.ui.train_box.addItems([ordinal(i + 1) for i in range(self.train_data.shape[0])])
        self.ui.test_box.addItems([ordinal(i + 1) for i in range(self.test_data.shape[0])])
        weight = np.zeros((self.size, self.size))
        for arr in self.train_data:
            weight += np.matmul(arr.T, arr)
        for i in range(self.size):
            weight[i][i] = 0
        theta = weight.sum(axis=0).reshape((self.size, 1))
        for process in self.result:
            x = process[0].copy().T
            for _ in range(self.ui.epoch_box.value()):
                if self.ui.sync_check.isChecked():
                    y = np.matmul(weight, x) - theta
                    for i in range(self.size):
                        if y[i][0] == 0:
                            y[i][0] = x[i][0]
                        else:
                            y[i][0] = 1 if y[i][0] > 0 else -1
                    x = y.copy()
                else:
                    for i in range(self.size):
                        tmp = np.dot(weight[i], x) - theta[i][0]
                        if tmp != 0:
                            x[i][0] = 1 if tmp > 0 else -1
                process.append(x.copy().T)

    def change_pattern(self, train: bool):
        if train:
            pattern = self.train_data[self.ui.train_box.currentIndex()].reshape((self.h, self.w))
        else:
            pattern = self.result[self.ui.test_box.currentIndex()][self.ui.slider.value()].reshape((self.h, self.w))
        for i in range(18):
            for j in range(18):
                item = QtWidgets.QTableWidgetItem()
                item.setBackground(self.GRAY if i >= self.h or j >= self.w
                                   else self.BLACK if pattern[i][j] == 1 else self.WHITE)
                if train:
                    self.ui.train_table.setItem(i, j, item)
                else:
                    self.ui.test_table.setItem(i, j, item)

    def get_data(self, file_name: str) -> np.ndarray:
        self.h = 0
        case = 1
        data = np.array([])
        with open(file_name) as file:
            file = file.readlines()
        self.w = len(file[0]) - int(file[0][-1] == '\n')
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            if line == '':
                case += 1
                continue
            if case == 1:
                self.h += 1
            data = np.append(data, list(map(lambda i: 1 if i == '1' else -1, list(line))))
        self.size = self.h * self.w
        return data.reshape((case, 1, self.size))


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = Controller()
    window.show()
    sys.exit(app.exec_())
