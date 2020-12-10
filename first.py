from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTableWidget, QMessageBox, QMainWindow, QTableWidgetItem
import sqlite3
import bearing_second
import os
import time
import torch
import torch.nn as nn
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from PyQt5.QtCore import QTime, QDateTime, QDate, Qt

conn = sqlite3.connect('bearing')
c = conn.cursor()

class Bearing_model(nn.Module):
    def __init__(self):
        super(Bearing_model, self).__init__()
        self.dw1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, groups=3)
        self.pw1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(12)

        self.dw2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1, groups=12)
        self.pw2 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=1, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3)

        self.fc1 = nn.Linear(in_features=2048, out_features=4)
        # self.fc2 = nn.Linear(in_features = 2048, out_features = 4)
        # self.dr = nn.Dropout(0.5)

    def forward(self, x):
        out = self.dw1(x)
        out = self.pw1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        out = self.dw2(out)
        out = self.pw2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.maxpool3(out)
        out = self.maxpool2(out)
        out = out.view(-1, 2048)
        out = self.fc1(out)

        return out



class Ui_bearing(object):
    def openSecondWindow(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = bearing_second.Ui_second_window()
        self.ui.setupUi(self.window)
        #bearing.hide()
        self.window.show()
    def msg_normal(self, mess):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(mess)
        msg.exec_()
    def msg_notify(self):
        flag = 0
        c.execute('SELECT Machine.ID FROM Machine')
        data = c.fetchall()
        m_id = self.line_id.text()
        for id in data:
            if str(id[0]) == m_id:
                flag = 1
                break
        if flag == 1:
            self.check_status()
            flag = 0
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Machine Doesn't exist")
            msg.setInformativeText("Is this a new one?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_ret = str(msg.exec_())
            if(msg_ret == "16384"):                                         #16384: YES
                self.openSecondWindow()
            elif(msg_ret == "65536"):                                       #65536: NO
                self.msg_normal("Please insert machine ID again!!!")

    def update_screen(self):
        c.execute('SELECT * FROM machine')
        data = c.fetchall()
        row_count = 0
        for row in data:
            row_count += 1
        self.tb_status.setRowCount(row_count)
        if row_count == 0:
            self.msg_normal('No Information in Database')
        else:
            idx = 0
            for row in data:
                self.tb_status.setItem(idx, 0, QTableWidgetItem(str(row[0])))
                self.tb_status.setItem(idx, 1, QTableWidgetItem(str(row[1])))
                self.tb_status.setItem(idx, 2, QTableWidgetItem(str(row[2])))
                self.tb_status.setItem(idx, 3, QTableWidgetItem(str(row[3])))
                idx += 1

    def remove_id(self):
        flag = 0
        get_id = self.line_id.text()
        c.execute("SELECT machine.ID FROM machine")
        data = c.fetchall()
        for i in data:
            if(str(i[0]) == get_id):
                flag = 1
                break
        if flag == 1:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Do you want to delete this machine?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_ret = str(msg.exec_())
            if (msg_ret == "16384"):  # 16384: YES
                c.execute("DELETE FROM machine WHERE ID = (?)", (int(get_id),))
                conn.commit()
                self.msg_normal("Deleted Successfully")
                self.update_screen()
            elif (msg_ret == "65536"):  # 65536: NO
                pass
            flag = 0
        else:
            self.msg_normal("Invalid Machine ID")

    def check_status(self):
        flag = 0
        file_before = os.listdir('/home/toan/bearing/simu_data')
        time.sleep(10)
        file_after = os.listdir('/home/toan/bearing/simu_data')
        for file in file_after:
            if file not in file_before:
                img = file
                flag = 1
                break
            else:
                flag = 0
        print(flag)
        if flag == 1:
            m_id = self.line_id.text()
            model = Bearing_model()
            model.load_state_dict(torch.load('/home/toan/Desktop/grad_thesis/model_weights.pth'))
            model.eval()
            img = Image.open('/home/toan/bearing/simu_data/'+img).convert("RGB")
            img = transforms.ToTensor()(img).unsqueeze_(0)
            output = model(img)
            _, predicted = torch.max(output, 1)
            if predicted.item() == 0:
                self.line_id.setText("normal")
            elif predicted.item() == 1:
                self.line_id.setText("inner")
            elif predicted.item() == 2:
                self.line_id.setText("outer")
            elif predicted.item() == 3:
                self.line_id.setText("balls")
            dtime = QDate.currentDate().toString(Qt.ISODate)
            print(type(dtime), dtime)
            status = self.line_id.text()
            c.execute("""UPDATE machine SET Status = (?), Time = (?)  WHERE ID = (?)""", (status, dtime, int(m_id)))
            conn.commit()
            flag = 0
        elif flag == 0:
            self.msg_normal("No New Data")

    def setupUi(self, bearing):
        bearing.setObjectName("bearing")
        bearing.resize(631, 600)
        self.tb_status = QtWidgets.QTableWidget(bearing)
        self.tb_status.setGeometry(QtCore.QRect(100, 100, 411, 181))
        self.tb_status.setObjectName("tb_status")
        self.tb_status.setColumnCount(4)
        self.tb_status.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tb_status.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tb_status.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tb_status.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tb_status.setHorizontalHeaderItem(3, item)
        self.lb_status = QtWidgets.QLabel(bearing)
        self.lb_status.setGeometry(QtCore.QRect(110, 60, 171, 41))
        self.lb_status.setObjectName("lb_status")
        self.lb_title = QtWidgets.QLabel(bearing)
        self.lb_title.setGeometry(QtCore.QRect(120, 0, 451, 81))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.lb_title.setFont(font)
        self.lb_title.setObjectName("lb_title")
        self.line_id = QtWidgets.QLineEdit(bearing)
        self.line_id.setGeometry(QtCore.QRect(240, 70, 141, 25))
        self.line_id.setInputMethodHints(QtCore.Qt.ImhNone)
        self.line_id.setObjectName("line_id")
        self.btn_check = QtWidgets.QPushButton(bearing)
        self.btn_check.setGeometry(QtCore.QRect(390, 70, 121, 25))
        self.btn_check.setObjectName("btn_check")
        self.btn_add = QtWidgets.QPushButton(bearing)
        self.btn_add.setGeometry(QtCore.QRect(100, 290, 91, 25))
        self.btn_add.setObjectName("btn_add")
        self.btn_rm = QtWidgets.QPushButton(bearing)
        self.btn_rm.setGeometry(QtCore.QRect(260, 290, 81, 25))
        self.btn_rm.setObjectName("btn_rm")
        self.btn_update = QtWidgets.QPushButton(bearing)
        self.btn_update.setGeometry(QtCore.QRect(420, 290, 91, 25))
        self.btn_update.setObjectName("btn_update")

        self.btn_check.clicked.connect(self.msg_notify)
        self.btn_add.clicked.connect(self.openSecondWindow)
        self.btn_update.clicked.connect(self.update_screen)
        self.btn_rm.clicked.connect(self.remove_id)

        self.retranslateUi(bearing)
        QtCore.QMetaObject.connectSlotsByName(bearing)

    def retranslateUi(self, bearing):
        _translate = QtCore.QCoreApplication.translate
        bearing.setWindowTitle(_translate("bearing", "bearing"))
        item = self.tb_status.horizontalHeaderItem(0)
        item.setText(_translate("bearing", "Machine ID"))
        item = self.tb_status.horizontalHeaderItem(1)
        item.setText(_translate("bearing", "Machine Name"))
        item = self.tb_status.horizontalHeaderItem(2)
        item.setText(_translate("bearing", "Status"))
        item = self.tb_status.horizontalHeaderItem(3)
        item.setText(_translate("bearing", "Time"))
        self.lb_status.setText(_translate("bearing", "Last Check Status"))
        self.lb_title.setText(_translate("bearing", "BEARING FAULT DIAGNOSIS SYSTEM"))
        self.btn_check.setText(_translate("bearing", "CHECK NOW"))
        self.btn_add.setText(_translate("bearing", "ADD"))
        self.btn_rm.setText(_translate("bearing", "REMOVE"))
        self.btn_update.setText(_translate("bearing", "UPDATE"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    bearing = QtWidgets.QWidget()
    ui = Ui_bearing()
    ui.setupUi(bearing)
    ui.update_screen()
    bearing.show()
    sys.exit(app.exec_())
