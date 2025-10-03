import sys
import os

#sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QSizePolicy, QScrollArea, QHBoxLayout, QPushButton, QListWidget, QListWidgetItem
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QSize, QUrl, QEvent, QFileInfo
# from PyQt5 import QtMultimedia
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

from util.image_warp import crop2_169, resize_img

from VITON.viton_upperbody import FrameProcessor
from util.camera_util import list_available_cameras


class VitonThread(QThread):
    frameCaptured = pyqtSignal(QImage)

    def __init__(self,garment_id_list):
        super().__init__()
        self.cap = self.get_camera()
        if not self.cap.isOpened():
            print("Failed to open the selected camera.")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running = True
        self.frame_processor = FrameProcessor(garment_id_list)

    def set_taregt_id(self, id):
        print(id)
        self.frame_processor.set_target_garment(id)

    def get_camera(self):
        cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        return cap

    def run(self):
        while self.running:
            ret, frame = self.cap.read()

            if ret:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                ## ichao: remove flip (Nov 13, 2024)
                frame=cv2.flip(frame, 1)
                frame=resize_img(frame,max_height=1024)
                #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame=crop2_169(frame)
                frame = self.frame_processor(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                step = channel * width
                q_img = QImage(frame.data, width, height, step, QImage.Format_RGB888)
                self.frameCaptured.emit(q_img)


    def stop(self):
        self.running = False
        self.cap.release()

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Virtual Try-On")
        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(200, 150)
        ## enable fullscreen
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.showFullScreen()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(False)

        layout = QHBoxLayout()
        layout.addWidget(self.image_label)

        # Create a scroll area for the horizontal layout
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Create a QWidget to hold the list
        container_list = QWidget()
        layout_list = QVBoxLayout(container_list)

        # Create a QListWidget
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)  # Ensure only one item can be selected at a time
        self.list_widget.setIconSize(QSize(150,150))

        # Add none item
        item = QListWidgetItem()
        item.setIcon(QIcon('./assets/none.png'))
        item.setData(Qt.UserRole, -1)
        self.list_widget.addItem(item)

        # Add 15 images to the horizontal layout
        #garment_id_list = [3, 2, 17, 18, 22]
        garment_name_list = ['lab_03','lab_04','lab_07','jin_17','jin_18','jin_22']
        self.gid_map = dict()
        self.gid_map[0] = -1
        self.gid_map[1] = 3
        self.gid_map[2] = 2
        self.gid_map[3] = 17
        self.gid_map[4] = 18
        self.gid_map[5] = 22
        
        for i, garment in enumerate(garment_name_list):
            item = QListWidgetItem()
            item.setIcon(QIcon('./assets/garment_images/'+garment+'_white_bg.jpg'))
            item.setData(Qt.UserRole, i)
            self.list_widget.addItem(item)
        #for i in range(9):
        #    item = QListWidgetItem()
        #    item.setIcon(QIcon('./figures/female_garments/lab_'+str(i).zfill(2)+'_white_bg.jpg'))
        #    item.setData(Qt.UserRole, 16+i)
        #    self.list_widget.addItem(item)
        layout_list.addWidget(self.list_widget)
        self.list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        self.list_widget.installEventFilter(self)


        
        scroll_area.setWidget(container_list)

        layout.addWidget(scroll_area)
        scroll_area.setFixedWidth(230)

        #layout.setStretch(0, 9)
        #layout.setStretch(1, 2)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        for i in range(len(garment_name_list)):
            garment_name_list[i] = garment_name_list[i]+'_vmsdp2ta'

        self.viton_thread = VitonThread(garment_name_list)
        self.viton_thread.frameCaptured.connect(self.update_image)
        self.viton_thread.start()

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.KeyPress):
            # print(f'keypress obj: {obj}')
            if obj == self.list_widget:
                self.list_widget.setCurrentRow(event.key()-48)
                return True
        
        return super(CameraApp, self).eventFilter(obj, event)
            
        
                
    def keyPressEvent(self, e):
        print(f'key: {e.key()}')
        if e.key() == Qt.Key_0:
            self.list_widget.setCurrentRow(0)
        elif e.key() == Qt.Key_1:
            self.list_widget.setCurrentRow(1)
        elif e.key() == Qt.Key_2:
            self.list_widget.setCurrentRow(2)
        elif e.key() == Qt.Key_3:
             self.list_widget.setCurrentRow(3)
        elif e.key() == Qt.Key_4:
             self.list_widget.setCurrentRow(4)
        elif e.key() == Qt.Key_5:
             self.list_widget.setCurrentRow(5)
        elif e.key() == Qt.Key_Q or e.key() == Qt.Key_Escape:
            print("Exiting application...")
            self.close()  # Triggers closeEvent

    def on_selection_changed(self):
        print('selection changed')
        selected_items = self.list_widget.selectedItems() 
        ## TODO: test if it can keep playing ...
        if selected_items:
            selected_item = selected_items[0]
            item_id = selected_item.data(Qt.UserRole)  # Retrieve the ID from the item

            ## change garment
            self.viton_thread.set_taregt_id(item_id)
            
            #self.trigger_function(item_id)  # Pass the ID to the trigger function



    def update_image(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):
        self.viton_thread.stop()
        self.viton_thread.wait()
        print("Viton thread stopped")
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
