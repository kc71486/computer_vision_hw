import sys
from PyQt5.QtWidgets import QApplication, QWidget


def constructapp():
    app = QApplication(sys.argv)

    qwidget = QWidget();
    qwidget.setWindowTitle("First GUI Window")
    qwidget.show()

    sys.exit(app.exec_())

