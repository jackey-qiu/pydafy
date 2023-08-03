from PyQt5.QtWidgets import QMessageBox
import logging

def error_pop_up(msg_text = 'error', window_title = ['Error','Information','Warning'][0]):
    msg = QMessageBox()
    if window_title == 'Error':
        msg.setIcon(QMessageBox.Critical)
    elif window_title == 'Warning':
        msg.setIcon(QMessageBox.Warning)
    else:
        msg.setIcon(QMessageBox.Information)

    msg.setText(msg_text)
    msg.setWindowTitle(window_title)
    msg.exec_()

#redirect the error stream to qt widget
class QTextEditLogger(logging.Handler):
    def __init__(self, textbrowser_widget, log_type = 'debug'):
        super().__init__()
        self.textBrowser_error_msg = textbrowser_widget
        self.setLevel(getattr(logging, log_type.upper()))
        self.records = 0
        self.log_type = log_type
        #self.color = self.get_color()
        # self.widget.setReadOnly(True)

    def set_level(self, level):
        if level.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            self.setLevel(getattr(logging, level.upper()))
            self.log_type = level

    def get_color(self, tag):
        if tag.upper() in ['ERROR','CRITICAL']:
            return 'red'
        elif tag.upper() == 'INFO':
            return 'white'
        elif tag.upper() == 'WARNING':
            return 'orange'
        else:
            return 'yellow'

    def emit(self, record):
        error_msg = self.format(record)
        separator = '-' * 80
        header = f"LOGGING INFORMATION WITH LEVEL > {self.log_type.upper()}"
        if self.records == 0:
            self.textBrowser_error_msg.clear()
        self.records += 1

        if self.records==1:
            notice = f'{header}\nrecord {self.records}'
        else:
            notice = f'record {self.records}'
        
        cursor = self.textBrowser_error_msg.textCursor()
        cursor.insertHtml('''<p><span style="color: {};">{} <br></span>'''.format(self.get_color(record.levelname), " "))
        self.textBrowser_error_msg.append(notice + '\n' +separator+'\n'+error_msg)
        verScrollBar = self.textBrowser_error_msg.verticalScrollBar()
        verScrollBar.setValue(verScrollBar.maximum())