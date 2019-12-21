# -*- coding: utf-8 -*-
import shutil
import fft2 as fft
import os
import sys,webbrowser
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtGui import *  
reload(sys)
sys.setdefaultencoding("utf8") 


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s
try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

#qtCreatorFile = "E:/mainwindow.ui" # Enter UI file here. 
qtCreatorFile = "mainwindow.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
 
class MyApp(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        #Menu and toolbar setting
        self.connect(self.importFile, QtCore.SIGNAL('triggered()'), self.showDialog)
        self.connect(self.resultFile, QtCore.SIGNAL('triggered()'), self.openResult)
        #self.connect(self.resultSave, QtCore.SIGNAL('triggered()'), self.saveFile)
        self.connect(self.exit, QtCore.SIGNAL('triggered()'), QtCore.SLOT('close()'))
        self.connect(self.actionHelp, QtCore.SIGNAL('triggered()'), self.openHelp)
        '''
        self.toolbar = self.addToolBar('import')
        self.toolbar.addAction(self.importFile)
        self.toolbar = self.addToolBar('ResultFile')
        self.toolbar.addAction(self.resultFile)
        self.toolbar = self.addToolBar('ResultSave AS')
        self.toolbar.addAction(self.resultSave)
        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(self.exit)     '''   
        self.connect(self.run, QtCore.SIGNAL('clicked()'),self.runs)
        #self.connect(self.ViewFig, QtCore.SIGNAL('clicked()'),self.imageOpen)
        self.connect(self.Delete, QtCore.SIGNAL('clicked()'),self.delete)
        self.connect(self.all, QtCore.SIGNAL('clicked()'),self.allselect)
        self.statusBar.showMessage('Ready')
                                             
        self.tw.setSelectionBehavior(QAbstractItemView.SelectRows)#select whole row
        #self.tw.setSelectionMode(QAbstractItemView.ExtendedSelection) 
        self.tw.setSelectionMode(QAbstractItemView.MultiSelection)
        global allfilepath,Imgpath
        allfilepath,Imgpath = [],[]


    def openResult(self):
        global allfilepath,Imgpath
        selectedRow = self.tw.selectionModel().selectedRows()
         #print selectedRow[0].row()
        if len(selectedRow) !=1:
            self.statusBar.showMessage('Please select ONE file to view figure...')
        else:
            rowId = selectedRow[0].row()
            resultPath = Imgpath[rowId]
            os.system('explorer '+resultPath)
        
    def openHelp(self):
        webbrowser.open('https://github.com/lzbbest/Rhythmic-Component-Analysis-Tool', new=0, autoraise=True)
        '''a = 'icon'
        b = '.png'        
        os.system(a+b)'''
        
    def showDialog(self):
        global allfilepath,Imgpath
        self.statusBar.showMessage('Running')
        #global filepath       
        filepath = QtGui.QFileDialog.getOpenFileNames(self, 'Open file', './')        
        lenFiles = len(filepath)
        for i in range(lenFiles):
            allfilepath.append(filepath[i])
            Imgpath.append('')
        rows = self.tw.rowCount()
        #self.startPoint.setText(str(lenFiles))       
        #for i in range(self.tw.rowCount()):
        #    self.tw.removeRow(0)       
        for rowcount in range(lenFiles):
            self.tw.insertRow(rowcount+rows)          
            self.newItem = QTableWidgetItem(filepath[rowcount])  
            self.tw.setItem(rowcount+rows,0,self.newItem)
        #print filepath[0] #E:\codetest\FFT\12-08-23\07ML.csv
        #self.tw.resizeRowsToContents()    #adjust row width       
        self.statusBar.showMessage('complete!')
        
    def saveFile(self):
        self.statusBar.showMessage('Running')
        global allfilepath,Imgpath
        selectedRow = self.tw.selectionModel().selectedRows()
         #print selectedRow[0].row()
        if len(selectedRow) !=1:
            self.statusBar.showMessage('Please select ONE file to view figure...')
        else:
            rowId = selectedRow[0].row()
            resultPath = Imgpath[rowId]
            #os.system('explorer '+resultPath)

            savepath = QtGui.QFileDialog.getExistingDirectory()       
            parent_path,name = os.path.split(resultPath)
            newdir = os.path.join(str(savepath),name)
            if os.path.exists(newdir):
                self.statusBar.showMessage('Error!The file exits!Please change Dir.')
            shutil.copytree(resultPath, newdir) #save as
            self.statusBar.showMessage('Complete!')
    
    def allselect(self):
        #self.tw.selectRow(0)
        for i in range(self.tw.rowCount()):
            self.tw.selectRow(i)        
        '''global allfilepath,allname
        for i in range(self.tw.rowCount()):
            self.tw.removeRow(0)'''
        self.statusBar.showMessage('All rows have been selected')
        
    def imageOpen(self):
         global allfilepath
         selectedRow = self.tw.selectionModel().selectedRows()
         #print selectedRow[0].row()
         if len(selectedRow) !=1:
             self.statusBar.showMessage('Please select ONE file to view figure...')
         else:
             rowId = selectedRow[0].row()
             name = self.tw.item(rowId,0)
             name = str(name.text())
             if name[0:3] == '***':
                 name = name[3:]
             #print name
             #A = os.path.join(resultPath,name+"-PartA.png")
             #B = os.path.join(resultPath,name+"-PartB.png")
             A = os.path.join(Imgpath[rowId],name+"-PartA.png")
             B = os.path.join(Imgpath[rowId],name+"-PartB.png")
             os.system('explorer ' + B)
             os.system('explorer ' + A)
    
    def delete(self):
        #currentRow = self.tw.currentRow()
        #row = self.tw.selectedRows()
        #rows = []       
        global allfilepath,Imgpath,allname
        try:
            currentRow = self.tw.selectionModel().selectedRows()
            rows = sorted(currentRow)
            rows.reverse() #delete must be from bottom to top!!!!
            for index in rows:
                print(index.row())
                #rows.append(int(index.row()))
                allfilepath.pop(int(index.row()))
                Imgpath.pop(int(index.row()))
                self.tw.removeRow(int(index.row()))
                    
            #print row,type(row)
            self.statusBar.showMessage('Delete complete!')
        except (KeyError, TypeError, IndexError, BaseException) as error:
            self.statusBar.showMessage('No file was selected! ')
        
    def runs(self):
        pic = 'no'
        if self.pic.isChecked():
            pic = 'yes'
        global allfilepath,Imgpath
        if not len(allfilepath):
            self.statusBar.showMessage('NO input files...')
        else:
            self.statusBar.showMessage('Running,please wait for a while...')
            point = int(self.point.text())
            minPeriod = int(self.minPeriod.text())
            maxPeriod = int(self.maxPeriod.text())
            start = int(self.startPoint.text())
            end = int(self.endPoint.text())
            #self.endPoint.setText(filepath[0])
            #global resultPath,allresult
            selectedRow = self.tw.selectionModel().selectedRows()
            if not len(selectedRow):
                self.statusBar.showMessage('Please select files to analyze...')
            else:
                selectedRow.sort()
                tempfilepath,Pathlist = [],[]
                for index in selectedRow:
                    tempfilepath.append(allfilepath[int(index.row())])
                try:
                    (allresult, Pathlist) = fft.cal(point, start, end, minPeriod, maxPeriod, tempfilepath, pic)
                except (KeyError, TypeError, IndexError, BaseException) as error:
                    self.statusBar.showMessage('System has error,please check inputFile.')
                
                for i in range(len(tempfilepath)):
                    index = int(selectedRow[i].row())
                    Imgpath[index] = Pathlist[i]
                    self.newItem = QTableWidgetItem('finished')
                    self.newItem.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                    self.tw.setItem(index,1,self.newItem)
                    
                self.statusBar.showMessage('All files have complete!')
         
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    