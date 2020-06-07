# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'VisImSorter.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_VisImSorter(object):
    def setupUi(self, VisImSorter):
        VisImSorter.setObjectName("VisImSorter")
        VisImSorter.resize(901, 616)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(VisImSorter.sizePolicy().hasHeightForWidth())
        VisImSorter.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        VisImSorter.setFont(font)
        VisImSorter.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.centralwidget = QtWidgets.QWidget(VisImSorter)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.group_input = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.group_input.sizePolicy().hasHeightForWidth())
        self.group_input.setSizePolicy(sizePolicy)
        self.group_input.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.group_input.setObjectName("group_input")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.group_input)
        self.verticalLayout.setObjectName("verticalLayout")
        self.select_folder_button = QtWidgets.QPushButton(self.group_input)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(255)
        sizePolicy.setHeightForWidth(self.select_folder_button.sizePolicy().hasHeightForWidth())
        self.select_folder_button.setSizePolicy(sizePolicy)
        self.select_folder_button.setMinimumSize(QtCore.QSize(0, 60))
        self.select_folder_button.setMaximumSize(QtCore.QSize(1000, 16777215))
        self.select_folder_button.setAcceptDrops(True)
        self.select_folder_button.setObjectName("select_folder_button")
        self.verticalLayout.addWidget(self.select_folder_button)
        self.check_subdirs_box = QtWidgets.QCheckBox(self.group_input)
        self.check_subdirs_box.setChecked(True)
        self.check_subdirs_box.setObjectName("check_subdirs_box")
        self.verticalLayout.addWidget(self.check_subdirs_box)
        self.gridLayout.addWidget(self.group_input, 0, 0, 1, 1)
        self.group_sizes = QtWidgets.QGroupBox(self.centralwidget)
        self.group_sizes.setObjectName("group_sizes")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.group_sizes)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.groupBox_2 = QtWidgets.QGroupBox(self.group_sizes)
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.V1 = QtWidgets.QRadioButton(self.groupBox_2)
        self.V1.setChecked(True)
        self.V1.setObjectName("V1")
        self.horizontalLayout_10.addWidget(self.V1)
        self.V2 = QtWidgets.QRadioButton(self.groupBox_2)
        self.V2.setObjectName("V2")
        self.horizontalLayout_10.addWidget(self.V2)
        self.verticalLayout_5.addWidget(self.groupBox_2)
        self.frame_2 = QtWidgets.QFrame(self.group_sizes)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.radio_folder_size = QtWidgets.QRadioButton(self.frame_2)
        self.radio_folder_size.setChecked(True)
        self.radio_folder_size.setObjectName("radio_folder_size")
        self.gridLayout_2.addWidget(self.radio_folder_size, 0, 0, 1, 1)
        self.spin_num_files = QtWidgets.QSpinBox(self.frame_2)
        self.spin_num_files.setMinimumSize(QtCore.QSize(75, 0))
        self.spin_num_files.setMinimum(2)
        self.spin_num_files.setMaximum(2000)
        self.spin_num_files.setProperty("value", 20)
        self.spin_num_files.setObjectName("spin_num_files")
        self.gridLayout_2.addWidget(self.spin_num_files, 0, 1, 1, 1)
        self.radio_number_folders = QtWidgets.QRadioButton(self.frame_2)
        self.radio_number_folders.setChecked(False)
        self.radio_number_folders.setObjectName("radio_number_folders")
        self.gridLayout_2.addWidget(self.radio_number_folders, 1, 0, 1, 1)
        self.spin_num_folders = QtWidgets.QSpinBox(self.frame_2)
        self.spin_num_folders.setMinimumSize(QtCore.QSize(75, 0))
        self.spin_num_folders.setMinimum(2)
        self.spin_num_folders.setMaximum(2000)
        self.spin_num_folders.setProperty("value", 30)
        self.spin_num_folders.setObjectName("spin_num_folders")
        self.gridLayout_2.addWidget(self.spin_num_folders, 1, 1, 1, 1)
        self.verticalLayout_5.addWidget(self.frame_2)
        self.groupBox_6 = QtWidgets.QGroupBox(self.group_sizes)
        self.groupBox_6.setObjectName("groupBox_6")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.groupBox_6)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.slider_enforce_equal = QtWidgets.QSlider(self.groupBox_6)
        self.slider_enforce_equal.setMinimum(0)
        self.slider_enforce_equal.setMaximum(300)
        self.slider_enforce_equal.setProperty("value", 80)
        self.slider_enforce_equal.setOrientation(QtCore.Qt.Horizontal)
        self.slider_enforce_equal.setInvertedAppearance(False)
        self.slider_enforce_equal.setInvertedControls(False)
        self.slider_enforce_equal.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.slider_enforce_equal.setTickInterval(10)
        self.slider_enforce_equal.setObjectName("slider_enforce_equal")
        self.horizontalLayout_4.addWidget(self.slider_enforce_equal)
        self.lbl_slider_enforce_equal = QtWidgets.QLabel(self.groupBox_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(162)
        sizePolicy.setVerticalStretch(95)
        sizePolicy.setHeightForWidth(self.lbl_slider_enforce_equal.sizePolicy().hasHeightForWidth())
        self.lbl_slider_enforce_equal.setSizePolicy(sizePolicy)
        self.lbl_slider_enforce_equal.setMinimumSize(QtCore.QSize(60, 0))
        self.lbl_slider_enforce_equal.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lbl_slider_enforce_equal.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_slider_enforce_equal.setObjectName("lbl_slider_enforce_equal")
        self.horizontalLayout_4.addWidget(self.lbl_slider_enforce_equal)
        self.verticalLayout_5.addWidget(self.groupBox_6)
        self.frame_3 = QtWidgets.QFrame(self.group_sizes)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label = QtWidgets.QLabel(self.frame_3)
        self.label.setObjectName("label")
        self.horizontalLayout_6.addWidget(self.label)
        self.spin_folder_size_min_files = QtWidgets.QSpinBox(self.frame_3)
        self.spin_folder_size_min_files.setMinimumSize(QtCore.QSize(60, 0))
        self.spin_folder_size_min_files.setMinimum(1)
        self.spin_folder_size_min_files.setMaximum(2000)
        self.spin_folder_size_min_files.setProperty("value", 3)
        self.spin_folder_size_min_files.setObjectName("spin_folder_size_min_files")
        self.horizontalLayout_6.addWidget(self.spin_folder_size_min_files)
        self.verticalLayout_5.addWidget(self.frame_3)
        self.groupBox_7 = QtWidgets.QGroupBox(self.group_sizes)
        self.groupBox_7.setObjectName("groupBox_7")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.groupBox_7)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.check_equal_name = QtWidgets.QCheckBox(self.groupBox_7)
        self.check_equal_name.setObjectName("check_equal_name")
        self.horizontalLayout_9.addWidget(self.check_equal_name)
        self.spin_equal_first_symbols = QtWidgets.QSpinBox(self.groupBox_7)
        self.spin_equal_first_symbols.setMinimumSize(QtCore.QSize(60, 0))
        self.spin_equal_first_symbols.setMinimum(1)
        self.spin_equal_first_symbols.setMaximum(200)
        self.spin_equal_first_symbols.setProperty("value", 8)
        self.spin_equal_first_symbols.setObjectName("spin_equal_first_symbols")
        self.horizontalLayout_9.addWidget(self.spin_equal_first_symbols)
        self.verticalLayout_5.addWidget(self.groupBox_7)
        self.gridLayout.addWidget(self.group_sizes, 0, 1, 4, 1)
        self.group_analyze = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.group_analyze.sizePolicy().hasHeightForWidth())
        self.group_analyze.setSizePolicy(sizePolicy)
        self.group_analyze.setObjectName("group_analyze")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.group_analyze)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.groupBox_4 = QtWidgets.QGroupBox(self.group_analyze)
        self.groupBox_4.setObjectName("groupBox_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.groupBox_4)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.slider_histogram_bands = QtWidgets.QSlider(self.groupBox_4)
        self.slider_histogram_bands.setMinimum(2)
        self.slider_histogram_bands.setMaximum(100)
        self.slider_histogram_bands.setPageStep(2)
        self.slider_histogram_bands.setProperty("value", 20)
        self.slider_histogram_bands.setOrientation(QtCore.Qt.Horizontal)
        self.slider_histogram_bands.setInvertedAppearance(False)
        self.slider_histogram_bands.setInvertedControls(False)
        self.slider_histogram_bands.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.slider_histogram_bands.setTickInterval(10)
        self.slider_histogram_bands.setObjectName("slider_histogram_bands")
        self.horizontalLayout_3.addWidget(self.slider_histogram_bands)
        self.lbl_histogram_bands = QtWidgets.QLabel(self.groupBox_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(162)
        sizePolicy.setVerticalStretch(95)
        sizePolicy.setHeightForWidth(self.lbl_histogram_bands.sizePolicy().hasHeightForWidth())
        self.lbl_histogram_bands.setSizePolicy(sizePolicy)
        self.lbl_histogram_bands.setMinimumSize(QtCore.QSize(60, 0))
        self.lbl_histogram_bands.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lbl_histogram_bands.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_histogram_bands.setObjectName("lbl_histogram_bands")
        self.horizontalLayout_3.addWidget(self.lbl_histogram_bands)
        self.verticalLayout_2.addWidget(self.groupBox_4)
        self.group_colorspaces_all = QtWidgets.QGroupBox(self.group_analyze)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.group_colorspaces_all.sizePolicy().hasHeightForWidth())
        self.group_colorspaces_all.setSizePolicy(sizePolicy)
        self.group_colorspaces_all.setObjectName("group_colorspaces_all")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.group_colorspaces_all)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.list_color_spaces = QtWidgets.QListWidget(self.group_colorspaces_all)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.list_color_spaces.sizePolicy().hasHeightForWidth())
        self.list_color_spaces.setSizePolicy(sizePolicy)
        self.list_color_spaces.setMinimumSize(QtCore.QSize(44, 85))
        self.list_color_spaces.setMaximumSize(QtCore.QSize(60, 85))
        self.list_color_spaces.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.list_color_spaces.setFrameShadow(QtWidgets.QFrame.Plain)
        self.list_color_spaces.setLineWidth(3)
        self.list_color_spaces.setMidLineWidth(2)
        self.list_color_spaces.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.list_color_spaces.setProperty("showDropIndicator", False)
        self.list_color_spaces.setAlternatingRowColors(False)
        self.list_color_spaces.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.list_color_spaces.setTextElideMode(QtCore.Qt.ElideMiddle)
        self.list_color_spaces.setLayoutMode(QtWidgets.QListView.SinglePass)
        self.list_color_spaces.setModelColumn(0)
        self.list_color_spaces.setObjectName("list_color_spaces")
        item = QtWidgets.QListWidgetItem()
        item.setText("HSV")
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        item.setFlags(QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsEnabled)
        self.list_color_spaces.addItem(item)
        item = QtWidgets.QListWidgetItem()
        item.setText("RGB")
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        item.setFlags(QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsEnabled)
        self.list_color_spaces.addItem(item)
        item = QtWidgets.QListWidgetItem()
        item.setText("CMYK")
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        item.setFlags(QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsEnabled)
        self.list_color_spaces.addItem(item)
        item = QtWidgets.QListWidgetItem()
        item.setText("YCbCr")
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        item.setFlags(QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsEnabled)
        self.list_color_spaces.addItem(item)
        self.horizontalLayout_7.addWidget(self.list_color_spaces)
        self.frame_sub_spaces = QtWidgets.QFrame(self.group_colorspaces_all)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_sub_spaces.sizePolicy().hasHeightForWidth())
        self.frame_sub_spaces.setSizePolicy(sizePolicy)
        self.frame_sub_spaces.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_sub_spaces.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_sub_spaces.setObjectName("frame_sub_spaces")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.frame_sub_spaces)
        self.horizontalLayout_8.setContentsMargins(-1, 0, 9, 0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.list_color_spaces_HSV = QtWidgets.QListWidget(self.frame_sub_spaces)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.list_color_spaces_HSV.sizePolicy().hasHeightForWidth())
        self.list_color_spaces_HSV.setSizePolicy(sizePolicy)
        self.list_color_spaces_HSV.setMinimumSize(QtCore.QSize(10, 85))
        self.list_color_spaces_HSV.setMaximumSize(QtCore.QSize(35, 85))
        self.list_color_spaces_HSV.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.list_color_spaces_HSV.setFrameShadow(QtWidgets.QFrame.Plain)
        self.list_color_spaces_HSV.setLineWidth(3)
        self.list_color_spaces_HSV.setMidLineWidth(2)
        self.list_color_spaces_HSV.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.list_color_spaces_HSV.setProperty("showDropIndicator", False)
        self.list_color_spaces_HSV.setAlternatingRowColors(False)
        self.list_color_spaces_HSV.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.list_color_spaces_HSV.setLayoutMode(QtWidgets.QListView.SinglePass)
        self.list_color_spaces_HSV.setModelColumn(0)
        self.list_color_spaces_HSV.setObjectName("list_color_spaces_HSV")
        item = QtWidgets.QListWidgetItem()
        item.setText("H")
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        item.setFlags(QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsEnabled)
        self.list_color_spaces_HSV.addItem(item)
        item = QtWidgets.QListWidgetItem()
        item.setText("S")
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        item.setFlags(QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsEnabled)
        self.list_color_spaces_HSV.addItem(item)
        item = QtWidgets.QListWidgetItem()
        item.setText("V")
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        item.setFlags(QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsEnabled)
        self.list_color_spaces_HSV.addItem(item)
        self.horizontalLayout_8.addWidget(self.list_color_spaces_HSV)
        self.list_color_spaces_RGB = QtWidgets.QListWidget(self.frame_sub_spaces)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.list_color_spaces_RGB.sizePolicy().hasHeightForWidth())
        self.list_color_spaces_RGB.setSizePolicy(sizePolicy)
        self.list_color_spaces_RGB.setMinimumSize(QtCore.QSize(0, 85))
        self.list_color_spaces_RGB.setMaximumSize(QtCore.QSize(35, 85))
        self.list_color_spaces_RGB.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.list_color_spaces_RGB.setFrameShadow(QtWidgets.QFrame.Plain)
        self.list_color_spaces_RGB.setLineWidth(3)
        self.list_color_spaces_RGB.setMidLineWidth(2)
        self.list_color_spaces_RGB.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.list_color_spaces_RGB.setProperty("showDropIndicator", False)
        self.list_color_spaces_RGB.setAlternatingRowColors(False)
        self.list_color_spaces_RGB.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.list_color_spaces_RGB.setLayoutMode(QtWidgets.QListView.SinglePass)
        self.list_color_spaces_RGB.setModelColumn(0)
        self.list_color_spaces_RGB.setObjectName("list_color_spaces_RGB")
        item = QtWidgets.QListWidgetItem()
        item.setText("R")
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        item.setFlags(QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsEnabled)
        self.list_color_spaces_RGB.addItem(item)
        item = QtWidgets.QListWidgetItem()
        item.setText("G")
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        item.setFlags(QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsEnabled)
        self.list_color_spaces_RGB.addItem(item)
        item = QtWidgets.QListWidgetItem()
        item.setText("B")
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        item.setFlags(QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsEnabled)
        self.list_color_spaces_RGB.addItem(item)
        self.horizontalLayout_8.addWidget(self.list_color_spaces_RGB)
        self.list_color_spaces_CMYK = QtWidgets.QListWidget(self.frame_sub_spaces)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.list_color_spaces_CMYK.sizePolicy().hasHeightForWidth())
        self.list_color_spaces_CMYK.setSizePolicy(sizePolicy)
        self.list_color_spaces_CMYK.setMinimumSize(QtCore.QSize(0, 85))
        self.list_color_spaces_CMYK.setMaximumSize(QtCore.QSize(35, 85))
        self.list_color_spaces_CMYK.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.list_color_spaces_CMYK.setFrameShadow(QtWidgets.QFrame.Plain)
        self.list_color_spaces_CMYK.setLineWidth(3)
        self.list_color_spaces_CMYK.setMidLineWidth(2)
        self.list_color_spaces_CMYK.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.list_color_spaces_CMYK.setProperty("showDropIndicator", False)
        self.list_color_spaces_CMYK.setAlternatingRowColors(False)
        self.list_color_spaces_CMYK.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.list_color_spaces_CMYK.setLayoutMode(QtWidgets.QListView.SinglePass)
        self.list_color_spaces_CMYK.setModelColumn(0)
        self.list_color_spaces_CMYK.setObjectName("list_color_spaces_CMYK")
        item = QtWidgets.QListWidgetItem()
        item.setText("C")
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        item.setFlags(QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsEnabled)
        self.list_color_spaces_CMYK.addItem(item)
        item = QtWidgets.QListWidgetItem()
        item.setText("M")
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        item.setFlags(QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsEnabled)
        self.list_color_spaces_CMYK.addItem(item)
        item = QtWidgets.QListWidgetItem()
        item.setText("Y")
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        item.setFlags(QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsEnabled)
        self.list_color_spaces_CMYK.addItem(item)
        self.horizontalLayout_8.addWidget(self.list_color_spaces_CMYK)
        self.list_color_spaces_YCbCr = QtWidgets.QListWidget(self.frame_sub_spaces)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.list_color_spaces_YCbCr.sizePolicy().hasHeightForWidth())
        self.list_color_spaces_YCbCr.setSizePolicy(sizePolicy)
        self.list_color_spaces_YCbCr.setMinimumSize(QtCore.QSize(0, 85))
        self.list_color_spaces_YCbCr.setMaximumSize(QtCore.QSize(35, 85))
        self.list_color_spaces_YCbCr.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.list_color_spaces_YCbCr.setFrameShadow(QtWidgets.QFrame.Plain)
        self.list_color_spaces_YCbCr.setLineWidth(3)
        self.list_color_spaces_YCbCr.setMidLineWidth(2)
        self.list_color_spaces_YCbCr.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.list_color_spaces_YCbCr.setProperty("showDropIndicator", False)
        self.list_color_spaces_YCbCr.setAlternatingRowColors(False)
        self.list_color_spaces_YCbCr.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.list_color_spaces_YCbCr.setLayoutMode(QtWidgets.QListView.SinglePass)
        self.list_color_spaces_YCbCr.setModelColumn(0)
        self.list_color_spaces_YCbCr.setObjectName("list_color_spaces_YCbCr")
        item = QtWidgets.QListWidgetItem()
        item.setText("Y")
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        item.setFlags(QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsEnabled)
        self.list_color_spaces_YCbCr.addItem(item)
        item = QtWidgets.QListWidgetItem()
        item.setText("Cb")
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        item.setFlags(QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsEnabled)
        self.list_color_spaces_YCbCr.addItem(item)
        item = QtWidgets.QListWidgetItem()
        item.setText("Cr")
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        item.setFlags(QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsEnabled)
        self.list_color_spaces_YCbCr.addItem(item)
        self.horizontalLayout_8.addWidget(self.list_color_spaces_YCbCr)
        self.horizontalLayout_7.addWidget(self.frame_sub_spaces)
        self.verticalLayout_2.addWidget(self.group_colorspaces_all)
        self.groupBox = QtWidgets.QGroupBox(self.group_analyze)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.slider_compare_limit = QtWidgets.QSlider(self.groupBox)
        self.slider_compare_limit.setMinimum(0)
        self.slider_compare_limit.setMaximum(100)
        self.slider_compare_limit.setSingleStep(1)
        self.slider_compare_limit.setPageStep(4)
        self.slider_compare_limit.setProperty("value", 10)
        self.slider_compare_limit.setOrientation(QtCore.Qt.Horizontal)
        self.slider_compare_limit.setInvertedAppearance(False)
        self.slider_compare_limit.setInvertedControls(False)
        self.slider_compare_limit.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.slider_compare_limit.setTickInterval(4)
        self.slider_compare_limit.setObjectName("slider_compare_limit")
        self.horizontalLayout.addWidget(self.slider_compare_limit)
        self.lbl_compare_limit = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(162)
        sizePolicy.setVerticalStretch(95)
        sizePolicy.setHeightForWidth(self.lbl_compare_limit.sizePolicy().hasHeightForWidth())
        self.lbl_compare_limit.setSizePolicy(sizePolicy)
        self.lbl_compare_limit.setMinimumSize(QtCore.QSize(60, 0))
        self.lbl_compare_limit.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lbl_compare_limit.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_compare_limit.setObjectName("lbl_compare_limit")
        self.horizontalLayout.addWidget(self.lbl_compare_limit)
        self.verticalLayout_2.addWidget(self.groupBox)
        self.gridLayout.addWidget(self.group_analyze, 0, 2, 2, 1)
        self.group_final = QtWidgets.QGroupBox(self.centralwidget)
        self.group_final.setObjectName("group_final")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.group_final)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.check_show_histograms = QtWidgets.QCheckBox(self.group_final)
        self.check_show_histograms.setObjectName("check_show_histograms")
        self.verticalLayout_3.addWidget(self.check_show_histograms)
        self.check_export_histograms = QtWidgets.QCheckBox(self.group_final)
        self.check_export_histograms.setObjectName("check_export_histograms")
        self.verticalLayout_3.addWidget(self.check_export_histograms)
        self.check_create_samples = QtWidgets.QCheckBox(self.group_final)
        self.check_create_samples.setChecked(False)
        self.check_create_samples.setObjectName("check_create_samples")
        self.verticalLayout_3.addWidget(self.check_create_samples)
        self.gridLayout.addWidget(self.group_final, 1, 0, 3, 1)
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setObjectName("groupBox_5")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_5)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.combo_final_sort = QtWidgets.QComboBox(self.groupBox_5)
        self.combo_final_sort.setObjectName("combo_final_sort")
        self.combo_final_sort.addItem("")
        self.combo_final_sort.addItem("")
        self.combo_final_sort.addItem("")
        self.combo_final_sort.addItem("")
        self.verticalLayout_4.addWidget(self.combo_final_sort)
        self.gridLayout.addWidget(self.groupBox_5, 2, 2, 1, 1)
        self.group_move = QtWidgets.QGroupBox(self.centralwidget)
        self.group_move.setObjectName("group_move")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.group_move)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.radio_copy = QtWidgets.QRadioButton(self.group_move)
        self.radio_copy.setChecked(False)
        self.radio_copy.setObjectName("radio_copy")
        self.horizontalLayout_2.addWidget(self.radio_copy)
        self.radio_move = QtWidgets.QRadioButton(self.group_move)
        self.radio_move.setChecked(True)
        self.radio_move.setObjectName("radio_move")
        self.horizontalLayout_2.addWidget(self.radio_move)
        self.gridLayout.addWidget(self.group_move, 3, 2, 1, 1)
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.btn_start = QtWidgets.QPushButton(self.frame)
        self.btn_start.setMinimumSize(QtCore.QSize(130, 60))
        self.btn_start.setObjectName("btn_start")
        self.horizontalLayout_5.addWidget(self.btn_start)
        self.btn_stop = QtWidgets.QPushButton(self.frame)
        self.btn_stop.setMinimumSize(QtCore.QSize(130, 60))
        self.btn_stop.setObjectName("btn_stop")
        self.horizontalLayout_5.addWidget(self.btn_stop)
        self.progressBar = QtWidgets.QProgressBar(self.frame)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout_5.addWidget(self.progressBar)
        self.gridLayout.addWidget(self.frame, 4, 0, 1, 3)
        VisImSorter.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(VisImSorter)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.statusbar.setFont(font)
        self.statusbar.setObjectName("statusbar")
        VisImSorter.setStatusBar(self.statusbar)

        self.retranslateUi(VisImSorter)
        self.list_color_spaces.setCurrentRow(0)
        self.list_color_spaces_RGB.setCurrentRow(-1)
        self.list_color_spaces_CMYK.setCurrentRow(-1)
        self.list_color_spaces_YCbCr.setCurrentRow(-1)
        QtCore.QMetaObject.connectSlotsByName(VisImSorter)

    def retranslateUi(self, VisImSorter):
        _translate = QtCore.QCoreApplication.translate
        VisImSorter.setWindowTitle(_translate("VisImSorter", "Visual Image Sorter"))
        self.group_input.setTitle(_translate("VisImSorter", "Input"))
        self.select_folder_button.setText(_translate("VisImSorter", "Select folder"))
        self.check_subdirs_box.setText(_translate("VisImSorter", "Search subdirectories"))
        self.group_sizes.setTitle(_translate("VisImSorter", "Grouping images"))
        self.groupBox_2.setTitle(_translate("VisImSorter", "Process version"))
        self.V1.setText(_translate("VisImSorter", "V1"))
        self.V2.setText(_translate("VisImSorter", "V2"))
        self.radio_folder_size.setText(_translate("VisImSorter", "Number of files in folders"))
        self.radio_number_folders.setText(_translate("VisImSorter", "Number of folders"))
        self.groupBox_6.setTitle(_translate("VisImSorter", "Equalize folder sizes"))
        self.lbl_slider_enforce_equal.setText(_translate("VisImSorter", "5%"))
        self.label.setText(_translate("VisImSorter", "Minimum # of files in group"))
        self.groupBox_7.setTitle(_translate("VisImSorter", "Consider a group of files as one image"))
        self.check_equal_name.setText(_translate("VisImSorter", "If they have \n"
"equal first symbols\n"
"in their name"))
        self.group_analyze.setTitle(_translate("VisImSorter", "Analisys depth"))
        self.groupBox_4.setTitle(_translate("VisImSorter", "Histogram bands"))
        self.lbl_histogram_bands.setText(_translate("VisImSorter", "5"))
        self.group_colorspaces_all.setTitle(_translate("VisImSorter", "Use colorspaces"))
        __sortingEnabled = self.list_color_spaces.isSortingEnabled()
        self.list_color_spaces.setSortingEnabled(False)
        self.list_color_spaces.setSortingEnabled(__sortingEnabled)
        __sortingEnabled = self.list_color_spaces_HSV.isSortingEnabled()
        self.list_color_spaces_HSV.setSortingEnabled(False)
        self.list_color_spaces_HSV.setSortingEnabled(__sortingEnabled)
        __sortingEnabled = self.list_color_spaces_RGB.isSortingEnabled()
        self.list_color_spaces_RGB.setSortingEnabled(False)
        self.list_color_spaces_RGB.setSortingEnabled(__sortingEnabled)
        __sortingEnabled = self.list_color_spaces_CMYK.isSortingEnabled()
        self.list_color_spaces_CMYK.setSortingEnabled(False)
        self.list_color_spaces_CMYK.setSortingEnabled(__sortingEnabled)
        __sortingEnabled = self.list_color_spaces_YCbCr.isSortingEnabled()
        self.list_color_spaces_YCbCr.setSortingEnabled(False)
        self.list_color_spaces_YCbCr.setSortingEnabled(__sortingEnabled)
        self.groupBox.setTitle(_translate("VisImSorter", "First step optimization"))
        self.lbl_compare_limit.setText(_translate("VisImSorter", "5"))
        self.group_final.setTitle(_translate("VisImSorter", "After grouping done"))
        self.check_show_histograms.setText(_translate("VisImSorter", "Show histograms"))
        self.check_export_histograms.setText(_translate("VisImSorter", "Export histograms"))
        self.check_create_samples.setText(_translate("VisImSorter", "Create sample\n"
"folder images"))
        self.groupBox_5.setTitle(_translate("VisImSorter", "Sort final folders by"))
        self.combo_final_sort.setItemText(0, _translate("VisImSorter", "Average color"))
        self.combo_final_sort.setItemText(1, _translate("VisImSorter", "File count (ascending)"))
        self.combo_final_sort.setItemText(2, _translate("VisImSorter", "File count (descending)"))
        self.combo_final_sort.setItemText(3, _translate("VisImSorter", "Average brightness"))
        self.group_move.setTitle(_translate("VisImSorter", "File operation"))
        self.radio_copy.setText(_translate("VisImSorter", "Copy files"))
        self.radio_move.setText(_translate("VisImSorter", "Move files"))
        self.btn_start.setText(_translate("VisImSorter", "Start"))
        self.btn_stop.setText(_translate("VisImSorter", "Stop"))
