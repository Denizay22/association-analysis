import sys
from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qdarktheme
from PyQt5.QtCore import Qt, QAbstractTableModel, QRect, QMetaObject, QCoreApplication
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWidgets import QLabel, QTableView, QWidget, QVBoxLayout, QHeaderView, QSizePolicy, QGridLayout, QComboBox, \
    QTextBrowser, QApplication, QDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

df_diff_dep = pd.read_csv('csv/diff_dep_class_enrollment_cnt.csv')
df_class_codes = pd.read_csv('csv/class_codes.csv')
df_data = pd.read_csv('csv/data_cleaned.csv', dtype={"std_no" : "string","department_no": "string","department_code": "string","class_department_code": "string", "class_code": "string","class_name": "string","period": "string"})
df_voc_el = pd.read_csv('csv/data_voc_el.csv', dtype={"std_no" : "string","department_no": "string","department_code": "string","class_department_code": "string", "class_code": "string","class_name": "string","period": "string", "cnt": "string"})
df_soc_el = pd.read_csv('csv/data_social_el.csv', dtype={"std_no" : "string","department_no": "string","department_code": "string","class_department_code": "string", "class_code": "string","class_name": "string","period": "string", "cnt": "string"})
df_mandatory = pd.read_csv('csv/data_mandatory.csv', dtype={"std_no" : "string","department_no": "string","department_code": "string","class_department_code": "string", "class_code": "string","class_name": "string","period": "string", "cnt": "string"})

df_dict = {}
df_dict['M'] = df_voc_el
df_dict['S'] = df_soc_el


vocational_elective_courses = pd.read_csv("csv/vocational_elective_courses.csv", 
                               dtype={'department_code': 'string'}, 
                               converters={"vocational_elective_courses": literal_eval})

mandatory_courses = pd.read_csv("csv/mandatory_courses.csv",
                                dtype={'department_code': 'string'},
                                converters={"mandatory_courses": literal_eval})

social_elective_courses = pd.read_csv("csv/social_elective_courses.csv",
                                dtype={'department_code': 'string'},
                                converters={"social_elective_courses": literal_eval})

dep_dict = dict(pd.read_csv('csv/department_dict.csv').set_index('department_code').T.to_dict('list'))

def dep_records_by_code(dep_code, df_in):
    """verilen bölüm kısaltmasının ders kayıtlarını içeren tablo döndürür"""
    """ilk değişken tablo ikinci değişken tablo uzunlugu döndürür"""
    tmp = df_in[df_in.department_code == dep_code].groupby('std_no').class_name.apply(list).reset_index()
    return tmp, tmp.shape[0]

def different_dep_class_enrollment_cnt(dep_code, df_in):
    df = df_in[df_in.department_code == dep_code]
    df = df[df.department_code != df.class_department_code]
    df['to_drop'] = df.apply(lambda row: 
                        row.class_code in social_elective_courses[social_elective_courses.department_code == row.department_code].reset_index().at[0, 'social_elective_courses'] or 
                        row.class_code in vocational_elective_courses[vocational_elective_courses.department_code == row.department_code].reset_index().at[0, 'vocational_elective_courses'] or 
                        row.class_code in mandatory_courses[mandatory_courses.department_code == row.department_code].reset_index().at[0, 'mandatory_courses'], axis=1)
    df = df[df.to_drop == False].reset_index().drop(['index', 'to_drop'], axis=1)
    df = df[['department_no', 'department_code', 'class_department_code', 'class_code', 'class_name']]
    df['cnt'] = df.groupby('class_department_code')['class_department_code'].transform('count')
    df = df[['class_department_code', 'cnt']]
    df = df.drop_duplicates()
    df = df[df['class_department_code'] != 'ITB']
    df = df.sort_values('cnt', ascending=False).reset_index().drop('index',axis=1)
    return df

def class_enrollment_cnt(dep_code, df_in):
    """verilen bölüm kodunun ders kayıtlarının büyükten küçüğe doğru sıralı bir şekilde
    her ders için kaç kayıt olduğunu döndürür"""
    df = df_in[df_in.department_code == dep_code]
    df = df[['department_no', 'department_code', 'class_department_code', 'class_code', 'class_name', 'period']]
    df['cnt'] = df.groupby(['class_code', 'period'])['class_code'].transform('count')
    df = df.drop_duplicates(subset=df.columns.difference(['class_name']))
    df = df.sort_values('cnt', ascending=False).reset_index().drop('index',axis=1)
    return df

def oneshot_df(dep_str, df_in):
    """verilen bölümün kayıtlarının oneshot table'ını oluşturur"""
    dep_records, _ = dep_records_by_code(dep_str, df_in)
    dep_records_class_codes_list = dep_records.class_name.to_list()
    te = TransactionEncoder()
    dep_records_oneshot = te.fit(dep_records_class_codes_list).transform(dep_records_class_codes_list)
    dep_records_oneshot_df = pd.DataFrame(dep_records_oneshot, columns=te.columns_)
    
    return dep_records_oneshot_df

def create_frequent_itemsets(dep_str, df_in):
    """verilen bölümün kayıtlarına göre frequent itemsets oluşturur"""
    dep_oneshot_df = oneshot_df(dep_str, df_in)
    dep_freq_itemsets = apriori(dep_oneshot_df, min_support = 0.01, use_colnames=True)
    return dep_freq_itemsets

def create_rules(dep_str, df_in):
    """verilen bölüm ve verilen ders için öneri yapar"""
    dep_freq_itemsets = create_frequent_itemsets(dep_str, df_in)
    rules = association_rules(dep_freq_itemsets, metric="lift", min_threshold=1)
    if len(rules) == 0:
        return rules
    rules = rules.sort_values('confidence', ascending=False)
    rules = rules.reset_index().drop('index', axis=1)
    _, cnt = dep_records_by_code(dep_str, df_in)
    rules["cnt"] = rules.apply(lambda row: round(cnt * row["support"]), axis=1)
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(map(str, x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(map(str, x)))
    rules = rules[rules["antecedents"].str.contains(',') == False]
    rules = rules[rules["consequents"].str.contains(',') == False]
    rules['antecedent support'] = rules['antecedent support'].apply(lambda x: round(x, 2))
    rules['consequent support'] = rules['consequent support'].apply(lambda x: round(x, 2))
    rules['support'] = rules['support'].apply(lambda x: round(x, 2))
    rules['confidence'] = rules['confidence'].apply(lambda x: round(x, 2))
    rules['lift'] = rules['lift'].apply(lambda x: round(x, 2))
    return rules

def rules10(dep_str, df, antecedents=""):
    """bölüm için veya ders için top10 kuralı döndürür\n
    antecedents girilmezse bölüm, girilirse girilen antecedents için oluşturulu\n
    antecedents list şeklinde olmalıdır. örn: ['Veri Madenciliğine Giriş']"""
    df = create_rules(dep_str, df)
    print(len(df))
    if len(df) == 0:
        df[['Antecedents', 'Consequents', 'Antec. Sup.', 'Conseq. Sup.', 'Support', 'Confidence', 'Count']] = ""
        return df[['Antecedents', 'Consequents', 'Antec. Sup.', 'Conseq. Sup.', 'Support', 'Confidence', 'Count']]
    if len(antecedents) > 0:
        df['to_drop'] = df.apply(lambda row: row.antecedents.find(antecedents), axis=1)
        df = df[df['to_drop'] == False].drop('to_drop', axis=1)
    df = df.sort_values('support', ascending=False)
    df = df.head(10)
    df = df.sort_values('confidence', ascending=False)
    df = df.reset_index().drop('index', axis=1)

    df[['Antecedents', 'Consequents', 'Antec. Sup.', 'Conseq. Sup.', 'Support', 'Confidence', 'Count']]  \
    = df[['antecedents', 'consequents', 'antecedent support', 'consequent support', 'support', 'confidence', 'cnt']]
    return df[['Antecedents', 'Consequents', 'Antec. Sup.', 'Conseq. Sup.', 'Support', 'Confidence', 'Count']]

def pie_inputs_normal(dep_str, df_in, class_str="", cnt_in=10, text_len=20):
    """dep_str: bölüm kısaltması\n
    df_in: df_voc_el/df_soc_el\n
    class_str:class_str"""
    df = class_enrollment_cnt(dep_str, df_in)
    if len(df) < cnt_in:
        cnt = len(df)
    else:
        cnt = cnt_in
    
    colors = ['#046dc8', '#1184a7', '#ea515f', '#b4418e', '#d94a8c', '#15a2a2', '#0450b4', '#6fb1a0', '#fe7434', '#fea802', '#d94a8c', '#0450b4']
    counts = df.head(cnt)['cnt'].to_list()
    explode = list(np.full(cnt, 0.02))
    labels_ = df.head(cnt)['class_name'].to_list()
    labels = list()
    for i in range(0, cnt):
        if len(df.at[i, 'class_name']) > text_len:
            labels.append(f"{df.at[i, 'class_name'][:text_len]}... [{df_class_codes[df_class_codes.class_name==df.at[i, 'class_name']].reset_index().at[0, 'class_code']}][{df.at[i, 'period'][:1]}]")
        else:
            labels.append(f"{df.at[i, 'class_name']} [{df_class_codes[df_class_codes.class_name==df.at[i, 'class_name']].reset_index().at[0, 'class_code']}][{df.at[i, 'period'][:1]}]")
    
    if class_str!="":
        tmp_df = df[df['class_name'] == class_str]
        if class_str not in labels_:
            labels_.append(class_str)
            for row in tmp_df.iterrows():
                if len(row[1][4]) > text_len:
                    labels.append(f"{row[1][4][:text_len]}... [{df_class_codes[df_class_codes.class_name==row[1][4]].reset_index().at[0, 'class_code']}][{row[1][5][:1]}]")
                else:
                    labels.append(f"{row[1][4]} [{df_class_codes[df_class_codes.class_name==row[1][4]].reset_index().at[0, 'class_code']}][{row[1][5][:1]}]")
                counts.append(row[1][6])
                explode.append(0.2)
        else:
            for row in tmp_df.iterrows():
                try:
                    explode[row[0]] = 0.2
                except:
                    #hata verirse o satırı eklicez
                    if len(row[1][4]) > text_len:
                        labels.append(f"{row[1][4][:text_len]}... [{df_class_codes[df_class_codes.class_name==row[1][4]].reset_index().at[0, 'class_code']}][{row[1][5][:1]}]")
                    else:
                        labels.append(f"{row[1][4]} [{df_class_codes[df_class_codes.class_name==row[1][4]].reset_index().at[0, 'class_code']}][{row[1][5][:1]}]")
                    counts.append(row[1][6])
                    explode.append(0.2)
    
    if (sum(df['cnt'].to_list()) - sum(counts)) > 0:
        labels.append('Diğerleri')
        counts.append(sum(df['cnt'].to_list()) - sum(counts))
        explode.append(0.02)

    colors = colors[:len(counts)-1]
    colors.append('#b4418e')
    return counts, labels, colors, explode

def pie_inputs_diff(dep_str, cnt_in=10, text_len=20):
    """dep_str: bölüm kısaltması"""
    df = df_diff_dep[df_diff_dep['department_code'] == dep_str]
    if len(df) < cnt_in:
        cnt = len(df)
    else:
        cnt = cnt_in
    colors = ['#046dc8', '#1184a7', '#ea515f', '#b4418e', '#d94a8c', '#15a2a2', '#0450b4', '#6fb1a0', '#fe7434', '#fea802', '#d94a8c', '#0450b4']
    
    explode = list(np.full(cnt, 0.02))
    counts = df.head(cnt)['cnt'].to_list()
    labels = list()

    for row in df.head(cnt).iterrows():
        if len(dep_dict[row[1][0]][0]) > text_len:
            labels.append(f"{dep_dict[row[1][1]][0][:text_len]}... [{row[1][1]}]")
        else:
            labels.append(f"{dep_dict[row[1][1]][0]} [{row[1][1]}]")
    
    if (sum(df['cnt'].to_list()) - sum(counts)) > 0:
        labels.append('Diğerleri')
        counts.append(sum(df['cnt'].to_list()) - sum(counts))
        explode.append(0.02)
    
    colors = colors[:len(counts)-1]
    colors.append('#b4418e')
    return counts, labels, colors, explode

def pie_inputs_rules(df_in, cnt_in=10, text_len=20):
    """df_in: rules10(dep_str, df_voc_el/df_soc_el, ["class_str"])"""
    if len(df_in) < cnt_in:
        cnt = len(df_in)
    else:
        cnt = cnt_in
    colors = ['#046dc8', '#1184a7', '#ea515f', '#b4418e', '#d94a8c', '#15a2a2', '#0450b4', '#6fb1a0', '#fe7434', '#fea802', '#b4418e', '#d94a8c', '#0450b4']
    explode = list(np.full(cnt, 0.02))
    counts = list()  
    labels = list()

    for row in df_in.iterrows():
        if len(row[1][1])>text_len:
            labels.append(f"{row[1][1][:text_len]}... [{df_class_codes[df_class_codes.class_name==row[1][1]].reset_index().at[0, 'class_code']}]")
        else:
            labels.append(f"{row[1][1]} [{df_class_codes[df_class_codes.class_name==row[1][1]].reset_index().at[0, 'class_code']}]")
        counts.append(row[1][6])
    
    return counts, labels, colors, explode

class pieChart(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        self.figure = plt.figure()
        self.figure.patch.set_facecolor('None')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.axes = self.figure.add_subplot(111)
        self.axes.set_visible(False)
        v_box = QVBoxLayout()
        v_box.addWidget(self.canvas)
        self.setLayout(v_box)

class pandasModel(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        
        self._data = data
    
    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]
    
    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        try:
            if orientation == Qt.Horizontal and role == Qt.DisplayRole:
                return self._data.columns[col]
            return None
        except:
            pass

class Ui_Form(object):
    def init(self, Form):
        Form.setObjectName("Form")
        Form.resize(1920, 1000)
        

        #combobox
        self.gridLayoutWidget = QWidget(Form)
        self.gridLayoutWidget.setGeometry(QRect(10, 290, 1891, 24))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridlayout_combo = QGridLayout(self.gridLayoutWidget)
        self.gridlayout_combo.setContentsMargins(0, 0, 0, 0)
        self.gridlayout_combo.setObjectName("gridlayout_combo")
        self.class_combo = QComboBox(self.gridLayoutWidget)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.class_combo.sizePolicy().hasHeightForWidth())
        self.class_combo.setSizePolicy(sizePolicy)
        self.class_combo.setObjectName("class_combo")
        self.gridlayout_combo.addWidget(self.class_combo, 0, 1, 1, 1)
        self.decision_combo = QComboBox(self.gridLayoutWidget)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.decision_combo.sizePolicy().hasHeightForWidth())
        self.decision_combo.setSizePolicy(sizePolicy)
        self.decision_combo.setObjectName("decision_combo")
        self.gridlayout_combo.addWidget(self.decision_combo, 0, 3, 1, 1)
        self.dep_combo = QComboBox(self.gridLayoutWidget)
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dep_combo.sizePolicy().hasHeightForWidth())
        self.dep_combo.setSizePolicy(sizePolicy)
        self.dep_combo.setObjectName("dep_combo")
        self.gridlayout_combo.addWidget(self.dep_combo, 0, 0, 1, 1)
        

        #piecharts
        self.gridLayoutWidget_2 = QWidget(Form)
        self.gridLayoutWidget_2.setGeometry(QRect(10, 410, 1891, 251))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridlayout_piechart = QGridLayout(self.gridLayoutWidget_2)
        self.gridlayout_piechart.setContentsMargins(0, 0, 0, 0)
        self.gridlayout_piechart.setVerticalSpacing(7)
        self.gridlayout_piechart.setObjectName("gridlayout_piechart")
        self.class_chart1 = pieChart(self.gridLayoutWidget_2)
        self.class_chart1.setObjectName("class_chart1")
        self.gridlayout_piechart.addWidget(self.class_chart1, 0, 2, 1, 1)
        self.class_chart2 = pieChart(self.gridLayoutWidget_2)
        self.class_chart2.setObjectName("class_chart2")
        self.gridlayout_piechart.addWidget(self.class_chart2, 0, 3, 1, 1)
        self.dep_chart2 = pieChart(self.gridLayoutWidget_2)
        self.dep_chart2.setObjectName("dep_chart2")
        self.gridlayout_piechart.addWidget(self.dep_chart2, 0, 1, 1, 1)
        self.dep_chart1 = pieChart(self.gridLayoutWidget_2)
        self.dep_chart1.setObjectName("dep_chart1")
        self.gridlayout_piechart.addWidget(self.dep_chart1, 0, 0, 1, 1)


        #tables
        self.gridLayoutWidget_3 = QWidget(Form)
        self.gridLayoutWidget_3.setGeometry(QRect(10, 740, 1891, 261))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        self.gridLayout_table = QGridLayout(self.gridLayoutWidget_3)
        self.gridLayout_table.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_table.setObjectName("gridLayout_table")
        self.dep_table_model = pandasModel(pd.DataFrame())
        self.dep_table = QTableView(self.gridLayoutWidget_3)
        self.dep_table.setObjectName("dep_table")
        self.gridLayout_table.addWidget(self.dep_table, 0, 0, 1, 1)
        self.class_table_model = pandasModel(pd.DataFrame())
        self.class_table = QTableView(self.gridLayoutWidget_3)
        self.class_table.setObjectName("class_table")
        self.gridLayout_table.addWidget(self.class_table, 0, 1, 1, 1)


        #combobox texts
        self.gridLayoutWidget_4 = QWidget(Form)
        self.gridLayoutWidget_4.setGeometry(QRect(10, 250, 1891, 41))
        self.gridLayoutWidget_4.setObjectName("gridLayoutWidget_4")
        self.gridLayout_text = QGridLayout(self.gridLayoutWidget_4)
        self.gridLayout_text.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_text.setObjectName("gridLayout_text")
        self.text_class = QTextBrowser(self.gridLayoutWidget_4)
        self.text_class.setObjectName("text_class")
        self.gridLayout_text.addWidget(self.text_class, 0, 1, 1, 1)
        self.text_dep = QTextBrowser(self.gridLayoutWidget_4)
        self.text_dep.setObjectName("text_dep")
        self.gridLayout_text.addWidget(self.text_dep, 0, 0, 1, 1)
        self.text_decision = QTextBrowser(self.gridLayoutWidget_4)
        self.text_decision.setObjectName("text_decision")
        self.gridLayout_text.addWidget(self.text_decision, 0, 2, 1, 1)


        #piechart texts
        self.gridLayoutWidget_5 = QWidget(Form)
        self.gridLayoutWidget_5.setGeometry(QRect(10, 370, 1891, 41))
        self.gridLayoutWidget_5.setObjectName("gridLayoutWidget_5")
        self.grid_pietytext = QGridLayout(self.gridLayoutWidget_5)
        self.grid_pietytext.setContentsMargins(0, 0, 0, 0)
        self.grid_pietytext.setObjectName("grid_pietytext")
        self.dep_counttext = QTextBrowser(self.gridLayoutWidget_5)
        self.dep_counttext.setObjectName("dep_counttext")
        self.grid_pietytext.addWidget(self.dep_counttext, 0, 0, 1, 1)
        self.dif_counttext = QTextBrowser(self.gridLayoutWidget_5)
        self.dif_counttext.setObjectName("dif_counttext")
        self.grid_pietytext.addWidget(self.dif_counttext, 0, 1, 1, 1)
        self.dep_class_counttext = QTextBrowser(self.gridLayoutWidget_5)
        self.dep_class_counttext.setObjectName("dep_class_counttext")
        self.grid_pietytext.addWidget(self.dep_class_counttext, 0, 2, 1, 1)
        self.rule_counttext = QTextBrowser(self.gridLayoutWidget_5)
        self.rule_counttext.setObjectName("rule_counttext")
        self.grid_pietytext.addWidget(self.rule_counttext, 0, 3, 1, 1)
        



        #table texts
        self.gridLayoutWidget_6 = QWidget(Form)
        self.gridLayoutWidget_6.setGeometry(QRect(10, 700, 1891, 41))
        self.gridLayoutWidget_6.setObjectName("gridLayoutWidget_6")
        self.grid_tabletext = QGridLayout(self.gridLayoutWidget_6)
        self.grid_tabletext.setContentsMargins(0, 0, 0, 0)
        self.grid_tabletext.setObjectName("grid_tabletext")
        self.dep_tabletext = QTextBrowser(self.gridLayoutWidget_6)
        self.dep_tabletext.setObjectName("dep_tabletext")
        self.grid_tabletext.addWidget(self.dep_tabletext, 0, 0, 1, 1)
        self.class_tabletext = QTextBrowser(self.gridLayoutWidget_6)
        self.class_tabletext.setObjectName("class_tabletext")
        self.grid_tabletext.addWidget(self.class_tabletext, 0, 1, 1, 1)


        #ytü image
        self.gridLayoutWidget_7 = QWidget(Form)
        self.gridLayoutWidget_7.setGeometry(QRect(10, 20, 231, 211))
        self.gridLayoutWidget_7.setObjectName("gridLayoutWidget_7")
        self.gridLayout = QGridLayout(self.gridLayoutWidget_7)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.ytu_image = QLabel()
        pixmap = QPixmap('images/amblem.png')
        pixmap = pixmap.scaled(231, 231, Qt.KeepAspectRatio)
        self.ytu_image.setPixmap(pixmap)
        self.ytu_image.setObjectName("widget")
        self.gridLayout.addWidget(self.ytu_image, 0, 1, 1, 1)
        
        #isimler
        self.names = QTextBrowser(Form)
        self.names.setGeometry(QRect(1640, 90, 256, 91))
        self.names.setObjectName("textBrowser")



        self.text_decision.setWindowFlags(Qt.FramelessWindowHint)
        self.text_decision.setAttribute(Qt.WA_TranslucentBackground)
        self.text_decision.setStyleSheet("background:transparent;")

        self.text_dep.setWindowFlags(Qt.FramelessWindowHint)
        self.text_dep.setAttribute(Qt.WA_TranslucentBackground)
        self.text_dep.setStyleSheet("background:transparent;")

        self.text_class.setWindowFlags(Qt.FramelessWindowHint)
        self.text_class.setAttribute(Qt.WA_TranslucentBackground)
        self.text_class.setStyleSheet("background:transparent;")

        self.class_table.setWindowFlags(Qt.FramelessWindowHint)
        self.class_table.setAttribute(Qt.WA_TranslucentBackground)
        self.class_table.setStyleSheet("background:transparent;")

        self.dep_table.setWindowFlags(Qt.FramelessWindowHint)
        self.dep_table.setAttribute(Qt.WA_TranslucentBackground)
        self.dep_table.setStyleSheet("background:transparent;")

        self.class_chart1.setWindowFlags(Qt.FramelessWindowHint)
        self.class_chart1.setAttribute(Qt.WA_TranslucentBackground)
        self.class_chart1.setStyleSheet("background:transparent;")

        self.class_chart2.setWindowFlags(Qt.FramelessWindowHint)
        self.class_chart2.setAttribute(Qt.WA_TranslucentBackground)
        self.class_chart2.setStyleSheet("background:transparent;")

        self.dep_chart1.setWindowFlags(Qt.FramelessWindowHint)
        self.dep_chart1.setAttribute(Qt.WA_TranslucentBackground)
        self.dep_chart1.setStyleSheet("background:transparent;")

        self.dep_chart2.setWindowFlags(Qt.FramelessWindowHint)
        self.dep_chart2.setAttribute(Qt.WA_TranslucentBackground)
        self.dep_chart2.setStyleSheet("background:transparent;")

        self.dep_counttext.setWindowFlags(Qt.FramelessWindowHint)
        self.dep_counttext.setAttribute(Qt.WA_TranslucentBackground)
        self.dep_counttext.setStyleSheet("background:transparent;")

        self.dep_class_counttext.setWindowFlags(Qt.FramelessWindowHint)
        self.dep_class_counttext.setAttribute(Qt.WA_TranslucentBackground)
        self.dep_class_counttext.setStyleSheet("background:transparent;")

        self.dif_counttext.setWindowFlags(Qt.FramelessWindowHint)
        self.dif_counttext.setAttribute(Qt.WA_TranslucentBackground)
        self.dif_counttext.setStyleSheet("background:transparent;")

        self.rule_counttext.setWindowFlags(Qt.FramelessWindowHint)
        self.rule_counttext.setAttribute(Qt.WA_TranslucentBackground)
        self.rule_counttext.setStyleSheet("background:transparent;")

        self.dep_tabletext.setWindowFlags(Qt.FramelessWindowHint)
        self.dep_tabletext.setAttribute(Qt.WA_TranslucentBackground)
        self.dep_tabletext.setStyleSheet("background:transparent;")

        self.class_tabletext.setWindowFlags(Qt.FramelessWindowHint)
        self.class_tabletext.setAttribute(Qt.WA_TranslucentBackground)
        self.class_tabletext.setStyleSheet("background:transparent;")

        self.names.setWindowFlags(Qt.FramelessWindowHint)
        self.names.setAttribute(Qt.WA_TranslucentBackground)
        self.names.setStyleSheet("background:transparent;")

        self.retranslateUi(Form)
        QMetaObject.connectSlotsByName(Form)

        self.setupUi()
        
    def retranslateUi(self, Form):
        _translate = QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Ders Kayıt Analiz Programı"))
        self.text_class.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:600; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt; font-weight:400;\">Ders Seçimi</span></p></body></html>"))
        self.text_dep.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">Bölüm Seçimi</span></p></body></html>"))
        self.text_decision.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">Mesleki/Sosyal Seçimi</span></p></body></html>"))
        self.dep_counttext.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">Bölüm Ders Kayıtları</span></p></body></html>"))
        self.dif_counttext.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">Farklı Bölümden Ders Kayıtları</span></p></body></html>"))
        self.dep_class_counttext.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">Seçilen Ders Kayıtları</span></p></body></html>"))
        self.rule_counttext.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">Ders Birliktelik Kuralları</span></p></body></html>"))
        self.dep_tabletext.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">Bölüm Birliktelik Kuralları</span></p></body></html>"))
        self.class_tabletext.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">Ders Birliktelik Kuralları Tablosu</span></p></body></html>"))
        self.names.setHtml(_translate("Form", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-style:italic;\">Hazırlayanlar</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-style:italic;\">Denizay BİBEROĞLU</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-style:italic;\">Yusuf Yemliha ÇELİK</span></p></body></html>"))

    def setupUi(self):
        self.dep_selection = ""
        self.class_selection = ""
        self.voc_soc_selection = "M"
        self.decision_combo.addItems(['Mesleki Seçmeliler', 'Sosyal Seçmeliler'])
        self.fill_dep_combo()
        self.dep_combo.currentTextChanged.connect(self.fill_classes_combo)
        self.dep_combo.currentTextChanged.connect(self.fill_dep_table)
        self.class_combo.currentTextChanged.connect(self.draw_class_chart1)
        self.class_combo.currentTextChanged.connect(self.draw_class_chart2)
        self.class_combo.currentTextChanged.connect(self.fill_class_table)
        self.decision_combo.currentTextChanged.connect(self.set_selection)

    def set_selection(self, s):
        self.voc_soc_selection = s[0]
        if self.dep_selection == "":
            return
        self.fill_dep_table(self.dep_selection)
        self.fill_class_table("")
        self.class_combo.clear()
        self.fill_classes_combo(self.dep_selection)
        self.draw_dep_chart1()
        self.draw_dep_chart2()
        self.draw_class_chart1("")
        self.draw_class_chart2("")
        print(self.voc_soc_selection)

    def fill_dep_table(self, s):
        if s == "":
            self.dep_table_model.__init__(pd.DataFrame())
            self.dep_table.setModel(self.dep_table_model)
            self.dep_table.show()
            return
        rules_table = rules10(self.dep_selection, df_dict[self.voc_soc_selection])
        if len(rules_table) == 0:
            self.dep_table_model.__init__(rules_table.drop('Count', axis=1))
            self.dep_table.setModel(self.dep_table_model)
            self.dep_table.show()
            return
        self.dep_table_model.__init__(rules_table.drop('Count', axis=1))
        self.dep_table.setModel(self.dep_table_model)
        self.dep_table.show()

        header = self.dep_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)

    def fill_class_table(self, s):
        if s == "" or self.dep_selection == "":
            self.class_table_model.__init__(pd.DataFrame())
            self.class_table.setModel(self.class_table_model)
            self.class_table.show()
            return
        rules_table = rules10(self.dep_selection, df_dict[self.voc_soc_selection], antecedents=s)
        if len(rules_table) == 0:
            self.class_table_model.__init__(rules_table)
            self.class_table.setModel(self.class_table_model)
            self.class_table.show()
            return

        self.class_table_model.__init__(rules_table)
        self.class_table.setModel(self.class_table_model)
        self.class_table.show()
        header = self.class_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
    
    def fill_dep_combo(self):
        self.dep_combo.addItem("")
        for k,v in dep_dict.items():
           self.dep_combo.addItem(v[0])

    def fill_classes_combo(self, s):
        if s == "":
            self.dep_chart1.axes.clear()
            self.dep_chart1.axes.set_visible(False)
            self.dep_chart1.canvas.draw()
            self.class_combo.clear()
            self.dep_selection = ""
            self.draw_dep_chart1()
            self.draw_dep_chart2()
            return
        self.class_combo.clear()
        if len(s) > 3:
            value = {i for i in dep_dict if s in dep_dict[i]}
            dep_code = list(value)[0]
        else:
            dep_code = s
        self.dep_selection = dep_code
        classes = df_dict[self.voc_soc_selection][df_dict[self.voc_soc_selection]['department_code'] == dep_code]['class_name'].drop_duplicates().to_list()
        self.class_combo.addItem("")
        for c in classes:
            self.class_combo.addItem(c)
        
        self.draw_dep_chart1()
        self.draw_dep_chart2()

    def draw_dep_chart1(self):
        if self.dep_selection=="":
            self.dep_chart1.axes.clear()
            self.dep_chart1.axes.set_visible(False)
            self.dep_chart1.canvas.draw()
            return

        
        counts, labels, colors, explode = pie_inputs_normal(self.dep_selection, df_dict[self.voc_soc_selection])

        self.dep_chart1.axes.clear()
        self.dep_chart1.axes.set_visible(True)
        _, texts, autotexts = self.dep_chart1.axes.pie(counts, labels=labels, startangle=45, autopct=lambda p: '{:.0f}'.format(p/100*sum(counts)), colors=colors, pctdistance=0.80, explode=explode)
        for text in texts:
            text.set_size('xx-small')
            text.set_color(colors[labels.index(text.get_text()) % len(colors)])
    
        for autotext in autotexts:
            autotext.set_size('xx-small')
            autotext.set_color('#113322')
    
        self.dep_chart1.canvas.draw()

    def draw_dep_chart2(self):
        if self.dep_selection=="":
            self.dep_chart2.axes.clear()
            self.dep_chart2.axes.set_visible(False)
            self.dep_chart2.canvas.draw()
            return

        counts, labels, colors, explode = pie_inputs_diff(self.dep_selection)

        self.dep_chart2.axes.clear()
        self.dep_chart2.axes.set_visible(True)
        _, texts, autotexts = self.dep_chart2.axes.pie(counts, labels=labels, startangle=45, autopct=lambda p: '{:.0f}'.format(p/100*sum(counts)), colors=colors, pctdistance=0.80, explode=explode)
        for text in texts:
            text.set_size('xx-small')
            text.set_color(colors[labels.index(text.get_text()) % len(colors)])
            
    
        for autotext in autotexts:
            autotext.set_size('xx-small')
            autotext.set_color('#113322')

        self.dep_chart2.canvas.draw()
    
    def draw_class_chart1(self, s):
        if s=="":
            self.class_chart1.axes.clear()
            self.class_chart1.axes.set_visible(False)
            self.class_chart1.canvas.draw()
            return

        
        counts, labels, colors, explode = pie_inputs_normal(self.dep_selection, df_dict[self.voc_soc_selection], class_str=s)
        self.class_chart1.axes.clear()
        self.class_chart1.axes.set_visible(True)
        _, texts, autotexts = self.class_chart1.axes.pie(counts, labels=labels, startangle=45, autopct=lambda p: '{:.0f}'.format(p/100*sum(counts)), colors=colors, pctdistance=0.80, explode=explode)
        for text in texts:
            text.set_size('xx-small')
            text.set_color(colors[labels.index(text.get_text()) % len(colors)])
            if text.get_text()[:-16] == s[:len(text.get_text()[:-16])]:
                text.set_weight('bold')

        for autotext in autotexts:
            autotext.set_size('xx-small')
            autotext.set_color('#113322')
    
        self.class_chart1.canvas.draw()

    def draw_class_chart2(self, s):
        if s=="":
            self.class_chart2.axes.clear()
            self.class_chart2.axes.set_visible(False)
            self.class_chart2.canvas.draw()
            return

        df = rules10(self.dep_selection, df_dict[self.voc_soc_selection], s)
        if len(df) == 0:
            return
        counts, labels, colors, explode = pie_inputs_rules(df_in=df)
        self.class_chart2.axes.clear()
        self.class_chart2.axes.set_visible(True)
        _, texts, autotexts = self.class_chart2.axes.pie(counts, labels=labels, startangle=45, autopct=lambda p: '{:.0f}'.format(p/100*sum(counts)), colors=colors, pctdistance=0.80, explode=explode)
        for text in texts:
            text.set_size('xx-small')
            text.set_color(colors[labels.index(text.get_text()) % len(colors)])
            if text.get_text()[:-16] == s[:len(text.get_text()[:-16])]:
                text.set_weight('bold')

        for autotext in autotexts:
            autotext.set_size('xx-small')
            autotext.set_color('#113322')
    
        self.class_chart2.canvas.draw()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    qdarktheme.setup_theme()
    Dialog = QDialog()
    ui = Ui_Form()
    ui.init(Dialog)
    Dialog.show()
    sys.exit(app.exec_())