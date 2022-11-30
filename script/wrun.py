import pandas as pd
import glob
import numpy as np
import os

h_path = "./吉村さんESN/"

mode_name = ["training", "validation", "test"]
files = glob.glob("./resamples/*")
member_no = 1
for file in files:
    data = pd.read_csv(file)
    data_nothing = data[data["nasi"] == 1]
    #data_sita_n = data[data["sita/n"]==1]
    #data_sita_p = data[data["sita/p"]==1]
    #data_zenmen_n = data[data["zenmen/n"]==1]
    #data_zenmen_p = data[data["zenmen/p"]==1]
    data_musi_n = data[data["musi/n"] == 1]
    #data_musi_p = data[data["musi/p"] == 1]
    #data_sita_p = data[data["sita/p"] == 1]
    data_nothing = data_nothing[["time", "average_std"]]
    #data_sita_n = data_sita_n[["time", "average_std"]]
    #data_sita_p = data_sita_p[["time", "average_std"]]
    #data_zenmen_n = data_zenmen_n[["time", "average_std"]]
    #data_zenmen_p = data_zenmen_p[["time", "average_std"]]
    data_musi_n = data_musi_n[["time", "average_std"]]
    #data_musi_p = data_musi_p[["time", "average_std"]]
    
    
    for class_name, class_data in [("nasi", data_nothing),("musi_n", data_musi_n)]:
        input_data, teacher_data = [], []
        prev_index, prev_row = -1, -1
        data_no = 0
         
        # 1行ずつ読み込む
        for index, row in class_data.iterrows():
            # 連番でなくなった場合（index,prev_indexの値が飛んだ場合）は、データを分ける
            if prev_index != -1 and (index - prev_index) > 1:
                #output_data = pd.DataFrame(teacher_data[9:], input_data[:-9])
                output_data = pd.DataFrame(teacher_data, input_data)
                os.makedirs(class_name + "_" + mode_name[data_no] , exist_ok=True)
                #csv形式で出力
                output_data.to_csv(class_name + "_" + mode_name[data_no] + str(member_no) + ".csv",header=False)
                input_data, teacher_data = [], []
                prev_row = -1
                data_no += 1
                
            if type(prev_row) != int:
                input_data.append(prev_row["average_std"])
                teacher_data.append(row["average_std"])
            prev_index = index
            prev_row = row
        #output_data = pd.DataFrame(teacher_data[9:], input_data[:-9])
        output_data = pd.DataFrame(teacher_data, input_data)
        output_data.to_csv(class_name + "_" + mode_name[data_no] + str(member_no)  + ".csv",header=False)
        os.makedirs(class_name + "_" + mode_name[data_no] , exist_ok=True)
      
    member_no += 1
    print(file)
    
    



