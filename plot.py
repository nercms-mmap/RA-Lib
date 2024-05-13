import csv
import matplotlib.pyplot as plt

file_path = "C:\Users\waii\Desktop\re-ID.csv"

with open(file_path, "r") as f:         
    reader = csv.reader(f)              
    list_data = list(reader)            
    f.close()

rows = len(list_data)                   # 行数
cols = len(list_data[0])                # 列数
print("rows =", rows)
print("cols =", cols)


col_0 = []
col_1 = []
col_2 = []

if (cols > 2):
    for i in range(0, rows):
        col_2.append(list_data[i][2])
        col_1.append(list_data[i][1])
        col_0.append(list_data[i][0])
elif (cols > 1):
    for i in range(0, rows):
        col_1.append(list_data[i][1])
        col_0.append(list_data[i][0])
else:
    for i in range(0, rows):
        col_0.append(list_data[i][0])

data_col_0=[int(x) for x in col_0]     
data_col_1=[int(x) for x in col_1]      
data_col_2=[int(x) for x in col_2]      

if (cols > 2):
    plt.subplot(311)
    plt.plot(data_col_0)
    plt.subplot(312)
    plt.plot(data_col_1)
    plt.subplot(313)
    plt.plot(data_col_2)
elif (cols > 1):
    plt.subplot(211)
    plt.plot(data_col_0)
    plt.subplot(212)
    plt.plot(data_col_1)
else:
    plt.plot(data_col_0)
    # plt.title("graph")
    # plt.xlabel("points", loc = "right")
    # plt.ylabel("data", loc = "top")
plt.show()