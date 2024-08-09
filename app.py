import os
from flask import Flask, request
from flask_cors import CORS

import pandas as pd
import torch
import torch.nn as nn
from keras.src.saving import load_model

from generate_bytecode.demo_main import demo_main
from generate_bytecode.preprocessing import extractPureBytecode,get_opcode,simplify_opcode,get_csv,get_label_csv
from data_processing.get_Embedding import get_embedding

app = Flask(__name__)
CORS(app)
app.static_folder = 'dist'

class MyModel(nn.Module):
    def __init__(self, input_size):
        super(MyModel, self).__init__()

        self.embedding = nn.Embedding(input_size, 20)
        self.gru = nn.GRU(20, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        gru_output, _ = self.gru(embedded)
        gru_output = gru_output.unsqueeze(0)  # 添加一个维度
        gru_output = gru_output[:, -1, :]  # 取最后一个时间步的输出
        fc1_output = self.relu(self.fc1(gru_output))
        fc2_output = self.sigmoid(self.fc2(fc1_output))
        return fc2_output

def predict(model, test_data):
    model.eval()  # 将模型设置为评估模式
    predictions = []
    with torch.no_grad():
        for inputs in test_data:
            outputs = model(inputs)
            predicted_labels = (outputs > 0.5).squeeze().int()
            predictions.append(predicted_labels.item())
    return predictions


def readfile(name):
    normal_data = pd.read_csv("contract/normal.csv")

    vul_data = pd.read_csv("contract/csv/" + name + ".csv")
    X_test = pd.concat([normal_data[-vul_data.shape[0]:], vul_data[vul_data.shape[0] // 2:]], axis=0)


    return X_test




def detect(filename, vul):
    # 读取数据
    X_test = readfile(filename)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    X_test = torch.Tensor(X_test.values).long()

    model = torch.load('model_detect/model_'+vul+'.pth')
    predictions2 = predict(model, X_test)
    #print(predictions2)
    return predictions2


def unknown_detect(name):
    # 读取数据
    X_test = pd.read_csv("contract/csv/" + name + ".csv")

    """加载模型"""
    model=load_model('model_detect/model_unknown.h5')
    # 对测试集进行预测
    test_pre = model.predict(X_test)

    return test_pre

# 生成字节码和操作码，并将操作码转换成向量
def processing(in_lines,file_name):
    #生成字节码
    #bytecode_path=demo_main(file_path)
    #print("生成字节码")
    #转换成操作码
    #extractPureBytecode(bytecode_path,file_name)
    #print("转换成操作码")
    #simplify_opcode(file_name)

    simplify_opcode(in_lines, file_name)
    print("简化操作码")
    get_csv(file_name)
    print("得到csv文件")

    #转换成向量
    get_embedding(file_name)
    print("转换成向量")
    #投入模型中进行判断
    dir=["reentrancy", "timestamp", "delegatecall","Transaction_Ordering_Dependency"]
    #dir = ["reentrancy"]
    result=[]

    for leak in dir:
        pred_result=detect(file_name,leak)
        result.append(pred_result[1])

    if sum(result)==0:
        pred_result = unknown_detect(file_name)
        if pred_result[0][1]<0.5:
            pred_result=1
        else:
            pred_result=0
        result.append(pred_result)
    else:
        result.append(0)

    return result


@app.route('/detect', methods=['POST'])
def detect():
    # in_file_path = request.json['in_file_path']
    # out_filename = request.json['out_filename']

    in_lines = request.files['file'].readlines()
    in_lines = [line.decode('utf-8') for line in in_lines]
    # with extension
    out_filename = request.files['file'].filename
    # strip extension
    out_filename = out_filename.split('.')[0]
    out_filename = f'{out_filename}_simple'

    #result = processing(in_file_path,out_filename)
    result = processing(in_lines, out_filename)

    print(result)

    return result

@app.route('/')
def main_page():
    return app.send_static_file('index.html')

@app.route('/<path:path>')
def static_file(path):
    if path.endswith('.js'):
        return app.send_static_file(path), 200, {'Content-Type': 'application/javascript'}
    return app.send_static_file(path)

if __name__ == '__main__':
    app.run()


# def main():
#     #file_path = "./contract/opcode/unknown.txt"
#     #file_path="./contract/raw_contract/reentrancy_test.sol"
#     file_name="unknown"
#     result=processing(file_name)
#     print(result)
#
# if __name__ == '__main__':
#     main()