import wfdb


# 读取年龄和性别
def read_age_sex(file_name):
    header = wfdb.rdheader(file_name)

    # 读取注释信息，获取年龄和性别
    first_comment_ls = header.comments[0].split()
    age, sex = int(first_comment_ls[0]), first_comment_ls[1]

    return age, sex


# 读取二导联信号
def read_signal(file_name):
    record = wfdb.rdrecord(file_name,
                           sampfrom=0,
                           sampto=650000,
                           physical=True,
                           channel_names=['MLII'])
    signal = record.p_signal[:, 0]
    fs = record.fs
    return signal, fs


# 读取标签
def read_annotation(file_name):
    annotation = wfdb.rdann(file_name, "atr", sampfrom=0, sampto=650000)
    return annotation
