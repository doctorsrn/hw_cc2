import pandas

def read_car_from_txt(path_):
    df = pandas.read_csv(path_, sep='[^\\-|0-9]+', header=None, skiprows=1, engine='python')

    # delete NaN columns--> first column and last column
    df.drop(columns=[0, df.shape[1] - 1], inplace=True)

    df.set_axis(['id', 'from', 'to', 'speed', 'planTime', 'priority', 'preset'], axis='columns', inplace=True)
    df.set_index(df['id'], inplace=True)
    df.rename_axis('index', inplace=True)

    return df


def read_road_from_txt(path_):
    df = pandas.read_csv(path_, sep='[^\\-|0-9]+', header=None, skiprows=1, engine='python')

    # delete NaN columns--> first column and last column
    df.drop(columns=[0, df.shape[1] - 1], inplace=True)

    df.set_axis(['id', 'length', 'speed', 'channel', 'from', 'to', 'isDuplex'], axis='columns', inplace=True)
    df.set_index(df['id'], inplace=True)
    df.rename_axis('index', inplace=True)

    return df


def read_cross_from_txt(path_):
    df = pandas.read_csv(path_, sep='[^\\-|0-9]+', header=None, skiprows=1, engine='python')

    # delete NaN columns--> first column and last column
    df.drop(columns=[0, df.shape[1] - 1], inplace=True)

    df.set_axis(['id', 'roadID1', 'roadID2', 'roadID3', 'roadID4'], axis='columns', inplace=True)
    df.set_index(df['id'], inplace=True)
    df.rename_axis('index', inplace=True)

    return df


def read_preset_answer_from_txt(path_, return_dict=False):
    """
    读取预设路径
    :param path_:
    :param return_dict: 若为真则返回字典形式的预设路径格式为：{carID:{'planTIme'：xx, 'path': []}}
                        否则返回dataframe形式的路径：格式为index = carID, columns=['planTIme', 'path', 'id']
    :return:
    """
    car_dict = {}
    pre_answer = open(path_, 'r').read().split('\n')
    count = 0
    for i, line in enumerate(pre_answer):
        if line.__len__() < 3:
            continue
        if line[0] == '#':
            continue
        line = line.strip()[1:-1].split(',')
        car_id = int(line[0])
        plan_time = int(line[1])
        car_path = [int(road) for road in line[2:]]

        if car_dict.__contains__(car_id):
            car_dict[car_id]['planTime'] = plan_time
            car_dict[car_id]['path'] = car_path
        else:
            car_dict[car_id] = {'planTime': plan_time, 'path': car_path}

        count += 1
    # print("There are %d cars' route preinstalled" % count)
    # print(len(car_dict.keys()))
    # print((car_dict[17023]['planTime']))
    # print((car_dict[17023]['path']))
    if return_dict:
        return car_dict

    else:
        df = pandas.DataFrame.from_dict(car_dict, orient='index')
        df['id'] = df.index
        return df
    

def write_answer2file(txt_path, answer_list):
    """
    :brief: write data to answer.txt, data pattern {carID, startTime, path series}
    :param txt_path: 要写入文件的路径
    :param answer_list: answer 2维数组，数据格式例如:[[100, 1, 203, 303], [101, 3, 213, 303, 304, 432]]
    :return:
    """
    with open(txt_path, 'w') as output:
        output.write('#carID, StartTime, RoadID...\n')
        for answer in answer_list:
            answer_str = "".join([str(x)+',' for x in answer])  # 将int list型的answer转换为str类型，并以逗号隔开
            output.writelines('(' + answer_str[:-1] + ')' + '\n')  # answer_str[:-1] 最后的逗号不写入


if __name__ == "__main__":
    rpath = '../config1'
    path = rpath + '/cross.txt'
    path1 = rpath + '/road.txt'
    path2 = rpath + '/car.txt'
    path3 = rpath + '/answer.txt'
    # preset_answer_path = rpath + '/presetAnswer.txt'
    preset_answer_path_ = '/home/srn/SRn/Competition/HUAWei2/SDK_python/CodeCraft-2019/config1/presetAnswer.txt'

#    cross_df = read_from_txt(path)
#    print(cross_df.head())
#    print(cross_df.shape)
#
#    road_df1 = read_from_txt(path1)
#    print(road_df1.head())
#    print(road_df1.shape)
#
#    car_df = read_from_txt(path2)
#    print(car_df.head())
#    print(car_df.shape)
    
    pre_answer_df = read_preset_answer_from_txt(preset_answer_path_)
    
    

