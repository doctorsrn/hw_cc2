import logging
import sys

from IOModule import *
from util import *
import util1 as u1
#
# logging.basicConfig(level=logging.DEBUG,
#                     filename='../../logs/CodeCraft-2019.log',
#                     format='[%(asctime)s] %(levelname)s [%(funcName)s: %(filename)s, %(lineno)d] %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S',
#                     filemode='a')


def main():
#    if len(sys.argv) != 6:
#        logging.info('please input args: car_path, road_path, cross_path, answerPath')
#        exit(1)
#
    car_path = sys.argv[1]
    road_path = sys.argv[2]
    cross_path = sys.argv[3]
    preset_answer_path = sys.argv[4]
    answer_path = sys.argv[5]

    # rpath = '../config1'
    # rpath = '/home/srn/SRn/Competition/HUAWei2/SDK_python/CodeCraft-2019/config1'
    # cross_path = rpath + '/cross.txt'
    # road_path = rpath + '/road.txt'
    # car_path = rpath + '/car.txt'
    # answer_path = rpath + '/answer.txt'
    # preset_answer_path = rpath + '/presetAnswer.txt'

    # to read input file
    car_df = read_car_from_txt(car_path)
    road_df = read_road_from_txt(road_path)
    cross_df = read_cross_from_txt(cross_path)
    pre_answer_df = read_preset_answer_from_txt(preset_answer_path, return_dict=True)
    pre_answer_d = read_preset_answer_from_txt(preset_answer_path, return_dict=False)

    # process
    al = build_adjacency_list(cross_df, road_df)

    # build adjacency list

    # 建立邻接表和后面权重更新相对应
    # 调试过效果良好的组合为:build_adjacency_list3--getallpaths_dj_cw  build_adjacency_list4--getallpaths_dj_cw2
    # ad_l = build_adjacency_list2(cross_df, road_df)
    # ad_l = build_adjacency_list3(cross_df, road_df)
    ad_l = build_adjacency_list4(cross_df, road_df)


    # 取出预置车辆
    car_preset_df = car_df.loc[car_df['preset'] == 1].copy(deep=True)
    preset_carlist = list(car_preset_df['id'])

    pre_paths = pre_answer_d['path'].to_dict()
    pre_times = pre_answer_d['planTime'].to_dict()
    # 取出非预置车辆
    car_not_preset_df = car_df.loc[car_df['preset'] != 1].copy(deep=True)
    notpreset_carlist = list(car_not_preset_df['id'])

    # 记录预置车辆实际出发时间
    # 将car_df_sort中预置车辆plantime改为presetAnswer中的数值
    for carid in preset_carlist:
        car_df['planTime'][carid] = pre_answer_df[carid]['planTime']
        car_preset_df['planTime'][carid] = pre_answer_df[carid]['planTime']


    # get time plans

    # 目前效果最好的为 get_time_plan5
    # time_plans, car_df_actual = u1.get_time_plan5(car_not_preset_df)
    # time_plans, car_df_actual = get_time_plan6(car_not_preset_df)
    # time_plans, car_df_actual = get_time_plan7(car_not_preset_df)
    # time_plans, car_df_actual = get_time_plan8(car_not_preset_df)
    time_plans, car_df_actual = get_time_plan9(car_df, car_preset_df, car_not_preset_df)

    # print(car_df_actual.head(100))

    # get path plans

    # 效果最好的是 getallpaths_dj_cw 和 getallpaths_dj_cw2
    # paths = get_allcarspaths_floyd(ad_l, car_df)
    # paths = get_allcarspaths_floyd_cw(ad_l, car_df)
    # paths = get_all_cars_paths(ad_l, car_df['id'], car_df['from'], car_df['to'], use_networkx=False)
    # paths = get_all_cars_paths_cw(ad_l, car_df, use_networkx=False)
    # paths = get_all_paths_with_hc(ad_l, road_df, car_df['id'], car_df['from'], car_df['to'])
    # paths = get_all_paths_with_hc_cw(ad_l, road_df, car_df_actual)
    # paths = getallpaths_dj_cw(ad_l, road_df, car_df_actual)
    # paths = getallpaths_dj_cw2(ad_l, road_df, car_df_actual)
    # paths = getallpaths_dj_cw3(ad_l, road_df, car_df_actual, pre_answer_df, preset_carlist)
    # paths = getallpaths_dj_cw3_slide(ad_l, road_df, car_df_actual, pre_answer_df, preset_carlist)
    # time_plans, car_df_actual = get_time_plan5(car_not_preset_df)
    # time_plans, car_df_actual = get_time_plan2(car_not_preset_df)
    # paths = get_all_cars_paths(al, car_df_actual['id'], car_df_actual['from'], car_df_actual['to'])
    # paths = get_all_paths_with_hc(al, road_df, car_df_actual['id'], car_df_actual['from'], car_df_actual['to'])
    paths = get_all_paths_with_weight_update(al, road_df, car_df_actual, cross_df, pathType=2, update_w=True)


    # 合并paths和timePlan
    paths.update(pre_paths)
    print(paths.__len__())
    origin_planTime = car_df_actual['planTime'].to_dict()
    origin_planTime.update(pre_times)
    print(origin_planTime.__len__())
    for carID in list(car_df['id']):
        car_df['planTime'][carID] = origin_planTime[carID]

    # replan
    time_plans, paths = u1.super_time_plan(paths, car_df, road_df, cross_df, al, pre_answer_d, preset_test=False)

    answers = get_answer(car_not_preset_df['id'], paths, time_plans)

    write_answer2file(answer_path, answers)

    # logging.info("car_path is %s" % (car_path))
    # logging.info("road_path is %s" % (road_path))
    # logging.info("cross_path is %s" % (cross_path))
    # logging.info("preset_answer_path is %s" % (preset_answer_path))
    # logging.info("answer_path is %s" % (answer_path))



# to write output file


if __name__ == "__main__":
    main()
