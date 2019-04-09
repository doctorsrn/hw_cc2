import logging
import sys
from util import *
from IOModule import *
import os
import profile

def main():
    # print("hello")
    start=time.time()
    rpath = '../config2'
    cross_path = rpath + '/cross.txt'
    road_path = rpath + '/road.txt'
    car_path = rpath + '/car.txt'
    answer_path = rpath + '/answer.txt'
    preset_answer_path = rpath + '/presetAnswer.txt'

    car_df = read_car_from_txt(car_path)
    # print(car_df.head())
    road_df = read_road_from_txt(road_path)
    cross_df = read_cross_from_txt(cross_path)
    pre_answer_df = read_preset_answer_from_txt(preset_answer_path, return_dict=True)


    # process

    # build adjacency list

    # 建立邻接表和后面权重更新相对应
    # 调试过效果良好的组合为:build_adjacency_list3--getallpaths_dj_cw  build_adjacency_list4--getallpaths_dj_cw2
    # ad_l = build_adjacency_list2(cross_df, road_df)
    # ad_l = build_adjacency_list3(cross_df, road_df)
    ad_l = build_adjacency_list4(cross_df, road_df)

    # 取出预置车辆
    car_preset_df = car_df.loc[car_df['preset'] == 1].copy(deep=True)
    preset_carlist = list(car_preset_df['id'])
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
    # time_plans, car_df_actual = get_time_plan5(car_not_preset_df)
    # time_plans, car_df_actual = get_time_plan6(car_not_preset_df)
    # time_plans, car_df_actual = get_time_plan7(car_not_preset_df)
    # time_plans, car_df_actual = get_time_plan8(car_not_preset_df)
    time_plans, car_df_actual = get_time_plan9(car_df, car_preset_df, car_not_preset_df)
    # time_plans, car_df_actual = get_time_plan10(car_df, car_preset_df, car_not_preset_df)

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
    paths = getallpaths_dj_cw3(ad_l, road_df, car_df_actual, pre_answer_df, preset_carlist)
    # paths = getallpaths_dj_cw3_slide(ad_l, road_df, car_df_actual, pre_answer_df, preset_carlist)


    # # 时间安排重规划
    # # 作用不大，后面可在没有啥招的时候用
    # # 路径最短的先发车
    # # timereplan3和get_time_plan5对应
    # car_df_extend = add_length_cardf(paths, road_df, car_df)
    # print(car_df_extend.head(20))
    # # time_plans, car_df_actual = timereplan(car_df_extend)
    # # time_plans, car_df_actual = timereplan2(car_df_extend)
    # time_plans, car_df_actual = timereplan3(car_df_extend)
    # print(car_df_actual.head(20))


    # get answer
    answers = get_answer(car_not_preset_df['id'], paths, time_plans)

    # to write output file
    write_answer2file(answer_path, answers)

    end=time.time()
    print(end-start)

    print("Good luck...")


if __name__ == "__main__":
    # profile.run('main()')
    main()
