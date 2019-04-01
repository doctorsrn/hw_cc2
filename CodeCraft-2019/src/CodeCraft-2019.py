import logging
import sys

from IOModule import *
from util import *
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
    pre_answer_df = read_preset_answer_from_txt(preset_answer_path, return_dict=False)

    # process
    al = build_adjacency_list(cross_df, road_df)

    car_not_preset_df = car_df.loc[car_df['preset'] != 1].copy(deep=True)

    time_plans, car_df_actual = get_time_plan5(car_not_preset_df)
    # time_plans, car_df_actual = get_time_plan2(car_not_preset_df)
    # paths = get_all_cars_paths(al, car_df_actual['id'], car_df_actual['from'], car_df_actual['to'])
    paths = get_all_paths_with_weight_update(al, road_df, car_df_actual, cross_df, pathType=2, update_w=True)

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