import fnmatch
import glob
import io
import itertools
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import sys
from itertools import chain
from multiprocessing import Process, Queue
from time import sleep
from timeit import default_timer as timer

import PyQt5.QtCore
import matplotlib.animation as animation
import matplotlib.colors as mc
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
import numpy.ma as ma
import psutil
import rawpy
from PIL import Image
# from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QApplication, QMainWindow
from PySide2.QtCore import QTimer
from scipy import ndimage

import VisImSorterInterface

image_DB = []
group_db = []
# grouped_image_db = []
distance_db = []
target_groups = 0
target_group_size = 0
progress_value = 0
progress_max = 0
image_count = 0
groups_count = 0
start_time = 0

start_folder = ""
new_folder = ""
new_vector_folder = ""
status_text = ""
search_subdirectories = False
# group_by_file_count = False
enforce_equal_folders = 0.
hg_bands_count = 0
compare_file_percent = 10
selected_color_spaces = []
selected_color_subspaces = []
move_files_not_copy = False
show_histograms = False
export_histograms = False
create_samples_enabled = False
run_index = 0
abort_reason = ""
# optimization_queue = []
gaussian_filter1d = ndimage.filters.gaussian_filter1d
median_filter = ndimage.filters.median_filter
gaussian_filter = ndimage.filters.gaussian_filter
la_norm = np.linalg.norm
folder_size_min_files = 0
# folder_size_min_percent = 0
# min_folder_size_set_in_percent = False
final_sort = 0
group_with_similar_name = 0
plain_file_types = ['jpg', 'png', 'jpeg', 'gif']
raw_file_types = ['nef', 'dng']
# process_version = 0
# power_coefficient = 1
folder_constraint_type = 0
folder_constraint_value = 0
stage1_grouping_type = 0
enable_stage2_grouping = False
enable_multiprocessing = True

# todo: option to change subfolder naming scheme (add color into the name, number of files)
# todo: search similar images to sample image or sample folder with images
# todo: sort final folders by: (average color, average lightness, file count, ...)
# todo: compare using 3D histograms
# todo: fix: in case of low
# todo: figure out what to do if two files have the same name and belong to the same folder

# todo: modify sorting algorithm so that final number of folders is roughly equal to desired
# todo: initial grouping is made only using image pairs rather than group pairs
# todo: when searching for closest pairs, sort also by other bands, not only by HSV hue

# todo: option to group only by pixel count
# todo: asynchronous interface: enable to change settings while scanning
# todo: scan forders and calculate difference between images in each
# todo: convert image_DB into numpy array


'''Алгоритм выделения связных компонент

В алгоритме выделения связных компонент задается входной параметр R и в графе удаляются все ребра, для которых 
«расстояния» больше R. Соединенными остаются только наиболее близкие пары объектов. Смысл алгоритма заключается в 
том, чтобы подобрать такое значение R, лежащее в диапазон всех «расстояний», при котором граф «развалится» на 
несколько связных компонент. Полученные компоненты и есть кластеры. 

Для подбора параметра R обычно строится гистограмма распределений попарных расстояний. В задачах с хорошо выраженной 
кластерной структурой данных на гистограмме будет два пика – один соответствует внутрикластерным расстояниям, 
второй – межкластерным расстояния. Параметр R подбирается из зоны минимума между этими пиками. При этом управлять 
количеством кластеров при помощи порога расстояния довольно затруднительно. 


Алгоритм минимального покрывающего дерева

Алгоритм минимального покрывающего дерева сначала строит на графе минимальное покрывающее дерево, 
а затем последовательно удаляет ребра с наибольшим весом. 



'''


def start_process():
    global progress_value
    global progress_max
    global run_index
    global status_text
    global start_time
    global show_histograms
    global export_histograms
    global start_folder
    global new_folder
    global move_files_not_copy
    global pool
    local_run_index = run_index

    parent = psutil.Process()
    if psutil.LINUX:
        parent.nice(15)
    else:
        parent.nice(psutil.IDLE_PRIORITY_CLASS)

    start_time = timer()
    progress_value = 0

    image_paths = scan_for_images()
    if local_run_index != run_index:
        return

    pool = mp.Pool()

    generate_image_vectors_and_groups(image_paths)
    if local_run_index != run_index:
        pool.terminate()
        pool.close()
        return

    if stage1_grouping_type == 0:
        create_simple_groups()
        if local_run_index != run_index:
            pool.terminate()
            pool.close()
            return
    elif stage1_grouping_type == 1:
        create_groups_by_similarity()
        if local_run_index != run_index:
            pool.terminate()
            pool.close()
            return
    elif stage1_grouping_type == 2:
        create_groups_v4()
        if local_run_index != run_index:
            pool.terminate()
            pool.close()
            return

    if enable_stage2_grouping:
        stage2_regroup()
        if local_run_index != run_index:
            pool.terminate()
            pool.close()
            return

    final_group_sort()

    pool.close()

    choose_destination_folder()
    move_files()
    if local_run_index != run_index:
        return

    if export_histograms:
        export_image_vectors()
    if local_run_index != run_index:
        return

    if create_samples_enabled:
        create_samples()

    if move_files_not_copy:
        start_folder = new_folder

    finish_time = timer()

    if show_histograms:
        AnimatedHistogram(image_DB, group_db)
    if local_run_index != run_index:
        return

    status_text = "Finished. Elapsed " + "%d:%02d" % divmod(finish_time - start_time, 60) + " minutes."
    print(status_text)
    progress_max = 100
    progress_value = 100


def scan_for_images():
    global status_text
    global start_folder
    global search_subdirectories
    global run_index
    global abort_reason
    status_text = "Scanning images..."
    print(status_text)
    QApplication.processEvents()

    types = plain_file_types + raw_file_types
    image_paths = []
    image_paths_grouped = []

    all_files = glob.glob(start_folder + "/**/*", recursive=search_subdirectories)

    for file_mask in types:
        # image_paths.extend(glob.glob(start_folder + subdir_mask + file_mask, recursive=search_subdirectories))
        image_paths.extend(fnmatch.filter(all_files, "*." + file_mask))

    # image_paths = [fnmatch.filter(all_files, "*." + ex) for ex in types]
    image_paths = [os.path.normpath(im) for im in image_paths]

    if len(image_paths) == 0:
        abort_reason = "No images found."
        run_index += 1
        QMessageBox.warning(None, "VisImSorter: Error", abort_reason)
        print(abort_reason)
    elif group_with_similar_name > 0:
        file_name_groups = sorted(image_paths, key=lambda x: os.path.basename(x)[:group_with_similar_name])
        image_paths_grouped = [list(it) for k, it in itertools.groupby(
            file_name_groups, key=lambda x: os.path.basename(x)[:group_with_similar_name])]
    else:
        image_paths_grouped = [[it] for it in image_paths]
    image_paths = image_paths_grouped

    return image_paths


def generate_image_vectors_and_groups(image_list):
    global image_DB
    global image_count
    global target_group_size
    global target_groups
    global progress_max
    global progress_value
    global group_db
    global groups_count
    global run_index
    global abort_reason
    global pool
    global folder_size_min_files
    global folder_constraint_type
    global folder_constraint_value

    local_run_index = run_index
    image_count = len(image_list)

    progress_max = image_count
    progress_value = 0
    image_DB = []
    results = []

    for image_path in image_list:
        result = pool.apply_async(generate_image_vector,
                                  (image_path, selected_color_spaces, selected_color_subspaces, hg_bands_count),
                                  callback=generate_image_vector_callback)
        results.append(result)

    for res in results:
        while not res.ready():
            QApplication.processEvents()
            sleep(.05)
            if local_run_index != run_index:
                return
        res.wait()

    image_DB.sort(key=lambda x: x[2])
    image_count = len(image_DB)
    real_image_count = sum([x[3] for x in image_DB])

    if image_count == 0:
        abort_reason = "Failed to create image database."
        run_index += 1
        print(abort_reason)

    if local_run_index != run_index:
        return

    if folder_constraint_type:
        target_groups = int(folder_constraint_value)
        target_group_size = int(round(real_image_count / target_groups))
    else:
        target_group_size = int(folder_constraint_value)
        target_groups = int(round(real_image_count / target_group_size))


def generate_image_vector_callback(vector_pack):
    global image_DB
    global progress_value
    global status_text

    if vector_pack is not None:
        image_DB.append(vector_pack)
    progress_value += 1
    status_text = "(1/4) Generating image histograms... (" \
                  + str(progress_value) + " of " + str(image_count) + ")"
    QApplication.processEvents()


def generate_image_vector(image_full_names, color_spaces, color_subspaces, hg_bands):
    hue_hg = np.zeros(256, dtype=np.int64)
    average_histogram = np.zeros(hg_bands * len(color_subspaces), dtype='int64')
    for image_full_name in image_full_names:
        try:
            if sum([image_full_name.lower().endswith(ex) for ex in raw_file_types]):
                f = open(image_full_name, 'rb', buffering=0)  # This is a workaround for opening cyrillic file names
                thumb = rawpy.imread(f).extract_thumb()
                img_to_read = io.BytesIO(thumb.data)
            else:
                img_to_read = image_full_name
            img = Image.open(img_to_read)
            img = img.resize([200, 200], resample=Image.BILINEAR)

            # img.save("Y:\\thumbs\\" + os.path.basename(image_full_name) + "_.png")
            hue_hg_one = np.histogram(img.convert("HSV").getchannel("H"), bins=256, range=(0., 255.))[0]
            hue_hg += hue_hg_one

            combined_histogram = np.array([], dtype='int64')
            for current_color_space in color_spaces:
                converted_image = img.convert(current_color_space)
                for band in converted_image.getbands():
                    if band in color_subspaces:
                        partial_hg = np.histogram(converted_image.getchannel(band), bins=hg_bands, range=(0., 255.))[0]
                        partial_hg = gaussian_filter(partial_hg, math.sqrt(hg_bands / 40), mode='nearest')
                        combined_histogram = np.concatenate((combined_histogram, partial_hg))
                        QApplication.processEvents()
            average_histogram += combined_histogram
        except Exception as e:
            print("Error reading ", image_full_name, e)
    images_in_row = len(image_full_names)
    hue_hg = gaussian_filter1d(np.divide(hue_hg, images_in_row), 15, mode='wrap').astype(np.uint16)
    max_hue = np.argmax(hue_hg).tolist()
    average_histogram = np.divide(average_histogram, images_in_row).astype(np.uint16)

    return [image_full_names, average_histogram, max_hue, images_in_row, hue_hg]


def create_simple_groups():
    global group_db
    global image_DB
    global target_group_size
    global target_groups
    global groups_count

    group_db = []
    group_im_indexes = np.array_split(range(image_count), target_groups)
    for new_group_image_list in group_im_indexes:
        new_group_vector = np.mean(np.array([image_DB[i][1] for i in new_group_image_list]), axis=0)
        new_group_hue = np.argmax(np.sum(np.array([image_DB[i][4] for i in new_group_image_list]), axis=0))
        new_group_image_count = sum([image_DB[i][3] for i in new_group_image_list])
        group_db.append([new_group_image_list.tolist(), new_group_vector, new_group_hue, new_group_image_count])

    group_db = sorted(group_db, key=lambda x: x[2])
    groups_count = len(group_db)


def create_groups_v4():
    global image_count
    global status_text
    global target_groups
    global groups_count
    global run_index
    global pool
    global group_db
    global image_DB
    global distance_db
    global progress_max
    global progress_value

    local_run_index = run_index
    batches = 1
    batch = 0
    compare_limit = int(image_count * .1 + 200)

    groups_history = []
    groups_open_history = []
    group_ratings_history = []
    group_invites_history = []

    status_text = "(2/4) Comparing images..."
    progress_max = image_count

    relations_db = create_relations_db(image_DB, batch, batches, compare_limit)

    status_text = "(3/4) Choosing center images..."

    eef01 = (enforce_equal_folders + .5) ** 2
    eef01_divisor = np.tanh(eef01) + 1

    groups_open = np.ones(image_count, dtype=bool)
    invite_count = np.ones(image_count) * target_group_size
    image_index_list = np.arange(image_count)
    image_belongings = image_index_list
    image_belongings_old = image_belongings
    image_belongings_final = image_belongings
    mutual_ratings = (np.tanh(eef01 * (1 - relations_db['rank'] / target_group_size)) + 1) / eef01_divisor
    group_ratings = np.sum((mutual_ratings - np.eye(image_count)) / (1 + relations_db['dist']), axis=1)
    group_ratings /= group_ratings.max()
    size_corrector = 1.
    minimal_distance = relations_db['dist'][relations_db['dist'] > 1].min()
    mutual_weights = 0

    progress_max = 1500
    for _ in range(progress_max):
        if (image_belongings == image_belongings_old).all():
            actual_groups = len(np.unique(image_belongings))
            group_sizes = np.bincount(image_belongings, minlength=image_count)
            # actual_groups2 = len(group_sizes.nonzero())
            image_belongings_final = image_belongings
            if actual_groups > target_groups:
                # image_belongings = image_index_list
                # todo: make an elegant a_coefficient function
                # todo: maybe reset every invite_count on some occasions
                # todo: use bigger steps for very large number of groups
                # todo: increase attraction to groups that have least mean distance or right file count

                # desmos
                # \tanh\left(1-\left(\frac{x}{b}\right)^{c}-\frac{b}{\left(15-a\right)x}\right)

                # cheap sigmoid
                # \frac{\left(x - b\right)}{2} / (1 +\operatorname{abs}(x-b))+.5
                # \frac{x-b}{2\left(1+\operatorname{abs}\left(x-b\right)\right)}+.5
                # \frac{.5}{\operatorname{sign}\left(x-b\right)+\frac{1}{x-b}}+.5

                # mega
                # \frac{1}{1+20^{5\left(\frac{x}{b}-1\right)}}\cdot\left(\frac{-.5}{\operatorname{sign}\left(x-b\right)+\frac{ab}{x-b}}+.5\right)
                # \frac{1}{1+20^{\frac{5\left(x-b\right)}{b}}}\cdot\left(\frac{-.5}{\operatorname{sign}\left(x-b\right)+\frac{ab}{x-b}}+.5\right)

                groups_open = (image_belongings == image_index_list)
                size_corrector *= 1.1
                distance_limit = minimal_distance * size_corrector ** .2
                mutual_distance = relations_db['dist'] - distance_limit
                distance_weights = .5 - .5 / (np.sign(mutual_distance) + 1 / mutual_distance)
                # distance_weights = (.5 - .5 / (np.sign(mutual_distance) + .05 * distance_limit / mutual_distance))
                # distance_weights /= (1 + 20 ** (5 * mutual_distance / distance_limit))
                mutual_weights = distance_weights # * group_size_weights
                group_ratings = np.sum(mutual_weights / (relations_db['dist'] + 500) * (1 - np.eye(image_count)), axis=1)
                group_ratings /= group_ratings.max()
                image_belongings = image_index_list
                # print(_, actual_groups, size_corrector)

            else:
                break

        # groups_history.append(image_belongings)
        # groups_open_history.append(groups_open)
        # group_ratings_history.append(group_ratings)
        # group_invites_history.append(group_invites)

        image_belongings_old = image_belongings
        groups_open_ratings = (image_belongings == image_index_list) * group_ratings
        group_invites = mutual_weights * groups_open_ratings
        image_belongings = np.argmax(group_invites, axis=1)

        closed_groups_indexes = (image_belongings != image_index_list).nonzero()
        images_in_closed_groups = np.isin(image_belongings, closed_groups_indexes)
        # invite_count = ma.array(invite_count, mask=images_in_closed_groups).filled(np.ones(image_count) * target_group_size)
        image_belongings = ma.array(image_belongings, mask=images_in_closed_groups).filled(image_index_list)

        progress_value = _
        QApplication.processEvents()

    # base_image_names = np.array([os.path.basename(i[0][0]) for i in image_DB])
    # save_log(groups_history, "groups_history", base_image_names)
    # save_log(groups_open_history, "groups_open_history", base_image_names)

    status_text = "(4/4) Grouping images..."
    progress_value = 0
    QApplication.processEvents()

    center_index_list = np.unique(image_belongings_final)

    group_db = []
    for im_n in center_index_list:
        indexes_in_group = (image_belongings_final == im_n).nonzero()[0].tolist()
        # indexes_in_group = [im_n]
        group_db.append([indexes_in_group, None, None, None])

    for grp in group_db:
        grp[1] = np.mean(np.array([image_DB[i][1] for i in grp[0]]), axis=0)
        grp[2] = np.argmax(np.sum(np.array([image_DB[i][4] for i in grp[0]]), axis=0))
        grp[3] = np.sum(np.array([image_DB[i][3] for i in grp[0]]), axis=0)
    groups_count = len(group_db)


#
# '''
# desmos
# wide
#     \frac{\tanh\left(2.5 + .5\cdot\frac{\left(a - x\right)}{10 - b}\right)}{2} + .5
#
# strict
#     \frac{\tanh\left(\left(1-\frac{x}{a}\right)\cdot\left(a\cdot\frac{b\cdot b}{5}\right)\right)}{2}+.5
#
# strict+
#     \frac{\tanh\left(\left(1-\frac{x}{a}\right)\cdot\left(b^{2}+0.1\right)\right)+1}{\tanh\left(b^{2}+0.1\right)+1}
#
# strict+2
#     \frac{\tanh\left(\left(1-\frac{x}{a}\right)\cdot\left(b^{2}+0.5\right)^{2}\right)+1}{\tanh\left(\left(b^{2}+0.5\right)^{2}\right)+1}
# '''


def create_relations_db(im_db, batch, batches, compare_lim):
    global progress_value
    im_count = len(im_db)
    image_index_list = np.arange(im_count)
    compare_count = min(compare_lim * 2 + 1, im_count)
    relations_db = []
    base_image_coordinates = np.array([i[1] for i in image_DB])

    for im_1 in np.array_split(image_index_list, batches)[batch]:
        second_range = np.roll(image_index_list, compare_lim - im_1)
        im_1_db = np.zeros(im_count, dtype=[('dist', np.float), ('rank', np.int32), ('im_1', np.int), ('im_2', np.int)])
        im_1_db['im_1'] = im_1
        im_1_db['im_2'] = second_range
        im_1_db['dist'] = np.inf
        # im_1_coordinates = im_db[im_1][1].astype(np.int32)
        # im_2_coordinates = base_image_coordinates[second_range[:compare_count]].astype(np.int32)
        im_1_coordinates = im_db[im_1][1]
        im_2_coordinates = base_image_coordinates[second_range[:compare_count]]
        vector_distance_line = la_norm((im_2_coordinates - im_1_coordinates).astype(np.int32), axis=1)
        im_1_db[:compare_count]['dist'] = vector_distance_line
        im_1_db.sort(order='dist')
        im_1_db['rank'] = image_index_list.astype(np.int32)
        im_1_db.sort(order='im_2')
        relations_db.append(im_1_db)
        progress_value = im_1
        QApplication.processEvents()
    relations_db = np.array(relations_db).T

    return relations_db


# noinspection PyTypeChecker
def save_log(log_values, log_name, base_image_names):
    logs_dir = "y:\\logs\\"
    if not os.path.isdir(logs_dir):
        try:
            os.makedirs(logs_dir)
        except Exception as e:
            print("Could not create folder ", e)
    np.savetxt(logs_dir + log_name + ".csv", np.vstack([base_image_names, log_values]), fmt='%s', delimiter=";", encoding="utf-8")


def create_groups_by_similarity():
    global image_count
    global status_text
    global target_groups
    global groups_count
    global run_index
    global pool
    global group_db
    global distance_db

    local_run_index = run_index
    step = 0

    group_db = [[[counter]] + image_record[1:4] for counter, image_record in enumerate(image_DB)]
    groups_count = len(group_db)

    feedback_queue = mp.Manager().Queue()
    while groups_count > target_groups:
        step += 1
        status_text = "Joining groups, pass %d. Now joined %d of %d . To join %d groups"
        status_text %= (step, image_count - groups_count, image_count - target_groups, groups_count - target_groups)

        fill_distance_db(feedback_queue, step)
        if local_run_index != run_index:
            return

        merge_group_batches()
        if local_run_index != run_index:
            return


def fill_distance_db(feedback_queue, step):
    global group_db
    global distance_db
    global image_count
    global target_groups
    global target_group_size
    global progress_max
    global progress_value
    global groups_count
    global run_index
    global enforce_equal_folders
    global pool

    local_run_index = run_index
    progress_max = groups_count

    distance_db = []
    results = []
    batches = min(groups_count // 200 + 1, mp.cpu_count())
    # batches = 1
    progress_value = 0

    compare_file_limit = int(compare_file_percent * (image_count + 2000) / 100)

    entries_limit = groups_count // 4 + 1
    entries_limit = groups_count - target_groups

    feedback_list = [0] * batches
    task_running = False
    if batches == 1:
        distance_db += fill_distance_db_sub(group_db, 1, 0, enforce_equal_folders, compare_file_limit,
                                            target_group_size, entries_limit)
    else:
        for batch in range(batches):
            task_running = True
            args = [group_db, batches, batch, enforce_equal_folders, compare_file_limit, target_group_size,
                    entries_limit, feedback_queue]
            result = pool.apply_async(fill_distance_db_sub, args=args, callback=fill_distance_db_callback)
            results.append(result)
            QApplication.processEvents()
            if local_run_index != run_index:
                return

    while task_running:
        task_running = False
        for res in results:
            if not res.ready():
                task_running = True
        while not feedback_queue.empty():
            feedback_bit = feedback_queue.get()
            feedback_list[feedback_bit[0]] = feedback_bit[1]
            progress_value = sum(feedback_list) // batches
        QApplication.processEvents()

    distance_db_unfiltered = sorted(distance_db, key=lambda x: x[0])  # [:entries_limit]

    used_groups = []
    distance_db = []
    for dist, g1, g2 in distance_db_unfiltered:
        if not ((g1 in used_groups) or (g2 in used_groups)):
            distance_db.append([g1, g2])
            used_groups += [g1, g2]
        if len(distance_db) > entries_limit:
            break


def fill_distance_db_callback(in_result):
    global progress_value
    global distance_db
    distance_db += in_result


def fill_distance_db_sub(group_db_l, batches, batch, enforce_ef, compare_lim, target_gs, entries_lim, feedback=None):
    global progress_value
    groups_count_l = len(group_db_l)
    compute_result = []
    simple_counter = 10
    for index_1 in range(batch, groups_count_l, batches):
        best_pair = None
        min_distance = float("inf")
        end_of_search_window = index_1 + compare_lim
        if groups_count_l >= end_of_search_window:
            second_range = range(index_1 + 1, end_of_search_window)
        else:
            rest_of_search_window = min(end_of_search_window - groups_count_l, index_1)
            second_range = chain(range(index_1 + 1, groups_count_l), range(0, rest_of_search_window))
        for index_2 in second_range:
            vector_distance = la_norm(group_db_l[index_1][1].astype(np.int32) - group_db_l[index_2][1].astype(np.int32))
            size_factor = (group_db_l[index_1][3] + group_db_l[index_2][3]) / target_gs
            vector_distance *= (size_factor ** enforce_ef)
            if min_distance > vector_distance:
                min_distance = vector_distance
                best_pair = [index_1, index_2]
            simple_counter -= 1
            if simple_counter == 0:
                simple_counter = 5000
                if feedback is None:
                    progress_value = index_1
                    QApplication.processEvents()
                else:
                    feedback.put([batch, index_1])
        if best_pair is not None:
            compute_result.append([min_distance] + best_pair)

    return compute_result


def merge_group_batches():
    global group_db
    global distance_db
    global target_groups
    global groups_count
    global run_index
    global image_DB

    local_run_index = run_index
    removed_count = 0

    for g1, g2 in distance_db:
        if groups_count - removed_count <= target_groups:
            break
        new_group_image_list = group_db[g1][0] + group_db[g2][0]
        new_group_vector = np.mean(np.array([image_DB[i][1] for i in new_group_image_list]), axis=0)
        new_group_hue = np.argmax(np.sum(np.array([image_DB[i][4] for i in new_group_image_list]), axis=0))
        new_group_image_count = sum([image_DB[i][3] for i in new_group_image_list])
        group_db.append([new_group_image_list, new_group_vector, new_group_hue, new_group_image_count])
        group_db[g1][3] = 0
        group_db[g2][3] = 0
        removed_count += 1
        if local_run_index != run_index:
            return
        QApplication.processEvents()
    group_db_filtered = filter(lambda x: x[3] > 0, group_db)
    group_db = sorted(group_db_filtered, key=lambda x: x[2])
    groups_count = len(group_db)


def stage2_sort_search(im_db, target_group_sz, enforce_ef, gr_db, work_group, target_groups_list):
    groups_count_l = len(gr_db)
    return_db = []
    for im_index in gr_db[work_group][0]:
        current_distance = la_norm(gr_db[work_group][1].astype(np.int32) - im_db[im_index][1].astype(np.int32))
        current_distance *= (gr_db[work_group][3] / target_group_sz) ** enforce_ef
        best_distance = current_distance
        new_group = -1
        distance_difference = 0
        for target_group in range(groups_count_l):
            if target_group == work_group:
                continue
            if not target_groups_list[work_group][target_group]:
                continue
            distance_to_target = la_norm(gr_db[target_group][1].astype(np.int32) - im_db[im_index][1].astype(np.int32))
            distance_to_target *= (gr_db[target_group][3] / target_group_sz) ** enforce_ef
            if distance_to_target < best_distance:
                new_group = target_group
                best_distance = distance_to_target
                distance_difference = current_distance - distance_to_target
        if new_group != -1:
            return_db.append([im_index, work_group, new_group, distance_difference])
            gr_db[work_group][0].remove(im_index)
            gr_db[new_group][0].append(im_index)
            gr_db[work_group][1] = np.mean(np.array([im_db[i][1] for i in gr_db[work_group][0]]), axis=0)
            gr_db[new_group][1] = np.mean(np.array([im_db[i][1] for i in gr_db[new_group][0]]), axis=0)
            gr_db[work_group][3] -= im_db[im_index][3]
            gr_db[new_group][3] += im_db[im_index][3]
    if len(return_db) > 0:
        return_db = sorted(return_db, key=lambda x: x[3], reverse=True)
        return return_db
    else:
        return [[None, work_group]]


def stage2_sort_worker(input_q, output_q, im_db, target_group_sz, enforce_ef):
    for args in iter(input_q.get, 'STOP'):
        result = stage2_sort_search(im_db, target_group_sz, enforce_ef, args[0], args[1], args[2])
        output_q.put(result)


def stage2_regroup():
    global image_DB
    global group_db
    global groups_count
    global progress_value
    global progress_max
    global target_group_size
    global enforce_equal_folders
    global run_index
    global status_text

    local_run_index = run_index

    moved_images = 0
    progress_value = 0
    progress_max = groups_count * groups_count
    idle_runs = 0

    status_text = "Resorting started... "
    QApplication.processEvents()

    parallel_processes = mp.cpu_count()
    task_index = 0
    lookup_order = list(range(groups_count))
    target_groups_list = np.ones((groups_count, groups_count), dtype=np.bool)

    task_queue = Queue()
    done_queue = Queue()

    for i in range(parallel_processes):
        Process(target=stage2_sort_worker, args=(task_queue, done_queue, image_DB, target_group_size,
                                                 enforce_equal_folders)).start()
        task_queue.put([group_db, 0, target_groups_list])
        QApplication.processEvents()

    while idle_runs < groups_count * 2:
        for i in range(parallel_processes):
            task_index += 1
            if task_index == groups_count:
                task_index = 0
                random.shuffle(lookup_order)
            task_queue.put([group_db, lookup_order[task_index], target_groups_list])

        for group_n in range(parallel_processes):
            next_result = done_queue.get()
            moved_in_this_step = 0
            if next_result[0][0] is not None:
                for move_order in next_result:
                    move_result = move_image_between_groups(move_order)
                    if move_result:
                        moved_in_this_step += 1
                        target_groups_list[move_order[2]] = True
                        target_groups_list[:, move_order[1]] = True
            if moved_in_this_step:
                moved_images += moved_in_this_step
                idle_runs = 0
            else:
                idle_runs += 1
                target_groups_list[next_result[0][1]] = False
        progress_value = progress_value * .99 + (progress_max - np.sum(target_groups_list)) * .01
        status_text = "Resorting. Relocated " + str(moved_images) + " files"
        QApplication.processEvents()

        if local_run_index != run_index:
            break

    for i in range(parallel_processes):
        task_queue.put('STOP')


def final_group_sort():
    global image_DB
    global group_db

    if final_sort == 0:
        for grp in group_db:
            grp[2] = np.argmax(np.sum(np.array([image_DB[i][3] for i in grp[0]]), axis=0))
        group_db.sort(key=lambda x: x[2])
    elif final_sort == 1:
        group_db.sort(key=lambda x: x[3])
    elif final_sort == 2:
        group_db.sort(key=lambda x: x[3], reverse=True)


def move_image_between_groups(task):
    global group_db
    global image_DB
    global target_group_size
    global enforce_equal_folders

    im_index = task[0]
    g1 = task[1]
    g2 = task[2]

    if im_index not in group_db[g1][0]:
        return False

    distance_to_g1 = la_norm(group_db[g1][1].astype(np.int32) - image_DB[im_index][1].astype(np.int32))
    distance_to_g1 *= (group_db[g1][3] / target_group_size) ** enforce_equal_folders
    distance_to_new_g2 = la_norm(group_db[g2][1].astype(np.int32) - image_DB[im_index][1].astype(np.int32))
    distance_to_new_g2 *= ((group_db[g2][3] + image_DB[im_index][3]) / target_group_size) ** enforce_equal_folders
    if distance_to_g1 > distance_to_new_g2:
        group_db[g1][0].remove(im_index)
        group_db[g2][0].append(im_index)
        for g in [g1, g2]:
            group_db[g][1] = np.mean(np.array([image_DB[i][1] for i in group_db[g][0]]), axis=0)
            group_db[g][3] = sum([image_DB[i][3] for i in group_db[g][0]])
        return True
    else:
        return False


def choose_destination_folder():
    global new_vector_folder
    global new_folder
    base_folder = start_folder
    destination_folder_index = -1
    if re.match(".*_sorted_\d\d\d$", start_folder):
        base_folder = start_folder[:-11]
    elif re.match(".*_sorted$", start_folder):
        base_folder = start_folder[:-7]
        destination_folder_index = 0

    list_of_indices = sorted(glob.glob(base_folder + "_sorted_[0-9][0-9][0-9]"), reverse=True)
    if len(list_of_indices) == 0:
        if len(glob.glob(base_folder + "_sorted")) > 0:
            destination_folder_index = 0
    if len(list_of_indices) > 0:
        destination_folder_index = int(list_of_indices[0][-3:]) + 1

    destination_folder_index_suffix = "/"
    if destination_folder_index >= 0:
        destination_folder_index_suffix = "_" + "%03d" % destination_folder_index + "/"

    new_folder = base_folder + "_sorted" + destination_folder_index_suffix
    new_vector_folder = base_folder + "_histograms" + destination_folder_index_suffix


def move_files():
    global start_time
    global group_db
    global image_DB
    global move_files_not_copy
    global status_text
    global progress_max
    global progress_value
    global groups_count
    global run_index
    global new_folder
    global folder_size_min_files

    local_run_index = run_index

    progress_max = groups_count
    status_text = "Sorting done. " + ["Copying ", "Moving "][move_files_not_copy] + "started"
    print(status_text)
    progress_value = 0
    QApplication.processEvents()

    new_folder_digits = int(math.log(target_groups, 10)) + 1

    action = shutil.copy
    if move_files_not_copy:
        action = shutil.move

    try:
        os.makedirs(new_folder)
    except Exception as e:
        print("Could not create folder ", e)

    ungroupables = []
    for group_index, grp in enumerate(group_db):
        if grp[3] >= folder_size_min_files:
            dir_name = get_group_color_name(group_index)
            dir_name = new_folder + str(group_index + 1).zfill(new_folder_digits) + " - (%03d)" % grp[3] + dir_name
            try:
                os.makedirs(dir_name)
            except Exception as e:
                print("Could not create folder ", e)
            for i in grp[0]:
                for im_path in image_DB[i][0]:
                    try:
                        action(im_path, dir_name + "/")
                    except Exception as e:
                        print("Could not complete file operation ", e)
                if local_run_index != run_index:
                    return
                QApplication.processEvents()
        else:
            ungroupables += grp[0]
        progress_value = group_index

    if len(ungroupables) > 0:
        files_count = sum([image_DB[i][3] for i in ungroupables])
        dir_name = new_folder + "_ungroupped" + " - (%03d)" % files_count
        try:
            os.makedirs(dir_name)
        except Exception as e:
            print("Could not create folder", e)
        for i in ungroupables:
            for im_path in image_DB[i][0]:
                try:
                    action(im_path, dir_name)
                except Exception as e:
                    print("Could not complete file operation ", e)
            if local_run_index != run_index:
                return
            QApplication.processEvents()


def get_group_color_name(group_n):
    global group_db
    global new_folder
    if "HSV" not in selected_color_spaces:
        return ""
    for subspace in "HSV":
        if subspace not in selected_color_subspaces:
            return ""

    # colors_list = mc.CSS4_COLORS
    colors_list = mc.XKCD_COLORS

    group_color = np.split(group_db[group_n][1][:hg_bands_count * 3], 3)
    group_color = ndimage.zoom(group_color, [1, 256 / hg_bands_count], mode='constant')
    group_color_one = np.zeros(3)
    group_color_one[0] = np.argmax(gaussian_filter1d(group_color[0], 30, mode="wrap")) / 256
    for i in [1, 2]:
        median_value = np.percentile(group_color[i], 70)
        filtered_channel = np.where(group_color[i] > median_value, group_color[i], 0)
        group_color_one[i] = ndimage.center_of_mass(filtered_channel)[0] / 256

    min_colours = {}
    for color_name, color_value in colors_list.items():
        color_value_hsv = mc.rgb_to_hsv(mc.to_rgb(color_value))
        min_colours[la_norm(color_value_hsv - group_color_one)] = color_name

    return " " + min_colours[min(min_colours.keys())][5:].replace("/", "-")


def export_image_vectors():
    global group_db
    global groups_count
    global run_index
    global new_vector_folder

    local_run_index = run_index

    try:
        os.makedirs(new_vector_folder)
    except Exception as e:
        print("Folder already exists", e)

    save_db = [row[1] for row in group_db]
    np.savetxt(new_vector_folder + "Groups.csv", save_db, fmt='%1.4f', delimiter=";")

    for group_n in range(groups_count):
        save_db = []
        for image_index in group_db[group_n][0]:
            save_db.append(
                [image_DB[image_index][0][0]] + image_DB[image_index][1] + [image_DB[image_index][2]])
        np.savetxt(new_vector_folder + "Group_vectors_" + str(group_n + 1) + ".csv", save_db, delimiter=";",
                   fmt='%s', newline='\n')
        if local_run_index != run_index:
            return


def create_samples():
    global group_db
    global groups_count
    global run_index
    global new_vector_folder
    global progress_max
    global progress_value
    global status_text

    if "HSV" not in selected_color_spaces:
        return
    for subspace in "HSV":
        if subspace not in selected_color_subspaces:
            return

    local_run_index = run_index

    roll_settings = [[0, 1500], [1, 1500],
                     [0, 1500], [1, 1500],
                     [0, 1500], [1, 1500],
                     [0, 1500], [1, 1500],
                     [0, 1500], [1, 1500]]

    try:
        os.makedirs(new_vector_folder + "/subvectors")
    except Exception as e:
        print("Folder already exists", e)

    status_text = "Generating folder images."
    QApplication.processEvents()

    progress_max = groups_count * 3
    progress_value = 0

    for group_n in range(groups_count):
        im_size = 2000, 1200
        random.seed()

        img_data = [Image] * 3
        for color_band in range(3):
            img_data_band = np.array([], dtype='uint8')
            bins_sum = 0.
            for hg_bin in range(hg_bands_count):
                bins_sum += group_db[group_n][1][color_band * hg_bands_count + hg_bin]
            band_coefficient = im_size[0] * im_size[1] / bins_sum / 2
            for hg_bin in range(hg_bands_count):
                bin_start = 256 * hg_bin // hg_bands_count
                bin_end = 256 * (hg_bin + 1) // hg_bands_count
                band_index = color_band * hg_bands_count + hg_bin
                bin_value = int(group_db[group_n][1][band_index] * band_coefficient)
                img_data_chunk = np.linspace(bin_start, bin_end, bin_value, endpoint=False, dtype='uint8')
                img_data_band = np.append(img_data_band, img_data_chunk)
            if im_size[0] * im_size[1] // 2 > len(img_data_band):
                img_data_chunk = np.full(im_size[0] * im_size[1] // 2 - len(img_data_band), 255, dtype='uint8')
                img_data_band = np.append(img_data_band, img_data_chunk)
            if im_size[0] * im_size[1] // 2 < len(img_data_band):
                img_data_band = np.resize(img_data_band, im_size[0] * im_size[1] // 2)

            img_data_band = np.reshape(img_data_band, (im_size[1] // 2, im_size[0]))
            img_data_band = np.vstack([img_data_band, np.flipud(img_data_band)])
            for one_set in roll_settings:
                img_data_band = roll_rows2(img_data_band, one_set[0], one_set[1])
                if local_run_index != run_index:
                    return

            gaussian_filter(img_data_band, 2, output=img_data_band, mode='wrap')
            median_filter(img_data_band, 7, output=img_data_band, mode='wrap')

            img_data[color_band] = Image.frombytes("L", im_size, img_data_band.copy(order='C'))

            progress_value += 1
            QApplication.processEvents()

            img_data[color_band].save(
                new_vector_folder + "/subvectors/" + "Group_" + "%03d_s_%d" % ((group_n + 1), color_band) + ".png")

        img = Image.merge("HSV", img_data)

        image_rgb = img.convert("RGB")
        image_rgb.save(new_vector_folder + "Group_" + "%03d" % (group_n + 1) + ".png")

        status_text = "Generating forlder images. " + "%1d of %1d done." % ((group_n + 1), groups_count)


def roll_rows2(input_layer, axis, max_speed):
    if axis == 1:
        input_layer = input_layer.T

    speed_row = np.linspace(-max_speed, max_speed, input_layer.shape[0])
    random.shuffle(speed_row)
    speed_row = gaussian_filter1d(speed_row, math.sqrt(max_speed) * 5, mode='wrap')
    displacement_row = np.zeros_like(speed_row)

    for i in range(10):
        displacement_row = np.roll(np.add(displacement_row * .9, speed_row), 1)
    displacement_row = displacement_row.astype('int32')

    moved_list = map(np.roll, input_layer, displacement_row)
    input_layer = np.array(list(moved_list), copy=False)

    if axis == 1:
        input_layer = input_layer.T

    return input_layer.copy(order='C')


# This function is currently not used
def rotate_squares(input_layer, square_size):
    result = []
    for chunk in np.split(input_layer, input_layer.shape[0] // square_size, 0):
        rows = []
        for square in np.split(chunk, input_layer.shape[1] // square_size, 1):
            rows.append(np.rot90(square, random.randrange(4)))
        result.append(rows)
        QApplication.processEvents()
    result = np.roll(np.block(result), square_size // 2, axis=(0, 1))
    return result


# This function is currently not used
def shuffle_single_layer(input_layer, square_size):
    w = input_layer.shape[1]
    h = input_layer.shape[0]
    squares = input_layer
    squares = np.hsplit(squares, w // square_size)
    squares = np.vstack(squares)
    squares = np.reshape(squares, -1)
    squares = np.hsplit(squares, w * h // (square_size * square_size))
    random.shuffle(squares)
    squares = np.reshape(squares, -1)
    squares = np.hsplit(squares, w * h // square_size)
    squares = np.vstack(squares)
    squares = np.vsplit(squares, w // square_size)
    squares = np.hstack(squares)
    squares = np.reshape(squares, input_layer.shape)
    return squares


class AnimatedHistogram:
    band_max = 0

    def __init__(self, im_db, gr_db):
        bands_count = len(im_db[0][1])
        AnimatedHistogram.band_max = 0
        left = np.arange(bands_count)
        right = left + 1
        bottom = np.zeros(bands_count)
        top = bottom

        verts_count = bands_count * (1 + 3 + 1)
        codes = np.ones(verts_count, int) * path.Path.LINETO
        codes[0::5] = path.Path.MOVETO
        codes[4::5] = path.Path.CLOSEPOLY

        img_verts = np.zeros((verts_count, 2))
        img_verts[0::5, 0] = left
        img_verts[0::5, 1] = bottom
        img_verts[1::5, 0] = left
        img_verts[1::5, 1] = top
        img_verts[2::5, 0] = right
        img_verts[2::5, 1] = top
        img_verts[3::5, 0] = right
        img_verts[3::5, 1] = bottom

        grp_verts = np.zeros((verts_count, 2))
        grp_verts[0::5, 0] = left + .4
        grp_verts[0::5, 1] = bottom
        grp_verts[1::5, 0] = left + .4
        grp_verts[1::5, 1] = top
        grp_verts[2::5, 0] = right - .4
        grp_verts[2::5, 1] = top
        grp_verts[3::5, 0] = right - .4
        grp_verts[3::5, 1] = bottom

        patch_img = None

        def animate(_):
            try:
                next_pair = next(AnimatedHistogram.image_n_group_iterator)
            except Exception as e:
                AnimatedHistogram.image_n_group_iterator = chain.from_iterable(
                    [[(grp[1], im_db[im][1], grp_n, im_db[im][0]) for im in grp[0]] for grp_n, grp in enumerate(gr_db)])
                return []
            gr_top = next_pair[0]
            im_top = next_pair[1]
            AnimatedHistogram.band_max = max(AnimatedHistogram.band_max, max(gr_top), max(im_top))
            ax.set_ylim(0, AnimatedHistogram.band_max)
            t.set_position((bands_count // 10, AnimatedHistogram.band_max * .9))
            new_font_size = fig.get_size_inches() * fig.dpi // 50
            t.set_fontsize(new_font_size[1])
            new_text = "Group " + "%03d " % (next_pair[2] + 1)
            for im_path in next_pair[3]:
                new_text += "\n" + os.path.basename(im_path)
            t.set_text(new_text)
            img_verts[1::5, 1] = im_top
            img_verts[2::5, 1] = im_top
            grp_verts[1::5, 1] = gr_top
            grp_verts[2::5, 1] = gr_top
            return [patch_img, patch_grp, t]

        fig, ax = plt.subplots()
        ax.set_xlim(0, bands_count)

        t = mtext.Text(3, 2.5, 'text label', ha='left', va='bottom', axes=ax)

        patch_img = patches.PathPatch(path.Path(img_verts, codes), facecolor='darkgreen', edgecolor='darkgreen',
                                      alpha=1)
        patch_grp = patches.PathPatch(path.Path(grp_verts, codes), facecolor='royalblue', edgecolor='royalblue',
                                      alpha=1)
        ax.add_patch(patch_img)
        ax.add_patch(patch_grp)
        ax.add_artist(t)

        # ani = animation.FuncAnimation(fig, animate, interval=800, save_count=10, repeat=False, blit=True)
        animation.FuncAnimation(fig, animate, interval=800, save_count=10, repeat=False, blit=True)
        plt.show()


class VisImSorterGUI(QMainWindow, VisImSorterInterface.Ui_VisImSorter):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.timer = QTimer()
        self.drag_timer = QTimer()
        self.font = PyQt5.QtGui.QFont()
        self.init_elements()

    def init_elements(self):
        global start_folder

        self.font.setPointSize(14)

        self.timer.timeout.connect(self.redraw_dialog)
        self.timer.setInterval(50)
        self.drag_timer.timeout.connect(self.drag_timeout)
        self.drag_timer.setInterval(3000)

        self.select_folder_button.clicked.connect(self.select_folder)
        self.select_folder_button.resizeEvent = self.directory_changed
        self.select_folder_button.dragLeaveEvent = self.directory_changed
        self.select_folder_button.dragEnterEvent = self.directory_entered
        self.select_folder_button.dropEvent = self.directory_dropped
        self.directory_changed()

        self.slider_histogram_bands.valueChanged.connect(self.slider_histogram_bands_changed)
        self.slider_histogram_bands_changed()

        self.slider_enforce_equal.valueChanged.connect(self.slider_equal_changed)
        self.slider_equal_changed()

        self.list_color_spaces.itemSelectionChanged.connect(self.color_spaces_reselected)
        self.color_spaces_reselected()

        self.init_color_space(self.list_color_spaces_CMYK)
        self.init_color_space(self.list_color_spaces_HSV)
        self.init_color_space(self.list_color_spaces_RGB)
        self.init_color_space(self.list_color_spaces_YCbCr)

        self.btn_stop.clicked.connect(self.stop_button_pressed)
        self.btn_start.clicked.connect(self.start_button_pressed)
        self.progressBar.setVisible(False)
        self.enable_elements()

        if len(sys.argv) > 1:
            if os.path.isdir(sys.argv[1]):
                start_folder = sys.argv[1]
                self.directory_changed()

    def start_button_pressed(self):
        global target_groups
        global target_group_size
        global search_subdirectories
        global selected_color_spaces
        global selected_color_subspaces
        global move_files_not_copy
        global show_histograms
        global export_histograms
        global run_index
        global status_text
        global abort_reason
        global start_folder
        global new_folder
        global create_samples_enabled
        global folder_size_min_files
        global folder_size_min_files
        global process_version
        global final_sort
        global group_with_similar_name
        global enable_stage2_grouping
        global folder_constraint_type
        global folder_constraint_value
        global stage1_grouping_type
        global enable_multiprocessing

        local_run_index = run_index

        folder_constraint_type = self.combo_folder_constraints.currentIndex()
        folder_constraint_value = self.spin_num_constraint.value()
        search_subdirectories = self.check_subdirs_box.isChecked()
        move_files_not_copy = self.radio_move.isChecked()
        show_histograms = self.check_show_histograms.isChecked()
        export_histograms = self.check_export_histograms.isChecked()
        create_samples_enabled = self.check_create_samples.isChecked()
        enable_stage2_grouping = self.check_stage2_grouping.isChecked()
        stage1_grouping_type = self.combo_stage1_grouping.currentIndex()
        enable_multiprocessing = self.check_multiprocessing.isChecked()

        final_sort = self.combo_final_sort.currentIndex()
        if self.check_equal_name.isChecked():
            group_with_similar_name = self.spin_equal_first_symbols.value()
        else:
            group_with_similar_name = 0

        selected_color_spaces = []
        selected_color_subspaces = []

        for item in self.list_color_spaces.selectedItems():
            selected_color_spaces.append(str(item.text()))
            selected_color_subspaces += self.get_sub_color_space(item.text())

        if len(selected_color_subspaces) == 0:
            QMessageBox.warning(None, "VisImSorter: Error",
                                "Please select at least one color space and at least one sub-color space")
            return

        self.disable_elements()
        start_process()
        if local_run_index != run_index:
            status_text = abort_reason
            self.btn_stop.setText(status_text)
            self.btn_stop.setStyleSheet("background-color: rgb(250,150,150)")
        else:
            self.btn_stop.setText(status_text)
            self.btn_stop.setStyleSheet("background-color: rgb(150,250,150)")
            run_index += 1
        self.btn_stop.setFont(self.font)
        self.progressBar.setVisible(False)

    def stop_button_pressed(self):
        global run_index
        global abort_reason
        if self.progressBar.isVisible():
            run_index += 1
            abort_reason = "Process aborted by user."
        else:
            self.directory_changed(True)
            self.enable_elements()

    def get_sub_color_space(self, color_space_name):
        w = self.group_colorspaces_all.findChild(PyQt5.QtWidgets.QListWidget, "list_color_spaces_" + color_space_name)
        subspaces = []
        for item in w.selectedItems():
            subspaces.append(item.text())
        return subspaces

    def color_spaces_reselected(self):
        for index in range(self.horizontalLayout_7.count()):
            short_name = self.horizontalLayout_7.itemAt(index).widget().objectName()[18:]
            items = self.list_color_spaces.findItems(short_name, PyQt5.QtCore.Qt.MatchExactly)
            if len(items) > 0:
                enabled = items[0].isSelected()
                self.horizontalLayout_7.itemAt(index).widget().setEnabled(enabled)

    def init_color_space(self, color_space_list):
        for i in range(color_space_list.count()):
            item = color_space_list.item(i)
            if item.text() != "K":
                item.setSelected(True)

    def enable_elements(self):
        global progress_value
        self.group_input.setEnabled(True)
        self.group_analyze.setEnabled(True)
        self.group_sizes.setEnabled(True)
        self.group_move.setEnabled(True)
        self.group_final.setEnabled(True)
        self.btn_start.setVisible(True)
        self.btn_stop.setVisible(False)
        self.btn_stop.setStyleSheet("background-color: rgb(200,150,150)")
        self.btn_stop.setFont(self.font)
        progress_value = 0
        self.redraw_dialog()
        self.timer.stop()

    def disable_elements(self):
        self.group_input.setEnabled(False)
        self.group_analyze.setEnabled(False)
        self.group_sizes.setEnabled(False)
        self.group_move.setEnabled(False)
        self.group_final.setEnabled(False)
        self.btn_start.setVisible(False)
        self.btn_stop.setText("Stop")
        self.btn_stop.setVisible(True)
        self.progressBar.setVisible(True)
        self.timer.start()

    def slider_histogram_bands_changed(self):
        global hg_bands_count
        hg_bands_count = self.slider_histogram_bands.value()
        self.lbl_histogram_bands.setText(str(hg_bands_count))

    def slider_equal_changed(self):
        global enforce_equal_folders
        enforce_equal_folders = self.slider_enforce_equal.value() / 100
        enforce_equal_folders *= enforce_equal_folders
        self.lbl_slider_enforce_equal.setText("%d" % (self.slider_enforce_equal.value() / 3) + "%")

    def drag_timeout(self):
        self.directory_changed()
        self.drag_timer.stop()

    def directory_entered(self, input_object):
        if input_object.mimeData().hasUrls():
            firt_path = input_object.mimeData().text()[8:]
            firt_path_start = input_object.mimeData().text()[:8]
            if firt_path_start == "file:///":
                if os.path.isdir(firt_path):
                    self.select_folder_button.setText("Drop folder here")
                    input_object.accept()
                    return
        self.select_folder_button.setText("Only directory accepted")
        input_object.ignore()
        self.drag_timer.start()

    def directory_dropped(self, in_dir):
        global start_folder
        firt_path = in_dir.mimeData().text()[8:]
        firt_path_start = in_dir.mimeData().text()[:8]
        if firt_path_start == "file:///":
            if os.path.isdir(firt_path):
                start_folder = firt_path
                self.directory_changed()

    def directory_changed(self, suppress_text=False):
        global start_folder
        global status_text

        line_length = self.select_folder_button.width() / 20

        if start_folder.endswith("/"):
            start_folder = start_folder[:-1]
        if len(start_folder) > 0:
            button_text = start_folder
            if not suppress_text:
                status_text = "Ready to start"
            if len(start_folder) > line_length:
                lines_count = len(start_folder) // line_length
                line_max = len(start_folder) // lines_count
                path_list = re.split("/", start_folder)
                button_text = ""
                line_length = 0
                for part_path in path_list:
                    extended_line_length = line_length + len(part_path) + 1
                    if (extended_line_length > line_max) and (extended_line_length > 5):
                        button_text += "\n" + part_path + "/"
                        line_length = len(part_path + "/")
                    else:
                        button_text += part_path + "/"
                        line_length += len(part_path + "/")

            self.select_folder_button.setText(button_text)
            self.btn_start.setEnabled(True)
        else:
            if not suppress_text:
                status_text = "Please select folder"
            self.select_folder_button.setText("Select folder")
            self.btn_start.setEnabled(False)

    def select_folder(self):
        global start_folder
        start_folder = QFileDialog.getExistingDirectory(self, "Choose directory", start_folder or "Y:/")
        self.directory_changed()

    def redraw_dialog(self):
        global progress_value
        global status_text
        global progress_max
        if self.progressBar.maximum() != progress_max:
            self.progressBar.setRange(0, progress_max)
        self.progressBar.setValue(int(progress_value))
        self.statusbar.showMessage(status_text)
        self.update()
        QApplication.processEvents()


def main():
    app = QApplication(sys.argv)
    sorter_window = VisImSorterGUI()
    sorter_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
