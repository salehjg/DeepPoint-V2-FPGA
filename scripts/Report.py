import json
import copy
import glob, os
import sys
from datetime import datetime
import numpy as np
from pathlib import Path
import zipfile
import colorama


class CProfiler:
    """
    duration: all reports (nano seconds) (cpu or xil) (layer or kernel)
    """

    def __init__(self, path_json):
        self.path_json = path_json
        self.file_json = open(path_json)
        self.src_json = json.load(self.file_json)  # dict
        self.file_json.close()

    def find_kernel(self, parent_layer_id):
        """
        returns a list of kernels with the requested id's
        :param parent_layer_id:
        :return:
        """
        matched = []
        for item in self.src_json['trace']:
            if item['type'] != "kernel":
                continue
            else:
                if item["id"] == parent_layer_id:
                    matched.append(item)
        return matched

    def recursive_find_layers_by_name(self, tree, layer_name, platform):
        if tree['type'] != 'layer':
            return []
        if len(tree['nested']) == 0:
            if tree['type'] == 'layer' and tree['name'] == layer_name and tree['platform'] == platform:
                dict_matched = copy.deepcopy(tree)
                del dict_matched['nested']
                return [dict_matched]
            else:
                return []
        else:
            list_matched_layers = []
            if tree['type'] == 'layer' and tree['name'] == layer_name and tree['platform'] == platform:
                dict_matched = copy.deepcopy(tree)
                del dict_matched['nested']
                list_matched_layers.append(dict_matched)
            if len(tree['nested']) != 0:
                for child in tree['nested']:
                    list_tmp = self.recursive_find_layers_by_name(child, layer_name, platform)
                    if len(list_tmp) != 0:
                        for e in list_tmp:
                            list_matched_layers.append(e)
            return list_matched_layers

    def find_layers_by_name_perkernel(self, layer_name, platform):
        assert platform == 'xil'
        list_results = []
        for parent in self.src_json['trace']:
            if parent['type'] == 'kernel':
                continue
            list_matched = self.recursive_find_layers_by_name(parent, layer_name, platform)
            if len(list_matched) != 0:
                for e in list_matched:
                    list_results.append(e)
        return list_results

    def report_info(self):
        return self.src_json['info']

    def is_layer_reducesum(self, tree):
        result = False
        if tree['type'] == 'layer':
            if tree['name'] == 'Reduce':
                if tree['args']['reduction_op'] == 0:  # sum
                    result = True
        return result

    def is_layer_reducemax(self, tree):
        result = False
        if tree['type'] == 'layer':
            if tree['name'] == 'Reduce':
                if tree['args']['reduction_op'] == 1:  # max
                    result = True
        return result

    def report_hosttimes_pertoplayer(self, platform):
        """
        Reports accumulated host time strictly for every unique the top layer names available.
        :param platform:
        :return:
        """
        dict_report = {}
        for parent in self.src_json['trace']:
            if parent['type'] != 'layer':
                continue
            if parent['platform'] == platform:
                reported_layer_name = ''
                if self.is_layer_reducesum(parent):
                    reported_layer_name = 'Reduce.Sum'
                else:
                    if self.is_layer_reducemax(parent):
                        reported_layer_name = 'Reduce.Max'
                    else:
                        reported_layer_name = parent['name']

                host_ns = (parent['time.stop'] - parent['time.start']) * 1000.0
                if not (reported_layer_name in dict_report.keys()):
                    dict_report[reported_layer_name] = {'pertoplayer.total.host': 0}
                dict_report[reported_layer_name]['pertoplayer.total.host'] += host_ns
        return dict_report

    def report_numcalls_pertoplayer(self, platform):
        """
        Reports accumulated number of calls strictly for every unique the top layer names available.
        :param platform:
        :return:
        """
        dict_report = {}
        for parent in self.src_json['trace']:
            if parent['type'] != 'layer':
                continue
            if parent['platform'] == platform:
                reported_layer_name = ''
                if self.is_layer_reducesum(parent):
                    reported_layer_name = 'Reduce.Sum'
                else:
                    if self.is_layer_reducemax(parent):
                        reported_layer_name = 'Reduce.Max'
                    else:
                        reported_layer_name = parent['name']
                if not (reported_layer_name in dict_report.keys()):
                    dict_report[reported_layer_name] = {'pertoplayer.total.calls': 0}
                dict_report[reported_layer_name]['pertoplayer.total.calls'] += 1
        return dict_report

    def recursive_xiltimes_cpuusage_for_a_tree(self, tree):
        """
        returns a tuple of the total device time and the list of cpu usages for the GIVEN top layer (parent).
        :param tree:
        :return:
        """
        # forced, plat = xil
        if tree['type'] == 'kernel':
            return 0, []
        if len(tree['nested']) == 0:
            if tree['type'] == 'layer' and tree['platform'] == 'xil':
                matched_kernels = self.find_kernel(tree['id'])
                if len(matched_kernels) != 0:
                    tmp = 0
                    for match in matched_kernels:
                        tmp += match['duration']
                    return tmp, [tree['cpu.usage']]
                else:
                    return 0, []
            else:
                return 0, []
        else:
            duration_total = 0
            cpu_usage_list = []
            if tree['type'] == 'layer' and tree['platform'] == 'xil':
                matched_kernels = self.find_kernel(tree['id'])
                if len(matched_kernels) != 0:
                    tmp = 0
                    for match in matched_kernels:
                        tmp += match['duration']
                    duration_total += tmp
            cpu_usage_list.append(tree['cpu.usage'])
            for child in tree['nested']:
                r_duration, r_cpu = self.recursive_xiltimes_cpuusage_for_a_tree(child)
                duration_total += r_duration
                for e in r_cpu:
                    cpu_usage_list.append(e)
            return duration_total, cpu_usage_list

    def report_xiltimes_cpuusages_pertoplayer(self):
        """
        Reports accumulated xilinx device times and sampled cpu usages strictly for every unique the top layer names available.
        :return:
        """
        dict_report = {}

        for parent in self.src_json['trace']:
            if parent['type'] != 'layer' or parent['platform'] != 'xil':
                continue
            reported_layer_name = ''
            if self.is_layer_reducesum(parent):
                reported_layer_name = 'Reduce.Sum'
            else:
                if self.is_layer_reducemax(parent):
                    reported_layer_name = 'Reduce.Max'
                else:
                    reported_layer_name = parent['name']
            if not (reported_layer_name in dict_report.keys()):
                dict_report[reported_layer_name] = {
                    'pertoplayer.total.xiltime': 0,
                    'pertoplayer.cpuusage.min': 0,
                    'pertoplayer.cpuusage.max': 0,
                    'pertoplayer.cpuusage.mean': 0,
                    'pertoplayer.cpuusage.list': []
                }

            total_device, list_cpuusages = self.recursive_xiltimes_cpuusage_for_a_tree(parent)
            for e in list_cpuusages:
                dict_report[reported_layer_name]['pertoplayer.cpuusage.list'].append(e)

            dict_report[reported_layer_name]['pertoplayer.total.xiltime'] += total_device

        for n in dict_report.keys():
            dict_report[n]['pertoplayer.cpuusage.min'] = np.min(dict_report[n]['pertoplayer.cpuusage.list'])
            dict_report[n]['pertoplayer.cpuusage.max'] = np.max(dict_report[n]['pertoplayer.cpuusage.list'])
            dict_report[n]['pertoplayer.cpuusage.mean'] = np.mean(dict_report[n]['pertoplayer.cpuusage.list'])
            del dict_report[n]['pertoplayer.cpuusage.list']

        return dict_report

    def report_detailed_relu_sqrt_square_perkernel(self):
        """
        returns detailed per kernel stats for the different modes of relusqrtsquare kernel.
        :return:
        """
        layers_xil_relu = self.find_layers_by_name_perkernel('ReLU', 'xil')
        layers_xil_sqrt = self.find_layers_by_name_perkernel('Sqrt', 'xil')
        layers_xil_square = self.find_layers_by_name_perkernel('Square', 'xil')

        # ----------------------------------------------------------
        dict_detailed = {}

        # ----------------------------------------------------------
        dict_detailed['relu.xil'] = {}
        dict_detailed['sqrt.xil'] = {}
        dict_detailed['square.xil'] = {}

        # ==========================================================
        dict_detailed['relu.xil']['total.time.host'] = 0
        dict_detailed['relu.xil']['cpu.usage.mean'] = 0
        dict_detailed['relu.xil']['total.time.xil'] = 0
        dict_detailed['relu.xil']['throughput.list'] = []
        dict_detailed['relu.xil']['throughput.max'] = 0
        for e in layers_xil_relu:
            dict_detailed['relu.xil']['total.time.host'] += (e['time.stop'] - e['time.start']) * 1000.0
            dict_detailed['relu.xil']['cpu.usage.mean'] += e['cpu.usage'] / float(len(layers_xil_relu))
            matched_kernels = self.find_kernel(e['id'])
            assert len(matched_kernels) == 1
            for k in matched_kernels:
                dict_detailed['relu.xil']['total.time.xil'] += k['duration']
                dict_detailed['relu.xil']['throughput.list'].append(
                    np.prod(e['args']['shape']) * 4.0 / k['duration'] * 953.674316406)

        dict_detailed['relu.xil']['matches'] = layers_xil_relu
        if len(dict_detailed['relu.xil']['throughput.list']) != 0:
            dict_detailed['relu.xil']['throughput.max'] = np.max(dict_detailed['relu.xil']['throughput.list'])
        else:
            dict_detailed['relu.xil']['throughput.max'] = 0

        # ----------------------------------------------------------
        dict_detailed['sqrt.xil']['total.time.host'] = 0
        dict_detailed['sqrt.xil']['cpu.usage.mean'] = 0
        dict_detailed['sqrt.xil']['total.time.xil'] = 0
        dict_detailed['sqrt.xil']['throughput.list'] = []
        dict_detailed['sqrt.xil']['throughput.max'] = 0
        for e in layers_xil_sqrt:
            dict_detailed['sqrt.xil']['total.time.host'] += (e['time.stop'] - e['time.start']) * 1000.0
            dict_detailed['sqrt.xil']['cpu.usage.mean'] += e['cpu.usage'] / float(len(layers_xil_sqrt))
            matched_kernels = self.find_kernel(e['id'])
            assert len(matched_kernels) == 1
            for k in matched_kernels:
                dict_detailed['sqrt.xil']['total.time.xil'] += k['duration']
                dict_detailed['sqrt.xil']['throughput.list'].append(
                    np.prod(e['args']['shape']) * 4.0 / k['duration'] * 953.674316406)

        dict_detailed['sqrt.xil']['matches'] = layers_xil_sqrt
        if len(dict_detailed['sqrt.xil']['throughput.list']) != 0:
            dict_detailed['sqrt.xil']['throughput.max'] = np.max(dict_detailed['sqrt.xil']['throughput.list'])
        else:
            dict_detailed['sqrt.xil']['throughput.max'] = 0

        # ----------------------------------------------------------
        dict_detailed['square.xil']['total.time.host'] = 0
        dict_detailed['square.xil']['cpu.usage.mean'] = 0
        dict_detailed['square.xil']['total.time.xil'] = 0
        dict_detailed['square.xil']['throughput.list'] = []
        dict_detailed['square.xil']['throughput.max'] = 0
        for e in layers_xil_square:
            dict_detailed['square.xil']['total.time.host'] += (e['time.stop'] - e['time.start']) * 1000.0
            dict_detailed['square.xil']['cpu.usage.mean'] += e['cpu.usage'] / float(len(layers_xil_square))
            matched_kernels = self.find_kernel(e['id'])
            assert len(matched_kernels) == 1
            for k in matched_kernels:
                dict_detailed['square.xil']['total.time.xil'] += k['duration']
                dict_detailed['square.xil']['throughput.list'].append(
                    np.prod(e['args']['shape']) * 4.0 / k['duration'] * 953.674316406)

        dict_detailed['square.xil']['matches'] = layers_xil_square
        if len(dict_detailed['square.xil']['throughput.list']) != 0:
            dict_detailed['square.xil']['throughput.max'] = np.max(dict_detailed['square.xil']['throughput.list'])
        else:
            dict_detailed['square.xil']['throughput.max'] = 0

        # ----------------------------------------------------------
        return dict_detailed

    def report_detailed_reduce_sum_max_perkernel(self):
        """
        returns detailed per kernel stats for the different modes of reduce kernel.
        :return:
        """
        dict_detailed = {}

        dict_detailed['reduce.sum.r3a2.xil'] = {}
        dict_detailed['reduce.sum.r4a012.xil'] = {}
        dict_detailed['reduce.max.xil'] = {}

        layers_xil_sum_and_max = self.find_layers_by_name_perkernel('Reduce', 'xil')

        layers_xil_sum_r3a2 = []
        layers_xil_sum_r4a012 = []
        layers_xil_max = []  # reduce max has only 1 kernel, no need to separate profiled layers

        for e in layers_xil_sum_and_max:
            if e['args']['reduction_op'] == 0:  # sum
                if e['args']['rank'] == 4 and e['args']['combination'] == [1, 1, 1, 0]:
                    layers_xil_sum_r4a012.append(e)
                else:
                    if e['args']['rank'] == 3 and e['args']['combination'] == [0, 0, 1]:
                        layers_xil_sum_r3a2.append(e)
                    else:
                        assert False
            else:
                if e['args']['reduction_op'] == 1:  # max
                    layers_xil_max.append(e)
                else:
                    assert False

        # ==============================================================================================================
        dict_detailed['reduce.max.xil']['total.time.host'] = 0
        dict_detailed['reduce.max.xil']['cpu.usage.mean'] = 0
        dict_detailed['reduce.max.xil']['total.time.xil'] = 0
        dict_detailed['reduce.max.xil']['throughput.list'] = []
        dict_detailed['reduce.max.xil']['throughput.max'] = 0
        for e in layers_xil_max:
            dict_detailed['reduce.max.xil']['total.time.host'] += (e['time.stop'] - e['time.start']) * 1000.0
            dict_detailed['reduce.max.xil']['cpu.usage.mean'] += e['cpu.usage'] / float(len(layers_xil_max))
            matched_kernels = self.find_kernel(e['id'])
            assert len(matched_kernels) == 1
            for k in matched_kernels:
                dict_detailed['reduce.max.xil']['total.time.xil'] += k['duration']
                dict_detailed['reduce.max.xil']['throughput.list'].append(
                    np.prod(e['args']['shape']) * 4.0 / k['duration'] * 953.674316406)

        dict_detailed['reduce.max.xil']['matches'] = layers_xil_max
        if len(dict_detailed['reduce.max.xil']['throughput.list']) != 0:
            dict_detailed['reduce.max.xil']['throughput.max'] = np.max(
                dict_detailed['reduce.max.xil']['throughput.list'])
        else:
            dict_detailed['reduce.max.xil']['throughput.max'] = 0

        # --------------------------------------------------------------------------------------------------------------
        # 2. detailed report for ReduceSum kernels (sum_r3a2 and sum_r4a012)

        dict_detailed['reduce.sum.r3a2.xil']['total.time.host'] = 0
        dict_detailed['reduce.sum.r3a2.xil']['cpu.usage.mean'] = 0
        dict_detailed['reduce.sum.r3a2.xil']['total.time.xil'] = 0
        dict_detailed['reduce.sum.r3a2.xil']['throughput.list'] = []
        dict_detailed['reduce.sum.r3a2.xil']['throughput.max'] = 0
        for e in layers_xil_sum_r3a2:
            dict_detailed['reduce.sum.r3a2.xil']['total.time.host'] += (e['time.stop'] - e['time.start']) * 1000.0
            dict_detailed['reduce.sum.r3a2.xil']['cpu.usage.mean'] += e['cpu.usage'] / float(len(layers_xil_sum_r3a2))
            matched_kernels = self.find_kernel(e['id'])
            assert len(matched_kernels) == 1
            for k in matched_kernels:
                dict_detailed['reduce.sum.r3a2.xil']['total.time.xil'] += k['duration']
                dict_detailed['reduce.sum.r3a2.xil']['throughput.list'].append(
                    np.prod(e['args']['shape']) * 4.0 / k['duration'] * 953.674316406)

        dict_detailed['reduce.sum.r3a2.xil']['matches'] = layers_xil_sum_r3a2
        if len(dict_detailed['reduce.sum.r3a2.xil']['throughput.list']) != 0:
            dict_detailed['reduce.sum.r3a2.xil']['throughput.max'] = np.max(
                dict_detailed['reduce.sum.r3a2.xil']['throughput.list'])
        else:
            dict_detailed['reduce.sum.r3a2.xil']['throughput.max'] = 0

        # -------------------
        dict_detailed['reduce.sum.r4a012.xil']['total.time.host'] = 0
        dict_detailed['reduce.sum.r4a012.xil']['cpu.usage.mean'] = 0
        dict_detailed['reduce.sum.r4a012.xil']['total.time.xil'] = 0
        dict_detailed['reduce.sum.r4a012.xil']['throughput.list'] = []
        dict_detailed['reduce.sum.r4a012.xil']['throughput.max'] = 0
        for e in layers_xil_sum_r4a012:
            dict_detailed['reduce.sum.r4a012.xil']['total.time.host'] += (e['time.stop'] - e['time.start']) * 1000.0
            dict_detailed['reduce.sum.r4a012.xil']['cpu.usage.mean'] += e['cpu.usage'] / float(
                len(layers_xil_sum_r4a012))
            matched_kernels = self.find_kernel(e['id'])
            assert len(matched_kernels) == 1
            for k in matched_kernels:
                dict_detailed['reduce.sum.r4a012.xil']['total.time.xil'] += k['duration']
                dict_detailed['reduce.sum.r4a012.xil']['throughput.list'].append(
                    np.prod(e['args']['shape']) * 4.0 / k['duration'] * 953.674316406)

        dict_detailed['reduce.sum.r4a012.xil']['matches'] = layers_xil_sum_r4a012
        if len(dict_detailed['reduce.sum.r4a012.xil']['throughput.list']) != 0:
            dict_detailed['reduce.sum.r4a012.xil']['throughput.max'] = np.max(
                dict_detailed['reduce.sum.r4a012.xil']['throughput.list'])
        else:
            dict_detailed['reduce.sum.r4a012.xil']['throughput.max'] = 0

        # -------------------
        return dict_detailed

    def report_detailed_pad_unpad_perkernel(self):
        """
        returns detailed per kernel stats for the different modes of padunpad kernel.
        :return:
        """
        dict_detailed = {}

        dict_detailed['padunpad.pad.xil'] = {}
        dict_detailed['padunpad.unpad.xil'] = {}

        layers_xil_pad = self.find_layers_by_name_perkernel('PadLastDim', 'xil')
        layers_xil_unpad = self.find_layers_by_name_perkernel('UnpadLastDim', 'xil')

        # -----------------------
        dict_detailed['padunpad.pad.xil']['total.time.host'] = 0
        dict_detailed['padunpad.pad.xil']['cpu.usage.mean'] = 0
        dict_detailed['padunpad.pad.xil']['total.time.xil'] = 0
        dict_detailed['padunpad.pad.xil']['throughput.list'] = []
        dict_detailed['padunpad.pad.xil']['throughput.max'] = 0
        for e in layers_xil_pad:
            dict_detailed['padunpad.pad.xil']['total.time.host'] += (e['time.stop'] - e['time.start']) * 1000.0
            dict_detailed['padunpad.pad.xil']['cpu.usage.mean'] += e['cpu.usage'] / float(
                len(layers_xil_pad))
            matched_kernels = self.find_kernel(e['id'])
            # assert len(matched_kernels) == 1
            for k in matched_kernels:
                if k['name'] == 'task_pad_unpad':
                    dict_detailed['padunpad.pad.xil']['total.time.xil'] += k['duration']
                    shape_out = copy.deepcopy(e['args']['shape'])
                    shape_out[-1] = e['args']['lastDimPadded']
                    dict_detailed['padunpad.pad.xil']['throughput.list'].append(
                        np.prod(shape_out) * 4.0 / k['duration'] * 953.674316406)
                else:
                    # This is to make sure that only conv2 is allowed to have the same id as pad unpad
                    if k['name'] != 'task_conv2_1x1_direct':
                        assert False

        dict_detailed['padunpad.pad.xil']['matches'] = layers_xil_pad
        if len(dict_detailed['padunpad.pad.xil']['throughput.list']) != 0:
            dict_detailed['padunpad.pad.xil']['throughput.max'] = np.max(
                dict_detailed['padunpad.pad.xil']['throughput.list'])
        else:
            dict_detailed['padunpad.pad.xil']['throughput.max'] = 0

        # -----------------------
        dict_detailed['padunpad.unpad.xil']['total.time.host'] = 0
        dict_detailed['padunpad.unpad.xil']['cpu.usage.mean'] = 0
        dict_detailed['padunpad.unpad.xil']['total.time.xil'] = 0
        dict_detailed['padunpad.unpad.xil']['throughput.list'] = []
        dict_detailed['padunpad.unpad.xil']['throughput.max'] = 0
        for e in layers_xil_unpad:
            dict_detailed['padunpad.unpad.xil']['total.time.host'] += (e['time.stop'] - e['time.start']) * 1000.0
            dict_detailed['padunpad.unpad.xil']['cpu.usage.mean'] += e['cpu.usage'] / float(
                len(layers_xil_unpad))
            matched_kernels = self.find_kernel(e['id'])
            assert len(matched_kernels) == 1
            for k in matched_kernels:
                dict_detailed['padunpad.unpad.xil']['total.time.xil'] += k['duration']
                shape_out = copy.deepcopy(e['args']['shape'])
                shape_out[-1] = e['args']['lastDimUnpadded']
                dict_detailed['padunpad.unpad.xil']['throughput.list'].append(
                    np.prod(shape_out) * 4.0 / k['duration'] * 953.674316406)

        dict_detailed['padunpad.unpad.xil']['matches'] = layers_xil_unpad
        if len(dict_detailed['padunpad.unpad.xil']['throughput.list']) != 0:
            dict_detailed['padunpad.unpad.xil']['throughput.max'] = np.max(
                dict_detailed['padunpad.unpad.xil']['throughput.list'])
        else:
            dict_detailed['padunpad.unpad.xil']['throughput.max'] = 0

        # ----------------
        return dict_detailed

    def report_detailed_transpose_topk_perkernel(self):
        """
        returns detailed per kernel stats for all the kernels with only one operational mode.
        :return:
        """
        dict_detailed = {}
        list_target_layers = [
            'Transpose',
            'TopK',
        ]

        for target_name in list_target_layers:
            if not (target_name in dict_detailed.keys()):
                dict_detailed[target_name] = {}
            layers = self.find_layers_by_name_perkernel(target_name, 'xil')
            dict_detailed[target_name]['total.time.host'] = 0
            dict_detailed[target_name]['cpu.usage.mean'] = 0
            dict_detailed[target_name]['total.time.xil'] = 0
            dict_detailed[target_name]['throughput.list'] = []
            dict_detailed[target_name]['throughput.max'] = 0
            for e in layers:
                dict_detailed[target_name]['total.time.host'] += (e['time.stop'] - e['time.start']) * 1000.0
                dict_detailed[target_name]['cpu.usage.mean'] += e['cpu.usage'] / float(len(layers))
                matched_kernels = self.find_kernel(e['id'])
                # assert len(matched_kernels) == 1
                for k in matched_kernels:
                    dict_detailed[target_name]['total.time.xil'] += k['duration']
                    dict_detailed[target_name]['throughput.list'].append(
                        np.prod(e['args']['shape']) * 4.0 / k['duration'] * 953.674316406)
            dict_detailed[target_name]['matches'] = layers
            if len(dict_detailed[target_name]['throughput.list']) != 0:
                dict_detailed[target_name]['throughput.max'] = np.max(
                    dict_detailed[target_name]['throughput.list'])
            else:
                dict_detailed[target_name]['throughput.max'] = 0
        return dict_detailed

    def report_detailed_basicops_perkernel(self):
        """
        returns detailed per kernel stats for all the kernels with only one operational mode.
        :return:
        """
        dict_detailed = {}
        list_target_layers = [
            'BasicOps',
        ]

        for target_name in list_target_layers:
            if not (target_name in dict_detailed.keys()):
                dict_detailed[target_name] = {}
            layers = self.find_layers_by_name_perkernel(target_name, 'xil')
            dict_detailed[target_name]['total.time.host'] = 0
            dict_detailed[target_name]['cpu.usage.mean'] = 0
            dict_detailed[target_name]['total.time.xil'] = 0
            dict_detailed[target_name]['throughput.list'] = []
            dict_detailed[target_name]['throughput.max'] = 0
            for e in layers:
                dict_detailed[target_name]['total.time.host'] += (e['time.stop'] - e['time.start']) * 1000.0
                dict_detailed[target_name]['cpu.usage.mean'] += e['cpu.usage'] / float(len(layers))
                matched_kernels = self.find_kernel(e['id'])
                # assert len(matched_kernels) == 1
                for k in matched_kernels:
                    dict_detailed[target_name]['total.time.xil'] += k['duration']
                    dict_detailed[target_name]['throughput.list'].append(
                        np.prod(e['args']['shape1']) * 4.0 / k['duration'] * 953.674316406)
            dict_detailed[target_name]['matches'] = layers
            if len(dict_detailed[target_name]['throughput.list']) != 0:
                dict_detailed[target_name]['throughput.max'] = np.max(
                    dict_detailed[target_name]['throughput.list'])
            else:
                dict_detailed[target_name]['throughput.max'] = 0
        return dict_detailed

    def report_detailed_tile_perkernel(self):
        """
        returns detailed per kernel stats for all the kernels with only one operational mode.
        :return:
        """
        dict_detailed = {}
        list_target_layers = [
            'Tile',
        ]

        for target_name in list_target_layers:
            if not (target_name in dict_detailed.keys()):
                dict_detailed[target_name] = {}
            layers = self.find_layers_by_name_perkernel(target_name, 'xil')
            dict_detailed[target_name]['total.time.host'] = 0
            dict_detailed[target_name]['cpu.usage.mean'] = 0
            dict_detailed[target_name]['total.time.xil'] = 0
            dict_detailed[target_name]['throughput.list'] = []
            dict_detailed[target_name]['throughput.max'] = 0
            for e in layers:
                dict_detailed[target_name]['total.time.host'] += (e['time.stop'] - e['time.start']) * 1000.0
                dict_detailed[target_name]['cpu.usage.mean'] += e['cpu.usage'] / float(len(layers))
                matched_kernels = self.find_kernel(e['id'])
                # assert len(matched_kernels) == 1
                for k in matched_kernels:
                    dict_detailed[target_name]['total.time.xil'] += k['duration']
                    shape_out = copy.deepcopy(e['args']['shape'])
                    shape_out.insert(e['args']['tileAxis'], e['args']['tileCount'])
                    print('shape out : ', shape_out)
                    dict_detailed[target_name]['throughput.list'].append(
                        np.prod(shape_out) * 4.0 / k['duration'] * 953.674316406)
            dict_detailed[target_name]['matches'] = layers
            if len(dict_detailed[target_name]['throughput.list']) != 0:
                dict_detailed[target_name]['throughput.max'] = np.max(
                    dict_detailed[target_name]['throughput.list'])
            else:
                dict_detailed[target_name]['throughput.max'] = 0
        return dict_detailed

    def report_detailed_gather_perkernel(self):
        """
        returns detailed per kernel stats for all the kernels with only one operational mode.
        :return:
        """
        dict_detailed = {}
        list_target_layers = [
            'Gather',
        ]

        for target_name in list_target_layers:
            if not (target_name in dict_detailed.keys()):
                dict_detailed[target_name] = {}
            layers = self.find_layers_by_name_perkernel(target_name, 'xil')
            dict_detailed[target_name]['total.time.host'] = 0
            dict_detailed[target_name]['cpu.usage.mean'] = 0
            dict_detailed[target_name]['total.time.xil'] = 0
            dict_detailed[target_name]['throughput.list'] = []
            dict_detailed[target_name]['throughput.max'] = 0
            for e in layers:
                dict_detailed[target_name]['total.time.host'] += (e['time.stop'] - e['time.start']) * 1000.0
                dict_detailed[target_name]['cpu.usage.mean'] += e['cpu.usage'] / float(len(layers))
                matched_kernels = self.find_kernel(e['id'])
                # assert len(matched_kernels) == 1
                for k in matched_kernels:
                    dict_detailed[target_name]['total.time.xil'] += k['duration']
                    shape_out = copy.deepcopy(e['args']['shape'])
                    assert e['args']['indicesOfAxis'] == 1
                    if 'shape.indices' in e['args'].keys():
                        shape_out.insert(2, e['args']['shape.indices'][-1])
                    else:
                        # if the profiler.json belongs to a commit at which the second shape is not recorded:
                        shape_out.insert(2, 20)
                    print('shape out : ', shape_out)
                    dict_detailed[target_name]['throughput.list'].append(
                        np.prod(shape_out) * 4.0 / k['duration'] * 953.674316406)
            dict_detailed[target_name]['matches'] = layers
            if len(dict_detailed[target_name]['throughput.list']) != 0:
                dict_detailed[target_name]['throughput.max'] = np.max(
                    dict_detailed[target_name]['throughput.list'])
            else:
                dict_detailed[target_name]['throughput.max'] = 0
        return dict_detailed

    def report_detailed_concat2_perkernel(self):
        """
        returns detailed per kernel stats for all the kernels with only one operational mode.
        :return:
        """
        dict_detailed = {}
        list_target_layers = [
            'Concat2',
        ]

        for target_name in list_target_layers:
            if not (target_name in dict_detailed.keys()):
                dict_detailed[target_name] = {}
            layers = self.find_layers_by_name_perkernel(target_name, 'xil')
            dict_detailed[target_name]['total.time.host'] = 0
            dict_detailed[target_name]['cpu.usage.mean'] = 0
            dict_detailed[target_name]['total.time.xil'] = 0
            dict_detailed[target_name]['throughput.list'] = []
            dict_detailed[target_name]['throughput.max'] = 0
            for e in layers:
                dict_detailed[target_name]['total.time.host'] += (e['time.stop'] - e['time.start']) * 1000.0
                dict_detailed[target_name]['cpu.usage.mean'] += e['cpu.usage'] / float(len(layers))
                matched_kernels = self.find_kernel(e['id'])
                # assert len(matched_kernels) == 1
                for k in matched_kernels:
                    dict_detailed[target_name]['total.time.xil'] += k['duration']
                    shape_out = copy.deepcopy(e['args']['shape1'])
                    shape_out[e['args']['concatAxis']] += e['args']['shape2'][e['args']['concatAxis']]
                    print('shape out : ', shape_out)
                    dict_detailed[target_name]['throughput.list'].append(
                        np.prod(shape_out) * 4.0 / k['duration'] * 953.674316406)
            dict_detailed[target_name]['matches'] = layers
            if len(dict_detailed[target_name]['throughput.list']) != 0:
                dict_detailed[target_name]['throughput.max'] = np.max(
                    dict_detailed[target_name]['throughput.list'])
            else:
                dict_detailed[target_name]['throughput.max'] = 0
        return dict_detailed

    def report_detailed_matmul_perkernel(self):
        """
        returns detailed per kernel stats for all the kernels with only one operational mode.
        :return:
        """
        dict_detailed = {}
        list_target_layers = [
            'MatMul',
        ]

        for target_name in list_target_layers:
            if not (target_name in dict_detailed.keys()):
                dict_detailed[target_name] = {}
            layers = self.find_layers_by_name_perkernel(target_name, 'xil')
            dict_detailed[target_name]['total.time.host'] = 0
            dict_detailed[target_name]['cpu.usage.mean'] = 0
            dict_detailed[target_name]['total.time.xil'] = 0
            dict_detailed[target_name]['GFlopPerSecond.list'] = []
            dict_detailed[target_name]['GFlopPerSecond.max'] = 0
            for e in layers:
                dict_detailed[target_name]['total.time.host'] += (e['time.stop'] - e['time.start']) * 1000.0
                dict_detailed[target_name]['cpu.usage.mean'] += e['cpu.usage'] / float(len(layers))
                matched_kernels = self.find_kernel(e['id'])
                # assert len(matched_kernels) == 1
                for k in matched_kernels:
                    dict_detailed[target_name]['total.time.xil'] += k['duration']
                    rank = len(e['args']['shape1'])
                    if rank == 2:
                        _B = 1
                        _N = e['args']['shape1'][0]
                        _K = e['args']['shape1'][1]
                        assert e['args']['shape2'][0] == _K
                        _M = e['args']['shape2'][1]
                    else:
                        if rank == 3:
                            _B = e['args']['shape1'][0]
                            _N = e['args']['shape1'][1]
                            _K = e['args']['shape1'][2]
                            assert e['args']['shape2'][0] == _B
                            assert e['args']['shape2'][1] == _K
                            _M = e['args']['shape2'][2]
                        else:
                            assert False
                    n_ops = 2 * _N * _K * _M * _B
                    print('nOps : ', n_ops)
                    dict_detailed[target_name]['GFlopPerSecond.list'].append(
                        float(n_ops) / float(k['duration']))
            dict_detailed[target_name]['matches'] = layers
            if len(dict_detailed[target_name]['GFlopPerSecond.list']) != 0:
                dict_detailed[target_name]['GFlopPerSecond.max'] = np.max(
                    dict_detailed[target_name]['GFlopPerSecond.list'])
            else:
                dict_detailed[target_name]['GFlopPerSecond.max'] = 0
        return dict_detailed

    def report_detailed_sharedmlp_perkernel(self):
        """
        returns detailed per kernel stats for all the kernels with only one operational mode.
        :return:
        """
        dict_detailed = {}
        list_target_layers = [
            'Conv2D',
        ]

        for target_name in list_target_layers:
            if not (target_name in dict_detailed.keys()):
                dict_detailed[target_name] = {}
            layers = self.find_layers_by_name_perkernel(target_name, 'xil')
            dict_detailed[target_name]['total.time.host'] = 0
            dict_detailed[target_name]['cpu.usage.mean'] = 0
            dict_detailed[target_name]['total.time.xil'] = 0
            dict_detailed[target_name]['GFlopPerSecond.list'] = []
            dict_detailed[target_name]['GFlopPerSecond.max'] = 0
            for e in layers:
                dict_detailed[target_name]['total.time.host'] += (e['time.stop'] - e['time.start']) * 1000.0
                dict_detailed[target_name]['cpu.usage.mean'] += e['cpu.usage'] / float(len(layers))
                matched_kernels = self.find_kernel(e['id'])
                # assert len(matched_kernels) == 1
                for k in matched_kernels:
                    dict_detailed[target_name]['total.time.xil'] += k['duration']

                    assert len(e['args']['shape.i']) == 4
                    assert len(e['args']['shape.w']) == 4
                    assert e['args']['shape.w'][0] == 1
                    assert e['args']['shape.w'][1] == 1
                    assert len(e['args']['shape.b']) == 1

                    _B = e['args']['shape.i'][0]
                    _N = e['args']['shape.i'][1]
                    _K = e['args']['shape.i'][2]
                    _C1 = e['args']['shape.i'][3]
                    assert _C1 == e['args']['shape.w'][2]
                    _C2 = e['args']['shape.w'][3]
                    assert _C2 == e['args']['shape.b'][0]
                    n_ops = _C2 * (1 + 2 * _B * _N * _K * _C1)
                    print('nOps : ', n_ops)
                    dict_detailed[target_name]['GFlopPerSecond.list'].append(
                        float(n_ops) / float(k['duration']))
            dict_detailed[target_name]['matches'] = layers
            if len(dict_detailed[target_name]['GFlopPerSecond.list']) != 0:
                dict_detailed[target_name]['GFlopPerSecond.max'] = np.max(
                    dict_detailed[target_name]['GFlopPerSecond.list'])
            else:
                dict_detailed[target_name]['GFlopPerSecond.max'] = 0
        return dict_detailed

    def report_detailed_datamover_perkernel(self):
        dict_detailed = {'DataMover': {'throughput.list': [], 'total.time.xil': 0}}
        for record in self.src_json['trace']:
            if record['type'] != 'kernel' or record['platform'] != 'xil':
                continue
            if record['name'] == 'task_datamover':
                datasize_bytes = record['bytes']
                duration_ns = record['duration']
                dict_detailed['DataMover']['throughput.list'].append(datasize_bytes / duration_ns * 953.674316406)
                dict_detailed['DataMover']['total.time.xil'] += duration_ns

        if len(dict_detailed['DataMover']['throughput.list']) != 0:
            dict_detailed['DataMover']['throughput.min'] = np.min(dict_detailed['DataMover']['throughput.list'])
            dict_detailed['DataMover']['throughput.max'] = np.max(dict_detailed['DataMover']['throughput.list'])
            dict_detailed['DataMover']['throughput.mean'] = np.mean(dict_detailed['DataMover']['throughput.list'])
        return dict_detailed

class CReporter:
    def __init__(self, path_json):
        self.obj = CProfiler(path_json)
        self.report = []
        self.today = datetime.now()

        path_only = path_json[0:path_json.rindex('/')]
        self.new_dump_dir = os.path.join(path_only, self.today.strftime('%Y%m%d%H%M%S'))
        os.mkdir(self.new_dump_dir)

        self.report.append(self.obj.report_info())
        self.print_report(self.report[-1], self.new_dump_dir + "/00Info.json")

        self.report.append(self.obj.report_hosttimes_pertoplayer(platform='cpu'))
        self.print_report(self.report[-1], self.new_dump_dir + "/01PerTopLayerHostTimeCPU.json")

        self.report.append(self.obj.report_hosttimes_pertoplayer(platform='xil'))
        self.print_report(self.report[-1], self.new_dump_dir + "/02PerTopLayerHostTimeXIL.json")

        self.report.append(self.obj.report_xiltimes_cpuusages_pertoplayer())
        self.print_report(self.report[-1], self.new_dump_dir + "/03PerTopLayerDeviceTimeAndCpuUsagesXIL.json")

        self.report.append(self.obj.report_numcalls_pertoplayer('cpu'))
        self.print_report(self.report[-1], self.new_dump_dir + "/04PerTopLayerNumberOfCallsCPU.json")

        self.report.append(self.obj.report_numcalls_pertoplayer('xil'))
        self.print_report(self.report[-1], self.new_dump_dir + "/05PerTopLayerNumberOfCallsXIL.json")

        self.report.append(self.obj.report_detailed_relu_sqrt_square_perkernel())
        self.print_report(self.report[-1], self.new_dump_dir + "/06PerKernelDetailedReluSqrtSquare.json")

        self.report.append(self.obj.report_detailed_reduce_sum_max_perkernel())
        self.print_report(self.report[-1], self.new_dump_dir + "/07PerKernelDetailedReduceSumMax.json")

        self.report.append(self.obj.report_detailed_pad_unpad_perkernel())
        self.print_report(self.report[-1], self.new_dump_dir + "/08PerKernelDetailedPadUnpad.json")

        self.report.append(self.obj.report_detailed_datamover_perkernel())
        self.print_report(self.report[-1], self.new_dump_dir + "/09PerKernelDetailedDataMover.json")

        self.report.append(self.obj.report_detailed_transpose_topk_perkernel())
        self.print_report(self.report[-1], self.new_dump_dir + "/10PerKernelDetailedTransposeAndTopK.json")

        self.report.append(self.obj.report_detailed_basicops_perkernel())
        self.print_report(self.report[-1], self.new_dump_dir + "/11PerKernelDetailedBasicOps.json")

        self.report.append(self.obj.report_detailed_tile_perkernel())
        self.print_report(self.report[-1], self.new_dump_dir + "/12PerKernelDetailedTile.json")

        self.report.append(self.obj.report_detailed_gather_perkernel())
        self.print_report(self.report[-1], self.new_dump_dir + "/13PerKernelDetailedGather.json")

        self.report.append(self.obj.report_detailed_concat2_perkernel())
        self.print_report(self.report[-1], self.new_dump_dir + "/14PerKernelDetailedConcat.json")

        self.report.append(self.obj.report_detailed_matmul_perkernel())
        self.print_report(self.report[-1], self.new_dump_dir + "/15PerKernelDetailedMatMul.json")

        self.report.append(self.obj.report_detailed_sharedmlp_perkernel())
        self.print_report(self.report[-1], self.new_dump_dir + "/16PerKernelDetailedSharedMLP.json")

    def print_report(self, report, dump_fname):
        print(
            json.dumps(report, sort_keys=True, indent=4),
            file=open(dump_fname, "w")
        )


    def single_mode(path_json):
        if not os.path.isfile(path_json):
            print("File does not exist: ", sys.argv[1])
            sys.exit(status=1)

        print("Analyzing the JSON file...")
        obj = CReporter(path_json)
        print("Done. Closing.")


    def batch_mode(path_dir):
        def get_fname_without_ext(fname_with_extension=''):
            fname_only = Path(fname_with_extension).stem
            return fname_only

        def get_fname_with_ext(fname_with_extension=''):
            fname_only = Path(fname_with_extension).name
            return fname_only

        for file in glob.glob(os.path.join(path_dir, '*.zip')):
            print('----------------------')
            folder_path = os.path.join('.', path_dir, get_fname_without_ext(file))
            # os.makedirs(folder_path)
            Path(folder_path).mkdir(parents=True, exist_ok=True)
            file_moved_path = os.path.join(folder_path, get_fname_with_ext(file))
            os.rename(file, file_moved_path)
            print("Unzipping ", file_moved_path)
            with zipfile.ZipFile(file_moved_path, "r") as zip_ref:
                unzipped_folder = os.path.join(folder_path, "unzipped")
                zip_ref.extractall(unzipped_folder)
                json_file_path = os.path.join(unzipped_folder, 'profiler.json')
                print("Analyzing ", json_file_path)
                try:
                    obj = CReporter(json_file_path)
                except:
                    print(
                        f"{colorama.Fore.RED}Error analyzing {json_file_path} of size (bytes) {os.path.getsize(json_file_path)} {colorama.Style.RESET_ALL}")


    def print_help():
        print("DeepPoint-V2-FPGA Report Script")
        print("Usage:")
        print("\tpython3 Report.py <mode: single, batch> args")
        print("\t\t <mode>=single:")
        print("\t\t\t python3 Report.py single <path to profiler.json>")
        print("\t\t <mode>=batch:")
        print("\t\t\t python3 Report.py batch <path to dir with multiple fpga_run zip files>")


    if __name__ == "__main__":
        if len(sys.argv) != 3:
            print_help()
            exit(1)
        else:
            arg_mode = sys.argv[1]
            if arg_mode == 'single':
                single_mode(sys.argv[2])
            else:
                if arg_mode == 'batch':
                    batch_mode(sys.argv[2])
                else:
                    print('Wrong mode.')
                    exit(1)
