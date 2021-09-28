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

    def __init__(self, path_json, datatype_size_bytes=4):
        self.path_json = path_json
        self.file_json = open(path_json)
        self.src_json = json.load(self.file_json)  # dict
        self.file_json.close()
        self.datatype_size_bytes = datatype_size_bytes

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

    def recursive_find_layer_by_id(self, tree, layer_id):
        if tree['type'] == 'kernel':
            return []
        if len(tree['nested']) == 0:
            if tree['id'] == layer_id:
                return [tree]
            else:
                return []
        else:
            matched = []
            for child in tree['nested']:
                k_list = self.recursive_find_layer_by_id(child, layer_id)
                for k in k_list:
                    matched.append(k)
            if tree['id'] == layer_id:
                matched.append(tree)
            return matched

    def find_layer_by_id(self, layer_id):
        rslt_list = []
        for element in self.src_json['trace']:
            k_list = self.recursive_find_layer_by_id(element, layer_id)
            for k in k_list:
                rslt_list.append(k)
        if len(rslt_list) > 1:
            print("find_layer_by_id: Warning, more than one layer is found for an specific layer id.")
            assert False

        return rslt_list

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
            if len(dict_report[n]['pertoplayer.cpuusage.list']) != 0:
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
            assert len(matched_kernels) <= 1
            for k in matched_kernels:
                dict_detailed['relu.xil']['total.time.xil'] += k['duration']
                dict_detailed['relu.xil']['throughput.list'].append(
                    np.prod(e['args']['shape']) * self.datatype_size_bytes / k['duration'] * 953.674316406)

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
            assert len(matched_kernels) <= 1
            for k in matched_kernels:
                dict_detailed['sqrt.xil']['total.time.xil'] += k['duration']
                dict_detailed['sqrt.xil']['throughput.list'].append(
                    np.prod(e['args']['shape']) * self.datatype_size_bytes / k['duration'] * 953.674316406)

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
            assert len(matched_kernels) <= 1
            for k in matched_kernels:
                dict_detailed['square.xil']['total.time.xil'] += k['duration']
                dict_detailed['square.xil']['throughput.list'].append(
                    np.prod(e['args']['shape']) * self.datatype_size_bytes / k['duration'] * 953.674316406)

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
            assert len(matched_kernels) <= 1
            for k in matched_kernels:
                dict_detailed['reduce.max.xil']['total.time.xil'] += k['duration']
                dict_detailed['reduce.max.xil']['throughput.list'].append(
                    np.prod(e['args']['shape']) * self.datatype_size_bytes / k['duration'] * 953.674316406)

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
            assert len(matched_kernels) <= 1
            for k in matched_kernels:
                dict_detailed['reduce.sum.r3a2.xil']['total.time.xil'] += k['duration']
                dict_detailed['reduce.sum.r3a2.xil']['throughput.list'].append(
                    np.prod(e['args']['shape']) * self.datatype_size_bytes / k['duration'] * 953.674316406)

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
            assert len(matched_kernels) <= 1
            for k in matched_kernels:
                dict_detailed['reduce.sum.r4a012.xil']['total.time.xil'] += k['duration']
                dict_detailed['reduce.sum.r4a012.xil']['throughput.list'].append(
                    np.prod(e['args']['shape']) * self.datatype_size_bytes / k['duration'] * 953.674316406)

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
                if k['name'] == 'task_pad_unpad' or k['name'] == 'task_conv2_1x1_direct':
                    # This is to make sure that only conv2 is allowed to have the same id as pad unpad
                    dict_detailed['padunpad.pad.xil']['total.time.xil'] += k['duration']
                    shape_out = copy.deepcopy(e['args']['shape'])
                    shape_out[-1] = e['args']['lastDimPadded']
                    dict_detailed['padunpad.pad.xil']['throughput.list'].append(
                        np.prod(shape_out) * self.datatype_size_bytes / k['duration'] * 953.674316406)

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
            assert len(matched_kernels) <= 1
            for k in matched_kernels:
                dict_detailed['padunpad.unpad.xil']['total.time.xil'] += k['duration']
                shape_out = copy.deepcopy(e['args']['shape'])
                shape_out[-1] = e['args']['lastDimUnpadded']
                dict_detailed['padunpad.unpad.xil']['throughput.list'].append(
                    np.prod(shape_out) * self.datatype_size_bytes / k['duration'] * 953.674316406)

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
                # assert len(matched_kernels) <= 1
                for k in matched_kernels:
                    dict_detailed[target_name]['total.time.xil'] += k['duration']
                    dict_detailed[target_name]['throughput.list'].append(
                        np.prod(e['args']['shape']) * self.datatype_size_bytes / k['duration'] * 953.674316406)
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
                # assert len(matched_kernels) <= 1
                for k in matched_kernels:
                    dict_detailed[target_name]['total.time.xil'] += k['duration']
                    dict_detailed[target_name]['throughput.list'].append(
                        np.prod(e['args']['shape1']) * self.datatype_size_bytes / k['duration'] * 953.674316406)
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
                # assert len(matched_kernels) <= 1
                for k in matched_kernels:
                    dict_detailed[target_name]['total.time.xil'] += k['duration']
                    shape_out = copy.deepcopy(e['args']['shape'])
                    shape_out.insert(e['args']['tileAxis'], e['args']['tileCount'])
                    print('shape out : ', shape_out)
                    dict_detailed[target_name]['throughput.list'].append(
                        np.prod(shape_out) * self.datatype_size_bytes / k['duration'] * 953.674316406)
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
                # assert len(matched_kernels) <= 1
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
                        np.prod(shape_out) * self.datatype_size_bytes / k['duration'] * 953.674316406)
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
                # assert len(matched_kernels) <= 1
                for k in matched_kernels:
                    dict_detailed[target_name]['total.time.xil'] += k['duration']
                    shape_out = copy.deepcopy(e['args']['shape1'])
                    shape_out[e['args']['concatAxis']] += e['args']['shape2'][e['args']['concatAxis']]
                    print('shape out : ', shape_out)
                    dict_detailed[target_name]['throughput.list'].append(
                        np.prod(shape_out) * self.datatype_size_bytes / k['duration'] * 953.674316406)
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
                # assert len(matched_kernels) <= 1
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
                # assert len(matched_kernels) <= 1
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

    def report_summary_perkernel(self):
        dict_detailed = {}
        for record in self.src_json['trace']:
            if record['type'] != 'kernel' or record['platform'] != 'xil':
                continue
            if not (record['name'] in dict_detailed.keys()):
                dict_detailed[record['name']] = {'num.calls': 0, 'total.xil': 0}
            dict_detailed[record['name']]['num.calls'] += 1
            dict_detailed[record['name']]['total.xil'] += record['duration']
        return dict_detailed

    def report_args_perkernel(self):
        dict_report = {}
        for element in self.src_json['trace']:
            if element['type'] == 'kernel' and element['platform'] == 'xil':
                parent_layer = self.find_layer_by_id(element['id'])
                if len(parent_layer) == 0:
                    if element['name'] != 'task_datamover':
                        print(
                            "report_per_kernel_args: Warning, failed to find the parent layer of a kernel that is not a datamover.")
                        assert False
                if not (element['name'] in dict_report.keys()):
                    dict_report[element['name']] = []

                dict_report[element['name']].append(copy.deepcopy(element))
                if element['name'] != 'task_datamover' and len(parent_layer) != 0:
                    dict_report[element['name']][-1]['args'] = parent_layer[0]['args']
                    dict_report[element['name']][-1]['name.parent'] = parent_layer[0]['name']
        return dict_report

    def get_the_output_tensor_sizes(self, list_kernel_args):
        """
        Takes a list of kernel args (only for 1 kernel) and sorts out and calculates the output tensor sizes in bytes.
        Note that pad/unpad kernel is:
            - used independently
            - used inside conv2d
        , meaning that to get results for pad/unpad, the user should run this method twice for args of padunpad and conv2d,
        and join the two resulted lists.

        :param list_kernel_args: one element of type list of output of report_args_perkernel()
        :return:
        """
        def get_bytes_of_shape(shape):
            return int(np.prod(shape) * self.datatype_size_bytes)
        dict_shapes_out = {}
        for item in list_kernel_args:
            task_name = item['name']
            if not (task_name in dict_shapes_out.keys()):
                dict_shapes_out[task_name] = {}
                dict_shapes_out[task_name]['all'] = []

            if task_name == 'task_transpose':
                dict_shapes_out['task_transpose']['all'].append(get_bytes_of_shape(item['args']['shape']))
            if task_name == 'task_matmul':
                rank = len(item['args']['shape1'])
                if rank == 2:
                    output_shape = [item['args']['shape1'][0], item['args']['shape2'][1]]
                else:
                    if rank == 3:
                        output_shape = [item['args']['shape1'][0], item['args']['shape1'][1], item['args']['shape2'][2]]
                    else:
                        assert False
                dict_shapes_out['task_matmul']['all'].append(get_bytes_of_shape(output_shape))
            if task_name == 'task_basicops':
                dict_shapes_out['task_basicops']['all'].append(get_bytes_of_shape(item['args']['shape1']))
            if task_name == 'task_tile':
                dict_shapes_out['task_tile']['all'].append(get_bytes_of_shape(item['args']['shape']) * item['args']['tileCount'])
            if task_name == 'task_topk':
                dict_shapes_out['task_topk']['all'].append(get_bytes_of_shape(item['args']['shape'][0:2]) * item['args']['k'])
            if task_name == 'task_gather':
                _k = 0
                # if the profiler.json belongs to a commit that the indicesTn's shape does not get recorded
                if 'shape.indices' in item['args'].keys():
                    _k = item['args']['shape.indices'][-1]
                else:
                    _k = 20
                dict_shapes_out['task_gather']['all'].append(get_bytes_of_shape(item['args']['shape']) * _k)
            if task_name == 'task_concat':
                output_shape = item['args']['shape1']
                output_shape[item['args']['concatAxis']] += item['args']['shape2'][item['args']['concatAxis']]
                dict_shapes_out['task_concat']['all'].append(get_bytes_of_shape(output_shape))
            if task_name == 'task_conv2_1x1_direct' and len(item['args']) == 3:
                # make sure that it's conv2d and not padunpad
                output_shape = item['args']['shape.i']
                output_shape[-1] = item['args']['shape.w'][-1]
                dict_shapes_out['task_conv2_1x1_direct']['all'].append(get_bytes_of_shape(output_shape))
            if task_name == 'task_relu_sqrt_square':
                if item['name.parent'] == 'ReLU':
                    if not('relu' in dict_shapes_out['task_relu_sqrt_square'].keys()):
                        dict_shapes_out['task_relu_sqrt_square']['relu'] = []
                    dict_shapes_out['task_relu_sqrt_square']['relu'].append(get_bytes_of_shape(item['args']['shape']))
                if item['name.parent'] == 'Sqrt':
                    if not('sqrt' in dict_shapes_out['task_relu_sqrt_square'].keys()):
                        dict_shapes_out['task_relu_sqrt_square']['sqrt'] = []
                    dict_shapes_out['task_relu_sqrt_square']['sqrt'].append(get_bytes_of_shape(item['args']['shape']))
                if item['name.parent'] == 'Square':
                    if not('square' in dict_shapes_out['task_relu_sqrt_square'].keys()):
                        dict_shapes_out['task_relu_sqrt_square']['square'] = []
                    dict_shapes_out['task_relu_sqrt_square']['square'].append(get_bytes_of_shape(item['args']['shape']))
            if task_name == 'task_reduce':
                if item['args']['reduction_op'] == 1:
                    # the only reduce max sub kernel
                    if not('max' in dict_shapes_out['task_reduce'].keys()):
                        dict_shapes_out['task_reduce']['max'] = []
                    dict_shapes_out['task_reduce']['max'].append(get_bytes_of_shape(item['args']['shape']))
                if item['args']['reduction_op'] == 0:
                    if item['args']['rank'] == 4 and item['args']['combination'] == [1, 1, 1, 0]:
                        # sum_r4a012
                        if not ('sum.r4a012' in dict_shapes_out['task_reduce'].keys()):
                            dict_shapes_out['task_reduce']['sum.r4a012'] = []
                        dict_shapes_out['task_reduce']['sum.r4a012'].append(get_bytes_of_shape(item['args']['shape']))
                    else:
                        if item['args']['rank'] == 3 and item['args']['combination'] == [0, 0, 1]:
                            # sum_r3a2
                            if not ('sum.r3a2' in dict_shapes_out['task_reduce'].keys()):
                                dict_shapes_out['task_reduce']['sum.r3a2'] = []
                            dict_shapes_out['task_reduce']['sum.r3a2'].append(get_bytes_of_shape(item['args']['shape']))
                        else:
                            assert False
            # ----------------------------------------------------------------------------------------------------------
            # pad unpad   or   pad/unpad of conv2ds
            if task_name == 'task_pad_unpad' or (task_name == 'task_conv2_1x1_direct' and len(item['args']) == 2):
                if not ('task_pad_unpad' in dict_shapes_out.keys()):
                    dict_shapes_out['task_pad_unpad'] = {}
                    dict_shapes_out['task_pad_unpad']['all'] = []
                if 'lastDimPadded' in item['args'].keys():
                    # pad
                    if not ('pad' in dict_shapes_out['task_pad_unpad'].keys()):
                        dict_shapes_out['task_pad_unpad']['pad'] = []
                    output_shape = item['args']['shape']
                    output_shape[-1] = item['args']['lastDimPadded']
                    dict_shapes_out['task_pad_unpad']['pad'].append(get_bytes_of_shape(output_shape))
                if 'lastDimUnpadded' in item['args'].keys():
                    # unpad
                    if not ('unpad' in dict_shapes_out['task_pad_unpad'].keys()):
                        dict_shapes_out['task_pad_unpad']['unpad'] = []
                    output_shape = item['args']['shape']
                    output_shape[-1] = item['args']['lastDimUnpadded']
                    dict_shapes_out['task_pad_unpad']['unpad'].append(get_bytes_of_shape(output_shape))
            # ----------------------------------------------------------------------------------------------------------
            if task_name == 'task_datamover':
                dict_shapes_out['task_datamover']['all'].append(item['bytes'])

        return dict_shapes_out

    def report_outputsizes_perkernel(self):
        """
        reports the output tensor sizes for each kernel.
        the report contains detailed sub-kernel stats for [relusqrtsuqare, reduce, and padunpad]
        :return: a dict of format: {<task_name>:{'all':[<for the kernels with no sub kernels>], '<subkernel>':[], ...}, ...}
        """
        dict_report = {}
        dict_allkernels_args = self.report_args_perkernel()
        for kernel_name in dict_allkernels_args.keys():
            results = self.get_the_output_tensor_sizes(dict_allkernels_args[kernel_name])
            for k in results.keys():
                if not(k in dict_report.keys()):
                    dict_report[k] = {}
                for q in results[k].keys():
                    if not (q in dict_report[k].keys()):
                        dict_report[k][q] = []
                    for i in results[k][q]:
                        dict_report[k][q].append(int(i))

        return dict_report

    def report_outputsizesboxplots_perkernel(self):
        def get_box_plot_five_nums(data):
            if len(data) == 0:
                return {}
            # sorted_ascend = np.sort(data, axis=0)
            r_median = np.median(data)
            r_q1 = np.quantile(data, 0.25)
            r_q2 = np.quantile(data, 0.75)
            r_min = np.min(data)
            r_max = np.max(data)
            return {'median': int(r_median), 'q1': int(r_q1), 'q3': int(r_q2), 'min': int(r_min), 'max': int(r_max)}
        dict_report = {}
        dict_outputsizes = self.report_outputsizes_perkernel()
        for kernel_name in dict_outputsizes.keys():
            if not(kernel_name in dict_report.keys()):
                dict_report[kernel_name] = {}
            for q in dict_outputsizes[kernel_name]:
                dict_report[kernel_name][q] = get_box_plot_five_nums(dict_outputsizes[kernel_name][q])
        return dict_report


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

        self.report.append(self.obj.report_summary_perkernel())
        self.print_report(self.report[-1], self.new_dump_dir + "/17PerKernelSummary.json")

        self.report.append(self.obj.report_args_perkernel())
        self.print_report(self.report[-1], self.new_dump_dir + "/18PerKernelArgs.json")

        self.report.append(self.obj.report_outputsizes_perkernel())
        self.print_report(self.report[-1], self.new_dump_dir + "/19PerKernelOutputSizes.json")

        self.report.append(self.obj.report_outputsizesboxplots_perkernel())
        self.print_report(self.report[-1], self.new_dump_dir + "/20PerKernelOutputSizeBoxPlots.json")

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
