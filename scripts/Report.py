import json
import copy
import os
import sys
from datetime import datetime
import numpy as np


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
        matched = []
        for item in self.src_json['trace']:
            if item['type'] != "kernel":
                continue
            else:
                if item["id"] == parent_layer_id:
                    matched.append(item)
        return matched

    def recursive_find_layers_by(self, out_dict_results, tree, platform='xil'):
        if tree['type'] == 'kernel':
            return
        if len(tree['nested']) == 0:
            if tree['type'] == 'layer' and tree['platform'] == platform:
                if not (tree['name'] in out_dict_results.keys()):
                    out_dict_results[tree['name']] = []
                out_dict_results[tree['name']].append(tree)
        else:
            for child in tree['nested']:
                self.recursive_find_layers_by(out_dict_results, child, platform)
            if tree['type'] == 'layer' and tree['platform'] == platform:
                if not (tree['name'] in out_dict_results.keys()):
                    out_dict_results[tree['name']] = []
                out_dict_results[tree['name']].append(tree)

    def report_per_layer_args(self, platform):
        # 1) finds recursively all the layers with unique 'name'
        # 2) gathers recursively all of the layers within the list assigned to their related dict.keys()
        # 3) returns the dict

        dict_layers_src = {}
        for base_element in self.src_json['trace']:
            self.recursive_find_layers_by(dict_layers_src, base_element, platform)
        dict_layers = copy.deepcopy(dict_layers_src)
        for layer_name in dict_layers.keys():
            # for base_element in self.src_json['trace']:
            #    self.recursive_gather_layers_by(dict_layers, base_element, layer_name, platform)
            for element in dict_layers[layer_name]:
                del element['time.stop']
                del element['time.start']
                del element['name']
                del element['type']
                del element['platform']
                del element['nested']
        return dict_layers

    def recursive_per_top_parent_layer_device_time(self, tree):
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
                r_duration, r_cpu = self.recursive_per_top_parent_layer_device_time(child)
                duration_total += r_duration
                for e in r_cpu:
                    cpu_usage_list.append(e)
            return duration_total, cpu_usage_list

    def report_per_layer_total_device_time(self):
        dict_total = {}
        for top_parent in self.src_json['trace']:
            total_device_time, all_cpu_usage_samples = self.recursive_per_top_parent_layer_device_time(top_parent)
            if top_parent['name'] in dict_total.keys():
                dict_total[top_parent['name']]['total.device.time'] += total_device_time
                for e in all_cpu_usage_samples:
                    dict_total[top_parent['name']]['cpu.usage.list'].append(e)
            else:
                dict_total[top_parent['name']] = {'total.device.time': total_device_time, 'cpu.usage.mean': 0, 'cpu.usage.min': 0, 'cpu.usage.max': 0, 'cpu.usage.list': all_cpu_usage_samples}

        for key in dict_total.keys():
            if len(dict_total[key]['cpu.usage.list'])!=0:
                dict_total[key]['cpu.usage.mean'] = np.mean(dict_total[key]['cpu.usage.list'])
                dict_total[key]['cpu.usage.min'] = np.min(dict_total[key]['cpu.usage.list'])
                dict_total[key]['cpu.usage.max'] = np.max(dict_total[key]['cpu.usage.list'])
            del dict_total[key]['cpu.usage.list']
        return dict_total

    def dict_add_integer_to_key_try(self, _dict, key, val):
        if key in _dict.keys():
            _dict[key] += val
        else:
            _dict[key] = val

    def recursive_per_layer_host_time(self, out_dict, tree, platform):
        if tree['type'] == 'kernel':
            return
        if len(tree['nested']) == 0:
            duration = tree['time.stop'] - tree['time.start']
            if tree['platform'] == 'cpu':
                duration = duration * 1000  # convert it to nano seconds
            if tree['platform'] == platform:
                self.dict_add_integer_to_key_try(out_dict, tree['name'], duration)
                # out_dict[tree['name']] = duration
            return tree['name'], duration, tree['platform']
        else:
            duration_raw = tree['time.stop'] - tree['time.start']
            if tree['platform'] == 'cpu':
                duration_raw = duration_raw * 1000  # convert it to nano seconds
            duration = duration_raw
            for child in tree['nested']:
                child_name, child_duration, child_platform = self.recursive_per_layer_host_time(out_dict, child,
                                                                                                platform=platform)
                if child_platform == platform:
                    if tree['name'] != child_name:
                        duration -= child_duration
                    else:
                        duration -= 0
            if tree['platform'] == platform:
                self.dict_add_integer_to_key_try(out_dict, tree['name'], duration)
                # out_dict[tree['name']] = duration
            return tree['name'], duration_raw, tree['platform']

    def report_per_layer_total_host_time(self, platform):
        dict_report = {}
        for element in self.src_json['trace']:
            self.recursive_per_layer_host_time(dict_report, element, platform)
        return dict_report

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

    def report_per_kernel_total_device_time(self):
        dict_report = {}
        for element in self.src_json['trace']:
            if element['type'] == 'kernel' and element['platform'] == 'xil':
                # parent_layer = self.find_layer_by_id(element['id'])
                if element['name'] in dict_report.keys():
                    dict_report[element['name']]['time'] += element['duration']
                    dict_report[element['name']]['launches'] += 1
                else:
                    dict_report[element['name']] = {'time':element['duration'], 'launches':1}
        return dict_report

    def report_per_kernel_args(self):
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
        return dict_report

    def report_info(self):
        return self.src_json['info']


class CReporter:
    def __init__(self, path_json):
        self.obj = CProfiler(path_json)
        self.report = []
        self.today = datetime.now()

        self.new_dump_dir = self.today.strftime('%Y%m%d%H%M%S')
        os.mkdir(self.new_dump_dir)

        self.report.append(self.obj.report_info())
        self.print_report(self.report[-1], self.new_dump_dir + "/0Info.json")

        self.report.append(self.obj.report_per_layer_args(platform='xil'))
        self.print_report(self.report[-1], self.new_dump_dir + "/1PerLayerArgsXil.json")

        self.report.append(self.obj.report_per_layer_args(platform='cpu'))
        self.print_report(self.report[-1], self.new_dump_dir + "/1PerLayerArgsCpu.json")

        self.report.append(self.obj.report_per_layer_total_device_time())
        self.print_report(self.report[-1], self.new_dump_dir + "/2PerLayerTotalDeviceTime.json")

        self.report.append(self.obj.report_per_layer_total_host_time(platform='cpu'))
        self.print_report(self.report[-1], self.new_dump_dir + "/3PerLayerTotalHostTimeCpu.json")

        self.report.append(self.obj.report_per_layer_total_host_time(platform='xil'))
        self.print_report(self.report[-1], self.new_dump_dir + "/3PerLayerTotalHostTimeXil.json")

        self.report.append(self.obj.report_per_kernel_total_device_time())
        self.print_report(self.report[-1], self.new_dump_dir + "/4PerKernelTotalDeviceTime.json")

        self.report.append(self.obj.report_per_kernel_args())
        self.print_report(self.report[-1], self.new_dump_dir + "/5PerKernelArgs.json")

    def print_report(self, report, dump_fname):
        print(
            json.dumps(report, sort_keys=True, indent=4),
            file=open(dump_fname, "w")
        )


if __name__ == "__main__":
    assert len(sys.argv) == 2 or len(sys.argv) == 1
    if len(sys.argv) == 2:
        if not os.path.isfile(sys.argv[1]):
            print("File does not exist: ", sys.argv[1])
            sys.exit(status=1)
    if len(sys.argv) == 1:
        print("No args, looking for profiler.json at the current directory...")
        if not os.path.isfile("profiler.json"):
            print("File does not exist: ", sys.argv[1])
            sys.exit(status=1)
    print("Analyzing the JSON file...")
    obj = CReporter(sys.argv[1] if len(sys.argv) == 2 else "profiler.json")
    print("Done. Closing.")

