import numpy as np


class SlrStats:
    def __init__(self, slr_index, util_bram, util_dsp, util_ff, util_lut, util_uram):
        self.slr_index = slr_index
        self.util_bram = util_bram
        self.util_dsp = util_dsp
        self.util_ff = util_ff
        self.util_lut = util_lut
        self.util_uram = util_uram

    def __add__(self, other):
        if not(self.slr_index == -1 and other.slr_index == -1):
            assert self.slr_index == other.slr_index
        return SlrStats(
            self.slr_index,
            self.util_bram + other.util_bram,
            self.util_dsp  + other.util_dsp ,
            self.util_ff   + other.util_ff  ,
            self.util_lut  + other.util_lut ,
            self.util_uram + other.util_uram
        )

    def __str__(self):
        return ''.join([
            "BRAM=",str(self.util_bram),'%, ',
            "DSP=",str(self.util_dsp),'%, ',
            "FF=",str(self.util_ff),'%, ',
            "LUT=",str(self.util_lut),'%, ',
            "URAM=",str(self.util_uram),'%, ',
        ])

    def __lt__(self, other):
        return self.util_bram < other.util_bram and \
               self.util_dsp < other.util_dsp and \
               self.util_ff < other.util_ff and \
               self.util_lut < other.util_lut and \
               self.util_uram < other.util_uram


class KernelObj:
    def __init__(self, kernel_name, possible_banks, util_bram, util_dsp, util_ff, util_lut, util_uram, assigned_bank=-1):
        self.kernel_name = kernel_name
        self.possible_banks = possible_banks
        self.assigned_bank = assigned_bank
        self.util_stats = SlrStats(-1, util_bram, util_dsp, util_ff, util_lut, util_uram)

    def get_slr(self):
        if self.assigned_bank == 1:
            return 2
        if self.assigned_bank == 2:
            return 1
        assert False

    def clone(self, assigned_bank=-1):
        return KernelObj(
            self.kernel_name,
            self.possible_banks,
            self.util_stats.util_bram,
            self.util_stats.util_dsp,
            self.util_stats.util_ff,
            self.util_stats.util_lut,
            self.util_stats.util_uram,
            assigned_bank)

