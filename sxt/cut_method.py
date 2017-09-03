import logging
import numpy as np
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(message)s")


class _CutMethods(object):

    def __init__(self):
        super(_CutMethods, self).__init__()

    @staticmethod
    def check_separate(series_, kwargs):
        separate, state = None, None
        if kwargs["keep_separate_value"] is not None:
            separate = kwargs["keep_separate_value"]
            if separate == series_.min():
                state = 0.01
            elif separate == series_.max():
                state = -0.01
            else:
                state, separate = None, None
        return separate, state

    @staticmethod
    def quantile_cut_flow(series_, part_, max_cut_part):
        for i in range(2, max_cut_part + 1):
            part = series_.quantile(np.linspace(
                0, 1, i + 1)).unique().tolist()
            part.sort()
            if part not in part_:
                part_.append(part)
        return part_, series_

    @staticmethod
    def separate_cut_flow(series_, separate, state, part_, max_cut_part):
        index = series_[series_ == separate].index
        series_.ix[index[0]] = separate + state
        for i in range(1, max_cut_part):
            part = series_[series_ != separate].quantile(
                np.linspace(0, 1, i + 1)).unique().tolist()
            part.append(separate)
            part.sort()
            if part not in part_:
                part_.append(part)
        return part_, series_

    @staticmethod
    def cut_method_flow(series_, kwargs):
        part_ = []
        separate, state = _CutMethods.check_separate(series_, kwargs)
        if separate is None:
            if kwargs["cut_method"] == "quantile":
                part_, series_ = _CutMethods.quantile_cut_flow(
                    series_, part_, kwargs["max_cut_part"])
            else:
                logging.warning("Only support 'quantile' !")
                part_, series_ = _CutMethods.quantile_cut_flow(
                    series_, part_, kwargs["max_cut_part"])
        else:
            part_, series_ = _CutMethods.separate_cut_flow(
                series_, separate, state, part_, kwargs["max_cut_part"])
        return part_, series_
