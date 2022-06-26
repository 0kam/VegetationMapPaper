from scripts.utils.utils import set_patches
from glob import glob
from pathlib import Path

set_patches("data_source/labels", "data_source/aligned/multi_days", "data/multidays_5x5/", (5,5))
set_patches("data_source/labels", "data_source/aligned/multi_days", "data/multidays_3x3/", (3,3))
set_patches("data_source/labels", "data_source/aligned/multi_days", "data/multidays_1x1/", (1,1))

for p in glob("data_source/aligned/single_day/*"):
    dname = Path(p).stem
    set_patches("data_source/labels", p, "data/single_day/{}/5x5/".format(dname), (5,5))
    set_patches("data_source/labels", p, "data/single_day/{}/3x3/".format(dname), (3,3))
    set_patches("data_source/labels", p, "data/single_day/{}/1x1/".format(dname), (1,1))