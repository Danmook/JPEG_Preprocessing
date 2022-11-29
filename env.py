import os
import shutil


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        # shutil.copyfile(file1,file2)
        # file1为需要复制的源文件的文件路径,file2为目标文件的文件路径+文件名.
        shutil.copyfile(config, os.path.join(path, config_name))
        # 将config.json复制到./checkpoint/image_compressor/config.json
