

def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r', encoding='UTF-8')
    print(file)
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]   # Remove all blank lines and lines beginning with #
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        # Read in the name of the current module
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0  # Pre-stored to prevent some layers from not having this parameter (only not before 3 yolo layers)
        # Read in the parameters of the current module
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()   # Remove the blanks on the left and right sides of "=""

    return module_defs


def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        # print(key,value)
        options[key.strip()] = value.strip()
    print(options)
    return options


if __name__=='__main__':
    parse_data_config('D:\Python\week5\PyTorch-YOLOv3\config\coco.data')