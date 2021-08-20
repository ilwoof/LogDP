
import numpy as np
import os


DATA_NAME = 'HDFS'  # 'bgl', 'spirit'
WINDOW = 'session'
MODE = 'semi-supervised'
Split_Group_Nums = 1
FEATURE_TYPE = 'raw'  # 'raw', 'wn'
# generate script for CFS (learning MBs)
# HDFS: 17; bgl-1hour:129; bgl-100:285; bgl-20:569;
# spirit-1hour:280; spirit-100:1041; spirit-20:1107
# tdb-1hour:1831; tdb-100:3621; tdb-20:3687
dimension = 17
# for mb learning
MB_METHOD = 'iamb'  # 'iamb' 'fbed'
ALPHA = 0.05  # 0.05 0.01 0.005
N_CON = -1

MB_PATH = f'../../data/{DATA_NAME}_{WINDOW}/{FEATURE_TYPE}'
TRN_PATH = MB_PATH
SCRIPT_PATH = f'./{DATA_NAME}-script-{WINDOW}'
CFS_PATH = '../CCD'
CFS_LOCAL_PATH = './CCD'

os.makedirs(SCRIPT_PATH, exist_ok=True)

if dimension % Split_Group_Nums == 0:
    iter_dimension = int(dimension/Split_Group_Nums)
else:
    iter_dimension = int(dimension/Split_Group_Nums)+1

fix_iter = iter_dimension

# cfs.sh
f_cfs = open(f'{SCRIPT_PATH}/cfs.sh', 'w')
f_cfs.write('#!/bin/bash\n\n')
f_cfs.write('dos2unix *.sh\n')
f_cfs.write('chmod +777 *.sh\n')

# script for each label
for label in np.arange(Split_Group_Nums):
    data_name = f'{TRN_PATH}/{DATA_NAME}_{WINDOW}_training_data.txt'
    # if there is just one group, we don't add label into the splitting mb file
    if Split_Group_Nums == 1:
        script_name = f'{DATA_NAME}_cfs_script.sh'
        ccd_dest = CFS_LOCAL_PATH
        mb_name = f'{MB_PATH}/{DATA_NAME}_{WINDOW}_training_data_mb.txt'
        tm_name = f'{MB_PATH}/{DATA_NAME}_{WINDOW}_training_data_tm.txt'
    else:
        script_name = f'{DATA_NAME}_cfs_{label}_script.sh'
        ccd_dest = CFS_LOCAL_PATH + str(label)
        mb_name = f'{MB_PATH}/{DATA_NAME}_{WINDOW}_training_data_{label}_mb.txt'
        tm_name = f'{MB_PATH}/{DATA_NAME}_{WINDOW}_training_data_{label}_tm.txt'

    script_name_path = f'{SCRIPT_PATH}/{script_name}'
    f_cfs.write('./'+script_name+' &\n')  # & means running in the background

    f = open(script_name_path, 'a')

    # copy cfs to a new folder to learning mbs parallel on all labels
    f.write("#!/bin/bash\n\n")
    f.write('cp -R '+CFS_PATH+' '+ccd_dest+' \n')
    f.write('cd '+ccd_dest+' \n')
    f.write('rm mb/mb.out\n')
    f.write('rm indicator1/indicator.out\n')

    if label + 1 == Split_Group_Nums:
        iter_dimension = dimension - iter_dimension * label
    for i_foc in range(iter_dimension):
        # i_foc = 0
        f.write('echo start focused variable {}/{} in label:{}----------\n'.format(i_foc+1, iter_dimension, label))
        f.write(f'./main {data_name} "" {ALPHA} {MB_METHOD.upper()} {i_foc+label*fix_iter} "" {N_CON}\n')
        f.write('sleep 3s\n')
    f.write(f'mv mb/mb.out {mb_name}\n')
    f.write(f'mv indicator1/indicator.out {tm_name}\n')

    f.write('cd ..\n')
    f.write(f'rm -rf {ccd_dest} \n')
    f.write(f'echo label {label} done.')
    f.close()

f_cfs.close()
print("generation CFS scripts done.")
