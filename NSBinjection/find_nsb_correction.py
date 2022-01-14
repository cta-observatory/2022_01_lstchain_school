from lstchain.io import standard_config
from lstchain.io.config import read_configuration_file
from lstchain.image.modifier import calculate_required_additional_nsb

input_filename = "../data/mc/DL0/proton_20deg_180deg_run1___cta-prod5-lapalma_4LSTs_MAGIC_desert-2158m_mono.simtel.gz"
target_data = "../data/DL1ab/dl1_LST-1.Run02977.0122.h5"

nsb_tuning_ratio = calculate_required_additional_nsb(input_filename,
                                                     target_data,
                                                     config=standard_config)

print("correction ratio, data_ped_variance, mc_ped_variance = ", nsb_tuning_ratio)
