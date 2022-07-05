from dataPreparation.crc.popDataPrep import popData_crc
from dataPreparation.crc.ancDataPrep import ancData_crc
from dataPreparation.rom.ancDataPrep import ancData_rom
years = [2015] #1990, 2000,

# Select city
crc = 'no'
rom = 'yes'

# Select Process
process_Population_Data = 'no'
process_Ancillary_Data = 'yes'

if rom == 'yes':
    if process_Ancillary_Data == 'yes':
        # For ESM use 2015
        init_esm = 'yes'
        #For GHS 2000, 2015
        init_ghs = 'no'
        process_ghs = 'no'
        
        for year in years:
            ancData_rom(init_esm, init_ghs, process_ghs, year)

if crc == 'yes':
    if process_Population_Data == 'yes':
        year_list = [2000]
        for year in year_list:
            popData_crc(year)
        
    if process_Ancillary_Data == 'yes':
        # For ESM use 2015
        init_esm = 'no'
        #For GHS 2000, 2015
        init_ghs = 'yes'
        process_ghs = 'yes'
        
        for year in years:
            ancData_crc(init_esm, init_ghs, process_ghs, year)