#########################################################################
#                                                                       #
# Python script for fire emissions preprocessing from RAVE FRP and FRE  #
# (Li et al.,2022). Written by Johana Romero-Alvarez and Haiqin Li      #
# based on Kai Wang and Jianping Huang prototype                        #
#                                                                       #
#########################################################################

import sys
import xarray as xr
import datetime as dt
from datetime import date, time,timedelta
import pandas as pd
import numpy as np
import ESMF
from netCDF4 import Dataset
import os
import time

#import fix files
staticdir = sys.argv[1]
ravedir = sys.argv[2]
newges_dir = sys.argv[3]
predef_grid = sys.argv[4]

#constants emissions estimation
beta= 0.38 # based on Wooster et al. 2005

#units conversion
to_s=3.6e3
fkg_to_ug=1e9
fg_to_ug=1e6

#Emission factors based on SERA US Forest Service
EF_FLM = dict({'frst':19,'hwd':9.4,'mxd':14.6,'shrb':9.3,'shrb_grs':10.7,'grs':13.3})
EF_SML = dict({'frst':28,'hwd':37.7,'mxd':17.6,'shrb':36.6,'shrb_grs':36.7,'grs':38.4})

#Land categories fractional map
land_cats=["Evergreen_Nleaf_Frst", "Evergreen_Bleaf_Frst", "Deciduous_NleafFrst","Deciduous_Bleaf_Frst","Mixed_Frst","Closed_Shrublands","Open_Shrublands","Woody_Savannas","Savannas","Grasslands","Permanent_Wetlands","Croplands","Urban_Lands","Cropland_NVeg"]

#list of variables to interpolate
vars_emis = ["FRP_MEAN","FRP_SD","FRE","PM2.5"]

#pass env vars from  workflow
current_day = os.environ.get("CDATE")
nwges_dir = os.environ.get("NWGES_DIR")

#Fixed files directories
normal_template_file = staticdir+'/pypost_conus_basic_template.grib2'
veg_map = '/scratch1/BMC/acomp/Johana/input_files/frac_LU/LU_frac_NA.nc'  #staticdir+'/veg_map.nc'
grid_in=  staticdir+'/grid_in.nc'
weightfile= staticdir+'/weight_file.nc'
grid_out  = staticdir+'/ds_out_base.nc'
dummy_hr_rave= staticdir+'/dummy_hr_rave.nc'

#RAVE data locations
RAVE='/scratch2/NAGAPE/epic/David.Huber/rrfs-sd/RAW_RAVE' #ravedir
intp_dir=newges_dir
rave_to_intp= predef_grid+"_intp_"

#Input weighting file for interpolation
filename =weightfile

#Set predefined grid
if predef_grid=='RRFS_NA_3km':
   cols,rows=2700,3950
else:
   cols,rows=1092,1820
print('PREDEF GRID',predef_grid,'cols,rows',cols,rows)

#Functions
#Create date range
def date_range(current_day):
   print('Searching for interpolated RAVE for',current_day)
   fcst_YYYYMMDDHH=dt.datetime.strptime(current_day, "%Y%m%d%H")
   previous_day=fcst_YYYYMMDDHH - timedelta(days = 1)
   date_list=pd.date_range(previous_day,periods=24,freq="H")
   fcst_dates=date_list.strftime("%Y%m%d%H")
   rave_to_intp= predef_grid+"_intp_"
   print('Current day', fcst_YYYYMMDDHH,'Persistance',previous_day)
   return fcst_dates

# Check if interoplated RAVE is available for the previous 24 hours.
def check_for_intp_rave(intp_dir,fcst_dates,rave_to_intp):
   os.chdir(intp_dir)
   # Listing of the working directory
   sorted_obj = sorted(os.listdir(intp_dir))
   # Lists of available and non-available hours
   intp_avail_hours=[]
   intp_non_avail_hours=[]
   # Loop through the list of forecast dates and determine if they have valid,
   # interpolated RAVE data
   # There are four situations here.
   #   1) the file is missing (interpolate a new file)
   #   2) the file is present (use it)
   #   3) there is a link, but it's broken (interpolate a new file)
   #   4) there is a valid link (use it)
   for date in fcst_dates:
      intp_name = rave_to_intp+date+'00_'+date+'00.nc'
      file_exists = intp_name in sorted_obj
      is_link = os.path.islink(intp_name)
      is_valid_link = is_link and os.path.exists(intp_name)
      if (file_exists and not is_link) or is_valid_link:
         print('RAVE interpolated available for',rave_to_intp+date+'00_'+date+'00.nc')
         intp_avail_hours.append(date)
      else:
         print('Create interpolated RAVE for',rave_to_intp+date+'00_'+date+'00.nc')
         intp_non_avail_hours.append(date)
   print('Avail_intp_hours',intp_avail_hours,'Non_avail_intp_hours',intp_non_avail_hours)
   return intp_avail_hours,intp_non_avail_hours

#Check if raw RAVE in intp_non_avail_hours is available to interpolate
def check_for_raw_rave(RAVE,intp_non_avail_hours):
   os.chdir(RAVE)
   raw_rave="Hourly_Emissions_3km_"
   updated_rave="RAVE-HrlyEmiss-3km_v1r0_blend_s"
   sorted_obj = sorted(os.listdir(RAVE))
   rave_avail=[]
   rave_avail_hours=[]
   rave_nonavail_hours_test=[]
   for date in intp_non_avail_hours:
      if raw_rave+date+'00_'+date+'00.nc' in sorted_obj:
         print('Raw RAVE available for interpolation',raw_rave+date+'00_'+date+'00.nc')
         rave_avail.append(raw_rave+date+'00_'+date+'00.nc')
         rave_avail_hours.append(date)
      else:
         print('Raw RAVE non_available for interpolation',raw_rave+date+'00_'+date+'00.nc')
         rave_nonavail_hours_test.append(date)
   print("Raw RAVE available",rave_avail_hours, "rave_nonavail_hours_test",rave_nonavail_hours_test)
   return rave_avail,rave_avail_hours,rave_nonavail_hours_test

#Create source and target fields
def creates_st_fields(grid_in,grid_out):
   os.chdir(intp_dir)
   #source RAW emission grid file
   ds_in=xr.open_dataset(grid_in)
   #target (3-km) grid file
   ds_out = xr.open_dataset(grid_out)
   #source center lat/lon
   src_latt = ds_in['grid_latt']
   #target center lat/lon
   tgt_latt = ds_out['grid_latt']
   tgt_lont = ds_out['grid_lont']
   #grid shapes
   src_shape = src_latt.shape
   tgt_shape = tgt_latt.shape
   #build the ESMF grid coordinates
   srcgrid = ESMF.Grid(np.array(src_shape), staggerloc=[ESMF.StaggerLoc.CENTER, ESMF.StaggerLoc.CORNER],coord_sys=ESMF.CoordSys.SPH_DEG)
   tgtgrid = ESMF.Grid(np.array(tgt_shape), staggerloc=[ESMF.StaggerLoc.CENTER, ESMF.StaggerLoc.CORNER],coord_sys=ESMF.CoordSys.SPH_DEG)
   #read in the pre-generated weight file
   tgt_area=ds_out['area']
   #dummy source and target fields
   srcfield = ESMF.Field(srcgrid, name='test',staggerloc=ESMF.StaggerLoc.CENTER)
   tgtfield = ESMF.Field(tgtgrid, name='test',staggerloc=ESMF.StaggerLoc.CENTER)
   print('Grid in and out files available. Generating target and source fields')
   return srcfield,tgtfield,tgt_latt,tgt_lont,srcgrid,tgtgrid,src_latt,tgt_area

#Define output and variable meta data
def create_emiss_file(fout,rave_file):
    fout.createDimension('t',None)
    fout.createDimension('lat',cols)
    fout.createDimension('lon',rows)
    setattr(fout,'PRODUCT_ALGORITHM_VERSION','Beta')
    setattr(fout,'TIME_RANGE','1 hour')
    setattr(fout,'RangeBeginningDate)',rave_file[21:25]+'-'+rave_file[25:27]+'-'+rave_file[27:29])
    setattr(fout,'RangeBeginningTime\(UTC-hour\)',rave_file[29:31])
def Store_latlon_by_Level(fout,varname,var,long_name,units,dim,fval,sfactor):
    if dim=='2D':
       var_out = fout.createVariable(varname,   'f4', ('lat','lon'))
       var_out.units=units
       var_out.long_name=long_name
       var_out.standard_name=varname
       fout.variables[varname][:]=var
       var_out.FillValue=fval
       var_out.coordinates='geolat geolon'
def Store_by_Level(fout,varname,long_name,units,dim,fval,sfactor):
    if dim=='3D':
       var_out = fout.createVariable(varname,   'f4', ('t','lat','lon'))
       var_out.units=units
       var_out.long_name = long_name
       var_out.standard_name=long_name
       var_out.FillValue=fval
       var_out.coordinates='t geolat geolon'

#Open LU map and extract land categories
def generate_EFs(veg_map,EF_FLM,EF_SML,land_cats):
   LU_map=(veg_map)
   nc_land= xr.open_dataset(LU_map)
   vtype_val= nc_land['vegetation_type_pct'][:,:,:]
   #Processing EF
   arr_parent_EFs=[]
   for lc,i in zip(land_cats,range(len(land_cats))):
      vtype_ind=vtype_val[i,:,:]
      if lc == "Evergreen_Nleaf_Frst":
         Nleaf_EF=vtype_ind*((0.75*EF_FLM['frst'])+(0.25*EF_SML['frst']))
         arr_parent_EFs.append(Nleaf_EF.values.flatten())
      elif lc ==  "Evergreen_Bleaf_Frst":
         Bleaf_EF=vtype_ind*((0.75*EF_FLM['frst'])+(0.25*EF_SML['frst']))
         arr_parent_EFs.append(Bleaf_EF.values.flatten())
      elif lc == "Deciduous_NleafFrst":
         Dec_Nleaf_EF=vtype_ind*((0.80*EF_FLM['hwd'])+(0.20*EF_SML['hwd']))
         arr_parent_EFs.append(Dec_Nleaf_EF.values.flatten())
      elif lc == "Deciduous_Bleaf_Frst":
         Dec_Bleaf_EF=vtype_ind*((0.80*EF_FLM['hwd'])+(0.20*EF_SML['hwd']))
         arr_parent_EFs.append(Dec_Bleaf_EF.values.flatten())
      elif lc == "Mixed_Frst":
         Mix_frst_EF=vtype_ind*((0.85*EF_FLM['mxd'])+(0.15*EF_SML['mxd']))
         arr_parent_EFs.append(Mix_frst_EF.values.flatten())
      elif lc == "Closed_Shrublands":
         Cls_shrub_EF=vtype_ind*((0.95*EF_FLM['shrb'])+(0.05*EF_SML['shrb']))
         arr_parent_EFs.append(Cls_shrub_EF.values.flatten())
      elif lc == "Open_Shrublands":
         Opn_shrub_EF=vtype_ind*((0.95*EF_FLM['shrb'])+(0.05*EF_SML['shrb']))
         arr_parent_EFs.append(Opn_shrub_EF.values.flatten())
      elif lc == "Woody_Savannas":
         Wdy_sv_EF=vtype_ind*((0.95*EF_FLM['shrb_grs'])+(0.05*EF_SML['shrb_grs']))
         arr_parent_EFs.append(Wdy_sv_EF.values.flatten())
      elif lc == "Savannas":
         Savn_EF=vtype_ind*((0.95*EF_FLM['grs'])+(0.05*EF_SML['grs']))
         arr_parent_EFs.append(Savn_EF.values.flatten())
      elif lc == "Grasslands":
         Grass_EF=vtype_ind*((0.95*EF_FLM['grs'])+(0.05*EF_SML['grs']))
         arr_parent_EFs.append(Grass_EF.values.flatten())
      elif lc == "Croplands":
         Crops_EF=vtype_ind*((0.95*EF_FLM['grs'])+(0.05*EF_SML['grs']))
         arr_parent_EFs.append(Crops_EF.values.flatten())
      elif lc == "Cropland_NVeg":
         NVeg_EF=vtype_ind*((0.95*EF_FLM['grs'])+(0.05*EF_SML['grs']))
         arr_parent_EFs.append(NVeg_EF.values.flatten())
      else:
         arr_parent_EFs.append(0.)
   arr_parent_EFs_flat=sum(arr_parent_EFs)
   arr_parent_EFs_reshape=np.reshape(arr_parent_EFs_flat,(cols,rows))
   arr_parent_EFs=arr_parent_EFs_reshape
   return arr_parent_EFs

#create a dummy hr rave interpolated file for rave_non_avail_hours and when regridder fails
def create_dummy(intp_dir,dummy_hr_rave,generate_hr_dummy,rave_avail,rave_nonavail_hours_test,rave_to_intp):
   os.chdir(intp_dir)
   if generate_hr_dummy==True:
      for i in rave_avail:
         print('Producing RAVE dummy files for all hrs:',i)
         dummy_rave=xr.open_dataset(dummy_hr_rave)
         missing_rave=xr.zeros_like(dummy_rave)
         missing_rave.attrs['RangeBeginningDate']=i[0:4]+'-'+i[4:6]+'-'+i[6:8]
         missing_rave.attrs['RangeBeginningTime\(UTC-hour\)']= i[8:10]
         missing_rave.to_netcdf(rave_to_intp+'dmy_'+i[21:49],unlimited_dims={'t':True})
   else:
      for i in rave_nonavail_hours_test:
         print('Producing RAVE dummy file for:',i)
         dummy_rave=xr.open_dataset(dummy_hr_rave)
         missing_rave=xr.zeros_like(dummy_rave)
         missing_rave.attrs['RangeBeginningDate']=i[0:4]+'-'+i[4:6]+'-'+i[6:8]
         missing_rave.attrs['RangeBeginningTime\(UTC-hour\)']= i[8:10]
         missing_rave.to_netcdf('dmy_'+rave_to_intp+i+'00_'+i+'00.nc',unlimited_dims={'t':True})
         print('dmy_'+rave_to_intp+i+'00_'+i+'00.nc')


#Sort raw RAVE, create source and target filelds, and compute emissions factors
fcst_dates=date_range(current_day)
intp_avail_hours,intp_non_avail_hours=check_for_intp_rave(intp_dir,fcst_dates,rave_to_intp)
rave_avail,rave_avail_hours,rave_nonavail_hours_test=check_for_raw_rave(RAVE,intp_non_avail_hours)
srcfield,tgtfield,tgt_latt,tgt_lont,srcgrid,tgtgrid,src_latt,tgt_area=creates_st_fields(grid_in,grid_out)
arr_parent_EFs=generate_EFs(veg_map,EF_FLM,EF_SML,land_cats)
#generate regridder
try:
   print('GENERATING REGRIDDER')
   regridder = ESMF.RegridFromFile(srcfield, tgtfield,filename)
   print('REGRIDDER FINISHED')
except ValueError:
   print('REGRIDDER FAILS USE DUMMY EMISSIONS')
   use_dummy_emiss=True
   generate_hr_dummy=True
else:
   use_dummy_emiss=False
   generate_hr_dummy=False
#process RAVE available for interpolation
sorted_obj = sorted(os.listdir(RAVE))
iii = 0
for rave_file in rave_avail:
   os.chdir(RAVE)
   if use_dummy_emiss==False and rave_file in sorted_obj:
      print('Interpolating:',rave_file)
      ds_togrid=xr.open_dataset(rave_file)
      QA=ds_togrid['QA']       #QC flags for fire emiss
      FRE_threshold= ds_togrid['FRE']
      print('=============before regridding===========','FRP_MEAN')
      print(np.sum(ds_togrid['FRP_MEAN'],axis=(1,2)))
      os.chdir(intp_dir)
      fout=Dataset(rave_to_intp+rave_file[21:33]+'_'+rave_file[21:33]+'.nc','w')
      create_emiss_file(fout,rave_file)
      Store_latlon_by_Level(fout,'geolat',tgt_latt,'cell center latitude','degrees_north','2D','-9999.f','1.f')
      Store_latlon_by_Level(fout,'geolon',tgt_lont,'cell center longitude','degrees_east','2D','-9999.f','1.f')
      for svar in vars_emis:
         print(svar)
         srcfield = ESMF.Field(srcgrid, name=svar)
         tgtfield = ESMF.Field(tgtgrid, name=svar)
         src_rate = ds_togrid[svar].fillna(0)
         #apply QC flags
         src_QA=xr.where(FRE_threshold>1000,src_rate,0.0)
         src_cut = src_QA[0,:,:]
         src_cut = xr.where(src_latt>7.22291,src_cut,0.0)
         srcfield.data[...] = src_cut
         tgtfield = regridder(srcfield, tgtfield)
         if svar=='FRP_MEAN':
            Store_by_Level(fout,'frp_avg_hr','Mean Fire Radiative Power','MW','3D','0.f','1.f')
            tgt_rate = tgtfield.data
            fout.variables['frp_avg_hr'][0,:,:] = tgt_rate
            print('=============after regridding==========='+svar)
            print(np.sum(tgt_rate))
         elif svar=='FRE':
            Store_by_Level(fout,'ebb_smoke_hr','PM2.5 emissions','ug m-2 h-1','3D','0.f','1.f')
            tgt_rate = tgtfield.data
            tgt_rate = tgt_rate*arr_parent_EFs*beta
            tgt_rate = (tgt_rate*fg_to_ug)/to_s
            tgt_rate = tgt_rate/tgt_area
            tgt_rate =xr.DataArray(tgt_rate)
            fout.variables['ebb_smoke_hr'][0,:,:] = tgt_rate
         elif svar=='FRP_SD':
            Store_by_Level(fout,'frp_std_hr','Standar Deviation of Fire Radiative Energy','MW','3D','0.f','1.f')
            tgt_rate = tgtfield.data
            fout.variables['frp_std_hr'][0,:,:] = tgt_rate
         elif svar=='PM2.5':
            Store_by_Level(fout,'ebu_oc','Particulate matter < 2.5 ug','ug m-2 s-1','3D','0.f','1.f')
            tgt_rate = tgtfield.data/to_s
            fout.variables['ebu_oc'][0,:,:] = tgt_rate
         else :
            tgt_rate = tgtfield.data/to_s
            fout.variables[svar][0,:,:] = tgt_rate
      ds_togrid.close()
      fout.close()
   iii += 1
#Create dummy hr files
create_dummy(intp_dir,dummy_hr_rave,generate_hr_dummy,rave_avail,rave_nonavail_hours_test,rave_to_intp)
