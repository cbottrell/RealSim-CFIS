#!/usr/bin/env python3

import numpy as np
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning
import os,time,vos,warnings
from astropy.io import fits
warnings.filterwarnings('ignore')

def CFIS_tile_radec(ra,dec):
    # find tile (see Stacking in docs)
    # https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/community/cfis/datadoc.html
    # CFIS tiles centers are on a cartesian grid spaced by exactly 0.5 degrees apart.
    yyy = int(np.rint((dec+90)*2))
    cosf = np.cos((yyy/2-90) * np.pi/180.)
    xxx = int(np.rint(ra*2*cosf))
    tile = f'CFIS.{xxx:03d}.{yyy:03d}.r.fits'
    return tile

def CFIS_cutout_radec(ra,dec,cutout_name='CFIS_cutout_radec.fits',fov_arcsec=100,dr='DR3'):
    '''
    Obtain CFIS cutout at ra,dec coordinates (degrees) with a square FOV set by `fov_arcsec` (arcsec).
    Parts of a FOV outside a tile edge are converted to NaNs.
    '''
    tile  = CFIS_tile_radec(ra,dec)
    
    # get output FOV dimensions
    arcsec_per_pixel = 0.1857
    fov_pixels = int(fov_arcsec / arcsec_per_pixel)
    if fov_pixels%2: fov_pixels+=1
    hw = int(fov_pixels/2)

    # get wcs file
    wcs_name = 'WCS-{}'.format(tile)
    if os.access(wcs_name,0): os.remove(wcs_name)
    attempts, max_attempts = 0, 10
    while not os.access(wcs_name,0) and attempts<max_attempts:
        attempts+=1
        vos_cmd = f'vcp vos:cfis/tiles_{dr}/{tile}[1:2,1:2] {wcs_name}'
        try:
            os.system(vos_cmd)
        except:
            time.sleep(0.1)   
            
    # get wcs mapping
    wcs = WCS(wcs_name)
    colc,rowc = wcs.all_world2pix(ra,dec,1,ra_dec_order=True)
    colc,rowc = int(np.around(colc)),int(np.around(rowc))
    if os.access(wcs_name,0): os.remove(wcs_name)
    
    # actual field of view params
    row_min = rowc-hw+1
    row_max = rowc+hw
    col_min = colc-hw+1
    col_max = colc+hw
    
    # cropped field of view params (for edges)
    crop_row_min = 0
    crop_row_max = fov_pixels
    crop_col_min = 0
    crop_col_max = fov_pixels
    if row_min < 1:
        crop_row_min = 1-row_min
        row_min = 1
    if row_max > 10000:
        crop_row_max -= (row_max-10000)
        row_max = 10000
    if col_min < 1:
        crop_col_min = 1-col_min
        col_min = 1
    if col_max > 10000:
        crop_col_max -= (col_max-10000)
        col_max = 10000
              
    # get temporary cutout file
    tmp_name = 'TMP-{}'.format(tile)
    if os.access(tmp_name,0): os.remove(tmp_name)
    attempts, max_attempts = 0, 10
    if not os.access(tmp_name,0) and attempts<max_attempts:
        attempts+=1
        vos_cmd = f'vcp vos:cfis/tiles_{dr}/{tile}[{col_min}:{col_max},{row_min}:{row_max}] {tmp_name}'
        try:
            os.system(vos_cmd)
        except:
            time.sleep(0.1)
            
    # extract data and header and delete
    with fits.open(tmp_name,mode='readonly') as hdu:
        header = hdu[0].header
        img_data = hdu[0].data
    if os.access(tmp_name,0): os.remove(tmp_name)
        
    # add to section of FOV where data are within the tile and save
    cutout_data = np.empty((fov_pixels,fov_pixels))*np.nan
    cutout_data[crop_row_min:crop_row_max,crop_col_min:crop_col_max]=img_data
    if os.access(cutout_name,0): os.remove(cutout_name)
    hdu_pri = fits.PrimaryHDU(cutout_data) 
    hdu_pri.header = header
    hdu_pri.writeto(cutout_name)
    return tile

def main():
    
    # ra,dec of target
    ra,dec = 236.4707075027352,36.93748162164066
    
    CFIS_cutout_radec(ra,dec,cutout_name='CFIS_cutout.fits',fov_arcsec=100)
    
if __name__ == '__main__':
    main()
    
            