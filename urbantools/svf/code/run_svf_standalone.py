"""
Standalone script for calculating the Sky View Factor (SVF) from a DSM in ASCII (.ASC) format.
The code corrects the header of the .ASC file (converting XLLCENTER/YLLCENTER to XLLCORNER/YLLCORNER), 
loads the raster, performs the SVF calculation with the UMEP algorithm (svfForProcessing153 or svfForProcessing655), 
and finally saves the result in GeoTIFF and CSV format.

Optionally, it generates a heatmap of the result.

Author: Aldo, UMEP
Dependencies: numpy, osgeo.gdal, pandas, matplotlib, UMEP functions (svf_functions.py)
"""

import numpy as np
from osgeo import gdal
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
# === Aggiungi percorso funzioni UMEP ===
sys.path.append(os.path.join(os.getcwd(), "functions"))
sys.path.append(os.path.join(os.getcwd(), "util"))
import svf_functions as svf  # usa svfForProcessing153 o svfForProcessing655

tile = "dsm_demo"

# === Parametri ===
input_asc_path = f"data/{tile}.ASC"       # <--- Inserisci il tuo DSM ASCII qui
corrected_asc_path = f"data/corrected_dsm_{tile}.asc"
output_svf_path = f"output/svf_{tile}.tif"

ANISOTROPIC = True  # True = 153 direzioni, False = 655 direzioni
TRANS_VEG = 0       # percentuale trasmissività vegetazione (0 se non presente)

# === Funzione per correggere intestazione .ASC ===
def fix_asc_header(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    header = {}
    data_start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().split()[0].upper() in ['NCOLS', 'NROWS', 'CELLSIZE', 'XLLCENTER', 'YLLCENTER', 'NODATA_VALUE']:
            key, val = line.strip().split()
            header[key.upper()] = float(val)
        else:
            data_start_idx = i
            break

    if 'XLLCENTER' in header and 'YLLCENTER' in header:
        header['XLLCORNER'] = header['XLLCENTER'] - header['CELLSIZE'] / 2
        header['YLLCORNER'] = header['YLLCENTER'] - header['CELLSIZE'] / 2

    # Ricostruisci intestazione
    new_header = [
        f"NCOLS         {int(header['NCOLS'])}",
        f"NROWS         {int(header['NROWS'])}",
        f"XLLCORNER     {header['XLLCORNER']:.4f}",
        f"YLLCORNER     {header['YLLCORNER']:.4f}",
        f"CELLSIZE      {header['CELLSIZE']:.4f}",
        f"NODATA_VALUE  {int(header['NODATA_VALUE'])}"
    ]

    with open(output_path, 'w') as f:
        f.write('\n'.join(new_header) + '\n')
        f.writelines(lines[data_start_idx:])

    print(f"[INFO] Intestazione .ASC corretta: {output_path}")

# === 1. Correggi intestazione .ASC ===
fix_asc_header(input_asc_path, corrected_asc_path)

# === 2. Carica DSM ===
ds = gdal.Open(corrected_asc_path)
dsm = ds.ReadAsArray().astype(float)

nd = ds.GetRasterBand(1).GetNoDataValue()
if nd is not None:
    dsm[dsm == nd] = 0

if dsm.min() < 0:
    dsm += abs(dsm.min())

gt = ds.GetGeoTransform()
pixel_res = gt[1]
scale = 1 / pixel_res
rows, cols = dsm.shape

# === 3. Dummy array vegetazione ===
veg = np.zeros_like(dsm)
veg2 = 0.0
useveg = 0


# === 4. Calcolo SVF ===
class FakeFeedback:
    def isCanceled(self):
        return False
    def setProgressText(self, text):
        print(f"[FEEDBACK] {text}")
    def setProgress(self, value):
        print(f"[PROGRESS] {value}%")

print("[INFO] Calcolo SVF...")
if ANISOTROPIC:
    feedback = FakeFeedback()
    result = svf.svfForProcessing153(dsm, veg, veg2, scale, useveg, pixel_res, False, None, feedback)
else:
    result = svf.svfForProcessing655(dsm, veg, veg2, scale, useveg, feedback=None)

svf_total = result["svf"]

# === 5. Salva output ===
driver = gdal.GetDriverByName('GTiff')
out_ds = driver.Create(output_svf_path, cols, rows, 1, gdal.GDT_Float32)
out_ds.SetGeoTransform(gt)
out_ds.SetProjection(ds.GetProjection())
out_ds.GetRasterBand(1).WriteArray(svf_total)
out_ds.GetRasterBand(1).SetNoDataValue(0)
out_ds.FlushCache()

print(f"[SUCCESS] SVF salvato in: {output_svf_path}")


# === 6. Crea CSV con coordinate e SVF ===
print("[INFO] Generazione CSV...")

x0, dx, _, y0, _, dy = gt
x_coords = x0 + dx * np.arange(cols)
y_coords = y0 + dy * np.arange(rows)
xx, yy = np.meshgrid(x_coords, y_coords)

flat_coords = np.column_stack((xx.flatten(), yy.flatten()))
flat_svf = svf_total.flatten()

# Rimuovi i punti nodata (dove SVF = 0 o NaN)
valid_mask = ~np.isnan(flat_svf) & (flat_svf > 0)
coords_strings = [f"{x:.2f},{y:.2f}" for x, y in flat_coords[valid_mask]]

df = pd.DataFrame({
    "coordinate": coords_strings,
    "svf": flat_svf[valid_mask]
})

csv_output_path = f"output/svf_{tile}.csv"
df.to_csv(csv_output_path, index=False)
print(f"[SUCCESS] CSV salvato in: {csv_output_path}")


print("[INFO] Visualizzazione Heatmap...")
plt.figure(figsize=(10, 8))
plt.imshow(svf_total, cmap='coolwarm', origin='upper')
plt.colorbar(label="Sky View Factor")
plt.title("SVF Heatmap")
plt.axis("off")
plt.tight_layout()
plt.show()