# Used ChatGPT to convert from MATLAB to Python

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# clear all / close all / clc (MATLAB equivalents)
plt.close('all')

# ===== USER SETTINGS =====
file = "9.25.25 Leaf Count.xlsx"
sheetName = "9-25-25 CONTROL (Non Solar)"  # change if needed

maxL = 28  # L1..L28 template
preferTwoRowsWhen28 = True  # if 28 appears, treat as 2 x 14 (True) or 1 x 28 (False)

# Border rule: how many plants to drop at each row end
endDrop = 1  # 1 means drop first+last plant in each physical row (common)
            # set to 2 if you want to exclude two plants at each end

# ===== READ RAW =====
raw_df = pd.read_excel(file, sheet_name=sheetName, header=None, engine="openpyxl")
raw = raw_df.to_numpy(dtype=object)

# Find the header row: first row containing something like P1Q1A, P2Q4B, etc.
pat = r"P\d+Q\d+[AB]"  # matches P#Q#[A/B]
headerRow = None
for r in range(raw.shape[0]):
    row = raw[r, :]
    found = False
    for v in row:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        s = str(v)
        if re.search(pat, s):
            found = True
            break
    if found:
        headerRow = r
        break

if headerRow is None:
    raise RuntimeError("Could not find header row with columns like P1Q1A. Check sheet formatting.")

headers = np.array([("" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)) for v in raw[headerRow, :]], dtype=object)

# Find the L1..L28 label column + first data row
Lcol = None
for c in range(raw.shape[1]):
    v = raw[headerRow + 1, c] if (headerRow + 1) < raw.shape[0] else None
    s = "" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
    if s == "L1":
        Lcol = c
        break

if Lcol is None:
    raise RuntimeError("Could not find L1 row labels under the header row. Check where L1..L28 are.")

dataStartRow = headerRow + 1

# Determine data columns = those whose header matches P#Q#[A/B]
dataCols = [i for i, h in enumerate(headers) if re.search(pat, str(h) if h is not None else "")]

# Restrict to rows L1..L28 (or fewer if sheet ends)
nRowsAvail = min(maxL, raw.shape[0] - dataStartRow)

# Pull L labels
Llabels = np.array(
    [("" if (raw[dataStartRow + i, Lcol] is None or (isinstance(raw[dataStartRow + i, Lcol], float) and np.isnan(raw[dataStartRow + i, Lcol])))
      else str(raw[dataStartRow + i, Lcol]))
     for i in range(nRowsAvail)],
    dtype=object
)

# status codes:
# 0 = blank
# 1 = numeric (measured)
# 2 = X (dead)
status = np.zeros((nRowsAvail, len(dataCols)), dtype=int)
measVals = np.full((nRowsAvail, len(dataCols)), np.nan, dtype=float)

for j, c in enumerate(dataCols):
    for i in range(nRowsAvail):
        v = raw[dataStartRow + i, c]

        if v is None:
            status[i, j] = 0
        elif isinstance(v, str):
            if len(v.strip()) == 0:
                status[i, j] = 0
            else:
                s = v.strip().upper()
                if s == "X":
                    status[i, j] = 2
                else:
                    # Unknown text -> treat as blank (or change if you prefer)
                    status[i, j] = 0
        elif isinstance(v, (int, float, np.integer, np.floating)):
            if isinstance(v, float) and np.isnan(v):
                status[i, j] = 0
            else:
                status[i, j] = 1
                measVals[i, j] = float(v)
        else:
            status[i, j] = 0

colNames = np.array([headers[c] for c in dataCols], dtype=object)

# ===== INFER DENSITY PER COLUMN =====
# We infer "effective planted length K" mainly from trailing blanks.
# K = last index that is non-blank (measured or X). Trailing blanks after K suggest fewer planted.
lastNonBlank = np.zeros(len(dataCols), dtype=int)
for j in range(len(dataCols)):
    idxs = np.where(status[:, j] != 0)[0]
    lastNonBlank[j] = (idxs[-1] + 1) if idxs.size else 0  # 1-based like MATLAB

# Snap to common densities if close (14 or 28), otherwise keep K
inferK = lastNonBlank.copy()
for j in range(len(dataCols)):
    k = inferK[j]
    if abs(k - 14) <= 1:
        inferK[j] = 14
    elif abs(k - 28) <= 1:
        inferK[j] = 28

# ===== COMPUTE BORDER POSITIONS (L indices) PER COLUMN =====
# borderMask(i,j)=True means L-index i is a border position (even if blank in the sheet)
borderMask = np.zeros((nRowsAvail, len(dataCols)), dtype=bool)

for j in range(len(dataCols)):
    K = int(inferK[j])

    if K == 0:
        continue

    if K == 14:
        # one row of 14
        rowStarts = [1]
        rowEnds = [14]

    elif K == 28 and preferTwoRowsWhen28:
        # two rows of 14: [1..14] and [15..28]
        rowStarts = [1, 15]
        rowEnds = [14, 28]

    elif K == 28 and (not preferTwoRowsWhen28):
        # one row of 28
        rowStarts = [1]
        rowEnds = [28]

    else:
        # fallback: treat as one row of length K
        rowStarts = [1]
        rowEnds = [K]

    # mark border positions at each row end (with endDrop)
    for rs, re_ in zip(rowStarts, rowEnds):
        leftBorders = list(range(rs, min(rs + endDrop - 1, re_) + 1))
        rightBorders = list(range(max(re_ - endDrop + 1, rs), re_ + 1))

        for ii in leftBorders:
            if 1 <= ii <= nRowsAvail:
                borderMask[ii - 1, j] = True
        for ii in rightBorders:
            if 1 <= ii <= nRowsAvail:
                borderMask[ii - 1, j] = True

# ===== SUMMARY TABLE =====
nMeasured = (status == 1).sum(axis=0)
nDeadX = (status == 2).sum(axis=0)
nBlank = (status == 0).sum(axis=0)

# Count how many border positions are blank vs filled
borderCount = borderMask.sum(axis=0)
borderBlank = (borderMask & (status == 0)).sum(axis=0)
borderNonBlank = (borderMask & (status != 0)).sum(axis=0)

T = pd.DataFrame({
    "Column": colNames,
    "InferredK": inferK,
    "Measured": nMeasured,
    "DeadX": nDeadX,
    "Blank": nBlank,
    "BorderPositions": borderCount,
    "BorderBlank": borderBlank,
    "BorderNonBlank": borderNonBlank
})
print(T.to_string(index=False))

# ===== PLOT: schematic with border overlay =====
fig1 = plt.figure(facecolor="w")
ax1 = plt.gca()
im1 = ax1.imshow(status, aspect="auto", origin="lower", interpolation="nearest")
ax1.set_title("Schematic: blank / measured / X with border positions outlined")
ax1.set_xlabel("Subplot column")
ax1.set_ylabel("Plant index")

# colormap: blank=white, measured=green, X=orange
cmap = plt.matplotlib.colors.ListedColormap([[1, 1, 1], [0.70, 0.90, 0.70], [0.95, 0.75, 0.40]])
im1.set_cmap(cmap)
im1.set_clim(0, 2)

ax1.set_xticks(np.arange(len(dataCols)))
ax1.set_xticklabels(colNames, rotation=45, ha="right")

ax1.set_yticks(np.arange(nRowsAvail))
ax1.set_yticklabels(Llabels)

# Overlay border positions with blue rectangles
rr, cc = np.where(borderMask)
for r, c in zip(rr, cc):
    ax1.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=(0, 0.3, 1), linewidth=1.2))

# Legend (dummy markers)
ax1.plot(np.nan, np.nan, 's', markerfacecolor=[1, 1, 1], markeredgecolor='k', label='Blank')
ax1.plot(np.nan, np.nan, 's', markerfacecolor=[0.70, 0.90, 0.70], markeredgecolor='k', label='Measured')
ax1.plot(np.nan, np.nan, 's', markerfacecolor=[0.95, 0.75, 0.40], markeredgecolor='k', label='X (dead)')
ax1.plot(np.nan, np.nan, 's', markerfacecolor='none', markeredgecolor=[0, 0.3, 1], linewidth=1.2, label='Border position')
ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()

# ===== OPTIONAL: value heatmap (NaN where blank/X) =====
fig2 = plt.figure(facecolor="w")
ax2 = plt.gca()

h = ax2.imshow(measVals, aspect="auto", origin="lower", interpolation="nearest")
ax2.set_title("Measured values (Gray = blank or X)")
ax2.set_xlabel("Subplot column")
ax2.set_ylabel("Plant index")

# colormap(parula);  % or keep your current colormap
h.set_cmap(plt.get_cmap("viridis"))

cb = plt.colorbar(h, ax=ax2)
cb.set_label("Lettuce leaf count per plant", fontsize=12, fontweight="bold")

# Make NaNs gray
ax2.set_facecolor((0.8, 0.8, 0.8))  # background color = gray
h.set_alpha((~np.isnan(measVals)).astype(float))  # hide NaNs (show background)

ax2.set_xticks(np.arange(len(dataCols)))
ax2.set_xticklabels(colNames, rotation=45, ha="right")

ax2.set_yticks(np.arange(nRowsAvail))
ax2.set_yticklabels(Llabels)

plt.tight_layout()
plt.show()