clear all
close all
clc
%% ===== USER SETTINGS =====
file = "9.5.25 leaf count.xlsx";
sheetName = "9-5-25 CONTROL (Non Solar)";   % change if needed

maxL = 28;                           % L1..L28 template
preferTwoRowsWhen28 = true;          % if 28 appears, treat as 2 x 14 (true) or 1 x 28 (false)

% Border rule: how many plants to drop at each row end
endDrop = 1;                         % 1 means drop first+last plant in each physical row (common)
                                    % set to 2 if you want to exclude two plants at each end

%% ===== READ RAW =====
raw = readcell(file, "Sheet", sheetName);

% Find the header row: first row containing something like P1Q1A, P2Q4B, etc.
pat = "P\d+Q\d+[AB]";  % matches P#Q#[A/B]
headerRow = [];
for r = 1:size(raw,1)
    rowStr = string(raw(r,:));
    if any(~ismissing(rowStr) & ~cellfun(@isempty, regexp(rowStr, pat, "once")))
        headerRow = r;
        break;
    end
end
if isempty(headerRow)
    error("Could not find header row with columns like P1Q1A. Check sheet formatting.");
end

headers = string(raw(headerRow,:));

% Find the L1..L28 label column + first data row
Lcol = [];
for c = 1:size(raw,2)
    if string(raw(headerRow+1,c)) == "L1"
        Lcol = c;
        break;
    end
end
if isempty(Lcol)
    error("Could not find L1 row labels under the header row. Check where L1..L28 are.");
end
dataStartRow = headerRow + 1;

% Determine data columns = those whose header matches P#Q#[A/B]
dataCols = find(~cellfun(@isempty, regexp(headers, pat, "once")));

% Restrict to rows L1..L28 (or fewer if sheet ends)
nRowsAvail = min(maxL, size(raw,1) - dataStartRow + 1);

% Pull L labels
Llabels = strings(nRowsAvail,1);
for i = 1:nRowsAvail
    Llabels(i) = string(raw{dataStartRow+i-1, Lcol});
end

% status codes:
% 0 = blank
% 1 = numeric (measured)
% 2 = X (dead)
status = zeros(nRowsAvail, numel(dataCols));
measVals = nan(nRowsAvail, numel(dataCols));

for j = 1:numel(dataCols)
    c = dataCols(j);
    for i = 1:nRowsAvail
        v = raw{dataStartRow+i-1, c};

        if isempty(v) || (isstring(v) && strlength(strtrim(v))==0)
            status(i,j) = 0;
        elseif isnumeric(v)
            if isnan(v)
                status(i,j) = 0;
            else
                status(i,j) = 1;
                measVals(i,j) = v;
            end
        elseif ischar(v) || isstring(v)
            s = upper(strtrim(string(v)));
            if s == "X"
                status(i,j) = 2;
            else
                % Unknown text -> treat as blank (or change if you prefer)
                status(i,j) = 0;
            end
        else
            status(i,j) = 0;
        end
    end
end

colNames = headers(dataCols);

%% ===== INFER DENSITY PER COLUMN =====
% We infer "effective planted length K" mainly from trailing blanks.
% K = last index that is non-blank (measured or X). Trailing blanks after K suggest fewer planted.
lastNonBlank = zeros(1, numel(dataCols));
for j = 1:numel(dataCols)
    idx = find(status(:,j) ~= 0, 1, "last");
    if isempty(idx), idx = 0; end
    lastNonBlank(j) = idx;
end

% Snap to common densities if close (14 or 28), otherwise keep K
inferK = lastNonBlank;
for j = 1:numel(dataCols)
    k = inferK(j);
    if abs(k-14) <= 1
        inferK(j) = 14;
    elseif abs(k-28) <= 1
        inferK(j) = 28;
    end
end

%% ===== COMPUTE BORDER POSITIONS (L indices) PER COLUMN =====
% borderMask(i,j)=true means L-index i is a border position (even if blank in the sheet)
borderMask = false(nRowsAvail, numel(dataCols));

for j = 1:numel(dataCols)
    K = inferK(j);

    if K == 0
        continue;
    end

    if K == 14
        % one row of 14
        rowStarts = 1;
        rowEnds   = 14;

    elseif K == 28 && preferTwoRowsWhen28
        % two rows of 14: [1..14] and [15..28]
        rowStarts = [1, 15];
        rowEnds   = [14, 28];

    elseif K == 28 && ~preferTwoRowsWhen28
        % one row of 28
        rowStarts = 1;
        rowEnds   = 28;

    else
        % fallback: treat as one row of length K
        rowStarts = 1;
        rowEnds   = K;
    end

    % mark border positions at each row end (with endDrop)
    for rr = 1:numel(rowStarts)
        rs = rowStarts(rr);
        re = rowEnds(rr);

        leftBorders  = rs : min(rs+endDrop-1, re);
        rightBorders = max(re-endDrop+1, rs) : re;

        borderMask(leftBorders, j) = true;
        borderMask(rightBorders, j) = true;
    end
end

%% ===== SUMMARY TABLE =====
nMeasured = sum(status==1,1);
nDeadX    = sum(status==2,1);
nBlank    = sum(status==0,1);

% Count how many border positions are blank vs filled
borderCount = sum(borderMask,1);
borderBlank = sum(borderMask & status==0, 1);
borderNonBlank = sum(borderMask & status~=0, 1);

T = table(colNames(:), inferK(:), nMeasured(:), nDeadX(:), nBlank(:), ...
          borderCount(:), borderBlank(:), borderNonBlank(:), ...
          'VariableNames', ["Column","InferredK","Measured","DeadX","Blank", ...
                            "BorderPositions","BorderBlank","BorderNonBlank"]);
disp(T);

%% ===== PLOT: schematic with border overlay =====
figure("Color","w");
imagesc(status);
axis tight;
set(gca,"YDir","normal");

% colormap: blank=white, measured=green, X=orange
colormap([1 1 1; 0.70 0.90 0.70; 0.95 0.75 0.40]);
caxis([0 2]);

xticks(1:numel(dataCols));
xticklabels(colNames);
xtickangle(45);

yticks(1:nRowsAvail);
yticklabels(Llabels);

title("Schematic: blank / measured / X with border positions outlined");
xlabel("Subplot column");
ylabel("Plant index");

hold on;
% Overlay border positions with blue rectangles
[rr, cc] = find(borderMask);
for k = 1:numel(rr)
    r = rr(k); c = cc(k);
    rectangle("Position",[c-0.5, r-0.5, 1, 1], "EdgeColor",[0 0.3 1], "LineWidth",1.2);
end

% Legend (dummy markers)
plot(nan,nan,'s','MarkerFaceColor',[1 1 1],'MarkerEdgeColor','k','DisplayName','Blank');
plot(nan,nan,'s','MarkerFaceColor',[0.70 0.90 0.70],'MarkerEdgeColor','k','DisplayName','Measured');
plot(nan,nan,'s','MarkerFaceColor',[0.95 0.75 0.40],'MarkerEdgeColor','k','DisplayName','X (dead)');
plot(nan,nan,'s','MarkerFaceColor','none','MarkerEdgeColor',[0 0.3 1],'LineWidth',1.2,'DisplayName','Border position');
legend("Location","eastoutside");
hold off;

%% ===== OPTIONAL: value heatmap (NaN where blank/X) =====
figure("Color","w");

h = imagesc(measVals);
axis tight;
set(gca,"YDir","normal");

colormap(parula);  % or keep your current colormap

cb = colorbar;
cb.Label.String = "Lettuce leaf count per plant";
cb.Label.FontSize = 12;
cb.Label.FontWeight = "bold";

% Make NaNs gray
set(gca, 'Color', [0.8 0.8 0.8]);  % background color = gray
set(h, 'AlphaData', ~isnan(measVals));  % hide NaNs (show background)

xticks(1:numel(dataCols));
xticklabels(colNames);
xtickangle(45);

yticks(1:nRowsAvail);
yticklabels(Llabels);

title("Measured values (Gray = blank or X)");
xlabel("Subplot column");
ylabel("Plant index");