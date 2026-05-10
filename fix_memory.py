import json

with open("2_comparison.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        if "X_tr, y_tr = X_train.iloc[:-val_size], y_train.iloc[:-val_size]" in src:
            src = src.replace("X_val, y_val = X_train.iloc[-val_size:], y_train.iloc[-val_size:]", "X_val, y_val = X_train.iloc[-val_size:].copy(), y_train.iloc[-val_size:].copy()")
            src = src.replace("X_tr, y_tr = X_train.iloc[:-val_size], y_train.iloc[:-val_size]", "X_tr, y_tr = X_train.iloc[:-val_size].copy(), y_train.iloc[:-val_size].copy()")
            lines = src.split('\n')
            new_lines = [l + '\n' for l in lines[:-1]]
            if len(lines) > 0:
                new_lines.append(lines[-1])
            cell['source'] = new_lines

with open("2_comparison.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
