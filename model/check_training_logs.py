# Read the training logs to understand what happened
with open('train_native_out.txt', 'r', encoding='utf-16') as f:
    lines = f.readlines()

# Find lines with epoch information
epoch_lines = []
for line in lines:
    if 'Epoch' in line and 'accuracy:' in line and 'val_accuracy:' in line:
        epoch_lines.append(line.strip())

print("Training progress (last 10 epochs):")
for line in epoch_lines[-10:]:
    print(line)

print(f"\nTotal epochs found: {len(epoch_lines)}")
if epoch_lines:
    first_epoch = epoch_lines[0]
    last_epoch = epoch_lines[-1]
    print(f"First epoch: {first_epoch}")
    print(f"Last epoch: {last_epoch}")
