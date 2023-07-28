# Open the original text file for reading
with open('/cvhci/temp/prgc/hdgcn_filtered/statistics/setup.txt', 'r') as f1:
    # Read the first 11653 lines
    lines = f1.readlines()[:11653]

# Open the new text file for writing
with open('/cvhci/temp/prgc/hdgcn_filtered/statistics/filtered_setup.txt', 'w') as f2:
    # Write the lines to the new file
    f2.writelines(lines)