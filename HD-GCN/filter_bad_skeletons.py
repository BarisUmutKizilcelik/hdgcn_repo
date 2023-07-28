

with open("/cvhci/temp/prgc/statistics/samples_with_missing_skeletons.txt", "r") as f1:
    lines_to_remove = set(f1.read().splitlines())

with open("/cvhci/temp/prgc/statistics/filtered_skes_avaliable_name.txt", "r+") as f2:
    lines = f2.readlines()
    f2.seek(0)
    for line in lines:
        if line.strip() not in lines_to_remove:
            f2.write(line)
    f2.truncate()
