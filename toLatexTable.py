"""
Generates a LaTex Table from the output CSV
"""

import csv
with open("output.csv", "r") as file:
    reader = csv.reader(file)
    print("\\midrule")
    next(reader)
    for row in reader:
        print(row[0] + " & highest-ability & " + row[1] + " ("+row[2]+")" + " & " + str(round(float(row[1])-float(row[3]),2)) + "\\\\")
        print("\t & random &  " + row[3] + " ("+row[4]+")" + "\\\\")
    print("\\bottomrule")